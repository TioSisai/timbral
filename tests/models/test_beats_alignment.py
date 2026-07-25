"""Alignment tests against the official BEATs implementation (requires explicit ``--run-alignment beats``).

The reference side is the official ``beats/`` source from a pinned commit
of microsoft/unilm (sparse-cloned into ``$TMPDIR``, SHA-256 verified, then
loaded via importlib); for fine-tuned entries, the official model has
``predictor = None`` set so it returns raw backbone features. See
``docs/designs/models/extra/beats-alignment.md`` for the contract details.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from timbral.models.encoders import BeatsEncoder
from timbral.models.helpers.beats import (
    BEATS_CHECKPOINTS,
    ensure_beats_checkpoint,
)
from timbral.models.transforms import BeatsKaldiFbankTransform

pytestmark = pytest.mark.alignment("beats")

_UNILM_URL = "https://github.com/microsoft/unilm"
_UNILM_COMMIT = "833df7e7832e5064a281131ee64a481afa8e5b95"
_OFFICIAL_FILE_SHA256 = {
    "BEATs.py": (
        "27f289db7c56ce26f2ceb50d3719854b"
        "91b2dec1c2830d8b1dd8de1bbee19eeb"
    ),
    "backbone.py": (
        "31c0378379a7e0f1d1069f9da444fb86"
        "890fe1ea078959a2dcd39640cdcadbaa"
    ),
    "modules.py": (
        "edeb6b6cd6a784da749f932c3e0783c0"
        "bce556fc768d0d23a4d53d4b819eb424"
    ),
}
_OFFICIAL_MODULE_NAMES = ("modules", "backbone", "BEATs")

_TARGET_SR = 16000
_MIN_TARGET_SAMPLES = 2800
_PATCH_SIZE = 16
_FREQ_PATCHES = 8
# The local pos_conv uses the parametrize version of weight_norm; before
# the value-by-value assertion, remap the official keys under the same
# rule (independent of the helpers implementation under test).
_POS_CONV_KEY_RENAMES = {
    "encoder.pos_conv.0.weight_g": (
        "encoder.pos_conv.0.parametrizations.weight.original0"
    ),
    "encoder.pos_conv.0.weight_v": (
        "encoder.pos_conv.0.parametrizations.weight.original1"
    ),
}

# The CUDA gate is relaxed based on measured behavior: the official
# per-sample kernel and the local batched kernel reduce cuFFT/GEMM in a
# different order, and pure-tone leakage into the noise-floor bins gets
# amplified by log into occasional ~2e-3 log-domain differences (the two
# sides are bit-exact on CPU). See the "numerical gates" in the alignment
# contract.
_TRANSFORM_TOLERANCES = {"cpu": (1e-4, 1e-4), "cuda": (5e-3, 1e-4)}
_ENCODER_TOLERANCES = {"cpu": (1e-4, 1e-4), "cuda": (2e-3, 1e-4)}
_AUDIT_RELATIVE_L2 = {"cpu": 1e-4, "cuda": 1e-3}
_MIN_COSINE = 0.99999

_TRANSFORM_DURATIONS = (0.02, 0.175, 1.0, 4.03, 10.0, 20.0)
_TRANSFORM_SIGNALS = (
    "random",
    "sine",
    "impulse",
    "multisine",
    "silence",
)
_ENCODER_DURATIONS = (1.0, 10.0)
_ENCODER_SIGNALS = ("random", "sine")
_FULL_MATRIX_ENTRY = "beats_iter3_plus_as2m"
_BATCH_CASES = (
    (0.02, 1.0, 4.03),
    (1.0, 4.03, 10.0),
    (1.0, 1.0, 1.0),
)
_INVALID_TAIL_FILL = 7.5

_SUMMARY: dict = {
    "unilm_commit": _UNILM_COMMIT,
    "official_file_sha256": _OFFICIAL_FILE_SHA256,
    "transform": {},
    "encoder": {},
    "mixed_batch": {},
}


def _alignment_root() -> Path:
    return (
        Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        / "timbral-beats-alignment"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_files_valid(beats_dir: Path) -> bool:
    return all(
        (beats_dir / name).exists()
        and _sha256(beats_dir / name) == expected
        for name, expected in _OFFICIAL_FILE_SHA256.items()
    )


def _devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def _make_signal(kind: str, num_samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    time_axis = torch.arange(num_samples, dtype=torch.float32) / _TARGET_SR
    if kind == "random":
        return torch.randn((num_samples,), generator=generator) * 0.5
    if kind == "sine":
        return torch.sin(2 * torch.pi * 997.0 * time_axis)
    if kind == "impulse":
        signal = torch.zeros(num_samples)
        signal[min(100, num_samples - 1)] = 1.0
        return signal
    if kind == "multisine":
        return (
            0.4 * torch.sin(2 * torch.pi * 440.0 * time_axis)
            + 0.3 * torch.sin(2 * torch.pi * 1237.0 * time_axis)
            + 0.2 * torch.cos(2 * torch.pi * 3313.0 * time_axis)
        )
    if kind == "silence":
        return torch.zeros(num_samples)
    raise ValueError(kind)


def _padded_for_official(waveform: torch.Tensor) -> torch.Tensor:
    """Recompute the local minimum zero-padding: the official reference receives the same waveform (zero-padded if needed)."""
    shortfall = _MIN_TARGET_SAMPLES - waveform.shape[-1]
    if shortfall > 0:
        return F.pad(waveform, (0, shortfall))
    return waveform


def _audit(
    local: torch.Tensor,
    reference: torch.Tensor,
    device: str,
) -> dict:
    """Compute float64 audit metrics and enforce the cosine and relative-L2 gates."""
    local64 = local.detach().double().flatten().cpu()
    reference64 = reference.detach().double().flatten().cpu()
    max_abs = (local64 - reference64).abs().max().item()
    reference_norm = reference64.norm().item()
    relative_l2 = (
        (local64 - reference64).norm().item() / reference_norm
        if reference_norm > 0
        else 0.0
    )
    cosine = (
        F.cosine_similarity(local64, reference64, dim=0).item()
        if reference_norm > 0 and local64.norm().item() > 0
        else 1.0
    )
    assert torch.isfinite(local).all()
    assert torch.isfinite(reference).all()
    assert cosine >= _MIN_COSINE
    assert relative_l2 <= _AUDIT_RELATIVE_L2[device]
    return {
        "max_abs": max_abs,
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def _merge_worst(bucket: dict, metrics: dict) -> None:
    bucket["max_abs"] = max(
        bucket.get("max_abs", 0.0), metrics["max_abs"]
    )
    bucket["relative_l2"] = max(
        bucket.get("relative_l2", 0.0), metrics["relative_l2"]
    )
    bucket["cosine"] = min(
        bucket.get("cosine", 1.0), metrics["cosine"]
    )
    bucket["cases"] = bucket.get("cases", 0) + 1


@pytest.fixture(scope="module")
def official_dir() -> Path:
    """Sparse-clone the official beats/ source at the pinned commit and verify its identity."""
    repo_dir = _alignment_root() / "unilm"
    beats_dir = repo_dir / "beats"

    if not _official_files_valid(beats_dir):
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True)
        commands = (
            ("git", "init"),
            ("git", "remote", "add", "origin", _UNILM_URL),
            ("git", "sparse-checkout", "init", "--cone"),
            ("git", "sparse-checkout", "set", "beats"),
            (
                "git",
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                _UNILM_COMMIT,
            ),
            ("git", "checkout", "--detach", "FETCH_HEAD"),
        )
        for command in commands:
            subprocess.run(
                command,
                cwd=repo_dir,
                check=True,
                capture_output=True,
            )

    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_dir,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert head == _UNILM_COMMIT
    for name, expected in _OFFICIAL_FILE_SHA256.items():
        assert _sha256(beats_dir / name) == expected, name
    return beats_dir


@pytest.fixture(scope="module")
def official_beats(official_dir):
    """Load the BEATs module using the official top-level absolute-import semantics."""
    saved_modules = {
        name: sys.modules.pop(name, None)
        for name in _OFFICIAL_MODULE_NAMES
    }
    sys.path.insert(0, str(official_dir))
    try:
        module = importlib.import_module("BEATs")
        yield module
    finally:
        sys.path.remove(str(official_dir))
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved


@pytest.fixture(scope="module")
def checkpoint_paths() -> dict[str, Path]:
    """Resolve and verify all 15 checkpoints; fail with download guidance if any is missing."""
    paths = {}
    for entry in BEATS_CHECKPOINTS:
        try:
            paths[entry] = ensure_beats_checkpoint(entry, None)
        except (FileNotFoundError, ValueError) as error:
            pytest.fail(str(error))
    return paths


@pytest.fixture(scope="module", autouse=True)
def summary_writer():
    yield
    output_path = _alignment_root() / "beats-alignment-summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY["torch"] = torch.__version__
    _SUMMARY["cuda_tested"] = torch.cuda.is_available()
    output_path.write_text(
        json.dumps(_SUMMARY, ensure_ascii=False, indent=1)
    )


class _Tf32Off:
    """Temporarily disable TF32 on CUDA."""

    def __enter__(self):
        self._matmul = torch.backends.cuda.matmul.allow_tf32
        self._cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def __exit__(self, *args):
        torch.backends.cuda.matmul.allow_tf32 = self._matmul
        torch.backends.cudnn.allow_tf32 = self._cudnn


def test_transform_alignment(official_beats):
    """Full-matrix alignment of the local batched frontend against the official preprocess."""
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _TRANSFORM_TOLERANCES[device]
            transform = BeatsKaldiFbankTransform().to(device).eval()
            worst: dict = {}
            for seconds in _TRANSFORM_DURATIONS:
                num_samples = round(seconds * _TARGET_SR)
                for kind in _TRANSFORM_SIGNALS:
                    waveform = _make_signal(
                        kind,
                        num_samples,
                        seed=num_samples,
                    ).unsqueeze(0)
                    local = transform(
                        waveform.to(device),
                        sample_rate=_TARGET_SR,
                    )
                    reference = official_beats.BEATs.preprocess(
                        None,
                        _padded_for_official(waveform).to(device),
                    )
                    frames = int(local["valid_feature_frames"][0])
                    assert frames == reference.shape[1]
                    torch.testing.assert_close(
                        local["input_features"],
                        reference,
                        atol=atol,
                        rtol=rtol,
                    )
                    _merge_worst(
                        worst,
                        _audit(
                            local["input_features"],
                            reference,
                            device,
                        ),
                    )
            _SUMMARY["transform"][device] = worst


def _official_backbone_features(
    official_beats,
    checkpoint: dict,
    finetuned: bool,
) -> torch.nn.Module:
    """Build the official model; fine-tuned entries set predictor=None to export backbone features."""
    config = official_beats.BEATsConfig(checkpoint["cfg"])
    model = official_beats.BEATs(config)
    model.load_state_dict(checkpoint["model"])
    if finetuned:
        model.predictor = None
    return model.eval()


def _build_local_encoders(
    entry: str,
) -> tuple[BeatsEncoder, BeatsEncoder]:
    """Build the local frame and clip Encoders, sharing a single weight load."""
    frame_encoder = BeatsEncoder(
        granularity="frame",
        checkpoint=entry,
        pretrained=True,
    ).eval()
    clip_encoder = BeatsEncoder(
        granularity="clip",
        checkpoint=entry,
        pretrained=False,
    ).eval()
    clip_encoder.load_state_dict(frame_encoder.state_dict())
    return frame_encoder, clip_encoder


def _assert_case(
    transform,
    frame_encoder,
    clip_encoder,
    official_model,
    waveform: torch.Tensor,
    device: str,
    worst: dict,
) -> None:
    """Single waveform: align the local clip/frame outputs against the official backbone features."""
    local_features = transform(
        waveform.to(device),
        sample_rate=_TARGET_SR,
    )
    frames = int(local_features["valid_feature_frames"][0])
    time_blocks = frames // _PATCH_SIZE

    reference_tokens, _ = official_model.extract_features(
        _padded_for_official(waveform).to(device)
    )
    reference_frame = reference_tokens.view(
        1,
        time_blocks,
        _FREQ_PATCHES,
        -1,
    ).mean(dim=2)
    reference_clip = reference_tokens.mean(dim=1)

    frame_output = frame_encoder(
        local_features["input_features"],
        valid_seconds=local_features["valid_seconds"],
        valid_feature_frames=local_features["valid_feature_frames"],
    )
    clip_output = clip_encoder(
        local_features["input_features"],
        valid_seconds=local_features["valid_seconds"],
        valid_feature_frames=local_features["valid_feature_frames"],
    )

    atol, rtol = _ENCODER_TOLERANCES[device]
    local_frame = frame_output["embedding"][:, :time_blocks]
    torch.testing.assert_close(
        local_frame,
        reference_frame,
        atol=atol,
        rtol=rtol,
    )
    torch.testing.assert_close(
        clip_output["embedding"],
        reference_clip,
        atol=atol,
        rtol=rtol,
    )
    _merge_worst(
        worst,
        _audit(local_frame, reference_frame, device),
    )
    _merge_worst(
        worst,
        _audit(clip_output["embedding"], reference_clip, device),
    )


def test_encoder_alignment_all_entries(
    official_beats,
    checkpoint_paths,
):
    """Loading assertions and clip/frame numerical alignment for all 15 entries."""
    with _Tf32Off(), torch.inference_mode():
        for entry, checkpoint_path in checkpoint_paths.items():
            metadata = BEATS_CHECKPOINTS[entry]
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
            frame_encoder, clip_encoder = _build_local_encoders(entry)

            # Value-by-value weight-loading assertion (250 keys after
            # dropping predictor)
            local_state = frame_encoder.state_dict()
            checkpoint_state = {
                _POS_CONV_KEY_RENAMES.get(key, key): value
                for key, value in checkpoint["model"].items()
                if not key.startswith("predictor.")
            }
            assert set(local_state) == set(checkpoint_state)
            for key, value in checkpoint_state.items():
                assert torch.equal(local_state[key], value), key

            official_model = _official_backbone_features(
                official_beats,
                checkpoint,
                metadata.finetuned,
            )

            durations = (
                _TRANSFORM_DURATIONS
                if entry == _FULL_MATRIX_ENTRY
                else _ENCODER_DURATIONS
            )
            for device in _devices():
                transform = BeatsKaldiFbankTransform().to(device)
                transform.eval()
                frame_encoder.to(device)
                clip_encoder.to(device)
                official_model.to(device)
                worst: dict = {}
                for seconds in durations:
                    num_samples = round(seconds * _TARGET_SR)
                    for kind in _ENCODER_SIGNALS:
                        waveform = _make_signal(
                            kind,
                            num_samples,
                            seed=num_samples,
                        ).unsqueeze(0)
                        _assert_case(
                            transform,
                            frame_encoder,
                            clip_encoder,
                            official_model,
                            waveform,
                            device,
                            worst,
                        )
                _SUMMARY["encoder"].setdefault(entry, {})[device] = worst

            del frame_encoder, clip_encoder, official_model, checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def test_mixed_batch_alignment(official_beats, checkpoint_paths):
    """A representative entry's mixed-length and equal-length B=3 batches against the official per-sample reference."""
    entry = _FULL_MATRIX_ENTRY
    checkpoint = torch.load(
        checkpoint_paths[entry],
        map_location="cpu",
        weights_only=True,
    )
    frame_encoder, clip_encoder = _build_local_encoders(entry)
    official_model = _official_backbone_features(
        official_beats,
        checkpoint,
        BEATS_CHECKPOINTS[entry].finetuned,
    )

    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _ENCODER_TOLERANCES[device]
            transform = BeatsKaldiFbankTransform().to(device)
            transform.eval()
            frame_encoder.to(device)
            clip_encoder.to(device)
            official_model.to(device)
            worst: dict = {}

            for batch_durations in _BATCH_CASES:
                sample_counts = [
                    round(seconds * _TARGET_SR)
                    for seconds in batch_durations
                ]
                max_samples = max(sample_counts)
                batch = torch.full(
                    (len(sample_counts), max_samples),
                    _INVALID_TAIL_FILL,
                )
                rows = []
                for index, num_samples in enumerate(sample_counts):
                    row = _make_signal(
                        "random",
                        num_samples,
                        seed=1000 + index,
                    )
                    rows.append(row)
                    batch[index, :num_samples] = row
                valid_seconds = torch.tensor(
                    batch_durations,
                    dtype=torch.float32,
                )

                local_features = transform(
                    batch.to(device),
                    sample_rate=_TARGET_SR,
                    valid_seconds=valid_seconds.to(device),
                )
                frame_output = frame_encoder(
                    local_features["input_features"],
                    valid_seconds=local_features["valid_seconds"],
                    valid_feature_frames=local_features[
                        "valid_feature_frames"
                    ],
                )
                clip_output = clip_encoder(
                    local_features["input_features"],
                    valid_seconds=local_features["valid_seconds"],
                    valid_feature_frames=local_features[
                        "valid_feature_frames"
                    ],
                )

                for index, row in enumerate(rows):
                    frames = int(
                        local_features["valid_feature_frames"][index]
                    )
                    time_blocks = frames // _PATCH_SIZE
                    reference_tokens, _ = (
                        official_model.extract_features(
                            _padded_for_official(
                                row.unsqueeze(0)
                            ).to(device)
                        )
                    )
                    reference_frame = reference_tokens.view(
                        1,
                        time_blocks,
                        _FREQ_PATCHES,
                        -1,
                    ).mean(dim=2)
                    local_frame = frame_output["embedding"][
                        index : index + 1,
                        :time_blocks,
                    ]
                    torch.testing.assert_close(
                        local_frame,
                        reference_frame,
                        atol=atol,
                        rtol=rtol,
                    )
                    torch.testing.assert_close(
                        clip_output["embedding"][index : index + 1],
                        reference_tokens.mean(dim=1),
                        atol=atol,
                        rtol=rtol,
                    )
                    _merge_worst(
                        worst,
                        _audit(local_frame, reference_frame, device),
                    )
                    # Invalid frame slots are exact zeros
                    assert torch.all(
                        frame_output["embedding"][index, time_blocks:]
                        == 0
                    )
            _SUMMARY["mixed_batch"][device] = worst

    del frame_encoder, clip_encoder, official_model, checkpoint
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
