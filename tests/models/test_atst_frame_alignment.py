"""Alignment tests for ATST-Frame against the official implementation (requires explicit ``--run-alignment atst_frame``).

The reference side is the official ``audiossl`` source at the same
pinned commit used by the ATST-Clip alignment module, sparse-cloned into
``$TMPDIR`` and SHA-256 verified per file. The official
``methods/atstframe/audio_transformer`` imports ``random_mask``, whose
first line pulls in ``fairseq`` for pretraining-time mask generation; a
stub module is injected into ``sys.modules`` so the import resolves
without altering a single byte of the official source. The stub's
function body raises, which asserts as a side effect that the inference
path never touches it.

The log-mel frontend is a single class shared with ATST-Clip and is
covered by ``test_atst_clip_alignment.py``; it is not repeated here. See
``docs/designs/models/extra/atst_frame-alignment.md`` for the contract.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from timbral.models.encoders import AtstFrameEncoder
from timbral.models.helpers.atst import (
    ATST_CHECKPOINTS,
    ATST_CHUNK_FRAMES,
    ATST_NUM_BLOCKS,
    ATST_PATCH_WIDTH,
    ensure_atst_checkpoint,
    load_atst_encoder_state,
)
from timbral.models.transforms import AtstMelspecTransform

pytestmark = pytest.mark.alignment("atst_frame")

_AUDIOSSL_URL = "https://github.com/Audio-WestlakeU/audiossl"
_AUDIOSSL_COMMIT = "ec3a14db086eaccfb69513e4a90fad89bf992e1f"
# The two atstframe package directories ship without __init__.py and are
# resolved as PEP 420 namespace packages, so they have nothing to pin.
_OFFICIAL_FILE_SHA256 = {
    "audiossl/__init__.py": (
        "c3edaf9c65e45b430aa06d4617abfb1b"
        "01b24c5c09f3a7e050eef021386b0d0b"
    ),
    "audiossl/methods/atstframe/audio_transformer.py": (
        "33554493407306db8efdba41082c5877"
        "0d021f5c9bf3e8a3c465cf1850e719bc"
    ),
    "audiossl/methods/atstframe/random_mask.py": (
        "ff6c335138e80578f2f5085c6c5066f1"
        "9f8d6a592795691d68d9aa2f76bb6507"
    ),
    "audiossl/modules/transformer.py": (
        "0048be36e211bfb00c177331a662b455"
        "e644a950424dc58406fd072a9f718297"
    ),
}
_SPARSE_DIRECTORIES = (
    "audiossl/methods/atstframe",
    "audiossl/modules",
)
_OFFICIAL_MODULE_PREFIX = "audiossl"
_FAIRSEQ_STUB_MODULES = (
    "fairseq",
    "fairseq.data",
    "fairseq.data.data_utils",
)

_TARGET_SR = 16000
_ENCODER_TOLERANCES = {"cpu": (1e-4, 1e-4), "cuda": (2e-3, 1e-4)}
_AUDIT_RELATIVE_L2 = {"cpu": 1e-4, "cuda": 1e-3}
_MIN_COSINE = 0.99999

_ENCODER_DURATIONS = (1.0, 10.0)
_MULTI_CHUNK_DURATIONS = (10.05, 21.0)
_BATCH_CASE = (0.5, 6.0, 10.0)

_SUMMARY: dict = {
    "audiossl_commit": _AUDIOSSL_COMMIT,
    "official_file_sha256": _OFFICIAL_FILE_SHA256,
    "frame": {},
    "clip": {},
    "multi_chunk": {},
    "mixed_batch": {},
}


def _alignment_root() -> Path:
    return (
        Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        / "timbral-atst-alignment"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _official_files_valid(repo_dir: Path) -> bool:
    return all(
        (repo_dir / name).exists()
        and _sha256(repo_dir / name) == expected
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
    if kind == "multisine":
        return (
            0.4 * torch.sin(2 * torch.pi * 440.0 * time_axis)
            + 0.3 * torch.sin(2 * torch.pi * 1237.0 * time_axis)
            + 0.2 * torch.cos(2 * torch.pi * 3313.0 * time_axis)
        )
    raise ValueError(kind)


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
    bucket["max_abs"] = max(bucket.get("max_abs", 0.0), metrics["max_abs"])
    bucket["relative_l2"] = max(
        bucket.get("relative_l2", 0.0), metrics["relative_l2"])
    bucket["cosine"] = min(bucket.get("cosine", 1.0), metrics["cosine"])
    bucket["cases"] = bucket.get("cases", 0) + 1


def _make_fairseq_stub() -> dict[str, types.ModuleType]:
    """Build the minimal fairseq stand-in the official import needs.

    ``random_mask`` is a pretraining-only helper; the inference path
    never calls it, which the raising body below turns into an
    assertion.
    """
    def compute_mask_indices(*args, **kwargs):
        raise AssertionError(
            "fairseq.compute_mask_indices is pretraining-only and must "
            "not be reached by the inference path."
        )

    fairseq = types.ModuleType("fairseq")
    fairseq.__path__ = []
    data = types.ModuleType("fairseq.data")
    data.__path__ = []
    data_utils = types.ModuleType("fairseq.data.data_utils")
    data_utils.compute_mask_indices = compute_mask_indices
    data.data_utils = data_utils
    fairseq.data = data
    return {
        "fairseq": fairseq,
        "fairseq.data": data,
        "fairseq.data.data_utils": data_utils,
    }


@pytest.fixture(scope="module")
def official_dir() -> Path:
    """Sparse-clone the official audiossl source at the pinned commit and verify its identity."""
    repo_dir = _alignment_root() / "audiossl-frame"

    if not _official_files_valid(repo_dir):
        if repo_dir.exists():
            shutil.rmtree(repo_dir)
        repo_dir.mkdir(parents=True)
        commands = (
            ("git", "init"),
            ("git", "remote", "add", "origin", _AUDIOSSL_URL),
            ("git", "sparse-checkout", "init", "--cone"),
            ("git", "sparse-checkout", "set", *_SPARSE_DIRECTORIES),
            (
                "git",
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                _AUDIOSSL_COMMIT,
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
    assert head == _AUDIOSSL_COMMIT
    for name, expected in _OFFICIAL_FILE_SHA256.items():
        assert _sha256(repo_dir / name) == expected, name
    return repo_dir


@pytest.fixture(scope="module")
def official_audiossl(official_dir):
    """Import the official FrameAST module under a stubbed fairseq."""
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == _OFFICIAL_MODULE_PREFIX
        or name.startswith(f"{_OFFICIAL_MODULE_PREFIX}.")
        or name in _FAIRSEQ_STUB_MODULES
    }
    for name in saved_modules:
        sys.modules.pop(name, None)
    sys.modules.update(_make_fairseq_stub())

    sys.path.insert(0, str(official_dir))
    try:
        yield importlib.import_module(
            "audiossl.methods.atstframe.audio_transformer")
    finally:
        sys.path.remove(str(official_dir))
        for name in [
            name
            for name in sys.modules
            if name == _OFFICIAL_MODULE_PREFIX
            or name.startswith(f"{_OFFICIAL_MODULE_PREFIX}.")
            or name in _FAIRSEQ_STUB_MODULES
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


@pytest.fixture(scope="module", autouse=True)
def summary_writer():
    yield
    output_path = _alignment_root() / "atst-frame-alignment-summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _SUMMARY["torch"] = torch.__version__
    _SUMMARY["cuda_tested"] = torch.cuda.is_available()
    output_path.write_text(
        json.dumps(_SUMMARY, ensure_ascii=False, indent=1))


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


def _official_model(module, arch: str, device: str):
    """Build the official FrameAST and load the pinned checkpoint."""
    factory = (
        module.FrameAST_small if arch == "small" else module.FrameAST_base
    )
    model = factory()
    metadata = ATST_CHECKPOINTS[("frame", arch)]
    state = load_atst_encoder_state(
        metadata, ensure_atst_checkpoint(metadata, None))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _official_chunk(
    model,
    features: torch.Tensor,
    *,
    scene: bool,
) -> torch.Tensor:
    """Run one chunk through the official extraction path."""
    mel = features.transpose(1, 2).unsqueeze(1)
    length = torch.full(
        (features.shape[0],),
        features.shape[1],
        dtype=torch.long,
        device=features.device,
    )
    return model.get_intermediate_layers(
        mel, length, n=ATST_NUM_BLOCKS, scene=scene)


def _chunk_bounds(num_frames: int) -> list[tuple[int, int]]:
    """Mirror the encoder's chunk split, for building the reference."""
    bounds = []
    for start in range(0, num_frames, ATST_CHUNK_FRAMES):
        end = min(start + ATST_CHUNK_FRAMES, num_frames)
        if (end - start) >= ATST_PATCH_WIDTH:
            bounds.append((start, end))
    return bounds


@pytest.mark.parametrize("arch", ("small", "base"))
@pytest.mark.parametrize("granularity", ("frame", "clip"))
def test_encoder_alignment_single_chunk(
    official_audiossl, arch, granularity
):
    """Within one chunk the local encoder matches the official extraction."""
    scene = granularity == "clip"
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _ENCODER_TOLERANCES[device]
            transform = AtstMelspecTransform().to(device).eval()
            encoder = AtstFrameEncoder(
                granularity=granularity,
                arch=arch,
                n_blocks=ATST_NUM_BLOCKS,
                pretrained=True,
            ).to(device).eval()
            reference_model = _official_model(
                official_audiossl, arch, device)
            worst: dict = {}
            for seconds in _ENCODER_DURATIONS:
                waveform = _make_signal(
                    "random",
                    round(seconds * _TARGET_SR),
                    seed=int(seconds * 1000),
                ).unsqueeze(0).to(device)
                features = transform(waveform, sample_rate=_TARGET_SR)
                local = encoder(
                    features["input_features"],
                    valid_seconds=features["valid_seconds"],
                    valid_feature_frames=features["valid_feature_frames"],
                )["embedding"]
                reference = _official_chunk(
                    reference_model,
                    features["input_features"],
                    scene=scene,
                )
                assert local.shape == reference.shape
                assert torch.allclose(
                    local, reference, atol=atol, rtol=rtol)
                _merge_worst(worst, _audit(local, reference, device))
            _SUMMARY[granularity][f"{arch}-{device}"] = worst


@pytest.mark.parametrize("granularity", ("frame", "clip"))
def test_encoder_alignment_multi_chunk(official_audiossl, granularity):
    """Across chunks the local encoder equals the assembled official result.

    Frame granularity concatenates the per-chunk token sequences along
    time; clip granularity combines the per-chunk vectors in proportion
    to the patches each chunk holds. The 1000-frame chunk grid and that
    weighting are this repository's decisions, so the reference is
    assembled here from official single-chunk calls, and the official
    equal-weight average is asserted to differ.
    """
    arch = "small"
    scene = granularity == "clip"
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _ENCODER_TOLERANCES[device]
            transform = AtstMelspecTransform().to(device).eval()
            encoder = AtstFrameEncoder(
                granularity=granularity,
                arch=arch,
                n_blocks=ATST_NUM_BLOCKS,
                pretrained=True,
            ).to(device).eval()
            reference_model = _official_model(
                official_audiossl, arch, device)
            worst: dict = {}
            for seconds in _MULTI_CHUNK_DURATIONS:
                waveform = _make_signal(
                    "multisine",
                    round(seconds * _TARGET_SR),
                    seed=int(seconds * 100),
                ).unsqueeze(0).to(device)
                features = transform(waveform, sample_rate=_TARGET_SR)
                frames = int(features["valid_feature_frames"][0])
                local = encoder(
                    features["input_features"],
                    valid_seconds=features["valid_seconds"],
                    valid_feature_frames=features["valid_feature_frames"],
                )["embedding"]

                chunk_outputs = [
                    _official_chunk(
                        reference_model,
                        features["input_features"][:, start:end],
                        scene=scene,
                    )
                    for start, end in _chunk_bounds(frames)
                ]
                assert len(chunk_outputs) > 1
                if scene:
                    stacked = torch.stack(chunk_outputs, dim=0)
                    chunk_patches = [
                        (end - start) // ATST_PATCH_WIDTH
                        for start, end in _chunk_bounds(frames)
                    ]
                    weights = stacked.new_tensor(chunk_patches)
                    weights = weights / weights.sum()
                    reference = (
                        stacked * weights[:, None, None]).sum(dim=0)
                    if min(chunk_patches) != max(chunk_patches):
                        assert not torch.allclose(
                            local, stacked.mean(dim=0),
                            atol=atol, rtol=rtol)
                else:
                    reference = torch.cat(chunk_outputs, dim=1)
                    assert reference.shape[1] == frames // ATST_PATCH_WIDTH
                assert local.shape == reference.shape
                assert torch.allclose(
                    local, reference, atol=atol, rtol=rtol)
                _merge_worst(worst, _audit(local, reference, device))
            _SUMMARY["multi_chunk"][f"{granularity}-{device}"] = worst


def test_mixed_batch_matches_single_calls(official_audiossl):
    """A mixed-length batch reproduces each sample's standalone official result."""
    arch = "small"
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _ENCODER_TOLERANCES[device]
            transform = AtstMelspecTransform().to(device).eval()
            encoder = AtstFrameEncoder(
                granularity="frame",
                arch=arch,
                n_blocks=ATST_NUM_BLOCKS,
                pretrained=True,
            ).to(device).eval()
            reference_model = _official_model(
                official_audiossl, arch, device)

            longest = round(max(_BATCH_CASE) * _TARGET_SR)
            waveform = torch.stack([
                F.pad(
                    _make_signal(
                        "random",
                        round(seconds * _TARGET_SR),
                        seed=int(seconds * 10),
                    ),
                    (0, longest - round(seconds * _TARGET_SR)),
                )
                for seconds in _BATCH_CASE
            ]).to(device)
            valid_seconds = torch.tensor(
                _BATCH_CASE, dtype=torch.float32, device=device)
            features = transform(
                waveform,
                sample_rate=_TARGET_SR,
                valid_seconds=valid_seconds,
            )
            output = encoder(
                features["input_features"],
                valid_seconds=features["valid_seconds"],
                valid_feature_frames=features["valid_feature_frames"],
            )
            local = output["embedding"]
            valid_mask = output["valid_mask"]

            worst: dict = {}
            for index, seconds in enumerate(_BATCH_CASE):
                single = transform(
                    waveform[index : index + 1],
                    sample_rate=_TARGET_SR,
                    valid_seconds=valid_seconds[index : index + 1],
                )
                reference = _official_chunk(
                    reference_model,
                    single["input_features"],
                    scene=False,
                )
                valid_frames = int(valid_mask[index].sum())
                assert valid_frames == reference.shape[1]
                local_valid = local[index, :valid_frames]
                assert torch.allclose(
                    local_valid, reference[0], atol=atol, rtol=rtol)
                # Everything past the valid region must be exactly zero.
                assert torch.equal(
                    local[index, valid_frames:],
                    torch.zeros_like(local[index, valid_frames:]),
                )
                _merge_worst(
                    worst, _audit(local_valid, reference[0], device))
            _SUMMARY["mixed_batch"][device] = worst
