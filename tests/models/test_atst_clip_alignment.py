"""Alignment tests for ATST-Clip against the official implementation (requires explicit ``--run-alignment atst_clip``).

The reference side is the official ``audiossl`` source at a pinned commit
of Audio-WestlakeU/audiossl (sparse-cloned into ``$TMPDIR``, SHA-256
verified per file, then imported as a normal package because its modules
use absolute intra-package imports). The official checkpoints ship in two
archive layouts, and both are exercised here: ``atst-clip-small`` is a
Lightning checkpoint while ``atst-clip-base`` is the earlier DINO-style
dict.

This module also covers the log-mel frontend, which both ATST families
share as one and the same ``AtstMelspecTransform`` class; the frame
alignment module therefore does not repeat it. See
``docs/designs/models/extra/atst_clip-alignment.md`` for the contract.
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
from pathlib import Path

import pytest
import torch
import torchaudio
from torch.nn import functional as F

from timbral.models.encoders import AtstClipEncoder
from timbral.models.helpers.atst import (
    ATST_CHECKPOINTS,
    ATST_CHUNK_FRAMES,
    ATST_NORM_MAX,
    ATST_NORM_MIN,
    ATST_NUM_BLOCKS,
    ensure_atst_checkpoint,
    load_atst_encoder_state,
)
from timbral.models.transforms import AtstMelspecTransform

pytestmark = pytest.mark.alignment("atst_clip")

_AUDIOSSL_URL = "https://github.com/Audio-WestlakeU/audiossl"
_AUDIOSSL_COMMIT = "ec3a14db086eaccfb69513e4a90fad89bf992e1f"
# Only the files the reference path actually imports are pinned. The
# three package directories below ship without __init__.py and are
# resolved as PEP 420 namespace packages, so they have nothing to pin.
_OFFICIAL_FILE_SHA256 = {
    "audiossl/__init__.py": (
        "c3edaf9c65e45b430aa06d4617abfb1b"
        "01b24c5c09f3a7e050eef021386b0d0b"
    ),
    "audiossl/models/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    ),
    "audiossl/models/atst/__init__.py": (
        "a825bbc89d2806d15fecb91d43f45a82"
        "946018c7b55b74c2fa0e3e19ff0e4388"
    ),
    "audiossl/models/atst/atst.py": (
        "c5e1bd0a622d7feb0cc5f6f7e8f3b742"
        "7d7033d4133e7aeec14dc9f28e186c59"
    ),
    "audiossl/models/atst/byol.py": (
        "3060ccef3662f5bf02d50b3048198ceb"
        "909c9057cd3007b42561e0f19d33b423"
    ),
    "audiossl/models/atst/audio_transformer.py": (
        "3ea4694fedd2d7ea1fb125440a81f5e4"
        "1a55f680a2f4a1ceb0237b32542fbb16"
    ),
    "audiossl/modules/transformer.py": (
        "0048be36e211bfb00c177331a662b455"
        "e644a950424dc58406fd072a9f718297"
    ),
    "audiossl/transforms/__init__.py": (
        "e3b0c44298fc1c149afbf4c8996fb924"
        "27ae41e4649b934ca495991b7852b855"
    ),
    "audiossl/transforms/common.py": (
        "9f99a260fef70da529455fec4b12f0ff"
        "fd297dab960ceec0e36148bc22ab49a1"
    ),
}
_SPARSE_DIRECTORIES = (
    "audiossl/models",
    "audiossl/modules",
    "audiossl/transforms",
)
_OFFICIAL_MODULE_PREFIX = "audiossl"

_TARGET_SR = 16000
# The official frontend cannot process a waveform shorter than the
# reflect padding it applies, so the reference receives the same
# zero-padded waveform the local transform builds internally.
_MIN_TARGET_SAMPLES = 513
_OFFICIAL_CHUNK_LEN = ATST_CHUNK_FRAMES + 1

_TRANSFORM_TOLERANCES = {"cpu": (1e-4, 1e-4), "cuda": (2e-3, 1e-4)}
_ENCODER_TOLERANCES = {"cpu": (1e-4, 1e-4), "cuda": (2e-3, 1e-4)}
_AUDIT_RELATIVE_L2 = {"cpu": 1e-4, "cuda": 1e-3}
_MIN_COSINE = 0.99999

_TRANSFORM_DURATIONS = (0.01, 0.2, 1.0, 6.0, 10.0, 12.5)
_TRANSFORM_SIGNALS = ("random", "sine", "impulse", "multisine", "silence")
# Single-chunk and multi-chunk durations: 10 s is exactly one chunk,
# anything beyond it exercises the chunk seam.
_ENCODER_DURATIONS = (1.0, 10.0)
_MULTI_CHUNK_DURATIONS = (10.05, 21.0)
_BATCH_CASE = (0.5, 6.0, 10.0)

_SUMMARY: dict = {
    "audiossl_commit": _AUDIOSSL_COMMIT,
    "official_file_sha256": _OFFICIAL_FILE_SHA256,
    "transform": {},
    "encoder": {},
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
    """Apply the same zero-padding floor the local transform applies."""
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
    bucket["max_abs"] = max(bucket.get("max_abs", 0.0), metrics["max_abs"])
    bucket["relative_l2"] = max(
        bucket.get("relative_l2", 0.0), metrics["relative_l2"])
    bucket["cosine"] = min(bucket.get("cosine", 1.0), metrics["cosine"])
    bucket["cases"] = bucket.get("cases", 0) + 1


@pytest.fixture(scope="module")
def official_dir() -> Path:
    """Sparse-clone the official audiossl source at the pinned commit and verify its identity."""
    repo_dir = _alignment_root() / "audiossl"

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
    """Import the official package with its own absolute-import semantics."""
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == _OFFICIAL_MODULE_PREFIX
        or name.startswith(f"{_OFFICIAL_MODULE_PREFIX}.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, str(official_dir))
    try:
        yield importlib.import_module(
            "audiossl.models.atst.audio_transformer")
    finally:
        sys.path.remove(str(official_dir))
        for name in [
            name
            for name in sys.modules
            if name == _OFFICIAL_MODULE_PREFIX
            or name.startswith(f"{_OFFICIAL_MODULE_PREFIX}.")
        ]:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


@pytest.fixture(scope="module")
def official_frontend(official_audiossl):
    """Rebuild the official frontend chain from the pinned source.

    Depends on ``official_audiossl`` so that the pinned checkout stays
    on ``sys.path`` while ``audiossl.transforms.common`` is imported.
    """
    common = importlib.import_module("audiossl.transforms.common")
    melspec = torchaudio.transforms.MelSpectrogram(
        _TARGET_SR,
        f_min=60,
        f_max=7800,
        hop_length=160,
        win_length=1024,
        n_fft=1024,
        n_mels=64,
    )
    to_db = torchaudio.transforms.AmplitudeToDB(
        stype="power", top_db=80)
    normalize = common.MinMax(min=ATST_NORM_MIN, max=ATST_NORM_MAX)
    return melspec, to_db, normalize


@pytest.fixture(scope="module", autouse=True)
def summary_writer():
    yield
    output_path = _alignment_root() / "atst-clip-alignment-summary.json"
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
    """Build the official AST and load the pinned checkpoint into it."""
    factory = module.AST_small if arch == "small" else module.AST_base
    model = factory()
    metadata = ATST_CHECKPOINTS[("clip", arch)]
    state = load_atst_encoder_state(
        metadata, ensure_atst_checkpoint(metadata, None))
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def _official_chunk(model, features: torch.Tensor) -> torch.Tensor:
    """Run one chunk through the official chunked extraction path.

    ``chunk_len`` is set one frame above the local chunk width so that a
    chunk of at most ``ATST_CHUNK_FRAMES`` frames is processed by the
    official code as a single chunk; what is compared is therefore the
    official single-chunk forward pass, with the local chunk splitting
    and cross-chunk combination assembled by the test itself.
    """
    mel = features.transpose(1, 2).unsqueeze(1)
    length = torch.full(
        (features.shape[0],),
        features.shape[1],
        dtype=torch.long,
        device=features.device,
    )
    return model.get_intermediate_layers_chunks(
        mel,
        length,
        ATST_NUM_BLOCKS,
        _OFFICIAL_CHUNK_LEN,
        avgpool=True,
    )


def test_transform_alignment(official_frontend):
    """The local batched frontend matches the official mel/dB/MinMax chain."""
    melspec, to_db, normalize = official_frontend
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _TRANSFORM_TOLERANCES[device]
            transform = AtstMelspecTransform().to(device).eval()
            reference_melspec = melspec.to(device)
            worst: dict = {}
            for seconds in _TRANSFORM_DURATIONS:
                num_samples = round(seconds * _TARGET_SR)
                for kind in _TRANSFORM_SIGNALS:
                    waveform = _make_signal(
                        kind, num_samples, seed=num_samples).unsqueeze(0)
                    local = transform(
                        waveform.to(device), sample_rate=_TARGET_SR)
                    # The official chain is fed [B, 1, N] so that its
                    # top_db floor reduces per sample, matching the
                    # official embedding.py entry point.
                    padded = _padded_for_official(waveform).to(device)
                    reference = normalize(
                        to_db(reference_melspec(padded.unsqueeze(1)))
                    ).squeeze(1)
                    frames = int(local["valid_feature_frames"][0])
                    assert reference.shape[-1] == frames
                    local_features = local["input_features"][0, :frames]
                    reference_features = reference[0].transpose(0, 1)
                    assert torch.allclose(
                        local_features,
                        reference_features,
                        atol=atol,
                        rtol=rtol,
                    )
                    _merge_worst(
                        worst,
                        _audit(
                            local_features, reference_features, device),
                    )
            _SUMMARY["transform"][device] = worst


@pytest.mark.parametrize("arch", ("small", "base"))
def test_encoder_alignment_single_chunk(official_audiossl, arch):
    """Within one chunk the local encoder matches the official extraction."""
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _ENCODER_TOLERANCES[device]
            transform = AtstMelspecTransform().to(device).eval()
            encoder = AtstClipEncoder(
                granularity="clip",
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
                    reference_model, features["input_features"])
                assert local.shape == reference.shape
                assert torch.allclose(
                    local, reference, atol=atol, rtol=rtol)
                _merge_worst(
                    worst, _audit(local, reference, device))
            _SUMMARY["encoder"][f"{arch}-{device}"] = worst


@pytest.mark.parametrize("arch", ("small",))
def test_encoder_alignment_multi_chunk(official_audiossl, arch):
    """Across chunks the local encoder equals the official per-chunk results combined by patch count.

    The local chunk grid (1000 mel frames, i.e. exactly the 250 patch
    slots of pos_embed) and the patch-count weighting are the
    repository's own decisions, so the reference is assembled here: each
    chunk is run through the official single-chunk path, and the results
    are combined in proportion to the patches each chunk holds. The
    official equal-weight average is asserted to differ, which pins the
    deviation rather than letting it pass silently.
    """
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _ENCODER_TOLERANCES[device]
            transform = AtstMelspecTransform().to(device).eval()
            encoder = AtstClipEncoder(
                granularity="clip",
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

                chunk_outputs = []
                chunk_patches = []
                for start in range(0, frames, ATST_CHUNK_FRAMES):
                    end = min(start + ATST_CHUNK_FRAMES, frames)
                    if (end - start) < 4:
                        continue
                    chunk_outputs.append(
                        _official_chunk(
                            reference_model,
                            features["input_features"][:, start:end],
                        )
                    )
                    chunk_patches.append((end - start) // 4)
                assert len(chunk_outputs) > 1
                stacked = torch.stack(chunk_outputs, dim=0)
                weights = stacked.new_tensor(chunk_patches)
                weights = weights / weights.sum()
                reference = (stacked * weights[:, None, None]).sum(dim=0)
                assert torch.allclose(
                    local, reference, atol=atol, rtol=rtol)
                if min(chunk_patches) != max(chunk_patches):
                    # The official rule weights every chunk equally; with
                    # uneven chunks that is a different vector.
                    assert not torch.allclose(
                        local, stacked.mean(dim=0), atol=atol, rtol=rtol)
                _merge_worst(
                    worst, _audit(local, reference, device))
            _SUMMARY["multi_chunk"][f"{arch}-{device}"] = worst


def test_mixed_batch_matches_single_calls(official_audiossl):
    """A mixed-length batch reproduces each sample's standalone official result."""
    arch = "small"
    with _Tf32Off(), torch.inference_mode():
        for device in _devices():
            atol, rtol = _ENCODER_TOLERANCES[device]
            transform = AtstMelspecTransform().to(device).eval()
            encoder = AtstClipEncoder(
                granularity="clip",
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
            local = encoder(
                features["input_features"],
                valid_seconds=features["valid_seconds"],
                valid_feature_frames=features["valid_feature_frames"],
            )["embedding"]

            worst: dict = {}
            for index, seconds in enumerate(_BATCH_CASE):
                single = transform(
                    waveform[index : index + 1],
                    sample_rate=_TARGET_SR,
                    valid_seconds=valid_seconds[index : index + 1],
                )
                reference = _official_chunk(
                    reference_model, single["input_features"])
                assert torch.allclose(
                    local[index : index + 1],
                    reference,
                    atol=atol,
                    rtol=rtol,
                )
                _merge_worst(
                    worst,
                    _audit(local[index : index + 1], reference, device),
                )
            _SUMMARY["mixed_batch"][device] = worst
