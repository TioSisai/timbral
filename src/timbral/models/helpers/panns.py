"""Types, checkpoint identity, and loading logic shared by PANNs components."""

from __future__ import annotations

import functools
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy._core.multiarray as _np_multiarray
import torch
from huggingface_hub.constants import HF_HUB_CACHE
from torch import Tensor

from .common import verify_file_sha256

PannsVariant: TypeAlias = Literal["max_mean", "decision_level_max"]
PannsTargetSampleRate: TypeAlias = Literal[16000, 32000]

PANNS_VARIANTS = frozenset(("max_mean", "decision_level_max"))
PANNS_TARGET_SAMPLE_RATES = frozenset((16000, 32000))

PANNS_OFFICIAL_FRONTENDS: dict[
    PannsTargetSampleRate, dict[str, int | float]
] = {
    16000: {
        "n_fft": 512,
        "win_length": 512,
        "hop_length": 160,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 8000.0,
    },
    32000: {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 320,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 14000.0,
    },
}


@dataclass(frozen=True, slots=True)
class PannsCheckpointMetadata:
    """Describes a fixed official PANNs checkpoint."""

    model_name: str
    filename: str
    url: str
    sha256: str
    requires_numpy_allowlist: bool


PANNS_CHECKPOINTS: dict[
    tuple[PannsTargetSampleRate, PannsVariant],
    PannsCheckpointMetadata,
] = {
    (32000, "max_mean"): PannsCheckpointMetadata(
        model_name="panns-32k-cnn14-max_mean",
        filename="Cnn14_mAP=0.431.pth",
        url=(
            "https://zenodo.org/records/3987831/files/"
            "Cnn14_mAP=0.431.pth"
        ),
        sha256=(
            "0dc499e40e9761ef5ea061ffc77697697"
            "f277f6a960894903df3ada000e34b31"
        ),
        requires_numpy_allowlist=False,
    ),
    (16000, "max_mean"): PannsCheckpointMetadata(
        model_name="panns-16k-cnn14-max_mean",
        filename="Cnn14_16k_mAP=0.438.pth",
        url=(
            "https://zenodo.org/records/3987831/files/"
            "Cnn14_16k_mAP=0.438.pth"
        ),
        sha256=(
            "e2ee543a27919542c2ea03eabaa70b24"
            "dcd4e6c8e05621de6b67a94e4c5058e6"
        ),
        requires_numpy_allowlist=True,
    ),
    (32000, "decision_level_max"): PannsCheckpointMetadata(
        model_name="panns-32k-cnn14-decision_level_max",
        filename="Cnn14_DecisionLevelMax_mAP=0.385.pth",
        url=(
            "https://zenodo.org/records/3987831/files/"
            "Cnn14_DecisionLevelMax_mAP=0.385.pth"
        ),
        sha256=(
            "dd3b4043a87d4ec13df8082c0fcfee3f"
            "b5084151808e47e060987a95eabdd142"
        ),
        requires_numpy_allowlist=False,
    ),
}

_CHECKPOINT_SAFE_GLOBALS = [
    (_np_multiarray._reconstruct, "numpy.core.multiarray._reconstruct"),
    np.ndarray,
    np.dtype,
    np.dtypes.Int64DType,
]


# Within the same process, (path, digest) is fully hashed only once:
# Transform and Encoder share the same checkpoint when constructed
# separately, avoiding reading a ~300 MB file twice.
_VERIFIED_CHECKPOINTS: set[tuple[str, str]] = set()


def _verify_checkpoint(path: Path, expected_sha256: str) -> None:
    """Verify a checkpoint digest (process-level memo; the same file is
    fully hashed only once).
    """
    key = (str(path), expected_sha256)
    if key in _VERIFIED_CHECKPOINTS:
        return
    verify_file_sha256(path, expected_sha256, label="checkpoint")
    _VERIFIED_CHECKPOINTS.add(key)


def ensure_panns_checkpoint(
    metadata: PannsCheckpointMetadata,
    pretrained_dir: str | Path | None,
) -> Path:
    """Resolve, download, and verify an official PANNs checkpoint.

    Args:
        metadata: Fixed official checkpoint metadata.
        pretrained_dir: Explicit checkpoint directory; when ``None``, uses
            the HF cache.

    Returns:
        The checkpoint path, verified against its SHA-256.
    """
    if pretrained_dir is None:
        directory = (
            Path(HF_HUB_CACHE)
            / "audioencoders"
            / metadata.model_name
        )
    else:
        directory = Path(pretrained_dir)
    checkpoint_path = directory / metadata.filename

    if checkpoint_path.exists():
        _verify_checkpoint(checkpoint_path, metadata.sha256)
        return checkpoint_path

    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{metadata.filename}.",
        suffix=".part",
        dir=directory,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)

    try:
        torch.hub.download_url_to_file(
            metadata.url,
            str(temporary_path),
            progress=True,
        )
        verify_file_sha256(temporary_path, metadata.sha256, label="checkpoint")
        os.replace(temporary_path, checkpoint_path)
        # The memo records the final path rather than the temp path, so a
        # checkpoint that was just downloaded hits the cache directly on a
        # second construction within the same process, without needing a
        # full re-hash.
        _VERIFIED_CHECKPOINTS.add((str(checkpoint_path), metadata.sha256))
    finally:
        temporary_path.unlink(missing_ok=True)

    return checkpoint_path


def load_panns_checkpoint_model(
    checkpoint_path: Path,
    *,
    requires_numpy_allowlist: bool,
) -> dict[str, Tensor]:
    """Load the checkpoint's model state in ``weights_only`` safe mode.

    Caches the most recent deserialization result at process level
    (maxsize=1): when Transform and Encoder are constructed back-to-back
    in create_model, the same checkpoint is ``torch.load``-ed only once;
    callers must only read the state dict, never modify it in place.

    Args:
        checkpoint_path: The checkpoint file to load.
        requires_numpy_allowlist: Whether to enable the minimal NumPy
            allowlist required by the 16 kHz checkpoint.

    Returns:
        The state dict corresponding to ``model`` in the checkpoint.
    """
    return _load_checkpoint_model_cached(
        str(checkpoint_path), requires_numpy_allowlist)


@functools.lru_cache(maxsize=1)
def _load_checkpoint_model_cached(
    checkpoint_path: str,
    requires_numpy_allowlist: bool,
) -> dict[str, Tensor]:
    """The actual implementation of ``load_panns_checkpoint_model`` (with
    maxsize=1 caching).
    """
    if requires_numpy_allowlist:
        with torch.serialization.safe_globals(_CHECKPOINT_SAFE_GLOBALS):
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
    else:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    return checkpoint["model"]


__all__ = (
    "PANNS_CHECKPOINTS",
    "PANNS_OFFICIAL_FRONTENDS",
    "PANNS_TARGET_SAMPLE_RATES",
    "PANNS_VARIANTS",
    "PannsCheckpointMetadata",
    "PannsTargetSampleRate",
    "PannsVariant",
    "ensure_panns_checkpoint",
    "load_panns_checkpoint_model",
)
