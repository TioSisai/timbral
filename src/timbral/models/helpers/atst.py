"""Checkpoint identity, archive-format normalization, and the frontend
geometry shared by the ATST-Clip and ATST-Frame components.

The two official families publish four checkpoints in two mutually
incompatible archive layouts: ATST-Clip small and both ATST-Frame
entries are PyTorch-Lightning checkpoints whose encoder weights live
under ``state_dict["model.teacher.encoder.*"]``, while ATST-Clip base is
an earlier DINO-style plain dict whose weights live under
``["teacher"]["module.backbone.*"]`` alongside a DINO projection head.
This module normalizes both into the one key set used by the local
Encoders, so the Encoders themselves never branch on archive format.
"""

from __future__ import annotations

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

AtstFamily: TypeAlias = Literal["clip", "frame"]
AtstArch: TypeAlias = Literal["small", "base"]
AtstArchiveFormat: TypeAlias = Literal["lightning", "dino"]

ATST_FAMILIES = frozenset(("clip", "frame"))
ATST_ARCHS = frozenset(("small", "base"))

# Frontend and patch geometry, identical across both families and both
# architectures (verified against the four official checkpoints: every
# pos_embed is [1, 251, D] and every patch_embed weight is [D, 256]).
ATST_TARGET_SAMPLE_RATE = 16000
ATST_HOP_LENGTH = 160
ATST_NUM_MELS = 64
ATST_PATCH_HEIGHT = 64
ATST_PATCH_WIDTH = 4
# The learned pos_embed holds 250 patch slots plus one cls slot, which
# caps a single forward pass at 250 patches; longer inputs are chunked.
ATST_CHUNK_PATCHES = 250
ATST_CHUNK_FRAMES = ATST_CHUNK_PATCHES * ATST_PATCH_WIDTH
ATST_POSITION_SLOTS = ATST_CHUNK_PATCHES + 1
# One patch spans 4 mel frames of 10 ms each.
ATST_FRAME_STEP_SECONDS = (
    ATST_PATCH_WIDTH * ATST_HOP_LENGTH / ATST_TARGET_SAMPLE_RATE
)
ATST_NUM_BLOCKS = 12
ATST_EMBED_DIMS: dict[AtstArch, int] = {"small": 384, "base": 768}
ATST_NUM_HEADS: dict[AtstArch, int] = {"small": 6, "base": 12}
# The official AmplitudeToDB(top_db=80) plus MinMax(min, max) rescaling
# to [-1, 1]; the constants are the AudioSet statistics hardcoded by the
# official transforms for n_mels=64.
ATST_TOP_DB = 80.0
ATST_NORM_MIN = -79.6482
ATST_NORM_MAX = 50.6842

_LIGHTNING_ENCODER_PREFIX = "model.teacher.encoder."
_DINO_ENCODER_PREFIX = "module.backbone."
# The DINO archive stores the projection head next to the backbone; it
# belongs to the pretraining objective, not to the encoder, and is
# dropped explicitly rather than tolerated via strict=False.
_DINO_DISCARDED_PREFIX = "module.head."

# Both archives pickle NumPy scalars (logged metrics and schedules);
# numpy.core is a compatibility shim in NumPy 2.x, so the allowlist maps
# the object under its pickled name (same technique as the PANNs
# helpers). The DINO archive additionally pickles its training args.
_CHECKPOINT_SAFE_GLOBALS = [
    (_np_multiarray.scalar, "numpy.core.multiarray.scalar"),
    np.dtype,
    np.dtypes.Float64DType,
]


@dataclass(frozen=True, slots=True)
class AtstCheckpointMetadata:
    """Describes one fixed official ATST checkpoint.

    Attributes:
        model_name: The registered name exposed by the registry.
        family: ``"clip"`` for ATST-Clip, ``"frame"`` for ATST-Frame.
        arch: ``"small"`` or ``"base"``.
        filename: The local filename under the weights directory.
        url: The official direct-download URL.
        sha256: The fixed digest of the official file.
        archive_format: Which archive layout the file uses.
        embed_dim: The backbone width ``D`` of a single block.
        num_heads: The attention head count of the architecture.
    """

    model_name: str
    family: AtstFamily
    arch: AtstArch
    filename: str
    url: str
    sha256: str
    archive_format: AtstArchiveFormat
    embed_dim: int
    num_heads: int


ATST_CHECKPOINTS: dict[
    tuple[AtstFamily, AtstArch], AtstCheckpointMetadata
] = {
    ("clip", "small"): AtstCheckpointMetadata(
        model_name="atst-clip-small",
        family="clip",
        arch="small",
        filename="atst_clip_small.ckpt",
        url=(
            "https://checkpointstorage.oss-cn-beijing.aliyuncs.com/"
            "atst/small.ckpt"
        ),
        sha256=(
            "fcadd6411881410d27cde47f4d540ef4"
            "16aa59e0197b195cf3ee7a81885a5f4a"
        ),
        archive_format="lightning",
        embed_dim=ATST_EMBED_DIMS["small"],
        num_heads=ATST_NUM_HEADS["small"],
    ),
    ("clip", "base"): AtstCheckpointMetadata(
        model_name="atst-clip-base",
        family="clip",
        arch="base",
        filename="atst_clip_base.ckpt",
        url=(
            "https://checkpointstorage.oss-cn-beijing.aliyuncs.com/"
            "atst/base.ckpt"
        ),
        sha256=(
            "7b20168cae0d1488a0e3334f17ca1cef"
            "b9365cbaa2401c11aa98d6ffaa668496"
        ),
        archive_format="dino",
        embed_dim=ATST_EMBED_DIMS["base"],
        num_heads=ATST_NUM_HEADS["base"],
    ),
    ("frame", "small"): AtstCheckpointMetadata(
        model_name="atst-frame-small",
        family="frame",
        arch="small",
        filename="atst_frame_small.ckpt",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1xZoOTuxV415icYONYbeFQzgrmJQf4a4B"
            "&export=download&confirm=t"
        ),
        sha256=(
            "1d85b290632dd26b8725f0ae73f53f99"
            "0a898888cfc2c4794c3055a8130ff5f1"
        ),
        archive_format="lightning",
        embed_dim=ATST_EMBED_DIMS["small"],
        num_heads=ATST_NUM_HEADS["small"],
    ),
    ("frame", "base"): AtstCheckpointMetadata(
        model_name="atst-frame-base",
        family="frame",
        arch="base",
        filename="atst_frame_base.ckpt",
        url=(
            "https://drive.usercontent.google.com/download"
            "?id=1bGJSZWlAIIJ6GL5Id5dW0PTB72DL-QDQ"
            "&export=download&confirm=t"
        ),
        sha256=(
            "9f812544983add849f45ef03c4ce1018"
            "4729adee082c0bdd347764f4835bb3da"
        ),
        archive_format="lightning",
        embed_dim=ATST_EMBED_DIMS["base"],
        num_heads=ATST_NUM_HEADS["base"],
    ),
}

ATST_CHECKPOINTS_BY_NAME: dict[str, AtstCheckpointMetadata] = {
    metadata.model_name: metadata
    for metadata in ATST_CHECKPOINTS.values()
}

# Within the same process, (path, digest) is fully hashed only once; the
# base entries are 1.4 GB files and rehashing them is measurable.
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


def ensure_atst_checkpoint(
    metadata: AtstCheckpointMetadata,
    pretrained_dir: str | Path | None,
) -> Path:
    """Resolve, download, and verify an official ATST checkpoint.

    The download lands on a temporary file inside the target directory
    and is moved to its final path only after the digest matches, so an
    interrupted or corrupted download never leaves a file that later
    constructions would accept.

    Args:
        metadata: Fixed official checkpoint metadata.
        pretrained_dir: Explicit checkpoint directory; when ``None``,
            uses ``audioencoders/atst`` under the HF cache.

    Returns:
        The checkpoint path, verified against its SHA-256.
    """
    if pretrained_dir is None:
        directory = Path(HF_HUB_CACHE) / "audioencoders" / "atst"
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
        verify_file_sha256(
            temporary_path, metadata.sha256, label="checkpoint")
        os.replace(temporary_path, checkpoint_path)
        _VERIFIED_CHECKPOINTS.add(
            (str(checkpoint_path), metadata.sha256))
    finally:
        temporary_path.unlink(missing_ok=True)

    return checkpoint_path


def load_atst_encoder_state(
    metadata: AtstCheckpointMetadata,
    checkpoint_path: Path,
) -> dict[str, Tensor]:
    """Load one checkpoint and normalize it to the local Encoder's keys.

    Both archive layouts are read in ``weights_only`` safe mode under a
    minimal allowlist. Only the teacher encoder subtree is kept: the
    student branch, the optimizer state, and (for the DINO layout) the
    projection head are dropped explicitly, so the caller can load with
    ``strict=True``.

    Args:
        metadata: Fixed official checkpoint metadata.
        checkpoint_path: The verified checkpoint file.

    Returns:
        A state dict whose keys match the local Encoder one-to-one.

    Raises:
        ValueError: The archive does not contain the expected teacher
            encoder subtree.
    """
    safe_globals = list(_CHECKPOINT_SAFE_GLOBALS)
    if metadata.archive_format == "dino":
        import argparse

        safe_globals.append(argparse.Namespace)

    with torch.serialization.safe_globals(safe_globals):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

    if metadata.archive_format == "lightning":
        archive = checkpoint.get("state_dict")
        if archive is None:
            raise ValueError(
                f"{checkpoint_path} is missing 'state_dict'; it does "
                "not look like the official Lightning checkpoint."
            )
        prefix = _LIGHTNING_ENCODER_PREFIX
    else:
        archive = checkpoint.get("teacher")
        if archive is None:
            raise ValueError(
                f"{checkpoint_path} is missing 'teacher'; it does not "
                "look like the official DINO-style checkpoint."
            )
        prefix = _DINO_ENCODER_PREFIX

    state = {
        key[len(prefix):]: value
        for key, value in archive.items()
        if key.startswith(prefix)
    }
    if not state:
        raise ValueError(
            f"{checkpoint_path} contains no tensor under {prefix!r}."
        )
    return state


def atst_feature_frames(valid_samples: Tensor) -> Tensor:
    """Return the mel frame count of each valid 16 kHz sample count.

    Mirrors ``torchaudio.transforms.MelSpectrogram`` with ``center=True``:
    a waveform of ``N`` samples yields ``N // hop + 1`` frames.

    Args:
        valid_samples: ``[B]`` valid sample counts at 16 kHz.

    Returns:
        ``[B]`` mel frame counts.
    """
    return torch.div(
        valid_samples, ATST_HOP_LENGTH, rounding_mode="floor"
    ) + 1


def atst_patch_frames(valid_feature_frames: Tensor) -> Tensor:
    """Return the patch count produced by each mel frame count.

    Patches tile the time axis in non-overlapping blocks of
    ``ATST_PATCH_WIDTH`` frames; a trailing remainder shorter than one
    patch is dropped, matching the official ``PatchEmbed_v2`` which
    slices the spectrogram to ``width - width % patch_width``.

    Args:
        valid_feature_frames: ``[B]`` mel frame counts.

    Returns:
        ``[B]`` patch counts.
    """
    return torch.div(
        valid_feature_frames, ATST_PATCH_WIDTH, rounding_mode="floor"
    )


__all__ = (
    "ATST_ARCHS",
    "ATST_CHECKPOINTS",
    "ATST_CHECKPOINTS_BY_NAME",
    "ATST_CHUNK_FRAMES",
    "ATST_CHUNK_PATCHES",
    "ATST_EMBED_DIMS",
    "ATST_FAMILIES",
    "ATST_FRAME_STEP_SECONDS",
    "ATST_HOP_LENGTH",
    "ATST_NORM_MAX",
    "ATST_NORM_MIN",
    "ATST_NUM_BLOCKS",
    "ATST_NUM_HEADS",
    "ATST_NUM_MELS",
    "ATST_PATCH_HEIGHT",
    "ATST_PATCH_WIDTH",
    "ATST_POSITION_SLOTS",
    "ATST_TARGET_SAMPLE_RATE",
    "ATST_TOP_DB",
    "AtstArch",
    "AtstArchiveFormat",
    "AtstCheckpointMetadata",
    "AtstFamily",
    "atst_feature_frames",
    "atst_patch_frames",
    "ensure_atst_checkpoint",
    "load_atst_encoder_state",
)
