"""Digest verification, snapshot download, and numeric utilities shared
by the four model helper groups.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from huggingface_hub.constants import HF_HUB_CACHE


def sha256_file(path: Path) -> str:
    """Compute the SHA-256 of a file.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal digest string.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: Path, expected_sha256: str, *,
                       label: str) -> None:
    """Verify the SHA-256 of a file.

    Args:
        path: File to verify.
        expected_sha256: Fixed hexadecimal digest.
        label: File identity prefix used in the error message (e.g.
            ``"checkpoint"``).

    Raises:
        ValueError: The file digest does not match the fixed identity.
    """
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{label} SHA-256 mismatch: {path}, "
            f"expected {expected_sha256}, got {actual_sha256}."
        )


def ensure_hf_snapshot(
    *,
    repo_id: str,
    revision: str,
    filenames: tuple[str, ...],
    sha256: dict[str, str],
    pretrained_dir: str | Path | None,
    label: str,
) -> Path:
    """Prepare and verify a fixed Hugging Face snapshot (shared by AST/CLAP).

    Missing files are first downloaded into a temporary directory inside
    the snapshot directory, then atomically moved to their final path
    after verification succeeds; a corrupt download or an interrupted
    leftover never ends up at the final path, which would otherwise cause
    later construction to hang permanently.

    Args:
        repo_id: Hugging Face repository name.
        revision: Fixed revision.
        filenames: Set of required filenames.
        sha256: Fixed digest for each file.
        pretrained_dir: Explicit snapshot directory; when ``None``, uses
            the project-specific directory ``audioencoders/{repo_id}``
            under the HF cache.
        label: File identity prefix used in digest error messages.

    Returns:
        The directory containing all fixed files, each verified against
        its digest.
    """
    if pretrained_dir is None:
        directory = Path(HF_HUB_CACHE) / "audioencoders" / repo_id
    else:
        directory = Path(pretrained_dir)
    directory.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        path = directory / filename
        if path.exists():
            verify_file_sha256(path, sha256[filename], label=label)
            continue
        with tempfile.TemporaryDirectory(
            prefix=".download-", dir=directory
        ) as download_dir:
            downloaded_path = Path(hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                local_dir=download_dir,
            ))
            verify_file_sha256(downloaded_path, sha256[filename], label=label)
            os.replace(downloaded_path, path)
    return directory


def round_positive_ratio(numerator: int, denominator: int) -> int:
    """Round a positive rational to the nearest integer using integer
    arithmetic (ties round to even).

    Matches the ties-to-even semantics of ``torch.round``. Used to convert
    a physical sample count exactly to a target sample rate, avoiding
    dropped samples on very long waveforms that would result from a
    float32 seconds round-trip.

    Args:
        numerator: Positive integer numerator.
        denominator: Positive integer denominator.

    Returns:
        The ties-to-even rounding result of ``numerator / denominator``.
    """
    quotient, remainder = divmod(numerator, denominator)
    doubled_remainder = 2 * remainder
    if doubled_remainder > denominator or (
        doubled_remainder == denominator and quotient % 2 == 1
    ):
        return quotient + 1
    return quotient


__all__ = (
    "ensure_hf_snapshot",
    "round_positive_ratio",
    "sha256_file",
    "verify_file_sha256",
)
