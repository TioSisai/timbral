"""Data preparation config resolution: default path derivation, default split generation, and config hash computation."""

import dataclasses
import os
from pathlib import Path

from datasets.fingerprint import Hasher

from timbral.datasets import split_generators
from timbral.paths import project_root


PROJECT_ROOT = project_root()


@dataclasses.dataclass(frozen=True)
class PrepConfig:
    """Complete configuration for one raw audio preparation run (see scripts/raw_prep.py for field semantics)."""

    dataset_name: str
    dataset_dir: str
    split_json: str
    cache_dir: str
    sr: int
    mono: bool
    seg_sec: float
    hop_sec: float
    tol_sec: float
    label_type: str
    num_proc: int
    batch_size: int
    overwrite: bool
    split_json_hash: str
    config_hash: str


def _resolve_split_json(dataset_name: str, dataset_dir: str,
                        split_json: str | None) -> str:
    """Resolve the split file path, invoking the corresponding split generator when the default split is missing.

    Args:
        dataset_name: Dataset name.
        dataset_dir: Root directory of the dataset's source files.
        split_json: Explicit split file path; when ``None``, the project's
            default split is used.

    Returns:
        A split file path ready to be read directly.

    Raises:
        FileNotFoundError: The explicit split does not exist, or the
            default split does not exist and no corresponding generator
            was found.
        Exception: Any exception raised inside the split generator
            propagates upward unchanged.
    """
    if split_json is not None:
        explicit_path = os.fspath(split_json)
        if not Path(explicit_path).is_file():
            raise FileNotFoundError(f"Explicitly specified split_json does not exist: {explicit_path}")
        return explicit_path

    default_path = (PROJECT_ROOT / "assets" / "datasets" / "splits"
                    / dataset_name / "default.json")
    if default_path.is_file():
        return str(default_path)

    if dataset_name not in split_generators.GENERATORS:
        raise FileNotFoundError(
            f"Default split does not exist, and no corresponding split "
            f"generator was found: {dataset_name}")

    split_generators.generate(dataset_name, dataset_dir,
                              output_path=default_path)
    return str(default_path)


def resolve_config(dataset_name, dataset_dir, cache_dir=None, split_json=None,
                   sr=16000, mono=True, seg_sec=10.0, hop_sec=10.0, tol_sec=0.0,
                   label_type="weak", num_proc=4, batch_size=16,
                   overwrite=False) -> PrepConfig:
    """Fill in default paths and compute the split and config hashes.

    Args:
        dataset_name: Dataset name, used for default path derivation and
            adapter dispatch.
        dataset_dir: Root directory of the dataset's source files; must be
            passed explicitly.
        cache_dir: Output directory. When ``None``, defaults to
            ``{current working directory}/{dataset_name}/{config_hash}``.
        split_json: Split file path. When ``None``, defaults to
            ``{repo root}/assets/datasets/splits/{dataset_name}/default.json``;
            if that file does not exist, the registered split generator is
            invoked to generate it.
        sr: Target sample rate.
        mono: Whether to convert to mono.
        seg_sec: Audio segment length (seconds).
        hop_sec: Audio segment hop length (seconds).
        tol_sec: Minimum retained length for a trailing segment (seconds).
        label_type: Label type, either ``weak`` or ``strong``.
        num_proc: Number of data processing worker processes.
        batch_size: Batch size for Hugging Face ``map``.
        overwrite: Whether to force a rebuild when the output already
            exists.

    Returns:
        A fully populated, immutable config object.

    Raises:
        ValueError: ``dataset_dir`` is ``None`` or the label type is
            unsupported.
        FileNotFoundError: The explicit split does not exist, or the
            default split does not exist and no corresponding generator
            was found.
        Exception: Any exception raised inside the split generator
            propagates upward unchanged.
    """
    if dataset_dir is None:
        raise ValueError("dataset_dir must be passed explicitly")
    if label_type not in ("weak", "strong"):
        raise ValueError(f"label_type only supports weak/strong, got: {label_type}")

    dataset_dir = os.fspath(dataset_dir)
    split_json = _resolve_split_json(dataset_name, dataset_dir, split_json)
    split_json_hash = Hasher.hash(Path(split_json).read_bytes())
    # Premise for config_hash's coverage: annotation content is frozen
    # together with dataset_dir -- neither the annotation source files nor
    # the adapter logic participate in the hash, so changes to either
    # require a manual --overwrite to rebuild the cache.
    config_hash = Hasher.hash({
        "dataset_name": dataset_name,
        "split_json_hash": split_json_hash,
        "sr": sr,
        "mono": mono,
        "seg_sec": seg_sec,
        "hop_sec": hop_sec,
        "tol_sec": tol_sec,
        "label_type": label_type,
    })
    if cache_dir is None:
        cache_dir = str(Path.cwd() / dataset_name / config_hash)
    else:
        cache_dir = os.fspath(cache_dir)

    return PrepConfig(
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        split_json=split_json,
        cache_dir=cache_dir,
        sr=sr,
        mono=mono,
        seg_sec=seg_sec,
        hop_sec=hop_sec,
        tol_sec=tol_sec,
        label_type=label_type,
        num_proc=num_proc,
        batch_size=batch_size,
        overwrite=overwrite,
        split_json_hash=split_json_hash,
        config_hash=config_hash,
    )
