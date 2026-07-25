"""BSD35K and BSD10K default split generators.

Both datasets share the same structure: metadata CSV (columns sound_id, class) +
audio/{sound_id}.wav; stratified random split by class 80/10/10; only audio files
that actually exist on disk are included.
"""

from pathlib import Path

import pandas as pd

from ..adapters.bsd import (BSD10K_METADATA_CSV, BSD35K_METADATA_CSV,
                            audio_relpath)
from . import base as u


def _generate(dataset_dir, metadata_filename):
    """Build a stratified random split from the metadata CSV (shared by the BSD
    series).

    Args:
        dataset_dir: Dataset root directory.
        metadata_filename: Annotation CSV filename under the metadata directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = Path(dataset_dir).resolve()
    metadata_path = dataset_dir / "metadata" / metadata_filename

    metadata = pd.read_csv(metadata_path)
    print("csv row count:", len(metadata))
    print("unique sound_id:", metadata["sound_id"].nunique())
    print("unique class:", metadata["class"].nunique())

    item_ids = []
    labels = []
    annotated_missing = 0
    sound_ids = metadata["sound_id"].tolist()
    class_names = metadata["class"].astype(str).tolist()
    for sound_id, class_name in zip(sound_ids, class_names):
        relative_path = audio_relpath(sound_id)
        if (dataset_dir / relative_path).is_file():
            item_ids.append(relative_path)
            labels.append(class_name)
        else:
            annotated_missing += 1

    print("samples included (present on disk):", len(item_ids))
    print("annotated but missing on disk:", annotated_missing)

    split_ids = u.stratified_split_single_label(
        item_ids, labels, ratios=(0.8, 0.1, 0.1)
    )
    print("split counts:", {key: len(value) for key, value in split_ids.items()})

    splits = {
        split_name: [
            u.make_entry(audio_path, start=0.0, end=u.INF)
            for audio_path in split_ids[split_name]
        ]
        for split_name in u.SPLIT_NAMES
    }

    splits, copy_note = u.fill_missing_splits(splits)
    print("copy_note:", repr(copy_note))
    return splits


def generate_bsd35k(dataset_dir):
    """Build the BSD35K default split.

    Args:
        dataset_dir: BSD35K dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    return _generate(dataset_dir, BSD35K_METADATA_CSV)


def generate_bsd10k(dataset_dir):
    """Build the BSD10K default split.

    Args:
        dataset_dir: BSD10K dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    return _generate(dataset_dir, BSD10K_METADATA_CSV)
