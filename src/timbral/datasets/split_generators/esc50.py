"""ESC-50 default split generator.

Specification (SPEC.md section 1):
  - annotations: meta/esc50.csv (columns filename,fold,target,category,...)
  - split: fold in {1,2,3}->train, fold==4->validation, fold==5->test
  - audio_path = "audio/" + filename, start=0, end=inf
  - all three splits are already present; copy is not triggered. Expected train
    1200 / val 400 / test 400 (all 2000 present)
"""

import os

import pandas as pd

from . import base as u

FOLD_TO_SPLIT = {1: "train", 2: "train", 3: "train", 4: "validation", 5: "test"}


def generate(dataset_dir):
    """Build the ESC-50 default split.

    Args:
        dataset_dir: ESC-50 dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    metadata_path = os.path.join(dataset_dir, "meta", "esc50.csv")

    metadata = pd.read_csv(metadata_path)
    splits = {split: [] for split in u.SPLIT_NAMES}
    dropped_missing = 0
    for filename, fold in zip(metadata["filename"], metadata["fold"]):
        split = FOLD_TO_SPLIT.get(int(fold))
        if split is None:
            raise ValueError(f"unknown fold: {fold}")
        relative_path = "audio/" + str(filename)
        audio_path = os.path.join(dataset_dir, relative_path)
        # Only include audio that actually exists on disk
        if not os.path.exists(audio_path):
            dropped_missing += 1
            continue
        splits[split].append(u.make_entry(relative_path, start=0.0, end=u.INF))

    # All three splits are already present; copy is not triggered (still calling
    # it for normalization and the note, which should be empty here)
    splits, copy_note = u.fill_missing_splits(splits)
    print("COPY_NOTE", repr(copy_note))
    print("DROPPED_MISSING", dropped_missing)
    return splits
