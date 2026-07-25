"""SONYC-UST default split generator.

Per SPEC section 4:
  - reads annotations.csv (multiple rows per audio = multiple annotators);
  - takes the unique audio_filename per split column value: train / validate / test;
  - audio_path = "audio/{audio_filename}"; only audio that actually exists on disk
    is included;
  - all three splits are officially present; copy is not triggered; start=0,
    end=inf; everything is kept (no content-duplicate removal).
"""

import os

import pandas as pd

from . import base as u

# split column value -> target split name
SPLIT_MAP = {"train": "train", "validate": "validation", "test": "test"}


def generate(dataset_dir):
    """Build the SONYC-UST default split.

    Args:
        dataset_dir: SONYC-UST dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    audio_dir = os.path.join(dataset_dir, "audio")

    df = pd.read_csv(os.path.join(dataset_dir, "annotations.csv"))
    disk = set(os.listdir(audio_dir))  # set of audio filenames that actually exist on disk

    splits = {"train": [], "validation": [], "test": []}
    stats = {}
    for raw_split, target in SPLIT_MAP.items():
        # unique audio_filename for this split (deterministic ordering)
        names = sorted(df.loc[df["split"] == raw_split, "audio_filename"].unique())
        kept, missing = [], 0
        for name in names:
            if name in disk:  # only include audio that actually exists on disk
                kept.append(u.make_entry("audio/" + name, start=0.0, end=u.INF))
            else:
                missing += 1
        splits[target] = kept
        stats[target] = {
            "unique_annotated": len(names),
            "kept": len(kept),
            "missing_on_disk": missing,
        }

    print("STATS:", stats)
    return splits
