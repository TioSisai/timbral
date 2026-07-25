"""FSDnoisy18k default split generator.

Per SPEC section 7:
  - train = all of FSDnoisy18k.meta/train.csv, audio_path = "FSDnoisy18k.audio_train/{fname}"
    (the 3 leaked files that duplicate test are kept, not deduplicated)
  - test  = all of FSDnoisy18k.meta/test.csv, audio_path = "FSDnoisy18k.audio_test/{fname}"
  - no official validation set -> validation = copy of test (copy rule)
  - only audio that actually exists on disk is included
"""

import os

import pandas as pd

from . import base as u


def build_split(dataset_dir, metadata_dir, csv_name, audio_subdir):
    """Build split entries from the annotation CSV, keeping only audio that
    actually exists on disk.

    Args:
        dataset_dir: FSDnoisy18k dataset root directory.
        metadata_dir: Dataset annotation directory.
        csv_name: Annotation CSV filename.
        audio_subdir: Audio subdirectory relative to the dataset root.

    Returns:
        tuple: Split entries, dropped paths, and total CSV row count.
    """
    df = pd.read_csv(os.path.join(metadata_dir, csv_name))
    entries = []
    dropped = []
    for fname in df["fname"].tolist():
        rel = "{}/{}".format(audio_subdir, fname)
        abs_path = os.path.join(dataset_dir, rel)
        if os.path.exists(abs_path):
            entries.append(u.make_entry(rel, start=0.0, end=u.INF))
        else:
            dropped.append(rel)
    return entries, dropped, len(df)


def generate(dataset_dir):
    """Build the FSDnoisy18k default split.

    Args:
        dataset_dir: FSDnoisy18k dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    metadata_dir = os.path.join(dataset_dir, "FSDnoisy18k.meta")

    train_entries, train_dropped, num_train_csv = build_split(
        dataset_dir,
        metadata_dir,
        "train.csv",
        "FSDnoisy18k.audio_train",
    )
    test_entries, test_dropped, num_test_csv = build_split(
        dataset_dir,
        metadata_dir,
        "test.csv",
        "FSDnoisy18k.audio_test",
    )

    print("train.csv rows={} -> present entries={} dropped={}".format(
        num_train_csv, len(train_entries), len(train_dropped)))
    print("test.csv  rows={} -> present entries={} dropped={}".format(
        num_test_csv, len(test_entries), len(test_dropped)))
    if train_dropped:
        print("train dropped samples:", train_dropped[:5])
    if test_dropped:
        print("test  dropped samples:", test_dropped[:5])

    splits = {"train": train_entries, "test": test_entries}
    splits, note = u.fill_missing_splits(splits)
    print("copy note:", note)
    return splits
