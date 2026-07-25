"""UrbanSound8K default split generator.

Per SPEC section 2: official fold-based split, fold 1-8 -> train, 9 -> validation,
10 -> test. audio_path = "audio/fold{fold}/{slice_file_name}"; .DS_Store is
ignored. Only audio that actually exists on disk is included. All three splits
are already present; copy is not triggered.
"""

import os

import pandas as pd

from . import base as u

# fold -> split mapping
FOLD_TO_SPLIT = {}
for f in range(1, 9):
    FOLD_TO_SPLIT[f] = "train"
FOLD_TO_SPLIT[9] = "validation"
FOLD_TO_SPLIT[10] = "test"


def generate(dataset_dir):
    """Build the UrbanSound8K default split.

    Args:
        dataset_dir: UrbanSound8K dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    csv_path = os.path.join(dataset_dir, "metadata", "UrbanSound8K.csv")

    df = pd.read_csv(csv_path)
    splits = {split_name: [] for split_name in u.SPLIT_NAMES}
    missing = 0
    seen_ds_store = 0
    for row in df.itertuples(index=False):
        name = row.slice_file_name
        if name == ".DS_Store":
            seen_ds_store += 1
            continue
        fold = int(row.fold)
        rel = "audio/fold{}/{}".format(fold, name)
        abs_path = os.path.join(dataset_dir, rel)
        if not os.path.exists(abs_path):
            missing += 1
            continue
        splits[FOLD_TO_SPLIT[fold]].append(u.make_entry(rel, start=0.0, end=u.INF))

    print("annotation rows (excluding .DS_Store):", len(df) - seen_ds_store)
    print("dropped as missing on disk:", missing)
    return splits
