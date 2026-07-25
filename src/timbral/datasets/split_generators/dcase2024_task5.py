"""DCASE-2024-Task-5 default split generator.

Strategy (see SPEC section 14):
  - train = all wav files under Development_Set/Training_Set/{BV,HT,JD,MT,WMW}/
  - validation = all wav files under
    Development_Set/Validation_Set/{HB,ME,PB,PB24,PW,RD}/
    (the two duplicate BUK5 copies under PB/PB24 are kept)
  - no official test set -> test = copy of validation (fill_missing_splits)
  - excludes __MACOSX/ and all .DS_Store
  - audio_path = posix path relative to the dataset root; start=0, end=inf
"""

import os
import tempfile
from collections import Counter

from ..adapters.dcase2024_task5 import PARTITION_SUBSETS
from . import base as u


def collect(dataset_dir, subdir, directories):
    """Collect the relative paths of existing WAV files under the given directory.

    Args:
        dataset_dir: DCASE-2024-Task-5 dataset root directory.
        subdir: Training-set or validation-set directory name under Development_Set.
        directories: Recording-site directory names to scan.

    Returns:
        list[str]: Ordered relative paths, with ``__MACOSX`` and ``.DS_Store`` excluded.
    """
    audio_paths = []
    for directory in directories:
        base_dir = os.path.join(dataset_dir, "Development_Set", subdir, directory)
        for current_dir, _, filenames in os.walk(base_dir):
            if "__MACOSX" in current_dir.split(os.sep):
                continue
            for filename in filenames:
                if filename == ".DS_Store":
                    continue
                if filename.lower().endswith(".wav"):
                    full_path = os.path.join(current_dir, filename)
                    # os.walk lists broken symlinks too; exists returns False for those
                    if not os.path.exists(full_path):
                        continue
                    audio_paths.append(os.path.relpath(full_path, dataset_dir))
    return sorted(audio_paths)


def breakdown(audio_paths):
    """Count the number of files per recording site among the relative paths.

    Args:
        audio_paths: Audio paths relative to the dataset root directory.

    Returns:
        dict[str, int]: File counts, sorted by recording-site name.
    """
    counts = Counter(path.split("/")[2] for path in audio_paths)
    return dict(sorted(counts.items()))


def generate(dataset_dir):
    """Build the DCASE-2024-Task-5 default split.

    Args:
        dataset_dir: DCASE-2024-Task-5 dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    # The subset manifest is defined once in adapters.dcase2024_task5
    subsets = dict(PARTITION_SUBSETS)
    train_rel = collect(dataset_dir, "Training_Set", subsets["Training_Set"])
    val_rel = collect(dataset_dir, "Validation_Set", subsets["Validation_Set"])

    # Write intermediate lists to disk (to avoid dumping to stdout); kept under
    # $TMPDIR for later inspection
    work_dir = tempfile.mkdtemp(prefix="timbral_gen_work_")
    with open(os.path.join(work_dir, "train_wavs.txt"), "w",
              encoding="utf-8") as file:
        file.write("\n".join(train_rel) + "\n")
    with open(os.path.join(work_dir, "val_wavs.txt"), "w",
              encoding="utf-8") as file:
        file.write("\n".join(val_rel) + "\n")
    print("work_dir:", work_dir)
    print("train_breakdown:", breakdown(train_rel))
    print("val_breakdown:", breakdown(val_rel))

    splits = {
        "train": [u.make_entry(path) for path in train_rel],
        "validation": [u.make_entry(path) for path in val_rel],
    }
    splits, note = u.fill_missing_splits(splits)
    print("copy note:", repr(note))
    return splits
