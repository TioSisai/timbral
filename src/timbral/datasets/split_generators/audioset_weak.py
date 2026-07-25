"""AudioSetWeak default split generator.

Strategy (per SPEC section 6 and global conventions):
  - based directly on the WAV files actually present on disk; no need to
    cross-reference a CSV;
  - train includes balanced_train_segments and all unbalanced partial directories;
  - test includes eval_segments;
  - no official validation set; use fill_missing_splits to copy test into validation;
  - audio_path is a POSIX path relative to the dataset root directory.
"""

import tempfile
from pathlib import Path

from . import base as u


def scan_wavs(dataset_dir, subdirectory):
    """Enumerate the WAV files in the given subdirectory.

    Args:
        dataset_dir: AudioSetWeak dataset root directory.
        subdirectory: Directory to scan, relative to the dataset root.

    Returns:
        list[str]: Sorted POSIX paths relative to the dataset root.
    """
    relative_paths = []
    audio_dir = dataset_dir / subdirectory
    for entry in audio_dir.iterdir():
        if entry.name.endswith(".wav") and entry.is_file():
            relative_paths.append(f"{subdirectory}/{entry.name}")
    return sorted(relative_paths)


def generate(dataset_dir):
    """Build the AudioSetWeak default split.

    Args:
        dataset_dir: AudioSetWeak dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = Path(dataset_dir).resolve()

    top_level_names = sorted(path.name for path in dataset_dir.iterdir())
    train_dirs = ["balanced_train_segments"] + sorted(
        name
        for name in top_level_names
        if name.startswith("unbalanced_train_segments_part")
        and name.endswith("_partial")
    )
    test_dirs = ["eval_segments"]

    print("train_dirs count:", len(train_dirs))
    print("test_dirs:", test_dirs)

    train_paths = []
    for subdirectory in train_dirs:
        part_paths = scan_wavs(dataset_dir, subdirectory)
        train_paths.extend(part_paths)
        print("  {}: {} wav".format(subdirectory, len(part_paths)))

    test_paths = []
    for subdirectory in test_dirs:
        part_paths = scan_wavs(dataset_dir, subdirectory)
        test_paths.extend(part_paths)
        print("  {}: {} wav".format(subdirectory, len(part_paths)))

    # Write the scanned raw path lists to disk (to avoid dumping them to stdout);
    # kept under $TMPDIR for later inspection
    work_dir = Path(tempfile.mkdtemp(prefix="timbral_gen_work_"))
    (work_dir / "train_paths.txt").write_text(
        "\n".join(train_paths), encoding="utf-8"
    )
    (work_dir / "test_paths.txt").write_text(
        "\n".join(test_paths), encoding="utf-8"
    )
    print("work_dir:", work_dir)

    print("train wav total:", len(train_paths))
    print("test  wav total:", len(test_paths))

    splits = {
        "train": [u.make_entry(path) for path in train_paths],
        "test": [u.make_entry(path) for path in test_paths],
    }
    splits, note = u.fill_missing_splits(splits)
    print("copy note:", note)
    return splits
