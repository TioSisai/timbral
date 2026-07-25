"""RealDESED default split generator.

The official release already partitions train/validation/test by directory, so
no split algorithm is applied here:
  - Each split's membership is determined by the unique filenames in
    {split}/annotations.csv (the released, post-adjudication merged version);
    audio with no annotation that passed review is naturally excluded (avoiding
    false negatives that would otherwise enter the split as "no event in the
    whole clip");
  - Only audio that actually exists on disk is included. All three splits are
    already present; copy is not triggered.
"""

import csv
import os

from ..adapters.realdesed import annotations_csv_path, audio_relpath
from . import base as u


def unique_filenames(csv_path):
    """Read the filename column of the CSV and return unique filenames in order
    of first appearance.

    Args:
        csv_path: Annotation CSV path to read.

    Returns:
        list[str]: Deduplicated list of filenames.
    """
    seen = set()
    ordered = []
    with open(csv_path, encoding="utf-8", newline="") as file:
        reader = csv.reader(file)
        header = next(reader)
        assert header[0] == "filename", "first column is not filename: {}".format(header)
        for row in reader:
            if not row:
                continue
            filename = row[0]
            if filename not in seen:
                seen.add(filename)
                ordered.append(filename)
    return ordered


def generate(dataset_dir):
    """Build the RealDESED default split.

    Args:
        dataset_dir: RealDESED dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))

    splits = {}
    stats = {}
    for split in u.SPLIT_NAMES:
        filenames = unique_filenames(annotations_csv_path(dataset_dir, split))
        entries = []
        missing = 0
        for filename in filenames:
            relative_path = audio_relpath(split, filename)
            if os.path.isfile(os.path.join(dataset_dir, relative_path)):
                entries.append(u.make_entry(relative_path, start=0.0, end=u.INF))
            else:
                missing += 1
        splits[split] = entries
        stats[split] = {
            "csv_unique": len(filenames),
            "present": len(entries),
            "missing_on_disk": missing,
        }

    print("STATS:", stats)
    return splits
