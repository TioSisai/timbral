"""DESED default split generator.

Per SPEC section 15:
  - train      = unique filenames from dcase_synth/.../train soundscapes.tsv
  - validation = unique filenames from dcase_synth/.../validation soundscapes.tsv
  - test       = unique filenames from metadata/eval/public.tsv
Only audio that actually exists on disk is included. All three splits are already
present; copy is not triggered.
"""

import os

from ..adapters.desed import SPLIT_SOURCES
from . import base as u


def unique_filenames(tsv_path):
    """Read the first column of the TSV and return unique filenames in order of
    first appearance.

    Args:
        tsv_path: Metadata TSV path to read.

    Returns:
        list[str]: Deduplicated list of filenames.
    """
    seen = set()
    ordered = []
    with open(tsv_path, encoding="utf-8") as file:
        header = file.readline().rstrip("\n").split("\t")
        assert header[0] == "filename", "first column is not filename: {}".format(header)
        for line in file:
            line = line.rstrip("\n")
            if not line:
                continue
            filename = line.split("\t", 1)[0]
            if filename not in seen:
                seen.add(filename)
                ordered.append(filename)
    return ordered


def generate(dataset_dir):
    """Build the DESED default split.

    Args:
        dataset_dir: DESED dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))

    splits = {}
    stats = {}
    for split, (tsv_relative_path, audio_prefix) in SPLIT_SOURCES.items():
        filenames = unique_filenames(os.path.join(dataset_dir, tsv_relative_path))
        entries = []
        missing = 0
        for filename in filenames:
            relative_path = "{}/{}".format(audio_prefix, filename)
            if os.path.isfile(os.path.join(dataset_dir, relative_path)):
                entries.append(u.make_entry(relative_path, start=0.0, end=u.INF))
            else:
                missing += 1
        splits[split] = entries
        stats[split] = {
            "tsv_unique": len(filenames),
            "present": len(entries),
            "missing_on_disk": missing,
        }

    print("STATS:", stats)
    return splits
