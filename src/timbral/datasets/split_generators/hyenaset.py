"""HyenaSET default split generator.

Built from fairseq manifests: the train/valid manifests each become one split;
no official test set -> test = copy of validation (copy rule); only files that
actually exist on disk are included.
"""

import os

from . import base as u

PREFIX = "wav/24000Hz/"


def read_manifest_rel_paths(dataset_dir, manifest_name):
    """Read the relative paths recorded in a fairseq manifest.

    Skips the first line if it contains no tab, and takes the first column of
    the remaining lines as the relative path.

    Args:
        dataset_dir: HyenaSET dataset root directory.
        manifest_name: Manifest filename.

    Returns:
        list[str]: Relative paths recorded in the manifest.
    """
    manifest_path = os.path.join(dataset_dir, "manifests", manifest_name)
    rels = []
    with open(manifest_path, encoding="utf-8") as manifest_file:
        for line in manifest_file:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue  # skip the first line, the raw root path (no TAB)
            rel = line.split("\t")[0]
            rels.append(rel)
    return rels


def build_entries(dataset_dir, rel_paths):
    """Map relative paths to split entries, including only files that actually
    exist on disk.

    Args:
        dataset_dir: HyenaSET dataset root directory.
        rel_paths: Relative paths recorded in the manifest.

    Returns:
        tuple: Split entries and the count of files missing from disk.
    """
    entries = []
    missing = 0
    for rel_path in rel_paths:
        audio_path = PREFIX + rel_path
        disk_path = os.path.join(dataset_dir, audio_path)
        if os.path.exists(disk_path):
            entries.append(u.make_entry(audio_path, start=0.0, end=u.INF))
        else:
            missing += 1
    return entries, missing


def generate(dataset_dir):
    """Build the HyenaSET default split.

    Args:
        dataset_dir: HyenaSET dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))

    train_rels = read_manifest_rel_paths(dataset_dir, "train_0_specific_indiv.tsv")
    val_rels = read_manifest_rel_paths(dataset_dir, "valid_0_specific_indiv.tsv")
    print("manifest train rels:", len(train_rels), "val rels:", len(val_rels))

    train_entries, train_missing = build_entries(dataset_dir, train_rels)
    val_entries, val_missing = build_entries(dataset_dir, val_rels)
    print("train on disk:", len(train_entries), "missing:", train_missing)
    print("val   on disk:", len(val_entries), "missing:", val_missing)

    splits = {"train": train_entries, "validation": val_entries}
    splits, note = u.fill_missing_splits(splits)
    print("copy note:", note)
    return splits
