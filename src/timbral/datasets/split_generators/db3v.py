"""DB3V default split generator.

Per SPEC section 9:
- Structure: data_wav_8s_2/{region}/{species}/{recid}_{seg}.wav
  (region∈{1,2,3}), no separate annotation file.
- Merge the three regions: label = the species segment (2nd path component, 10
  classes); item = each wav file.
- Split: stratified_split_single_label(audio_paths, species_labels, 80/10/10).
- audio_path = actual path relative to the root, data_wav_8s_2/{region}/{species}/{file}
  (spaces preserved).
- Everything is kept (no grouping by recid, no near-duplicate removal). All three
  splits are present (no copy).
"""

import os
from collections import Counter

from ..adapters.db3v import AUDIO_DIR
from . import base as u


def generate(dataset_dir):
    """Build the DB3V default split.

    Args:
        dataset_dir: DB3V dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    data_dir = os.path.join(dataset_dir, AUDIO_DIR)

    # 1. Enumerate wav files that actually exist on disk, building
    #    (relative path, species label) pairs
    audio_paths = []
    labels = []
    for region in sorted(os.listdir(data_dir)):
        region_path = os.path.join(data_dir, region)
        if not os.path.isdir(region_path):
            continue
        for species in sorted(os.listdir(region_path)):
            species_path = os.path.join(region_path, species)
            if not os.path.isdir(species_path):
                continue
            for filename in sorted(os.listdir(species_path)):
                file_path = os.path.join(species_path, filename)
                if not os.path.isfile(file_path) or not filename.lower().endswith(".wav"):
                    continue
                relative_path = os.path.join(
                    AUDIO_DIR, region, species, filename
                )
                audio_paths.append(relative_path)
                labels.append(species)  # merging the three regions; label is species only

    n_total = len(audio_paths)
    n_classes = len(set(labels))
    print("total samples", n_total, "num classes", n_classes)
    print("class distribution:")
    for label, count in sorted(Counter(labels).items()):
        print("  ", repr(label), count)

    # 2. Stratified random split 80/10/10
    split_ids = u.stratified_split_single_label(
        audio_paths, labels, ratios=(0.8, 0.1, 0.1)
    )

    # 3. Map to entry
    splits = {
        split: [u.make_entry(path, start=0.0, end=u.INF) for path in split_ids[split]]
        for split in u.SPLIT_NAMES
    }

    # All three splits are already present; no copy needed
    splits, copy_note = u.fill_missing_splits(splits)
    print("copy_note:", repr(copy_note))
    return splits
