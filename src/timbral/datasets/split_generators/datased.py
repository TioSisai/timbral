"""DataSED default split generator (strong multi-label, greedy duration split)."""

import os

import soundfile as sf

from ..adapters.datased import WAV_DIR, read_merged_ground_truth
from . import base as u


def generate(dataset_dir):
    """Build the DataSED default split.

    Args:
        dataset_dir: DataSED dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    wav_dir = os.path.join(dataset_dir, WAV_DIR)

    # 1. Annotation merging logic is defined once in adapters.datased
    #    (Polyphonic + Monophonic supplemental rows)
    merged = read_merged_ground_truth(dataset_dir)
    print(
        "merged rows:",
        len(merged),
        "unique sound_name:",
        merged["sound_name"].nunique(),
    )

    # 2. Only include audio that actually exists on disk
    sound_names = sorted(os.listdir(wav_dir))  # 717 files: S-0001.wav..S-0717.wav
    print("disk wav:", len(sound_names))

    # Each file's class set = the deduplicated set of all class_name values for
    # that sound_name in the merged annotations
    labelsets = {sound_name: set() for sound_name in sound_names}
    for sound_name, class_name in zip(merged["sound_name"], merged["class_name"]):
        if sound_name in labelsets:
            labelsets[sound_name].add(class_name)
    n_no_label = sum(1 for sound_name in sound_names if not labelsets[sound_name])
    print("files without any annotation:", n_no_label)
    all_classes = set()
    for labels in labelsets.values():
        all_classes |= labels
    print("num distinct classes:", len(all_classes))

    # 3. File duration = measured via soundfile.info
    weights = {}
    for sound_name in sound_names:
        audio_info = sf.info(os.path.join(wav_dir, sound_name))
        weights[sound_name] = float(audio_info.frames) / float(audio_info.samplerate)
    total_duration = sum(weights.values())
    print("total duration (s):", round(total_duration, 2))

    # 4. Greedy duration split
    split_ids = u.greedy_multilabel_duration_split(
        sound_names, labelsets, weights, ratios=(0.8, 0.1, 0.1)
    )
    for split in u.SPLIT_NAMES:
        duration = sum(weights[filename] for filename in split_ids[split])
        print(
            f"{split}: n={len(split_ids[split])} dur={duration:.1f}s "
            f"({duration / total_duration * 100:.1f}%)"
        )

    # 5. Map to audio_path = "SED_wav/{sound_name}", start=0, end=inf
    splits = {
        split: [u.make_entry(f"{WAV_DIR}/{sound_name}")
                for sound_name in split_ids[split]]
        for split in u.SPLIT_NAMES
    }

    # All three splits are already present; copy is not triggered
    splits, note = u.fill_missing_splits(splits)
    print("copy note:", repr(note))
    return splits
