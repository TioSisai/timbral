"""FSD50K default split generator.

FSD50K uses the official predefined split (the split column in dev.csv, plus
eval.csv in its entirety as test), so no stratified random split is needed. Only
audio files that actually exist on disk are included, and overlong files are
excluded per the official clip-duration cap (30s) (TODO item 8: 13 files in
dev/eval were measured at >30s, the longest being 550s, and must be excluded
manually; the threshold is strictly >30.0s, so eval/30303.wav, at exactly
30.01s, is also excluded).
"""

import os

import pandas as pd
import soundfile as sf

from ..adapters.fsd50k import DEV_AUDIO_DIR, EVAL_AUDIO_DIR, GROUND_TRUTH_DIR
from . import base as u

# FSD50K's official clip-duration cap; files measured at >30.0s are annotation
# anomalies, excluded per TODO item 8
MAX_DURATION_SEC = 30.0


def _is_over_length(dataset_dir, audio_rel):
    """Determine whether the audio's measured duration exceeds the official cap.

    Args:
        dataset_dir: FSD50K dataset root directory.
        audio_rel: Audio path relative to the dataset root directory.

    Returns:
        bool: Whether the audio duration strictly exceeds the official cap.
    """
    info = sf.info(os.path.join(dataset_dir, audio_rel))
    return info.frames / info.samplerate > MAX_DURATION_SEC


def generate(dataset_dir):
    """Build the FSD50K default split.

    Args:
        dataset_dir: FSD50K dataset root directory.

    Returns:
        dict: ``{"train": [entry, ...], "validation": [...], "test": [...]}``.
    """
    dataset_dir = os.path.abspath(os.fspath(dataset_dir))
    ground_truth_dir = os.path.join(dataset_dir, GROUND_TRUTH_DIR)

    # Set of wav basenames that actually exist on disk (extension stripped)
    dev_present = {
        filename[:-4]
        for filename in os.listdir(os.path.join(dataset_dir, DEV_AUDIO_DIR))
        if filename.endswith(".wav")
    }
    eval_present = {
        filename[:-4]
        for filename in os.listdir(os.path.join(dataset_dir, EVAL_AUDIO_DIR))
        if filename.endswith(".wav")
    }
    print("disk dev wav:", len(dev_present), "eval wav:", len(eval_present))

    dev = pd.read_csv(os.path.join(ground_truth_dir, "dev.csv"))
    ev = pd.read_csv(os.path.join(ground_truth_dir, "eval.csv"))
    # fname may be read in as int; normalize to str
    dev["fname"] = dev["fname"].astype(str)
    ev["fname"] = ev["fname"].astype(str)

    splits = {"train": [], "validation": [], "test": []}
    missing = {"train": 0, "validation": 0, "test": 0}
    over_length = {"train": [], "validation": [], "test": []}

    # train / validation come from the split column in dev.csv
    split_map = {"train": "train", "val": "validation"}
    for filename, split_value in zip(dev["fname"], dev["split"]):
        target_split = split_map.get(split_value)
        if target_split is None:
            raise ValueError("unknown split value: {}".format(split_value))
        if filename not in dev_present:
            missing[target_split] += 1
            continue
        audio_rel = "{}/{}.wav".format(DEV_AUDIO_DIR, filename)
        if _is_over_length(dataset_dir, audio_rel):  # overlong files excluded per official 30s cap
            over_length[target_split].append(audio_rel)
            continue
        splits[target_split].append(u.make_entry(audio_rel))

    # test comes from the entirety of eval.csv
    for filename in ev["fname"]:
        if filename not in eval_present:
            missing["test"] += 1
            continue
        audio_rel = "{}/{}.wav".format(EVAL_AUDIO_DIR, filename)
        if _is_over_length(dataset_dir, audio_rel):  # overlong files excluded per official 30s cap
            over_length["test"].append(audio_rel)
            continue
        splits["test"].append(u.make_entry(audio_rel))

    print("annotated but missing on disk (should be 0):", missing)
    num_over_length = sum(len(paths) for paths in over_length.values())
    print(
        "overlong (>{}s) excluded count:".format(MAX_DURATION_SEC),
        {split_name: len(paths) for split_name, paths in over_length.items()},
        "total",
        num_over_length,
    )
    for split_name in ("train", "validation", "test"):
        for audio_rel in over_length[split_name]:
            print("  EXCLUDED_OVERLEN", split_name, audio_rel)

    # All three splits are already present; copy is not triggered; still run
    # through it to normalize keys
    splits, note = u.fill_missing_splits(splits)
    print("copy note:", repr(note))
    return splits
