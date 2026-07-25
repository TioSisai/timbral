"""DataSED annotation adapter: Polyphonic csv (plus Monophonic patch rows) -> multilabel strong.

All events have value=1.0 (confirmed present). The dataset layout constants and
annotation-merging logic are defined here as the single source of truth,
referenced directly by the split generator.
"""

import os

import pandas as pd

from .base import DatasetAnnotation

WAV_DIR = "SED_wav"
_GROUND_TRUTH_DIR = "SED_ground_truth"
# Audio missing Polyphonic annotations that must be patched in from Monophonic
# (see dataset survey conclusions)
_MONO_PATCH_FILES = {f"S-{i:04d}.wav" for i in range(704, 718)}


def read_merged_ground_truth(dataset_dir: str) -> pd.DataFrame:
    """Load and merge the Polyphonic annotations with the Monophonic patch rows (shared with the split generator).

    Args:
        dataset_dir: DataSED dataset root directory.

    Returns:
        pd.DataFrame: The merged result with the full Polyphonic table first,
        followed by the Monophonic patch rows; columns are
        sound_name/class_name/start_time/end_time.
    """
    gt = os.path.join(dataset_dir, _GROUND_TRUTH_DIR)
    poly = pd.read_csv(os.path.join(gt, "Polyphonic_sound_detection.csv"))
    mono = pd.read_csv(os.path.join(gt, "Monophonic_sound_detection.csv"))
    return pd.concat(
        [poly, mono[mono["sound_name"].isin(_MONO_PATCH_FILES)]],
        ignore_index=True)


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load DataSED event-level annotations; audio_path looks like 'SED_wav/{sound_name}'."""
    merged = read_merged_ground_truth(dataset_dir)
    strong_events = {}
    for sn, cn, st, et in zip(merged["sound_name"], merged["class_name"],
                              merged["start_time"], merged["end_time"]):
        strong_events.setdefault(f"{WAV_DIR}/{sn}", []).append(
            (cn, float(st), float(et), 1.0))
    return DatasetAnnotation(label_kind="multilabel", annotation_kind="strong",
                             classes=sorted(merged["class_name"].unique()),
                             strong_events=strong_events)
