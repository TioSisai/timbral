"""DESED annotation adapter: event tsv files for synthetic train/val plus real public eval -> multilabel strong.

All events have value=1.0 (confirmed present).
"""

import os

import pandas as pd

from .base import DatasetAnnotation

# {split: (tsv relative path, audio_path prefix)} -- the prefix is relative to
# the dataset root, posix-style; layout constants are defined here as the
# single source of truth, referenced directly by the split generator
SPLIT_SOURCES = {
    "train": (
        "dcase_synth/metadata/train/synthetic21_train/soundscapes.tsv",
        "dcase_synth/audio/train/synthetic21_train/soundscapes",
    ),
    "validation": (
        "dcase_synth/metadata/validation/synthetic21_validation/soundscapes.tsv",
        "dcase_synth/audio/validation/synthetic21_validation/soundscapes",
    ),
    "test": (
        "metadata/eval/public.tsv",
        "audio/eval/public",
    ),
}


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load DESED event-level annotations (onset/offset/event_label)."""
    strong_events = {}
    classes = set()
    for tsv_rel, audio_prefix in SPLIT_SOURCES.values():
        df = pd.read_csv(os.path.join(dataset_dir, tsv_rel), sep="\t")
        classes.update(df["event_label"].unique())
        for fn, onset, offset, label in zip(df["filename"], df["onset"],
                                            df["offset"], df["event_label"]):
            strong_events.setdefault(f"{audio_prefix}/{fn}", []).append(
                (label, float(onset), float(offset), 1.0))
    return DatasetAnnotation(label_kind="multilabel", annotation_kind="strong",
                             classes=sorted(classes), strong_events=strong_events)
