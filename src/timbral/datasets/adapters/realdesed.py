"""RealDESED annotation adapter: adjudicated annotations.csv per split -> multilabel strong.

Only annotations.csv (the adjudicated, merged release version) is read;
annotations_raw.csv (unadjudicated per-annotator raw events) and metadata.csv
(recording intent, which is mutually inconsistent with the adjudicated ground
truth) are not used. The full class set is a dynamic union of the class
columns across the three splits; all events have value=1.0 (confirmed present).
"""

import os

import pandas as pd

from .base import DatasetAnnotation

# Official directory split; layout constants defined here as the single source
# of truth, referenced directly by the split generator
_SPLITS = ("train", "validation", "test")


def annotations_csv_path(dataset_dir: str, split: str) -> str:
    """The absolute path of the split's adjudicated annotation CSV (shared with the split generator).

    Args:
        dataset_dir: RealDESED dataset root directory.
        split: Official split name; see ``_SPLITS`` for possible values.

    Returns:
        The absolute path of the annotations.csv for that split.
    """
    return os.path.join(dataset_dir, split, "annotations.csv")


def audio_relpath(split: str, filename: str) -> str:
    """The dataset-relative path of an audio file within a split (shared with the split generator).

    Args:
        split: Official split name; see ``_SPLITS`` for possible values.
        filename: The audio filename in annotations.csv.

    Returns:
        Dataset-relative path, of the form ``"{split}/audio/{filename}"``.
    """
    return f"{split}/audio/{filename}"


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load RealDESED event-level annotations (onset/offset/class)."""
    strong_events = {}
    classes = set()
    for split in _SPLITS:
        df = pd.read_csv(annotations_csv_path(dataset_dir, split))
        classes.update(df["class"].unique())
        for fn, onset, offset, label in zip(df["filename"], df["onset"],
                                            df["offset"], df["class"]):
            strong_events.setdefault(audio_relpath(split, fn), []).append(
                (label, float(onset), float(offset), 1.0))
    return DatasetAnnotation(label_kind="multilabel", annotation_kind="strong",
                             classes=sorted(classes), strong_events=strong_events)
