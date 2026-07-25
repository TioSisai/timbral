"""ESC-50 annotation adapter: meta/esc50.csv -> multiclass weak."""

import os

import pandas as pd

from .base import DatasetAnnotation


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load ESC-50 official annotations; audio_path looks like 'audio/{filename}'."""
    df = pd.read_csv(os.path.join(dataset_dir, "meta", "esc50.csv"))
    weak_labels = {f"audio/{fn}": cat
                   for fn, cat in zip(df["filename"], df["category"])}
    return DatasetAnnotation(label_kind="multiclass", annotation_kind="weak",
                             classes=sorted(df["category"].unique()),
                             weak_labels=weak_labels)
