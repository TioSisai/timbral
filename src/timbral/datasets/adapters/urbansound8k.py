"""UrbanSound8K annotation adapter: metadata/UrbanSound8K.csv -> multiclass weak."""

import os

import pandas as pd

from .base import DatasetAnnotation


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load UrbanSound8K official annotations; audio_path looks like 'audio/fold{fold}/{slice_file_name}'."""
    df = pd.read_csv(os.path.join(dataset_dir, "metadata", "UrbanSound8K.csv"))
    df = df[df["slice_file_name"] != ".DS_Store"]  # Ignore stray rows, consistent with the split generation script
    weak_labels = {f"audio/fold{int(fold)}/{fn}": cls
                   for fn, fold, cls in zip(df["slice_file_name"], df["fold"],
                                            df["class"])}
    return DatasetAnnotation(label_kind="multiclass", annotation_kind="weak",
                             classes=sorted(df["class"].unique()),
                             weak_labels=weak_labels)
