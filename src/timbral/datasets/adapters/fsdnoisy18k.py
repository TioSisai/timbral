"""FSDnoisy18k annotation adapter: FSDnoisy18k.meta/train|test.csv -> multiclass weak."""

import os

import pandas as pd

from .base import DatasetAnnotation


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load FSDnoisy18k official annotations (one csv each for train/test, single-label with 20 classes)."""
    meta = os.path.join(dataset_dir, "FSDnoisy18k.meta")
    weak_labels = {}
    classes = set()
    for audio_dir, csv_name in (("FSDnoisy18k.audio_train", "train.csv"),
                                ("FSDnoisy18k.audio_test", "test.csv")):
        df = pd.read_csv(os.path.join(meta, csv_name))
        classes.update(df["label"].unique())
        for fname, label in zip(df["fname"], df["label"]):
            weak_labels[f"{audio_dir}/{fname}"] = label
    return DatasetAnnotation(label_kind="multiclass", annotation_kind="weak",
                             classes=sorted(classes), weak_labels=weak_labels)
