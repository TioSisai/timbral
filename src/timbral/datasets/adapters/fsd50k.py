"""FSD50K annotation adapter: ground_truth dev/eval csv -> multilabel weak."""

import os

import pandas as pd

from .base import DatasetAnnotation

# Single source of truth for the directory layout, referenced directly by the split generator
GROUND_TRUTH_DIR = "FSD50K.ground_truth"
DEV_AUDIO_DIR = "FSD50K.dev_audio"
EVAL_AUDIO_DIR = "FSD50K.eval_audio"


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load FSD50K official annotations; the labels column is comma-separated; the full class set comes from vocabulary.csv."""
    gt = os.path.join(dataset_dir, GROUND_TRUTH_DIR)
    weak_labels = {}
    for audio_dir, csv_name in ((DEV_AUDIO_DIR, "dev.csv"),
                                (EVAL_AUDIO_DIR, "eval.csv")):
        df = pd.read_csv(os.path.join(gt, csv_name), dtype={"fname": str})
        for fname, label_str in zip(df["fname"], df["labels"]):
            weak_labels[f"{audio_dir}/{fname}.wav"] = label_str.split(",")
    vocab = pd.read_csv(os.path.join(gt, "vocabulary.csv"), header=None)
    return DatasetAnnotation(label_kind="multilabel", annotation_kind="weak",
                             classes=vocab[1].tolist(), weak_labels=weak_labels)
