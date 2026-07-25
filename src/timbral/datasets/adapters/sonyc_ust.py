"""SONYC-UST annotation adapter: annotations.csv -> multilabel weak (8 coarse classes).

Multi-annotator aggregation rule (as specified by the user): for audio with an
annotator_id==0 row (the SONYC team's two-stage verified ground truth), that row
takes precedence; for the remaining audio, an any-vote is applied across all
annotation rows (positive if any annotator marks presence==1); a presence value
of -1 (unannotated) is treated as 0.
"""

import os

import pandas as pd

from .base import DatasetAnnotation


def aggregate_presence(df: pd.DataFrame, presence_cols: list) -> pd.DataFrame:
    """Aggregate multi-annotator rows into a per-audio 0/1 presence matrix.

    Args:
        df: The annotations.csv DataFrame (must contain audio_filename/annotator_id
            and presence columns).
        presence_cols: List of presence column names to aggregate.

    Returns:
        pd.DataFrame: A 0/1 matrix indexed by audio_filename, with verified rows
        taking precedence.
    """
    positive = (df[presence_cols] == 1).astype("int8")
    positive.index = df["audio_filename"]
    crowd = positive.groupby(level=0).max()
    verified_mask = (df["annotator_id"] == 0).to_numpy()
    verified = positive[verified_mask].groupby(level=0).max()
    crowd.loc[verified.index] = verified
    return crowd


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load SONYC-UST annotations; audio_path looks like 'audio/{audio_filename}'."""
    df = pd.read_csv(os.path.join(dataset_dir, "annotations.csv"))
    coarse_cols = [c for c in df.columns
                   if c.endswith("_presence") and "-" not in c.split("_")[0]]
    agg = aggregate_presence(df, coarse_cols)
    classes = [c.removesuffix("_presence") for c in coarse_cols]
    weak_labels = {
        f"audio/{fn}": [cls for cls, v in zip(classes, row) if v]
        for fn, row in zip(agg.index, agg.to_numpy())
    }
    return DatasetAnnotation(label_kind="multilabel", annotation_kind="weak",
                             classes=classes, weak_labels=weak_labels)
