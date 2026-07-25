"""AudioSetWeak annotation adapter: the three official_metadata segments csv files -> multilabel weak.

Audio is scattered across the balanced/eval/41 unbalanced part directories, with
filenames of the form "{YTID}.wav"; annotations are keyed globally by YTID. To
avoid scanning roughly 2 million files, weak_labels uses a lazy mapping that
resolves YTID from the basename (it only needs to support the builder's `in`
and `[]` queries).
"""

import os

import pandas as pd

from .base import DatasetAnnotation

_SEGMENT_CSVS = ("balanced_train_segments.csv", "unbalanced_train_segments.csv",
                 "eval_segments.csv")


class _YtidKeyedLabels:
    """A label mapping keyed by audio_path for lookups, internally stored by basename-derived YTID."""

    def __init__(self, ytid_to_classes: dict):
        self._by_ytid = ytid_to_classes

    @staticmethod
    def _ytid(audio_path: str) -> str:
        return os.path.basename(audio_path).removesuffix(".wav")

    def __contains__(self, audio_path: str) -> bool:
        return self._ytid(audio_path) in self._by_ytid

    def __getitem__(self, audio_path: str) -> list:
        return self._by_ytid[self._ytid(audio_path)]

    def get(self, audio_path: str, default=None):
        return self._by_ytid.get(self._ytid(audio_path), default)


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load AudioSetWeak official annotations (positive_labels is a comma-separated list of mids)."""
    meta = os.path.join(dataset_dir, "official_metadata")
    cli = pd.read_csv(os.path.join(meta, "class_labels_indices.csv"))
    mid_to_name = dict(zip(cli["mid"], cli["display_name"]))
    ytid_to_classes = {}
    for csv_name in _SEGMENT_CSVS:
        df = pd.read_csv(os.path.join(meta, csv_name), header=None, comment="#",
                         names=["YTID", "start_seconds", "end_seconds",
                                "positive_labels"],
                         skipinitialspace=True)
        for ytid, mids in zip(df["YTID"], df["positive_labels"]):
            ytid_to_classes[ytid] = [mid_to_name[m] for m in mids.split(",")]
    return DatasetAnnotation(label_kind="multilabel", annotation_kind="weak",
                             classes=cli["display_name"].tolist(),
                             weak_labels=_YtidKeyedLabels(ytid_to_classes))
