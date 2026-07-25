"""AudioSetStrong annotation adapter: official_metadata event tsv files -> multilabel strong.

The full class set is taken from mid_to_display_name.tsv (456 classes); all events
have value=1.0 (confirmed present), and any mid outside the vocabulary fails fast
via KeyError.
"""

import os

import pandas as pd

from .base import DatasetAnnotation

# Single source of truth for the audio directory layout, referenced directly by the split generator
TRAIN_AUDIO_DIR = "audio/train"
EVAL_AUDIO_DIR = "audio/eval"

# (tsv filename, audio_path prefix) -- the wav filename is "{segment_id}.wav"
_SOURCES = (
    ("audioset_train_strong.tsv", TRAIN_AUDIO_DIR),
    ("audioset_eval_strong.tsv", EVAL_AUDIO_DIR),
)


def load_annotation(dataset_dir: str) -> DatasetAnnotation:
    """Load AudioSetStrong event-level annotations; audio_path looks like 'audio/train/{segment_id}.wav'."""
    meta = os.path.join(dataset_dir, "official_metadata")
    mid_map = pd.read_csv(os.path.join(meta, "mid_to_display_name.tsv"),
                          sep="\t", header=None, names=["mid", "display_name"])
    mid_to_name = dict(zip(mid_map["mid"], mid_map["display_name"]))
    strong_events = {}
    for tsv_name, audio_prefix in _SOURCES:
        df = pd.read_csv(os.path.join(meta, tsv_name), sep="\t")
        for seg_id, start, end, mid in zip(df["segment_id"],
                                           df["start_time_seconds"],
                                           df["end_time_seconds"], df["label"]):
            strong_events.setdefault(f"{audio_prefix}/{seg_id}.wav", []).append(
                (mid_to_name[mid], float(start), float(end), 1.0))
    return DatasetAnnotation(label_kind="multilabel", annotation_kind="strong",
                             classes=sorted(mid_to_name.values()),
                             strong_events=strong_events)
