"""BSD35K / BSD10K annotation adapter: BST second-level short codes from metadata csv -> multiclass weak.

The two datasets share the Broad Sound Taxonomy (BST); class names use the
qualified format "top-level readable name / second-level readable name" (the
README's second-level readable names collide, e.g. all 5 top-level categories
have an "Other", so a top-level qualifier is required).
"""

import os

import pandas as pd

from .base import DatasetAnnotation

# BST second-level short code -> "top-level readable name / second-level readable name"
# (per the BSD35K/BSD10K README category list)
_CODE_TO_NAME = {
    "fx-a": "Sound effects / Animals",
    "fx-el": "Sound effects / Electronic / Design",
    "fx-ex": "Sound effects / Experimental",
    "fx-h": "Sound effects / Human sounds and actions",
    "fx-m": "Sound effects / Other mechanisms, engines, machines",
    "fx-n": "Sound effects / Natural elements and explosions",
    "fx-o": "Sound effects / Objects / House appliances",
    "fx-other": "Sound effects / Other",
    "fx-v": "Sound effects / Vehicles",
    "is-e": "Instrument samples / Synths / Electronic",
    "is-k": "Instrument samples / Piano / Keyboard instruments",
    "is-other": "Instrument samples / Other",
    "is-p": "Instrument samples / Percussion",
    "is-s": "Instrument samples / String",
    "is-w": "Instrument samples / Wind",
    "m-m": "Music / Multiple instruments",
    "m-other": "Music / Other",
    "m-si": "Music / Solo instrument",
    "m-sp": "Music / Solo percussion",
    "sp-c": "Speech / Conversation / Crowd",
    "sp-other": "Speech / Other",
    "sp-p": "Speech / Processed / Synthetic",
    "sp-s": "Speech / Solo speech",
    "ss-i": "Soundscapes / Indoors",
    "ss-n": "Soundscapes / Nature",
    "ss-other": "Soundscapes / Other",
    "ss-s": "Soundscapes / Synthetic / Artificial",
    "ss-u": "Soundscapes / Urban",
}


# Single source of truth for the metadata filename layout, referenced directly by the split generator
BSD35K_METADATA_CSV = "BSD35k-CS_metadata.csv"
BSD10K_METADATA_CSV = "BSD10k_metadata.csv"


def audio_relpath(sound_id) -> str:
    """The relative audio path corresponding to sound_id (shared with the split generator).

    Args:
        sound_id: The sound_id in metadata csv.

    Returns:
        Dataset-relative path, of the form ``"audio/{sound_id}.wav"``.
    """
    return f"audio/{sound_id}.wav"


def _load(dataset_dir: str, csv_name: str) -> DatasetAnnotation:
    """Load a BSD-family metadata csv; audio_path looks like 'audio/{sound_id}.wav'."""
    df = pd.read_csv(os.path.join(dataset_dir, "metadata", csv_name))
    weak_labels = {audio_relpath(sid): _CODE_TO_NAME[cls]
                   for sid, cls in zip(df["sound_id"], df["class"].astype(str))}
    classes = sorted(_CODE_TO_NAME[c] for c in df["class"].astype(str).unique())
    return DatasetAnnotation(label_kind="multiclass", annotation_kind="weak",
                             classes=classes, weak_labels=weak_labels)


def load_annotation_bsd35k(dataset_dir: str) -> DatasetAnnotation:
    """BSD35K (crowdsourced annotations, 28 classes)."""
    return _load(dataset_dir, BSD35K_METADATA_CSV)


def load_annotation_bsd10k(dataset_dir: str) -> DatasetAnnotation:
    """BSD10K (expert annotations, 23 classes)."""
    return _load(dataset_dir, BSD10K_METADATA_CSV)
