"""Unit tests for the RealDESED adapter and split generator using synthetic data."""

import pandas as pd

from timbral.datasets.adapters import realdesed
from timbral.datasets.split_generators import realdesed as realdesed_split


def _write_csv(path, rows):
    """Create the parent directory and write a synthetic CSV.

    Args:
        path: Output CSV path.
        rows: Row records passed to ``pandas.DataFrame``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_synthetic_dataset(root):
    """Build a minimal dataset covering the class-vocabulary union, unannotated audio, and audio missing from disk.

    Args:
        root: Root directory of the synthetic dataset.

    Layout:
      - train: 000001.wav (2 events, including a repeated filename) and
        000002.wav (1 event) are annotated and present on disk; 000009.wav is
        annotated but missing from disk; 000003.wav is present on disk but
        unannotated (should be excluded from the split).
      - validation: 000004.wav carries a class unique to validation (verifies
        the vocabulary is unioned across splits).
      - test: 000005.wav has a single event.
    """
    _write_csv(root / "train" / "annotations.csv", [
        {"filename": "000001.wav", "class": "phone_ringing",
         "onset": 9.1, "offset": 13.8},
        {"filename": "000001.wav", "class": "keyboard_typing",
         "onset": 0.0, "offset": 31.2},
        {"filename": "000002.wav", "class": "keychain",
         "onset": 24.9, "offset": 25.9},
        {"filename": "000001.wav", "class": "phone_ringing",
         "onset": 14.8, "offset": 17.8},
        {"filename": "000009.wav", "class": "keychain",
         "onset": 1.0, "offset": 2.0},
    ])
    _write_csv(root / "validation" / "annotations.csv", [
        {"filename": "000004.wav", "class": "vacuum_cleaner",
         "onset": 3.0, "offset": 8.5},
    ])
    _write_csv(root / "test" / "annotations.csv", [
        {"filename": "000005.wav", "class": "footsteps",
         "onset": 0.5, "offset": 4.0},
    ])
    for split, filenames in (("train", ("000001.wav", "000002.wav",
                                        "000003.wav")),
                             ("validation", ("000004.wav",)),
                             ("test", ("000005.wav",))):
        audio_dir = root / split / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            (audio_dir / filename).touch()


def test_realdesed_adapter_events_and_class_union(tmp_path):
    """Events are converted row-by-row into (class, onset, offset, 1.0); the vocabulary is the union across the three splits."""
    _make_synthetic_dataset(tmp_path)
    annotation = realdesed.load_annotation(str(tmp_path))

    assert annotation.label_kind == "multilabel"
    assert annotation.annotation_kind == "strong"
    assert annotation.classes == ["footsteps", "keyboard_typing", "keychain",
                                  "phone_ringing", "vacuum_cleaner"]
    assert annotation.strong_events["train/audio/000001.wav"] == [
        ("phone_ringing", 9.1, 13.8, 1.0),
        ("keyboard_typing", 0.0, 31.2, 1.0),
        ("phone_ringing", 14.8, 17.8, 1.0),
    ]
    assert annotation.strong_events["validation/audio/000004.wav"] == [
        ("vacuum_cleaner", 3.0, 8.5, 1.0)]
    assert annotation.strong_events["test/audio/000005.wav"] == [
        ("footsteps", 0.5, 4.0, 1.0)]
    # Unannotated audio (000003.wav) produces no event key
    assert "train/audio/000003.wav" not in annotation.strong_events


def test_realdesed_split_membership_and_exclusions(tmp_path):
    """Membership = unique filenames in annotations intersected with what exists on disk; both unannotated and missing-from-disk files are excluded."""
    _make_synthetic_dataset(tmp_path)
    splits = realdesed_split.generate(str(tmp_path))

    # 000001 recurring is counted only once and keeps first-occurrence order; 000009 missing from disk is dropped
    assert [entry["audio_path"] for entry in splits["train"]] == [
        "train/audio/000001.wav", "train/audio/000002.wav"]
    assert [entry["audio_path"] for entry in splits["validation"]] == [
        "validation/audio/000004.wav"]
    assert [entry["audio_path"] for entry in splits["test"]] == [
        "test/audio/000005.wav"]
    # 000003.wav is present on disk but unannotated, so it enters no split
    all_paths = {entry["audio_path"]
                 for entries in splits.values() for entry in entries}
    assert "train/audio/000003.wav" not in all_paths
    # Whole-file entry: start=0, end=inf
    assert all(entry["start"] == 0.0 and entry["end"] == float("inf")
               for entries in splits.values() for entry in entries)
