"""Unit tests for the DCASE-2024-Task-5 adapter using synthetic metadata."""

import math

import pandas as pd

from timbral.datasets.adapters import dcase2024_task5


def _write_csv(path, rows):
    """Create the parent directory and write a synthetic CSV.

    Args:
        path: Output CSV path.
        rows: Row records passed to ``pandas.DataFrame``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _make_synthetic_dataset(root):
    """Build a minimal dataset covering the train/validation mapping, colliding
    class names, all three states, and POS/UNK conflicts.

    Args:
        root: Root directory of the synthetic dataset.
    """
    _write_csv(root / "DCASE2024_task5_training_set_classes.csv", [
        {"dataset": "BV", "class_code": "A", "class_name": "Bird A"},
        {"dataset": "BV", "class_code": "B", "class_name": "Bird B"},
        {"dataset": "HT", "class_code": "H", "class_name": "Hyena"},
        {"dataset": "WMW", "class_code": "c_4", "class_name": "Same bird"},
        {"dataset": "WMW", "class_code": "c_5", "class_name": "Same bird"},
    ])
    _write_csv(root / "DCASE2024_task5_validation_set_classes.csv", [
        {"dataset": "HB", "recording": "unused", "class_code": "Q",
         "class_name": "Mosquito"},
        {"dataset": "PB", "recording": "shared", "class_code": "Q",
         "class_name": "Blackbird call"},
        {"dataset": "PB24", "recording": "shared", "class_code": "Q",
         "class_name": "Song thrush call"},
    ])

    for partition, subsets in dcase2024_task5.PARTITION_SUBSETS:
        for subset in subsets:
            (root / "Development_Set" / partition / subset).mkdir(
                parents=True, exist_ok=True)

    _write_csv(root / "Development_Set/Training_Set/BV/train_clip.csv", [
        {"Audiofilename": "wrong-name.wav", "Starttime": 0.0,
         "Endtime": 0.5, "A": "POS", "B": "NEG"},
        {"Audiofilename": "wrong-name.wav", "Starttime": 1.0,
         "Endtime": 1.5, "A": "NEG", "B": "POS"},
        {"Audiofilename": "wrong-name.wav", "Starttime": 2.0,
         "Endtime": 2.5, "A": "UNK", "B": "UNK"},
        {"Audiofilename": "wrong-name.wav", "Starttime": 3.0,
         "Endtime": 3.5, "A": "NEG", "B": "NEG"},
        {"Audiofilename": "wrong-name.wav", "Starttime": 4.0,
         "Endtime": 4.5, "A": "UNK", "B": "NEG"},
    ])
    _write_csv(root / "Development_Set/Training_Set/BV/mixed_clip.csv", [
        {"Audiofilename": "mixed_clip.wav", "Starttime": 0.0,
         "Endtime": 0.5, "A": "POS", "B": "UNK"},
    ])
    _write_csv(root / "Development_Set/Training_Set/HT/hyena.csv", [
        {"Audiofilename": "hyena.csv", "Starttime": 5.0, "Endtime": 5.5,
         "H": "POS"},
    ])
    _write_csv(root / "Development_Set/Training_Set/WMW/wetland.csv", [
        {"Audiofilename": "wetland.wav", "Starttime": 6.0, "Endtime": 6.5,
         "c_4": "POS", "c_5": "NEG"},
        {"Audiofilename": "wetland.wav", "Starttime": 7.0, "Endtime": 7.5,
         "c_4": "NEG", "c_5": "POS"},
        {"Audiofilename": "wetland.wav", "Starttime": 8.0, "Endtime": 8.5,
         "c_4": "POS", "c_5": "POS"},
    ])
    _write_csv(root / "Development_Set/Training_Set/WMW/conflict.csv", [
        {"Audiofilename": "conflict.wav", "Starttime": 1.0, "Endtime": 1.5,
         "c_4": "POS", "c_5": "UNK"},
        {"Audiofilename": "conflict.wav", "Starttime": 2.0, "Endtime": 2.5,
         "c_4": "UNK", "c_5": "POS"},
        {"Audiofilename": "conflict.wav", "Starttime": 3.0, "Endtime": 3.5,
         "c_4": "UNK", "c_5": "UNK"},
    ])
    _write_csv(root / "Development_Set/Validation_Set/PB/shared.csv", [
        {"Audiofilename": "shared.wav", "Starttime": 8.0, "Endtime": 8.5,
         "Q": "POS"},
        {"Audiofilename": "shared.wav", "Starttime": 9.0, "Endtime": 9.5,
         "Q": "UNK"},
    ])
    _write_csv(root / "Development_Set/Validation_Set/PB24/shared.csv", [
        {"Audiofilename": "shared.wav", "Starttime": 10.0,
         "Endtime": 10.5, "Q": "POS"},
    ])


def test_dcase2024_task5_class_universe_and_path_mapping(tmp_path):
    """The class table provides the full universe; classes map by subset + recording, and the path never trusts the filename in the table."""
    _make_synthetic_dataset(tmp_path)
    annotation = dcase2024_task5.load_annotation(str(tmp_path))

    assert annotation.label_kind == "multilabel"
    assert annotation.annotation_kind == "strong"
    assert set(annotation.classes) == {
        "Bird A", "Bird B", "Hyena", "Mosquito", "Blackbird call",
        "Song thrush call", "Same bird",
    }
    assert annotation.strong_events[
        "Development_Set/Training_Set/HT/hyena.wav"] == [
            ("Hyena", 5.0, 5.5, 1.0)]
    assert annotation.strong_events[
        "Development_Set/Validation_Set/PB/shared.wav"][0] == (
            "Blackbird call", 8.0, 8.5, 1.0)
    assert annotation.strong_events[
        "Development_Set/Validation_Set/PB24/shared.wav"] == [
            ("Song thrush call", 10.0, 10.5, 1.0)]


def test_dcase2024_task5_pos_unknown_and_negative_semantics(tmp_path):
    """POS yields a value=1.0 event; UNK yields a per-class value=NaN event without collapsing; NEG and unannotated rows are ignored."""
    _make_synthetic_dataset(tmp_path)
    strong_events = dcase2024_task5.load_annotation(str(tmp_path)).strong_events

    events = strong_events["Development_Set/Training_Set/BV/train_clip.wav"]
    assert [event[:3] for event in events] == [
        ("Bird A", 0.0, 0.5),
        ("Bird B", 1.0, 1.5),
        ("Bird A", 2.0, 2.5),
        ("Bird B", 2.0, 2.5),
        ("Bird A", 4.0, 4.5),
    ]
    assert [event[3] for event in events[:2]] == [1.0, 1.0]
    assert all(math.isnan(event[3]) for event in events[2:])

    validation_events = strong_events[
        "Development_Set/Validation_Set/PB/shared.wav"]
    assert validation_events[1][:3] == ("Blackbird call", 9.0, 9.5)
    assert math.isnan(validation_events[1][3])


def test_dcase2024_task5_duplicate_display_names_are_merged(tmp_path):
    """Two WMW class_codes sharing the same display name are merged into one readable class, per the confirmed convention."""
    _make_synthetic_dataset(tmp_path)
    events = dcase2024_task5.load_annotation(str(tmp_path)).strong_events[
        "Development_Set/Training_Set/WMW/wetland.wav"]

    assert events == [
        ("Same bird", 6.0, 6.5, 1.0),
        ("Same bird", 7.0, 7.5, 1.0),
        ("Same bird", 8.0, 8.5, 1.0),
    ]


def test_dcase2024_task5_row_mixing_and_pos_priority(tmp_path):
    """A row mixing POS+UNK yields one event per class; when same-name codes collide, POS takes priority without duplication."""
    _make_synthetic_dataset(tmp_path)
    strong_events = dcase2024_task5.load_annotation(str(tmp_path)).strong_events

    mixed_events = strong_events[
        "Development_Set/Training_Set/BV/mixed_clip.wav"]
    assert [event[:3] for event in mixed_events] == [
        ("Bird A", 0.0, 0.5),
        ("Bird B", 0.0, 0.5),
    ]
    assert mixed_events[0][3] == 1.0
    assert math.isnan(mixed_events[1][3])

    conflict_events = strong_events[
        "Development_Set/Training_Set/WMW/conflict.wav"]
    assert [event[:3] for event in conflict_events] == [
        ("Same bird", 1.0, 1.5),
        ("Same bird", 2.0, 2.5),
        ("Same bird", 3.0, 3.5),
    ]
    assert [conflict_events[0][3], conflict_events[1][3]] == [1.0, 1.0]
    assert math.isnan(conflict_events[2][3])
