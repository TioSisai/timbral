"""Unit tests for the HyenaSET adapter's pure HDF5 logic."""

import h5py
import pytest

from timbral.datasets.adapters import hyenaset


def _write_label_file(path, labels, starts, ends, focal_flags):
    """Write a minimal HyenaSET HDF5 test label file.

    Args:
        path: Output HDF5 path.
        labels: Sequence of call types.
        starts: Sequence of event start times in seconds.
        ends: Sequence of event end times in seconds.
        focal_flags: Sequence of event focal flags.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as h5_file:
        if labels:
            string_dtype = h5py.string_dtype(encoding="utf-8")
            h5_file.create_dataset("lbl", data=labels, dtype=string_dtype)
        else:
            # In real data, an empty lbl dataset is float64, not string dtype.
            h5_file.create_dataset("lbl", data=[])
        h5_file.create_dataset("start_time_lbl", data=starts)
        h5_file.create_dataset("end_time_lbl", data=ends)
        h5_file.create_dataset("foc", data=focal_flags)


def test_load_annotation_reads_only_call_types(tmp_path):
    """Regardless of its value, foc must not produce extra events beyond call types."""
    label_path = tmp_path / "lbl" / "24000Hz" / "7" / "sample.h5"
    _write_label_file(
        label_path,
        labels=["groan", "whoop", "snore"],
        starts=[0.25, 2.0, 8.5],
        ends=[1.5, 3.25, 10.0],
        focal_flags=[1, 0, 1],
    )

    annotation = hyenaset.load_annotation(str(tmp_path))
    events = annotation.strong_events[
        "wav/24000Hz/7/sample.wav"
    ]

    assert annotation.label_kind == "multilabel"
    assert annotation.annotation_kind == "strong"
    assert set(annotation.classes) == {
        "groan", "oth", "whoop", "alarm_rumble", "squitter", "squeal",
        "feeding", "giggle", "growl", "snore",
    }
    assert events == [
        ("groan", 0.25, 1.5, 1.0),
        ("whoop", 2.0, 3.25, 1.0),
        ("snore", 8.5, 10.0, 1.0),
    ]


def test_empty_label_file_and_missing_label_are_distinguished(tmp_path):
    """A reviewed but empty HDF5 returns empty events; a missing label follows the mapping's get default."""
    label_path = tmp_path / "lbl" / "24000Hz" / "0" / "empty.h5"
    _write_label_file(label_path, labels=[], starts=[], ends=[], focal_flags=[])
    annotation = hyenaset.load_annotation(str(tmp_path))

    assert annotation.strong_events["wav/24000Hz/0/empty.wav"] == []
    sentinel = object()
    assert annotation.strong_events.get(
        "wav/24000Hz/0/missing.wav", sentinel
    ) is sentinel
    assert annotation.strong_events.get("not-a-hyenaset-path", sentinel) is sentinel


def test_malformed_hdf5_structure_is_not_treated_as_missing_label(tmp_path):
    """When a label exists but is missing a field, it must raise directly rather than being silently treated as no events."""
    label_path = tmp_path / "lbl" / "24000Hz" / "2" / "malformed.h5"
    label_path.parent.mkdir(parents=True)
    with h5py.File(label_path, "w") as h5_file:
        string_dtype = h5py.string_dtype(encoding="utf-8")
        h5_file.create_dataset("lbl", data=["groan"], dtype=string_dtype)
        h5_file.create_dataset("start_time_lbl", data=[0.0])
        h5_file.create_dataset("foc", data=[1])

    annotation = hyenaset.load_annotation(str(tmp_path))
    with pytest.raises(KeyError, match="end_time_lbl"):
        annotation.strong_events.get("wav/24000Hz/2/malformed.wav", [])
