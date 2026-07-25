"""Unit tests for timbral.datasets.split_io: parsing and validation rules."""

import json

import pytest

from timbral.datasets import split_io


def _write_split(tmp_path, obj):
    path = tmp_path / "split.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def _entry(audio_path, start=0.0, end="inf"):
    return {"audio_path": audio_path, "start": start, "end": end}


def test_parses_inf_string(tmp_path):
    path = _write_split(tmp_path, {
        "train": [_entry("a.wav")],
        "validation": [_entry("b.wav")],
        "test": [_entry("c.wav", 0.0, 20.0)],
    })
    splits = split_io.load_split_json(path)
    assert set(splits) == set(split_io.SPLIT_NAMES)
    assert splits["train"][0] == split_io.SplitEntry("a.wav", 0.0, float("inf"))
    assert splits["test"][0].end == 20.0


def test_missing_split_key_raises(tmp_path):
    path = _write_split(tmp_path, {"train": [], "validation": []})
    with pytest.raises(ValueError, match="is missing split"):
        split_io.load_split_json(path)


def test_duplicate_audio_in_same_split_raises(tmp_path):
    path = _write_split(tmp_path, {
        "train": [_entry("a.wav", 0.0, 10.0), _entry("a.wav", 10.0, 20.0)],
        "validation": [], "test": [],
    })
    with pytest.raises(ValueError, match="Duplicate audio_path"):
        split_io.load_split_json(path)


def test_cross_split_identical_interval_is_copy(tmp_path):
    path = _write_split(tmp_path, {
        "train": [_entry("a.wav")],
        "validation": [_entry("b.wav")],
        "test": [_entry("b.wav")],
    })
    splits = split_io.load_split_json(path)
    assert splits["validation"][0] == splits["test"][0]


def test_cross_split_adjacent_intervals_ok(tmp_path):
    path = _write_split(tmp_path, {
        "train": [_entry("a.wav", 0.0, 20.0)],
        "validation": [_entry("a.wav", 20.0, 40.0)],
        "test": [_entry("b.wav")],
    })
    split_io.load_split_json(path)


def test_cross_split_partial_overlap_raises(tmp_path):
    path = _write_split(tmp_path, {
        "train": [_entry("a.wav", 0.0, 25.0)],
        "validation": [_entry("a.wav", 20.0, 40.0)],
        "test": [_entry("b.wav")],
    })
    with pytest.raises(ValueError, match="partially overlapping"):
        split_io.load_split_json(path)


def test_invalid_interval_raises(tmp_path):
    path = _write_split(tmp_path, {
        "train": [_entry("a.wav", 20.0, 20.0)],
        "validation": [], "test": [],
    })
    with pytest.raises(ValueError, match="Invalid interval"):
        split_io.load_split_json(path)
