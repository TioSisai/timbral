"""End-to-end unit tests for timbral.datasets.builder: a synthetic small dataset runs the full pipeline."""

import importlib
import json
import os
from pathlib import Path

import fsspec
import numpy as np
import pytest
import soundfile as sf
from datasets import ClassLabel, Dataset, DatasetDict, load_from_disk
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from timbral import storage
from timbral.datasets import adapters, builder
from timbral.datasets.adapters.base import DatasetAnnotation
from timbral.datasets.config import resolve_config

SR = 8000
SEG_SEC = 1.0

# Synthetic dataset: 3 8kHz wavs and their multiclass annotations
_DURATIONS = {"a.wav": 2.5, "b.wav": 1.0, "c.wav": 3.0}
_WEAK_LABELS = {"a.wav": "dog", "b.wav": "cat", "c.wav": "dog"}
# strong events: (class_name, start, end, value); value 1.0=confirmed present, NaN=uncertain
_STRONG_EVENTS = {
    "a.wav": [("dog", 0.2, 1.4, 1.0), ("cat", 1.6, 1.9, float("nan")),
              ("cat", 2.2, 2.9, 1.0)],
    "b.wav": [("cat", 0.0, 1.0, 1.0)],
    "c.wav": [("dog", 0.5, 0.7, 1.0), ("cat", 2.0, 3.0, 1.0)],
}


@pytest.fixture()
def synth_dataset(tmp_path, monkeypatch):
    """Build a synthetic dataset directory + split.json, and inject two test adapters (weak/strong)."""
    dataset_dir = tmp_path / "TestSet"
    dataset_dir.mkdir()
    rng = np.random.default_rng(7)
    for name, dur in _DURATIONS.items():
        data = (rng.standard_normal(int(dur * SR)) * 0.1).astype(np.float32)
        sf.write(dataset_dir / name, data, SR, subtype="FLOAT")

    split_json = tmp_path / "split.json"
    split_json.write_text(json.dumps({
        "train": [{"audio_path": "a.wav", "start": 0.0, "end": "inf"}],
        "validation": [{"audio_path": "b.wav", "start": 0.0, "end": "inf"}],
        "test": [{"audio_path": "c.wav", "start": 0.0, "end": "inf"}],
    }), encoding="utf-8")

    monkeypatch.setitem(adapters.ADAPTERS, "TestSet", lambda d: DatasetAnnotation(
        label_kind="multiclass", annotation_kind="weak",
        classes=["dog", "cat"], weak_labels=dict(_WEAK_LABELS)))
    monkeypatch.setitem(adapters.ADAPTERS, "TestSetStrong", lambda d: DatasetAnnotation(
        label_kind="multilabel", annotation_kind="strong",
        classes=["dog", "cat"], strong_events=dict(_STRONG_EVENTS)))
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    monkeypatch.chdir(work_dir)
    return {"dataset_dir": str(dataset_dir), "split_json": str(split_json)}


def _make_config(synth, dataset_name, **kwargs):
    defaults = dict(dataset_dir=synth["dataset_dir"], split_json=synth["split_json"],
                    sr=SR, seg_sec=SEG_SEC, hop_sec=SEG_SEC, num_proc=1, batch_size=2)
    defaults.update(kwargs)
    return resolve_config(dataset_name, **defaults)


def test_multiclass_weak_end_to_end(synth_dataset, monkeypatch):
    save_paths = []
    original_save_to_disk = DatasetDict.save_to_disk

    def record_save_to_disk(self, dataset_path, *args, **kwargs):
        """Record the local save_to_disk target, then call the original implementation."""
        save_paths.append(dataset_path)
        return original_save_to_disk(self, dataset_path, *args, **kwargs)

    monkeypatch.setattr(DatasetDict, "save_to_disk", record_save_to_disk)
    cfg = _make_config(synth_dataset, "TestSet")
    cache_dir = builder.prepare_dataset(cfg)
    dsd = load_from_disk(cache_dir)

    assert save_paths == [cache_dir]
    assert not any(
        path.name.startswith(".building-")
        for path in Path(cache_dir).parent.iterdir())

    # a.wav 2.5s -> 3 segs (0.5s tail segment); b 1.0s -> 1; c 3.0s -> 3
    assert {s: len(d) for s, d in dsd.items()} == {
        "train": 3, "validation": 1, "test": 3}

    train = dsd["train"]
    assert isinstance(train.features["label"], ClassLabel)
    assert train.features["label"].names == ["cat", "dog"]
    assert train.features["raw"].length == SR
    rows = list(train)
    assert [r["segment_id"] for r in rows] == [0, 1, 2]
    assert all(r["audio_path"] == "a.wav" for r in rows)
    assert [r["start"] for r in rows] == [0.0, 1.0, 2.0]
    assert [r["valid_sec"] for r in rows] == [1.0, 1.0, 0.5]
    assert rows[2]["end"] == pytest.approx(2.5)
    # label: dog -> 1 (lexicographic order)
    assert all(r["label"] == 1 for r in rows)
    assert all(r["sr"] == SR for r in rows)
    # audio_id: lexicographic order a=0, b=1, c=2
    assert rows[0]["audio_id"] == 0
    assert dsd["validation"][0]["audio_id"] == 1
    # Tail-segment padding: 0.5s valid, the rest is 0
    tail = np.asarray(rows[2]["raw"], dtype=np.float32)
    assert tail.shape == (SR,)
    assert np.abs(tail[:SR // 2]).sum() > 0
    np.testing.assert_array_equal(tail[SR // 2:], 0.0)

    # cache auxiliary files
    with open(os.path.join(cache_dir, "label_index.json"), encoding="utf-8") as f:
        assert json.load(f) == {"cat": 0, "dog": 1}
    with open(os.path.join(cache_dir, "prep_config.json"), encoding="utf-8") as f:
        assert json.load(f)["config_hash"] == cfg.config_hash


def test_skip_existing_and_overwrite(synth_dataset, capsys):
    cfg = _make_config(synth_dataset, "TestSet")
    cache_dir = builder.prepare_dataset(cfg)
    mtime = os.path.getmtime(cache_dir)

    assert builder.prepare_dataset(cfg) == cache_dir
    assert "skipping" in capsys.readouterr().out
    assert os.path.getmtime(cache_dir) == mtime

    cfg_overwrite = _make_config(synth_dataset, "TestSet", overwrite=True)
    builder.prepare_dataset(cfg_overwrite)
    assert os.path.getmtime(cache_dir) != mtime
    assert len(load_from_disk(cache_dir)["train"]) == 3


def test_s3_cache_end_to_end(synth_dataset, tmp_path, monkeypatch, capsys):
    """S3 writes the final cache directly; the map's Arrow cache only lands in TMPDIR."""
    class S3MemoryFileSystem(MemoryFileSystem):
        """An in-memory S3 filesystem for offline testing."""

        protocol = "s3"
        store = {}
        pseudo_dirs = [""]

        @classmethod
        def _strip_protocol(cls, path):
            """Strip the path prefix per the ``s3://`` protocol."""
            return AbstractFileSystem._strip_protocol.__func__(cls, path)

    registry = importlib.import_module("fsspec.registry")
    monkeypatch.setitem(registry._registry, "s3", S3MemoryFileSystem)
    monkeypatch.setenv("AWS_S3_ENDPOINT", "a3s.fi")
    map_tmp_root = tmp_path / "map-tmp"
    map_tmp_root.mkdir()
    monkeypatch.setenv("TMPDIR", str(map_tmp_root))

    map_cache_files = []
    save_paths = []
    original_map = Dataset.map
    original_save_to_disk = DatasetDict.save_to_disk

    def record_map(self, *args, **kwargs):
        """Record the map's intermediate cache path, then call the original implementation."""
        map_cache_files.append(kwargs["cache_file_name"])
        return original_map(self, *args, **kwargs)

    def record_save_to_disk(self, dataset_path, *args, **kwargs):
        """Record the save_to_disk target and storage options, then call the original implementation."""
        save_paths.append((dataset_path, kwargs.get("storage_options")))
        return original_save_to_disk(self, dataset_path, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", record_map)
    monkeypatch.setattr(DatasetDict, "save_to_disk", record_save_to_disk)

    root_cfg = _make_config(synth_dataset, "TestSet",
                            cache_dir="s3://test-bucket")
    with pytest.raises(ValueError, match="directory within the bucket"):
        builder.prepare_dataset(root_cfg)

    cache_dir = "s3://test-bucket/cache"
    cfg = _make_config(synth_dataset, "TestSet", cache_dir=cache_dir)
    assert builder.prepare_dataset(cfg) == cache_dir
    assert not any(map_tmp_root.iterdir())

    options = storage.s3_storage_options()
    assert options["endpoint_url"] == "https://a3s.fi"
    fs, cache_path = fsspec.core.url_to_fs(cache_dir, **options)
    dsd = load_from_disk(cache_dir, storage_options=options)
    assert {split: len(ds) for split, ds in dsd.items()} == {
        "train": 3, "validation": 1, "test": 3}
    assert fs.isdir(cache_path)
    with fs.open(f"{cache_path}/label_index.json", "r", encoding="utf-8") as f:
        assert json.load(f) == {"cat": 0, "dog": 1}
    with fs.open(f"{cache_path}/prep_config.json", "r", encoding="utf-8") as f:
        assert json.load(f)["config_hash"] == cfg.config_hash

    assert len(map_cache_files) == 3
    assert all(os.path.commonpath([str(map_tmp_root), path]) == str(map_tmp_root)
               for path in map_cache_files)
    assert all(not path.startswith("s3://") for path in map_cache_files)
    assert save_paths == [(cache_dir, options)]
    assert not any(".building-" in path for path in fs.find("test-bucket"))

    map_call_count = len(map_cache_files)
    assert builder.prepare_dataset(cfg) == cache_dir
    assert "skipping" in capsys.readouterr().out
    assert len(map_cache_files) == map_call_count

    changed = _make_config(synth_dataset, "TestSet", cache_dir=cache_dir,
                           sr=4000)
    with pytest.raises(ValueError, match="config_hash does not match"):
        builder.prepare_dataset(changed)

    fs.rm(cache_path, recursive=True)
    with fs.open(f"{cache_path}/partial", "w") as f:
        f.write("partial")
    with pytest.raises(ValueError, match="missing prep_config"):
        builder.prepare_dataset(cfg)
    assert fs.isfile(f"{cache_path}/partial")
    partial_overwrite = _make_config(
        synth_dataset, "TestSet", cache_dir=cache_dir, overwrite=True)
    assert builder.prepare_dataset(partial_overwrite) == cache_dir
    assert not fs.exists(f"{cache_path}/partial")
    assert fs.isfile(f"{cache_path}/prep_config.json")

    with fs.open(f"{cache_path}/stale", "w") as f:
        f.write("stale")
    overwrite = _make_config(synth_dataset, "TestSet", cache_dir=cache_dir,
                             overwrite=True)
    assert builder.prepare_dataset(overwrite) == cache_dir
    assert not fs.exists(f"{cache_path}/stale")
    assert len(load_from_disk(cache_dir, storage_options=options)["train"]) == 3


def test_local_incomplete_cache_requires_overwrite(synth_dataset, tmp_path):
    """When a local target is missing its completion marker, it is only cleaned up and rebuilt after an explicit overwrite."""
    cache_dir = tmp_path / "partial-cache"
    cache_dir.mkdir()
    partial_file = cache_dir / "partial"
    partial_file.write_text("partial", encoding="utf-8")
    cfg = _make_config(
        synth_dataset, "TestSet", cache_dir=str(cache_dir))

    with pytest.raises(ValueError, match="missing prep_config"):
        builder.prepare_dataset(cfg)
    assert partial_file.is_file()

    overwrite_cfg = _make_config(
        synth_dataset, "TestSet", cache_dir=str(cache_dir), overwrite=True)
    assert builder.prepare_dataset(overwrite_cfg) == str(cache_dir)
    assert not partial_file.exists()
    assert (cache_dir / "prep_config.json").is_file()
    assert len(load_from_disk(cache_dir)["train"]) == 3


def test_strong_labels_clipped_per_segment(synth_dataset):
    cfg = _make_config(synth_dataset, "TestSetStrong", label_type="strong")
    dsd = load_from_disk(builder.prepare_dataset(cfg))

    # target is always a valid ClassLabel; value is a float32 three-state
    # annotation value (1.0/NaN)
    event_feature = dsd["train"].features["label"].feature
    target_feature = event_feature["target"]
    assert isinstance(target_feature, ClassLabel)
    assert target_feature.names == ["cat", "dog"]
    assert event_feature["value"].dtype == "float32"

    # a.wav (2.5s): events dog[0.2,1.4]=1.0, cat[1.6,1.9]=NaN, cat[2.2,2.9]=1.0;
    # seg0 [0,1) seg1 [1,2) seg2 [2,2.5)
    seg0, seg1, seg2 = list(dsd["train"])
    assert seg0["label"] == [
        {"target": 1, "start": 0.2, "end": 1.0, "value": 1.0}]
    assert len(seg1["label"]) == 2
    dog_ev = seg1["label"][0]
    assert dog_ev["target"] == 1
    assert dog_ev["start"] == pytest.approx(0.0)
    assert dog_ev["end"] == pytest.approx(0.4)
    assert dog_ev["value"] == 1.0
    unknown_ev = seg1["label"][1]
    assert unknown_ev["target"] == 0
    assert unknown_ev["start"] == pytest.approx(0.6)
    assert unknown_ev["end"] == pytest.approx(0.9)
    assert np.isnan(unknown_ev["value"])
    # tail segment seg2 valid=0.5s: cat[2.2,2.9] clipped to the valid end -> rel [0.2, 0.5]
    assert len(seg2["label"]) == 1
    assert seg2["label"][0]["target"] == 0
    assert seg2["label"][0]["start"] == pytest.approx(0.2)
    assert seg2["label"][0]["end"] == pytest.approx(0.5)
    assert seg2["label"][0]["value"] == 1.0

    # c.wav (3.0s) seg2 [2,3): cat[2,3] fully covers the segment -> rel [0, 1]
    c_seg2 = dsd["test"][2]
    assert c_seg2["label"] == [
        {"target": 0, "start": 0.0, "end": 1.0, "value": 1.0}]


def test_strong_to_weak_aggregation(synth_dataset):
    cfg = _make_config(synth_dataset, "TestSetStrong", label_type="weak")
    dsd = load_from_disk(builder.prepare_dataset(cfg))

    # three-state multi-hot is uniformly float32
    assert dsd["train"].features["label"].feature.dtype == "float32"
    # a.wav seg1 [1,2): dog POS overlaps -> 1; cat UNK overlaps -> NaN (NaN
    # != NaN, so assert with isnan)
    seg1_label = dsd["train"][1]["label"]
    assert np.isnan(seg1_label[0])
    assert seg1_label[1] == 1.0
    # a.wav seg2 [2,2.5): cat[2.2,2.9] POS overlaps
    assert dsd["train"][2]["label"] == [1.0, 0.0]
    # c.wav seg0 [0,1): dog[0.5,0.7]
    assert dsd["test"][0]["label"] == [0.0, 1.0]


def test_multilabel_weak_labels_float32(synth_dataset, monkeypatch):
    # Pure weak multilabel's multi-hot is likewise uniformly float32 (0/1)
    monkeypatch.setitem(
        adapters.ADAPTERS, "TestSetMulti", lambda d: DatasetAnnotation(
            label_kind="multilabel", annotation_kind="weak",
            classes=["dog", "cat"],
            weak_labels={"a.wav": ["dog", "cat"], "b.wav": ["cat"],
                         "c.wav": ["dog"]}))
    cfg = _make_config(synth_dataset, "TestSetMulti")
    dsd = load_from_disk(builder.prepare_dataset(cfg))

    assert dsd["train"].features["label"].feature.dtype == "float32"
    assert dsd["train"][0]["label"] == [1.0, 1.0]
    assert dsd["validation"][0]["label"] == [1.0, 0.0]
    assert dsd["test"][0]["label"] == [0.0, 1.0]


def test_explicit_cache_dir_config_mismatch_raises(synth_dataset, tmp_path):
    # Re-running with changed parameters under an explicit cache_dir: silently hitting a stale artifact is not allowed
    explicit = str(tmp_path / "explicit_cache")
    cfg = _make_config(synth_dataset, "TestSet", cache_dir=explicit)
    builder.prepare_dataset(cfg)
    cfg_changed = _make_config(synth_dataset, "TestSet", cache_dir=explicit,
                               sr=4000)
    with pytest.raises(ValueError, match="config_hash does not match"):
        builder.prepare_dataset(cfg_changed)
    # Re-running with the same config still skips normally
    assert builder.prepare_dataset(cfg) == explicit


def test_multiprocess_map(synth_dataset):
    # num_proc > 1 requires the map closure to be serializable, and the result must match the single-process run
    cfg = _make_config(synth_dataset, "TestSet", num_proc=2, batch_size=1)
    dsd = load_from_disk(builder.prepare_dataset(cfg))
    assert {s: len(d) for s, d in dsd.items()} == {
        "train": 3, "validation": 1, "test": 3}
    assert [r["segment_id"] for r in dsd["train"]] == [0, 1, 2]


def test_weak_dataset_strong_label_type_raises(synth_dataset):
    cfg = _make_config(synth_dataset, "TestSet", label_type="strong")
    with pytest.raises(ValueError, match="cannot produce label_type=strong"):
        builder.prepare_dataset(cfg)


def test_weak_missing_annotation_raises(synth_dataset, monkeypatch):
    monkeypatch.setitem(adapters.ADAPTERS, "TestSet", lambda d: DatasetAnnotation(
        label_kind="multiclass", annotation_kind="weak",
        classes=["dog", "cat"], weak_labels={"a.wav": "dog"}))
    cfg = _make_config(synth_dataset, "TestSet")
    with pytest.raises(ValueError, match="are missing annotations"):
        builder.prepare_dataset(cfg)
