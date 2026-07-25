"""End-to-end unit tests for timbral.embeddings.builder: real raw cache + a registered deterministic fake model."""

import dataclasses
import importlib
import json
import os
from pathlib import Path

import fsspec
import numpy as np
import pytest
import soundfile as sf
import torch
from datasets import (Array2D, ClassLabel, Dataset, DatasetDict, Sequence,
                      Value, load_from_disk)
from datasets.fingerprint import Hasher
from fsspec import AbstractFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from timbral import storage
from timbral.datasets import adapters
from timbral.datasets.adapters.base import DatasetAnnotation
from timbral.datasets.builder import prepare_dataset
from timbral.datasets.config import resolve_config as resolve_raw_config
from timbral.embeddings import builder
from timbral.embeddings.builder import prepare_embeddings
from timbral.embeddings.config import resolve_config
from timbral.models import ModelSpec
from timbral.models import registry as models_registry
from timbral.models.encoders.base import BaseEncoder
from timbral.models.transforms.base import BaseTransform

SR = 8000
SEG_SEC = 1.0
EMB_DIM = 4
FILL_VALUE = 0.5
MODEL_NAME = "fake/enc"
FRAME_SEC = 0.25
NUM_FRAMES = 3

# Synthetic dataset: 3 wavs at 8kHz; a 2.5s -> 3 segments (0.5s tail), b 1.0s
# -> 1 segment, c 3.0s -> 3 segments
_DURATIONS = {"a.wav": 2.5, "b.wav": 1.0, "c.wav": 3.0}
_WEAK_LABELS = {"a.wav": "dog", "b.wav": "cat", "c.wav": "dog"}
_MULTI_LABELS = {"a.wav": ["dog", "cat"], "b.wav": ["cat"], "c.wav": ["dog"]}

# strong events: (class_name, start, end, value); class indices in
# lexicographic order: bird=0 cat=1 dog=2.
# Covers: pure POS (a seg0 dog / c each segment dog), pure UNK (a seg0 cat /
# b seg0 bird), same-class POS+UNK conflict (a seg1 cat -> 1), empty-event
# slice (a seg2 -> all 0), cross-segment event (c dog).
_STRONG_CLASSES = ["dog", "cat", "bird"]
_STRONG_EVENTS = {
    "a.wav": [("dog", 0.2, 0.9, 1.0), ("cat", 0.1, 0.5, float("nan")),
              ("cat", 1.2, 1.5, 1.0), ("cat", 1.3, 1.8, float("nan"))],
    "b.wav": [("bird", 0.0, 0.6, float("nan"))],
    "c.wav": [("dog", 0.5, 2.2, 1.0), ("cat", 2.5, 2.9, float("nan"))],
}

# Synthetic data dedicated to frame boundary cases: exactly one segment per
# audio (SEG_SEC=1.0), fake frame Encoder T_full=3 / FRAME_SEC=0.25 (exact
# in binary). A full-length segment (1.0s) has valid frame count
# ceil(1.0/0.25)=4 clamped to 3, with the last valid frame's endpoint
# [0.5, 0.75) absorbed to 1.0 -> absorption region [0.75, 1.0); b.wav's
# 0.5s short segment has 2 valid frames + 1 padding frame.
_FRAME_DURATIONS = {"a.wav": 1.0, "b.wav": 0.5, "c.wav": 1.0, "d.wav": 1.0,
                    "e.wav": 0.2}
_FRAME_EVENTS = {
    # a: dog crosses the frame 0/1 boundary; cat POS start == frame 0
    # slot's end (zero-length intersection, frame 1 only); cat UNK end ==
    # frame 1 slot's start (frame 0 only); bird falls in the last frame's
    # absorption region (frame 2 only)
    "a.wav": [("dog", 0.2, 0.3, 1.0), ("cat", 0.25, 0.4, 1.0),
              ("cat", 0.1, 0.25, float("nan")), ("bird", 0.8, 0.95, 1.0)],
    "b.wav": [("cat", 0.3, 0.45, 1.0)],
    # c: frame 0 same-class POS+UNK -> 1; frame 1 has different-class POS
    # (bird) and UNK (cat) coexisting
    "c.wav": [("dog", 0.05, 0.1, 1.0), ("dog", 0.12, 0.2, float("nan")),
              ("bird", 0.26, 0.45, 1.0), ("cat", 0.3, 0.4, float("nan"))],
    # d: no events -> all valid frames are 0; e: 0.2s extremely short
    # segment -> 1 valid frame + 2 padding frames
}


def _write_synth_source(tmp_path):
    """Write out a synthetic wav source directory and a three-split split.json; return (dataset_dir, split_json)."""
    dataset_dir = tmp_path / "source"
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
    return dataset_dir, split_json


class _FakeTransform(BaseTransform):
    """Deterministic fake Transform: input_features is the raw waveform value, valid_seconds passes through unchanged in seconds."""

    def __init__(self) -> None:
        super().__init__()
        self.target_sample_rate = SR
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def forward(self, waveform, *, sample_rate, valid_seconds=None):
        waveform = waveform.to(device=self.device, dtype=torch.float32)
        valid_seconds = valid_seconds.to(device=self.device,
                                         dtype=torch.float32)
        return {"input_features": waveform, "valid_seconds": valid_seconds}


class _FakeHighSrTransform(_FakeTransform):
    """Fake Transform whose native sample rate is higher than the cache's sr, used for the low-sample-rate hint test case."""

    def __init__(self) -> None:
        super().__init__()
        self.target_sample_rate = SR * 2


class _FakeEncoder(BaseEncoder):
    """Deterministic fake Encoder: outputs embeddings from which the waveform and valid_seconds paths can be recovered.

    clip: embedding[i, 0] = waveform[i].sum(), embedding[i, 1] =
    valid_seconds[i], remaining dims are always FILL_VALUE. frame follows
    the new contract of outputting a zero-padded result for the maximum
    valid frame count within the batch (mimicking PANNs' batch-dependent
    T): embedding[b, t, 0] = b*10 + t, valid frame count is
    ceil(valid_seconds / FRAME_SEC) clamped to NUM_FRAMES, geometry uses
    the nominal grid points with the last valid frame's endpoint absorbed
    into valid_seconds, and invalid frames have embedding/geometry filled
    with 0 and mask False.
    """

    supported_granularities = frozenset(("clip", "frame"))
    embedding_dim = EMB_DIM

    def __init__(self, *, granularity):
        super().__init__(granularity)
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

    @property
    def device(self) -> torch.device:
        return self.anchor.device

    def _encode_clip(self, input_features, *, valid_seconds):
        num_rows = input_features.shape[0]
        embedding = input_features.new_full((num_rows, EMB_DIM), FILL_VALUE)
        embedding[:, 0] = input_features.sum(dim=1)
        embedding[:, 1] = valid_seconds
        geometry = torch.stack(
            (torch.zeros_like(valid_seconds), valid_seconds), dim=1)
        valid_mask = torch.ones((num_rows,), dtype=torch.bool,
                                device=input_features.device)
        return {"embedding": embedding, "geometry": geometry,
                "valid_mask": valid_mask}

    def _encode_frame(self, input_features, *, valid_seconds):
        num_rows = input_features.shape[0]
        device = input_features.device
        num_valid = (valid_seconds / FRAME_SEC).ceil().long().clamp(
            max=NUM_FRAMES)
        num_frames = int(num_valid.max().item())
        rows = torch.arange(num_rows, device=device).unsqueeze(1)
        frames = torch.arange(num_frames, device=device).unsqueeze(0)
        embedding = input_features.new_full((num_rows, num_frames, EMB_DIM),
                                            FILL_VALUE)
        embedding[:, :, 0] = (rows * 10 + frames).to(torch.float32)
        starts = (frames * FRAME_SEC).expand(num_rows, -1)
        ends = torch.where(frames == (num_valid - 1).unsqueeze(1),
                           valid_seconds.unsqueeze(1), starts + FRAME_SEC)
        geometry = torch.stack([starts, ends], dim=-1)
        valid_mask = frames < num_valid.unsqueeze(1)
        embedding = embedding * valid_mask.unsqueeze(-1)
        geometry = geometry * valid_mask.unsqueeze(-1)
        return {"embedding": embedding, "geometry": geometry,
                "valid_mask": valid_mask}


@pytest.fixture()
def raw_caches(tmp_path, monkeypatch):
    """Build tiny weak caches (one multiclass, one multilabel) via the real prepare_dataset."""
    dataset_dir, split_json = _write_synth_source(tmp_path)

    monkeypatch.setitem(adapters.ADAPTERS, "EmbSet", lambda d: DatasetAnnotation(
        label_kind="multiclass", annotation_kind="weak",
        classes=["dog", "cat"], weak_labels=dict(_WEAK_LABELS)))
    monkeypatch.setitem(
        adapters.ADAPTERS, "EmbSetMulti", lambda d: DatasetAnnotation(
            label_kind="multilabel", annotation_kind="weak",
            classes=["dog", "cat"], weak_labels=dict(_MULTI_LABELS)))

    caches = {"output_root": str(tmp_path / "emb_out")}
    for kind, dataset_name in (("multiclass", "EmbSet"),
                               ("multilabel", "EmbSetMulti")):
        caches[kind] = prepare_dataset(resolve_raw_config(
            dataset_name, dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "raw" / dataset_name),
            split_json=str(split_json), sr=SR, seg_sec=SEG_SEC,
            hop_sec=SEG_SEC, num_proc=1, batch_size=2))
    return caches


@pytest.fixture()
def strong_caches(tmp_path, monkeypatch):
    """Build a label_type=strong cache and a label_type=weak cache from the same strong synthetic annotations."""
    dataset_dir, split_json = _write_synth_source(tmp_path)
    monkeypatch.setitem(
        adapters.ADAPTERS, "EmbSetStrong", lambda d: DatasetAnnotation(
            label_kind="multilabel", annotation_kind="strong",
            classes=list(_STRONG_CLASSES), strong_events=dict(_STRONG_EVENTS)))

    caches = {"output_root": str(tmp_path / "emb_out")}
    for label_type in ("strong", "weak"):
        caches[label_type] = prepare_dataset(resolve_raw_config(
            "EmbSetStrong", dataset_dir=str(dataset_dir),
            cache_dir=str(tmp_path / "raw" / f"EmbSetStrong-{label_type}"),
            split_json=str(split_json), sr=SR, seg_sec=SEG_SEC,
            hop_sec=SEG_SEC, label_type=label_type, num_proc=1, batch_size=2))
    return caches


@pytest.fixture()
def strong_frame_cache(tmp_path, monkeypatch):
    """Strong cache dedicated to frame boundary cases: see _FRAME_DURATIONS/_FRAME_EVENTS for audio durations and events."""
    dataset_dir = tmp_path / "frame_source"
    dataset_dir.mkdir()
    rng = np.random.default_rng(11)
    for name, dur in _FRAME_DURATIONS.items():
        data = (rng.standard_normal(int(dur * SR)) * 0.1).astype(np.float32)
        sf.write(dataset_dir / name, data, SR, subtype="FLOAT")
    split_json = tmp_path / "frame_split.json"
    split_json.write_text(json.dumps({
        "train": [{"audio_path": "a.wav", "start": 0.0, "end": "inf"},
                  {"audio_path": "d.wav", "start": 0.0, "end": "inf"}],
        "validation": [{"audio_path": "b.wav", "start": 0.0, "end": "inf"},
                       {"audio_path": "e.wav", "start": 0.0, "end": "inf"}],
        "test": [{"audio_path": "c.wav", "start": 0.0, "end": "inf"}],
    }), encoding="utf-8")

    monkeypatch.setitem(
        adapters.ADAPTERS, "EmbSetFrame", lambda d: DatasetAnnotation(
            label_kind="multilabel", annotation_kind="strong",
            classes=list(_STRONG_CLASSES), strong_events=dict(_FRAME_EVENTS)))
    cache = prepare_dataset(resolve_raw_config(
        "EmbSetFrame", dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "raw" / "EmbSetFrame"),
        split_json=str(split_json), sr=SR, seg_sec=SEG_SEC, hop_sec=SEG_SEC,
        label_type="strong", num_proc=1, batch_size=2))
    return {"cache": cache, "output_root": str(tmp_path / "emb_out")}


@pytest.fixture()
def fake_model(monkeypatch):
    """Register a fake model pairing and wrap builder's create_model reference to record call args."""
    monkeypatch.setitem(
        models_registry.MODELS, MODEL_NAME,
        ModelSpec(transform_cls=_FakeTransform, encoder_cls=_FakeEncoder))
    calls = []
    real_create_model = builder.create_model

    def recording_create_model(name, **kwargs):
        """Record the call args, then delegate to the real create_model."""
        calls.append({"name": name, **kwargs})
        return real_create_model(name, **kwargs)

    monkeypatch.setattr(builder, "create_model", recording_create_model)
    return calls


def _resolve(caches, kind="multiclass", **kwargs):
    defaults = dict(output_dir=caches["output_root"])
    defaults.update(kwargs)
    return resolve_config(caches[kind], MODEL_NAME, "clip", **defaults)


def test_multiclass_passthrough_end_to_end(raw_caches, fake_model):
    cfg = _resolve(raw_caches)
    out_dir = prepare_embeddings(cfg)
    emb = load_from_disk(out_dir)
    raw = load_from_disk(raw_caches["multiclass"])

    # Three-level path: {output_root}/{dataset_name}/{model_name with --
    # substitution}/{emb_hash}
    with open(os.path.join(raw_caches["multiclass"], "prep_config.json"),
              encoding="utf-8") as f:
        raw_config_hash = json.load(f)["config_hash"]
    expected_hash = Hasher.hash({"raw_config_hash": raw_config_hash,
                                 "model_name": MODEL_NAME,
                                 "granularity": "clip"})
    assert cfg.emb_hash == expected_hash
    assert out_dir == os.path.join(raw_caches["output_root"], "EmbSet",
                                   "fake--enc", expected_hash)

    # create_model call args: registered name + granularity + weights dir
    assert fake_model == [dict(name=MODEL_NAME, granularity="clip",
                               pretrained_dir=None)]

    # Row order in each split matches the input one-to-one; raw/sr columns
    # are excluded from the output
    assert set(emb) == {"train", "validation", "test"}
    for split in emb:
        assert emb[split].column_names == [
            "audio_path", "audio_id", "segment_id", "start", "end",
            "valid_sec", "embedding", "label"]
        for key in ("audio_path", "audio_id", "segment_id", "start", "end",
                    "valid_sec"):
            assert emb[split][key] == raw[split][key]

    # multiclass passthrough: ClassLabel names and label values match the
    # input
    label_feature = emb["train"].features["label"]
    assert isinstance(label_feature, ClassLabel)
    assert label_feature.names == raw["train"].features["label"].names
    for split in emb:
        assert emb[split]["label"] == raw[split]["label"]

    # embedding: [D] float32 fixed length; values match the fake Encoder's
    # deterministic function
    emb_feature = emb["train"].features["embedding"]
    assert emb_feature.feature.dtype == "float32"
    assert emb_feature.length == EMB_DIM
    for split in emb:
        raw_wave = np.asarray(raw[split]["raw"], dtype=np.float32)
        embeddings = np.asarray(emb[split]["embedding"], dtype=np.float32)
        np.testing.assert_allclose(embeddings[:, 0], raw_wave.sum(axis=1),
                                   rtol=1e-4)
        np.testing.assert_array_equal(
            embeddings[:, 1],
            np.asarray(raw[split]["valid_sec"], dtype=np.float32))
        np.testing.assert_array_equal(embeddings[:, 2:], FILL_VALUE)
    # valid_seconds passthrough path for the tail segment (a.wav seg2, valid
    # 0.5s)
    assert emb["train"][2]["embedding"][1] == 0.5


def test_multilabel_passthrough(raw_caches, fake_model):
    out_dir = prepare_embeddings(_resolve(raw_caches, kind="multilabel"))
    emb = load_from_disk(out_dir)
    raw = load_from_disk(raw_caches["multilabel"])

    assert emb["train"].features["label"].feature.dtype == "float32"
    assert emb["train"].features["label"].length == 2
    for split in emb:
        assert emb[split]["label"] == raw[split]["label"]
    # a.wav -> [dog, cat] -> multi-hot [1, 1]; b.wav -> [cat] -> [1, 0] (cat=0
    # lexicographically)
    assert emb["train"][0]["label"] == [1.0, 1.0]
    assert emb["validation"][0]["label"] == [1.0, 0.0]


def test_passthrough_bitexact_and_default_format(tmp_path, monkeypatch,
                                                 fake_model):
    """Passthrough fields stay bit-exact as float64 + the on-disk cache keeps the default format.

    Taking a batch inside map with the numpy format downcasts float64 to
    float32, but provenance fields pass through unchanged at the Arrow
    layer (neither dropped nor rebuilt), keeping them bit-exact; the test
    durations are deliberately chosen as values float32 cannot represent
    exactly (0.3/0.7/1.1 -> 0.1 tail), so any precision-loss regression is
    necessarily caught by the bit-exact equality assertion. It also
    asserts that the numpy format was not persisted into state.json by
    save_to_disk.
    """
    durations = {"p.wav": 0.3, "q.wav": 0.7, "r.wav": 1.1}
    dataset_dir = tmp_path / "bitexact_source"
    dataset_dir.mkdir()
    rng = np.random.default_rng(3)
    for name, dur in durations.items():
        data = (rng.standard_normal(int(dur * SR)) * 0.1).astype(np.float32)
        sf.write(dataset_dir / name, data, SR, subtype="FLOAT")
    split_json = tmp_path / "bitexact_split.json"
    split_json.write_text(json.dumps({
        "train": [{"audio_path": "p.wav", "start": 0.0, "end": "inf"}],
        "validation": [{"audio_path": "q.wav", "start": 0.0, "end": "inf"}],
        "test": [{"audio_path": "r.wav", "start": 0.0, "end": "inf"}],
    }), encoding="utf-8")
    monkeypatch.setitem(
        adapters.ADAPTERS, "EmbSetExact", lambda d: DatasetAnnotation(
            label_kind="multiclass", annotation_kind="weak",
            classes=["dog", "cat"],
            weak_labels={name: "dog" for name in durations}))
    cache = prepare_dataset(resolve_raw_config(
        "EmbSetExact", dataset_dir=str(dataset_dir),
        cache_dir=str(tmp_path / "raw" / "EmbSetExact"),
        split_json=str(split_json), sr=SR, seg_sec=SEG_SEC, hop_sec=SEG_SEC,
        num_proc=1, batch_size=2))

    out_dir = prepare_embeddings(resolve_config(
        cache, MODEL_NAME, "clip", output_dir=str(tmp_path / "emb_out")))
    emb = load_from_disk(out_dir)
    raw = load_from_disk(cache)

    for split in emb:
        assert emb[split].format["type"] is None
        for key in ("start", "end", "valid_sec"):
            assert emb[split][key] == raw[split][key]
    # Discriminative self-check: the input cache does contain float64 values
    # that float32 cannot represent exactly
    tail_valid_sec = raw["train"][0]["valid_sec"]
    assert tail_valid_sec != float(np.float32(tail_valid_sec))


def test_weak_frame_embedding_with_clip_label(raw_caches, fake_model):
    """Weak cache + frame granularity: frame embeddings + passthrough clip labels (weakly-labeled SED scenario)."""
    out_dir = prepare_embeddings(resolve_config(
        raw_caches["multiclass"], MODEL_NAME, "frame",
        output_dir=raw_caches["output_root"]))
    emb = load_from_disk(out_dir)
    raw = load_from_disk(raw_caches["multiclass"])

    for split in emb:
        features = emb[split].features
        assert emb[split].column_names == [
            "audio_path", "audio_id", "segment_id", "start", "end",
            "valid_sec", "embedding", "label", "geometry", "valid_mask"]
        assert features["embedding"] == Array2D(
            shape=(NUM_FRAMES, EMB_DIM), dtype="float32")
        assert features["geometry"] == Array2D(shape=(NUM_FRAMES, 2),
                                               dtype="float32")
        assert features["valid_mask"] == Sequence(Value("bool"),
                                                  length=NUM_FRAMES)
        # label passes through in its weak clip shape; no frame labels are
        # generated
        assert isinstance(features["label"], ClassLabel)
        assert emb[split]["label"] == raw[split]["label"]

    # Tail segment (a.wav seg2, valid 0.5s): 2 valid frames + 1 zero-padded
    # frame, mask is the sole source of truth
    row = emb["train"][2]
    assert row["valid_mask"] == [True, True, False]
    np.testing.assert_array_equal(
        np.asarray(row["geometry"], dtype=np.float32),
        [[0.0, 0.25], [0.25, 0.5], [0.0, 0.0]])
    np.testing.assert_array_equal(
        np.asarray(row["embedding"], dtype=np.float32)[2], 0.0)


def test_skip_existing_and_overwrite(raw_caches, fake_model, capsys):
    cfg = _resolve(raw_caches)
    out_dir = prepare_embeddings(cfg)
    mtime = os.path.getmtime(out_dir)

    # Re-run with the same args: prints skip message, does not recompute
    # (no model build), artifacts untouched
    assert prepare_embeddings(cfg) == out_dir
    assert "skipping" in capsys.readouterr().out
    assert len(fake_model) == 1
    assert os.path.getmtime(out_dir) == mtime

    # overwrite: delete and rebuild
    prepare_embeddings(_resolve(raw_caches, overwrite=True))
    assert os.path.getmtime(out_dir) != mtime
    assert len(fake_model) == 2
    assert len(load_from_disk(out_dir)["train"]) == 3


def test_incomplete_output_requires_overwrite(raw_caches, fake_model):
    cfg = _resolve(raw_caches)
    os.makedirs(cfg.output_dir)
    partial_file = Path(cfg.output_dir) / "partial"
    partial_file.write_text("partial", encoding="utf-8")

    with pytest.raises(ValueError, match="missing emb_config.json"):
        prepare_embeddings(cfg)
    assert partial_file.is_file()

    overwrite_cfg = _resolve(raw_caches, overwrite=True)
    assert prepare_embeddings(overwrite_cfg) == cfg.output_dir
    assert not partial_file.exists()
    assert (Path(cfg.output_dir) / "emb_config.json").is_file()


def test_s3_output_end_to_end(raw_caches, fake_model, tmp_path,
                              monkeypatch, capsys):
    """S3 output writes the three-level path directly; the map Arrow cache only lands under TMPDIR."""
    class S3MemoryFileSystem(MemoryFileSystem):
        """In-memory S3 filesystem for offline testing."""

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

    # Local baseline: used to compare embedding / label for exact equality
    # against the S3 output read-back
    local_emb = load_from_disk(prepare_embeddings(_resolve(raw_caches)))

    map_cache_files = []
    save_paths = []
    original_map = Dataset.map
    original_save_to_disk = DatasetDict.save_to_disk

    def record_map(self, *args, **kwargs):
        """Record the map intermediate cache path, then call the real implementation."""
        map_cache_files.append(kwargs["cache_file_name"])
        return original_map(self, *args, **kwargs)

    def record_save_to_disk(self, dataset_path, *args, **kwargs):
        """Record the save_to_disk target and storage options, then call the real implementation."""
        save_paths.append((dataset_path, kwargs.get("storage_options")))
        return original_save_to_disk(self, dataset_path, *args, **kwargs)

    monkeypatch.setattr(Dataset, "map", record_map)
    monkeypatch.setattr(DatasetDict, "save_to_disk", record_save_to_disk)

    # Bucket-root path rejection (resolve_config never actually produces this
    # under the three-level structure; construct the config directly to
    # trigger it)
    cfg = resolve_config(raw_caches["multiclass"], MODEL_NAME, "clip",
                         output_dir="s3://test-bucket/emb/")
    with pytest.raises(ValueError, match="directory within the bucket"):
        prepare_embeddings(
            dataclasses.replace(cfg, output_dir="s3://test-bucket"))

    # Output root's trailing slash is normalized + posix three-level join
    assert cfg.output_dir == ("s3://test-bucket/emb/EmbSet/fake--enc/"
                              f"{cfg.emb_hash}")
    out_dir = prepare_embeddings(cfg)
    assert out_dir == cfg.output_dir
    assert not any(map_tmp_root.iterdir())

    options = storage.s3_storage_options()
    assert options["endpoint_url"] == "https://a3s.fi"
    fs, out_path = fsspec.core.url_to_fs(out_dir, **options)
    emb = load_from_disk(out_dir, storage_options=options)
    assert set(emb) == {"train", "validation", "test"}
    for split in emb:
        assert emb[split]["label"] == local_emb[split]["label"]
        np.testing.assert_array_equal(
            np.asarray(emb[split]["embedding"], dtype=np.float32),
            np.asarray(local_emb[split]["embedding"], dtype=np.float32))
    with fs.open(f"{out_path}/label_index.json", "r", encoding="utf-8") as f:
        assert json.load(f) == {"cat": 0, "dog": 1}
    with fs.open(f"{out_path}/emb_config.json", "r", encoding="utf-8") as f:
        assert json.load(f)["emb_hash"] == cfg.emb_hash

    # map's temporary Arrow files only land under the fake TMPDIR; artifacts
    # are written directly to S3 via save_to_disk
    assert len(map_cache_files) == 3
    assert all(os.path.commonpath([str(map_tmp_root), path]) == str(map_tmp_root)
               for path in map_cache_files)
    assert all(not path.startswith("s3://") for path in map_cache_files)
    assert save_paths == [(out_dir, options)]

    # Re-run with the same args: prints skip message, does not recompute (no
    # more map / model build)
    map_call_count = len(map_cache_files)
    model_call_count = len(fake_model)
    assert prepare_embeddings(cfg) == out_dir
    assert "skipping" in capsys.readouterr().out
    assert len(map_cache_files) == map_call_count
    assert len(fake_model) == model_call_count

    # Half-finished directory missing the completion marker: errors without
    # overwrite, cleans up and rebuilds after overwrite
    fs.rm(out_path, recursive=True)
    with fs.open(f"{out_path}/partial", "w") as f:
        f.write("partial")
    with pytest.raises(ValueError, match="missing emb_config.json.*--overwrite"):
        prepare_embeddings(cfg)
    assert fs.isfile(f"{out_path}/partial")
    overwrite_cfg = resolve_config(
        raw_caches["multiclass"], MODEL_NAME, "clip",
        output_dir="s3://test-bucket/emb", overwrite=True)
    assert prepare_embeddings(overwrite_cfg) == out_dir
    assert not fs.exists(f"{out_path}/partial")
    assert fs.isfile(f"{out_path}/emb_config.json")
    assert len(load_from_disk(out_dir, storage_options=options)["train"]) == 3


def test_aux_files_copied_and_snapshot(raw_caches, fake_model):
    cfg = _resolve(raw_caches)
    out_dir = prepare_embeddings(cfg)

    # label_index.json is copied verbatim
    with open(os.path.join(raw_caches["multiclass"], "label_index.json"),
              encoding="utf-8") as f:
        raw_label_index = json.load(f)
    with open(os.path.join(out_dir, "label_index.json"),
              encoding="utf-8") as f:
        assert json.load(f) == raw_label_index == {"cat": 0, "dog": 1}

    # emb_config.json: emb_hash + a full snapshot of raw prep_config
    with open(os.path.join(raw_caches["multiclass"], "prep_config.json"),
              encoding="utf-8") as f:
        raw_prep_config = json.load(f)
    with open(os.path.join(out_dir, "emb_config.json"),
              encoding="utf-8") as f:
        emb_config = json.load(f)
    assert emb_config["emb_hash"] == cfg.emb_hash
    assert emb_config["raw_config_hash"] == raw_prep_config["config_hash"]
    assert emb_config["raw_prep_config"] == raw_prep_config
    assert emb_config["model_name"] == MODEL_NAME
    assert emb_config["granularity"] == "clip"
    assert emb_config["output_dir"] == out_dir


def test_strong_clip_tri_state_aggregation(strong_caches, fake_model):
    out_dir = prepare_embeddings(resolve_config(
        strong_caches["strong"], MODEL_NAME, "clip",
        output_dir=strong_caches["output_root"]))
    emb = load_from_disk(out_dir)

    # label: [C] float32 fixed-length tri-state multi-hot (C=3, bird=0 cat=1
    # dog=2 lexicographically)
    label_feature = emb["train"].features["label"]
    assert label_feature.feature.dtype == "float32"
    assert label_feature.length == len(_STRONG_CLASSES)

    # a.wav seg0 [0,1): dog pure POS -> 1, cat pure UNK -> NaN, bird no event
    # -> 0
    train = np.asarray(emb["train"]["label"], dtype=np.float32)
    assert train[0][0] == 0.0
    assert np.isnan(train[0][1])
    assert train[0][2] == 1.0
    # a.wav seg1 [1,2): cat POS[1.2,1.5] + UNK[1.3,1.8] same-class conflict
    # -> 1
    np.testing.assert_array_equal(train[1], [0.0, 1.0, 0.0])
    # a.wav seg2 [2,2.5): empty event slice -> all 0
    np.testing.assert_array_equal(train[2], [0.0, 0.0, 0.0])

    # b.wav seg0: bird pure UNK -> NaN, others no event -> 0
    validation = np.asarray(emb["validation"]["label"], dtype=np.float32)
    assert np.isnan(validation[0][0])
    np.testing.assert_array_equal(validation[0][1:], [0.0, 0.0])

    # c.wav: dog[0.5,2.2] is POS across all three segments; seg2 [2,3) also
    # has cat UNK[2.5,2.9] -> NaN
    test = np.asarray(emb["test"]["label"], dtype=np.float32)
    np.testing.assert_array_equal(test[0], [0.0, 0.0, 1.0])
    np.testing.assert_array_equal(test[1], [0.0, 0.0, 1.0])
    assert test[2][0] == 0.0
    assert np.isnan(test[2][1])
    assert test[2][2] == 1.0


def test_strong_clip_matches_strong_to_weak_passthrough(strong_caches,
                                                        fake_model):
    # Same strong annotations: strong cache + in-path aggregation vs.
    # strong-to-weak cache + passthrough; labels for matching slices are
    # elementwise equal (assert_array_equal requires NaN positions to match
    # too)
    root = strong_caches["output_root"]
    emb_strong = load_from_disk(prepare_embeddings(resolve_config(
        strong_caches["strong"], MODEL_NAME, "clip", output_dir=root)))
    emb_weak = load_from_disk(prepare_embeddings(resolve_config(
        strong_caches["weak"], MODEL_NAME, "clip", output_dir=root)))

    assert set(emb_strong) == set(emb_weak) == {"train", "validation", "test"}
    for split in emb_strong:
        np.testing.assert_array_equal(
            np.asarray(emb_strong[split]["label"], dtype=np.float32),
            np.asarray(emb_weak[split]["label"], dtype=np.float32))


def _prepare_frame(strong_frame_cache):
    """Run frame-granularity extraction through the main seam; return the output directory."""
    return prepare_embeddings(resolve_config(
        strong_frame_cache["cache"], MODEL_NAME, "frame",
        output_dir=strong_frame_cache["output_root"]))


def _assert_tri_state(actual, expected):
    """Tri-state matrix assertion: NaN positions are checked via isnan, other positions elementwise equal."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    nan_mask = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan_mask)
    np.testing.assert_array_equal(actual[~nan_mask], expected[~nan_mask])


def test_strong_frame_features_and_factory(strong_frame_cache, fake_model):
    emb = load_from_disk(_prepare_frame(strong_frame_cache))

    assert fake_model == [dict(name=MODEL_NAME, granularity="frame",
                               pretrained_dir=None)]
    # Fixed-shape schema: the probed frame count NUM_FRAMES is constant
    # across the whole dataset
    for split in emb:
        features = emb[split].features
        assert features["embedding"] == Array2D(
            shape=(NUM_FRAMES, EMB_DIM), dtype="float32")
        assert features["label"] == Array2D(
            shape=(NUM_FRAMES, len(_STRONG_CLASSES)), dtype="float32")
        assert features["geometry"] == Array2D(shape=(NUM_FRAMES, 2),
                                               dtype="float32")
        assert features["valid_mask"] == Sequence(Value("bool"),
                                                  length=NUM_FRAMES)
        assert emb[split].column_names == [
            "audio_path", "audio_id", "segment_id", "start", "end",
            "valid_sec", "embedding", "label", "geometry", "valid_mask"]
        for row in emb[split]:
            assert np.asarray(row["embedding"]).shape == (NUM_FRAMES, EMB_DIM)
            assert np.asarray(row["label"]).shape == (NUM_FRAMES,
                                                      len(_STRONG_CLASSES))
            assert np.asarray(row["geometry"]).shape == (NUM_FRAMES, 2)
            assert len(row["valid_mask"]) == NUM_FRAMES


def test_strong_frame_label_boundary_cases(strong_frame_cache, fake_model):
    emb = load_from_disk(_prepare_frame(strong_frame_cache))

    # a.wav (classes in lexicographic order bird=0 cat=1 dog=2):
    #   dog[0.2,0.3) crosses the frame 0/1 boundary -> both frames' dog = 1;
    #   cat POS[0.25,0.4) start == frame 0 slot's end (zero-length
    #   intersection) -> frame 0 not set, only frame 1;
    #   cat UNK[0.1,0.25) end == frame 1 slot's start -> only frame 0 NaN;
    #   bird[0.8,0.95) falls in the last frame's absorption region (nominal
    #   end 0.75, valid 1.0) -> only frame 2
    _assert_tri_state(emb["train"][0]["label"],
                      [[0.0, np.nan, 1.0],
                       [0.0, 1.0, 1.0],
                       [1.0, 0.0, 0.0]])
    # d.wav no events -> all valid frames are 0
    _assert_tri_state(emb["train"][1]["label"], np.zeros((NUM_FRAMES, 3)))
    # c.wav: frame 0 same-class (dog) POS+UNK -> 1; frame 1 different-class
    # POS (bird) and UNK (cat) coexist -> 1 and NaN respectively
    _assert_tri_state(emb["test"][0]["label"],
                      [[0.0, 0.0, 1.0],
                       [1.0, np.nan, 0.0],
                       [0.0, 0.0, 0.0]])


def test_strong_frame_padding_sync_and_geometry(strong_frame_cache,
                                                fake_model):
    """Zero-padding + valid_mask contract: invalid frames are zero-padded in all three places, mask is the sole source of truth.

    The validation split (b's 0.5s + e's 0.2s in the same batch) has a
    within-batch maximum valid frame count of 2; the fake Encoder only
    outputs 2 frames, and builder uniformly pads to the probed frame count
    of 3 -- this also exercises the batch-dependent-T padding path.
    """
    emb = load_from_disk(_prepare_frame(strong_frame_cache))

    # b.wav's 0.5s short segment: 2 valid frames + 1 padding frame
    row = emb["validation"][0]
    embedding = np.asarray(row["embedding"], dtype=np.float32)
    label = np.asarray(row["label"], dtype=np.float32)
    geometry = np.asarray(row["geometry"], dtype=np.float32)
    assert row["valid_mask"] == [True, True, False]
    # Invalid frames' embedding / label / geometry are all zero-padded in
    # sync
    np.testing.assert_array_equal(embedding[2], 0.0)
    np.testing.assert_array_equal(label[2], 0.0)
    np.testing.assert_array_equal(geometry[2], [0.0, 0.0])
    # No NaN leakage in valid frames (aside from the label's tri-state NaN);
    # the last valid frame's endpoint is absorbed to 0.5
    assert not np.isnan(embedding[:2]).any()
    assert not np.isnan(geometry[:2]).any()
    np.testing.assert_array_equal(geometry[:2], [[0.0, 0.25], [0.25, 0.5]])
    _assert_tri_state(label, [[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]])

    # e.wav's 0.2s extremely short segment: 1 valid frame + 2 padding frames
    row = emb["validation"][1]
    embedding = np.asarray(row["embedding"], dtype=np.float32)
    label = np.asarray(row["label"], dtype=np.float32)
    geometry = np.asarray(row["geometry"], dtype=np.float32)
    assert row["valid_mask"] == [True, False, False]
    np.testing.assert_array_equal(embedding[1:], 0.0)
    np.testing.assert_array_equal(label[1:], 0.0)
    np.testing.assert_array_equal(geometry[1:], 0.0)
    # The sole valid frame: endpoint absorbed to valid_seconds=0.2, no
    # events -> all 0
    np.testing.assert_array_equal(geometry[0],
                                  np.asarray([0.0, 0.2], dtype=np.float32))
    _assert_tri_state(label[0], [0.0, 0.0, 0.0])

    # Full-length segments: the geometry written to disk is the encoder's raw
    # output: nominal grid points + last frame's endpoint absorption
    # (nominal [0.5, 0.75) -> [0.5, 1.0))
    train_geometry = np.asarray(emb["train"]["geometry"], dtype=np.float32)
    np.testing.assert_array_equal(
        train_geometry,
        np.broadcast_to([[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]],
                        (2, NUM_FRAMES, 2)))


def test_strong_frame_embedding_values(strong_frame_cache, fake_model):
    emb = load_from_disk(_prepare_frame(strong_frame_cache))

    # Default batch_size=32 -> a single batch per split, embedding[b, t, 0]
    # = b*10 + t
    train = np.asarray(emb["train"]["embedding"], dtype=np.float32)
    np.testing.assert_array_equal(train[..., 0],
                                  [[0.0, 1.0, 2.0], [10.0, 11.0, 12.0]])
    np.testing.assert_array_equal(train[..., 1:], FILL_VALUE)
    # validation: row b has 2 valid frames, row e has 1 valid frame; invalid
    # and padding frames are all 0
    validation = np.asarray(emb["validation"]["embedding"], dtype=np.float32)
    np.testing.assert_array_equal(validation[..., 0],
                                  [[0.0, 1.0, 0.0], [10.0, 0.0, 0.0]])


def test_low_sr_hint_printed(raw_caches, monkeypatch, capsys):
    # Fake Transform's native sample rate is higher than the cache's sr ->
    # prints a hint line but does not block
    monkeypatch.setitem(
        models_registry.MODELS, MODEL_NAME,
        ModelSpec(transform_cls=_FakeHighSrTransform,
                  encoder_cls=_FakeEncoder))
    out_dir = prepare_embeddings(_resolve(raw_caches))
    assert "below the model's native sample rate" in capsys.readouterr().out
    assert len(load_from_disk(out_dir)["train"]) == 3
