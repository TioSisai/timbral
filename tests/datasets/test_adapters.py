"""Unit tests for the timbral.datasets.adapters registry and pure logic (correctness of real-data adapters is covered by end-to-end validation)."""

import pandas as pd
import pytest

from timbral.datasets import adapters
from timbral.datasets.adapters import audioset_strong, audioset_weak, bsd, sonyc_ust
from timbral.datasets.adapters.base import DatasetAnnotation


def test_builtin_adapters_registered():
    assert {"ESC-50", "FSD50K", "DataSED", "UrbanSound8K", "FSDnoisy18k",
            "BSD35K", "BSD10K", "SONYC-UST", "DESED", "AudioSetStrong",
            "AudioSetWeak", "BirdVox-14SD", "DB3V", "DCASE-2024-Task-5",
            "HyenaSET", "RealDESED"} <= set(adapters.ADAPTERS)


def test_unknown_dataset_raises_with_registry_listing():
    with pytest.raises(KeyError, match="registered adapters"):
        adapters.load_annotation("NoSuchSet", "/nowhere")


def test_register_adapter_dispatches(monkeypatch):
    monkeypatch.setitem(adapters.ADAPTERS, "TestSet", lambda dataset_dir: DatasetAnnotation(
        label_kind="multiclass", annotation_kind="weak",
        classes=["a"], weak_labels={"x.wav": "a"}))
    annotation = adapters.load_annotation("TestSet", "/nowhere")
    assert annotation.weak_labels == {"x.wav": "a"}


def test_bsd_code_to_name_unique():
    # The README's secondary readable names have duplicates ("Other"x5); the
    # restricted format must guarantee uniqueness across the whole table.
    assert len(set(bsd._CODE_TO_NAME.values())) == len(bsd._CODE_TO_NAME)


def test_sonyc_aggregate_presence_verified_overrides_crowd():
    cols = ["1_engine_presence", "8_dog_presence"]
    df = pd.DataFrame({
        "audio_filename": ["a.wav", "a.wav", "b.wav", "b.wav", "c.wav"],
        "annotator_id":   [1,       0,       1,       2,       3],
        # a: crowd label engine=1, but verified (annotator 0) says
        #    0/dog=1 -> verified takes precedence
        # b: no verified label, any-vote (-1 counts as 0) -> engine=1, dog=0
        # c: single annotator, all 0 -> empty label
        cols[0]: [1, 0, 1, -1, 0],
        cols[1]: [0, 1, 0, -1, 0],
    })
    agg = sonyc_ust.aggregate_presence(df, cols)
    assert agg.loc["a.wav"].tolist() == [0, 1]
    assert agg.loc["b.wav"].tolist() == [1, 0]
    assert agg.loc["c.wav"].tolist() == [0, 0]


def _write_audioset_strong_metadata(root):
    """Write synthetic AudioSetStrong official_metadata (vocabulary + train/eval event tsv).

    Args:
        root: Root directory of the synthetic dataset.
    """
    meta = root / "official_metadata"
    meta.mkdir()
    (meta / "mid_to_display_name.tsv").write_text(
        "/m/01\tSpeech\n/m/02\tDog\n", encoding="utf-8")
    header = "segment_id\tstart_time_seconds\tend_time_seconds\tlabel\n"
    (meta / "audioset_train_strong.tsv").write_text(
        header + "seg1\t0.0\t1.5\t/m/01\nseg1\t2.0\t3.0\t/m/02\n",
        encoding="utf-8")
    (meta / "audioset_eval_strong.tsv").write_text(
        header + "seg2\t0.5\t1.0\t/m/02\n", encoding="utf-8")


def test_audioset_strong_events_are_four_tuples(tmp_path):
    """A normal event yields a (class_name, start, end, 1.0) four-tuple."""
    _write_audioset_strong_metadata(tmp_path)
    annotation = audioset_strong.load_annotation(str(tmp_path))

    assert annotation.label_kind == "multilabel"
    assert annotation.annotation_kind == "strong"
    assert annotation.classes == ["Dog", "Speech"]
    assert annotation.strong_events["audio/train/seg1.wav"] == [
        ("Speech", 0.0, 1.5, 1.0), ("Dog", 2.0, 3.0, 1.0)]
    assert annotation.strong_events["audio/eval/seg2.wav"] == [
        ("Dog", 0.5, 1.0, 1.0)]


def test_audioset_strong_oov_mid_raises(tmp_path):
    """An out-of-vocabulary mid fails fast via direct-indexing KeyError, with no silent fallback."""
    _write_audioset_strong_metadata(tmp_path)
    (tmp_path / "official_metadata" / "audioset_train_strong.tsv").write_text(
        "segment_id\tstart_time_seconds\tend_time_seconds\tlabel\n"
        "seg1\t0.0\t1.0\t/m/oov\n", encoding="utf-8")

    with pytest.raises(KeyError, match="/m/oov"):
        audioset_strong.load_annotation(str(tmp_path))


def test_audioset_weak_ytid_keyed_labels():
    labels = audioset_weak._YtidKeyedLabels({"--PJHxphWEs": ["Speech"]})
    assert "balanced_train_segments/--PJHxphWEs.wav" in labels
    assert "unbalanced_train_segments_part03_partial/--PJHxphWEs.wav" in labels
    assert labels["eval_segments/--PJHxphWEs.wav"] == ["Speech"]
    assert "balanced_train_segments/other.wav" not in labels
