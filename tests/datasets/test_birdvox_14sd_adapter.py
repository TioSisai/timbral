"""Adapter tests for BirdVox-14SD annotations and HDF5 audio reading."""

import json

import h5py
import numpy as np
import pytest
from datasets import load_from_disk

from timbral.datasets import audio_io, builder
from timbral.datasets.adapters import birdvox_14sd
from timbral.datasets.config import resolve_config


@pytest.fixture()
def birdvox_dir(tmp_path):
    """Build a minimal BirdVox HDF5 directory covering all 14 target classes."""
    expected_count = 0
    for index, taxonomy_code in enumerate(
            birdvox_14sd.TAXONOMY_CODE_TO_NAME):
        filename = (
            f"BirdVox-14SD_{taxonomy_code.replace('.', '-')}_original.h5"
        )
        with h5py.File(tmp_path / filename, "w") as h5_file:
            h5_file.create_dataset("sample_rate", data=100)
            waveforms = h5_file.create_group("waveforms")
            data = (np.linspace(-0.5, 0.5, 100, dtype=np.float32)
                    if index == 0 else np.array([0.0, 0.1], dtype=np.float32))
            waveforms.create_dataset(f"clip-{index}", data=data)
            expected_count += 1
            if index == 0:
                waveforms.create_dataset("second-clip", data=[0.2])
                expected_count += 1
    return tmp_path, expected_count


def test_load_annotation_reads_target_waveform_keys(birdvox_dir):
    """The waveform keys of the 14 target HDF5 files should map to the corresponding readable species names."""
    dataset_dir, expected_count = birdvox_dir

    annotation = birdvox_14sd.load_annotation(str(dataset_dir))

    assert annotation.label_kind == "multiclass"
    assert annotation.annotation_kind == "weak"
    assert len(annotation.classes) == 14
    assert len(annotation.weak_labels) == expected_count
    assert annotation.weak_labels[
        "BirdVox-14SD_1-1-1_original.h5::waveforms/clip-0"
    ] == "American tree sparrow"
    assert annotation.weak_labels[
        "BirdVox-14SD_1-4-7_original.h5::waveforms/clip-13"
    ] == "Ovenbird"


def test_audio_io_reads_birdvox_virtual_path(birdvox_dir):
    """audio_io should dispatch BirdVox HDF5 virtual paths directly, without needing adapter side effects."""
    dataset_dir, _ = birdvox_dir
    path = (
        dataset_dir / "BirdVox-14SD_1-1-1_original.h5"
    ).as_posix() + "::waveforms/clip-0"

    assert audio_io.probe_duration(path) == pytest.approx(1.0)
    segment = audio_io.load_segment(
        path, offset_sec=0.25, duration_sec=0.5,
        sr=50, mono=True, seg_len=40)
    assert segment.shape == (40,)
    assert segment.dtype == np.float32
    assert np.abs(segment[:25]).sum() > 0
    np.testing.assert_array_equal(segment[25:], 0.0)


def test_birdvox_virtual_paths_work_in_multiprocess_builder(birdvox_dir):
    """HDF5 virtual paths should complete segment reading within a multiprocess map."""
    dataset_dir, _ = birdvox_dir

    def entry(taxonomy_code, clip_index):
        filename = (
            f"BirdVox-14SD_{taxonomy_code.replace('.', '-')}_original.h5"
        )
        return {
            "audio_path": f"{filename}::waveforms/clip-{clip_index}",
            "start": 0.0,
            "end": "inf",
        }

    split_json = dataset_dir / "split.json"
    split_json.write_text(json.dumps({
        "train": [entry("1.1.1", 0), entry("1.1.2", 1)],
        "validation": [entry("1.1.3", 2)],
        "test": [entry("1.1.4", 3)],
    }), encoding="utf-8")
    config = resolve_config(
        "BirdVox-14SD",
        dataset_dir=str(dataset_dir),
        split_json=str(split_json),
        cache_dir=str(dataset_dir / "cache"),
        sr=50,
        seg_sec=0.5,
        hop_sec=0.5,
        num_proc=2,
        batch_size=1,
    )

    dataset = load_from_disk(builder.prepare_dataset(config))

    assert {split: len(rows) for split, rows in dataset.items()} == {
        "train": 3,
        "validation": 1,
        "test": 1,
    }
    assert [row["segment_id"] for row in dataset["train"]] == [0, 1, 0]
    assert all(len(row["raw"]) == 25 for row in dataset["train"])
