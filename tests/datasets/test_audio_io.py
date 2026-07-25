"""Unit tests for timbral.datasets.audio_io: segment reading/padding/multi-channel/HDF5 virtual paths."""

import h5py
import numpy as np
import pytest
import soundfile as sf

from timbral.datasets import audio_io

SR = 16000


@pytest.fixture()
def mono_wav(tmp_path):
    """A 3-second 16kHz mono wav; the waveform is random noise (allows sample-by-sample comparison)."""
    rng = np.random.default_rng(0)
    data = (rng.standard_normal(3 * SR) * 0.1).astype(np.float32)
    path = tmp_path / "mono.wav"
    sf.write(path, data, SR, subtype="FLOAT")
    return str(path), data


@pytest.fixture()
def stereo_wav(tmp_path):
    """A 2-second 16kHz stereo wav."""
    rng = np.random.default_rng(1)
    data = (rng.standard_normal((2 * SR, 2)) * 0.1).astype(np.float32)
    path = tmp_path / "stereo.wav"
    sf.write(path, data, SR, subtype="FLOAT")
    return str(path), data


def test_probe_duration(mono_wav):
    path, _ = mono_wav
    assert audio_io.probe_duration(path) == pytest.approx(3.0)


def test_load_segment_offset_exact(mono_wav):
    # target sr == native sr, no resampling, should match sample-by-sample
    path, data = mono_wav
    seg = audio_io.load_segment(path, offset_sec=1.0,
                                duration_sec=1.0, sr=SR, mono=True, seg_len=SR)
    assert seg.shape == (SR,) and seg.dtype == np.float32
    np.testing.assert_allclose(seg, data[SR:2 * SR], atol=1e-6)


def test_load_segment_tail_zero_padded(mono_wav):
    path, data = mono_wav
    seg = audio_io.load_segment(path, offset_sec=2.0,
                                duration_sec=1.0, sr=SR, mono=True, seg_len=2 * SR)
    assert seg.shape == (2 * SR,)
    np.testing.assert_allclose(seg[:SR], data[2 * SR:], atol=1e-6)
    np.testing.assert_array_equal(seg[SR:], np.zeros(SR, dtype=np.float32))


def test_load_segment_truncates_to_seg_len(mono_wav):
    path, data = mono_wav
    seg = audio_io.load_segment(path, offset_sec=0.0,
                                duration_sec=3.0, sr=SR, mono=True, seg_len=SR)
    assert seg.shape == (SR,)
    np.testing.assert_allclose(seg, data[:SR], atol=1e-6)


def test_load_segment_stereo_shape(stereo_wav):
    path, data = stereo_wav
    seg = audio_io.load_segment(path, offset_sec=0.0,
                                duration_sec=2.0, sr=SR, mono=False, seg_len=3 * SR)
    assert seg.shape == (2, 3 * SR)
    np.testing.assert_allclose(seg[:, :2 * SR], data.T, atol=1e-6)
    np.testing.assert_array_equal(seg[:, 2 * SR:], 0.0)


def test_load_segment_resamples(mono_wav):
    path, _ = mono_wav
    seg = audio_io.load_segment(path, offset_sec=0.0,
                                duration_sec=3.0, sr=8000, mono=True, seg_len=24000)
    assert seg.shape == (24000,)
    assert np.abs(seg).sum() > 0


def test_h5_virtual_path_roundtrip(tmp_path):
    # Generic HDF5 virtual path: top-level sample_rate + a 1-D waveform dataset
    data = np.linspace(-0.5, 0.5, 200, dtype=np.float32)
    h5_path = tmp_path / "container.h5"
    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("sample_rate", data=100)
        h5_file.create_dataset("waveforms/k1", data=data)
    path = f"{h5_path.as_posix()}::waveforms/k1"

    assert audio_io.probe_duration(path) == pytest.approx(2.0)
    seg = audio_io.load_segment(path, offset_sec=0.5, duration_sec=1.0,
                                sr=100, mono=True, seg_len=150)
    assert seg.shape == (150,) and seg.dtype == np.float32
    np.testing.assert_allclose(seg[:100], data[50:150], atol=1e-6)
    np.testing.assert_array_equal(seg[100:], 0.0)
