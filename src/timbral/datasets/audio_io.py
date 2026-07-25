"""Audio segment loading: librosa reads by offset/duration + fixed-length zero-padding.

Two path forms are supported: regular audio file paths, and virtual paths
for waveforms embedded in an HDF5 container, ``{h5_path}::{dataset_path}``
(the container must contain a top-level ``sample_rate`` dataset, with the
waveform stored as a 1-D array, e.g. BirdVox-14SD); dispatch is decoupled
from the concrete dataset name -- the adapter is only responsible for
producing the virtual path.

The HDF5 branch only performs dtype conversion; waveforms inside the
container must already be floating-point amplitude (integer PCM is not
normalized via ``/32768``); segment boundaries are rounded with ``round``,
which differs from librosa's truncation convention of ``int(offset*sr)``
plus ``int(duration*sr)`` -- the same audio segment can therefore differ
by 1 frame between the two branches.
"""

import h5py
import librosa
import numpy as np
import soundfile as sf


def _split_container_path(path: str):
    """Parse an HDF5 virtual path by ``::``; returns ``None`` for a regular path."""
    h5_path, separator, dataset_path = path.partition("::")
    return (h5_path, dataset_path) if separator else None


def _fit_length(y: np.ndarray, seg_len: int, mono: bool) -> np.ndarray:
    """Zero-pad or truncate to the fixed length seg_len."""
    y = np.atleast_2d(y)
    n = y.shape[1]
    if n > seg_len:
        y = y[:, :seg_len]
    elif n < seg_len:
        y = np.pad(y, ((0, 0), (0, seg_len - n)))
    return y[0] if mono else y


def probe_duration(path: str) -> float:
    """Probe the total audio duration (seconds); reads only the file header / dataset metadata without decoding the payload."""
    container = _split_container_path(path)
    if container is None:
        info = sf.info(path)
        return info.frames / info.samplerate
    h5_path, dataset_path = container
    with h5py.File(h5_path, "r") as h5_file:
        return len(h5_file[dataset_path]) / float(h5_file["sample_rate"][()])


def load_segment(path: str, offset_sec: float, duration_sec: float,
                 sr: int, mono: bool, seg_len: int) -> np.ndarray:
    """Load the [offset, offset+duration) segment and resample, zero-padding/truncating to seg_len.

    Args:
        path: Audio file path or HDF5 virtual path.
        offset_sec: Segment start time within the original audio, in seconds.
        duration_sec: Segment valid length, in seconds.
        sr: Target sample rate.
        mono: Whether to merge to mono.
        seg_len: Fixed number of output samples (= round(seg_sec * sr)).

    Returns:
        np.ndarray: float32, shape (seg_len,) when mono=True, otherwise (ch, seg_len).
    """
    container = _split_container_path(path)
    if container is None:
        y, _ = librosa.load(path, sr=sr, mono=mono, offset=offset_sec,
                            duration=duration_sec, dtype=np.float32)
        return _fit_length(y, seg_len, mono)

    h5_path, dataset_path = container
    with h5py.File(h5_path, "r") as h5_file:
        source_sr = int(h5_file["sample_rate"][()])
        start_frame = round(offset_sec * source_sr)
        end_frame = start_frame + round(duration_sec * source_sr)
        waveform = np.asarray(h5_file[dataset_path][start_frame:end_frame],
                              dtype=np.float32)
    if source_sr != sr:
        waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=sr)
    return _fit_length(waveform, seg_len, mono)
