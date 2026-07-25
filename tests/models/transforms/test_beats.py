"""Unit tests for ``BeatsKaldiFbankTransform`` without weights."""

from __future__ import annotations

import pytest
import torch
import torchaudio.compliance.kaldi as ta_kaldi
from torch.nn import functional as F

from timbral.models.transforms import BeatsKaldiFbankTransform

_TARGET_SR = 16000
_FBANK_MEAN = 15.41663
_FBANK_STD = 6.55582
_MIN_TARGET_SAMPLES = 2800


@pytest.fixture(scope="module")
def transform() -> BeatsKaldiFbankTransform:
    return BeatsKaldiFbankTransform().eval()


def _reference_fbank(waveform: torch.Tensor) -> torch.Tensor:
    """Official reference: per-sample ta_kaldi.fbank + BEATs normalization."""
    fbanks = []
    for row in waveform:
        fbank = ta_kaldi.fbank(
            row.unsqueeze(0) * 2**15,
            num_mel_bins=128,
            sample_frequency=16000,
            frame_length=25,
            frame_shift=10,
        )
        fbanks.append(fbank)
    stacked = torch.stack(fbanks, dim=0)
    return (stacked - _FBANK_MEAN) / (2 * _FBANK_STD)


def _make_signal(kind: str, num_samples: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    if kind == "random":
        return torch.randn((num_samples,), generator=generator) * 0.5
    if kind == "sine":
        time_axis = torch.arange(num_samples, dtype=torch.float32)
        return torch.sin(
            2 * torch.pi * 997.0 * time_axis / _TARGET_SR
        )
    if kind == "impulse":
        signal = torch.zeros(num_samples)
        signal[min(100, num_samples - 1)] = 1.0
        return signal
    raise ValueError(kind)


def test_invalid_inputs_raise(transform):
    with pytest.raises(TypeError, match="floating-point"):
        transform(
            torch.zeros((1, 16000), dtype=torch.int16),
            sample_rate=16000,
        )
    with pytest.raises(ValueError, match="shape"):
        transform(torch.zeros(16000), sample_rate=16000)
    with pytest.raises(TypeError, match="Python int"):
        transform(torch.zeros((1, 16000)), sample_rate=16000.0)
    with pytest.raises(ValueError, match="positive"):
        transform(torch.zeros((1, 16000)), sample_rate=0)
    with pytest.raises(TypeError, match="Tensor or None"):
        transform(
            torch.zeros((1, 16000)),
            sample_rate=16000,
            valid_seconds=[1.0],
        )
    with pytest.raises(ValueError, match="shape \\[B\\]"):
        transform(
            torch.zeros((2, 16000)),
            sample_rate=16000,
            valid_seconds=torch.tensor([1.0]),
        )
    with pytest.raises(ValueError, match="exceeds the physical"):
        transform(
            torch.zeros((1, 16000)),
            sample_rate=16000,
            valid_seconds=torch.tensor([2.0]),
        )
    with pytest.raises(ValueError, match="at least 1 sample"):
        transform(
            torch.zeros((1, 16000)),
            sample_rate=16000,
            valid_seconds=torch.tensor([1e-5]),
        )
    with pytest.raises(TypeError):
        transform(
            torch.zeros((1, 16000)),
            sample_rate=16000,
            unknown_argument=1,
        )


def test_output_contract(transform):
    output = transform(torch.randn(2, 16000), sample_rate=16000)

    assert set(output) == {
        "input_features",
        "valid_feature_frames",
        "valid_seconds",
    }
    assert output["input_features"].shape == (2, 98, 128)
    assert output["input_features"].dtype == torch.float32
    assert output["valid_feature_frames"].dtype == torch.int64
    assert torch.equal(
        output["valid_feature_frames"],
        torch.tensor([98, 98]),
    )
    assert output["valid_seconds"].dtype == torch.float32
    assert torch.allclose(
        output["valid_seconds"],
        torch.tensor([1.0, 1.0]),
    )
    assert transform.target_sample_rate == 16000


@pytest.mark.parametrize(
    ("num_samples", "expected_frames"),
    [(320, 16), (2799, 16), (2800, 16), (2960, 17), (16000, 98)],
)
def test_valid_feature_frames_formula(
    transform,
    num_samples,
    expected_frames,
):
    output = transform(
        torch.randn(1, num_samples),
        sample_rate=16000,
    )
    assert int(output["valid_feature_frames"][0]) == expected_frames
    assert output["input_features"].shape[1] == expected_frames


@pytest.mark.parametrize("kind", ("random", "sine", "impulse"))
@pytest.mark.parametrize("seconds", (0.175, 1.0, 4.03))
def test_fbank_matches_official_reference(transform, kind, seconds):
    num_samples = round(seconds * _TARGET_SR)
    waveform = _make_signal(kind, num_samples, seed=num_samples).unsqueeze(0)

    output = transform(waveform, sample_rate=16000)
    reference = _reference_fbank(waveform)

    torch.testing.assert_close(
        output["input_features"],
        reference,
        atol=1e-5,
        rtol=1e-5,
    )


def test_short_input_zero_padding_matches_official(transform):
    waveform = _make_signal("random", 320, seed=7).unsqueeze(0)

    output = transform(waveform, sample_rate=16000)
    padded = F.pad(waveform, (0, _MIN_TARGET_SAMPLES - 320))
    reference = _reference_fbank(padded)

    assert int(output["valid_feature_frames"][0]) == 16
    torch.testing.assert_close(
        output["input_features"],
        reference,
        atol=1e-5,
        rtol=1e-5,
    )
    assert torch.allclose(
        output["valid_seconds"],
        torch.tensor([0.02]),
    )


def test_channel_downmix_and_resample(transform):
    generator = torch.Generator().manual_seed(11)
    stereo = torch.randn((1, 2, 32000), generator=generator)

    output_stereo = transform(stereo, sample_rate=32000)
    output_mono = transform(stereo.mean(dim=1), sample_rate=32000)

    torch.testing.assert_close(
        output_stereo["input_features"],
        output_mono["input_features"],
    )
    assert output_stereo["input_features"].shape == (1, 98, 128)


def test_nonzero_invalid_tail_does_not_change_output(transform):
    generator = torch.Generator().manual_seed(13)
    clean = torch.zeros(2, 32000)
    clean[:, :16000] = torch.randn((2, 16000), generator=generator)
    dirty = clean.clone()
    dirty[:, 16000:] = 9.0
    valid_seconds = torch.tensor([1.0, 1.0])

    output_clean = transform(
        clean,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )
    output_dirty = transform(
        dirty,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    torch.testing.assert_close(
        output_clean["input_features"],
        output_dirty["input_features"],
    )


def test_mixed_batch_matches_single_calls(transform):
    generator = torch.Generator().manual_seed(17)
    valid_seconds = torch.tensor([0.02, 1.0, 4.03])
    max_samples = round(4.03 * _TARGET_SR)
    batch = torch.full((3, max_samples), 5.0)
    rows = []
    for index, seconds in enumerate(valid_seconds.tolist()):
        num_samples = round(seconds * _TARGET_SR)
        row = torch.randn((num_samples,), generator=generator)
        rows.append(row)
        batch[index, :num_samples] = row

    output_batch = transform(
        batch,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    for index, row in enumerate(rows):
        output_single = transform(row.unsqueeze(0), sample_rate=16000)
        frames = int(output_single["valid_feature_frames"][0])
        assert (
            int(output_batch["valid_feature_frames"][index]) == frames
        )
        torch.testing.assert_close(
            output_batch["input_features"][index, :frames],
            output_single["input_features"][0],
        )
        # Padding outside each group is exactly 0
        assert torch.all(
            output_batch["input_features"][index, frames:] == 0
        )


def test_buffers_match_torchaudio_reference(transform):
    expected_window = torch.hann_window(400, periodic=False).pow(0.85)
    torch.testing.assert_close(transform.window, expected_window)

    mel_weight, _ = ta_kaldi.get_mel_banks(
        num_bins=128,
        window_length_padded=512,
        sample_freq=16000.0,
        low_freq=20.0,
        high_freq=0.0,
        vtln_low=100.0,
        vtln_high=-500.0,
        vtln_warp_factor=1.0,
    )
    torch.testing.assert_close(
        transform.mel_weight,
        F.pad(mel_weight, (0, 1), value=0.0),
    )
    assert transform.state_dict() == {}


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_device_transfer():
    transform = BeatsKaldiFbankTransform().to("cuda").eval()
    output = transform(torch.randn(1, 16000), sample_rate=16000)

    assert transform.device.type == "cuda"
    assert output["input_features"].device.type == "cuda"
    assert output["valid_feature_frames"].device.type == "cuda"
    assert output["valid_seconds"].device.type == "cuda"


def test_zero_source_samples_group_under_resample(transform):
    # Under 8 kHz upsampling, an extremely small valid_seconds rounds the
    # source sample count to 0 (target still has 1); this group directly
    # constructs a zero waveform of the target length and must not crash
    # inside resample.
    waveform = torch.randn(2, 4800)
    valid_seconds = torch.tensor([4e-5, 0.5])

    output = transform(
        waveform, sample_rate=8000, valid_seconds=valid_seconds)

    features = output["input_features"]
    assert features.shape[0] == 2
    assert torch.isfinite(features).all()
