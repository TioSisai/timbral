"""Unit tests for ``AtstMelspecTransform`` without weights."""

from __future__ import annotations

import inspect
import math

import pytest
import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

from timbral.models.transforms import AtstMelspecTransform

_TARGET_SR = 16000
_HOP_LENGTH = 160
_NUM_MELS = 64
_TOP_DB = 80.0
_NORM_MIN = -79.6482
_NORM_MAX = 50.6842
_POWER_MULTIPLIER = 10.0
# ``AmplitudeToDB`` clamps power at ``amin`` before the logarithm, so a
# digitally silent sample lands on exactly -100 dB and, after the fixed
# MinMax mapping, on one constant feature value. Derived here from the
# documented constants rather than read back off the implementation.
_AMPLITUDE_FLOOR = 1e-10
_SILENCE_DECIBEL = _POWER_MULTIPLIER * math.log10(_AMPLITUDE_FLOOR)
_SILENCE_FEATURE = (_SILENCE_DECIBEL - _NORM_MIN) / (
    _NORM_MAX - _NORM_MIN
) * 2.0 - 1.0
# ``center=True`` reflect-pads by ``n_fft // 2``; 513 is therefore the
# shortest waveform torch.stft accepts here, and it yields 4 mel frames.
_MIN_TARGET_SAMPLES = 513


@pytest.fixture(scope="module")
def transform() -> AtstMelspecTransform:
    """Reuse the parameterless official frontend."""
    return AtstMelspecTransform().eval()


def _reference_melspec() -> torchaudio.transforms.MelSpectrogram:
    """Build the official 16 kHz, 64-mel, 1024/1024/160 frontend."""
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=_TARGET_SR,
        n_fft=1024,
        win_length=1024,
        hop_length=_HOP_LENGTH,
        f_min=60.0,
        f_max=7800.0,
        n_mels=_NUM_MELS,
    )


def _rescale(decibel: Tensor) -> Tensor:
    """Apply the fixed MinMax rescaling and put time first."""
    features = (decibel - _NORM_MIN) / (
        _NORM_MAX - _NORM_MIN
    ) * 2.0 - 1.0
    return features.transpose(1, 2)


def _reference_features(waveform: Tensor) -> Tensor:
    """Official reference with a per-sample ``top_db`` reduction.

    ``AmplitudeToDB`` takes the peak per sample once the input is 4-D,
    which is the behaviour the transform reproduces by hand.
    """
    power = _reference_melspec()(waveform)
    to_decibel = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=_TOP_DB,
    )
    return _rescale(to_decibel(power.unsqueeze(1)).squeeze(1))


def _shared_peak_features(waveform: Tensor) -> Tensor:
    """Reference with the batch-shared ``top_db`` reduction.

    A 3-D input makes ``AmplitudeToDB`` reduce the peak over the batch
    axis as well, so one loud sample raises the floor of every other
    sample. The transform must not behave this way.
    """
    power = _reference_melspec()(waveform)
    to_decibel = torchaudio.transforms.AmplitudeToDB(
        stype="power",
        top_db=_TOP_DB,
    )
    return _rescale(to_decibel(power))


def test_public_export_and_no_trainable_state():
    from timbral.models.transforms.atst import (
        AtstMelspecTransform as DirectTransform,
    )

    assert AtstMelspecTransform is DirectTransform
    assert inspect.signature(AtstMelspecTransform).parameters == {}

    transform = AtstMelspecTransform()
    assert transform.target_sample_rate == _TARGET_SR
    assert transform.device.type == "cpu"
    assert list(transform.parameters()) == []
    assert set(transform.state_dict()) == {
        "melspec.spectrogram.window",
        "melspec.mel_scale.fb",
    }
    assert not any(
        buffer.requires_grad for buffer in transform.buffers()
    )


def test_output_contract(transform):
    output = transform(torch.randn(2, 16000), sample_rate=16000)

    assert set(output) == {
        "input_features",
        "valid_feature_frames",
        "valid_seconds",
    }
    assert output["input_features"].shape == (2, 101, _NUM_MELS)
    assert output["input_features"].dtype == torch.float32
    assert output["input_features"].device == transform.device
    assert output["valid_feature_frames"].dtype == torch.int64
    assert output["valid_feature_frames"].device == transform.device
    assert torch.equal(
        output["valid_feature_frames"],
        torch.tensor([101, 101]),
    )
    assert output["valid_seconds"].dtype == torch.float32
    assert output["valid_seconds"].device == transform.device
    assert torch.allclose(
        output["valid_seconds"],
        torch.tensor([1.0, 1.0]),
    )


@pytest.mark.parametrize(
    ("num_samples", "expected_frames"),
    [
        (160, 4),
        (512, 4),
        (513, 4),
        (640, 5),
        (1600, 11),
        (16000, 101),
        (160000, 1001),
    ],
)
def test_valid_feature_frames_formula(
    transform,
    num_samples,
    expected_frames,
):
    # Above the 513-sample floor the law is exactly N // 160 + 1.
    output = transform(torch.randn(1, num_samples), sample_rate=16000)

    assert int(output["valid_feature_frames"][0]) == expected_frames
    assert output["input_features"].shape == (
        1,
        expected_frames,
        _NUM_MELS,
    )


@pytest.mark.parametrize("seconds", (0.2, 1.0, 4.03, 10.0))
def test_features_match_official_reference(transform, seconds):
    generator = torch.Generator().manual_seed(round(seconds * 100))
    num_samples = round(seconds * _TARGET_SR)
    waveform = torch.randn((2, num_samples), generator=generator)

    output = transform(waveform, sample_rate=16000)

    assert torch.equal(
        output["input_features"],
        _reference_features(waveform),
    )


def test_short_input_is_zero_padded_to_the_melspec_floor(transform):
    # 513 samples is the smallest length the underlying MelSpectrogram
    # accepts at all; the transform pads up to it instead of crashing.
    with pytest.raises(RuntimeError, match="Padding size"):
        transform.melspec(torch.zeros(1, _MIN_TARGET_SAMPLES - 1))
    assert transform.melspec(
        torch.zeros(1, _MIN_TARGET_SAMPLES)
    ).shape == (1, _NUM_MELS, 4)

    generator = torch.Generator().manual_seed(3)
    waveform = torch.randn((1, 160), generator=generator)

    output = transform(waveform, sample_rate=16000)

    assert int(output["valid_feature_frames"][0]) == 4
    assert output["input_features"].shape == (1, 4, _NUM_MELS)
    assert torch.equal(
        output["input_features"],
        _reference_features(
            F.pad(waveform, (0, _MIN_TARGET_SAMPLES - 160))
        ),
    )
    assert torch.allclose(
        output["valid_seconds"],
        torch.tensor([0.01]),
    )


def test_digital_silence_lands_on_the_amplitude_floor(transform):
    """Silent power clamps at ``amin`` instead of diverging to -inf.

    Every mel bin of a silent sample sits at exactly 1e-10, so the whole
    map collapses onto the one value the documented dB and MinMax
    constants predict. No other test drives any bin down to the floor.
    """
    output = transform(torch.zeros(2, 16000), sample_rate=16000)
    features = output["input_features"]

    assert features.shape == (2, 101, _NUM_MELS)
    assert torch.isfinite(features).all()
    torch.testing.assert_close(
        features,
        torch.full_like(features, _SILENCE_FEATURE),
        atol=1e-6,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    ("sample_rate", "num_samples", "expected_frames"),
    [
        (8000, 4000, 51),
        (8000, 8000, 101),
        (8000, 80000, 1001),
        (44100, 441, 4),
        (44100, 22050, 51),
        (44100, 44100, 101),
    ],
)
def test_resampled_frame_counts_follow_the_16k_target(
    transform,
    sample_rate,
    num_samples,
    expected_frames,
):
    generator = torch.Generator().manual_seed(num_samples)
    waveform = torch.randn((1, num_samples), generator=generator)

    output = transform(waveform, sample_rate=sample_rate)

    assert int(output["valid_feature_frames"][0]) == expected_frames
    assert output["input_features"].shape == (
        1,
        expected_frames,
        _NUM_MELS,
    )


def test_resampled_features_match_the_resampled_reference(transform):
    generator = torch.Generator().manual_seed(23)
    waveform = torch.randn((1, 8000), generator=generator)

    output = transform(waveform, sample_rate=8000)
    resampled = torchaudio.functional.resample(
        waveform,
        orig_freq=8000,
        new_freq=_TARGET_SR,
    )

    assert resampled.shape == (1, 16000)
    assert torch.equal(
        output["input_features"],
        _reference_features(resampled),
    )


def test_zero_source_samples_group_at_an_extremely_low_rate(transform):
    """A valid region rounding to 0 source samples still yields silence.

    At 10 Hz, 0.04 s rounds to 0 source samples while the 16 kHz target
    is 640 samples; the group must materialize as zeros rather than be
    handed to ``resample``, which cannot consume an empty waveform.
    """
    output = transform(
        torch.randn(2, 5),
        sample_rate=10,
        valid_seconds=torch.tensor([0.04, 0.3]),
    )
    features = output["input_features"]

    # 640 // 160 + 1 = 5 frames, and 4800 // 160 + 1 = 31 frames.
    assert output["valid_feature_frames"].tolist() == [5, 31]
    assert features.shape == (2, 31, _NUM_MELS)
    torch.testing.assert_close(
        features[0, :5],
        torch.full((5, _NUM_MELS), _SILENCE_FEATURE),
        atol=1e-6,
        rtol=0.0,
    )
    assert torch.all(features[0, 5:] == 0)


def test_multichannel_downmix_matches_the_pre_averaged_input(transform):
    generator = torch.Generator().manual_seed(29)
    mono = torch.randn((2, 16000), generator=generator)
    stereo = torch.stack((mono - 0.4, mono + 0.4), dim=1)
    valid_seconds = torch.tensor([0.5, 1.0])

    output_stereo = transform(
        stereo,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )
    output_mono = transform(
        stereo.mean(dim=1),
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    assert output_stereo["input_features"].shape == (2, 101, _NUM_MELS)
    assert torch.equal(
        output_stereo["input_features"],
        output_mono["input_features"],
    )
    assert torch.equal(
        output_stereo["valid_feature_frames"],
        output_mono["valid_feature_frames"],
    )


def test_padding_never_influences_the_valid_region(transform):
    generator = torch.Generator().manual_seed(31)
    valid_prefix = torch.randn((2, 8000), generator=generator)
    quiet_tail = torch.cat(
        (valid_prefix, torch.zeros(2, 8000)),
        dim=1,
    )
    loud_tail = torch.cat(
        (valid_prefix, torch.full((2, 8000), 7.0)),
        dim=1,
    )
    valid_seconds = torch.tensor([0.5, 0.5])

    output_quiet = transform(
        quiet_tail,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )
    output_loud = transform(
        loud_tail,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    assert output_quiet["valid_feature_frames"].tolist() == [51, 51]
    assert output_quiet["input_features"].shape == (2, 51, _NUM_MELS)
    assert torch.equal(
        output_quiet["input_features"],
        output_loud["input_features"],
    )
    assert torch.equal(
        output_quiet["input_features"],
        _reference_features(valid_prefix),
    )


def test_invalid_tail_is_dropped_before_resampling(transform):
    """The crop to the valid region precedes ``resample``.

    At the native rate a later crop to the target length would hide a
    missing one, but resampling convolves with a wide sinc kernel, so a
    tail left in place bleeds back across the crop point and moves the
    final mel frames.
    """
    generator = torch.Generator().manual_seed(47)
    valid_prefix = torch.randn((2, 22050), generator=generator)
    quiet_tail = torch.cat(
        (valid_prefix, torch.zeros(2, 22050)),
        dim=1,
    )
    loud_tail = torch.cat(
        (valid_prefix, torch.full((2, 22050), 8.0)),
        dim=1,
    )
    valid_seconds = torch.tensor([0.5, 0.5])

    output_quiet = transform(
        quiet_tail,
        sample_rate=44100,
        valid_seconds=valid_seconds,
    )
    output_loud = transform(
        loud_tail,
        sample_rate=44100,
        valid_seconds=valid_seconds,
    )

    assert output_quiet["valid_feature_frames"].tolist() == [51, 51]
    assert torch.equal(
        output_quiet["input_features"],
        output_loud["input_features"],
    )
    assert torch.equal(
        output_quiet["input_features"],
        _reference_features(
            torchaudio.functional.resample(
                valid_prefix,
                orig_freq=44100,
                new_freq=_TARGET_SR,
            )
        ),
    )


def test_mixed_batch_matches_single_calls(transform):
    generator = torch.Generator().manual_seed(37)
    valid_seconds = torch.tensor([0.01, 0.5, 1.0, 4.03])
    max_samples = round(4.03 * _TARGET_SR)
    batch = torch.full((4, max_samples), -3.0)
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
        # Each unique valid length is its own group, so batching cannot
        # perturb a sample by even one ulp.
        assert torch.equal(
            output_batch["input_features"][index, :frames],
            output_single["input_features"][0],
        )
        # Padding outside each group is exactly 0
        assert torch.all(
            output_batch["input_features"][index, frames:] == 0
        )


def test_top_db_floor_is_taken_per_sample(transform):
    generator = torch.Generator().manual_seed(41)
    loud = torch.randn((16000,), generator=generator) * 10.0
    quiet = torch.randn((16000,), generator=generator) * 1e-4
    batch = torch.stack((loud, quiet))

    output = transform(batch, sample_rate=16000)
    features = output["input_features"]

    assert torch.equal(features, _reference_features(batch))
    # Sharing one peak across the batch would flatten the quiet sample
    # onto the loud sample's floor, i.e. a constant feature map.
    shared = _shared_peak_features(batch)
    assert torch.equal(shared[1], torch.full_like(shared[1], shared[1][0, 0]))
    assert (features[1] - shared[1]).abs().max() > 0.1

    for index, row in enumerate((loud, quiet)):
        single = transform(row.unsqueeze(0), sample_rate=16000)
        # Batching keeps the numerics within one float32 ulp of the
        # single-sample path; only the shared-peak variant would differ
        # in a visible way.
        torch.testing.assert_close(
            features[index],
            single["input_features"][0],
            atol=1e-6,
            rtol=1e-6,
        )


def test_top_db_floor_clamps_a_burst_followed_by_silence(transform):
    """Pin the floor to 80 dB below the peak, not merely per sample.

    Every randn waveform used elsewhere spans under 30 dB within one
    sample, so ``top_db`` never changes a value there and any width
    would do. A loud burst ahead of digital silence spans far more, and
    the clamped minimum then reads the constant back directly.
    """
    generator = torch.Generator().manual_seed(43)
    waveform = torch.zeros(1, 16000)
    waveform[:, :4000] = torch.randn((1, 4000), generator=generator) * 5.0

    output = transform(waveform, sample_rate=16000)
    features = output["input_features"]

    decibel = _POWER_MULTIPLIER * torch.log10(
        torch.clamp(_reference_melspec()(waveform), min=_AMPLITUDE_FLOOR)
    )
    assert float(decibel.amax() - decibel.amin()) > _TOP_DB
    expected_floor = (
        float(decibel.amax()) - _TOP_DB - _NORM_MIN
    ) / (_NORM_MAX - _NORM_MIN) * 2.0 - 1.0

    assert torch.equal(features, _reference_features(waveform))
    torch.testing.assert_close(
        float(features.min()),
        expected_floor,
        atol=1e-6,
        rtol=0.0,
    )
    # The floor is a plateau, not one stray bin: whole silent frames sit
    # on it once the burst has decayed out of the analysis window.
    on_floor = (features[0] <= features.min() + 1e-6).all(dim=-1)
    assert int(on_floor.sum()) > 50


def test_invalid_inputs_raise(transform):
    with pytest.raises(TypeError, match="floating-point"):
        transform(
            torch.zeros((1, 16000), dtype=torch.int16),
            sample_rate=16000,
        )
    with pytest.raises(ValueError, match="shape"):
        transform(torch.zeros(16000), sample_rate=16000)
    with pytest.raises(ValueError, match="shape"):
        transform(torch.zeros((1, 1, 2, 16000)), sample_rate=16000)
    with pytest.raises(TypeError, match="Python int"):
        transform(torch.zeros((1, 16000)), sample_rate=16000.0)
    with pytest.raises(ValueError, match="positive"):
        transform(torch.zeros((1, 16000)), sample_rate=0)
    with pytest.raises(ValueError, match="positive"):
        transform(torch.zeros((1, 16000)), sample_rate=-16000)
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
            valid_seconds=torch.tensor([1.5]),
        )
    with pytest.raises(ValueError, match="exceeds the physical"):
        transform(
            torch.zeros((1, 16000)),
            sample_rate=16000,
            valid_seconds=torch.tensor([0.0]),
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


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_device_transfer():
    transform = AtstMelspecTransform().to("cuda").eval()
    output = transform(torch.randn(1, 16000), sample_rate=16000)

    assert transform.device.type == "cuda"
    assert output["input_features"].device.type == "cuda"
    assert output["valid_feature_frames"].device.type == "cuda"
    assert output["valid_seconds"].device.type == "cuda"
