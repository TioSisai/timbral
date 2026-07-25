"""Unit tests for PannsLogmelTransform without weights."""

from __future__ import annotations

import inspect

import pytest
import torch

from timbral.models.helpers.panns import PannsVariant as HelperPannsVariant
from timbral.models.transforms import PannsLogmelTransform, PannsVariant


def _transform(
    *,
    target_sample_rate: int = 16000,
    variant: str = "max_mean",
    **overrides,
) -> PannsLogmelTransform:
    parameters = {
        "target_sample_rate": target_sample_rate,
        "n_fft": 512 if target_sample_rate == 16000 else 1024,
        "win_length": 512 if target_sample_rate == 16000 else 1024,
        "hop_length": 160 if target_sample_rate == 16000 else 320,
        "n_mels": 64,
        "f_min": 50.0,
        "f_max": 8000.0 if target_sample_rate == 16000 else 14000.0,
        "variant": variant,
        "pretrained": False,
    }
    parameters.update(overrides)
    return PannsLogmelTransform(**parameters)


@pytest.fixture(scope="module")
def transform_16k() -> PannsLogmelTransform:
    """Reuse the expensive DFT/mel initialization."""
    return _transform().eval()


def test_public_export_and_keyword_only_constructor():
    from timbral.models.transforms.panns import (
        PannsLogmelTransform as DirectTransform,
        PannsVariant as DirectVariant,
    )

    assert PannsLogmelTransform is DirectTransform
    assert PannsVariant is DirectVariant
    assert PannsVariant is HelperPannsVariant
    parameters = inspect.signature(PannsLogmelTransform).parameters
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY
        for parameter in parameters.values()
    )


def test_pretrained_false_allows_random_16k_decision_level_max():
    transform = _transform(variant="decision_level_max")

    assert transform.variant == "decision_level_max"
    assert not transform.pretrained


def test_pretrained_true_rejects_nonexistent_or_mismatched_configuration():
    with pytest.raises(ValueError, match="No official weights exist"):
        _transform(
            variant="decision_level_max",
            pretrained=True,
        )
    with pytest.raises(ValueError, match="exactly match"):
        _transform(
            hop_length=200,
            pretrained=True,
        )


def test_odd_n_fft_is_rejected():
    with pytest.raises(ValueError, match="n_fft must be even"):
        _transform(
            n_fft=511,
            win_length=511,
        )


def test_custom_mel_dimension_is_reflected_in_output():
    transform = _transform(n_mels=32).eval()
    waveform = torch.randn(1, 16000)

    with torch.inference_mode():
        output = transform(waveform, sample_rate=16000)

    assert output["input_features"].shape == (1, 101, 32)


def test_power_spectrogram_matches_torch_stft(
    transform_16k: PannsLogmelTransform,
):
    generator = torch.Generator().manual_seed(7)
    waveform = torch.randn(2, 4096, generator=generator)
    window = torch.hann_window(512, periodic=True)

    with torch.inference_mode():
        actual = transform_16k._power_spectrogram(waveform)
        expected = (
            torch.stft(
                waveform,
                n_fft=512,
                hop_length=160,
                win_length=512,
                window=window,
                center=True,
                pad_mode="reflect",
                return_complex=True,
            )
            .abs()
            .square()
            .transpose(1, 2)
        )

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


def test_multichannel_mean_and_float32(
    transform_16k: PannsLogmelTransform,
):
    mono = torch.randn(2, 4000, dtype=torch.float64)
    stereo = torch.stack((mono - 0.25, mono + 0.25), dim=1)
    valid_seconds = torch.tensor([0.2, 0.25], dtype=torch.float64)

    with torch.inference_mode():
        mono_output = transform_16k(
            mono,
            sample_rate=16000,
            valid_seconds=valid_seconds,
        )
        stereo_output = transform_16k(
            stereo,
            sample_rate=16000,
            valid_seconds=valid_seconds,
        )

    assert stereo_output["input_features"].dtype == torch.float32
    assert stereo_output["valid_seconds"].dtype == torch.float32
    torch.testing.assert_close(
        stereo_output["input_features"],
        mono_output["input_features"],
    )


def test_invalid_padding_is_cleared_before_frontend(
    transform_16k: PannsLogmelTransform,
):
    generator = torch.Generator().manual_seed(11)
    valid_prefix = torch.randn(1, 8000, generator=generator)
    waveform_a = torch.cat((valid_prefix, torch.full((1, 8000), 3.0)), 1)
    waveform_b = torch.cat((valid_prefix, torch.full((1, 8000), -4.0)), 1)
    valid_seconds = torch.tensor([0.5])

    with torch.inference_mode():
        output_a = transform_16k(
            waveform_a,
            sample_rate=16000,
            valid_seconds=valid_seconds,
        )
        output_b = transform_16k(
            waveform_b,
            sample_rate=16000,
            valid_seconds=valid_seconds,
        )

    torch.testing.assert_close(
        output_a["input_features"],
        output_b["input_features"],
    )


def test_short_input_uses_pre_bn_silence_completion(
    transform_16k: PannsLogmelTransform,
):
    waveform = torch.randn(1, 320)

    with torch.inference_mode():
        native_logmel = transform_16k._logmel(waveform)
        output = transform_16k(waveform, sample_rate=16000)
        expected_logmel = torch.nn.functional.pad(
            native_logmel,
            (0, 0, 0, 29),
            value=-100.0,
        )
        expected = (
            transform_16k.bn0(
                expected_logmel.transpose(1, 2).unsqueeze(-1)
            )
            .squeeze(-1)
            .transpose(1, 2)
        )

    assert output["valid_feature_frames"].tolist() == [3]
    assert output["input_features"].shape == (1, 32, 64)
    torch.testing.assert_close(output["input_features"], expected)


def test_normal_input_retains_nonmultiple_native_frames(
    transform_16k: PannsLogmelTransform,
):
    waveform = torch.randn(1, 16000)

    with torch.inference_mode():
        output = transform_16k(waveform, sample_rate=16000)

    assert output["valid_feature_frames"].tolist() == [101]
    assert output["input_features"].shape == (1, 101, 64)


def test_joint_source_target_length_grouping_matches_individual_calls():
    transform = _transform(target_sample_rate=32000).eval()
    waveform = torch.randn(2, 1200)
    valid_seconds = torch.tensor([0.062475, 0.062525])

    with torch.inference_mode():
        batch_output = transform(
            waveform,
            sample_rate=16000,
            valid_seconds=valid_seconds,
        )
        individual_outputs = [
            transform(
                waveform[index : index + 1],
                sample_rate=16000,
                valid_seconds=valid_seconds[index : index + 1],
            )
            for index in range(2)
        ]

    source_lengths = torch.round(valid_seconds * 16000).long()
    target_lengths = torch.round(valid_seconds * 32000).long()
    assert source_lengths.tolist() == [1000, 1000]
    assert target_lengths.tolist() == [1999, 2001]
    for index, individual in enumerate(individual_outputs):
        torch.testing.assert_close(
            batch_output["input_features"][index],
            individual["input_features"][0],
        )


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "valid_seconds", "error"),
    [
        (torch.ones(10), 16000, None, ValueError),
        (torch.ones(1, 100, dtype=torch.int16), 16000, None, TypeError),
        (torch.ones(1, 100), 16000.0, None, TypeError),
        (torch.ones(1, 100), 0, None, ValueError),
        (torch.ones(1, 100), 16000, torch.ones(2), ValueError),
        (torch.ones(1, 100), 16000, torch.tensor([0.0]), ValueError),
        (torch.ones(1, 100), 16000, torch.tensor([1.0]), ValueError),
        # At the target sample rate, the valid sample count rounds to 0
        (torch.ones(1, 100), 16000, torch.tensor([1e-5]), ValueError),
    ],
)
def test_input_contract_errors(
    transform_16k,
    waveform,
    sample_rate,
    valid_seconds,
    error,
):
    with pytest.raises(error):
        transform_16k(
            waveform,
            sample_rate=sample_rate,
            valid_seconds=valid_seconds,
        )


@pytest.mark.parametrize(
    ("num_samples", "pad_mode"),
    [(256, "constant"), (257, "reflect")],
)
def test_padding_mode_switches_at_half_n_fft(
    transform_16k: PannsLogmelTransform,
    num_samples: int,
    pad_mode: str,
):
    """Degenerates to zero padding when the length is <= n_fft // 2, and the frame-count formula is consistent on both sides of the boundary."""
    generator = torch.Generator().manual_seed(11)
    waveform = torch.randn(1, num_samples, generator=generator)
    window = torch.hann_window(512, periodic=True)
    native_frames = num_samples // 160 + 1

    with torch.inference_mode():
        actual = transform_16k._power_spectrogram(waveform)
        expected = (
            torch.stft(
                torch.nn.functional.pad(waveform, (256, 256), mode=pad_mode),
                n_fft=512,
                hop_length=160,
                win_length=512,
                window=window,
                center=False,
                return_complex=True,
            )
            .abs()
            .square()
            .transpose(1, 2)
        )
        output = transform_16k(waveform, sample_rate=16000)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert actual.shape[1] == native_frames
    assert output["input_features"].shape == (1, 32, 64)
    assert torch.equal(
        output["valid_feature_frames"],
        torch.tensor([native_frames]),
    )


def test_training_path_preserves_waveform_and_bn_gradients(
    transform_16k: PannsLogmelTransform,
):
    transform_16k.train()
    transform_16k.zero_grad(set_to_none=True)
    waveform = torch.randn(2, 640, requires_grad=True)
    try:
        output = transform_16k(
            waveform,
            sample_rate=16000,
            valid_seconds=torch.tensor([0.02, 0.04]),
        )
        output["input_features"].square().mean().backward()
    finally:
        transform_16k.eval()

    assert waveform.grad is not None
    assert transform_16k.bn0.weight.grad is not None


def test_zero_source_samples_group_under_resample(
    transform_16k: PannsLogmelTransform,
):
    # Under 8 kHz upsampling, an extremely small valid_seconds rounds the
    # source sample count to 0 (target still has 1); this group directly
    # constructs a zero waveform of the target length and must not crash
    # inside resample.
    waveform = torch.randn(2, 4800)
    valid_seconds = torch.tensor([4e-5, 0.5])

    output = transform_16k(
        waveform, sample_rate=8000, valid_seconds=valid_seconds)

    features = output["input_features"]
    assert features.shape[0] == 2
    assert torch.isfinite(features).all()
    assert output["valid_feature_frames"].tolist()[0] == 1
