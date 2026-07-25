"""Offline unit tests for ClapLogmelTransform."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from timbral.models.transforms import ClapLogmelTransform
from timbral.models.transforms import clap as transform_module


@pytest.fixture(scope="module")
def transform() -> ClapLogmelTransform:
    """Reuse the fixed CLAP frontend buffers."""
    return ClapLogmelTransform()


def _rng_states_equal(left, right) -> bool:
    """Compare NumPy legacy RNG state."""
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def test_public_export_parameterless_constructor_and_empty_state():
    from timbral.models.transforms.clap import (
        ClapLogmelTransform as DirectTransform,
    )

    assert ClapLogmelTransform is DirectTransform
    assert not inspect.signature(ClapLogmelTransform).parameters
    assert ClapLogmelTransform().state_dict() == {}


def test_fixed_output_and_dtype_conversion_preserves_internal_precision(
    transform: ClapLogmelTransform,
):
    output = transform(
        torch.zeros(1, 48000),
        sample_rate=48000,
    )

    assert set(output) == {"input_features", "valid_seconds"}
    assert output["input_features"].shape == (1, 4, 1001, 64)
    assert output["input_features"].dtype == torch.float32
    assert output["valid_seconds"].dtype == torch.float32
    assert transform.mel_filters.dtype == torch.float64
    assert transform.window.dtype == torch.float64

    transform.float()
    assert transform.mel_filters.dtype == torch.float64
    assert transform.window.dtype == torch.float64


def test_multichannel_mean_and_invalid_tail_are_isolated(
    transform: ClapLogmelTransform,
):
    generator = torch.Generator().manual_seed(13)
    mono = torch.randn(2, 9600, generator=generator)
    stereo_a = torch.stack((mono - 0.4, mono + 0.4), dim=1)
    stereo_b = stereo_a.clone()
    stereo_b[0, :, 4800:] = 9.0
    valid_seconds = torch.tensor([0.1, 0.2])

    mono_output = transform(
        mono,
        sample_rate=48000,
        valid_seconds=valid_seconds,
    )
    stereo_output = transform(
        stereo_a,
        sample_rate=48000,
        valid_seconds=valid_seconds,
    )
    changed_tail_output = transform(
        stereo_b,
        sample_rate=48000,
        valid_seconds=valid_seconds,
    )

    torch.testing.assert_close(
        stereo_output["input_features"],
        mono_output["input_features"],
    )
    torch.testing.assert_close(
        changed_tail_output["input_features"][0],
        stereo_output["input_features"][0],
    )


def test_resampled_mixed_batch_matches_individual_prefixes(
    transform: ClapLogmelTransform,
):
    generator = torch.Generator().manual_seed(17)
    waveform = torch.randn(2, 8820, generator=generator)
    valid_seconds = torch.tensor([0.1, 0.2])

    batch = transform(
        waveform,
        sample_rate=44100,
        valid_seconds=valid_seconds,
    )
    individual = [
        transform(
            waveform[index : index + 1, :valid_samples],
            sample_rate=44100,
        )
        for index, valid_samples in enumerate((4410, 8820))
    ]

    for index, expected in enumerate(individual):
        torch.testing.assert_close(
            batch["input_features"][index],
            expected["input_features"][0],
        )


def test_resampling_only_consumes_group_valid_prefixes(
    transform: ClapLogmelTransform,
    monkeypatch,
):
    resampled_shapes = []

    def fake_resample(
        waveform,
        *,
        orig_freq,
        new_freq,
    ):
        resampled_shapes.append(tuple(waveform.shape))
        target_length = transform_module._round_positive_ratio(
            waveform.shape[1] * new_freq,
            orig_freq,
        )
        return torch.nn.functional.interpolate(
            waveform.unsqueeze(1),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

    monkeypatch.setattr(
        transform_module.torchaudio.functional,
        "resample",
        fake_resample,
    )
    transform(
        torch.randn(2, 4000),
        sample_rate=1000,
        valid_seconds=torch.tensor([0.01, 0.02]),
    )

    assert resampled_shapes == [(1, 10), (1, 20)]


def test_integer_ratio_rounding_uses_nearest_even():
    assert transform_module._round_positive_ratio(1, 2) == 0
    assert transform_module._round_positive_ratio(3, 2) == 2
    assert (
        transform_module._round_positive_ratio(
            6_147_839 * 48_000,
            48_000,
        )
        == 6_147_839
    )


def test_fusion_boundary_channels_and_no_rng_consumption(
    transform: ClapLogmelTransform,
):
    generator = torch.Generator().manual_seed(19)
    waveform = torch.randn(2, 960000, generator=generator)

    original_state = np.random.get_state()
    try:
        np.random.seed(1729)
        state_before = np.random.get_state()
        short = transform(
            waveform[:, :480479],
            sample_rate=48000,
        )
        boundary = transform(
            waveform[:, :480480],
            sample_rate=48000,
        )
        long = transform(waveform, sample_rate=48000)
        state_after = np.random.get_state()
    finally:
        np.random.set_state(original_state)

    assert _rng_states_equal(state_before, state_after)
    assert torch.equal(
        short["input_features"][:, 0],
        short["input_features"][:, 1],
    )
    assert not torch.equal(
        boundary["input_features"][:, 0],
        boundary["input_features"][:, 1],
    )
    assert not torch.equal(
        long["input_features"][:, 0],
        long["input_features"][:, 1],
    )


def test_anchored_crop_starts_match_formula():
    starts = ClapLogmelTransform._anchored_crop_starts
    assert starts(1002) == (0, 1, 1)
    assert starts(1137) == (0, 68, 136)
    assert starts(2001) == (0, 500, 1000)
    assert starts(6001) == (500, 2500, 4501)


def test_long_audio_repeat_and_batch_independent(
    transform: ClapLogmelTransform,
):
    generator = torch.Generator().manual_seed(29)
    waveform = torch.randn(2, 960000, generator=generator)
    valid_seconds = torch.tensor([20.0, 5.0])

    mixed = transform(
        waveform,
        sample_rate=48000,
        valid_seconds=valid_seconds,
    )
    repeated = transform(
        waveform,
        sample_rate=48000,
        valid_seconds=valid_seconds,
    )
    long_alone = transform(
        waveform[:1],
        sample_rate=48000,
    )
    short_alone = transform(
        waveform[1:, :240000],
        sample_rate=48000,
    )

    assert torch.equal(
        mixed["input_features"],
        repeated["input_features"],
    )
    assert torch.equal(
        mixed["input_features"][0],
        long_alone["input_features"][0],
    )
    assert torch.equal(
        mixed["input_features"][1],
        short_alone["input_features"][0],
    )


def test_gradient_reaches_valid_waveform(
    transform: ClapLogmelTransform,
):
    waveform = torch.randn(1, 4800, requires_grad=True)

    output = transform(waveform, sample_rate=48000)
    output["input_features"].square().mean().backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()
    assert torch.count_nonzero(waveform.grad) > 0


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "valid_seconds", "error"),
    [
        (torch.ones(100), 48000, None, ValueError),
        (torch.ones(1, 100, dtype=torch.int16), 48000, None, TypeError),
        (torch.ones(1, 100), 48000.0, None, TypeError),
        (torch.ones(1, 100), 0, None, ValueError),
        (torch.ones(1, 100), 48000, torch.ones(2), ValueError),
        (torch.ones(1, 100), 48000, torch.tensor([0.0]), ValueError),
        (torch.ones(1, 100), 48000, torch.tensor([1.0]), ValueError),
    ],
)
def test_invalid_public_inputs_raise(
    transform,
    waveform,
    sample_rate,
    valid_seconds,
    error,
):
    with pytest.raises(error):
        transform(
            waveform,
            sample_rate=sample_rate,
            valid_seconds=valid_seconds,
        )


def test_target_grid_zero_length_raises(
    transform: ClapLogmelTransform,
):
    with pytest.raises(ValueError, match="at least 1"):
        transform(
            torch.ones(1, 1),
            sample_rate=192000,
        )


def test_unknown_parameter_raises(transform: ClapLogmelTransform):
    with pytest.raises(TypeError, match="unknown"):
        transform(
            torch.ones(1, 4800),
            sample_rate=48000,
            unknown=True,
        )
