"""Offline unit tests for AstKaldiFbankTransform."""

from __future__ import annotations

import inspect

import pytest
import torch

from timbral.models.transforms import AstKaldiFbankTransform
from timbral.models.transforms import ast_transform as transform_module


@pytest.fixture(scope="module")
def transform() -> AstKaldiFbankTransform:
    """Reuse the fixed mel filter."""
    return AstKaldiFbankTransform()


def test_public_export_parameterless_constructor_and_empty_state():
    from timbral.models.transforms.ast_transform import (
        AstKaldiFbankTransform as DirectTransform,
    )

    assert AstKaldiFbankTransform is DirectTransform
    assert not inspect.signature(AstKaldiFbankTransform).parameters
    assert AstKaldiFbankTransform().state_dict() == {}


def test_fixed_output_shape_dtype_and_spectral_padding(
    transform: AstKaldiFbankTransform,
):
    waveform = torch.zeros(2, 400, dtype=torch.float64)
    valid_seconds = torch.tensor(
        [1.0 / 16000.0, 400.0 / 16000.0],
        dtype=torch.float64,
    )

    output = transform(
        waveform,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    assert set(output) == {"input_features", "valid_seconds"}
    assert output["input_features"].shape == (2, 1024, 128)
    assert output["input_features"].dtype == torch.float32
    assert output["valid_seconds"].dtype == torch.float32
    padding_value = -transform_module._FEATURE_MEAN / (
        2.0 * transform_module._FEATURE_STD
    )
    torch.testing.assert_close(
        output["input_features"][0],
        torch.full((1024, 128), padding_value),
    )
    assert not torch.equal(
        output["input_features"][1, 0],
        output["input_features"][1, 1],
    )


def test_multichannel_uses_arithmetic_mean(
    transform: AstKaldiFbankTransform,
):
    generator = torch.Generator().manual_seed(13)
    mono = torch.randn(2, 1600, generator=generator)
    stereo = torch.stack((mono - 0.4, mono + 0.4), dim=1)
    valid_seconds = torch.tensor([0.05, 0.1])

    mono_output = transform(
        mono,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )
    stereo_output = transform(
        stereo,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    torch.testing.assert_close(
        stereo_output["input_features"],
        mono_output["input_features"],
    )


def test_invalid_tail_is_isolated_before_and_after_resampling(
    transform: AstKaldiFbankTransform,
):
    generator = torch.Generator().manual_seed(17)
    valid_prefix = torch.randn(2, 2205, generator=generator)
    waveform_a = torch.cat(
        (valid_prefix, torch.full((2, 2205), 3.0)),
        dim=1,
    )
    waveform_b = torch.cat(
        (valid_prefix, torch.full((2, 2205), -4.0)),
        dim=1,
    )
    valid_seconds = torch.tensor([0.05, 0.1])

    output_a = transform(
        waveform_a,
        sample_rate=22050,
        valid_seconds=valid_seconds,
    )
    output_b = transform(
        waveform_b,
        sample_rate=22050,
        valid_seconds=valid_seconds,
    )

    torch.testing.assert_close(
        output_a["input_features"],
        output_b["input_features"],
    )


def test_mixed_batch_matches_individual_valid_prefixes(
    transform: AstKaldiFbankTransform,
):
    generator = torch.Generator().manual_seed(19)
    waveform = torch.randn(3, 16400, generator=generator)
    valid_seconds = torch.tensor([0.025, 0.9999375, 1.025])

    batch_output = transform(
        waveform,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )
    individual_outputs = [
        transform(
            waveform[index : index + 1, :num_samples],
            sample_rate=16000,
        )
        for index, num_samples in enumerate((400, 15999, 16400))
    ]

    for index, individual in enumerate(individual_outputs):
        torch.testing.assert_close(
            batch_output["input_features"][index],
            individual["input_features"][0],
        )


def test_resampled_mixed_batch_matches_individual_valid_prefixes(
    transform: AstKaldiFbankTransform,
):
    generator = torch.Generator().manual_seed(23)
    waveform = torch.randn(3, 23000, generator=generator)
    valid_seconds = torch.tensor([0.025, 0.999, 1.0])

    batch_output = transform(
        waveform,
        sample_rate=22050,
        valid_seconds=valid_seconds,
    )
    for index, seconds in enumerate(valid_seconds):
        valid_samples = round(float(seconds) * 22050)
        individual = transform(
            waveform[index : index + 1, :valid_samples],
            sample_rate=22050,
        )
        torch.testing.assert_close(
            batch_output["input_features"][index],
            individual["input_features"][0],
        )


def test_discrete_valid_frame_boundaries(
    transform: AstKaldiFbankTransform,
):
    valid_samples = torch.tensor(
        [1, 399, 400, 559, 560, 164079, 164080]
    )
    waveform = torch.randn(valid_samples.shape[0], 164080)

    output = transform(
        waveform,
        sample_rate=16000,
        valid_seconds=valid_samples.to(torch.float32) / 16000,
    )
    expected_rows = torch.tensor([0, 0, 1, 1, 2, 1023, 1024])
    padding_value = -transform_module._FEATURE_MEAN / (
        2.0 * transform_module._FEATURE_STD
    )
    row_indices = torch.arange(1024).unsqueeze(0)
    padding_mask = row_indices >= expected_rows.unsqueeze(1)

    actual_padding = output["input_features"].eq(padding_value).all(dim=2)
    assert torch.equal(actual_padding, padding_mask)


def test_physical_long_padding_is_legal_but_effective_overflow_errors(
    transform: AstKaldiFbankTransform,
):
    waveform = torch.zeros(1, 20 * 16000)

    output = transform(
        waveform,
        sample_rate=16000,
        valid_seconds=torch.tensor([5.0]),
    )
    assert output["input_features"].shape == (1, 1024, 128)

    with pytest.raises(ValueError, match="10.255"):
        transform(waveform, sample_rate=16000)
    with pytest.raises(ValueError, match="10.255"):
        transform(
            waveform,
            sample_rate=16000,
            valid_seconds=torch.tensor([10.2550625]),
        )


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "valid_seconds", "error"),
    [
        (torch.ones(100), 16000, None, ValueError),
        (torch.ones(1, 100, dtype=torch.int16), 16000, None, TypeError),
        (torch.ones(1, 100), 16000.0, None, TypeError),
        (torch.ones(1, 100), 0, None, ValueError),
        (torch.ones(1, 100), 16000, [0.001], TypeError),
        (torch.ones(1, 100), 16000, torch.ones(2), ValueError),
        (torch.ones(1, 100), 16000, torch.tensor([0.0]), ValueError),
        (torch.ones(1, 100), 16000, torch.tensor([1.0]), ValueError),
    ],
)
def test_input_contract_errors(
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


def test_unknown_argument_raises(transform: AstKaldiFbankTransform):
    with pytest.raises(TypeError, match="unknown"):
        transform(
            torch.ones(1, 400),
            sample_rate=16000,
            unknown=True,
        )


def test_waveform_gradient_propagates(
    transform: AstKaldiFbankTransform,
):
    waveform = torch.randn(2, 800, requires_grad=True)
    output = transform(
        waveform,
        sample_rate=16000,
        valid_seconds=torch.tensor([0.025, 0.05]),
    )
    output["input_features"].square().mean().backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="No CUDA available in the current environment.",
)
def test_cuda_device_and_output():
    transform = AstKaldiFbankTransform().cuda()
    output = transform(
        torch.randn(1, 400),
        sample_rate=16000,
    )

    assert output["input_features"].is_cuda
    assert output["valid_seconds"].is_cuda
