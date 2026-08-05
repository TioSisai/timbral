"""Offline unit tests for Wav2Vec2WaveformTransform."""

from __future__ import annotations

import inspect

import pytest
import torch

from timbral.models.transforms import Wav2Vec2WaveformTransform

_TARGET_SR = 16000
_MIN_TARGET_SAMPLES = 400
_NORMALIZATION_EPSILON = 1e-7


@pytest.fixture(scope="module")
def transform() -> Wav2Vec2WaveformTransform:
    """Reuse the parameterless default (do_normalize=True) frontend."""
    return Wav2Vec2WaveformTransform()


def test_public_export_keyword_only_constructor_and_empty_state():
    from timbral.models.transforms.wav2vec2 import (
        Wav2Vec2WaveformTransform as DirectTransform,
    )

    assert Wav2Vec2WaveformTransform is DirectTransform
    parameters = inspect.signature(Wav2Vec2WaveformTransform).parameters
    assert list(parameters) == ["do_normalize"]
    assert parameters["do_normalize"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["do_normalize"].default is True
    with pytest.raises(TypeError):
        Wav2Vec2WaveformTransform(True)
    assert Wav2Vec2WaveformTransform().state_dict() == {}


@pytest.mark.parametrize("do_normalize", (1, 0, "yes", None))
def test_do_normalize_rejects_non_strict_bool(do_normalize):
    with pytest.raises(TypeError, match="bool"):
        Wav2Vec2WaveformTransform(do_normalize=do_normalize)


def test_output_contract(transform: Wav2Vec2WaveformTransform):
    output = transform(torch.randn(2, 16000), sample_rate=16000)

    assert set(output) == {
        "input_features",
        "valid_samples",
        "valid_seconds",
    }
    assert output["input_features"].shape == (2, 16000)
    assert output["input_features"].dtype == torch.float32
    assert output["valid_samples"].dtype == torch.int64
    assert torch.equal(
        output["valid_samples"],
        torch.tensor([16000, 16000]),
    )
    assert output["valid_seconds"].dtype == torch.float32
    assert torch.allclose(
        output["valid_seconds"],
        torch.tensor([1.0, 1.0]),
    )
    assert transform.target_sample_rate == 16000
    assert transform.device.type == "cpu"


def test_do_normalize_false_passes_16k_waveform_through():
    transform = Wav2Vec2WaveformTransform(do_normalize=False)
    waveform = torch.randn(2, 800)

    output = transform(waveform, sample_rate=16000)

    assert torch.equal(output["input_features"], waveform)


def test_normalization_statistics_use_only_the_valid_region(
    transform: Wav2Vec2WaveformTransform,
):
    generator = torch.Generator().manual_seed(9)
    waveform = torch.randn(1, 1000, generator=generator) * 3.0 + 2.0
    valid_seconds = torch.tensor([800.0 / _TARGET_SR])

    output = transform(
        waveform,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    valid_prefix = waveform[0, :800]
    mean = valid_prefix.sum() / 800
    variance = (valid_prefix - mean).square().sum() / 800
    reference = (valid_prefix - mean) / torch.sqrt(
        variance + _NORMALIZATION_EPSILON
    )
    torch.testing.assert_close(
        output["input_features"][0, :800],
        reference,
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.all(output["input_features"][0, 800:] == 0)


def test_multichannel_uses_arithmetic_mean(
    transform: Wav2Vec2WaveformTransform,
):
    generator = torch.Generator().manual_seed(21)
    mono = torch.randn(2, 8000, generator=generator)
    stereo = torch.stack((mono - 0.3, mono + 0.3), dim=1)
    valid_seconds = torch.tensor([0.25, 0.5])

    stereo_output = transform(
        stereo,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )
    mono_output = transform(
        stereo.mean(dim=1),
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    assert torch.equal(
        stereo_output["input_features"],
        mono_output["input_features"],
    )


def test_float64_input_produces_float32_output(
    transform: Wav2Vec2WaveformTransform,
):
    output = transform(
        torch.randn(1, 800, dtype=torch.float64),
        sample_rate=16000,
        valid_seconds=torch.tensor([0.05], dtype=torch.float64),
    )

    assert output["input_features"].dtype == torch.float32
    assert output["valid_seconds"].dtype == torch.float32
    assert torch.isfinite(output["input_features"]).all()


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "valid_seconds", "error"),
    [
        (torch.ones(400), 16000, None, ValueError),
        (torch.ones(1, 1, 1, 400), 16000, None, ValueError),
        (torch.ones(1, 400, dtype=torch.int16), 16000, None, TypeError),
        (torch.ones(1, 400), 16000.0, None, TypeError),
        (torch.ones(1, 400), 0, None, ValueError),
        (torch.ones(1, 400), 16000, [0.025], TypeError),
        (torch.ones(1, 400), 16000, torch.ones(2), ValueError),
        (torch.ones(1, 400), 16000, torch.tensor([0.0]), ValueError),
        (torch.ones(1, 400), 16000, torch.tensor([1.0]), ValueError),
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


def test_unknown_argument_raises(transform: Wav2Vec2WaveformTransform):
    with pytest.raises(TypeError, match="unexpected keyword"):
        transform(
            torch.ones(1, 400),
            sample_rate=16000,
            unknown=True,
        )


def test_minimum_length_boundary_at_native_rate(
    transform: Wav2Vec2WaveformTransform,
):
    waveform = torch.randn(1, 16000)

    output = transform(
        waveform,
        sample_rate=16000,
        valid_seconds=torch.tensor([400.0 / _TARGET_SR]),
    )
    assert int(output["valid_samples"][0]) == _MIN_TARGET_SAMPLES

    with pytest.raises(ValueError, match="at least 400"):
        transform(
            waveform,
            sample_rate=16000,
            valid_seconds=torch.tensor([399.0 / _TARGET_SR]),
        )
    with pytest.raises(ValueError, match="at least 400"):
        transform(torch.randn(1, 399), sample_rate=16000)
    assert int(
        transform(torch.randn(1, 400), sample_rate=16000)[
            "valid_samples"
        ][0]
    ) == _MIN_TARGET_SAMPLES


def test_minimum_length_boundary_via_foreign_rate(
    transform: Wav2Vec2WaveformTransform,
):
    waveform = torch.randn(1, 4410)

    output = transform(
        waveform,
        sample_rate=44100,
        valid_seconds=torch.tensor([0.025]),
    )
    assert int(output["valid_samples"][0]) == _MIN_TARGET_SAMPLES

    with pytest.raises(ValueError, match="at least 400"):
        transform(
            waveform,
            sample_rate=44100,
            valid_seconds=torch.tensor([399.0 / _TARGET_SR]),
        )


def test_long_physical_padding_with_short_valid_region_is_legal(
    transform: Wav2Vec2WaveformTransform,
):
    output = transform(
        torch.randn(1, 20 * 16000),
        sample_rate=16000,
        valid_seconds=torch.tensor([0.5]),
    )

    assert output["input_features"].shape == (1, 8000)
    assert int(output["valid_samples"][0]) == 8000


def test_mixed_batch_matches_cropped_calls_at_native_rate(
    transform: Wav2Vec2WaveformTransform,
):
    generator = torch.Generator().manual_seed(3)
    # A duplicated length exercises a multi-row group.
    valid_seconds = torch.tensor([0.025, 0.5, 0.5, 1.0])
    source_counts = (400, 8000, 8000, 16000)
    batch = torch.full((4, 16000), 7.0)
    for index, num_samples in enumerate(source_counts):
        batch[index, :num_samples] = torch.randn(
            (num_samples,), generator=generator
        )

    batch_output = transform(
        batch,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    for index, num_samples in enumerate(source_counts):
        single_output = transform(
            batch[index : index + 1, :num_samples],
            sample_rate=16000,
        )
        target_samples = int(single_output["valid_samples"][0])
        assert int(batch_output["valid_samples"][index]) == target_samples
        assert torch.equal(
            batch_output["input_features"][index, :target_samples],
            single_output["input_features"][0],
        )
        assert torch.all(
            batch_output["input_features"][index, target_samples:] == 0
        )


@pytest.mark.parametrize("sample_rate", (44100, 22050))
def test_mixed_batch_matches_cropped_calls_at_foreign_rates(
    transform: Wav2Vec2WaveformTransform,
    sample_rate: int,
):
    generator = torch.Generator().manual_seed(sample_rate)
    source_counts = (
        round(0.1 * sample_rate),
        round(0.4 * sample_rate),
        round(0.4 * sample_rate),
        sample_rate,
    )
    valid_seconds = torch.tensor(
        [num_samples / sample_rate for num_samples in source_counts]
    )
    batch = torch.full((4, sample_rate), 6.0)
    for index, num_samples in enumerate(source_counts):
        batch[index, :num_samples] = torch.randn(
            (num_samples,), generator=generator
        )

    batch_output = transform(
        batch,
        sample_rate=sample_rate,
        valid_seconds=valid_seconds,
    )

    for index, num_samples in enumerate(source_counts):
        single_output = transform(
            batch[index : index + 1, :num_samples],
            sample_rate=sample_rate,
        )
        target_samples = int(single_output["valid_samples"][0])
        assert int(batch_output["valid_samples"][index]) == target_samples
        assert torch.equal(
            batch_output["input_features"][index, :target_samples],
            single_output["input_features"][0],
        )
        assert torch.all(
            batch_output["input_features"][index, target_samples:] == 0
        )


def test_nonzero_invalid_tail_does_not_change_output_at_native_rate(
    transform: Wav2Vec2WaveformTransform,
):
    generator = torch.Generator().manual_seed(11)
    clean = torch.zeros(2, 32000)
    clean[:, :16000] = torch.randn((2, 16000), generator=generator)
    dirty = clean.clone()
    dirty[:, 16000:] = 9.0
    valid_seconds = torch.tensor([1.0, 1.0])

    clean_output = transform(
        clean,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )
    dirty_output = transform(
        dirty,
        sample_rate=16000,
        valid_seconds=valid_seconds,
    )

    assert torch.equal(
        clean_output["input_features"],
        dirty_output["input_features"],
    )


def test_nonzero_invalid_tail_does_not_change_output_at_foreign_rate(
    transform: Wav2Vec2WaveformTransform,
):
    generator = torch.Generator().manual_seed(23)
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

    assert torch.equal(
        output_a["input_features"],
        output_b["input_features"],
    )


def test_all_zero_valid_region_produces_all_zero_output(
    transform: Wav2Vec2WaveformTransform,
):
    output = transform(torch.zeros(2, 800), sample_rate=16000)

    assert torch.all(output["input_features"] == 0)
    assert torch.isfinite(output["input_features"]).all()


def test_valid_samples_integer_fast_path_for_full_waveform():
    # float32 seconds cannot represent (2^24 + 1) / 16000 exactly; the
    # seconds round-trip would yield 2^24, while the integer ratio keeps
    # every sample.
    num_samples = 2**24 + 1
    transform = Wav2Vec2WaveformTransform(do_normalize=False)

    output = transform(torch.zeros(1, num_samples), sample_rate=16000)

    assert int(output["valid_samples"][0]) == num_samples
    assert output["input_features"].shape == (1, num_samples)


def test_valid_samples_fast_path_rounds_ties_to_even(
    transform: Wav2Vec2WaveformTransform,
):
    # 1601 samples at 32 kHz map to 800.5 target samples (ties to even
    # 800); 1603 map to 801.5 (ties to even 802).
    for num_samples, expected in ((1601, 800), (1603, 802)):
        output = transform(
            torch.randn(1, num_samples),
            sample_rate=32000,
        )
        assert int(output["valid_samples"][0]) == expected


def test_zero_source_samples_group_at_extremely_low_rate(
    transform: Wav2Vec2WaveformTransform,
):
    # At 10 Hz, valid_seconds=0.025 rounds the source sample count to 0
    # while the target count is exactly 400; the group materializes as an
    # all-zero waveform instead of crashing inside resample.
    output = transform(
        torch.randn(2, 5),
        sample_rate=10,
        valid_seconds=torch.tensor([0.025, 0.3]),
    )

    assert torch.equal(
        output["valid_samples"],
        torch.tensor([400, 4800]),
    )
    assert torch.all(output["input_features"][0] == 0)
    assert torch.isfinite(output["input_features"]).all()


def test_waveform_gradient_propagates(
    transform: Wav2Vec2WaveformTransform,
):
    waveform = torch.randn(2, 2205, requires_grad=True)
    output = transform(
        waveform,
        sample_rate=22050,
        valid_seconds=torch.tensor([0.05, 0.1]),
    )
    output["input_features"].square().mean().backward()

    assert waveform.grad is not None
    assert torch.isfinite(waveform.grad).all()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is not available.",
)
def test_cuda_device_and_output():
    transform = Wav2Vec2WaveformTransform().cuda()
    output = transform(torch.randn(1, 800), sample_rate=16000)

    assert transform.device.type == "cuda"
    assert output["input_features"].device.type == "cuda"
    assert output["valid_samples"].device.type == "cuda"
    assert output["valid_seconds"].device.type == "cuda"
