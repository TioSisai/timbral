"""Unit tests for the BaseTransform abstract interface."""

import inspect
from typing import Any

import pytest
import torch
from torch import Tensor

from timbral.models.transforms import BaseTransform


class _DummyTransform(BaseTransform):
    """Minimal test Transform implementing the final interface contract."""

    def __init__(self, device: torch.device | str = "cpu") -> None:
        super().__init__()
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, device=device),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
        scale: Tensor | float = 1.0,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported argument: {names}")
        if not isinstance(waveform, Tensor) or not waveform.is_floating_point():
            raise TypeError("waveform must be a floating-point Tensor.")
        if waveform.ndim not in (2, 3):
            raise ValueError("waveform must have shape [B, N] or [B, C, N].")
        if type(sample_rate) is not int:
            raise TypeError("sample_rate must be a Python int.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        waveform = waveform.to(device=self.device)
        batch_size, num_samples = waveform.shape[0], waveform.shape[-1]
        if valid_seconds is None:
            valid_seconds = torch.full(
                (batch_size,),
                num_samples / sample_rate,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            if not isinstance(valid_seconds, Tensor):
                raise TypeError("valid_seconds must be a Tensor or None.")
            valid_seconds = valid_seconds.to(
                device=self.device,
                dtype=torch.float32,
            )
            if valid_seconds.shape != (batch_size,):
                raise ValueError("valid_seconds must have shape [B].")
            max_seconds = num_samples / sample_rate
            valid_range = (valid_seconds > 0) & (
                valid_seconds <= max_seconds
            )
            if not torch.all(valid_range):
                raise ValueError("valid_seconds exceeds the physical waveform range.")

        if isinstance(scale, Tensor):
            scale = scale.to(device=self.device)
        else:
            scale = torch.tensor(scale, device=self.device)

        valid_samples = torch.round(valid_seconds * sample_rate).to(torch.long)
        sample_indices = torch.arange(num_samples, device=self.device)
        valid_mask = sample_indices.unsqueeze(0) < valid_samples.unsqueeze(1)
        if waveform.ndim == 3:
            valid_mask = valid_mask.unsqueeze(1)

        return {
            "input_features": waveform * valid_mask * scale,
            "valid_seconds": valid_seconds,
            "scale": scale,
        }


def test_base_transform_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseTransform()


def test_subclass_must_implement_all_abstract_members():
    class IncompleteTransform(BaseTransform):
        pass

    with pytest.raises(TypeError):
        IncompleteTransform()


def test_public_subpackage_export():
    from timbral.models.transforms.base import BaseTransform as DirectBaseTransform

    assert BaseTransform is DirectBaseTransform


def test_common_parameters_are_keyword_only():
    parameters = inspect.signature(BaseTransform.forward).parameters

    assert parameters["sample_rate"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["valid_seconds"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("shape", [(2, 4), (2, 3, 4)])
def test_accepts_mono_and_multichannel_waveform(shape):
    transform = _DummyTransform()
    waveform = torch.ones(shape)

    output = transform(waveform, sample_rate=4)

    assert output["input_features"].shape == shape
    torch.testing.assert_close(output["valid_seconds"], torch.ones(2))


def test_normalizes_valid_seconds_and_masks_padding():
    transform = _DummyTransform()
    waveform_a = torch.tensor([[1.0, 2.0, 30.0, 40.0],
                               [1.0, 2.0, 3.0, 4.0]])
    waveform_b = torch.tensor([[1.0, 2.0, -3.0, -4.0],
                               [1.0, 2.0, 3.0, 4.0]])
    valid_seconds = torch.tensor([0.5, 1.0], dtype=torch.float64)

    output_a = transform(
        waveform_a,
        sample_rate=4,
        valid_seconds=valid_seconds,
    )
    output_b = transform(
        waveform_b,
        sample_rate=4,
        valid_seconds=valid_seconds,
    )

    assert output_a["valid_seconds"].dtype == torch.float32
    torch.testing.assert_close(
        output_a["input_features"],
        torch.tensor([[1.0, 2.0, 0.0, 0.0],
                      [1.0, 2.0, 3.0, 4.0]]),
    )
    torch.testing.assert_close(
        output_a["input_features"],
        output_b["input_features"],
    )


def test_moves_top_level_model_tensor_and_allows_extra_output():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = _DummyTransform(device)

    output = transform(
        torch.ones(1, 4),
        sample_rate=4,
        scale=torch.tensor(2.0),
    )

    assert output["input_features"].device == transform.device
    assert output["valid_seconds"].device == transform.device
    assert output["scale"].device == transform.device
    torch.testing.assert_close(
        output["input_features"].cpu(),
        torch.full((1, 4), 2.0),
    )


@pytest.mark.parametrize(
    ("waveform", "sample_rate", "valid_seconds", "error"),
    [
        (torch.ones(4), 4, None, ValueError),
        (torch.ones(1, 4, dtype=torch.int16), 4, None, TypeError),
        (torch.ones(1, 4), 4.0, None, TypeError),
        (torch.ones(1, 4), 0, None, ValueError),
        (torch.ones(1, 4), 4, torch.ones(2), ValueError),
        (torch.ones(1, 4), 4, torch.tensor([0.0]), ValueError),
        (torch.ones(1, 4), 4, torch.tensor([1.1]), ValueError),
        (torch.ones(1, 4), 4, torch.tensor([float("nan")]), ValueError),
    ],
)
def test_dummy_transform_enforces_input_contract(
    waveform,
    sample_rate,
    valid_seconds,
    error,
):
    transform = _DummyTransform()

    with pytest.raises(error):
        transform(
            waveform,
            sample_rate=sample_rate,
            valid_seconds=valid_seconds,
        )


def test_unknown_model_parameter_raises():
    transform = _DummyTransform()

    with pytest.raises(TypeError, match="unknown"):
        transform(torch.ones(1, 4), sample_rate=4, unknown=True)
