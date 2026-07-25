"""Unit tests for the BaseEncoder abstract interface."""

import inspect
from typing import Any

import pytest
import torch
from torch import Tensor

from timbral.models.encoders import BaseEncoder, Granularity


class _DummyEncoder(BaseEncoder):
    """Minimal test Encoder implementing the final interface contract."""

    supported_granularities = frozenset(("clip", "frame"))

    def __init__(
        self,
        granularity: Granularity,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__(granularity)
        self.register_buffer(
            "_device_anchor",
            torch.empty(0, device=device),
            persistent=False,
        )
        self.called_granularity: Granularity | None = None

    @property
    def device(self) -> torch.device:
        return self._device_anchor.device

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        scale: Tensor | float = 1.0,
    ) -> dict[str, Tensor]:
        self.called_granularity = "clip"
        embedding = input_features.mean(dim=1) * scale
        geometry = torch.stack(
            (torch.zeros_like(valid_seconds), valid_seconds),
            dim=-1,
        ).to(torch.float32)
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": torch.ones(
                valid_seconds.shape,
                dtype=torch.bool,
                device=self.device,
            ),
            "model_output": embedding,
        }

    def _encode_frame(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        scale: Tensor | float = 1.0,
    ) -> dict[str, Tensor]:
        self.called_granularity = "frame"
        num_frames = input_features.shape[1]
        frame_step_seconds = 1.0
        valid_frames = torch.ceil(
            valid_seconds / frame_step_seconds
        ).to(torch.long)
        valid_frames = valid_frames.clamp(min=1, max=num_frames)
        frame_indices = torch.arange(num_frames, device=self.device)
        valid_mask = frame_indices.unsqueeze(0) < valid_frames.unsqueeze(1)

        starts = (
            frame_indices.unsqueeze(0)
            .expand(valid_seconds.shape[0], -1)
            * frame_step_seconds
        )
        ends = starts + frame_step_seconds
        ends = torch.minimum(ends, valid_seconds.unsqueeze(1))
        geometry = torch.stack((starts, ends), dim=-1).to(torch.float32)

        return {
            "embedding": input_features * valid_mask.unsqueeze(-1) * scale,
            "geometry": geometry * valid_mask.unsqueeze(-1),
            "valid_mask": valid_mask,
            "model_output": input_features,
        }


class _RecordingEncoder(BaseEncoder):
    """Test Encoder that records the base class's device-transfer results."""

    supported_granularities = frozenset(("clip",))

    def __init__(self) -> None:
        super().__init__("clip")
        self.received: dict[str, Any] = {}

    @property
    def device(self) -> torch.device:
        return torch.device("meta")

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        self.received = {
            "input_features": input_features,
            "valid_seconds": valid_seconds,
            **kwargs,
        }
        return {
            "embedding": input_features,
            "geometry": torch.empty(
                input_features.shape[0], 2, device=self.device
            ),
            "valid_mask": torch.ones(
                input_features.shape[0],
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def _encode_frame(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        raise AssertionError("A clip instance should not call the frame hook.")


def test_base_encoder_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseEncoder("clip")


def test_default_supported_granularities_is_empty():
    class DeviceOnlyEncoder(BaseEncoder):
        @property
        def device(self) -> torch.device:
            return torch.device("cpu")

    assert DeviceOnlyEncoder.supported_granularities == frozenset()
    with pytest.raises(ValueError, match="granularities: none"):
        DeviceOnlyEncoder("clip")


def test_clip_only_encoder_rejects_frame_during_initialization():
    class ClipOnlyEncoder(_RecordingEncoder):
        def __init__(self, granularity: Granularity) -> None:
            BaseEncoder.__init__(self, granularity)

    assert ClipOnlyEncoder.supported_granularities == frozenset(("clip",))
    with pytest.raises(ValueError, match="does not support 'frame'.*clip"):
        ClipOnlyEncoder("frame")


def test_public_subpackage_exports():
    from timbral.models.encoders.base import (
        BaseEncoder as DirectBaseEncoder,
        Granularity as DirectGranularity,
    )

    assert BaseEncoder is DirectBaseEncoder
    assert Granularity is DirectGranularity


@pytest.mark.parametrize("granularity", ["token", "", "CLIP"])
def test_invalid_granularity_raises(granularity):
    with pytest.raises(ValueError, match="granularity"):
        _DummyEncoder(granularity)  # type: ignore[arg-type]


def test_valid_seconds_is_keyword_only():
    parameter = inspect.signature(BaseEncoder.forward).parameters[
        "valid_seconds"
    ]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_clip_dispatch_and_output_contract():
    encoder = _DummyEncoder("clip")
    features = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    valid_seconds = torch.tensor([2.0, 4.0])

    output = encoder(features, valid_seconds=valid_seconds)

    assert encoder.called_granularity == "clip"
    assert output["embedding"].shape == (2, 3)
    assert output["geometry"].shape == (2, 2)
    assert output["geometry"].dtype == torch.float32
    assert output["valid_mask"].shape == (2,)
    assert output["valid_mask"].dtype == torch.bool
    assert output["valid_mask"].all()
    torch.testing.assert_close(
        output["geometry"],
        torch.tensor([[0.0, 2.0], [0.0, 4.0]]),
    )
    assert "model_output" in output


def test_frame_dispatch_geometry_mask_and_zero_padding():
    encoder = _DummyEncoder("frame")
    features = torch.ones(2, 4, 3)
    valid_seconds = torch.tensor([2.0, 4.0])

    output = encoder(features, valid_seconds=valid_seconds)

    assert encoder.called_granularity == "frame"
    assert output["embedding"].shape == (2, 4, 3)
    assert output["geometry"].shape == (2, 4, 2)
    assert output["geometry"].dtype == torch.float32
    assert output["valid_mask"].dtype == torch.bool
    torch.testing.assert_close(
        output["valid_mask"],
        torch.tensor([[True, True, False, False],
                      [True, True, True, True]]),
    )
    torch.testing.assert_close(output["embedding"][0, 2:], torch.zeros(2, 3))
    torch.testing.assert_close(output["geometry"][0, 2:], torch.zeros(2, 2))
    torch.testing.assert_close(
        output["geometry"][0, :2],
        torch.tensor([[0.0, 1.0], [1.0, 2.0]]),
    )
    torch.testing.assert_close(
        output["geometry"][1],
        torch.tensor([[0.0, 1.0], [1.0, 2.0],
                      [2.0, 3.0], [3.0, 4.0]]),
    )


def test_frame_geometry_uses_time_grid_and_truncates_last_frame():
    encoder = _DummyEncoder("frame")

    output = encoder(
        torch.ones(1, 4, 3),
        valid_seconds=torch.tensor([2.6]),
    )

    torch.testing.assert_close(
        output["geometry"],
        torch.tensor([[[0.0, 1.0],
                       [1.0, 2.0],
                       [2.0, 2.6],
                       [0.0, 0.0]]]),
    )
    torch.testing.assert_close(
        output["valid_mask"],
        torch.tensor([[True, True, True, False]]),
    )


def test_moves_top_level_tensors_without_recursing_or_casting():
    encoder = _RecordingEncoder()
    nested_tensor = torch.ones(1)
    nested = {"tensor": nested_tensor}

    output = encoder(
        torch.ones(1, 2),
        valid_seconds=torch.ones(1, dtype=torch.float64),
        auxiliary=torch.ones(1),
        nested=nested,
        label="example",
    )

    assert encoder.received["input_features"].device.type == "meta"
    assert encoder.received["valid_seconds"].device.type == "meta"
    assert encoder.received["valid_seconds"].dtype == torch.float64
    assert encoder.received["auxiliary"].device.type == "meta"
    assert encoder.received["nested"] is nested
    assert encoder.received["nested"]["tensor"] is nested_tensor
    assert encoder.received["label"] == "example"
    assert output["embedding"].device.type == "meta"


def test_unknown_model_parameter_raises():
    encoder = _DummyEncoder("clip")

    with pytest.raises(TypeError, match="unknown"):
        encoder(
            torch.ones(1, 2, 3),
            valid_seconds=torch.ones(1),
            unknown=True,
        )


def test_base_does_not_validate_concrete_output():
    class EmptyOutputEncoder(_DummyEncoder):
        def _encode_clip(
            self,
            input_features: Tensor,
            *,
            valid_seconds: Tensor,
            **kwargs: Any,
        ) -> dict[str, Tensor]:
            return {}

    encoder = EmptyOutputEncoder("clip")

    assert encoder(
        torch.ones(1, 2, 3),
        valid_seconds=torch.ones(1),
    ) == {}
