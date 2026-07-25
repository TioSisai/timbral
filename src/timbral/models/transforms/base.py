"""Abstract interface for audio transforms."""

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import Tensor, nn


class BaseTransform(nn.Module, ABC):
    """Abstract base class for all audio transforms.

    Concrete transforms are responsible for the full waveform preprocessing
    and feature extraction pipeline, including input validation, device
    transfer, valid-region handling, and model-specific transformations. The
    base class only defines the common interface; it does not prescribe
    resampling, channel conversion, padding, or maximum-input-length
    policies.

    Attributes:
        target_sample_rate: The target sample rate used for feature
            extraction, set by each concrete subclass at construction time;
            the concrete implementation is responsible for resampling when
            the input sample rate differs.
    """

    target_sample_rate: int

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device on which this transform receives input."""

    @abstractmethod
    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Convert a batch of waveforms into the input required by the encoder.

        Args:
            waveform: Floating-point waveform tensor of shape ``[B, N]`` or
                ``[B, C, N]``.
            sample_rate: Sample rate of the input waveform. Must be a
                positive Python ``int``, and the entire batch shares the
                same sample rate.
            valid_seconds: Valid duration of each sample, of shape ``[B]``,
                in seconds. ``None`` means the entire physical tensor of
                each sample is valid.
            **kwargs: Model-specific forward arguments. Concrete
                implementations must not silently ignore unknown arguments.

        Returns:
            A tensor dict containing at least ``input_features`` and
            ``valid_seconds``. ``valid_seconds`` must reside on the
            transform's device and use ``float32``. Concrete
            implementations may add model-specific tensors.

        Raises:
            TypeError: The input type or an unknown model-specific
                argument is invalid.
            ValueError: The input shape, sample rate, valid duration, or an
                inherent model constraint is invalid.

        Notes:
            Concrete implementations must automatically move ``waveform``,
            ``valid_seconds``, and top-level tensors in ``kwargs`` to
            :attr:`device`, and must ensure that padding outside the valid
            duration does not affect the output. Tensors nested inside
            containers are not covered by this public transfer contract.
        """
