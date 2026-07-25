"""Abstract audio encoder interface."""

from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, TypeAlias

import torch
from torch import Tensor, nn

Granularity: TypeAlias = Literal["clip", "frame"]
_GRANULARITIES = frozenset(("clip", "frame"))


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all audio encoders.

    The base class is only responsible for fixing the output granularity,
    automatically moving top-level tensors to the encoder's device, and
    dispatching the forward pass to the concrete model's clip or frame
    encoding hook. Transform-encoder pairing, output construction, and
    model-specific constraints are the responsibility of the caller or the
    concrete implementation.

    Args:
        granularity: Output granularity, only ``"clip"`` or ``"frame"`` are
            supported.

    Raises:
        ValueError: If ``granularity`` is not a supported value.

    Attributes:
        supported_granularities: The set of output granularities supported
            by the concrete encoder.
        embedding_dim: The embedding dimension ``D`` of the concrete
            encoder, declared by each concrete class as a ClassVar.
    """

    supported_granularities: ClassVar[
        frozenset[Granularity]
    ] = frozenset()
    embedding_dim: ClassVar[int]

    def __init__(self, granularity: Granularity) -> None:
        """Initialize the encoder and fix the output granularity."""
        super().__init__()
        if granularity not in _GRANULARITIES:
            raise ValueError(
                "granularity only supports 'clip' or 'frame', "
                f"but received {granularity!r}."
            )
        if granularity not in self.supported_granularities:
            supported = ", ".join(
                sorted(self.supported_granularities)
            ) or "none"
            raise ValueError(
                f"{type(self).__name__} does not support {granularity!r} "
                f"granularity; supported granularities: {supported}."
            )
        self.granularity: Granularity = granularity

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device on which this encoder receives inputs."""

    def forward(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Move inputs to the encoder's device and dispatch to the concrete
        encoding hook based on the instance's granularity.

        Args:
            input_features: The primary feature tensor produced by the
                transform.
            valid_seconds: The valid duration of each sample, shaped
                ``[B]``, in seconds. The input should use ``float32``; the
                base class only moves the device and does not convert
                dtype.
            **kwargs: Model-specific inputs. Values that are top-level
                tensors are automatically moved to the device by the base
                class; nested containers are handled by the concrete
                encoder.

        Returns:
            A dict of tensors containing at least ``embedding``,
            ``geometry``, and ``valid_mask``. Concrete encoders may add
            model-specific outputs.
        """
        device = self.device
        input_features = input_features.to(device=device)
        valid_seconds = valid_seconds.to(device=device)
        model_inputs = {
            name: value.to(device=device) if isinstance(value, Tensor) else value
            for name, value in kwargs.items()
        }

        if self.granularity == "clip":
            return self._encode_clip(
                input_features,
                valid_seconds=valid_seconds,
                **model_inputs,
            )
        return self._encode_frame(
            input_features,
            valid_seconds=valid_seconds,
            **model_inputs,
        )

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Produce clip-granularity outputs.

        The return value contains at least:

        - ``embedding``: shaped ``[B, D]``;
        - ``geometry``: shaped ``[B, 2]``, in seconds, dtype ``float32``,
          with each row equal to ``[0, valid_seconds]``;
        - ``valid_mask``: shaped ``[B]``, dtype ``bool``, all ``True``.

        Args:
            input_features: The primary feature tensor already moved to
                the encoder's device.
            valid_seconds: The valid duration tensor already moved to the
                encoder's device.
            **kwargs: Model-specific inputs whose top-level tensors have
                already been moved to the device.

        Returns:
            The complete clip output dict.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the clip encoding "
            "hook."
        )

    def _encode_frame(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        """Produce frame-granularity outputs.

        The return value contains at least:

        - ``embedding``: shaped ``[B, T, D]``;
        - ``geometry``: shaped ``[B, T, 2]``, in seconds, dtype
          ``float32``;
        - ``valid_mask``: shaped ``[B, T]``, dtype ``bool``.

        Valid geometry must construct non-overlapping ownership intervals
        along the model's time grid, covering from 0 to
        ``valid_seconds``. The embedding and geometry of invalid frames
        must be filled with 0, and the corresponding mask must be
        ``False``.

        Args:
            input_features: The primary feature tensor already moved to
                the encoder's device.
            valid_seconds: The valid duration tensor already moved to the
                encoder's device.
            **kwargs: Model-specific inputs whose top-level tensors have
                already been moved to the device.

        Returns:
            The complete frame output dict.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement the frame encoding "
            "hook."
        )
