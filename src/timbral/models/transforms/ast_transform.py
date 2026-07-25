"""Batch Kaldi fbank frontend for the fixed MIT AST model."""

from __future__ import annotations

import math

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

from .base import BaseTransform

_TARGET_SAMPLE_RATE = 16000
_NUM_MEL_BINS = 128
_WIN_LENGTH = 400
_HOP_LENGTH = 160
_PADDED_WINDOW_SIZE = 512
_LOW_FREQUENCY = 20.0
_PREEMPHASIS = 0.97
_MAX_LENGTH = 1024
_FEATURE_MEAN = -4.2677393
_FEATURE_STD = 4.5689974
_MAX_TARGET_SAMPLES = 164080
_MAX_VALID_SECONDS = 10.255


class AstKaldiFbankTransform(BaseTransform):
    """Convert a waveform into normalized fbank features for the fixed
    AST checkpoint.

    This frontend performs batched computation on a fixed 10.255-second
    canvas. Waveform padding outside the valid region is zeroed both before
    and after resampling, so it does not affect the output.
    """

    def __init__(self) -> None:
        """Construct a fixed AST frontend with no trainable parameters."""
        super().__init__()
        self.target_sample_rate = _TARGET_SAMPLE_RATE
        mel_weight, _ = torchaudio.compliance.kaldi.get_mel_banks(
            num_bins=_NUM_MEL_BINS,
            window_length_padded=_PADDED_WINDOW_SIZE,
            sample_freq=float(_TARGET_SAMPLE_RATE),
            low_freq=_LOW_FREQUENCY,
            high_freq=0.0,
            vtln_low=100.0,
            vtln_high=-500.0,
            vtln_warp_factor=1.0,
        )
        self.register_buffer(
            "mel_weight",
            F.pad(mel_weight, (0, 1), value=0.0),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(_WIN_LENGTH, periodic=False),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Return the current transform's device."""
        return self.mel_weight.device

    def _extract(
        self,
        waveform: Tensor,
        target_valid_samples: Tensor,
    ) -> Tensor:
        """Extract batched AST fbank features from the fixed 16 kHz canvas."""
        frames = waveform.unfold(
            -1,
            _WIN_LENGTH,
            _HOP_LENGTH,
        ).contiguous()
        frames = frames - frames.mean(dim=-1, keepdim=True)
        shifted = F.pad(frames, (1, 0), mode="replicate")[..., :-1]
        frames = (frames - _PREEMPHASIS * shifted) * self.window
        frames = F.pad(
            frames,
            (0, _PADDED_WINDOW_SIZE - _WIN_LENGTH),
        )
        spectrum = torch.fft.rfft(frames, dim=-1)
        power = spectrum.real.square() + spectrum.imag.square()
        mel = power @ self.mel_weight.t()
        fbank = torch.log(
            mel.clamp_min(torch.finfo(mel.dtype).eps)
        )

        num_valid_rows = (
            torch.div(
                target_valid_samples - _WIN_LENGTH,
                _HOP_LENGTH,
                rounding_mode="floor",
            )
            + 1
        ).clamp(min=0, max=_MAX_LENGTH)
        row_indices = torch.arange(
            _MAX_LENGTH,
            device=self.device,
        )
        padding_rows = (
            row_indices.unsqueeze(0) >= num_valid_rows.unsqueeze(1)
        )
        fbank = fbank.masked_fill(padding_rows.unsqueeze(2), 0.0)
        return (fbank - _FEATURE_MEAN) / (2.0 * _FEATURE_STD)

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract fixed AST features isolated by valid length.

        Args:
            waveform: Floating-point waveform of shape ``[B,N]`` or
                ``[B,C,N]``.
            sample_rate: Positive integer sample rate shared across the
                batch.
            valid_seconds: Valid duration of each sample; ``None`` means
                the entire waveform.

        Returns:
            A tensor dict containing ``input_features`` and
            ``valid_seconds``.

        Raises:
            TypeError: The input type does not comply with the public
                contract.
            ValueError: The input shape, sample rate, valid duration, or
                maximum length is invalid.
        """
        if not isinstance(waveform, Tensor) or not waveform.is_floating_point():
            raise TypeError("waveform must be a floating-point Tensor.")
        if waveform.ndim not in (2, 3):
            raise ValueError("waveform must have shape [B,N] or [B,C,N].")
        if type(sample_rate) is not int:
            raise TypeError("sample_rate must be a Python int.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        waveform = waveform.to(device=self.device, dtype=torch.float32)
        batch_size, num_samples = waveform.shape[0], waveform.shape[-1]
        physical_seconds = num_samples / sample_rate
        if valid_seconds is None:
            valid_seconds = torch.full(
                (batch_size,),
                physical_seconds,
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

        valid_range = (
            (valid_seconds > 0)
            & (valid_seconds <= physical_seconds)
            & (valid_seconds <= _MAX_VALID_SECONDS)
        )
        if not torch.all(valid_range):
            raise ValueError(
                "valid_seconds must be greater than 0, and must not "
                "exceed the physical waveform range or AST's 10.255-second "
                "upper limit."
            )

        source_valid_samples = torch.round(
            valid_seconds * sample_rate
        ).to(torch.long)
        target_valid_samples = torch.round(
            valid_seconds * _TARGET_SAMPLE_RATE
        ).to(torch.long)
        if not torch.all(target_valid_samples <= _MAX_TARGET_SAMPLES):
            raise ValueError(
                "Valid audio must not exceed 164080 16 kHz target samples."
            )

        max_source_samples = math.ceil(
            _MAX_VALID_SECONDS * sample_rate
        )
        waveform = waveform[..., :max_source_samples]
        source_indices = torch.arange(
            waveform.shape[-1],
            device=self.device,
        )
        source_mask = (
            source_indices.unsqueeze(0)
            < source_valid_samples.unsqueeze(1)
        )
        if waveform.ndim == 3:
            source_mask = source_mask.unsqueeze(1)
        waveform = waveform.masked_fill(~source_mask, 0.0)
        if waveform.ndim == 3:
            waveform = waveform.mean(dim=1)

        if sample_rate != _TARGET_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=_TARGET_SAMPLE_RATE,
            )
        waveform = waveform[..., :_MAX_TARGET_SAMPLES]
        waveform = F.pad(
            waveform,
            (0, _MAX_TARGET_SAMPLES - waveform.shape[-1]),
        )
        target_indices = torch.arange(
            _MAX_TARGET_SAMPLES,
            device=self.device,
        )
        target_mask = (
            target_indices.unsqueeze(0)
            < target_valid_samples.unsqueeze(1)
        )
        waveform = waveform.masked_fill(~target_mask, 0.0)

        return {
            "input_features": self._extract(
                waveform,
                target_valid_samples,
            ),
            "valid_seconds": valid_seconds,
        }


__all__ = ("AstKaldiFbankTransform",)
