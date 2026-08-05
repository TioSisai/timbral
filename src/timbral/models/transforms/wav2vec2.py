"""Waveform passthrough frontend for the fixed wav2vec2 checkpoint."""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

from ..helpers.common import round_positive_ratio
from ..helpers.grouping import assemble_padded_groups, iter_length_groups
from ..helpers.wav2vec2 import WAV2VEC2_MIN_TARGET_SAMPLES
from .base import BaseTransform

_TARGET_SAMPLE_RATE = 16000
_NORMALIZATION_EPSILON = 1e-7


class Wav2Vec2WaveformTransform(BaseTransform):
    """Convert a waveform into the raw 16 kHz input expected by the
    wav2vec2 encoder.

    Unlike the spectrogram frontends, ``input_features`` is the waveform
    itself, shaped ``[B, N]``: downmixed, resampled to 16 kHz, and (per
    the official preprocessor) optionally normalized per sample to zero
    mean and unit variance over the valid region. Resampling runs per
    (source length, target length) group on the exact valid prefix, and the
    results are scattered back onto a zero canvas, so padding outside the
    valid region never influences the output.

    Args:
        do_normalize: Whether to apply the official per-sample zero-mean
            unit-variance normalization over the valid region. The fixed
            ``facebook/wav2vec2-base`` preprocessor enables it; large
            wav2vec2 variants may fix a different value at registration
            time.
    """

    def __init__(self, *, do_normalize: bool = True) -> None:
        """Construct the wav2vec2 frontend with no trainable parameters."""
        super().__init__()
        if type(do_normalize) is not bool:
            raise TypeError("do_normalize must be a bool.")
        self.do_normalize = do_normalize
        self.target_sample_rate = _TARGET_SAMPLE_RATE
        self.register_buffer(
            "device_anchor",
            torch.empty(0),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Return the current transform's device."""
        return self.device_anchor.device

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Produce the 16 kHz waveform canvas isolated by valid length.

        Args:
            waveform: Floating-point waveform of shape ``[B, N]`` or
                ``[B, C, N]``.
            sample_rate: Positive integer sample rate shared across the
                batch.
            valid_seconds: Valid duration of each sample; ``None`` means
                the entire waveform.

        Returns:
            A tensor dict containing ``input_features`` (``[B, N16k]``
            float32 waveform), ``valid_samples`` (``[B]`` int64 valid
            16 kHz sample counts), and ``valid_seconds``.

        Raises:
            TypeError: The input type does not comply with the public
                contract.
            ValueError: The input shape, sample rate, or valid duration
                is invalid, or a sample is shorter than the 400-sample
                (0.025 s) conv receptive field.
        """
        if not isinstance(waveform, Tensor) or not waveform.is_floating_point():
            raise TypeError("waveform must be a floating-point Tensor.")
        if waveform.ndim not in (2, 3):
            raise ValueError("waveform must have shape [B, N] or [B, C, N].")
        if type(sample_rate) is not int:
            raise TypeError("sample_rate must be a Python int.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        waveform = waveform.to(device=self.device, dtype=torch.float32)
        batch_size, num_samples = waveform.shape[0], waveform.shape[-1]
        if valid_seconds is None:
            valid_seconds = torch.full(
                (batch_size,),
                num_samples / sample_rate,
                dtype=torch.float32,
                device=self.device,
            )
            # The entire waveform is valid: the sample count is converted
            # directly via integer ratio to avoid losing samples from
            # float32-second round-trips on very long waveforms
            # (> 2^24 samples).
            source_valid_samples = torch.full(
                (batch_size,), num_samples,
                dtype=torch.long, device=self.device,
            )
            target_valid_samples = torch.full(
                (batch_size,),
                round_positive_ratio(
                    num_samples * _TARGET_SAMPLE_RATE, sample_rate),
                dtype=torch.long, device=self.device,
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
            valid_range = (valid_seconds > 0) & (
                valid_seconds <= num_samples / sample_rate
            )
            if not torch.all(valid_range):
                raise ValueError(
                    "valid_seconds exceeds the physical waveform range."
                )
            source_valid_samples = torch.round(
                valid_seconds * sample_rate
            ).to(torch.long)
            target_valid_samples = torch.round(
                valid_seconds * _TARGET_SAMPLE_RATE
            ).to(torch.long)

        if not torch.all(
            target_valid_samples >= WAV2VEC2_MIN_TARGET_SAMPLES
        ):
            raise ValueError(
                "Valid audio must map to at least 400 samples (0.025 s) "
                "at 16 kHz to cover one wav2vec2 conv receptive field."
            )

        # Each group forwards only the valid prefix ([:, :source_length]);
        # anything outside it cannot influence the result, so there is no
        # need to zero the whole batch with masked_fill.
        if waveform.ndim == 3:
            waveform = waveform.mean(dim=1)

        length_pairs = torch.stack(
            (source_valid_samples, target_valid_samples),
            dim=1,
        )
        batch_index_groups: list[Tensor] = []
        waveform_groups: list[Tensor] = []
        max_target_samples = 0

        for (source_length, target_length), batch_indices in (
                iter_length_groups(length_pairs)):
            group_waveform = waveform.index_select(0, batch_indices)[
                :, :source_length
            ]

            if source_length == 0:
                # 0 valid samples (caused by rounding a very small
                # valid_seconds at a very low source rate) cannot go
                # through resample, so we directly build a zero waveform
                # of the target length (same handling as BEATs).
                group_waveform = waveform.new_zeros(
                    (batch_indices.shape[0], target_length))
            elif sample_rate != _TARGET_SAMPLE_RATE:
                group_waveform = torchaudio.functional.resample(
                    group_waveform,
                    orig_freq=sample_rate,
                    new_freq=_TARGET_SAMPLE_RATE,
                )
            if group_waveform.shape[-1] > target_length:
                group_waveform = group_waveform[:, :target_length]
            if group_waveform.shape[-1] < target_length:
                group_waveform = F.pad(
                    group_waveform,
                    (0, target_length - group_waveform.shape[-1]),
                )

            # Normalization must run on the exact-length group rows, not
            # on the assembled canvas: float reductions depend on the
            # reduced width, so canvas-level statistics would break the
            # bit-identity between mixed batches and per-sample calls.
            if self.do_normalize:
                mean = group_waveform.mean(dim=1, keepdim=True)
                centered = group_waveform - mean
                variance = centered.square().mean(dim=1, keepdim=True)
                group_waveform = centered / torch.sqrt(
                    variance + _NORMALIZATION_EPSILON
                )

            max_target_samples = max(max_target_samples, target_length)
            batch_index_groups.append(batch_indices)
            waveform_groups.append(group_waveform)

        input_features = assemble_padded_groups(
            batch_size,
            batch_index_groups,
            [group.unsqueeze(-1) for group in waveform_groups],
            waveform,
            total_frames=max_target_samples,
        ).squeeze(-1)

        return {
            "input_features": input_features,
            "valid_samples": target_valid_samples,
            "valid_seconds": valid_seconds,
        }


__all__ = ("Wav2Vec2WaveformTransform",)
