"""Batch Kaldi fbank frontend for BEATs."""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

from ..helpers.common import round_positive_ratio
from ..helpers.grouping import assemble_padded_groups, iter_length_groups
from .base import BaseTransform

_TARGET_SAMPLE_RATE = 16000
_NUM_MEL_BINS = 128
_WIN_LENGTH = 400
_HOP_LENGTH = 160
_PADDED_WINDOW_SIZE = 512
_LOW_FREQUENCY = 20.0
_PREEMPHASIS = 0.97
_WAVEFORM_SCALE = 2**15
_FBANK_MEAN = 15.41663
_FBANK_STD = 6.55582
_EPSILON = torch.finfo(torch.float32).eps
# The patch convolution of the paired encoder requires at least 16 fbank
# frames; under snip_edges framing this corresponds to
# 400 + 15 * 160 = 2800 16 kHz samples.
_MIN_TARGET_SAMPLES = 2800


class BeatsKaldiFbankTransform(BaseTransform):
    """Convert a variable-length waveform into normalized Kaldi fbank
    features for BEATs.

    Replicates the official ``BEATs.preprocess``: the waveform is multiplied
    by ``2**15`` and then passed through a 16 kHz, 128-mel, 25/10 ms, povey
    window, snip_edges kaldi fbank, followed by normalization via
    ``(x - 15.41663) / (2 * 6.55582)``. There is no fixed canvas and no
    duration limit; samples are grouped by unique valid length to remain
    batch-independent.
    """

    def __init__(self) -> None:
        """Construct a fixed BEATs frontend with no trainable parameters."""
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
            torch.hann_window(_WIN_LENGTH, periodic=False).pow(0.85),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Return the current transform's device."""
        return self.mel_weight.device

    def _extract_fbank(self, waveform: Tensor) -> Tensor:
        """Batch-compute normalized kaldi fbank features for a group of
        equal-length waveforms.

        Args:
            waveform: 16 kHz mono float32 waveform of shape ``[B, N]``,
                with ``N >= 2800``.

        Returns:
            Normalized fbank of shape ``[B, 1 + (N - 400) // 160, 128]``.
        """
        frames = (waveform * _WAVEFORM_SCALE).unfold(
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
        spectrum = torch.fft.rfft(frames, dim=-1).abs().pow(2.0)
        fbank = torch.log(
            torch.clamp_min(spectrum @ self.mel_weight.t(), _EPSILON)
        )
        return (fbank - _FBANK_MEAN) / (2.0 * _FBANK_STD)

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract BEATs features isolated by valid length.

        Args:
            waveform: Floating-point waveform of shape ``[B, N]`` or
                ``[B, C, N]``.
            sample_rate: Input sample rate shared across the batch.
            valid_seconds: Valid duration of each sample; ``None`` means
                the entire waveform.

        Returns:
            A tensor dict containing ``input_features``,
            ``valid_feature_frames``, and ``valid_seconds``.
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

        if not torch.all(target_valid_samples > 0):
            raise ValueError(
                "Valid audio must contain at least 1 sample after mapping "
                "to 16 kHz."
            )

        valid_feature_frames = (
            torch.div(
                torch.clamp(
                    target_valid_samples,
                    min=_MIN_TARGET_SAMPLES,
                )
                - _WIN_LENGTH,
                _HOP_LENGTH,
                rounding_mode="floor",
            )
            + 1
        )

        # Each group keeps only the valid prefix ([:, :source_length]);
        # anything outside the valid region is never read, so there is no
        # need to zero the whole batch with masked_fill.
        if waveform.ndim == 3:
            waveform = waveform.mean(dim=1)

        length_pairs = torch.stack(
            (source_valid_samples, target_valid_samples),
            dim=1,
        )
        grouped_features: list[tuple[Tensor, Tensor]] = []
        max_physical_frames = 0

        for (source_length, target_length), batch_indices in (
                iter_length_groups(length_pairs)):
            group_waveform = waveform.index_select(0, batch_indices)[
                :, :source_length
            ]

            if source_length == 0:
                # 0 valid samples (caused by rounding a very small
                # valid_seconds) cannot go through resample, so we
                # directly build a zero waveform of the target length
                # (same handling as CLAP).
                group_waveform = waveform.new_zeros(
                    (batch_indices.shape[0], target_length))
            elif sample_rate != _TARGET_SAMPLE_RATE:
                group_waveform = torchaudio.functional.resample(
                    group_waveform,
                    orig_freq=sample_rate,
                    new_freq=_TARGET_SAMPLE_RATE,
                )
            physical_length = max(target_length, _MIN_TARGET_SAMPLES)
            if group_waveform.shape[-1] > target_length:
                group_waveform = group_waveform[:, :target_length]
            if group_waveform.shape[-1] < physical_length:
                group_waveform = F.pad(
                    group_waveform,
                    (0, physical_length - group_waveform.shape[-1]),
                )

            features = self._extract_fbank(group_waveform)
            max_physical_frames = max(
                max_physical_frames,
                features.shape[1],
            )
            grouped_features.append((batch_indices, features))

        input_features = assemble_padded_groups(
            batch_size,
            [batch_indices for batch_indices, _ in grouped_features],
            [features for _, features in grouped_features],
            waveform,
            total_frames=max_physical_frames,
        )

        return {
            "input_features": input_features,
            "valid_feature_frames": valid_feature_frames,
            "valid_seconds": valid_seconds,
        }


__all__ = ("BeatsKaldiFbankTransform",)
