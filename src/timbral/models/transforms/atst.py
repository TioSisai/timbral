"""Batch log-mel frontend shared by ATST-Clip and ATST-Frame."""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F

from ..helpers.atst import (
    ATST_HOP_LENGTH,
    ATST_NORM_MAX,
    ATST_NORM_MIN,
    ATST_NUM_MELS,
    ATST_TARGET_SAMPLE_RATE,
    ATST_TOP_DB,
    atst_feature_frames,
)
from ..helpers.common import round_positive_ratio
from ..helpers.grouping import assemble_padded_groups, iter_length_groups
from .base import BaseTransform

_N_FFT = 1024
_WIN_LENGTH = 1024
_F_MIN = 60.0
_F_MAX = 7800.0
_AMPLITUDE_FLOOR = 1e-10
_POWER_MULTIPLIER = 10.0
# ``center=True`` reflect-pads by ``n_fft // 2`` on both sides, which
# torch.stft only accepts while the input is strictly longer than the
# padding; 513 samples is therefore the shortest waveform the official
# frontend can process at all, and it yields exactly 4 mel frames, i.e.
# the one patch the paired Encoder needs at minimum.
_MIN_TARGET_SAMPLES = _N_FFT // 2 + 1


class AtstMelspecTransform(BaseTransform):
    """Convert a variable-length waveform into ATST log-mel features.

    Replicates the official frontend shared by both families: a 16 kHz,
    64-mel, 1024/1024/160 ``MelSpectrogram`` (60-7800 Hz), then
    ``AmplitudeToDB(stype="power", top_db=80)``, then the fixed
    ``MinMax(-79.6482, 50.6842)`` rescaling to ``[-1, 1]``.

    The dB stage is written out rather than delegated to
    ``torchaudio.transforms.AmplitudeToDB`` because that module's
    ``top_db`` reduction range depends on the input rank: a ``[B, F, T]``
    input shares one peak across the whole batch, while ``[B, 1, F, T]``
    takes the peak per sample. Only the per-sample reduction keeps the
    output independent of batch composition, so it is spelled out here.

    Features are computed over each sample's whole valid region in one
    pass, never per chunk: the ``top_db`` floor is derived from the peak
    of the full valid region, and chunking it would make the floor
    depend on where the chunk boundaries happen to fall. Samples are
    grouped by unique valid length so that padding never enters any
    computation. There is no duration cap.
    """

    def __init__(self) -> None:
        """Construct the fixed ATST frontend with no trainable state."""
        super().__init__()
        self.target_sample_rate = ATST_TARGET_SAMPLE_RATE
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=ATST_TARGET_SAMPLE_RATE,
            n_fft=_N_FFT,
            win_length=_WIN_LENGTH,
            hop_length=ATST_HOP_LENGTH,
            f_min=_F_MIN,
            f_max=_F_MAX,
            n_mels=ATST_NUM_MELS,
        )

    @property
    def device(self) -> torch.device:
        """Return the current transform's device."""
        return self.melspec.mel_scale.fb.device

    def _extract_features(self, waveform: Tensor) -> Tensor:
        """Compute normalized log-mel features for equal-length waveforms.

        Args:
            waveform: 16 kHz mono float32 waveform of shape ``[B, N]``
                with ``N >= 513``.

        Returns:
            Features of shape ``[B, N // 160 + 1, 64]``, time first.
        """
        power = self.melspec(waveform)
        decibel = _POWER_MULTIPLIER * torch.log10(
            torch.clamp(power, min=_AMPLITUDE_FLOOR)
        )
        # Per-sample top_db floor: the peak is reduced over this
        # sample's mel and time axes only.
        peak = decibel.amax(dim=(-2, -1), keepdim=True)
        decibel = torch.maximum(decibel, peak - ATST_TOP_DB)
        features = (decibel - ATST_NORM_MIN) / (
            ATST_NORM_MAX - ATST_NORM_MIN
        ) * 2.0 - 1.0
        return features.transpose(1, 2)

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract ATST features isolated by valid length.

        Args:
            waveform: Floating-point waveform of shape ``[B, N]`` or
                ``[B, C, N]``.
            sample_rate: Input sample rate shared across the batch.
            valid_seconds: Valid duration of each sample; ``None`` means
                the entire waveform.

        Returns:
            A tensor dict containing ``input_features``,
            ``valid_feature_frames``, and ``valid_seconds``.

        Raises:
            TypeError: The waveform or sample rate has the wrong type.
            ValueError: The shape, sample rate, or valid duration is
                invalid.
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
                    num_samples * ATST_TARGET_SAMPLE_RATE, sample_rate),
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
                valid_seconds * ATST_TARGET_SAMPLE_RATE
            ).to(torch.long)

        if not torch.all(target_valid_samples > 0):
            raise ValueError(
                "Valid audio must contain at least 1 sample after mapping "
                "to 16 kHz."
            )

        valid_feature_frames = atst_feature_frames(
            torch.clamp(target_valid_samples, min=_MIN_TARGET_SAMPLES)
        )

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
                # directly build a zero waveform of the target length.
                group_waveform = waveform.new_zeros(
                    (batch_indices.shape[0], target_length))
            elif sample_rate != ATST_TARGET_SAMPLE_RATE:
                group_waveform = torchaudio.functional.resample(
                    group_waveform,
                    orig_freq=sample_rate,
                    new_freq=ATST_TARGET_SAMPLE_RATE,
                )
            physical_length = max(target_length, _MIN_TARGET_SAMPLES)
            if group_waveform.shape[-1] > target_length:
                group_waveform = group_waveform[:, :target_length]
            if group_waveform.shape[-1] < physical_length:
                group_waveform = F.pad(
                    group_waveform,
                    (0, physical_length - group_waveform.shape[-1]),
                )

            features = self._extract_features(group_waveform)
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


__all__ = ("AtstMelspecTransform",)
