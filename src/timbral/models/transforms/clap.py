"""Batch log-mel frontend for the fixed CLAP HTSAT fused checkpoint."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torchaudio
from torch import Tensor
from torch.nn import functional as F
from transformers.audio_utils import mel_filter_bank

from ..helpers.common import round_positive_ratio as _round_positive_ratio
from .base import BaseTransform

_TARGET_SAMPLE_RATE = 48000
_SHORT_CANVAS_SAMPLES = 480000
_FUSION_MIN_SAMPLES = 480480
_FFT_WINDOW_SIZE = 1024
_HOP_LENGTH = 480
_NUM_MELS = 64
_MIN_FREQUENCY = 50.0
_MAX_FREQUENCY = 14000.0
_OUTPUT_FRAMES = 1001


class ClapLogmelTransform(BaseTransform):
    """Convert a waveform into four-channel log-mel features for the
    fixed CLAP model.
    """

    def __init__(self) -> None:
        """Construct a fixed CLAP frontend with no trainable parameters."""
        super().__init__()
        self.target_sample_rate = _TARGET_SAMPLE_RATE
        filters = mel_filter_bank(
            num_frequency_bins=_FFT_WINDOW_SIZE // 2 + 1,
            num_mel_filters=_NUM_MELS,
            min_frequency=_MIN_FREQUENCY,
            max_frequency=_MAX_FREQUENCY,
            sampling_rate=_TARGET_SAMPLE_RATE,
            norm=None,
            mel_scale="htk",
        )
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(filters).to(torch.float64),
            persistent=False,
        )
        self.register_buffer(
            "window",
            torch.hann_window(
                _FFT_WINDOW_SIZE,
                periodic=True,
                dtype=torch.float64,
            ),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        """Return the current transform's device."""
        return self.mel_filters.device

    def _apply(
        self,
        fn: Callable[[Tensor], Tensor],
        recurse: bool = True,
    ) -> ClapLogmelTransform:
        """Apply the device transform while keeping the fixed frontend
        buffers as float64.
        """
        mel_filters = self.mel_filters
        window = self.window
        super()._apply(fn, recurse)
        self.mel_filters = mel_filters.to(
            device=self.mel_filters.device,
            dtype=torch.float64,
        )
        self.window = window.to(
            device=self.window.device,
            dtype=torch.float64,
        )
        return self

    def _repeatpad(self, waveform: Tensor) -> Tensor:
        """Pad to a fixed 10-second canvas following the official
        repeatpad rule.
        """
        num_samples = waveform.shape[1]
        if num_samples >= _SHORT_CANVAS_SAMPLES:
            return waveform
        num_repeats = _SHORT_CANVAS_SAMPLES // num_samples
        repeated = waveform.repeat(1, num_repeats)
        return F.pad(
            repeated,
            (0, _SHORT_CANVAS_SAMPLES - repeated.shape[1]),
        )

    def _extract_logmel(self, waveform: Tensor) -> Tensor:
        """Extract `[B,T,64]` float32 log-mel features."""
        waveform = waveform.to(torch.float64)
        half_window = _FFT_WINDOW_SIZE // 2
        padded = F.pad(
            waveform,
            (half_window, half_window),
            mode="reflect",
        )
        spectrum = torch.stft(
            padded,
            n_fft=_FFT_WINDOW_SIZE,
            hop_length=_HOP_LENGTH,
            win_length=_FFT_WINDOW_SIZE,
            window=self.window,
            center=False,
            return_complex=True,
        )
        power = spectrum.real.square() + spectrum.imag.square()
        mel = torch.matmul(
            self.mel_filters.t(),
            power,
        ).clamp_min(1e-10)
        return (10.0 * torch.log10(mel)).transpose(1, 2).float()

    @staticmethod
    def _anchored_crop_starts(total_frames: int) -> tuple[int, int, int]:
        """Generate three crop starts from anchors at 1/6, 1/2, and 5/6
        along the frame axis.
        """
        max_start = total_frames - _OUTPUT_FRAMES
        half_span = (_OUTPUT_FRAMES - 1) // 2
        starts: list[int] = []
        for anchor_index in range(3):
            center = _round_positive_ratio(
                (2 * anchor_index + 1) * total_frames,
                6,
            )
            starts.append(
                min(max(center - half_span, 0), max_start)
            )
        return starts[0], starts[1], starts[2]

    def _extract_group(
        self,
        waveform: Tensor,
        target_length: int,
    ) -> Tensor:
        """Extract four-channel CLAP features for one group of equal
        valid length.
        """
        if target_length < _FUSION_MIN_SAMPLES:
            log_mel = self._extract_logmel(
                self._repeatpad(waveform)
            )
            return (
                log_mel.unsqueeze(1)
                .expand(-1, 4, -1, -1)
                .contiguous()
            )

        log_mel = self._extract_logmel(waveform)
        global_mel = F.interpolate(
            log_mel.unsqueeze(1),
            size=(_OUTPUT_FRAMES, _NUM_MELS),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        crop_start_tensor = torch.tensor(
            self._anchored_crop_starts(
                target_length // _HOP_LENGTH + 1
            ),
            dtype=torch.long,
            device=log_mel.device,
        )
        crop_indices = crop_start_tensor.unsqueeze(-1) + torch.arange(
            _OUTPUT_FRAMES,
            device=log_mel.device,
        )
        local_mels = log_mel[:, crop_indices]
        return torch.cat((global_mel.unsqueeze(1), local_mels), dim=1)

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract CLAP features routed by per-sample valid length.

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
            ValueError: The input shape, sample rate, or valid duration is
                invalid.
        """
        if not isinstance(waveform, Tensor) or not waveform.is_floating_point():
            raise TypeError("waveform must be a floating-point Tensor.")
        if waveform.ndim not in (2, 3):
            raise ValueError("waveform must have shape [B,N] or [B,C,N].")
        if type(sample_rate) is not int:
            raise TypeError("sample_rate must be a Python int.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")

        batch_size, physical_samples = (
            waveform.shape[0],
            waveform.shape[-1],
        )
        physical_seconds = physical_samples / sample_rate
        if valid_seconds is None:
            valid_seconds = torch.full(
                (batch_size,),
                physical_seconds,
                dtype=torch.float32,
                device=self.device,
            )
            source_valid_samples = torch.full(
                (batch_size,),
                physical_samples,
                dtype=torch.long,
                device=self.device,
            )
            target_valid_length = _round_positive_ratio(
                physical_samples * _TARGET_SAMPLE_RATE,
                sample_rate,
            )
            target_valid_samples = torch.full(
                (batch_size,),
                target_valid_length,
                dtype=torch.long,
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
            source_valid_samples = torch.round(
                valid_seconds * sample_rate
            ).to(torch.long)
            target_valid_samples = torch.round(
                valid_seconds * _TARGET_SAMPLE_RATE
            ).to(torch.long)

        valid_range = (
            (valid_seconds > 0)
            & (valid_seconds <= physical_seconds)
        )
        if not torch.all(valid_range):
            raise ValueError(
                "valid_seconds must be greater than 0, and must not "
                "exceed the physical waveform range."
            )

        if not torch.all(target_valid_samples > 0):
            raise ValueError(
                "Valid audio must contain at least 1 sample after mapping "
                "to 48 kHz."
            )

        length_pairs = torch.stack(
            (source_valid_samples, target_valid_samples),
            dim=1,
        ).tolist()

        grouped_sample_indices: dict[
            tuple[int, int],
            list[int],
        ] = {}
        for sample_index, lengths in enumerate(
            length_pairs
        ):
            grouped_sample_indices.setdefault(
                (lengths[0], lengths[1]),
                [],
            ).append(
                sample_index
            )

        feature_groups = []
        index_groups = []
        for (
            source_length,
            target_length,
        ), sample_indices in grouped_sample_indices.items():
            source_group_indices = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=waveform.device,
            )
            group_indices = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=self.device,
            )
            group_waveform = waveform.index_select(
                0,
                source_group_indices,
            )[..., :source_length].to(
                device=self.device,
                dtype=torch.float64,
            )
            if waveform.ndim == 3:
                group_waveform = group_waveform.mean(dim=1)
            if source_length == 0:
                group_waveform = torch.zeros(
                    (len(sample_indices), target_length),
                    dtype=torch.float64,
                    device=self.device,
                )
            elif sample_rate != _TARGET_SAMPLE_RATE:
                group_waveform = torchaudio.functional.resample(
                    group_waveform,
                    orig_freq=sample_rate,
                    new_freq=_TARGET_SAMPLE_RATE,
                )
            if group_waveform.shape[1] < target_length:
                group_waveform = F.pad(
                    group_waveform,
                    (0, target_length - group_waveform.shape[1]),
                )
            else:
                group_waveform = group_waveform[:, :target_length]
            feature_groups.append(
                self._extract_group(
                    group_waveform,
                    int(target_length),
                )
            )
            index_groups.append(group_indices)

        grouped_features = torch.cat(feature_groups, dim=0)
        grouped_indices = torch.cat(index_groups, dim=0)
        input_features = grouped_features[
            torch.argsort(grouped_indices)
        ]
        return {
            "input_features": input_features,
            "valid_seconds": valid_seconds,
        }


__all__ = ("ClapLogmelTransform",)
