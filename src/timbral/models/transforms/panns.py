"""Log-mel frontend for PANNs Cnn14."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np
import torch
import torchaudio
from torch import Tensor, nn
from torch.nn import functional as F

from ..helpers.common import round_positive_ratio
from ..helpers.grouping import assemble_padded_groups, iter_length_groups
from ..helpers.panns import (
    PANNS_CHECKPOINTS,
    PANNS_OFFICIAL_FRONTENDS,
    PANNS_TARGET_SAMPLE_RATES,
    PANNS_VARIANTS,
    PannsTargetSampleRate,
    PannsVariant,
    ensure_panns_checkpoint,
    load_panns_checkpoint_model,
)
from .base import BaseTransform

_MIN_ENCODER_MELS = 32
_MIN_ENCODER_FEATURE_FRAMES = 32
_SILENCE_LOGMEL_DB = -100.0

_FRONTEND_KEY_MAP = {
    "spectrogram_extractor.stft.conv_real.weight": "stft_conv_real",
    "spectrogram_extractor.stft.conv_imag.weight": "stft_conv_imag",
    "logmel_extractor.melW": "mel_weight",
    "bn0.weight": "bn0.weight",
    "bn0.bias": "bn0.bias",
    "bn0.running_mean": "bn0.running_mean",
    "bn0.running_var": "bn0.running_var",
    "bn0.num_batches_tracked": "bn0.num_batches_tracked",
}


class PannsLogmelTransform(BaseTransform):
    """Convert a variable-length waveform into PANNs post-BN log-mel features.

    Args:
        target_sample_rate: Target sample rate for the frontend; only
            16 kHz or 32 kHz is supported.
        n_fft: Number of FFT points.
        win_length: Length of the Hann window.
        hop_length: Sample-point interval between adjacent feature frames.
        n_mels: Number of mel bands, at least 32.
        f_min: Lower bound frequency of the mel scale.
        f_max: Upper bound frequency of the mel scale.
        variant: PANNs pooling/weight variant.
        pretrained: Whether to download and load the official frontend
            state.
        pretrained_dir: Directory where the checkpoint resides. When
            ``None``, the HF cache is used.
    """

    def __init__(
        self,
        *,
        target_sample_rate: PannsTargetSampleRate,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        variant: PannsVariant,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self._validate_configuration(
            target_sample_rate=target_sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            variant=variant,
            pretrained=pretrained,
        )

        self.target_sample_rate = target_sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = float(f_min)
        self.f_max = float(f_max)
        self.variant = variant
        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )

        fft_window = librosa.filters.get_window(
            "hann",
            win_length,
            fftbins=True,
        )
        fft_window = librosa.util.pad_center(fft_window, size=n_fft)
        grid_x, grid_y = np.meshgrid(np.arange(n_fft), np.arange(n_fft))
        dft = np.power(
            np.exp(-2.0 * np.pi * 1j / n_fft),
            grid_x * grid_y,
        )
        num_bins = n_fft // 2 + 1
        real_weight = np.real(
            dft[:, :num_bins] * fft_window[:, None]
        ).T
        imag_weight = np.imag(
            dft[:, :num_bins] * fft_window[:, None]
        ).T
        self.register_buffer(
            "stft_conv_real",
            torch.from_numpy(real_weight).float().unsqueeze(1),
        )
        self.register_buffer(
            "stft_conv_imag",
            torch.from_numpy(imag_weight).float().unsqueeze(1),
        )

        mel_weight = librosa.filters.mel(
            sr=target_sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max,
        ).T
        self.register_buffer(
            "mel_weight",
            torch.from_numpy(mel_weight.astype(np.float32)),
        )
        self.bn0 = nn.BatchNorm2d(n_mels)

        if pretrained:
            metadata = PANNS_CHECKPOINTS[(target_sample_rate, variant)]
            checkpoint_path = ensure_panns_checkpoint(
                metadata,
                pretrained_dir,
            )
            checkpoint_state = load_panns_checkpoint_model(
                checkpoint_path,
                requires_numpy_allowlist=metadata.requires_numpy_allowlist,
            )
            frontend_state = {
                local_key: checkpoint_state[checkpoint_key]
                for checkpoint_key, local_key in _FRONTEND_KEY_MAP.items()
            }
            self.load_state_dict(frontend_state, strict=True)

    @staticmethod
    def _validate_configuration(
        *,
        target_sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float,
        variant: str,
        pretrained: bool,
    ) -> None:
        """Validate the constructor configuration."""
        if target_sample_rate not in PANNS_TARGET_SAMPLE_RATES:
            raise ValueError("target_sample_rate only supports 16000 or 32000.")
        if variant not in PANNS_VARIANTS:
            raise ValueError(
                "variant only supports 'max_mean' or 'decision_level_max'."
            )
        if type(pretrained) is not bool:
            raise TypeError("pretrained must be a bool.")
        if not all(
            type(value) is int and value > 0
            for value in (n_fft, win_length, hop_length, n_mels)
        ):
            raise ValueError(
                "n_fft, win_length, hop_length, and n_mels must be "
                "positive integers."
            )
        if win_length > n_fft:
            raise ValueError("win_length must not be greater than n_fft.")
        if n_fft % 2 != 0:
            raise ValueError("n_fft must be even.")
        if n_mels < _MIN_ENCODER_MELS:
            raise ValueError("n_mels must be at least 32.")
        if not 0 <= f_min < f_max <= target_sample_rate / 2:
            raise ValueError(
                "The mel frequency range must satisfy "
                "0 <= f_min < f_max <= Nyquist."
            )

        if not pretrained:
            return
        if (target_sample_rate, variant) not in PANNS_CHECKPOINTS:
            raise ValueError(
                "No official weights exist for this sample rate and "
                "variant combination when pretrained=True."
            )
        expected = PANNS_OFFICIAL_FRONTENDS[target_sample_rate]
        actual = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "f_min": float(f_min),
            "f_max": float(f_max),
        }
        if actual != expected:
            raise ValueError(
                "When pretrained=True, the frontend parameters must "
                "exactly match the official checkpoint."
            )

    @property
    def device(self) -> torch.device:
        """Return the current transform's device."""
        return self.stft_conv_real.device

    def _power_spectrogram(self, waveform: Tensor) -> Tensor:
        """Compute the conv1d STFT power spectrogram.

        When the waveform is shorter than or equal to ``n_fft // 2``,
        reflect padding is undefined and falls back to zero padding, i.e.
        padding under the local semantics of "silence outside the valid
        audio"; both modes use the same padding width, and the output
        frame count is the same in both cases: ``floor(N / hop_length) + 1``.
        """
        pad_width = self.n_fft // 2
        audio = F.pad(
            waveform.unsqueeze(1),
            (pad_width, pad_width),
            mode="reflect" if waveform.shape[-1] > pad_width else "constant",
        )
        real = F.conv1d(
            audio,
            self.stft_conv_real,
            stride=self.hop_length,
        )
        imag = F.conv1d(
            audio,
            self.stft_conv_imag,
            stride=self.hop_length,
        )
        return (real.square() + imag.square()).transpose(1, 2)

    def _logmel(self, waveform: Tensor) -> Tensor:
        """Compute the log-mel features prior to bn0."""
        mel = torch.clamp(
            self._power_spectrogram(waveform) @ self.mel_weight,
            min=1e-10,
        )
        return 10.0 * torch.log10(mel)

    def forward(
        self,
        waveform: Tensor,
        *,
        sample_rate: int,
        valid_seconds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Extract PANNs features isolated by valid length.

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
                    num_samples * self.target_sample_rate, sample_rate),
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
                valid_seconds * self.target_sample_rate
            ).to(torch.long)

        if not torch.all(target_valid_samples > 0):
            raise ValueError(
                f"Valid audio must contain at least 1 sample after "
                f"mapping to {self.target_sample_rate} Hz."
            )

        valid_feature_frames = (
            torch.div(
                target_valid_samples,
                self.hop_length,
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
            elif sample_rate != self.target_sample_rate:
                group_waveform = torchaudio.functional.resample(
                    group_waveform,
                    orig_freq=sample_rate,
                    new_freq=self.target_sample_rate,
                )
            if group_waveform.shape[-1] > target_length:
                group_waveform = group_waveform[:, :target_length]
            elif group_waveform.shape[-1] < target_length:
                group_waveform = F.pad(
                    group_waveform,
                    (0, target_length - group_waveform.shape[-1]),
                )

            logmel = self._logmel(group_waveform)
            native_feature_frames = target_length // self.hop_length + 1
            if native_feature_frames < _MIN_ENCODER_FEATURE_FRAMES:
                logmel = F.pad(
                    logmel,
                    (
                        0,
                        0,
                        0,
                        _MIN_ENCODER_FEATURE_FRAMES
                        - native_feature_frames,
                    ),
                    value=_SILENCE_LOGMEL_DB,
                )
            features = (
                self.bn0(logmel.transpose(1, 2).unsqueeze(-1))
                .squeeze(-1)
                .transpose(1, 2)
            )
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


__all__ = ("PannsLogmelTransform", "PannsVariant")
