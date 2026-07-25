"""Frozen Hugging Face CLAP HTSAT audio-tower clip encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import ClapAudioModelWithProjection

from ..helpers.clap import (
    ensure_clap_checkpoint,
    fixed_clap_audio_config,
    load_and_validate_clap_audio_config,
    load_clap_audio_state,
)
from ..helpers.geometry import build_clip_outputs
from .base import BaseEncoder, Granularity

_TARGET_SAMPLE_RATE = 48000
_FUSION_MIN_SAMPLES = 480480
_INPUT_CHANNELS = 4
_INPUT_FRAMES = 1001
_INPUT_MELS = 64
_PROJECTION_DIM = 512


class ClapHtsatEncoder(BaseEncoder):
    """Clip encoder wrapping the frozen CLAP HTSAT fused audio tower."""

    supported_granularities = frozenset(("clip",))
    embedding_dim = _PROJECTION_DIM

    def __init__(
        self,
        *,
        granularity: Granularity,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        """Construct the frozen CLAP audio tower and projection."""
        super().__init__(granularity)
        if type(pretrained) is not bool:
            raise TypeError("pretrained must be a bool.")

        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )
        if pretrained:
            checkpoint_dir = ensure_clap_checkpoint(pretrained_dir)
            config = load_and_validate_clap_audio_config(
                checkpoint_dir / "config.json",
                checkpoint_dir / "preprocessor_config.json",
            )
            self.backbone = ClapAudioModelWithProjection(config)
            audio_state = load_clap_audio_state(
                checkpoint_dir / "model.safetensors"
            )
            self.backbone.load_state_dict(audio_state, strict=True)
        else:
            self.backbone = ClapAudioModelWithProjection(
                fixed_clap_audio_config()
            )

    @property
    def device(self) -> torch.device:
        """Return the current encoder device."""
        return self.backbone.audio_projection.linear1.weight.device

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
    ) -> dict[str, Tensor]:
        """Produce the official CLAP contrastive-space clip embedding and
        project geometry.
        """
        batch_size = input_features.shape[0]
        expected_shape = (
            batch_size,
            _INPUT_CHANNELS,
            _INPUT_FRAMES,
            _INPUT_MELS,
        )
        if input_features.shape != expected_shape:
            raise ValueError(
                "input_features must have shape [B,4,1001,64]."
            )
        if input_features.dtype != torch.float32:
            raise TypeError("input_features must use float32.")
        if valid_seconds.shape != (batch_size,):
            raise ValueError("valid_seconds must have shape [B].")

        target_valid_samples = torch.round(
            valid_seconds * _TARGET_SAMPLE_RATE
        ).to(torch.long)
        fusion_mask = (
            target_valid_samples >= _FUSION_MIN_SAMPLES
        ).unsqueeze(1)
        outputs = self.backbone(
            input_features=input_features,
            is_longer=fusion_mask,
        )
        embedding = F.normalize(outputs.audio_embeds, dim=-1)
        geometry, valid_mask = build_clip_outputs(valid_seconds)
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": valid_mask,
        }


__all__ = ("ClapHtsatEncoder",)
