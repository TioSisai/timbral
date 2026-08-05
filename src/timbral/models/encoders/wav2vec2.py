"""Frozen Hugging Face Wav2Vec2Model clip/frame embedding encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from transformers import Wav2Vec2Model

from ..helpers.geometry import build_clip_outputs, build_frame_geometry
from ..helpers.grouping import (
    assemble_flat_groups,
    assemble_padded_groups,
    iter_length_groups,
)
from ..helpers.wav2vec2 import (
    WAV2VEC2_HOP_SAMPLES,
    ensure_wav2vec2_checkpoint,
    fixed_wav2vec2_config,
    load_and_validate_wav2vec2_config,
    load_wav2vec2_backbone_state,
    wav2vec2_feature_frames,
)
from .base import BaseEncoder, Granularity

_TARGET_SAMPLE_RATE = 16000
_HIDDEN_SIZE = 768
_FRAME_STEP_SECONDS = WAV2VEC2_HOP_SAMPLES / _TARGET_SAMPLE_RATE


class Wav2Vec2Encoder(BaseEncoder):
    """Embedding encoder wrapping the frozen wav2vec2-base backbone.

    The conv frontend, its group normalization, and the conv positional
    embedding all leak padding into valid positions, so the backbone runs
    per valid-length group on the exact valid prefix and the results are
    scattered back onto a zero canvas; padding therefore never influences
    any output. Batch outputs match per-sample calls bit-identically when
    a group holds a single sample, and up to floating-point kernel
    batching inside the Transformer layers when a group holds several
    same-length samples.

    Args:
        granularity: Output ``"clip"`` or ``"frame"``.
        pretrained: Whether to prepare and load the frozen official
            checkpoint.
        pretrained_dir: Explicit snapshot directory; uses the HF cache
            when ``None``.
    """

    supported_granularities = frozenset(("clip", "frame"))
    embedding_dim = _HIDDEN_SIZE

    def __init__(
        self,
        *,
        granularity: Granularity,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        """Construct the frozen wav2vec2 backbone."""
        super().__init__(granularity)
        if type(pretrained) is not bool:
            raise TypeError("pretrained must be a bool.")

        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )
        if pretrained:
            checkpoint_dir = ensure_wav2vec2_checkpoint(pretrained_dir)
            config = load_and_validate_wav2vec2_config(
                checkpoint_dir / "config.json"
            )
            self.backbone = Wav2Vec2Model(config)
            backbone_state = load_wav2vec2_backbone_state(
                checkpoint_dir / "pytorch_model.bin"
            )
            self.backbone.load_state_dict(backbone_state, strict=True)
        else:
            self.backbone = Wav2Vec2Model(fixed_wav2vec2_config())

    @property
    def device(self) -> torch.device:
        """Return the current encoder device."""
        return self.backbone.feature_projection.projection.weight.device

    def _backbone_hidden(self, input_features: Tensor) -> Tensor:
        """Encode the exact-length waveforms within a group into the last
        hidden state ``[B, T, 768]``.
        """
        return self.backbone(input_values=input_features).last_hidden_state

    @staticmethod
    def _validate_batch_cardinality(
        input_features: Tensor,
        valid_seconds: Tensor,
        valid_samples: Tensor,
    ) -> int:
        """Validate per-sample tensor shapes and return the batch size."""
        batch_size = input_features.shape[0]
        if valid_seconds.shape != (batch_size,):
            raise ValueError("valid_seconds must have shape [B].")
        if valid_samples.shape != (batch_size,):
            raise ValueError("valid_samples must have shape [B].")
        return batch_size

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        valid_samples: Tensor,
    ) -> dict[str, Tensor]:
        """Produce the valid-frame mean clip embedding, geometry, and
        mask.
        """
        batch_size = self._validate_batch_cardinality(
            input_features,
            valid_seconds,
            valid_samples,
        )
        batch_index_groups = []
        embedding_groups = []

        for sample_length, batch_indices in iter_length_groups(
                valid_samples):
            hidden = self._backbone_hidden(
                input_features.index_select(0, batch_indices)[
                    :, :sample_length
                ]
            )
            batch_index_groups.append(batch_indices)
            embedding_groups.append(hidden.mean(dim=1))

        embedding = assemble_flat_groups(
            batch_size, batch_index_groups, embedding_groups, input_features)
        geometry, valid_mask = build_clip_outputs(valid_seconds)
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": valid_mask,
        }

    def _encode_frame(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        valid_samples: Tensor,
    ) -> dict[str, Tensor]:
        """Produce the frame embedding, geometry, and mask."""
        batch_size = self._validate_batch_cardinality(
            input_features,
            valid_seconds,
            valid_samples,
        )
        valid_frames = wav2vec2_feature_frames(valid_samples)
        max_frames = int(valid_frames.max().item())
        batch_index_groups = []
        embedding_groups = []

        for sample_length, batch_indices in iter_length_groups(
                valid_samples):
            hidden = self._backbone_hidden(
                input_features.index_select(0, batch_indices)[
                    :, :sample_length
                ]
            )
            batch_index_groups.append(batch_indices)
            embedding_groups.append(hidden)

        # Grouped forward passes plus zero-canvas index_copy already
        # guarantee that invalid-frame embeddings are exactly 0, so
        # there is no need to multiply by valid_mask again.
        embedding = assemble_padded_groups(
            batch_size,
            batch_index_groups,
            embedding_groups,
            input_features,
            total_frames=max_frames,
        )
        geometry, valid_mask = build_frame_geometry(
            valid_frames,
            valid_seconds,
            total_frames=max_frames,
            step_seconds=_FRAME_STEP_SECONDS,
        )
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": valid_mask,
        }


__all__ = ("Wav2Vec2Encoder",)
