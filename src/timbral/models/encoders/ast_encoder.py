"""Frozen Hugging Face ASTModel clip/frame embedding encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor
from transformers import ASTModel

from ..helpers.ast_helpers import (
    ensure_ast_checkpoint,
    fixed_ast_config,
    load_and_validate_ast_config,
    load_ast_backbone_state,
)
from ..helpers.geometry import build_clip_outputs, build_frame_geometry
from .base import BaseEncoder, Granularity

_TARGET_SAMPLE_RATE = 16000
_FREQUENCY_OUT = 12
_TIME_OUT = 101
_HIDDEN_SIZE = 768
_FRAME_TARGET_SAMPLES = 1600
_FRAME_STEP_SECONDS = 0.1


class AstEncoder(BaseEncoder):
    """Embedding encoder wrapping the frozen MIT AudioSet AST backbone.

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
        """Construct the frozen AST backbone."""
        super().__init__(granularity)
        if type(pretrained) is not bool:
            raise TypeError("pretrained must be a bool.")

        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )
        if pretrained:
            checkpoint_dir = ensure_ast_checkpoint(pretrained_dir)
            config = load_and_validate_ast_config(
                checkpoint_dir / "config.json"
            )
            self.backbone = ASTModel(config)
            backbone_state = load_ast_backbone_state(
                checkpoint_dir / "model.safetensors"
            )
            self.backbone.load_state_dict(backbone_state, strict=True)
        else:
            self.backbone = ASTModel(fixed_ast_config())

    @property
    def device(self) -> torch.device:
        """Return the current encoder device."""
        return (
            self.backbone.embeddings.patch_embeddings.projection.weight.device
        )

    def _backbone_outputs(self, input_features: Tensor):
        """Run the frozen AST backbone."""
        return self.backbone(input_values=input_features)

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
    ) -> dict[str, Tensor]:
        """Produce the official pooler clip embedding and project geometry."""
        embedding = self._backbone_outputs(input_features).pooler_output
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
    ) -> dict[str, Tensor]:
        """Produce frame embeddings aggregated by frequency mean and
        ownership geometry.
        """
        hidden = self._backbone_outputs(
            input_features
        ).last_hidden_state
        patch_tokens = hidden[:, 2:, :]
        embedding = patch_tokens.reshape(
            -1,
            _FREQUENCY_OUT,
            _TIME_OUT,
            _HIDDEN_SIZE,
        ).mean(dim=1)

        target_valid_samples = torch.round(
            valid_seconds * _TARGET_SAMPLE_RATE
        ).to(torch.long)
        valid_frames = torch.div(
            target_valid_samples + _FRAME_TARGET_SAMPLES - 1,
            _FRAME_TARGET_SAMPLES,
            rounding_mode="floor",
        ).clamp(min=1, max=_TIME_OUT)
        geometry, valid_mask = build_frame_geometry(
            valid_frames,
            valid_seconds,
            total_frames=_TIME_OUT,
            step_seconds=_FRAME_STEP_SECONDS,
        )
        # AST performs a fixed-canvas whole-batch forward pass, so invalid
        # frame slots still carry real backbone outputs and must be
        # explicitly zeroed by multiplying with the mask (unlike
        # PANNs/BEATs, which forward in groups).
        embedding = embedding * valid_mask.unsqueeze(2)
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": valid_mask,
        }


__all__ = ("AstEncoder",)
