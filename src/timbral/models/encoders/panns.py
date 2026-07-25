"""PANNs Cnn14 embedding encoder."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..helpers.geometry import build_clip_outputs, build_frame_geometry
from ..helpers.grouping import (
    assemble_flat_groups,
    assemble_padded_groups,
    iter_length_groups,
)
from ..helpers.panns import (
    PANNS_CHECKPOINTS,
    PANNS_TARGET_SAMPLE_RATES,
    PANNS_VARIANTS,
    PannsTargetSampleRate,
    PannsVariant,
    ensure_panns_checkpoint,
    load_panns_checkpoint_model,
)
from .base import BaseEncoder, Granularity

_ENCODER_DOWNSAMPLE_RATIO = 32
_FEATURE_HOP_SECONDS = 0.01
_FRAME_STEP_SECONDS = (
    _ENCODER_DOWNSAMPLE_RATIO * _FEATURE_HOP_SECONDS
)


def _initialize_layer(layer: nn.Conv2d | nn.Linear) -> None:
    """Initialize a conv or linear layer using the official PANNs rules."""
    nn.init.xavier_uniform_(layer.weight)
    if layer.bias is not None:
        layer.bias.data.fill_(0.0)


def _initialize_batch_norm(batch_norm: nn.BatchNorm2d) -> None:
    """Initialize a BatchNorm layer using the official PANNs rules."""
    batch_norm.bias.data.fill_(0.0)
    batch_norm.weight.data.fill_(1.0)


class _ConvBlock(nn.Module):
    """Two-layer 3x3 conv block of Cnn14."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        _initialize_layer(self.conv1)
        _initialize_layer(self.conv2)
        _initialize_batch_norm(self.bn1)
        _initialize_batch_norm(self.bn2)

    def forward(
        self,
        inputs: Tensor,
        *,
        pool_size: tuple[int, int],
    ) -> Tensor:
        """Run convolution, normalization, activation, and average
        pooling.
        """
        outputs = F.relu_(self.bn1(self.conv1(inputs)))
        outputs = F.relu_(self.bn2(self.conv2(outputs)))
        if pool_size != (1, 1):
            outputs = F.avg_pool2d(outputs, kernel_size=pool_size)
        return outputs


class PannsCnn14Encoder(BaseEncoder):
    """PANNs Cnn14 backbone with clip/frame embedding branches.

    Args:
        granularity: Output granularity, ``"clip"`` or ``"frame"``.
        target_sample_rate: Target sample rate matching the PANNs front
            end.
        variant: Pooling and checkpoint variant.
        pretrained: Whether to download and load the official backbone
            state.
        pretrained_dir: The directory containing the checkpoint. Uses the
            HF cache when ``None``.
    """

    supported_granularities = frozenset(("clip", "frame"))
    embedding_dim = 2048

    def __init__(
        self,
        *,
        granularity: Granularity,
        target_sample_rate: PannsTargetSampleRate,
        variant: PannsVariant,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        super().__init__(granularity)
        if type(pretrained) is not bool:
            raise TypeError("pretrained must be a bool.")
        if target_sample_rate not in PANNS_TARGET_SAMPLE_RATES:
            raise ValueError(
                "target_sample_rate only supports 16000 or 32000."
            )
        if variant not in PANNS_VARIANTS:
            raise ValueError(
                "variant only supports 'max_mean' or 'decision_level_max'."
            )
        if (
            pretrained
            and (target_sample_rate, variant) not in PANNS_CHECKPOINTS
        ):
            raise ValueError(
                "No official weights exist for this sample rate and "
                "variant combination when pretrained=True."
            )

        self.target_sample_rate = target_sample_rate
        self.variant = variant
        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )

        self.conv_block1 = _ConvBlock(1, 64)
        self.conv_block2 = _ConvBlock(64, 128)
        self.conv_block3 = _ConvBlock(128, 256)
        self.conv_block4 = _ConvBlock(256, 512)
        self.conv_block5 = _ConvBlock(512, 1024)
        self.conv_block6 = _ConvBlock(1024, 2048)
        self.fc1 = nn.Linear(2048, 2048)
        _initialize_layer(self.fc1)

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
            encoder_state = {
                key: value
                for key, value in checkpoint_state.items()
                if key.startswith("conv_block") or key.startswith("fc1.")
            }
            self.load_state_dict(encoder_state, strict=True)

    @property
    def device(self) -> torch.device:
        """Return the current encoder device."""
        return self.fc1.weight.device

    def _conv_features(self, input_features: Tensor) -> Tensor:
        """Extract Cnn14 sequence features after averaging over
        frequency.
        """
        outputs = input_features.unsqueeze(1)
        outputs = self.conv_block1(outputs, pool_size=(2, 2))
        outputs = F.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block2(outputs, pool_size=(2, 2))
        outputs = F.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block3(outputs, pool_size=(2, 2))
        outputs = F.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block4(outputs, pool_size=(2, 2))
        outputs = F.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block5(outputs, pool_size=(2, 2))
        outputs = F.dropout(outputs, p=0.2, training=self.training)
        outputs = self.conv_block6(outputs, pool_size=(1, 1))
        outputs = F.dropout(outputs, p=0.2, training=self.training)
        return outputs.mean(dim=3)

    def _max_mean_clip(self, features: Tensor) -> Tensor:
        """Run the official max+mean clip branch."""
        pooled = features.amax(dim=2) + features.mean(dim=2)
        pooled = F.dropout(pooled, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(pooled))
        return F.dropout(embedding, p=0.5, training=self.training)

    def _max_mean_frames(self, features: Tensor) -> Tensor:
        """Run the frame branch derived from max_mean weights."""
        features = F.dropout(features, p=0.5, training=self.training)
        embedding = F.relu_(self.fc1(features.transpose(1, 2)))
        return F.dropout(embedding, p=0.5, training=self.training)

    def _decision_level_frames(self, features: Tensor) -> Tensor:
        """Run the official DecisionLevelMax segment hidden branch."""
        smoothed = F.max_pool1d(
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        ) + F.avg_pool1d(
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        smoothed = F.dropout(
            smoothed,
            p=0.5,
            training=self.training,
        )
        embedding = F.relu_(self.fc1(smoothed.transpose(1, 2)))
        return F.dropout(embedding, p=0.5, training=self.training)

    def _encode_clip(
        self,
        input_features: Tensor,
        *,
        valid_seconds: Tensor,
        valid_feature_frames: Tensor,
    ) -> dict[str, Tensor]:
        """Produce the clip embedding, geometry, and mask."""
        batch_size = input_features.shape[0]
        batch_index_groups = []
        embedding_groups = []

        for logical_length, batch_indices in iter_length_groups(
                valid_feature_frames):
            physical_length = max(
                _ENCODER_DOWNSAMPLE_RATIO,
                logical_length,
            )
            group_input = input_features.index_select(
                0,
                batch_indices,
            )[:, :physical_length]
            features = self._conv_features(group_input)
            if self.variant == "max_mean":
                group_embedding = self._max_mean_clip(features)
            else:
                frame_embedding = self._decision_level_frames(features)
                group_embedding = frame_embedding.amax(dim=1)
            batch_index_groups.append(batch_indices)
            embedding_groups.append(group_embedding)

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
        valid_feature_frames: Tensor,
    ) -> dict[str, Tensor]:
        """Produce the frame embedding, geometry, and mask."""
        valid_embedding_frames = torch.clamp(
            torch.div(
                valid_feature_frames,
                _ENCODER_DOWNSAMPLE_RATIO,
                rounding_mode="floor",
            ),
            min=1,
        )
        batch_size = input_features.shape[0]
        max_embedding_frames = int(
            valid_embedding_frames.max().item()
        )
        batch_index_groups = []
        embedding_groups = []

        for logical_length, batch_indices in iter_length_groups(
                valid_feature_frames):
            physical_length = max(
                _ENCODER_DOWNSAMPLE_RATIO,
                logical_length,
            )
            group_input = input_features.index_select(
                0,
                batch_indices,
            )[:, :physical_length]
            features = self._conv_features(group_input)
            if self.variant == "max_mean":
                group_embedding = self._max_mean_frames(features)
            else:
                group_embedding = self._decision_level_frames(features)
            batch_index_groups.append(batch_indices)
            embedding_groups.append(group_embedding)

        # Grouped forward passes plus zero-canvas index_copy already
        # guarantee that invalid-frame embeddings are exactly 0, so
        # there is no need to multiply by valid_mask again.
        embedding = assemble_padded_groups(
            batch_size,
            batch_index_groups,
            embedding_groups,
            input_features,
            total_frames=max_embedding_frames,
        )
        geometry, valid_mask = build_frame_geometry(
            valid_embedding_frames,
            valid_seconds,
            total_frames=max_embedding_frames,
            step_seconds=_FRAME_STEP_SECONDS,
        )
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": valid_mask,
        }


__all__ = ("PannsCnn14Encoder", "PannsVariant")
