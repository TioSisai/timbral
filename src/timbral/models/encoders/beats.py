"""BEATs embedding encoder and its inference-only backbone.

The private backbone classes replicate, operator by operator, the code
paths actually reached by the official ``backbone.py`` under the cfg of
the 15 official checkpoints (post-norm + deep_norm + T5 relative
position bias + gru gating). The state_dict keys correspond one-to-one
with the official ones (only the two weight_norm keys of pos_conv are
named per the parametrize convention, mapped by the helpers at load
time). Training-only or unreachable branches (layerdrop, GradMultiply,
incremental_state, quant_noise, non-gelu activations, the padding_mask
path, per-layer export) are not ported.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..helpers.beats import (
    BEATS_CHECKPOINTS,
    ensure_beats_checkpoint,
    load_beats_checkpoint_state,
)
from ..helpers.geometry import build_clip_outputs, build_frame_geometry
from ..helpers.grouping import (
    assemble_flat_groups,
    assemble_padded_groups,
    iter_length_groups,
)
from .base import BaseEncoder, Granularity

_PATCH_SIZE = 16
_PATCH_EMBED_DIM = 512
_ENCODER_EMBED_DIM = 768
_FREQ_PATCHES = 8
_FEATURE_HOP_SECONDS = 0.01
_FRAME_STEP_SECONDS = _PATCH_SIZE * _FEATURE_HOP_SECONDS

_FFN_DIM = 3072
_NUM_HEADS = 12
_NUM_LAYERS = 12
_HEAD_DIM = _ENCODER_EMBED_DIM // _NUM_HEADS
_CONV_POS = 128
_CONV_POS_GROUPS = 16
_NUM_BUCKETS = 320
_MAX_DISTANCE = 800
_ATTENTION_ALPHA = 32
_DEEP_NORM_ALPHA = math.pow(2 * _NUM_LAYERS, 1 / 4)
_DEEP_NORM_BETA = math.pow(8 * _NUM_LAYERS, -1 / 4)
_BERT_INIT_STD = 0.02

# dropout p values are taken from the checkpoint cfg's measured values
# per entry type (all are identity under eval):
# (dropout, attention_dropout, dropout_input)
_DROPOUTS_PRETRAINED = (0.1, 0.1, 0.1)
_DROPOUTS_FINETUNED = (0.0, 0.0, 0.0)
_ACTIVATION_DROPOUT = 0.0


class _SamePad(nn.Module):
    """Trim the extra trailing column produced by even-kernel
    symmetric-padding convolution (official SamePad).
    """

    def forward(self, x: Tensor) -> Tensor:
        return x[:, :, :-1]


class _BeatsSelfAttention(nn.Module):
    """Inference-only self-attention path of the official
    MultiheadAttention.

    Includes the alpha=32 numerical stability trick, T5 bidirectional
    relative-position bucket bias, and gru_rel_pos gating;
    ``position_bias`` is passed through across layers and computed only
    once, in the first layer.

    Args:
        attention_dropout: Dropout applied to attention probabilities.
        has_relative_attention_bias: Whether this layer holds the
            relative position bias table. Layer 1 and onward are rebound
            by :class:`_BeatsTransformerEncoder` to the layer-0 instance.
    """

    def __init__(
        self,
        *,
        attention_dropout: float,
        has_relative_attention_bias: bool,
    ) -> None:
        super().__init__()
        self.k_proj = nn.Linear(_ENCODER_EMBED_DIM, _ENCODER_EMBED_DIM)
        self.v_proj = nn.Linear(_ENCODER_EMBED_DIM, _ENCODER_EMBED_DIM)
        self.q_proj = nn.Linear(_ENCODER_EMBED_DIM, _ENCODER_EMBED_DIM)
        self.out_proj = nn.Linear(_ENCODER_EMBED_DIM, _ENCODER_EMBED_DIM)
        self.dropout_module = nn.Dropout(attention_dropout)
        self.grep_linear = nn.Linear(_HEAD_DIM, 8)
        self.grep_a = nn.Parameter(torch.ones(1, _NUM_HEADS, 1, 1))
        if has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                _NUM_BUCKETS,
                _NUM_HEADS,
            )
        self.scaling = _HEAD_DIM**-0.5

    def _relative_positions_bucket(
        self,
        relative_positions: Tensor,
    ) -> Tensor:
        """T5-style bidirectional bucketing: exact for short distances,
        logarithmic bucketing for long distances.
        """
        num_buckets = _NUM_BUCKETS // 2
        relative_buckets = (
            (relative_positions > 0).to(torch.long) * num_buckets
        )
        relative_positions = torch.abs(relative_positions)
        max_exact = num_buckets // 2
        is_small = relative_positions < max_exact
        position_if_large = max_exact + (
            torch.log(relative_positions.float() / max_exact)
            / math.log(_MAX_DISTANCE / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        position_if_large = torch.min(
            position_if_large,
            torch.full_like(position_if_large, num_buckets - 1),
        )
        return relative_buckets + torch.where(
            is_small,
            relative_positions,
            position_if_large,
        )

    def _compute_bias(self, length: int) -> Tensor:
        """Compute the ``[heads, T, T]`` relative position bias (the
        official implementation buckets on CPU).
        """
        context_position = torch.arange(length, dtype=torch.long)[:, None]
        memory_position = torch.arange(length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        bucket = self._relative_positions_bucket(relative_position).to(
            self.relative_attention_bias.weight.device
        )
        return self.relative_attention_bias(bucket).permute(2, 0, 1)

    def forward(
        self,
        x: Tensor,
        position_bias: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run self-attention on a ``[T, B, C]`` input.

        Args:
            x: The ``[T, B, C]`` input sequence.
            position_bias: The ``[B*heads, T, T]`` bias passed through
                from the previous layer; computed by this layer when
                ``None``.

        Returns:
            ``(output sequence, position_bias)``.
        """
        length, batch_size, _ = x.size()
        if position_bias is None:
            position_bias = (
                self._compute_bias(length)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1, 1)
                .view(batch_size * _NUM_HEADS, length, length)
            )

        query = self.q_proj(x)
        query = query * self.scaling
        query = query * (1 / _ATTENTION_ALPHA)
        query = (
            query.contiguous()
            .view(length, batch_size * _NUM_HEADS, _HEAD_DIM)
            .transpose(0, 1)
        )
        key = (
            self.k_proj(x)
            .contiguous()
            .view(length, batch_size * _NUM_HEADS, _HEAD_DIM)
            .transpose(0, 1)
        )
        value = (
            self.v_proj(x)
            .contiguous()
            .view(length, batch_size * _NUM_HEADS, _HEAD_DIM)
            .transpose(0, 1)
        )

        attn_weights = torch.bmm(query, key.transpose(1, 2))
        attn_weights = (
            attn_weights - attn_weights.max(dim=-1, keepdim=True)[0]
        ) * _ATTENTION_ALPHA

        # gru_rel_pos gating: per-position gates are generated from the
        # unscaled q, scaling the shared bias.
        query_layer = (
            query.view(batch_size, _NUM_HEADS, length, _HEAD_DIM)
            * _ATTENTION_ALPHA
            / self.scaling
        )
        gate_a, gate_b = torch.sigmoid(
            self.grep_linear(query_layer)
            .view(batch_size, _NUM_HEADS, length, 2, 4)
            .sum(-1, keepdim=False)
        ).chunk(2, dim=-1)
        gate = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
        attn_weights = attn_weights + (
            gate.view(batch_size * _NUM_HEADS, length, 1) * position_bias
        ).view(attn_weights.size())

        attn_probs = self.dropout_module(
            F.softmax(attn_weights, dim=-1)
        )
        attn = torch.bmm(attn_probs, value)
        attn = (
            attn.transpose(0, 1)
            .contiguous()
            .view(length, batch_size, _ENCODER_EMBED_DIM)
        )
        return self.out_proj(attn), position_bias


class _BeatsEncoderLayer(nn.Module):
    """Post-norm + deep_norm Transformer layer.

    Args:
        dropout: Dropout applied to the output before the residual
            connection.
        attention_dropout: Dropout applied to attention probabilities.
        activation_dropout: Dropout applied after the FFN activation.
        has_relative_attention_bias: See :class:`_BeatsSelfAttention`.
    """

    def __init__(
        self,
        *,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        has_relative_attention_bias: bool,
    ) -> None:
        super().__init__()
        self.self_attn = _BeatsSelfAttention(
            attention_dropout=attention_dropout,
            has_relative_attention_bias=has_relative_attention_bias,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(activation_dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.self_attn_layer_norm = nn.LayerNorm(_ENCODER_EMBED_DIM)
        self.fc1 = nn.Linear(_ENCODER_EMBED_DIM, _FFN_DIM)
        self.fc2 = nn.Linear(_FFN_DIM, _ENCODER_EMBED_DIM)
        self.final_layer_norm = nn.LayerNorm(_ENCODER_EMBED_DIM)

    def forward(
        self,
        x: Tensor,
        position_bias: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Run the attention and FFN sublayers with deep_norm residual
        connections.
        """
        residual = x
        x, position_bias = self.self_attn(x, position_bias)
        x = self.dropout1(x)
        x = residual * _DEEP_NORM_ALPHA + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.gelu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.dropout3(x)
        x = residual * _DEEP_NORM_ALPHA + x
        return self.final_layer_norm(x), position_bias


class _BeatsTransformerEncoder(nn.Module):
    """Inference-only path of the official TransformerEncoder.

    Args:
        dropout: Dropout applied to the layer input and output.
        attention_dropout: Dropout applied to attention probabilities.
        activation_dropout: Dropout applied after the FFN activation.
    """

    def __init__(
        self,
        *,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
    ) -> None:
        super().__init__()
        self.dropout = dropout

        conv = nn.Conv1d(
            _ENCODER_EMBED_DIM,
            _ENCODER_EMBED_DIM,
            kernel_size=_CONV_POS,
            padding=_CONV_POS // 2,
            groups=_CONV_POS_GROUPS,
        )
        nn.init.normal_(
            conv.weight,
            mean=0,
            std=math.sqrt(4 / (_CONV_POS * _ENCODER_EMBED_DIM)),
        )
        nn.init.constant_(conv.bias, 0)
        # The forward pass uses the same ``torch._weight_norm`` operator
        # as the legacy weight_norm; the checkpoint's weight_g/weight_v
        # keys are mapped by the helpers at load time to
        # parametrizations.weight.original{0,1}.
        conv = nn.utils.parametrizations.weight_norm(
            conv,
            name="weight",
            dim=2,
        )
        self.pos_conv = nn.Sequential(conv, _SamePad(), nn.GELU())

        self.layers = nn.ModuleList(
            _BeatsEncoderLayer(
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
                has_relative_attention_bias=True,
            )
            for _ in range(_NUM_LAYERS)
        )
        # The relative position bias is shared as a single instance
        # across all 12 layers (rebound after official construction;
        # every layer's key exists in the state_dict and points to the
        # same tensor).
        shared_bias = self.layers[0].self_attn.relative_attention_bias
        for layer in list(self.layers)[1:]:
            del layer.self_attn.relative_attention_bias
            layer.self_attn.relative_attention_bias = shared_bias

        self.layer_norm = nn.LayerNorm(_ENCODER_EMBED_DIM)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Replicate the final initialization after the official
        init_bert_params and deep_norm gain.
        """
        nn.init.normal_(
            self.layers[0].self_attn.relative_attention_bias.weight,
            mean=0.0,
            std=_BERT_INIT_STD,
        )
        for layer in self.layers:
            attention = layer.self_attn
            nn.init.xavier_normal_(attention.k_proj.weight, gain=1)
            nn.init.xavier_normal_(attention.q_proj.weight, gain=1)
            nn.init.xavier_normal_(
                attention.v_proj.weight,
                gain=_DEEP_NORM_BETA,
            )
            nn.init.xavier_normal_(
                attention.out_proj.weight,
                gain=_DEEP_NORM_BETA,
            )
            nn.init.xavier_normal_(layer.fc1.weight, gain=_DEEP_NORM_BETA)
            nn.init.xavier_normal_(layer.fc2.weight, gain=_DEEP_NORM_BETA)
            nn.init.normal_(
                attention.grep_linear.weight,
                mean=0.0,
                std=_BERT_INIT_STD,
            )
            for bias in (
                attention.k_proj.bias,
                attention.q_proj.bias,
                attention.v_proj.bias,
                attention.out_proj.bias,
                attention.grep_linear.bias,
                layer.fc1.bias,
                layer.fc2.bias,
            ):
                nn.init.zeros_(bias)

    def forward(self, x: Tensor) -> Tensor:
        """Encode a ``[B, N, C]`` token sequence.

        Args:
            x: The ``[B, N, C]`` patch token sequence, unpadded within
                the group.

        Returns:
            The ``[B, N, C]`` encoded sequence.
        """
        x = x + self.pos_conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = x.transpose(0, 1)
        position_bias = None
        for layer in self.layers:
            x, position_bias = layer(x, position_bias)
        return x.transpose(0, 1)


class BeatsEncoder(BaseEncoder):
    """BEATs backbone with clip/frame embedding branches.

    Uniformly outputs 768-dimensional backbone features; the predictor
    classification head of fine-tuned checkpoints is discarded at load
    time.

    Args:
        granularity: Output granularity, ``"clip"`` or ``"frame"``.
        checkpoint: One of the 15 official entry names, determining the
            weights and dropout p group.
        pretrained: Whether to load official pretrained weights.
        pretrained_dir: The directory containing the checkpoint. Uses the
            HF cache when ``None``.
    """

    supported_granularities = frozenset(("clip", "frame"))
    embedding_dim = _ENCODER_EMBED_DIM

    def __init__(
        self,
        *,
        granularity: Granularity,
        checkpoint: str,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        super().__init__(granularity)
        if type(pretrained) is not bool:
            raise TypeError("pretrained must be a bool.")
        if checkpoint not in BEATS_CHECKPOINTS:
            raise ValueError(
                f"Unknown BEATs entry {checkpoint!r}, "
                f"available: {sorted(BEATS_CHECKPOINTS)}."
            )

        self.checkpoint = checkpoint
        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )

        metadata = BEATS_CHECKPOINTS[checkpoint]
        dropout, attention_dropout, dropout_input = (
            _DROPOUTS_FINETUNED
            if metadata.finetuned
            else _DROPOUTS_PRETRAINED
        )
        self.patch_embedding = nn.Conv2d(
            1,
            _PATCH_EMBED_DIM,
            kernel_size=_PATCH_SIZE,
            stride=_PATCH_SIZE,
            bias=False,
        )
        self.layer_norm = nn.LayerNorm(_PATCH_EMBED_DIM)
        self.post_extract_proj = nn.Linear(
            _PATCH_EMBED_DIM,
            _ENCODER_EMBED_DIM,
        )
        self.dropout_input = nn.Dropout(dropout_input)
        self.encoder = _BeatsTransformerEncoder(
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=_ACTIVATION_DROPOUT,
        )

        if pretrained:
            checkpoint_path = ensure_beats_checkpoint(
                checkpoint,
                pretrained_dir,
            )
            state = load_beats_checkpoint_state(
                checkpoint,
                checkpoint_path,
            )
            self.load_state_dict(state, strict=True)

    @property
    def device(self) -> torch.device:
        """Return the current encoder device."""
        return self.post_extract_proj.weight.device

    def _backbone_tokens(self, input_features: Tensor) -> Tensor:
        """Encode the fbank features within a group into a
        ``[B, N, 768]`` token sequence.

        Tokens are flattened with time as the outer dimension and
        frequency as the inner dimension; each patch time block produces
        8 frequency tokens.
        """
        features = self.patch_embedding(input_features.unsqueeze(1))
        features = features.reshape(
            features.shape[0],
            features.shape[1],
            -1,
        ).transpose(1, 2)
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)
        features = self.dropout_input(features)
        return self.encoder(features)

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

        for frame_length, batch_indices in iter_length_groups(
                valid_feature_frames):
            tokens = self._backbone_tokens(
                input_features.index_select(0, batch_indices)[
                    :, :frame_length
                ]
            )
            batch_index_groups.append(batch_indices)
            embedding_groups.append(tokens.mean(dim=1))

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
        valid_embedding_frames = torch.div(
            valid_feature_frames,
            _PATCH_SIZE,
            rounding_mode="floor",
        )
        batch_size = input_features.shape[0]
        max_embedding_frames = int(valid_embedding_frames.max().item())
        batch_index_groups = []
        embedding_groups = []

        for frame_length, batch_indices in iter_length_groups(
                valid_feature_frames):
            tokens = self._backbone_tokens(
                input_features.index_select(0, batch_indices)[
                    :, :frame_length
                ]
            )
            group_embedding = tokens.view(
                tokens.shape[0],
                -1,
                _FREQ_PATCHES,
                self.embedding_dim,
            ).mean(dim=2)
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


__all__ = ("BeatsEncoder",)
