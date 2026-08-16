"""ATST-Clip and ATST-Frame embedding encoders and their inference-only
backbones.

The private backbone classes replicate, operator by operator, the code
paths actually reached by the official ``AST`` (ATST-Clip) and
``FrameAST`` (ATST-Frame) under the configuration of the four official
checkpoints: linear ``PatchEmbed_v2`` over 64x4 patches, ``pos_type
= "cut"`` positional slicing, ``nprompt = 0``, ``avg_blocks = 0``, and
``qkv_bias = False``. The ``state_dict`` keys correspond one-to-one with
the official ones. Training-only or unreachable branches (masking,
prompt tokens, positional interpolation, CNN patch embedding, the
data2vec block-averaging teacher, DropPath, and every dropout, all of
which are identities under eval) are not ported.

Both families cap a single forward pass at the 250 patch slots held by
the learned ``pos_embed``; longer inputs are split into consecutive
1000-mel-frame chunks, which keeps every patch aligned to the global
40 ms grid. Clip granularity averages the per-chunk results, frame
granularity concatenates them along time.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from ..helpers.atst import (
    ATST_ARCHS,
    ATST_CHECKPOINTS,
    ATST_CHUNK_FRAMES,
    ATST_EMBED_DIMS,
    ATST_FRAME_STEP_SECONDS,
    ATST_NUM_BLOCKS,
    ATST_NUM_HEADS,
    ATST_NUM_MELS,
    ATST_PATCH_WIDTH,
    ATST_POSITION_SLOTS,
    AtstArch,
    atst_patch_frames,
    ensure_atst_checkpoint,
    load_atst_encoder_state,
)
from ..helpers.geometry import build_clip_outputs, build_frame_geometry
from ..helpers.grouping import (
    assemble_flat_groups,
    assemble_padded_groups,
    iter_length_groups,
)
from .base import BaseEncoder, Granularity

_PATCH_FEATURES = ATST_NUM_MELS * ATST_PATCH_WIDTH
_MLP_RATIO = 4
_LAYER_NORM_EPS = 1e-6
_INIT_STD = 0.02
# The official pooling divides by the patch count offset by this epsilon
# rather than taking an exact mean; it is reproduced verbatim.
_POOLING_EPSILON = 1e-6


class _AtstAttention(nn.Module):
    """Inference-only path of the official ``Attention``.

    The official module adds an additive attention mask built from each
    sample's valid length. Callers here group samples by valid length
    and slice to the exact prefix, so that mask is uniformly zero and is
    omitted; the arithmetic is unchanged.

    Args:
        embed_dim: Backbone width.
        num_heads: Attention head count.
    """

    def __init__(self, *, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Run self-attention on a ``[B, N, C]`` token sequence."""
        batch_size, length, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_size, length, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query @ key.transpose(-2, -1)) * self.scale
        attention = attention.softmax(dim=-1)
        x = (attention @ value).transpose(1, 2).reshape(
            batch_size, length, channels)
        return self.proj(x)


class _AtstMlp(nn.Module):
    """Inference-only path of the official ``Mlp``.

    Args:
        embed_dim: Backbone width.
    """

    def __init__(self, *, embed_dim: int) -> None:
        super().__init__()
        hidden_dim = embed_dim * _MLP_RATIO
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the two-layer feed-forward network."""
        return self.fc2(self.act(self.fc1(x)))


class _AtstBlock(nn.Module):
    """Pre-norm Transformer block (official ``Block`` under eval).

    Args:
        embed_dim: Backbone width.
        num_heads: Attention head count.
    """

    def __init__(self, *, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim, eps=_LAYER_NORM_EPS)
        self.attn = _AtstAttention(
            embed_dim=embed_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(embed_dim, eps=_LAYER_NORM_EPS)
        self.mlp = _AtstMlp(embed_dim=embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the attention and feed-forward sublayers."""
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class _AtstPatchEmbed(nn.Module):
    """Linear 64x4 patch embedding (official ``PatchEmbed_v2``).

    The official module rearranges ``b c (h p1) (w p2) -> b (w h)
    (p1 p2 c)`` with ``p1 = 64`` mel bins and ``p2 = 4`` frames. Since
    the mel axis holds exactly one patch row and the input carries a
    single channel, the same layout is produced here by reshaping the
    time-major features directly, without an einops dependency.

    Args:
        embed_dim: Backbone width.
    """

    def __init__(self, *, embed_dim: int) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(_PATCH_FEATURES, embed_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Embed ``[B, T, 64]`` features into ``[B, T // 4, D]`` tokens.

        A trailing remainder shorter than one patch is dropped, matching
        the official slice to ``width - width % patch_width``.
        """
        batch_size, num_frames, _ = features.shape
        num_patches = num_frames // ATST_PATCH_WIDTH
        patches = features[:, : num_patches * ATST_PATCH_WIDTH, :]
        # [B, W, 4, 64] -> [B, W, 64, 4] -> [B, W, 256] reproduces the
        # official (p1 p2 c) ordering: mel bin major, frame minor.
        patches = patches.view(
            batch_size, num_patches, ATST_PATCH_WIDTH, ATST_NUM_MELS
        ).transpose(2, 3).reshape(batch_size, num_patches, _PATCH_FEATURES)
        return self.patch_embed(patches)


def _build_blocks(*, embed_dim: int, num_heads: int) -> nn.ModuleList:
    """Build the 12 Transformer blocks shared by both families."""
    return nn.ModuleList(
        _AtstBlock(embed_dim=embed_dim, num_heads=num_heads)
        for _ in range(ATST_NUM_BLOCKS)
    )


def _init_module(module: nn.Module) -> None:
    """Replicate the official ``_init_weights`` for one submodule."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=_INIT_STD)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)


def _validate_common_arguments(
    *,
    arch: AtstArch,
    n_blocks: int,
    pretrained: bool,
) -> None:
    """Validate the constructor arguments shared by both encoders.

    Raises:
        TypeError: ``n_blocks`` or ``pretrained`` has the wrong type.
        ValueError: ``arch`` is unknown or ``n_blocks`` is out of range.
    """
    if arch not in ATST_ARCHS:
        raise ValueError(
            f"Unknown ATST arch {arch!r}, available: "
            f"{sorted(ATST_ARCHS)}."
        )
    if type(n_blocks) is not int:
        raise TypeError("n_blocks must be an int.")
    if not 1 <= n_blocks <= ATST_NUM_BLOCKS:
        raise ValueError(
            f"n_blocks must be within [1, {ATST_NUM_BLOCKS}], but "
            f"received {n_blocks}."
        )
    if type(pretrained) is not bool:
        raise TypeError("pretrained must be a bool.")


def _iter_chunk_bounds(num_frames: int) -> list[tuple[int, int]]:
    """Split a mel frame count into consecutive chunk bounds.

    Chunks are 1000 frames long, i.e. exactly the 250 patch slots the
    learned positional embedding covers, so every patch keeps its place
    on the global 40 ms grid. A trailing chunk shorter than one patch
    contributes nothing and is dropped, which matches the total patch
    count ``num_frames // 4`` exactly because the chunk length is itself
    a multiple of the patch width.

    Args:
        num_frames: Valid mel frame count of one length group.

    Returns:
        ``[(start, end), ...]`` frame bounds of the usable chunks.
    """
    bounds = []
    for start in range(0, num_frames, ATST_CHUNK_FRAMES):
        end = min(start + ATST_CHUNK_FRAMES, num_frames)
        if (end - start) >= ATST_PATCH_WIDTH:
            bounds.append((start, end))
    return bounds


class AtstClipEncoder(BaseEncoder):
    """ATST-Clip backbone exposing a clip-granularity embedding.

    Reproduces the official downstream feature extraction
    (``get_intermediate_layers_chunks`` with ``avgpool=True``): the last
    ``n_blocks`` block outputs each pass through the final norm, and the
    cls token and the mean over patch tokens are concatenated, giving a
    ``2 * n_blocks * D`` vector. Chunks are averaged with equal weight,
    matching the official ``get_scene_embedding``.

    The official family provides no frame-level usage for ATST-Clip, so
    only clip granularity is exposed.

    Args:
        granularity: Output granularity; only ``"clip"`` is supported.
        arch: ``"small"`` (384) or ``"base"`` (768).
        n_blocks: How many trailing blocks to concatenate, 1 to 12.
        pretrained: Whether to load the official pretrained weights.
        pretrained_dir: The directory holding the checkpoint. Uses the
            HF cache when ``None``.
    """

    supported_granularities = frozenset(("clip",))

    def __init__(
        self,
        *,
        granularity: Granularity,
        arch: AtstArch,
        n_blocks: int = 1,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        """Construct the ATST-Clip backbone."""
        super().__init__(granularity)
        _validate_common_arguments(
            arch=arch, n_blocks=n_blocks, pretrained=pretrained)

        embed_dim = ATST_EMBED_DIMS[arch]
        self.arch = arch
        self.n_blocks = n_blocks
        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )
        # The cls branch and the patch-mean branch are concatenated per
        # block, hence the factor of two.
        self.embedding_dim = 2 * n_blocks * embed_dim

        # mask_embed is a pretraining-only parameter; it is declared so
        # the state_dict stays in one-to-one correspondence with the
        # official checkpoint, and is never read during inference.
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, ATST_POSITION_SLOTS, embed_dim))
        self.patch_embed = _AtstPatchEmbed(embed_dim=embed_dim)
        self.blocks = _build_blocks(
            embed_dim=embed_dim, num_heads=ATST_NUM_HEADS[arch])
        self.norm = nn.LayerNorm(embed_dim, eps=_LAYER_NORM_EPS)

        self._reset_parameters()
        if pretrained:
            metadata = ATST_CHECKPOINTS[("clip", arch)]
            checkpoint_path = ensure_atst_checkpoint(
                metadata, pretrained_dir)
            state = load_atst_encoder_state(metadata, checkpoint_path)
            self.load_state_dict(state, strict=True)

    def _reset_parameters(self) -> None:
        """Replicate the official initialization order."""
        nn.init.trunc_normal_(self.pos_embed, std=_INIT_STD)
        nn.init.trunc_normal_(self.mask_embed, std=_INIT_STD)
        nn.init.trunc_normal_(self.cls_token, std=_INIT_STD)
        self.apply(_init_module)

    @property
    def device(self) -> torch.device:
        """Return the current encoder device."""
        return self.patch_embed.patch_embed.weight.device

    def _chunk_features(self, features: Tensor) -> Tensor:
        """Encode one chunk into its ``[B, 2 * n_blocks * D]`` vector.

        Args:
            features: ``[B, T, 64]`` mel features of a single chunk,
                whose frame count is a multiple of the patch width plus
                at most a dropped remainder.

        Returns:
            The concatenated cls and patch-mean branches.
        """
        tokens = self.patch_embed(features)
        batch_size, num_patches, _ = tokens.shape
        tokens = torch.cat(
            (self.cls_token.expand(batch_size, -1, -1), tokens), dim=1)
        tokens = tokens + self.pos_embed[:, : num_patches + 1, :]

        collected = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if len(self.blocks) - index <= self.n_blocks:
                collected.append(self.norm(tokens))

        cls_branch = [layer[:, 0] for layer in collected]
        patch_branch = [
            layer[:, 1:].sum(dim=1) / (num_patches + _POOLING_EPSILON)
            for layer in collected
        ]
        return torch.cat(cls_branch + patch_branch, dim=-1)

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
            group_features = input_features.index_select(0, batch_indices)
            chunk_embeddings = [
                self._chunk_features(group_features[:, start:end])
                for start, end in _iter_chunk_bounds(frame_length)
            ]
            batch_index_groups.append(batch_indices)
            embedding_groups.append(
                torch.stack(chunk_embeddings, dim=0).mean(dim=0))

        embedding = assemble_flat_groups(
            batch_size, batch_index_groups, embedding_groups, input_features)
        geometry, valid_mask = build_clip_outputs(valid_seconds)
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": valid_mask,
        }


class AtstFrameEncoder(BaseEncoder):
    """ATST-Frame backbone exposing clip and frame granularities.

    Reproduces the official ``get_intermediate_layers``: the last
    ``n_blocks`` block outputs each pass through the final frame norm
    and are concatenated, giving ``n_blocks * D``. At clip granularity
    the official ``scene=True`` mean over patch tokens is used (this
    family has no cls token, so there is no second branch), and chunks
    are averaged with equal weight; at frame granularity the per-chunk
    token sequences are concatenated along time, one 40 ms frame per
    patch.

    Args:
        granularity: Output granularity, ``"clip"`` or ``"frame"``.
        arch: ``"small"`` (384) or ``"base"`` (768).
        n_blocks: How many trailing blocks to concatenate, 1 to 12.
        pretrained: Whether to load the official pretrained weights.
        pretrained_dir: The directory holding the checkpoint. Uses the
            HF cache when ``None``.
    """

    supported_granularities = frozenset(("clip", "frame"))

    def __init__(
        self,
        *,
        granularity: Granularity,
        arch: AtstArch,
        n_blocks: int = 1,
        pretrained: bool = True,
        pretrained_dir: str | Path | None = None,
    ) -> None:
        """Construct the ATST-Frame backbone."""
        super().__init__(granularity)
        _validate_common_arguments(
            arch=arch, n_blocks=n_blocks, pretrained=pretrained)

        embed_dim = ATST_EMBED_DIMS[arch]
        self.arch = arch
        self.n_blocks = n_blocks
        self.pretrained = pretrained
        self.pretrained_dir = (
            None if pretrained_dir is None else Path(pretrained_dir)
        )
        self.embedding_dim = n_blocks * embed_dim

        # Pretraining-only parameter, declared for state_dict parity.
        self.mask_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, ATST_POSITION_SLOTS, embed_dim))
        self.patch_embed = _AtstPatchEmbed(embed_dim=embed_dim)
        self.blocks = _build_blocks(
            embed_dim=embed_dim, num_heads=ATST_NUM_HEADS[arch])
        self.norm_frame = nn.LayerNorm(embed_dim, eps=_LAYER_NORM_EPS)

        self._reset_parameters()
        if pretrained:
            metadata = ATST_CHECKPOINTS[("frame", arch)]
            checkpoint_path = ensure_atst_checkpoint(
                metadata, pretrained_dir)
            state = load_atst_encoder_state(metadata, checkpoint_path)
            self.load_state_dict(state, strict=True)

    def _reset_parameters(self) -> None:
        """Replicate the official initialization order."""
        nn.init.trunc_normal_(self.pos_embed, std=_INIT_STD)
        nn.init.trunc_normal_(self.mask_embed, std=_INIT_STD)
        self.apply(_init_module)

    @property
    def device(self) -> torch.device:
        """Return the current encoder device."""
        return self.patch_embed.patch_embed.weight.device

    def _chunk_tokens(self, features: Tensor) -> Tensor:
        """Encode one chunk into its ``[B, P, n_blocks * D]`` tokens.

        The positional embedding is sliced from index 1 onward: this
        family holds no cls token, and the official ``FrameAST`` keeps
        slot 0 unused.
        """
        tokens = self.patch_embed(features)
        num_patches = tokens.shape[1]
        tokens = tokens + self.pos_embed[:, 1 : num_patches + 1, :]

        collected = []
        for index, block in enumerate(self.blocks):
            tokens = block(tokens)
            if len(self.blocks) - index <= self.n_blocks:
                collected.append(self.norm_frame(tokens))
        return torch.cat(collected, dim=-1)

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
            group_features = input_features.index_select(0, batch_indices)
            chunk_embeddings = []
            for start, end in _iter_chunk_bounds(frame_length):
                tokens = self._chunk_tokens(group_features[:, start:end])
                chunk_embeddings.append(
                    tokens.sum(dim=1)
                    / (tokens.shape[1] + _POOLING_EPSILON)
                )
            batch_index_groups.append(batch_indices)
            embedding_groups.append(
                torch.stack(chunk_embeddings, dim=0).mean(dim=0))

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
        valid_patches = atst_patch_frames(valid_feature_frames)
        batch_size = input_features.shape[0]
        max_patches = int(valid_patches.max().item())
        batch_index_groups = []
        embedding_groups = []

        for frame_length, batch_indices in iter_length_groups(
                valid_feature_frames):
            group_features = input_features.index_select(0, batch_indices)
            chunk_tokens = [
                self._chunk_tokens(group_features[:, start:end])
                for start, end in _iter_chunk_bounds(frame_length)
            ]
            batch_index_groups.append(batch_indices)
            embedding_groups.append(torch.cat(chunk_tokens, dim=1))

        # Grouped forward passes plus zero-canvas index_copy already
        # guarantee that invalid-frame embeddings are exactly 0, so
        # there is no need to multiply by valid_mask again.
        embedding = assemble_padded_groups(
            batch_size,
            batch_index_groups,
            embedding_groups,
            input_features,
            total_frames=max_patches,
        )
        geometry, valid_mask = build_frame_geometry(
            valid_patches,
            valid_seconds,
            total_frames=max_patches,
            step_seconds=ATST_FRAME_STEP_SECONDS,
        )
        return {
            "embedding": embedding,
            "geometry": geometry,
            "valid_mask": valid_mask,
        }


__all__ = ("AtstClipEncoder", "AtstFrameEncoder")
