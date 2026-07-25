"""Model-agnostic scaffolding for grouping by unique length (shared by
Transform and Encoder).

Grouping pattern: split into groups by unique valid length -> forward
pass within each group -> pad and index_copy back into a zero canvas,
ensuring the output is independent of batch composition. This module only
provides assembly utilities that are independent of model numerics; it
does not involve any frontend/backbone algorithm.
"""

from __future__ import annotations

from collections.abc import Iterator

import torch
from torch import Tensor
from torch.nn import functional as F


def iter_length_groups(
    lengths: Tensor,
) -> Iterator[tuple[int | list[int], Tensor]]:
    """Iterate over groups by unique length, yielding
    ``(length value, batch indices)``.

    All length values are fetched via a single ``tolist`` call before the
    loop; scalars are not pulled per group inside the loop body. The
    ``torch.nonzero`` call within each group still incurs one
    device-host synchronization.

    Args:
        lengths: An integer Tensor of shape ``[B]`` (single length) or
            ``[B, K]`` (length tuple).

    Yields:
        ``(value, batch_indices)``: value is a Python int (for 1-D input)
        or a ``list[int]`` (for 2-D input); batch_indices are the row
        indices of the samples in that group.
    """
    if lengths.ndim == 1:
        unique_values, inverse = torch.unique(
            lengths, sorted=True, return_inverse=True)
    else:
        unique_values, inverse = torch.unique(
            lengths, dim=0, sorted=True, return_inverse=True)
    for group_index, value in enumerate(unique_values.tolist()):
        batch_indices = torch.nonzero(
            inverse == group_index, as_tuple=False).squeeze(1)
        yield value, batch_indices


def assemble_flat_groups(
    batch_size: int,
    index_groups: list[Tensor],
    feature_groups: list[Tensor],
    reference: Tensor,
) -> Tensor:
    """Scatter each group's ``[n, D]`` features back into a ``[B, D]``
    zero canvas by batch index.

    Args:
        batch_size: Batch dimension of the output canvas.
        index_groups: Batch indices of the samples in each group.
        feature_groups: Features for each group, with a consistent
            feature dimension.
        reference: Reference Tensor that determines the canvas's
            dtype/device.

    Returns:
        ``[B, D]``, with uncovered positions set to 0.
    """
    canvas = reference.new_zeros((batch_size, feature_groups[0].shape[-1]))
    return canvas.index_copy(
        0, torch.cat(index_groups), torch.cat(feature_groups))


def assemble_padded_groups(
    batch_size: int,
    index_groups: list[Tensor],
    feature_groups: list[Tensor],
    reference: Tensor,
    *,
    total_frames: int | None = None,
) -> Tensor:
    """Pad each group's ``[n, T_g, D]`` features along the time dimension
    and scatter them back into a ``[B, T, D]`` zero canvas.

    Args:
        batch_size: Batch dimension of the output canvas.
        index_groups: Batch indices of the samples in each group.
        feature_groups: Features for each group; the time dimension
            length may differ.
        reference: Reference Tensor that determines the canvas's
            dtype/device.
        total_frames: Target time dimension; when ``None``, the maximum
            across groups is used.

    Returns:
        ``[B, total_frames, D]``, with uncovered positions set to 0.
    """
    if total_frames is None:
        total_frames = max(f.shape[1] for f in feature_groups)
    padded_groups = [
        F.pad(f, (0, 0, 0, total_frames - f.shape[1]))
        for f in feature_groups
    ]
    canvas = reference.new_zeros(
        (batch_size, total_frames, feature_groups[0].shape[-1]))
    return canvas.index_copy(
        0, torch.cat(index_groups), torch.cat(padded_groups))


__all__ = (
    "assemble_flat_groups",
    "assemble_padded_groups",
    "iter_length_groups",
)
