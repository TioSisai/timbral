"""Model-agnostic construction of clip/frame output geometry and
validity masks (shared by all Encoders).
"""

from __future__ import annotations

import torch
from torch import Tensor


def build_clip_outputs(valid_seconds: Tensor) -> tuple[Tensor, Tensor]:
    """Construct clip-level geometry ``[B, 2]`` and an all-True valid_mask
    ``[B]``.

    At clip granularity, each sample has exactly one slot
    ``[0, valid_seconds)``, which is always valid.

    Args:
        valid_seconds: ``[B]`` valid seconds per sample.

    Returns:
        ``(geometry, valid_mask)``.
    """
    valid_seconds = valid_seconds.to(torch.float32)
    geometry = torch.stack(
        (torch.zeros_like(valid_seconds), valid_seconds), dim=1)
    valid_mask = torch.ones(
        (valid_seconds.shape[0],),
        dtype=torch.bool,
        device=valid_seconds.device,
    )
    return geometry, valid_mask


def build_frame_geometry(
    valid_frames: Tensor,
    valid_seconds: Tensor,
    *,
    total_frames: int,
    step_seconds: float,
) -> tuple[Tensor, Tensor]:
    """Construct geometry ``[B, T, 2]`` and valid_mask ``[B, T]`` for a
    fixed-step frame grid.

    The slot for frame t is ``[t*step, (t+1)*step)``; the endpoint of the
    last valid frame is absorbed into ``valid_seconds`` (intermediate
    boundaries that overshoot are also clamped to ``valid_seconds``);
    invalid frame slots are zeroed to ``[0, 0]``, with valid_mask as the
    sole source of truth for validity.

    Args:
        valid_frames: ``[B]`` valid frame count per sample (>= 1).
        valid_seconds: ``[B]`` valid seconds per sample.
        total_frames: Length T of the output frame axis.
        step_seconds: Interval (in seconds) between consecutive frame
            starts.

    Returns:
        ``(geometry, valid_mask)``.
    """
    device = valid_seconds.device
    valid_seconds = valid_seconds.to(torch.float32)
    frame_indices = torch.arange(total_frames, device=device)
    valid_mask = frame_indices.unsqueeze(0) < valid_frames.unsqueeze(1)
    frame_boundaries = (
        torch.arange(total_frames + 1, device=device, dtype=torch.float32)
        * step_seconds
    )
    starts = frame_boundaries[:-1].unsqueeze(0).expand(
        valid_seconds.shape[0], -1)
    ends = torch.minimum(
        frame_boundaries[1:].unsqueeze(0),
        valid_seconds.unsqueeze(1),
    )
    ends = ends.scatter(
        1,
        (valid_frames - 1).unsqueeze(1),
        valid_seconds.unsqueeze(1),
    )
    geometry = torch.stack((starts, ends), dim=2) * valid_mask.unsqueeze(2)
    return geometry, valid_mask


__all__ = ("build_clip_outputs", "build_frame_geometry")
