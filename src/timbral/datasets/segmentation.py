"""Segmentation plan: given an entry region and the audio duration, compute each segment's start time and valid length."""

import numpy as np


def plan_segments(entry_start: float, entry_end: float, audio_duration: float,
                  seg_sec: float, hop_sec: float, tol_sec: float) -> np.ndarray:
    """Compute the segmentation plan for a single split entry.

    The valid region is [entry_start, min(entry_end, audio_duration));
    segment starts advance from entry_start with a step of hop_sec;
    valid_sec = min(seg_sec, region end - seg_start); the tol_sec filter
    applies only to segments with seg_id > 0 (segments with
    valid_sec > tol_sec are kept); segment 0 is always kept unconditionally,
    guaranteeing that every entry produces at least one segment.

    Args:
        entry_start: Entry start time within the original audio, in
            seconds.
        entry_end: Entry end time within the original audio, in seconds
            (inf for the whole recording).
        audio_duration: Measured total audio duration, in seconds.
        seg_sec: Segment length, in seconds.
        hop_sec: Segment hop length, in seconds.
        tol_sec: Minimum retained length for a trailing segment, in
            seconds; does not apply to segment 0.

    Returns:
        np.ndarray: shape (n, 2) float64, each row is (seg_start,
        valid_sec); seg_start is the absolute time within the original
        audio, in seconds.
    """
    region_end = min(entry_end, audio_duration)
    if region_end <= entry_start:
        return np.empty((0, 2), dtype=np.float64)
    num = int(np.ceil((region_end - entry_start) / hop_sec))
    starts = entry_start + hop_sec * np.arange(num, dtype=np.float64)
    valid = np.minimum(seg_sec, region_end - starts)
    keep = valid > tol_sec
    keep[0] = True
    return np.stack([starts[keep], valid[keep]], axis=1)
