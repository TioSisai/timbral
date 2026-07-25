"""Label indexing/encoding and clipping/aggregation of strong events."""

import numpy as np


def build_label_index(classes) -> dict:
    """Number class names 0-based in lexicographic order, returning {class_name: index}."""
    return {name: idx for idx, name in enumerate(sorted(classes))}


def encode_multihot(label_indices, num_classes: int) -> np.ndarray:
    """Convert a list of label indices to a float32 multi-hot vector (empty list -> all zeros)."""
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[np.asarray(label_indices, dtype=np.int64)] = 1.0
    return vec


def intersects_positive(ev_start, ev_end, region_start, region_end):
    """Positive-length intersection test (arguments are broadcastable): ev_start < region_end and ev_end > region_start.

    Segment clipping on the datasets side and frame label generation on the
    embeddings side share this predicate, ensuring the intersection
    convention is enforced by shared code rather than by comment
    convention alone.

    Args:
        ev_start: Event start time (seconds), broadcastable.
        ev_end: Event end time (seconds), broadcastable.
        region_start: Target region start time (seconds), broadcastable.
        region_end: Target region end time (seconds), broadcastable.

    Returns:
        np.ndarray: Bool array of the broadcast shape; True means the
        intersection length is positive.
    """
    return (ev_start < region_end) & (ev_end > region_start)


def scatter_tri_state(vec: np.ndarray, ev_target: np.ndarray,
                      ev_value: np.ndarray) -> None:
    """Write the tri-state event values into vec at the corresponding positions in place, following the priority 1 > NaN > 0.

    NaN positions are written first, then 1.0 positions are written to
    overwrite them, so conflicts at the same position naturally satisfy
    the priority 1 > NaN > 0; positions not written by any event keep
    vec's original value (the caller gets 0.0 by initializing to all
    zeros). The frame path reuses this primitive by flattening the
    [T, C] matrix and indexing with frame*C + class.

    Args:
        vec: 1-D float32 vector to write into, modified in place.
        ev_target: Array of event write-position indices (int64).
        ev_value: Array of tri-state event annotation values (float32,
            1.0 = confirmed present, NaN = uncertain).
    """
    is_unknown = np.isnan(ev_value)
    vec[ev_target[is_unknown]] = np.nan
    vec[ev_target[~is_unknown]] = 1.0


def clip_events(ev_target, ev_start, ev_end, ev_value,
                seg_start: float, seg_end: float) -> list:
    """Clip the events of an entire audio recording to a segment and convert to time relative to the segment.

    Args:
        ev_target: Array of event class indices (always valid class
            indices).
        ev_start: Array of event absolute start times within the original
            audio, in seconds.
        ev_end: Array of event absolute end times within the original
            audio, in seconds.
        ev_value: Array of tri-state event annotation values (1.0 =
            confirmed present, NaN = uncertain).
        seg_start: Absolute start time of the segment's valid region
            within the original audio, in seconds.
        seg_end: Absolute end time of the segment's valid region within
            the original audio, in seconds.

    Returns:
        list[dict]: [{"target": int, "start": float, "end": float,
        "value": float}, ...]; value is passed through unchanged, times
        are relative to the segment and clipped to
        [0, seg_end - seg_start]; events with no intersection are
        excluded.
    """
    ev_target = np.asarray(ev_target, dtype=np.int64)
    ev_start = np.asarray(ev_start, dtype=np.float64)
    ev_end = np.asarray(ev_end, dtype=np.float64)
    ev_value = np.asarray(ev_value, dtype=np.float32)
    keep = intersects_positive(ev_start, ev_end, seg_start, seg_end)
    rel_start = np.maximum(ev_start[keep], seg_start) - seg_start
    rel_end = np.minimum(ev_end[keep], seg_end) - seg_start
    return [{"target": t, "start": s, "end": e, "value": v}
            for t, s, e, v in zip(ev_target[keep].tolist(), rel_start.tolist(),
                                  rel_end.tolist(), ev_value[keep].tolist())]


def clipped_events_multihot(ev_target, ev_start, ev_end, ev_value,
                            seg_start: float, seg_end: float,
                            num_classes: int) -> np.ndarray:
    """Aggregate segment-intersecting events per class into a multi-hot vector using the tri-state annotation (strong -> weak downgrade).

    Per class, the value is taken by priority 1 > NaN > 0: if there is an
    intersecting event with value=1 for that class -> 1.0; else if there
    is an intersecting event with value=NaN -> NaN; else 0.0.

    Args:
        ev_target: Array of event class indices (always valid class
            indices).
        ev_start: Array of event absolute start times within the original
            audio, in seconds.
        ev_end: Array of event absolute end times within the original
            audio, in seconds.
        ev_value: Array of tri-state event annotation values (1.0 =
            confirmed present, NaN = uncertain).
        seg_start: Absolute start time of the segment's valid region
            within the original audio, in seconds.
        seg_end: Absolute end time of the segment's valid region within
            the original audio, in seconds.
        num_classes: Total number of classes.

    Returns:
        np.ndarray: float32 multi-hot vector with values in
        {0.0, 1.0, NaN}.
    """
    ev_target = np.asarray(ev_target, dtype=np.int64)
    ev_start = np.asarray(ev_start, dtype=np.float64)
    ev_end = np.asarray(ev_end, dtype=np.float64)
    ev_value = np.asarray(ev_value, dtype=np.float32)
    keep = intersects_positive(ev_start, ev_end, seg_start, seg_end)
    vec = np.zeros(num_classes, dtype=np.float32)
    scatter_tri_state(vec, ev_target[keep], ev_value[keep])
    return vec
