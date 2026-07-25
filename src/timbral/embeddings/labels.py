"""Tri-state label aggregation: generates clip-granularity [C] and
frame-granularity [T, C] labels for a strong cache.

The two primitives, tri-state write and positive-length intersection, are
defined in a single place in ``timbral.datasets.labels`` and reused directly
by this module, keeping the semantics consistent between the two sides.
"""

import numpy as np

from timbral.datasets.labels import intersects_positive, scatter_tri_state

__all__ = ["clip_multihot", "frame_multihot", "scatter_tri_state"]


def clip_multihot(ev_target, ev_value, num_classes: int) -> np.ndarray:
    """Aggregate the events within a slice per class by their tri-state
    annotation into a clip-granularity [C] multi-hot.

    Per class, values are taken in the order 1 > NaN > 0: if a value=1 event
    exists for that class -> 1.0; otherwise if a value=NaN event exists ->
    NaN; otherwise 0.0. Events in the raw cache have already been clipped to
    within the slice (all have a positive-length intersection), so no
    further temporal filtering is needed.

    Args:
        ev_target: Array of event class indices (always valid class
            indices).
        ev_value: Array of event tri-state annotation values (1.0 =
            confirmed present, NaN = uncertain).
        num_classes: Total number of classes.

    Returns:
        np.ndarray: A [C] float32 vector with values in {0.0, 1.0, NaN};
        classes with no events are 0.0.
    """
    vec = np.zeros(num_classes, dtype=np.float32)
    scatter_tri_state(vec, np.asarray(ev_target, dtype=np.int64),
                      np.asarray(ev_value, dtype=np.float32))
    return vec


def frame_multihot(ev_target, ev_start, ev_end, ev_value, geometry,
                   num_classes: int) -> np.ndarray:
    """Aggregate the events within a slice per frame by temporal geometry
    into a frame-granularity [T, C] frame label.

    The slot for frame t is geometry[t] = [s_t, e_t); an event is counted
    towards that frame's class if it has a positive-length intersection with
    the slot (sharing the same intersects_positive predicate used on the
    datasets side, with no minimum overlap threshold); hit events are
    aggregated in the order 1 > NaN > 0 (the flattened (frame, class) index
    reuses scatter_tri_state), and classes with no hits are 0.0. The slot for
    an invalid frame is [0, 0] (zero length, so the intersection comparison
    is always False), so the whole row naturally stays 0.0; frame validity is
    expressed by the separate valid_mask column.

    Args:
        ev_target: Array of event class indices (always valid class
            indices).
        ev_start: Array of event start times relative to the segment, in
            seconds.
        ev_end: Array of event end times relative to the segment, in
            seconds.
        ev_value: Array of event tri-state annotation values (1.0 =
            confirmed present, NaN = uncertain).
        geometry: [T, 2] temporal geometry; each frame is [start, end) in
            seconds, [0, 0] for an invalid frame.
        num_classes: Total number of classes.

    Returns:
        np.ndarray: [T, C] float32; valid frames have values in
        {0.0, 1.0, NaN}, invalid frames are 0.0 across the whole row.
    """
    ev_target = np.asarray(ev_target, dtype=np.int64)
    ev_value = np.asarray(ev_value, dtype=np.float32)
    # Events and slots are unified to float32 before comparison, to avoid a
    # float64/float32 precision mismatch misjudging a zero-length
    # intersection (where a start exactly equals a slot's end) as
    # positive-length.
    ev_start = np.asarray(ev_start, dtype=np.float32)
    ev_end = np.asarray(ev_end, dtype=np.float32)
    geometry = np.asarray(geometry, dtype=np.float32)

    # [T, E] positive-length intersection matrix, computed in one broadcast
    hit = intersects_positive(ev_start[None, :], ev_end[None, :],
                              geometry[:, :1], geometry[:, 1:])
    mat = np.zeros((geometry.shape[0], num_classes), dtype=np.float32)
    frame_idx, event_idx = np.nonzero(hit)
    scatter_tri_state(mat.reshape(-1),
                      frame_idx * num_classes + ev_target[event_idx],
                      ev_value[event_idx])
    return mat
