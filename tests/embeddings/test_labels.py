"""Pure-function tests for timbral.embeddings.labels tri-state label aggregation."""

import numpy as np

from timbral.embeddings.labels import (clip_multihot, frame_multihot,
                                       scatter_tri_state)

# Geometry shared by frame test cases: 3 valid frames (FRAME_SEC=0.25, the
# last frame's endpoint absorbed to valid=1.0)
_GEOMETRY = np.asarray([[0.0, 0.25], [0.25, 0.5], [0.5, 1.0]],
                       dtype=np.float32)


def _assert_tri_state(actual, expected):
    """Tri-state matrix assertion: NaN positions are checked via isnan, other positions elementwise equal."""
    actual = np.asarray(actual, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    nan_mask = np.isnan(expected)
    np.testing.assert_array_equal(np.isnan(actual), nan_mask)
    np.testing.assert_array_equal(actual[~nan_mask], expected[~nan_mask])


def test_scatter_tri_state_priority_and_untouched():
    vec = np.zeros(4, dtype=np.float32)
    # position 0 same-slot POS+UNK conflict -> 1; position 1 pure UNK -> NaN;
    # position 3 no event -> 0
    scatter_tri_state(vec, np.asarray([0, 0, 1, 2], dtype=np.int64),
                      np.asarray([np.nan, 1.0, np.nan, 1.0],
                                 dtype=np.float32))
    _assert_tri_state(vec, [1.0, np.nan, 1.0, 0.0])


def test_clip_multihot_pos_unk_and_empty():
    _assert_tri_state(clip_multihot([2, 1], [1.0, np.nan], 3),
                      [0.0, np.nan, 1.0])
    # same-class POS+UNK conflict -> 1
    _assert_tri_state(clip_multihot([1, 1], [1.0, np.nan], 3),
                      [0.0, 1.0, 0.0])
    # empty event slice -> all 0
    _assert_tri_state(clip_multihot([], [], 3), [0.0, 0.0, 0.0])


def test_frame_multihot_positive_intersection_only():
    # dog(2)[0.2,0.3) crosses the frame 0/1 boundary -> both frames set;
    # cat(1) POS start == frame 0 slot's end (zero-length intersection) ->
    # frame 1 only;
    # cat(1) UNK end == frame 1 slot's start -> frame 0 only;
    # bird(0) falls in the last frame's absorption region [0.75, 1.0) ->
    # frame 2 only
    mat = frame_multihot([2, 1, 1, 0],
                         [0.2, 0.25, 0.1, 0.8],
                         [0.3, 0.4, 0.25, 0.95],
                         [1.0, 1.0, np.nan, 1.0],
                         _GEOMETRY, 3)
    _assert_tri_state(mat, [[0.0, np.nan, 1.0],
                            [0.0, 1.0, 1.0],
                            [1.0, 0.0, 0.0]])


def test_frame_multihot_same_class_conflict_and_empty():
    # frame 0 same-class (dog=2) POS+UNK -> 1; frame 1 different-class
    # POS (bird=0) and UNK (cat=1) coexist
    mat = frame_multihot([2, 2, 0, 1],
                         [0.05, 0.12, 0.26, 0.3],
                         [0.1, 0.2, 0.45, 0.4],
                         [1.0, np.nan, 1.0, np.nan],
                         _GEOMETRY, 3)
    _assert_tri_state(mat, [[0.0, 0.0, 1.0],
                            [1.0, np.nan, 0.0],
                            [0.0, 0.0, 0.0]])
    _assert_tri_state(frame_multihot([], [], [], [], _GEOMETRY, 3),
                      np.zeros((3, 3)))


def test_frame_multihot_padding_slot_stays_zero():
    # Invalid frame slot [0, 0]: zero-length intersection is always False,
    # so even an event covering the vicinity of 0 seconds does not set it;
    # the whole row stays 0
    geometry = np.asarray([[0.0, 0.25], [0.25, 0.5], [0.0, 0.0]],
                          dtype=np.float32)
    mat = frame_multihot([0, 1], [0.0, 0.3], [0.5, 0.5], [1.0, np.nan],
                         geometry, 2)
    _assert_tri_state(mat, [[1.0, 0.0],
                            [1.0, np.nan],
                            [0.0, 0.0]])


def test_frame_multihot_mixed_precision_boundary():
    # The event start is float64 0.25 + a tiny epsilon, which rounds back to
    # 0.25 in float32: this is a zero-length intersection with frame 0
    # slot's end, and must not be misjudged as a positive-length one
    mat = frame_multihot([0], [0.25 + 1e-12], [0.4], [1.0], _GEOMETRY, 1)
    _assert_tri_state(mat, [[0.0], [1.0], [0.0]])
