"""Unit tests for timbral.datasets.labels: indexing/multi-hot/event clipping and aggregation."""

import numpy as np

from timbral.datasets import labels

NAN = float("nan")


def test_build_label_index_lexicographic():
    assert labels.build_label_index(["dog", "cat", "bird"]) == {
        "bird": 0, "cat": 1, "dog": 2}


def test_encode_multihot():
    vec = labels.encode_multihot([0, 2], 4)
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, np.array([1, 0, 1, 0], dtype=np.float32))


def test_encode_multihot_empty():
    vec = labels.encode_multihot([], 3)
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, [0, 0, 0])


def test_clip_events_relative_time_and_clipping():
    # seg interval [10, 20): event 1 fully inside, event 2 crosses the left boundary, event 3 crosses the right boundary
    events = labels.clip_events(
        ev_target=[3, 1, 2],
        ev_start=[12.0, 5.0, 18.0],
        ev_end=[13.0, 11.0, 30.0],
        ev_value=[1.0, 1.0, NAN],
        seg_start=10.0, seg_end=20.0)
    assert events[0] == {"target": 3, "start": 2.0, "end": 3.0, "value": 1.0}
    assert events[1] == {"target": 1, "start": 0.0, "end": 1.0, "value": 1.0}
    # NaN passthrough: NaN != NaN, so assert separately with isnan
    assert events[2]["target"] == 2
    assert events[2]["start"] == 8.0
    assert events[2]["end"] == 10.0
    assert np.isnan(events[2]["value"])
    assert all(isinstance(e["target"], int) for e in events)


def test_clip_events_excludes_non_intersecting():
    # No positive-length intersection with [10, 20): events ending exactly at the left boundary or starting exactly at the right boundary are both excluded
    events = labels.clip_events(
        ev_target=[0, 1, 2],
        ev_start=[0.0, 20.0, 3.0],
        ev_end=[10.0, 25.0, 8.0],
        ev_value=[1.0, 1.0, 1.0],
        seg_start=10.0, seg_end=20.0)
    assert events == []


def test_clipped_events_multihot_pure_pos():
    vec = labels.clipped_events_multihot(
        ev_target=[0, 2, 1],
        ev_start=[11.0, 15.0, 30.0],
        ev_end=[12.0, 25.0, 40.0],
        ev_value=[1.0, 1.0, 1.0],
        seg_start=10.0, seg_end=20.0, num_classes=4)
    # Classes 0 and 2 intersect -> 1; class 1 has no intersection -> 0
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, np.array([1, 0, 1, 0], dtype=np.float32))


def test_clipped_events_multihot_pure_unk():
    vec = labels.clipped_events_multihot(
        ev_target=[1], ev_start=[12.0], ev_end=[13.0], ev_value=[NAN],
        seg_start=10.0, seg_end=20.0, num_classes=3)
    # Only UNK intersects -> that class is NaN, the rest are 0
    assert vec.dtype == np.float32
    assert np.isnan(vec[1])
    np.testing.assert_array_equal(vec[[0, 2]], [0.0, 0.0])


def test_clipped_events_multihot_pos_overrides_unk():
    # Same class has both POS and UNK intersecting events; priority 1 > NaN > 0 takes 1
    vec = labels.clipped_events_multihot(
        ev_target=[1, 1], ev_start=[11.0, 14.0], ev_end=[12.0, 15.0],
        ev_value=[NAN, 1.0], seg_start=10.0, seg_end=20.0, num_classes=2)
    np.testing.assert_array_equal(vec, np.array([0, 1], dtype=np.float32))


def test_clipped_events_multihot_empty_events():
    vec = labels.clipped_events_multihot(
        ev_target=[], ev_start=[], ev_end=[], ev_value=[],
        seg_start=0.0, seg_end=10.0, num_classes=3)
    assert vec.dtype == np.float32
    np.testing.assert_array_equal(vec, [0.0, 0.0, 0.0])
