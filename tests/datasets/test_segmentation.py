"""Unit tests for timbral.datasets.segmentation: segment-plan boundary semantics."""

import numpy as np

from timbral.datasets.segmentation import plan_segments

INF = float("inf")


def test_whole_file_exact_division():
    plan = plan_segments(0.0, INF, 20.0, 10.0, 10.0, 0.0)
    np.testing.assert_allclose(plan, [[0.0, 10.0], [10.0, 10.0]])


def test_tail_kept_when_tol_zero():
    plan = plan_segments(0.0, INF, 25.0, 10.0, 10.0, 0.0)
    np.testing.assert_allclose(plan, [[0.0, 10.0], [10.0, 10.0], [20.0, 5.0]])


def test_tail_filtered_by_tol():
    plan = plan_segments(0.0, INF, 20.5, 10.0, 10.0, 1.0)
    np.testing.assert_allclose(plan, [[0.0, 10.0], [10.0, 10.0]])


def test_first_segment_kept_despite_tol():
    plan = plan_segments(0.0, INF, 0.5, 10.0, 10.0, 1.0)
    np.testing.assert_allclose(plan, [[0.0, 0.5]])


def test_first_segment_kept_despite_tol_in_subregion():
    plan = plan_segments(20.0, 40.0, 20.5, 10.0, 10.0, 1.0)
    np.testing.assert_allclose(plan, [[20.0, 0.5]])


def test_short_file_single_padded_segment():
    plan = plan_segments(0.0, INF, 5.0, 10.0, 10.0, 0.0)
    np.testing.assert_allclose(plan, [[0.0, 5.0]])


def test_overlapping_hop():
    plan = plan_segments(0.0, INF, 5.0, 10.0, 2.0, 0.0)
    np.testing.assert_allclose(plan, [[0.0, 5.0], [2.0, 3.0], [4.0, 1.0]])


def test_entry_subregion():
    plan = plan_segments(20.0, 40.0, 100.0, 10.0, 10.0, 0.0)
    np.testing.assert_allclose(plan, [[20.0, 10.0], [30.0, 10.0]])


def test_entry_clamped_by_duration():
    plan = plan_segments(20.0, 40.0, 25.0, 10.0, 10.0, 0.0)
    np.testing.assert_allclose(plan, [[20.0, 5.0]])


def test_empty_region():
    plan = plan_segments(30.0, 40.0, 25.0, 10.0, 10.0, 0.0)
    assert plan.shape == (0, 2)
