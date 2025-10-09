import numpy as np

from optimization.constraints import penalty_ordering, round_to_nearest_50


def test_penalty_ordering_hierarchy_satisfied():
    # To yield zero penalty we need:
    # segment_means[left] - segment_means[right] <= 0 (non-increasing)
    # size_means[right] - size_means[left] <= 0 (i.e. non-decreasing sequence)
    # We'll make all equal so diffs == 0.
    nr = np.array([[100, 100, 100]])
    vol = np.array([[10, 10, 10]])
    segment_labels = np.array([["Value", "Core", "Premium"]])
    size_labels = np.array([["Large", "Regular", "Small"]])
    segment_order = ["Value", "Core", "Premium"]
    size_order = ["Large", "Regular", "Small"]
    p = penalty_ordering(
        nr, vol, segment_labels, size_labels,
        segment_order, size_order,
        seg_lambda=1000, size_lambda=1000,
    )
    assert p == 0.0


def test_penalty_ordering_violation_segment():
    nr = np.array([[300, 200, 100]])  # reversed revenue
    vol = np.array([[10, 10, 10]])
    segment_labels = np.array([["Value", "Core", "Premium"]])
    size_labels = np.array([["Large", "Regular", "Small"]])
    segment_order = ["Value", "Core", "Premium"]
    size_order = ["Large", "Regular", "Small"]
    p = penalty_ordering(
        nr, vol, segment_labels, size_labels,
        segment_order, size_order,
        seg_lambda=1.0, size_lambda=0.0,
    )
    assert p > 0  # segment violation adds penalty


def test_round_to_nearest_50():
    ref = np.array([100, 150, 200])
    new = np.array([120, 193, 141])
    rounded = round_to_nearest_50(new, ref)
    # diffs: +20 -> +0 (nearest 0); +43 -> +50; -59 -> -50
    assert np.array_equal(rounded, np.array([100, 200, 150]))
