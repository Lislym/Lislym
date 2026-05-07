"""Tests for the auto-capture buffer used by both calibration tools."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.tools.auto_capture import (
    AutoCaptureBuffer,
    FrameCandidate,
    diversity_distance,
    laplacian_sharpness,
    points_bbox,
    pose_signature,
)


def _candidate(signature: np.ndarray, sharpness: float = 200.0) -> FrameCandidate:
    return FrameCandidate(
        image=np.zeros((4, 4, 3), dtype=np.uint8), signature=signature, sharpness=sharpness
    )


def test_first_offer_accepted_unconditionally() -> None:
    buf = AutoCaptureBuffer(target_size=5, min_diversity=10.0)
    out = buf.offer(_candidate(np.array([0.0, 0.0, 0.0, 0.0, 0.0])))
    assert out == "accepted"
    assert len(buf) == 1


def test_low_sharpness_rejected() -> None:
    buf = AutoCaptureBuffer(target_size=5, min_sharpness=100.0, min_diversity=1.0)
    out = buf.offer(_candidate(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), sharpness=50.0))
    assert out == "rejected_sharpness"
    assert len(buf) == 0


def test_similar_frame_rejected_for_diversity() -> None:
    buf = AutoCaptureBuffer(target_size=5, min_diversity=50.0)
    buf.offer(_candidate(np.array([100.0, 100.0, 200.0, 200.0, 100.0])))
    out = buf.offer(_candidate(np.array([105.0, 102.0, 198.0, 201.0, 100.0])))
    assert out == "rejected_diversity"
    assert len(buf) == 1


def test_diverse_frame_accepted_until_full() -> None:
    buf = AutoCaptureBuffer(target_size=3, min_diversity=10.0)
    for k in range(5):
        out = buf.offer(_candidate(np.array([100.0 * k, 0.0, 0.0, 0.0, 0.0])))
        if k < 3:
            assert out == "accepted"
    assert buf.is_full()
    assert len(buf) == 3


def test_better_candidate_replaces_worst_when_buffer_full() -> None:
    buf = AutoCaptureBuffer(target_size=2, min_diversity=10.0, min_sharpness=10.0)
    buf.offer(_candidate(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), sharpness=50.0))
    buf.offer(_candidate(np.array([100.0, 0.0, 0.0, 0.0, 0.0]), sharpness=200.0))
    # Buffer full: low-quality frame at 50 should be evicted by a high-quality
    # diverse newcomer.
    out = buf.offer(_candidate(np.array([200.0, 0.0, 0.0, 0.0, 0.0]), sharpness=400.0))
    assert out == "replaced"
    assert all(c.sharpness >= 200 for c in buf.accepted)


def test_replacement_respects_diversity_against_remaining() -> None:
    """A high-quality but redundant frame should still be rejected."""
    buf = AutoCaptureBuffer(target_size=2, min_diversity=50.0, min_sharpness=10.0)
    buf.offer(_candidate(np.array([0.0, 0.0, 0.0, 0.0, 0.0]), sharpness=80.0))
    buf.offer(_candidate(np.array([200.0, 0.0, 0.0, 0.0, 0.0]), sharpness=80.0))
    # New frame is sharper but lives ON TOP of an existing one.
    out = buf.offer(_candidate(np.array([2.0, 0.0, 0.0, 0.0, 0.0]), sharpness=400.0))
    assert out == "rejected_diversity"
    assert len(buf) == 2


def test_diversity_distance_inf_when_no_accepted() -> None:
    assert diversity_distance(np.zeros(3), []) == float("inf")


def test_pose_signature_shape_and_orientation() -> None:
    pts = np.array([[10.0, 20.0], [30.0, 20.0], [10.0, 60.0], [30.0, 60.0]])
    sig = pose_signature(pts)
    assert sig.shape == (5,)
    np.testing.assert_allclose(sig[0:2], [20.0, 40.0])
    np.testing.assert_allclose(sig[2:4], [20.0, 40.0])  # bbox w, h


def test_points_bbox_is_inclusive() -> None:
    pts = np.array([[5.5, 7.2], [9.1, 11.9]])
    bbox = points_bbox(pts)
    assert bbox == (5, 7, 5, 5)  # x=5, y=7, w=ceil(9.1)-5=5, h=ceil(11.9)-7=5


def test_laplacian_sharpness_higher_for_edged_image() -> None:
    flat = np.full((100, 100), 128, dtype=np.uint8)
    striped = np.tile(np.array([0, 255], dtype=np.uint8), (100, 50))
    assert laplacian_sharpness(striped) > laplacian_sharpness(flat) * 100
