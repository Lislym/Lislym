"""Tests for the occlusion-and-return contract from plan §C-4 / C-5.

The user emphasised: a marker leaving any camera's view (or all cameras'
views) must not break tracking when it returns. These tests exercise the
EKF, the gating schedule, identity re-association, and the K=1 fallback.
"""

from __future__ import annotations

import numpy as np
import pytest

from lislym.fusion.marker_state import (
    CandidateDetection,
    GatingSchedule,
    MarkerState,
    MarkerTracker,
    reassign_markers,
)
from lislym.fusion.single_ray import (
    Ray,
    solve_with_distance_to_anchor,
    solve_with_distance_to_prior,
)


# --- MarkerState ---


def test_state_initialises_on_first_observation() -> None:
    s = MarkerState(name="hip", color="red")
    s.update(np.array([0.1, 0.2, 1.0]), np.eye(3) * 1e-4, observation_time=0.0)
    np.testing.assert_allclose(s.position, [0.1, 0.2, 1.0])
    assert s.initialised


def test_state_predict_extrapolates_velocity() -> None:
    """The filter learns velocity from a sequence of predict→update steps."""
    s = MarkerState(name="ankle_left", color="green")
    s.update(np.array([0.0, 0.0, 0.0]), np.eye(3) * 1e-4, observation_time=0.0)
    # Walk along +x at 2 m/s for 0.5 s to seed velocity.
    for k in range(1, 11):
        s.predict(dt=0.05)
        s.update(np.array([0.1 * k, 0.0, 0.0]), np.eye(3) * 1e-4, observation_time=0.05 * k)
    # Now predict 0.1 s further with no observation: should extrapolate.
    pos_before = s.position[0]
    s.predict(dt=0.1)
    assert s.position[0] - pos_before == pytest.approx(0.2, abs=0.05)
    assert abs(s.velocity[0] - 2.0) < 0.3


def test_state_uncertainty_grows_during_occlusion() -> None:
    s = MarkerState(name="hip", color="red")
    s.update(np.array([0.0, 0.0, 1.0]), np.eye(3) * 1e-4, observation_time=0.0)
    sigma_before = s.position_uncertainty()
    for _ in range(20):  # 20 * 16ms ≈ 320 ms occluded
        s.predict(dt=0.016)
    sigma_after = s.position_uncertainty()
    assert sigma_after > sigma_before * 5  # growth, not stagnation


# --- Gating schedule ---


def test_gating_radius_grows_with_time() -> None:
    g = GatingSchedule(initial_m=0.5, growth_m_per_s=0.5, max_m=2.0)
    assert g.radius(0.0) == pytest.approx(0.5)
    assert g.radius(1.0) == pytest.approx(1.0)
    assert g.radius(10.0) == pytest.approx(2.0)  # capped


# --- Tracker / re-association ---


def _make_tracker() -> MarkerTracker:
    tr = MarkerTracker()
    tr.register("hip", "red")
    tr.register("ankle_left", "green")
    tr.register("ankle_right", "blue")
    return tr


def test_reassign_picks_color_match() -> None:
    tr = _make_tracker()
    R = np.eye(3) * 1e-4
    tr.update("hip", np.array([0.0, 0.0, 1.0]), R, 0.0)
    tr.update("ankle_left", np.array([-0.1, 0.0, 0.05]), R, 0.0)
    tr.update("ankle_right", np.array([0.1, 0.0, 0.05]), R, 0.0)

    # New detections, all near the predictions.
    detections = [
        CandidateDetection(np.array([0.05, 0.0, 1.0]), color="red"),
        CandidateDetection(np.array([-0.1, 0.0, 0.05]), color="green"),
        CandidateDetection(np.array([0.1, 0.0, 0.05]), color="blue"),
    ]
    out = reassign_markers(tr, detections, t_now=0.05)
    assert out == {"hip": 0, "ankle_left": 1, "ankle_right": 2}


def test_reassign_rejects_detection_outside_gate_after_short_loss() -> None:
    tr = _make_tracker()
    tr.gating = GatingSchedule(initial_m=0.5, growth_m_per_s=0.5, max_m=2.0)
    R = np.eye(3) * 1e-4
    tr.update("hip", np.array([0.0, 0.0, 1.0]), R, 0.0)

    # 0.4 s later, a red blob appears 1.5 m away — beyond the gate of
    # 0.5 + 0.5 * 0.4 = 0.7 m → must NOT be assigned.
    detections = [CandidateDetection(np.array([1.5, 0.0, 1.0]), color="red")]
    out = reassign_markers(tr, detections, t_now=0.4)
    assert "hip" not in out


def test_reassign_after_long_occlusion_uses_grown_gate() -> None:
    tr = _make_tracker()
    tr.gating = GatingSchedule(initial_m=0.5, growth_m_per_s=0.5, max_m=2.0)
    R = np.eye(3) * 1e-4
    tr.update("hip", np.array([0.0, 0.0, 1.0]), R, 0.0)
    # 3 s later: the gate has grown to 2.0 m (capped) so a red blob 1.8 m
    # away is acceptable.
    detections = [CandidateDetection(np.array([1.8, 0.0, 1.0]), color="red")]
    out = reassign_markers(tr, detections, t_now=3.0)
    assert out.get("hip") == 0


def test_reassign_with_bone_constraints_resolves_left_right_swap() -> None:
    """Two same-distance ankles should still go to the right hip-anchored slot."""
    tr = _make_tracker()
    tr.gating = GatingSchedule(initial_m=2.0, growth_m_per_s=0.0, max_m=2.0)
    R = np.eye(3) * 1e-4
    tr.update("hip", np.array([0.0, 0.0, 1.0]), R, 0.0)
    tr.update("ankle_left", np.array([-0.10, 0.0, 0.05]), R, 0.0)
    tr.update("ankle_right", np.array([0.10, 0.0, 0.05]), R, 0.0)

    # Detections come back with correctly colored ankles but slightly
    # shifted; the bone-length sanity check should keep colors paired
    # with the corresponding tracked names.
    detections = [
        CandidateDetection(np.array([0.0, 0.0, 1.05]), color="red"),
        CandidateDetection(np.array([-0.12, 0.0, 0.06]), color="green"),
        CandidateDetection(np.array([0.12, 0.0, 0.06]), color="blue"),
    ]
    out = reassign_markers(
        tr,
        detections,
        bone_constraints={
            ("hip", "ankle_left"): (0.7, 1.1),
            ("hip", "ankle_right"): (0.7, 1.1),
        },
        t_now=0.05,
    )
    # Even with the bone constraint penalising the (currently-near) ankle
    # configuration, color match still pins the assignment correctly.
    assert out["ankle_left"] == 1
    assert out["ankle_right"] == 2


# --- Single-ray (K=1) fallback ---


def test_single_ray_anchor_intersect_picks_root_near_prior() -> None:
    # Anchor at origin; ray along +x at z=1; bone length 1 m.
    ray = Ray(origin=np.array([2.0, 0.0, 1.0]), direction=np.array([-1.0, 0.0, 0.0]))
    point, reason = solve_with_distance_to_anchor(
        ray=ray,
        anchor=np.array([0.0, 0.0, 1.0]),
        target_distance_m=1.0,
        prior=np.array([1.05, 0.0, 1.0]),
    )
    assert reason == "intersect"
    np.testing.assert_allclose(point, [1.0, 0.0, 1.0], atol=1e-9)


def test_single_ray_no_intersection_projects_prior() -> None:
    ray = Ray(origin=np.array([5.0, 0.0, 1.0]), direction=np.array([-1.0, 0.0, 0.0]))
    point, reason = solve_with_distance_to_anchor(
        ray=ray,
        anchor=np.array([0.0, 5.0, 1.0]),  # 5 m from the closest point on the ray
        target_distance_m=1.0,
        prior=np.array([2.0, 0.0, 1.0]),
    )
    assert reason == "projected"
    np.testing.assert_allclose(point, [2.0, 0.0, 1.0])


def test_single_ray_distance_to_prior_clamps_step() -> None:
    ray = Ray(origin=np.array([0.0, 0.0, 0.0]), direction=np.array([1.0, 0.0, 0.0]))
    out = solve_with_distance_to_prior(
        ray=ray,
        prior=np.array([0.5, 1.0, 0.0]),
        max_step_m=0.2,
    )
    delta = float(np.linalg.norm(out - np.array([0.5, 1.0, 0.0])))
    assert delta <= 0.2 + 1e-9


# --- End-to-end: occlusion across frames ---


def test_full_occlusion_then_return_keeps_identity() -> None:
    """Simulate: hip seen → occluded for 2 s → returns near predicted position.

    This is the headline scenario from plan §C-5: identity persistence
    across full visual loss. We verify the tracker re-acquires the right
    name with no swap.
    """
    tr = _make_tracker()
    R = np.eye(3) * 1e-4
    # Walk forward at 0.1 m/s for 1 s to seed velocity.
    for k, t in enumerate(np.arange(0.0, 1.0, 0.05)):
        pos = np.array([0.1 * t, 0.0, 1.0])
        tr.update("hip", pos, R, t)
        del k

    # 2 s of occlusion: predict only.
    for _ in range(40):
        for s in tr.states.values():
            s.predict(0.05)

    # New detection appears near the predicted position with the right color.
    pred = tr.states["hip"].position
    detections = [
        CandidateDetection(pred + np.array([0.05, 0.0, 0.0]), color="red"),
        # A distractor on the floor — wrong color, must be ignored.
        CandidateDetection(np.array([2.0, 1.5, 0.0]), color="purple"),
    ]
    out = reassign_markers(tr, detections, t_now=3.0)
    assert out == {"hip": 0}


def test_color_distractor_does_not_capture_identity() -> None:
    """A red object across the room must not steal the hip identity."""
    tr = _make_tracker()
    R = np.eye(3) * 1e-4
    tr.update("hip", np.array([0.0, 0.0, 1.0]), R, 0.0)

    detections = [
        CandidateDetection(np.array([3.0, 0.0, 1.0]), color="red"),  # too far
    ]
    out = reassign_markers(tr, detections, t_now=0.1)
    assert "hip" not in out
