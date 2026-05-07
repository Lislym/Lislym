"""Tests for analytical two-bone IK and upper-body solver."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.personal import BodySegments
from lislym.ik.two_bone import solve_two_bone
from lislym.ik.upper_body import HipState, HMDState, solve_upper_body


def test_two_bone_reachable_target_satisfies_bone_lengths() -> None:
    root = np.array([0.0, 0.0, 1.0])
    end = np.array([0.0, 0.0, 0.0])  # 1.0 m below root
    a = 0.45
    b = 0.55
    pole = np.array([1.0, 0.0, 0.0])  # bend forward

    sol = solve_two_bone(root, end, a, b, pole)
    assert not sol.reach_clipped
    assert np.linalg.norm(sol.intermediate - root) == pytest.approx(a, abs=1e-6)
    assert np.linalg.norm(sol.end - sol.intermediate) == pytest.approx(b, abs=1e-6)
    np.testing.assert_allclose(sol.end, end, atol=1e-6)


def test_two_bone_pole_chooses_bend_direction() -> None:
    root = np.array([0.0, 0.0, 1.0])
    end = np.array([0.0, 0.0, 0.2])  # within reach with bend room
    a = 0.5
    b = 0.5

    forward = solve_two_bone(root, end, a, b, np.array([1.0, 0.0, 0.0]))
    backward = solve_two_bone(root, end, a, b, np.array([-1.0, 0.0, 0.0]))

    assert forward.intermediate[0] > 0
    assert backward.intermediate[0] < 0
    # Both solutions have the knee at the same height.
    assert forward.intermediate[2] == pytest.approx(backward.intermediate[2])


def test_two_bone_unreachable_target_clips_to_full_extension() -> None:
    root = np.array([0.0, 0.0, 0.0])
    end = np.array([2.0, 0.0, 0.0])  # 2 m away with 1 m total reach
    a = 0.5
    b = 0.5

    sol = solve_two_bone(root, end, a, b, np.array([0.0, 0.0, 1.0]))
    assert sol.reach_clipped
    assert np.linalg.norm(sol.end - root) == pytest.approx(a + b)
    assert np.linalg.norm(sol.intermediate - root) == pytest.approx(a)


def test_two_bone_coincident_root_and_end_keeps_chain_straight() -> None:
    root = np.array([0.0, 0.0, 0.5])
    end = root.copy()
    sol = solve_two_bone(root, end, 0.4, 0.4, np.array([0.0, 0.0, 1.0]))
    assert np.linalg.norm(sol.intermediate - root) == pytest.approx(0.4)


@pytest.mark.parametrize(
    "target",
    [np.array([0.05, 0.0, 0.2]), np.array([-0.05, 0.0, 0.2])],
)
def test_two_bone_pole_dictates_bend_side_regardless_of_step(target: np.ndarray) -> None:
    """The pole picks bend direction independently of small lateral steps."""
    root = np.array([0.0, 0.0, 1.0])
    pole = np.array([1.0, 0.0, 0.0])  # forward
    sol = solve_two_bone(root, target, 0.5, 0.5, pole)
    # Pole=+X means the bend should be on the +X side.
    assert sol.intermediate[0] > 0


def _segments() -> BodySegments:
    return BodySegments(
        leg_length=0.95,
        thigh_length=0.45,
        shin_length=0.50,
        arm_length=0.65,
        spine_length=0.50,
        shoulder_half_width=0.20,
    )


def test_upper_body_neck_and_head_align_with_hmd() -> None:
    seg = _segments()
    hip = HipState(position=np.array([0.0, 0.0, 1.0]), yaw_radians=0.0)
    hmd = HMDState(position=np.array([0.0, 0.0, 1.65]), yaw_radians=0.0)
    sol = solve_upper_body(hip, hmd, np.array([0.55, 0.0, 1.45]), np.array([-0.55, 0.0, 1.45]), seg)
    np.testing.assert_allclose(sol.head, hmd.position)
    assert sol.neck[2] < hmd.position[2]


def test_upper_body_arm_satisfies_two_bone_lengths() -> None:
    seg = _segments()
    hip = HipState(position=np.array([0.0, 0.0, 1.0]), yaw_radians=0.0)
    hmd = HMDState(position=np.array([0.0, 0.0, 1.65]), yaw_radians=0.0)
    cl = np.array([-0.35, 0.10, 1.40])
    cr = np.array([0.35, 0.10, 1.40])
    sol = solve_upper_body(hip, hmd, cl, cr, seg)

    upper_arm = float(np.linalg.norm(sol.elbow_left - sol.shoulder_left))
    forearm = float(np.linalg.norm(sol.wrist_left - sol.elbow_left))
    assert upper_arm == pytest.approx(seg.arm_length / 2, abs=1e-6)
    assert forearm == pytest.approx(seg.arm_length / 2, abs=1e-6)
    np.testing.assert_allclose(sol.wrist_left, cl, atol=1e-6)


def test_upper_body_shoulders_swing_with_hmd_yaw() -> None:
    seg = _segments()
    hip = HipState(position=np.array([0.0, 0.0, 1.0]), yaw_radians=0.0)
    hmd_neutral = HMDState(position=np.array([0.0, 0.0, 1.65]), yaw_radians=0.0)
    hmd_left = HMDState(position=np.array([0.0, 0.0, 1.65]), yaw_radians=np.deg2rad(45))

    cl = np.array([-0.55, 0.0, 1.40])
    cr = np.array([0.55, 0.0, 1.40])

    base = solve_upper_body(hip, hmd_neutral, cl, cr, seg)
    twisted = solve_upper_body(hip, hmd_left, cl, cr, seg)

    # The shoulder line should have rotated CCW (right shoulder moves +Y).
    assert twisted.shoulder_right[1] > base.shoulder_right[1] + 0.02
