"""Upper-body IK from HMD + controllers.

Plan §D-4. With:

* hip 6 DoF (position + yaw), recovered from the two-ball hip belt
* HMD 6 DoF
* left and right controller 6 DoF

we estimate the spine and arm joints by:

1. Stacking spine vertebrae linearly between the hip and a neck point
   derived from the HMD (head shifted down by ``head_offset_m`` along
   world up).
2. Twisting the spine: the rotation difference between hip yaw and HMD
   yaw is split evenly across the spine joints (chest gets ~60 %, upper
   spine ~40 % to keep the head looking where the HMD points).
3. Placing the shoulders rigidly to the chest joint at
   ``±shoulder_half_width`` from the chest centre.
4. Solving each arm with two-bone IK from the shoulder to the controller,
   with a pole vector along the body's forward axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lislym.calibration.personal import BodySegments
from lislym.ik.two_bone import solve_two_bone


@dataclass(frozen=True)
class HipState:
    """Hip pose driving the spine root."""

    position: np.ndarray
    yaw_radians: float


@dataclass(frozen=True)
class HMDState:
    """HMD pose. Yaw is the heading; pitch/roll are unused here."""

    position: np.ndarray
    yaw_radians: float


@dataclass(frozen=True)
class UpperBodyJoints:
    """Output joints from the upper-body solve, all in world coordinates."""

    chest: np.ndarray
    neck: np.ndarray
    head: np.ndarray
    shoulder_left: np.ndarray
    elbow_left: np.ndarray
    wrist_left: np.ndarray
    shoulder_right: np.ndarray
    elbow_right: np.ndarray
    wrist_right: np.ndarray


def _yaw_to_forward(yaw: float) -> np.ndarray:
    """+X-forward at yaw=0; rotates CCW around world +Z."""
    return np.array([np.cos(yaw), np.sin(yaw), 0.0])


def _yaw_to_right(yaw: float) -> np.ndarray:
    """Right-hand axis of the body at the given yaw."""
    return np.array([np.sin(yaw), -np.cos(yaw), 0.0])


def solve_upper_body(
    hip: HipState,
    hmd: HMDState,
    controller_left: np.ndarray,
    controller_right: np.ndarray,
    segments: BodySegments,
    *,
    head_offset_m: float = 0.10,
    chest_fraction: float = 0.60,
) -> UpperBodyJoints:
    """Solve the spine + both arms in one closed-form pass.

    The chest is placed ``chest_fraction * spine_length`` along the
    hip→neck axis so the upper torso bends with the HMD's heading while
    the lower torso stays aligned with the hip.
    """
    head = hmd.position.copy()
    neck = hmd.position - np.array([0.0, 0.0, head_offset_m])

    # Spine direction is just hip→neck; length is enforced from the
    # personal segment lengths but we keep direction from observation
    # so a slight forward lean is preserved.
    spine_axis = neck - hip.position
    spine_norm = float(np.linalg.norm(spine_axis))
    if spine_norm < 1e-6:
        spine_axis = np.array([0.0, 0.0, 1.0])
    else:
        spine_axis /= spine_norm
    chest = hip.position + chest_fraction * segments.spine_length * spine_axis

    # Distribute yaw twist: the chest takes the larger share so the
    # shoulders track the head naturally; the lower spine takes the rest.
    chest_yaw = hip.yaw_radians + chest_fraction * (hmd.yaw_radians - hip.yaw_radians)
    right_axis = _yaw_to_right(chest_yaw)
    forward_axis = _yaw_to_forward(chest_yaw)

    shoulder_left = chest - segments.shoulder_half_width * right_axis
    shoulder_right = chest + segments.shoulder_half_width * right_axis

    pole = forward_axis  # natural elbow bend forward
    half = segments.arm_length * 0.5
    arm_a = half  # upper arm
    arm_b = half  # forearm
    sol_left = solve_two_bone(shoulder_left, controller_left, arm_a, arm_b, pole)
    sol_right = solve_two_bone(shoulder_right, controller_right, arm_a, arm_b, pole)

    return UpperBodyJoints(
        chest=chest,
        neck=neck,
        head=head,
        shoulder_left=shoulder_left,
        elbow_left=sol_left.intermediate,
        wrist_left=sol_left.end,
        shoulder_right=shoulder_right,
        elbow_right=sol_right.intermediate,
        wrist_right=sol_right.end,
    )
