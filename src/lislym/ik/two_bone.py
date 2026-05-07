"""Analytical two-bone IK.

Solves for the position of an intermediate joint (knee or elbow) given
the chain root, end effector, and the two bone lengths. The classic
problem: a triangle with known side lengths ``a`` (root→intermediate)
and ``b`` (intermediate→end) and known endpoint separation ``c`` (root→
end). The intermediate joint sits at one of the two intersection points
of two spheres of radius ``a`` and ``b``; the ambiguity is resolved by a
*pole* vector that selects which side of the root–end axis the bend
should fall on.

This is the same formula every game engine ships under the name
"TwoBoneIK". We implement it explicitly so the rest of the code stays
NumPy-only with no engine dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TwoBoneSolution:
    """Result of a two-bone IK solve."""

    intermediate: np.ndarray  # knee or elbow
    end: np.ndarray  # may differ from the requested ``end`` if it was out of reach
    reach_clipped: bool


def solve_two_bone(
    root: np.ndarray,
    end_target: np.ndarray,
    a: float,
    b: float,
    pole: np.ndarray,
    *,
    minimum_separation_m: float = 1e-4,
) -> TwoBoneSolution:
    """Solve a two-bone IK chain.

    Parameters
    ----------
    root:
        Position of the chain's first joint (hip / shoulder).
    end_target:
        Desired position of the end effector (ankle / wrist). If beyond
        ``a + b`` the chain is fully extended along ``end - root`` and the
        actual end position is clipped to that reach (``reach_clipped``
        is set in the result).
    a, b:
        Bone lengths in metres.
    pole:
        Direction in which the bend should fall. Project this vector onto
        the plane perpendicular to the root–end axis to choose the side.
        For a leg with knee bending forward, pass ``hip_forward`` (the
        body's forward axis). For an arm bending naturally, ``hip_down``
        works well.
    minimum_separation_m:
        If the root and end are essentially coincident the bend direction
        becomes ill-defined; we then place the intermediate joint
        ``a`` metres along the pole from the root.
    """
    diff = end_target - root
    c = float(np.linalg.norm(diff))

    reach_clipped = False
    if c > a + b:
        # Out of reach — fully extend along the requested direction.
        if c < 1e-9:
            return TwoBoneSolution(
                intermediate=root + a * _safe_normalise(pole),
                end=root + (a + b) * _safe_normalise(pole),
                reach_clipped=True,
            )
        end_dir = diff / c
        end = root + (a + b) * end_dir
        intermediate = root + a * end_dir
        return TwoBoneSolution(intermediate=intermediate, end=end, reach_clipped=True)

    if c < minimum_separation_m:
        intermediate = root + a * _safe_normalise(pole)
        return TwoBoneSolution(intermediate=intermediate, end=end_target.copy(), reach_clipped=False)

    # Distance along root→end at which the intermediate joint projects.
    # cos law for triangle (a, b, c): cos(α) = (a² + c² − b²)/(2ac)
    cos_alpha = float(np.clip((a * a + c * c - b * b) / (2.0 * a * c), -1.0, 1.0))
    along = a * cos_alpha
    # Perpendicular distance from the root–end axis to the intermediate joint.
    height = a * float(np.sqrt(max(0.0, 1.0 - cos_alpha * cos_alpha)))

    axis = diff / c
    # Component of pole orthogonal to ``axis`` is the bend direction.
    pole_perp = pole - np.dot(pole, axis) * axis
    norm = float(np.linalg.norm(pole_perp))
    if norm < 1e-9:
        # The pole is parallel to the chain direction; pick any orthogonal
        # vector so the solver still returns something sensible. The
        # downstream IK can refine the bend direction once geometry is
        # less degenerate.
        helper = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        pole_perp = helper - np.dot(helper, axis) * axis
        norm = float(np.linalg.norm(pole_perp))
    pole_perp /= norm

    intermediate = root + along * axis + height * pole_perp
    return TwoBoneSolution(intermediate=intermediate, end=end_target.copy(), reach_clipped=False)


def _safe_normalise(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0])
    return v / n
