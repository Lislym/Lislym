"""Multi-view triangulation with weighted DLT and RANSAC.

The weighted DLT follows Hartley & Zisserman §12.2: each 2D observation
``(x_i, y_i)`` from camera ``i`` with projection matrix ``P_i = [p1; p2; p3]``
contributes two rows to a homogeneous system

::

    x_i * p3 - p1
    y_i * p3 - p2

scaled by an observation weight ``w_i``. The solution is the right-singular
vector of the smallest singular value, normalised so the homogeneous
coordinate is one.

RANSAC iterates over 2-camera samples (the minimum that yields a 3D point),
counts how many remaining cameras have a re-projection error below a
threshold, and refines on the largest consensus set with the weighted DLT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lislym.calibration.camera_model import CameraModel, undistort_points


@dataclass(frozen=True)
class TriangulationResult:
    """Triangulation outcome for one marker at one timestamp.

    ``mode`` is one of:

    * ``"multi"``  — at least two inlier cameras, weighted DLT applied.
    * ``"single"`` — exactly one camera; caller should fall back to the K=1
      kinematic-constrained solver.
    * ``"none"``   — no usable observations; caller should use prediction.
    """

    point: np.ndarray | None
    inliers: tuple[int, ...]
    reprojection_rms: float
    mode: str


def _project(camera: CameraModel, point_world: np.ndarray) -> np.ndarray:
    p = camera.projection_matrix @ np.append(point_world, 1.0)
    return p[:2] / p[2]


def _weighted_dlt(
    cameras: list[CameraModel],
    obs: list[tuple[float, float]],
    weights: list[float],
    indices: list[int],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for k, idx in enumerate(indices):
        cam = cameras[idx]
        x, y = obs[idx]
        w = float(weights[idx])
        P = cam.projection_matrix
        rows.append(w * (x * P[2] - P[0]))
        rows.append(w * (y * P[2] - P[1]))
        del k
    A = np.stack(rows, axis=0)
    _, _, vt = np.linalg.svd(A)
    Xh = vt[-1]
    if abs(Xh[3]) < 1e-12:
        raise np.linalg.LinAlgError("DLT solution is at infinity")
    return Xh[:3] / Xh[3]


def _reproj_error(camera: CameraModel, obs_xy: tuple[float, float], world_xyz: np.ndarray) -> float:
    proj = _project(camera, world_xyz)
    dx = proj[0] - obs_xy[0]
    dy = proj[1] - obs_xy[1]
    return float(np.hypot(dx, dy))


def triangulate(
    cameras: list[CameraModel],
    observations: list[tuple[float, float] | None],
    pixel_residuals: list[float] | None = None,
    *,
    ransac_threshold_px: float = 4.0,
    min_inliers: int = 2,
    max_iterations: int = 16,
    rng: np.random.Generator | None = None,
) -> TriangulationResult:
    """Triangulate one 3D point from multi-view 2D observations.

    Parameters
    ----------
    cameras:
        Calibrated cameras in a fixed order.
    observations:
        Same length as ``cameras``. ``None`` for cameras that did not see
        the marker this frame; otherwise a ``(px, py)`` pixel observation in
        the *original distorted* image.
    pixel_residuals:
        Optional 1-sigma localisation error in pixels per camera. Inverse
        variance is used as the DLT weight; defaults to 1 px for everyone.
    ransac_threshold_px:
        Re-projection error (pixels) below which a camera is an inlier.
    min_inliers:
        Minimum cameras required to claim a multi-view solution.
    max_iterations:
        Cap on RANSAC samples; the routine also terminates as soon as it
        finds an all-cameras consensus.
    rng:
        Optional generator for reproducibility.
    """
    if len(cameras) != len(observations):
        raise ValueError("cameras and observations must have the same length")
    n = len(cameras)
    if pixel_residuals is None:
        pixel_residuals = [1.0] * n
    if len(pixel_residuals) != n:
        raise ValueError("pixel_residuals must match cameras length")

    valid_indices = [i for i, obs in enumerate(observations) if obs is not None]
    if len(valid_indices) == 0:
        return TriangulationResult(point=None, inliers=(), reprojection_rms=float("inf"), mode="none")
    if len(valid_indices) == 1:
        return TriangulationResult(
            point=None,
            inliers=tuple(valid_indices),
            reprojection_rms=float("inf"),
            mode="single",
        )

    # Undistort observations once so re-projection error is computed on the
    # ideal pinhole that DLT actually solved.
    undistorted: list[tuple[float, float] | None] = [None] * n
    for i in valid_indices:
        obs = observations[i]
        assert obs is not None
        undistorted[i] = tuple(undistort_points(cameras[i], np.array([obs])).reshape(-1).tolist())

    weights = [1.0 / max(pr, 1e-3) ** 2 for pr in pixel_residuals]
    obs_for_solver: list[tuple[float, float]] = [
        (0.0, 0.0) if u is None else u for u in undistorted
    ]

    if rng is None:
        rng = np.random.default_rng(0xA1F5)

    best_inliers: list[int] = []
    best_point: np.ndarray | None = None
    best_rms: float = float("inf")

    if len(valid_indices) == 2:
        candidates = [tuple(valid_indices)]
    else:
        candidates = []
        seen: set[tuple[int, int]] = set()
        # Enumerate small problems exactly, otherwise sample.
        from itertools import combinations

        all_pairs = list(combinations(valid_indices, 2))
        if len(all_pairs) <= max_iterations:
            candidates = all_pairs
        else:
            while len(candidates) < max_iterations:
                pair = tuple(sorted(rng.choice(valid_indices, size=2, replace=False).tolist()))
                if pair in seen:
                    continue
                seen.add(pair)
                candidates.append(pair)

    for sample in candidates:
        try:
            X = _weighted_dlt(cameras, obs_for_solver, weights, list(sample))
        except np.linalg.LinAlgError:
            continue

        inliers: list[int] = []
        sq_errors: list[float] = []
        for i in valid_indices:
            err = _reproj_error(cameras[i], obs_for_solver[i], X)
            if err <= ransac_threshold_px:
                inliers.append(i)
                sq_errors.append(err * err)

        if len(inliers) > len(best_inliers) or (
            len(inliers) == len(best_inliers)
            and np.sqrt(np.mean(sq_errors) if sq_errors else np.inf) < best_rms
        ):
            best_inliers = inliers
            best_rms = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("inf")
            best_point = X

        if len(best_inliers) == len(valid_indices):
            break

    if best_point is None or len(best_inliers) < min_inliers:
        # Could not find a clean consensus. Fall back to weighted DLT on all
        # available cameras — better an estimate than nothing, and the caller
        # will see the high RMS.
        try:
            X = _weighted_dlt(cameras, obs_for_solver, weights, valid_indices)
            sq_errors = [
                _reproj_error(cameras[i], obs_for_solver[i], X) ** 2 for i in valid_indices
            ]
            return TriangulationResult(
                point=X,
                inliers=tuple(valid_indices),
                reprojection_rms=float(np.sqrt(np.mean(sq_errors))),
                mode="multi",
            )
        except np.linalg.LinAlgError:
            return TriangulationResult(
                point=None,
                inliers=(),
                reprojection_rms=float("inf"),
                mode="none",
            )

    # Refine on the inlier set with the weighted DLT.
    refined = _weighted_dlt(cameras, obs_for_solver, weights, best_inliers)
    sq_errors = [_reproj_error(cameras[i], obs_for_solver[i], refined) ** 2 for i in best_inliers]
    return TriangulationResult(
        point=refined,
        inliers=tuple(best_inliers),
        reprojection_rms=float(np.sqrt(np.mean(sq_errors))),
        mode="multi",
    )
