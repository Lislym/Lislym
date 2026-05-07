"""Tests for wand-based extrinsic calibration."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.extrinsic import (
    WandObservation,
    WandSpec,
    calibrate_extrinsic_wand,
)
from lislym.calibration.store import Intrinsics


def _intrinsics(name: str) -> Intrinsics:
    K = np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]], dtype=np.float64)
    return Intrinsics(
        camera_name=name,
        K=K,
        dist=np.zeros(5),
        image_size=(1280, 720),
        reprojection_rms_px=0.2,
        captured_frames=80,
    )


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> tuple[np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R = np.stack([right, down, forward], axis=0)
    t = -R @ eye
    return R, t


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = R @ X + t
    p = K @ Xc
    return p[:2] / p[2]


def _make_synthetic_observations(
    n_frames: int, rng: np.random.Generator, noise_sigma: float = 0.0
) -> tuple[list[Intrinsics], list[list[WandObservation]], list[tuple[np.ndarray, np.ndarray]], WandSpec]:
    wand = WandSpec(length_m=0.50, mid_offset_m=0.25)
    template = wand.reference_points()

    # Cameras placed around a 1.5 m volume centred at the world origin (which
    # ends up being camera 0's frame after calibration).
    eyes = [
        np.array([1.5, 0.0, 1.5]),
        np.array([-1.5, 0.2, 1.5]),
        np.array([0.0, 1.5, 1.5]),
    ]
    target = np.array([0.0, 0.0, 1.0])
    cams_extrinsic = [_look_at(e, target) for e in eyes]
    intrinsics = [_intrinsics(f"cam{i}") for i in range(len(eyes))]

    # In camera-0's frame the world origin sits at -R0 @ eye0.
    # We don't actually transform anything: the calibrator returns poses in
    # whichever frame camera 0 is at, and we compare against the *relative*
    # poses between cameras. So generate world points freely.

    observations: list[list[WandObservation]] = [[] for _ in cams_extrinsic]
    ground_truth_wand_poses: list[tuple[np.ndarray, np.ndarray]] = []

    for f in range(n_frames):
        # Random rigid pose for the wand each frame.
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        K_skew = np.array(
            [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
        )
        R_wand = np.eye(3) * c + s * K_skew + (1 - c) * np.outer(axis, axis)
        # Translate roughly within a 1.5 m × 1.5 m × 0.4 m volume.
        t_wand = np.array(
            [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.8, 1.2)]
        )
        ground_truth_wand_poses.append((R_wand, t_wand))

        world_pts = (R_wand @ template.T).T + t_wand

        for cam_idx, (R, t) in enumerate(cams_extrinsic):
            pix = np.empty((3, 2))
            for k in range(3):
                p = _project(intrinsics[cam_idx].K, R, t, world_pts[k])
                if noise_sigma > 0:
                    p = p + rng.normal(0, noise_sigma, size=2)
                pix[k] = p
            observations[cam_idx].append(
                WandObservation(camera_index=cam_idx, frame_index=f, points=pix)
            )

    return intrinsics, observations, cams_extrinsic, wand


def _relative_pose_distance(
    R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray
) -> tuple[float, float]:
    """Translational distance and rotational angle between two poses."""
    dR = R_gt @ R_est.T
    angle = float(np.arccos(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)))
    dt = float(np.linalg.norm(t_gt - t_est))
    return dt, angle


def test_wand_extrinsic_recovers_relative_pose_no_noise() -> None:
    """With perfect 2D observations and 30 frames, BA should converge cleanly."""
    rng = np.random.default_rng(42)
    intrinsics, obs, gt_extrinsic, wand = _make_synthetic_observations(
        n_frames=30, rng=rng, noise_sigma=0.0
    )
    session = calibrate_extrinsic_wand(intrinsics, obs, wand)

    # Camera 0 is the world frame in the recovered solution. Compute the
    # ground-truth pose of every other camera relative to camera 0, and
    # compare against the estimate.
    R0_gt, t0_gt = gt_extrinsic[0]
    for i in range(1, len(intrinsics)):
        Ri_gt, ti_gt = gt_extrinsic[i]
        # World→cam_i in camera-0 coordinates: cam_i pose w.r.t. cam_0.
        R_rel = Ri_gt @ R0_gt.T
        t_rel = ti_gt - R_rel @ t0_gt

        Ri_est = session.cameras[i].R
        ti_est = session.cameras[i].t
        dt, dang = _relative_pose_distance(R_rel, t_rel, Ri_est, ti_est)
        # Allow 5 cm / 2°. Real-world targets are tighter; this checks the
        # solver's correctness given clean data.
        assert dt < 0.05, f"camera {i} translation off by {dt*1000:.1f} mm"
        assert np.degrees(dang) < 2.0, f"camera {i} rotation off by {np.degrees(dang):.2f}°"

    assert session.bundle_adjustment_rms_px < 0.5
    assert session.locked is True


def test_wand_extrinsic_robust_to_subpixel_noise() -> None:
    """0.3 px noise should still give sub-cm extrinsics with 60+ frames."""
    rng = np.random.default_rng(7)
    intrinsics, obs, gt_extrinsic, wand = _make_synthetic_observations(
        n_frames=60, rng=rng, noise_sigma=0.3
    )
    session = calibrate_extrinsic_wand(intrinsics, obs, wand)

    R0_gt, t0_gt = gt_extrinsic[0]
    for i in range(1, len(intrinsics)):
        Ri_gt, ti_gt = gt_extrinsic[i]
        R_rel = Ri_gt @ R0_gt.T
        t_rel = ti_gt - R_rel @ t0_gt
        Ri_est = session.cameras[i].R
        ti_est = session.cameras[i].t
        dt, dang = _relative_pose_distance(R_rel, t_rel, Ri_est, ti_est)
        assert dt < 0.03, f"camera {i} translation off by {dt*1000:.1f} mm"
        assert np.degrees(dang) < 1.0, f"camera {i} rotation off by {np.degrees(dang):.2f}°"


def test_wand_extrinsic_rejects_too_few_seed_frames() -> None:
    rng = np.random.default_rng(0)
    intrinsics, obs, _, wand = _make_synthetic_observations(n_frames=3, rng=rng)
    with pytest.raises(RuntimeError, match="≥6 frames"):
        calibrate_extrinsic_wand(intrinsics, obs, wand)


def test_wand_session_marked_locked_with_default_notes() -> None:
    rng = np.random.default_rng(11)
    intrinsics, obs, _, wand = _make_synthetic_observations(n_frames=30, rng=rng)
    session = calibrate_extrinsic_wand(intrinsics, obs, wand)
    assert session.locked is True
    assert "wand" in session.notes.lower()
    assert len(session.cameras) == 3
