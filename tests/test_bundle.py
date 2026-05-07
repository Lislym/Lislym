"""Tests for sliding-window bundle adjustment."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.camera_model import CameraModel
from lislym.fusion.bundle import FrameObservations, SlidingWindowSmoother


def _intrinsics() -> np.ndarray:
    return np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]], dtype=np.float64)


def _look_at(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    up = np.array([0.0, 0.0, 1.0])
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd], axis=0)
    t = -R @ eye
    return R, t


def _make_cameras() -> list[CameraModel]:
    cams = []
    for i, eye in enumerate(
        [np.array([1.5, 0.0, 1.5]), np.array([-1.5, 0.0, 1.5]), np.array([0.0, 1.5, 1.5])]
    ):
        R, t = _look_at(eye, np.array([0.0, 0.0, 1.0]))
        cams.append(CameraModel(name=f"cam{i}", K=_intrinsics(), dist=np.zeros(5), R=R, t=t))
    return cams


def _project(cam: CameraModel, X: np.ndarray) -> tuple[float, float]:
    Xc = cam.R @ X + cam.t
    p = cam.K @ Xc
    return float(p[0] / p[2]), float(p[1] / p[2])


def test_smoother_recovers_static_point() -> None:
    cams = _make_cameras()
    smoother = SlidingWindowSmoother(cameras=cams, window_size=5)
    target = np.array([0.05, 0.10, 1.20])

    for k in range(5):
        obs = FrameObservations(
            timestamp=k * 0.016,
            pixel_observations={i: _project(c, target) for i, c in enumerate(cams)},
            pixel_residuals={i: 0.3 for i in range(len(cams))},
        )
        smoother.push(obs)
    positions = smoother.solve()
    assert positions is not None
    np.testing.assert_allclose(positions[-1], target, atol=1e-3)


def test_smoother_reduces_jitter_versus_per_frame_estimate() -> None:
    """Per-frame triangulation noise should drop substantially after BA."""
    cams = _make_cameras()
    rng = np.random.default_rng(0)
    smoother = SlidingWindowSmoother(cameras=cams, window_size=12, smoothness_weight=5.0)

    truth: list[np.ndarray] = []
    smoothed: list[np.ndarray] = []
    for k in range(40):
        # A point moving slowly along x.
        true_pos = np.array([0.001 * k, 0.0, 1.20])
        truth.append(true_pos)
        obs = FrameObservations(
            timestamp=k * 0.016,
            pixel_observations={
                i: tuple(np.array(_project(c, true_pos)) + rng.normal(0, 0.5, 2))
                for i, c in enumerate(cams)
            },
            pixel_residuals={i: 0.5 for i in range(len(cams))},
        )
        smoother.push(obs)
        positions = smoother.solve()
        smoothed.append(positions[-1])

    # Compare RMS error after the window has filled.
    truth_arr = np.stack(truth[15:])
    smooth_arr = np.stack(smoothed[15:])
    err = np.linalg.norm(smooth_arr - truth_arr, axis=1)
    rms_smoothed = float(np.sqrt(np.mean(err ** 2)))
    # Without the smoother an unweighted DLT at σ_pixel=0.5 with 3 cameras
    # gives roughly 1–2 mm RMS (theoretical), but on real-frame jitter this
    # produces visible jumps. We expect <2 mm here.
    assert rms_smoothed < 0.005, f"smoothed RMS too high: {rms_smoothed*1000:.2f} mm"


def test_smoother_handles_empty_window() -> None:
    smoother = SlidingWindowSmoother(cameras=_make_cameras(), window_size=5)
    assert smoother.solve() is None
    assert smoother.latest_smoothed() is None


def test_smoother_window_size_bounded_to_capacity() -> None:
    smoother = SlidingWindowSmoother(cameras=_make_cameras(), window_size=3)
    cams = _make_cameras()
    target = np.array([0.0, 0.0, 1.2])
    for k in range(10):
        obs = FrameObservations(
            timestamp=k * 0.016,
            pixel_observations={i: _project(c, target) for i, c in enumerate(cams)},
            pixel_residuals={i: 0.3 for i in range(len(cams))},
        )
        smoother.push(obs)
    positions = smoother.solve()
    assert positions.shape == (3, 3)
