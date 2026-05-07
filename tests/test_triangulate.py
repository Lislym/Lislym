"""Tests for multi-view triangulation."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.camera_model import CameraModel
from lislym.fusion.triangulate import triangulate


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0, 0, 1])) -> tuple[np.ndarray, np.ndarray]:
    """Build world-to-camera (R, t) for a camera at ``eye`` looking at ``target``.

    Returns OpenCV convention: camera +Z forward, +Y down, +X right.
    """
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R_wc = np.stack([right, down, forward], axis=0)  # rows are camera axes
    t_wc = -R_wc @ eye
    return R_wc, t_wc


def _intrinsics(fx: float = 900.0, fy: float = 900.0, cx: float = 640.0, cy: float = 360.0) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def _make_cameras(positions: list[np.ndarray], target: np.ndarray) -> list[CameraModel]:
    cams = []
    for i, eye in enumerate(positions):
        R, t = _look_at(eye, target)
        cams.append(
            CameraModel(
                name=f"cam{i}",
                K=_intrinsics(),
                dist=np.zeros(5),
                R=R,
                t=t,
                image_size=(1280, 720),
            )
        )
    return cams


def _project(camera: CameraModel, X: np.ndarray) -> tuple[float, float]:
    p = camera.projection_matrix @ np.append(X, 1.0)
    return float(p[0] / p[2]), float(p[1] / p[2])


def test_triangulate_two_cameras_perfect_observations() -> None:
    cams = _make_cameras(
        positions=[np.array([1.5, 0.0, 1.5]), np.array([-1.5, 0.0, 1.5])],
        target=np.array([0.0, 0.0, 1.5]),
    )
    X = np.array([0.0, 0.1, 1.5])
    obs = [_project(c, X) for c in cams]

    result = triangulate(cams, obs)
    assert result.mode == "multi"
    assert result.point is not None
    assert np.linalg.norm(result.point - X) < 1e-6


def test_triangulate_four_cameras_with_one_outlier() -> None:
    cams = _make_cameras(
        positions=[
            np.array([1.5, 0.0, 1.5]),
            np.array([-1.5, 0.0, 1.5]),
            np.array([0.0, 1.5, 1.5]),
            np.array([0.0, -1.5, 1.5]),
        ],
        target=np.array([0.0, 0.0, 1.5]),
    )
    X = np.array([0.05, -0.1, 1.4])
    obs: list[tuple[float, float] | None] = [_project(c, X) for c in cams]
    # Inject a 50 px outlier into camera 2.
    bad = obs[2]
    assert bad is not None
    obs[2] = (bad[0] + 50.0, bad[1] - 50.0)

    result = triangulate(cams, obs, ransac_threshold_px=2.0)
    assert result.mode == "multi"
    assert 2 not in result.inliers
    assert {0, 1, 3}.issubset(set(result.inliers))
    assert result.point is not None
    assert np.linalg.norm(result.point - X) < 5e-3


def test_triangulate_single_camera_signals_fallback() -> None:
    cams = _make_cameras(
        positions=[np.array([1.5, 0.0, 1.5]), np.array([-1.5, 0.0, 1.5])],
        target=np.array([0.0, 0.0, 1.5]),
    )
    X = np.array([0.0, 0.0, 1.5])
    obs: list[tuple[float, float] | None] = [_project(cams[0], X), None]
    result = triangulate(cams, obs)
    assert result.mode == "single"
    assert result.point is None
    assert result.inliers == (0,)


def test_triangulate_zero_cameras_returns_none() -> None:
    cams = _make_cameras(
        positions=[np.array([1.5, 0.0, 1.5]), np.array([-1.5, 0.0, 1.5])],
        target=np.array([0.0, 0.0, 1.5]),
    )
    result = triangulate(cams, [None, None])
    assert result.mode == "none"
    assert result.point is None


def test_triangulate_weight_favours_lower_residual_camera() -> None:
    """A camera with a 0.1 px residual should dominate one with 5 px."""
    cams = _make_cameras(
        positions=[np.array([1.5, 0.0, 1.5]), np.array([-1.5, 0.0, 1.5])],
        target=np.array([0.0, 0.0, 1.5]),
    )
    X = np.array([0.05, 0.05, 1.4])
    obs0 = _project(cams[0], X)
    obs1 = _project(cams[1], X)
    # Add a small noise to cam 1 only; if the weights were equal the result
    # would split the noise, but with high weight on cam 0 it should stay close.
    obs1_noisy = (obs1[0] + 1.0, obs1[1] - 0.5)

    result_weighted = triangulate(
        cams,
        [obs0, obs1_noisy],
        pixel_residuals=[0.1, 5.0],
        ransac_threshold_px=10.0,
    )
    result_equal = triangulate(
        cams,
        [obs0, obs1_noisy],
        pixel_residuals=[1.0, 1.0],
        ransac_threshold_px=10.0,
    )
    err_weighted = np.linalg.norm(result_weighted.point - X)
    err_equal = np.linalg.norm(result_equal.point - X)
    assert err_weighted < err_equal


@pytest.mark.parametrize("noise_sigma", [0.0, 0.3])
def test_triangulate_three_cameras_noise_budget(noise_sigma: float) -> None:
    cams = _make_cameras(
        positions=[
            np.array([1.5, 0.0, 1.5]),
            np.array([-1.5, 0.0, 1.5]),
            np.array([0.0, 1.5, 1.5]),
        ],
        target=np.array([0.0, 0.0, 1.5]),
    )
    rng = np.random.default_rng(7)
    errors: list[float] = []
    for _ in range(50):
        X = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(1.2, 1.6)])
        obs = []
        for c in cams:
            px, py = _project(c, X)
            obs.append((px + rng.normal(0, noise_sigma), py + rng.normal(0, noise_sigma)))
        result = triangulate(cams, obs, ransac_threshold_px=2.0)
        assert result.mode == "multi"
        errors.append(float(np.linalg.norm(result.point - X)))
    rms = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    if noise_sigma == 0.0:
        assert rms < 1e-6
    else:
        # σ_pixel = 0.3, baseline ~3m, depth ~1.5m → ~0.5–2 mm.
        assert rms < 0.005, f"3D RMS too high: {rms*1000:.3f} mm"
