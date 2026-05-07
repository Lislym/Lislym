"""Tests for floor plane calibration."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.floor import (
    FloorPlane,
    fit_plane_from_points,
)


def test_fit_plane_horizontal_z_zero() -> None:
    pts = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    )
    plane = fit_plane_from_points(pts)
    np.testing.assert_allclose(plane.normal, [0.0, 0.0, 1.0], atol=1e-9)
    assert plane.offset == pytest.approx(0.0)


def test_fit_plane_signed_distance() -> None:
    plane = FloorPlane(normal=np.array([0.0, 0.0, 1.0]), offset=0.0)
    assert plane.signed_distance(np.array([0.5, 0.5, 0.05])) == pytest.approx(0.05)
    assert plane.signed_distance(np.array([0.0, 0.0, -0.02])) == pytest.approx(-0.02)


def test_fit_plane_projects_onto_surface() -> None:
    plane = FloorPlane(normal=np.array([0.0, 0.0, 1.0]), offset=0.0)
    p = np.array([0.3, -0.7, 0.42])
    proj = plane.project(p)
    np.testing.assert_allclose(proj[:2], [0.3, -0.7])
    assert proj[2] == pytest.approx(0.0)


def test_fit_plane_orientation_prefers_up() -> None:
    # Same plane sampled in either CCW or CW order should still yield +Z.
    pts = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    plane = fit_plane_from_points(pts, prefer_up=np.array([0.0, 0.0, 1.0]))
    assert plane.normal[2] > 0


def test_fit_plane_handles_noisy_points() -> None:
    rng = np.random.default_rng(1)
    pts = []
    for _ in range(40):
        x = rng.uniform(-1, 1)
        y = rng.uniform(-1, 1)
        z = 0.05 + rng.normal(0, 0.001)  # 1 mm noise around z=0.05
        pts.append([x, y, z])
    plane = fit_plane_from_points(np.array(pts))
    assert abs(plane.normal[2] - 1.0) < 1e-3
    assert plane.offset == pytest.approx(-0.05, abs=1e-3)


def test_fit_plane_rejects_too_few_points() -> None:
    with pytest.raises(ValueError):
        fit_plane_from_points(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


def test_fit_plane_height_above_sign() -> None:
    plane = FloorPlane(normal=np.array([0.0, 0.0, 1.0]), offset=0.0)
    assert plane.height_above(np.array([0.0, 0.0, 0.1])) > 0
    assert plane.height_above(np.array([0.0, 0.0, -0.1])) < 0
