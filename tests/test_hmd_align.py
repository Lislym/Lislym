"""Tests for HMD-to-camera frame alignment."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.hmd_align import (
    FrameAlignment,
    align_yaw_and_translation,
)


def test_alignment_recovers_rigid_transform_no_noise() -> None:
    rng = np.random.default_rng(0)
    cam_pts = rng.normal(size=(20, 3))
    yaw_truth = np.deg2rad(40.0)
    t_truth = np.array([1.0, -0.5, 0.3])
    c, s = np.cos(yaw_truth), np.sin(yaw_truth)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    vr_pts = cam_pts @ R.T + t_truth

    alignment = align_yaw_and_translation(cam_pts, vr_pts)
    assert alignment.yaw_radians == pytest.approx(yaw_truth, abs=1e-6)
    np.testing.assert_allclose(alignment.translation, t_truth, atol=1e-6)


def test_alignment_apply_round_trip() -> None:
    rng = np.random.default_rng(1)
    cam_pts = rng.normal(size=(10, 3))
    yaw = np.deg2rad(15.0)
    t = np.array([0.2, -0.4, 0.05])

    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    vr_pts = cam_pts @ R.T + t

    alignment = align_yaw_and_translation(cam_pts, vr_pts)
    transformed = alignment.apply(cam_pts)
    np.testing.assert_allclose(transformed, vr_pts, atol=1e-6)


def test_alignment_apply_handles_single_point() -> None:
    alignment = FrameAlignment(yaw_radians=np.deg2rad(90.0), translation=np.array([1.0, 0.0, 0.0]))
    out = alignment.apply(np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(out, [1.0, 1.0, 0.0], atol=1e-6)


def test_alignment_robust_to_small_noise() -> None:
    rng = np.random.default_rng(2)
    cam_pts = rng.normal(size=(50, 3))
    yaw = np.deg2rad(30.0)
    t = np.array([0.5, 0.5, 0.0])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    vr_pts = cam_pts @ R.T + t + rng.normal(0, 0.005, cam_pts.shape)

    alignment = align_yaw_and_translation(cam_pts, vr_pts)
    assert abs(alignment.yaw_radians - yaw) < np.deg2rad(0.5)
