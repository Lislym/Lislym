"""Pinhole camera model used by every downstream component.

A camera is represented by:

* ``K`` — 3x3 intrinsics (focal length and principal point in pixels).
* ``dist`` — distortion coefficients in OpenCV's order ``(k1, k2, p1, p2, k3)``.
* ``R`` — 3x3 world-to-camera rotation.
* ``t`` — 3-vector world-to-camera translation; the camera centre is
  ``-R^T @ t``.
* ``image_size`` — ``(width, height)`` in pixels.

We deliberately store the world-to-camera form because that is what
``cv2.projectPoints`` expects, and because the projection matrix
``P = K [R | t]`` falls out trivially.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Calibrated pinhole camera with optional radial/tangential distortion."""

    name: str
    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int] = field(default=(1920, 1080))

    def __post_init__(self) -> None:
        if self.K.shape != (3, 3):
            raise ValueError("K must be 3x3")
        if self.R.shape != (3, 3):
            raise ValueError("R must be 3x3")
        if self.t.shape != (3,):
            raise ValueError("t must be a 3-vector")
        if self.dist.ndim != 1:
            raise ValueError("dist must be 1-D")

    @property
    def projection_matrix(self) -> np.ndarray:
        """Return ``P = K [R | t]`` as a 3x4 matrix."""
        Rt = np.empty((3, 4), dtype=np.float64)
        Rt[:, :3] = self.R
        Rt[:, 3] = self.t
        return self.K @ Rt

    @property
    def center(self) -> np.ndarray:
        """World-space camera centre ``C = -R^T t``."""
        return -self.R.T @ self.t

    def ray_through_pixel(self, px: float, py: float) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(origin, direction)`` of the world-space ray for a pixel.

        Direction is unit length. Uses the inverse intrinsics; distortion is
        not undone here — call :func:`undistort_points` first if needed.
        """
        Kinv = np.linalg.inv(self.K)
        ray_cam = Kinv @ np.array([px, py, 1.0])
        ray_world = self.R.T @ ray_cam
        ray_world /= np.linalg.norm(ray_world)
        return self.center, ray_world


def undistort_points(camera: CameraModel, pts: np.ndarray) -> np.ndarray:
    """Undistort and normalise pixel points to ideal pinhole coordinates.

    Returns an ``(N, 2)`` array of pixel coordinates as if the camera had
    zero distortion. Works for an empty distortion vector.
    """
    if pts.size == 0:
        return pts.copy()
    if not np.any(camera.dist):
        return pts.astype(np.float64)
    import cv2  # local import keeps the data class import-light

    pts_cv = pts.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts_cv, camera.K, camera.dist, P=camera.K)
    return undist.reshape(-1, 2)
