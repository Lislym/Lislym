"""Floor plane calibration.

Two equally valid paths:

* **AprilTag**: a single 10 cm AprilTag is taped flat on the floor;
  ``cv2.aruco`` recovers the tag's 6 DoF pose per camera, the tag's local
  +Z axis is the floor normal, and the tag origin pins a point on the
  plane. Multiple cameras' estimates are averaged for robustness.
* **Point cloud fit**: any 3D points known to lie on the floor (e.g.,
  the lowest sphere positions of the ankle markers during a quiet
  standing pose) are fit with a least-squares plane.

The calibration result is a :class:`FloorPlane` carrying a unit normal
``n`` and a signed offset ``d`` such that floor points satisfy
``n · x + d = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from lislym.calibration.camera_model import CameraModel


@dataclass(frozen=True)
class FloorPlane:
    """Plane in world frame: ``n · x + d = 0`` with ``||n|| = 1``."""

    normal: np.ndarray
    offset: float

    def signed_distance(self, point: np.ndarray) -> float:
        return float(np.dot(self.normal, point) + self.offset)

    def project(self, point: np.ndarray) -> np.ndarray:
        return point - self.signed_distance(point) * self.normal

    def height_above(self, point: np.ndarray) -> float:
        """Positive = above floor, negative = below floor (penetrating)."""
        return self.signed_distance(point)


def fit_plane_from_points(points: np.ndarray, *, prefer_up: np.ndarray | None = None) -> FloorPlane:
    """Least-squares plane fit through ``(N, 3)`` points.

    The orientation of the normal is ambiguous; ``prefer_up`` picks a sign
    that aligns the normal with the supplied direction (defaults to world
    +Z which is the convention for "up").
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if points.shape[0] < 3:
        raise ValueError("need at least 3 points to fit a plane")

    centroid = points.mean(axis=0)
    centred = points - centroid
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)

    if prefer_up is None:
        prefer_up = np.array([0.0, 0.0, 1.0])
    if np.dot(normal, prefer_up) < 0:
        normal = -normal
    offset = -float(np.dot(normal, centroid))
    return FloorPlane(normal=normal, offset=offset)


@dataclass(frozen=True)
class AprilTagFloorObservation:
    """AprilTag detection from one camera."""

    camera_index: int
    corners_px: np.ndarray  # (4, 2)
    tag_id: int


def _tag_object_points(tag_size_m: float) -> np.ndarray:
    """OpenCV's expected ordering of ArUco/AprilTag corners (CCW from top-left)."""
    h = tag_size_m / 2.0
    return np.array(
        [
            [-h,  h, 0.0],
            [ h,  h, 0.0],
            [ h, -h, 0.0],
            [-h, -h, 0.0],
        ],
        dtype=np.float64,
    )


def floor_plane_from_apriltag(
    cameras: list[CameraModel],
    observations: list[AprilTagFloorObservation],
    tag_size_m: float = 0.10,
) -> FloorPlane:
    """Estimate the floor plane from AprilTag detections in 1+ cameras.

    Each camera that saw the tag contributes a 6 DoF estimate via
    ``solvePnP``; we average the resulting plane normals (after sign
    alignment) and the world-space tag origins. With more cameras the
    estimate is more robust to per-camera tag-corner localisation noise.
    """
    if not observations:
        raise ValueError("need at least one AprilTag observation")

    obj = _tag_object_points(tag_size_m)
    normals: list[np.ndarray] = []
    origins: list[np.ndarray] = []

    for obs in observations:
        cam = cameras[obs.camera_index]
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            obs.corners_px.astype(np.float64),
            cam.K,
            cam.dist if cam.dist.size else None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
        if not ok:
            continue
        R_tag, _ = cv2.Rodrigues(rvec)
        # Tag local +Z points up out of the tag face. In world coordinates:
        normal_cam = R_tag[:, 2]
        normal_world = cam.R.T @ normal_cam
        normal_world /= np.linalg.norm(normal_world)
        # Tag origin in camera frame is tvec; transform to world.
        origin_world = cam.R.T @ (tvec.reshape(3) - cam.t)
        normals.append(normal_world)
        origins.append(origin_world)

    if not normals:
        raise RuntimeError("no usable AprilTag PnP solutions")

    # Sign-align normals so they all point up before averaging.
    ref = normals[0]
    aligned = [n if np.dot(n, ref) >= 0 else -n for n in normals]
    n_avg = np.mean(aligned, axis=0)
    n_avg /= np.linalg.norm(n_avg)
    if n_avg[2] < 0:
        n_avg = -n_avg
    origin_avg = np.mean(origins, axis=0)
    offset = -float(np.dot(n_avg, origin_avg))
    return FloorPlane(normal=n_avg, offset=offset)
