"""HMD-to-camera world frame alignment via Procrustes.

Plan §D / "HMD ↔ カメラ系アライメント": the cameras define one world
frame (calibrated by the wand, with camera 0 at the origin); SteamVR
defines another (with the play-area origin somewhere on the floor and
+Y pointing up in OpenVR convention). To send VRChat OSC trackers in
the same frame the HMD operates in, we need a rigid transform
``T_steamvr_from_camera`` that maps a marker position from the camera
world frame to the SteamVR frame.

Procedure:

1. The user stands T-pose for ~3 s.
2. We sample the HMD position and the hip-marker position once per frame.
3. We assume the world up axis is unchanged (gravity), so the rotation
   reduces to a yaw + a 2D translation in the floor plane plus a height
   offset.

The yaw aligns the hip-to-HMD vector (projected onto the floor) with
SteamVR's +Z axis (or whichever axis the user is reported facing in
SteamVR's frame). For the alignment-only purpose of this module we
take a vector pair ``(camera_world_vec, steamvr_vec)`` and solve for
the in-plane rotation and the translation offset that maps one to the
other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameAlignment:
    """Rigid transform: ``q_steamvr = R @ q_camera + t``.

    Restricted to a yaw + translation (no pitch/roll), which is the
    correct DOF for room-scale VR where gravity is shared across both
    frames.
    """

    yaw_radians: float
    translation: np.ndarray  # 3-vector

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Map ``(N, 3)`` or ``(3,)`` points from camera frame to SteamVR frame."""
        c = np.cos(self.yaw_radians)
        s = np.sin(self.yaw_radians)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        single = points.ndim == 1
        pts = points.reshape(-1, 3)
        out = pts @ R.T + self.translation
        return out.reshape(-1) if single else out


def align_yaw_and_translation(
    camera_points: np.ndarray,
    steamvr_points: np.ndarray,
) -> FrameAlignment:
    """Procrustes alignment restricted to yaw + translation.

    The two arrays are corresponding samples (``(N, 3)`` each) of the
    same physical positions observed in both frames. Returns the rigid
    transform that maps the camera frame to the SteamVR frame with the
    smallest total squared error in the floor plane plus a constant
    height offset.
    """
    if camera_points.shape != steamvr_points.shape:
        raise ValueError("camera and steamvr point sets must have equal shape")
    if camera_points.shape[0] < 1:
        raise ValueError("need at least one correspondence")

    # Centroids in both frames.
    c_cam = camera_points.mean(axis=0)
    c_vr = steamvr_points.mean(axis=0)

    cam_xy = (camera_points - c_cam)[:, :2]
    vr_xy = (steamvr_points - c_vr)[:, :2]

    # Solve for yaw via the 2D Procrustes formula.
    H = vr_xy.T @ cam_xy
    a = H[0, 0] + H[1, 1]
    b = H[1, 0] - H[0, 1]
    yaw = float(np.arctan2(b, a))

    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = c_vr - R @ c_cam

    return FrameAlignment(yaw_radians=yaw, translation=t)
