"""Wand-based extrinsic calibration (plan §C-6 method B).

A 50 cm rigid wand carries three colored spheres at known distances
(red — green — blue, end-to-end). When the user waves the wand through
the capture volume:

1. Each camera detects the three colored balls per frame and sub-pixel
   localises them.
2. Across all observed frames we have many 3-point rigid configurations,
   one per frame, where the inter-ball distances are known and constant.
3. Using one camera as the world origin (R = I, t = 0) and a *seed* pair
   of cameras, we recover an initial relative pose by triangulating the
   wand points up to scale and rescaling so the recovered red↔blue
   distance equals the wand length.
4. The remaining cameras are localised by ``cv2.solvePnP`` against the
   seed-recovered 3D points.
5. A bundle adjustment (``scipy.optimize.least_squares``) refines all
   camera poses *and* per-frame wand positions jointly to minimise
   re-projection error, with a soft constraint that the wand's three
   inter-ball distances remain constant.

The end product is a session-level :class:`ExtrinsicSession` that the
calibration store writes once, locks, and never updates at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares

from lislym.calibration.camera_model import CameraModel
from lislym.calibration.store import (
    ExtrinsicEntry,
    ExtrinsicSession,
    Intrinsics,
    utc_now_isoformat,
)


@dataclass(frozen=True)
class WandSpec:
    """Geometry of the calibration wand.

    The three balls are co-linear; ``mid_offset_m`` is the distance from
    the red ball to the green (middle) ball, and ``length_m`` is the
    full red-to-blue distance.
    """

    length_m: float = 0.50
    mid_offset_m: float = 0.25

    def reference_points(self) -> np.ndarray:
        """3-point template in the wand-local frame (red at origin, +x to blue)."""
        return np.array(
            [
                [0.0, 0.0, 0.0],
                [self.mid_offset_m, 0.0, 0.0],
                [self.length_m, 0.0, 0.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class WandObservation:
    """One frame's worth of wand detections from one camera.

    ``points`` is shape ``(3, 2)`` — pixel coordinates of (red, green, blue)
    in that order. Use NaNs for balls the camera didn't see; the bundle
    adjuster will skip them.
    """

    camera_index: int
    frame_index: int
    points: np.ndarray  # (3, 2) float64


def _triangulate_pair(
    cam_a: CameraModel,
    cam_b: CameraModel,
    pts_a: np.ndarray,
    pts_b: np.ndarray,
) -> np.ndarray:
    """Linear triangulation of N corresponding pixel points between two cameras.

    Returns ``(N, 3)`` world-space points. Both observations must already
    be undistorted into ideal pinhole pixel coordinates.
    """
    Pa = cam_a.projection_matrix
    Pb = cam_b.projection_matrix
    n = pts_a.shape[0]
    out = np.empty((n, 3))
    for i in range(n):
        A = np.stack(
            [
                pts_a[i, 0] * Pa[2] - Pa[0],
                pts_a[i, 1] * Pa[2] - Pa[1],
                pts_b[i, 0] * Pb[2] - Pb[0],
                pts_b[i, 1] * Pb[2] - Pb[1],
            ],
            axis=0,
        )
        _, _, vt = np.linalg.svd(A)
        Xh = vt[-1]
        out[i] = Xh[:3] / Xh[3]
    return out


def _rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    return R


def _R_to_rodrigues(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def _project(K: np.ndarray, R: np.ndarray, t: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xc = R @ X + t
    if Xc[2] <= 0:
        return np.array([np.nan, np.nan])
    p = K @ Xc
    return p[:2] / p[2]


def _wand_template_fit(points_3d: np.ndarray, wand: WandSpec) -> np.ndarray:
    """Project a 3-point measurement onto the rigid wand model.

    Uses Procrustes alignment between the measured points and the
    wand-local template. Returns the fitted points (still in world
    coordinates) which by construction obey the wand geometry.
    """
    template = wand.reference_points()
    centroid_meas = points_3d.mean(axis=0)
    centroid_tmpl = template.mean(axis=0)
    P = points_3d - centroid_meas
    Q = template - centroid_tmpl
    H = Q.T @ P
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (template - centroid_tmpl) @ R.T + centroid_meas


def calibrate_extrinsic_wand(
    intrinsics: list[Intrinsics],
    observations_per_camera: list[list[WandObservation]],
    wand: WandSpec,
    *,
    seed_camera_pair: tuple[int, int] = (0, 1),
    bundle_max_iterations: int = 50,
    notes: str = "",
    session_id: str | None = None,
) -> ExtrinsicSession:
    """Recover camera extrinsics from synchronised wand observations.

    Parameters
    ----------
    intrinsics:
        Per-camera :class:`Intrinsics`, indexed in the same order as
        ``observations_per_camera``. Camera 0's pose is fixed at world
        origin so the result is in camera-0 coordinates.
    observations_per_camera:
        ``observations_per_camera[i]`` is the list of frames in which
        camera ``i`` saw the wand. Frame indices align across cameras.
    wand:
        Wand geometry (defaults to the 50/25 cm spec from the plan).
    seed_camera_pair:
        Pair to use for the initial relative-pose recovery.

    Returns
    -------
    A locked :class:`ExtrinsicSession` ready to be saved.
    """
    n_cams = len(intrinsics)
    if len(observations_per_camera) != n_cams:
        raise ValueError("observations_per_camera must match intrinsics length")
    if n_cams < 2:
        raise ValueError("need at least two cameras for extrinsic calibration")

    # Group observations by frame index to build correspondences.
    by_frame: dict[int, dict[int, np.ndarray]] = {}
    for cam_idx, observations in enumerate(observations_per_camera):
        for obs in observations:
            by_frame.setdefault(obs.frame_index, {})[cam_idx] = obs.points

    a_idx, b_idx = seed_camera_pair
    seed_frames = [
        f
        for f, cams in by_frame.items()
        if a_idx in cams and b_idx in cams
        and not np.isnan(cams[a_idx]).any()
        and not np.isnan(cams[b_idx]).any()
    ]
    if len(seed_frames) < 6:
        raise RuntimeError(
            f"need ≥6 frames where both seed cameras see all 3 balls; got {len(seed_frames)}"
        )

    # --- Step 1: relative pose between the two seed cameras via essential matrix.
    pts_a = np.concatenate([by_frame[f][a_idx] for f in seed_frames])  # (3F, 2)
    pts_b = np.concatenate([by_frame[f][b_idx] for f in seed_frames])
    K_a = intrinsics[a_idx].K
    K_b = intrinsics[b_idx].K

    E, _ = cv2.findEssentialMat(pts_a, pts_b, K_a, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("essential matrix recovery failed")
    _, R_b, t_b, _ = cv2.recoverPose(E, pts_a, pts_b, K_b)
    R_b = R_b.astype(np.float64)
    t_b = t_b.reshape(3).astype(np.float64)

    # Triangulate seed-frame wand points up to scale.
    cam_a = CameraModel(
        name=intrinsics[a_idx].camera_name,
        K=K_a,
        dist=np.zeros(5),
        R=np.eye(3),
        t=np.zeros(3),
    )
    cam_b_init = CameraModel(
        name=intrinsics[b_idx].camera_name,
        K=K_b,
        dist=np.zeros(5),
        R=R_b,
        t=t_b,
    )
    seed_3d = _triangulate_pair(cam_a, cam_b_init, pts_a, pts_b)  # (3F, 3) up to scale

    # Recover scale: median over frames of (red↔blue distance / wand.length_m).
    scale_estimates: list[float] = []
    for f_idx, frame in enumerate(seed_frames):
        red = seed_3d[3 * f_idx + 0]
        blue = seed_3d[3 * f_idx + 2]
        d = float(np.linalg.norm(blue - red))
        if d > 1e-6:
            scale_estimates.append(wand.length_m / d)
    if not scale_estimates:
        raise RuntimeError("could not recover scale from seed triangulations")
    scale = float(np.median(scale_estimates))
    t_b *= scale
    seed_3d *= scale

    # --- Step 2: localise remaining cameras with PnP against the seed 3D points.
    extrinsic_R: list[np.ndarray] = [np.eye(3) for _ in range(n_cams)]
    extrinsic_t: list[np.ndarray] = [np.zeros(3) for _ in range(n_cams)]
    extrinsic_R[a_idx] = np.eye(3)
    extrinsic_t[a_idx] = np.zeros(3)
    extrinsic_R[b_idx] = R_b
    extrinsic_t[b_idx] = t_b

    for cam_idx in range(n_cams):
        if cam_idx in (a_idx, b_idx):
            continue
        # Find frames where this camera and at least one seed camera both saw the wand.
        usable_3d: list[np.ndarray] = []
        usable_2d: list[np.ndarray] = []
        for f_idx, frame in enumerate(seed_frames):
            cams = by_frame[frame]
            if cam_idx not in cams:
                continue
            pts_2d = cams[cam_idx]
            if np.isnan(pts_2d).any():
                continue
            for k in range(3):
                usable_3d.append(seed_3d[3 * f_idx + k])
                usable_2d.append(pts_2d[k])
        if len(usable_3d) < 6:
            raise RuntimeError(
                f"camera {cam_idx} has only {len(usable_3d)} 3D↔2D correspondences (need ≥6)"
            )
        obj = np.asarray(usable_3d, dtype=np.float64)
        img = np.asarray(usable_2d, dtype=np.float64)
        ok, rvec, tvec = cv2.solvePnP(
            obj, img, intrinsics[cam_idx].K, None, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            raise RuntimeError(f"solvePnP failed for camera {cam_idx}")
        extrinsic_R[cam_idx] = _rodrigues_to_R(rvec.reshape(3))
        extrinsic_t[cam_idx] = tvec.reshape(3)

    # --- Step 3: bundle adjustment with rigid-wand constraint.
    frames_sorted = sorted(by_frame.keys())
    # Per-frame wand pose: 6 DoF (rotation + translation) of the wand frame.
    # Use the seed triangulation as initialisation where available; otherwise
    # triangulate from the now-localised cameras.
    wand_template = wand.reference_points()
    wand_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    cams_initialised = [
        CameraModel(
            name=intrinsics[i].camera_name,
            K=intrinsics[i].K,
            dist=np.zeros(5),
            R=extrinsic_R[i],
            t=extrinsic_t[i],
        )
        for i in range(n_cams)
    ]
    for frame in frames_sorted:
        cams_seeing = [c for c in by_frame[frame].keys() if not np.isnan(by_frame[frame][c]).any()]
        if len(cams_seeing) < 2:
            continue
        i, j = cams_seeing[:2]
        pts_3d = _triangulate_pair(
            cams_initialised[i],
            cams_initialised[j],
            by_frame[frame][i],
            by_frame[frame][j],
        )
        fitted = _wand_template_fit(pts_3d, wand)
        # Recover rigid pose of the wand template that produced ``fitted``.
        centroid = fitted.mean(axis=0) - wand_template.mean(axis=0)
        H = (wand_template - wand_template.mean(axis=0)).T @ (
            fitted - fitted.mean(axis=0)
        )
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        Rw = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
        tw = fitted.mean(axis=0) - Rw @ wand_template.mean(axis=0)
        wand_poses[frame] = (Rw, tw)

    if not wand_poses:
        raise RuntimeError("bundle adjustment has no usable frames")

    valid_frames = sorted(wand_poses.keys())
    # Pack parameters: [r_cam_1, t_cam_1, ..., r_cam_{n-1}, t_cam_{n-1},
    #                   r_wand_f, t_wand_f for each frame].
    # Camera 0 (a_idx) stays at identity. Reorder so a_idx is at position 0.
    cam_order = [a_idx] + [i for i in range(n_cams) if i != a_idx]

    def pack() -> np.ndarray:
        x: list[float] = []
        for cam_idx in cam_order[1:]:
            x.extend(_R_to_rodrigues(extrinsic_R[cam_idx]).tolist())
            x.extend(extrinsic_t[cam_idx].tolist())
        for f in valid_frames:
            Rw, tw = wand_poses[f]
            x.extend(_R_to_rodrigues(Rw).tolist())
            x.extend(tw.tolist())
        return np.asarray(x, dtype=np.float64)

    n_free_cams = n_cams - 1
    n_frames = len(valid_frames)

    def unpack(x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
        Rs = [np.eye(3)]
        ts = [np.zeros(3)]
        offset = 0
        for _ in range(n_free_cams):
            r = x[offset : offset + 3]
            t = x[offset + 3 : offset + 6]
            Rs.append(_rodrigues_to_R(r))
            ts.append(t.copy())
            offset += 6
        wand_list: list[tuple[np.ndarray, np.ndarray]] = []
        for _ in range(n_frames):
            r = x[offset : offset + 3]
            t = x[offset + 3 : offset + 6]
            wand_list.append((_rodrigues_to_R(r), t.copy()))
            offset += 6
        # Reorder Rs/ts back to original camera indexing.
        R_full = [None] * n_cams
        t_full = [None] * n_cams
        for slot, cam_idx in enumerate(cam_order):
            R_full[cam_idx] = Rs[slot]
            t_full[cam_idx] = ts[slot]
        return R_full, t_full, wand_list

    def residuals(x: np.ndarray) -> np.ndarray:
        R_full, t_full, wand_list = unpack(x)
        out: list[float] = []
        for w_idx, frame in enumerate(valid_frames):
            Rw, tw = wand_list[w_idx]
            world_pts = (Rw @ wand_template.T).T + tw
            obs_per_cam = by_frame[frame]
            for cam_idx, pixel_pts in obs_per_cam.items():
                if np.isnan(pixel_pts).any():
                    continue
                K = intrinsics[cam_idx].K
                Rc = R_full[cam_idx]
                tc = t_full[cam_idx]
                for k in range(3):
                    proj = _project(K, Rc, tc, world_pts[k])
                    if np.isnan(proj).any():
                        out.extend([1e3, 1e3])
                    else:
                        out.extend((proj - pixel_pts[k]).tolist())
        return np.asarray(out, dtype=np.float64)

    x0 = pack()
    result = least_squares(
        residuals,
        x0,
        method="lm",
        max_nfev=bundle_max_iterations * len(x0),
        xtol=1e-9,
        ftol=1e-9,
    )
    R_final, t_final, _ = unpack(result.x)

    rms = float(np.sqrt(np.mean(result.fun ** 2))) if result.fun.size else float("inf")

    entries = tuple(
        ExtrinsicEntry(
            camera_name=intrinsics[i].camera_name,
            R=R_final[i],
            t=t_final[i],
        )
        for i in range(n_cams)
    )
    return ExtrinsicSession(
        session_id=session_id or utc_now_isoformat().replace(":", "").replace("-", ""),
        created_at=utc_now_isoformat(),
        cameras=entries,
        bundle_adjustment_rms_px=rms,
        notes=notes or "wand calibration (3-ball, 50/25 cm)",
        locked=True,
    )
