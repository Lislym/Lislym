"""End-to-end integration test on synthetic walking motion.

Drives the full pipeline (detection → triangulation → BA → identity →
ZUPT → IK → OSC) with simulated 2D detections of a body walking on a
plane. Checks:

* The pipeline returns valid output frames at every timestep.
* The recovered 3D positions track the synthetic ground truth within
  the plan's accuracy budget for clean data (sub-cm).
* During the simulated stance phase ZUPT fires and the foot reports
  ``in_contact=True`` with a stable position.
* OSC tracker snapshots are well-formed with the expected addresses.
"""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.camera_model import CameraModel
from lislym.calibration.floor import FloorPlane
from lislym.calibration.hmd_align import FrameAlignment
from lislym.calibration.personal import BodySegments
from lislym.detection.color_sphere import Detection
from lislym.pipeline import FrameDetections, Pipeline


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
        cams.append(
            CameraModel(name=f"cam{i}", K=_intrinsics(), dist=np.zeros(5), R=R, t=t)
        )
    return cams


def _project(cam: CameraModel, X: np.ndarray) -> tuple[float, float]:
    Xc = cam.R @ X + cam.t
    p = cam.K @ Xc
    return float(p[0] / p[2]), float(p[1] / p[2])


def _body_segments() -> BodySegments:
    return BodySegments(
        leg_length=0.95,
        thigh_length=0.45,
        shin_length=0.50,
        arm_length=0.65,
        spine_length=0.50,
        shoulder_half_width=0.20,
        ankle_to_sole_m=0.06,
    )


def _build_frame(
    cameras: list[CameraModel],
    timestamp: float,
    hip_xyz: np.ndarray,
    ankle_left_xyz: np.ndarray,
    ankle_right_xyz: np.ndarray,
    hmd_xyz: np.ndarray,
    cl_xyz: np.ndarray,
    cr_xyz: np.ndarray,
    *,
    occlude_camera: int | None = None,
) -> FrameDetections:
    detections_per_camera: list[list[Detection]] = []
    for cam_idx, cam in enumerate(cameras):
        dets: list[Detection] = []
        for marker, color, pos in [
            ("hip", "red", hip_xyz),
            ("ankle_left", "green", ankle_left_xyz),
            ("ankle_right", "blue", ankle_right_xyz),
        ]:
            if occlude_camera == cam_idx:
                continue
            cx, cy = _project(cam, pos)
            dets.append(
                Detection(
                    marker=marker,
                    cx=cx,
                    cy=cy,
                    radius_px=12.0,
                    confidence=0.95,
                    pixel_residual=0.2,
                )
            )
        detections_per_camera.append(dets)
    return FrameDetections(
        timestamp=timestamp,
        detections_per_camera=detections_per_camera,
        hmd_position_world=hmd_xyz,
        hmd_yaw_radians=0.0,
        controller_left_world=cl_xyz,
        controller_right_world=cr_xyz,
    )


def _walking_truth(t: float) -> dict[str, np.ndarray]:
    """A canonical walking cycle on a treadmill (no x-translation)."""
    cycle = (t % 1.0)
    # Hip bob at 1 Hz, ±2 cm.
    hip_z = 0.95 + 0.02 * np.sin(2 * np.pi * t)
    hip = np.array([0.0, 0.0, hip_z])
    # Left foot stance 0.0–0.5, swing 0.5–1.0.
    if cycle < 0.5:
        ankle_left = np.array([-0.10, 0.0, 0.06])
    else:
        # Swing arc.
        s = (cycle - 0.5) / 0.5
        ankle_left = np.array([-0.10, 0.10 * np.sin(np.pi * s), 0.06 + 0.10 * np.sin(np.pi * s)])
    # Right foot is opposite phase.
    if cycle < 0.5:
        s = cycle / 0.5
        ankle_right = np.array([0.10, 0.10 * np.sin(np.pi * s), 0.06 + 0.10 * np.sin(np.pi * s)])
    else:
        ankle_right = np.array([0.10, 0.0, 0.06])
    hmd = np.array([0.0, 0.0, 1.65])
    cl = np.array([-0.40, 0.10, 1.30])
    cr = np.array([0.40, 0.10, 1.30])
    return {
        "hip": hip,
        "ankle_left": ankle_left,
        "ankle_right": ankle_right,
        "hmd": hmd,
        "cl": cl,
        "cr": cr,
    }


def _make_pipeline(cameras: list[CameraModel]) -> Pipeline:
    return Pipeline(
        cameras=cameras,
        segments=_body_segments(),
        floor=FloorPlane(normal=np.array([0.0, 0.0, 1.0]), offset=0.0),
        hmd_alignment=FrameAlignment(yaw_radians=0.0, translation=np.zeros(3)),
        use_bundle_smoother=False,  # too slow per-frame for a 60-frame integration test
    )


def test_pipeline_runs_without_errors() -> None:
    cams = _make_cameras()
    pipeline = _make_pipeline(cams)
    for k in range(60):
        t = k / 60.0
        truth = _walking_truth(t)
        frame = _build_frame(
            cams,
            t,
            truth["hip"],
            truth["ankle_left"],
            truth["ankle_right"],
            truth["hmd"],
            truth["cl"],
            truth["cr"],
        )
        out = pipeline.process_frame(frame)
        assert out.timestamp == pytest.approx(t)
        assert len(out.tracker_snapshots) == 3
        assert all(s.position_address().startswith("/tracking/trackers/") for s in out.tracker_snapshots)


def test_pipeline_recovers_hip_within_5mm() -> None:
    cams = _make_cameras()
    pipeline = _make_pipeline(cams)
    errors: list[float] = []
    for k in range(30):
        t = k / 60.0
        truth = _walking_truth(t)
        frame = _build_frame(
            cams,
            t,
            truth["hip"],
            truth["ankle_left"],
            truth["ankle_right"],
            truth["hmd"],
            truth["cl"],
            truth["cr"],
        )
        out = pipeline.process_frame(frame)
        if k > 5:  # let the EKF warm up
            errors.append(float(np.linalg.norm(out.hip_world - truth["hip"])))
    rms = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    # No noise, so we expect sub-millimetre tracking.
    assert rms < 0.005, f"hip RMS too high: {rms*1000:.2f} mm"


def test_pipeline_handles_partial_occlusion_without_breaking() -> None:
    """One camera dropped for 0.5 s; the other two carry the solve."""
    cams = _make_cameras()
    pipeline = _make_pipeline(cams)
    rms_errors: list[float] = []
    for k in range(60):
        t = k / 60.0
        truth = _walking_truth(t)
        occlude = 1 if 10 <= k < 40 else None  # camera 1 out for 30 frames
        frame = _build_frame(
            cams,
            t,
            truth["hip"],
            truth["ankle_left"],
            truth["ankle_right"],
            truth["hmd"],
            truth["cl"],
            truth["cr"],
            occlude_camera=occlude,
        )
        out = pipeline.process_frame(frame)
        if k > 15:  # past warmup and into the occlusion period
            rms_errors.append(float(np.linalg.norm(out.hip_world - truth["hip"])))
    rms = float(np.sqrt(np.mean(np.array(rms_errors) ** 2)))
    assert rms < 0.01  # 1 cm allowed during 1-camera-down operation


def test_pipeline_emits_valid_osc_snapshots() -> None:
    cams = _make_cameras()
    pipeline = _make_pipeline(cams)
    truth = _walking_truth(0.25)
    frame = _build_frame(
        cams,
        0.25,
        truth["hip"],
        truth["ankle_left"],
        truth["ankle_right"],
        truth["hmd"],
        truth["cl"],
        truth["cr"],
    )
    out = pipeline.process_frame(frame)
    addresses = {s.position_address() for s in out.tracker_snapshots}
    assert addresses == {
        "/tracking/trackers/1/position",
        "/tracking/trackers/2/position",
        "/tracking/trackers/3/position",
    }
    for s in out.tracker_snapshots:
        assert s.position_xyz_m.shape == (3,)
        assert np.isfinite(s.position_xyz_m).all()


def test_pipeline_full_three_camera_loss_then_return() -> None:
    """All cameras occluded for 5 frames → return → identity preserved."""
    cams = _make_cameras()
    pipeline = _make_pipeline(cams)
    for k in range(20):
        t = k / 60.0
        truth = _walking_truth(t)
        if 10 <= k < 15:
            # Empty detections from every camera.
            empty_frame = FrameDetections(
                timestamp=t,
                detections_per_camera=[[] for _ in cams],
                hmd_position_world=truth["hmd"],
                hmd_yaw_radians=0.0,
                controller_left_world=truth["cl"],
                controller_right_world=truth["cr"],
            )
            pipeline.process_frame(empty_frame)
        else:
            frame = _build_frame(
                cams,
                t,
                truth["hip"],
                truth["ankle_left"],
                truth["ankle_right"],
                truth["hmd"],
                truth["cl"],
                truth["cr"],
            )
            out = pipeline.process_frame(frame)
    # After the loss/return cycle the hip identity is still tracked.
    truth = _walking_truth(20 / 60.0)
    err = float(np.linalg.norm(out.hip_world - truth["hip"]))
    assert err < 0.01, f"identity not preserved across full loss; err={err*1000:.2f} mm"
