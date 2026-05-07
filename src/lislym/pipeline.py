"""End-to-end tracking pipeline.

Wires the per-frame stages together:

    detections ──► triangulation ──► (optional) BA smoothing
                                       │
                                       ▼
                                  EKF + identity ──► ZUPT/floor snap
                                       │                 │
                                       ▼                 ▼
                                  IK (2-bone legs +       hip yaw via 2-ball belt
                                       upper body)
                                       │
                                       ▼
                                  HMD alignment ──► OSC trackers

The orchestrator is intentionally synchronous and stateless across
calls *except* for the explicit state objects it owns (EKF tracker,
foot-contact state, smoothers, One Euro filters). That makes it easy to
unit test by feeding synthetic detections frame-by-frame.

The pipeline does *not* own a capture loop or a clock. Call
:meth:`Pipeline.process_frame` once per synchronised camera bundle and
let your application decide whether to drive that from a real
``MultiCameraReader`` or from a synthetic test harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lislym.calibration.camera_model import CameraModel
from lislym.calibration.floor import FloorPlane
from lislym.calibration.hmd_align import FrameAlignment
from lislym.calibration.personal import BodySegments
from lislym.detection.color_sphere import Detection
from lislym.fusion.bundle import FrameObservations, SlidingWindowSmoother
from lislym.fusion.marker_state import (
    CandidateDetection,
    MarkerTracker,
    reassign_markers,
)
from lislym.fusion.triangulate import TriangulationResult, triangulate
from lislym.ik.two_bone import solve_two_bone
from lislym.ik.upper_body import HipState, HMDState, UpperBodyJoints, solve_upper_body
from lislym.output.osc_trackers import (
    TrackerSnapshot,
    build_tracker_snapshots,
)
from lislym.postprocess.one_euro import OneEuroFilter
from lislym.postprocess.zupt import FootContactState, update_foot_contact


@dataclass(frozen=True)
class FrameDetections:
    """Per-camera detections for a single synchronised frame.

    ``detections_per_camera[i]`` is the (possibly empty) list returned
    by :func:`color_sphere.detect_spheres` for camera ``i``. The HMD
    and controller poses come from SteamVR for the same timestamp.
    """

    timestamp: float
    detections_per_camera: list[list[Detection]]
    hmd_position_world: np.ndarray
    hmd_yaw_radians: float
    controller_left_world: np.ndarray
    controller_right_world: np.ndarray


@dataclass
class FrameOutput:
    """Everything the pipeline produces in a single frame."""

    timestamp: float
    hip_world: np.ndarray
    ankle_left_world: np.ndarray
    ankle_right_world: np.ndarray
    knee_left_world: np.ndarray
    knee_right_world: np.ndarray
    upper_body: UpperBodyJoints
    tracker_snapshots: list[TrackerSnapshot]
    triangulation_modes: dict[str, str]
    in_contact_left: bool
    in_contact_right: bool


@dataclass
class Pipeline:
    """All-in-one synchronous orchestrator.

    The expected per-frame call pattern is::

        out = pipeline.process_frame(detections)

    where ``detections`` is built by your capture + detection thread.
    """

    cameras: list[CameraModel]
    segments: BodySegments
    floor: FloorPlane
    hmd_alignment: FrameAlignment
    marker_colors: dict[str, str] = field(
        default_factory=lambda: {"hip": "red", "ankle_left": "green", "ankle_right": "blue"}
    )
    bone_constraints_m: dict[tuple[str, str], tuple[float, float]] = field(default_factory=dict)
    use_bundle_smoother: bool = True
    smoother_window: int = 18
    knee_filter_min_cutoff_hz: float = 1.5
    knee_filter_beta: float = 0.05
    forward_axis_world: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))

    tracker: MarkerTracker = field(init=False)
    foot_left_state: FootContactState = field(init=False)
    foot_right_state: FootContactState = field(init=False)
    smoothers: dict[str, SlidingWindowSmoother] = field(init=False)
    knee_left_filter: OneEuroFilter = field(init=False)
    knee_right_filter: OneEuroFilter = field(init=False)
    last_hip_yaw_radians: float = 0.0
    last_position: dict[str, np.ndarray] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        self.tracker = MarkerTracker()
        for name, color in self.marker_colors.items():
            self.tracker.register(name, color)
        self.foot_left_state = FootContactState(
            ankle_to_sole_m=self.segments.ankle_to_sole_m
        )
        self.foot_right_state = FootContactState(
            ankle_to_sole_m=self.segments.ankle_to_sole_m
        )
        self.smoothers = {
            name: SlidingWindowSmoother(
                cameras=self.cameras, window_size=self.smoother_window
            )
            for name in self.marker_colors
        }
        if not self.bone_constraints_m:
            # Reasonable defaults from the segment lengths: ±15 % around
            # the calibrated leg length.
            tol = 0.15 * self.segments.leg_length
            self.bone_constraints_m = {
                ("hip", "ankle_left"): (
                    self.segments.leg_length - tol,
                    self.segments.leg_length + tol,
                ),
                ("hip", "ankle_right"): (
                    self.segments.leg_length - tol,
                    self.segments.leg_length + tol,
                ),
            }
        self.knee_left_filter = OneEuroFilter(
            min_cutoff=self.knee_filter_min_cutoff_hz, beta=self.knee_filter_beta
        )
        self.knee_right_filter = OneEuroFilter(
            min_cutoff=self.knee_filter_min_cutoff_hz, beta=self.knee_filter_beta
        )

    def process_frame(self, frame: FrameDetections) -> FrameOutput:
        # 1. Triangulate every marker that any pair of cameras saw.
        per_marker_observations = self._gather_observations_per_marker(frame)
        marker_world: dict[str, np.ndarray | None] = {}
        triangulation_modes: dict[str, str] = {}
        for marker_name in self.marker_colors:
            obs, residuals = per_marker_observations.get(marker_name, ([], []))
            tri = triangulate(self.cameras, obs, residuals)
            triangulation_modes[marker_name] = tri.mode
            marker_world[marker_name] = tri.point if tri.point is not None else None

            # Push observations into the smoother regardless of mode so the
            # window stays current; only run BA when we have a multi-view
            # solution this frame.
            if tri.point is not None and self.use_bundle_smoother:
                pixel_obs = {
                    cam_idx: pt for cam_idx, pt in enumerate(obs) if pt is not None
                }
                pixel_residuals = {
                    cam_idx: residuals[cam_idx] for cam_idx in pixel_obs
                }
                self.smoothers[marker_name].push(
                    FrameObservations(
                        timestamp=frame.timestamp,
                        pixel_observations=pixel_obs,
                        pixel_residuals=pixel_residuals,
                    )
                )
                smoothed = self.smoothers[marker_name].solve()
                if smoothed is not None and len(smoothed) > 0:
                    marker_world[marker_name] = smoothed[-1]

        # 2. Re-association — match the freshly triangulated points back
        #    to the right names by color × spatial × kinematic.
        candidates: list[CandidateDetection] = []
        candidate_owner: list[str] = []
        for marker_name, point in marker_world.items():
            if point is None:
                continue
            candidates.append(
                CandidateDetection(position=point, color=self.marker_colors[marker_name])
            )
            candidate_owner.append(marker_name)
        assignment = reassign_markers(
            self.tracker, candidates, bone_constraints=self.bone_constraints_m, t_now=frame.timestamp
        )

        # 3. EKF predict + update.
        for state in self.tracker.states.values():
            last_t = state.last_observation_time
            dt = (frame.timestamp - last_t) if last_t is not None else 0.0
            if dt > 0:
                state.predict(dt=dt)
        for name, det_idx in assignment.items():
            cand = candidates[det_idx]
            self.tracker.update(name, cand.position, np.eye(3) * 1e-4, frame.timestamp)

        # 4. Foot-contact / floor snap on the ankle states.
        ankle_left_state = self.tracker.states["ankle_left"]
        ankle_right_state = self.tracker.states["ankle_right"]
        hip_state = self.tracker.states["hip"]

        ankle_left_pos = ankle_left_state.position if ankle_left_state.initialised else self.last_position.get("ankle_left", np.zeros(3))
        ankle_right_pos = ankle_right_state.position if ankle_right_state.initialised else self.last_position.get("ankle_right", np.zeros(3))
        hip_pos = hip_state.position if hip_state.initialised else self.last_position.get("hip", np.zeros(3))

        contact_l = update_foot_contact(self.foot_left_state, ankle_left_pos, frame.timestamp, self.floor)
        contact_r = update_foot_contact(self.foot_right_state, ankle_right_pos, frame.timestamp, self.floor)
        ankle_left_pos = contact_l.snapped_position
        ankle_right_pos = contact_r.snapped_position

        self.last_position["hip"] = hip_pos
        self.last_position["ankle_left"] = ankle_left_pos
        self.last_position["ankle_right"] = ankle_right_pos

        # 5. Hip yaw — for the 3-marker case it must be inferred from the
        #    ankle-to-hip geometry plus a temporal smoothness assumption,
        #    OR taken from the HMD if no two-ball hip belt is available.
        #    We default to "follow the HMD's yaw filtered toward the
        #    leg-spanning direction".
        hip_yaw = self._estimate_hip_yaw(
            hip_pos, ankle_left_pos, ankle_right_pos, fallback_yaw=frame.hmd_yaw_radians
        )
        self.last_hip_yaw_radians = hip_yaw

        # 6. Two-bone IK for each leg (knee from hip + ankle).
        forward = np.array([np.cos(hip_yaw), np.sin(hip_yaw), 0.0])
        knee_left = solve_two_bone(
            hip_pos,
            ankle_left_pos,
            self.segments.thigh_length,
            self.segments.shin_length,
            forward,
        ).intermediate
        knee_right = solve_two_bone(
            hip_pos,
            ankle_right_pos,
            self.segments.thigh_length,
            self.segments.shin_length,
            forward,
        ).intermediate
        knee_left = self.knee_left_filter.filter(knee_left, frame.timestamp)
        knee_right = self.knee_right_filter.filter(knee_right, frame.timestamp)

        # 7. Upper-body IK.
        upper_body = solve_upper_body(
            hip=HipState(position=hip_pos, yaw_radians=hip_yaw),
            hmd=HMDState(position=frame.hmd_position_world, yaw_radians=frame.hmd_yaw_radians),
            controller_left=frame.controller_left_world,
            controller_right=frame.controller_right_world,
            segments=self.segments,
        )

        # 8. World → SteamVR alignment for the OSC channel.
        hip_steamvr = self.hmd_alignment.apply(hip_pos)
        ankle_l_steamvr = self.hmd_alignment.apply(ankle_left_pos)
        ankle_r_steamvr = self.hmd_alignment.apply(ankle_right_pos)
        snapshots = build_tracker_snapshots(
            hip_world=hip_steamvr,
            hip_yaw_radians=hip_yaw,
            foot_left_world=ankle_l_steamvr,
            foot_left_yaw_radians=hip_yaw,
            foot_right_world=ankle_r_steamvr,
            foot_right_yaw_radians=hip_yaw,
        )

        return FrameOutput(
            timestamp=frame.timestamp,
            hip_world=hip_pos,
            ankle_left_world=ankle_left_pos,
            ankle_right_world=ankle_right_pos,
            knee_left_world=knee_left,
            knee_right_world=knee_right,
            upper_body=upper_body,
            tracker_snapshots=snapshots,
            triangulation_modes=triangulation_modes,
            in_contact_left=contact_l.in_contact,
            in_contact_right=contact_r.in_contact,
        )

    def _gather_observations_per_marker(
        self, frame: FrameDetections
    ) -> dict[str, tuple[list[tuple[float, float] | None], list[float]]]:
        out: dict[str, tuple[list[tuple[float, float] | None], list[float]]] = {}
        n_cams = len(self.cameras)
        for marker_name in self.marker_colors:
            observations: list[tuple[float, float] | None] = [None] * n_cams
            residuals: list[float] = [1.0] * n_cams
            for cam_idx, dets in enumerate(frame.detections_per_camera):
                for d in dets:
                    if d.marker == marker_name:
                        observations[cam_idx] = (d.cx, d.cy)
                        residuals[cam_idx] = max(d.pixel_residual, 0.05)
                        break
            out[marker_name] = (observations, residuals)
        return out

    def _estimate_hip_yaw(
        self,
        hip: np.ndarray,
        ankle_left: np.ndarray,
        ankle_right: np.ndarray,
        *,
        fallback_yaw: float,
    ) -> float:
        """Yaw from the L→R ankle vector, fallback to HMD yaw on collapse."""
        lateral = ankle_right - ankle_left
        lateral[2] = 0.0
        norm = float(np.linalg.norm(lateral))
        if norm < 0.05:
            return fallback_yaw
        # The hip's forward direction is perpendicular to the ankle line in
        # the floor plane, on the side the user faces. We disambiguate
        # using the HMD yaw (always available, may be stale at session
        # start by ~30°).
        right = lateral / norm
        forward_candidate = np.array([-right[1], right[0], 0.0])
        hmd_forward = np.array([np.cos(fallback_yaw), np.sin(fallback_yaw), 0.0])
        if np.dot(forward_candidate, hmd_forward) < 0:
            forward_candidate = -forward_candidate
        return float(np.arctan2(forward_candidate[1], forward_candidate[0]))
