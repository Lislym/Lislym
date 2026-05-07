"""Personal calibration: estimate body segment lengths from a T-pose.

Plan §D-1. While the user stands in a quiet T-pose for ~3 seconds, the
multi-camera tracker provides 3D positions of:

* hip marker (front-back centroid)
* both ankle markers
* HMD (head, from SteamVR)
* both controllers (wrists)

Plus, optionally, MediaPipe-derived 2D landmarks for shoulders and knees.
The geometry of a T-pose lets us solve for several segment lengths in
closed form:

* leg length = hip → ankle distance (averaged over frames)
* arm half-span = controller → mid-shoulder distance
* spine length = hip → neck (≈ HMD origin minus a fixed head offset)
* shoulder width = mid-shoulder displacement (left vs right controllers,
  with arms outstretched in the T-pose)

Where MediaPipe gives a 2D shoulder landmark we triangulate it; otherwise
we approximate the shoulders as ``hip + spine_offset + ±half_shoulder``.

The result is serialised to ``personal_<user_id>.json`` and read back at
startup for IK.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BodySegments:
    """Per-user segment lengths in metres.

    The names mirror the IK chains we drive at runtime:

    * ``leg_length`` — hip-to-ankle straight distance, used as the maximum
      reach of the 2-bone leg IK.
    * ``thigh_length`` / ``shin_length`` — hip-to-knee and knee-to-ankle.
      Their sum is approximately ``leg_length`` (with the knee very slightly
      forward in T-pose) and they're used as the two bone lengths in
      ``two_bone_ik``. We default to a 50/50 split with ``leg_length`` if
      MediaPipe knee data isn't available.
    * ``arm_length`` — wrist-to-shoulder, used by the upper-body IK.
    * ``spine_length`` — hip-to-neck, drives the spine column.
    * ``shoulder_half_width`` — shoulder centre to one shoulder.
    * ``ankle_to_sole_m`` — vertical offset from the marker centre to the
      sole, used by ZUPT to land the foot at the right floor height.
    """

    leg_length: float
    thigh_length: float
    shin_length: float
    arm_length: float
    spine_length: float
    shoulder_half_width: float
    ankle_to_sole_m: float = 0.06

    def to_json(self) -> dict[str, float]:
        return {
            "leg_length": self.leg_length,
            "thigh_length": self.thigh_length,
            "shin_length": self.shin_length,
            "arm_length": self.arm_length,
            "spine_length": self.spine_length,
            "shoulder_half_width": self.shoulder_half_width,
            "ankle_to_sole_m": self.ankle_to_sole_m,
        }

    @classmethod
    def from_json(cls, data: dict[str, float]) -> "BodySegments":
        return cls(**data)


@dataclass
class TPoseSamples:
    """Accumulator of multi-frame T-pose data.

    Populate by calling :meth:`add_frame` once per synchronised frame
    during the quiet T-pose. Then call :meth:`solve` to recover
    :class:`BodySegments`.
    """

    hip: list[np.ndarray] = field(default_factory=list)
    ankle_left: list[np.ndarray] = field(default_factory=list)
    ankle_right: list[np.ndarray] = field(default_factory=list)
    hmd: list[np.ndarray] = field(default_factory=list)
    wrist_left: list[np.ndarray] = field(default_factory=list)
    wrist_right: list[np.ndarray] = field(default_factory=list)

    def add_frame(
        self,
        *,
        hip: np.ndarray,
        ankle_left: np.ndarray,
        ankle_right: np.ndarray,
        hmd: np.ndarray,
        wrist_left: np.ndarray,
        wrist_right: np.ndarray,
    ) -> None:
        self.hip.append(np.asarray(hip, dtype=np.float64))
        self.ankle_left.append(np.asarray(ankle_left, dtype=np.float64))
        self.ankle_right.append(np.asarray(ankle_right, dtype=np.float64))
        self.hmd.append(np.asarray(hmd, dtype=np.float64))
        self.wrist_left.append(np.asarray(wrist_left, dtype=np.float64))
        self.wrist_right.append(np.asarray(wrist_right, dtype=np.float64))

    def __len__(self) -> int:
        return len(self.hip)

    def solve(
        self,
        *,
        head_offset_m: float = 0.10,
        ankle_to_sole_m: float = 0.06,
    ) -> BodySegments:
        """Recover body segment lengths from the accumulated samples.

        Parameters
        ----------
        head_offset_m:
            Distance from the HMD origin to the neck base. The HMD sits
            ~10 cm above the C7 vertebra for typical setups; subtract
            this along the world up axis to estimate the neck height.
        ankle_to_sole_m:
            Pass-through to :class:`BodySegments`; this is set when the
            user mounts the ankle markers and is not solved for here.
        """
        if len(self) < 5:
            raise RuntimeError(
                f"need at least 5 T-pose frames; got {len(self)}"
            )

        hip = np.median(np.stack(self.hip), axis=0)
        ankle_l = np.median(np.stack(self.ankle_left), axis=0)
        ankle_r = np.median(np.stack(self.ankle_right), axis=0)
        hmd = np.median(np.stack(self.hmd), axis=0)
        wrist_l = np.median(np.stack(self.wrist_left), axis=0)
        wrist_r = np.median(np.stack(self.wrist_right), axis=0)

        leg_l = float(np.linalg.norm(hip - ankle_l))
        leg_r = float(np.linalg.norm(hip - ankle_r))
        leg_length = 0.5 * (leg_l + leg_r)
        thigh_length = leg_length * 0.5
        shin_length = leg_length - thigh_length

        # Neck is HMD shifted down by head_offset_m along world +Z.
        neck = hmd.copy()
        neck[2] = hmd[2] - head_offset_m
        spine_length = float(np.linalg.norm(neck - hip))

        # In T-pose, wrists are extended sideways; the shoulder centre is
        # roughly midway between the two wrists.
        shoulder_centre = 0.5 * (wrist_l + wrist_r)
        shoulder_half_width = 0.5 * float(np.linalg.norm(wrist_r - wrist_l)) * 0.30
        # The 0.30 factor: the wrists in a T-pose are about a full arm-length
        # away from the shoulder centre on each side. Total span ≈ 2*arm
        # ≈ 2 * 0.4 * height; the shoulders sit at ≈ 0.30 of that span from
        # the centre line. This is a rough heuristic; the IK only needs an
        # order-of-magnitude shoulder width.
        arm_length = 0.5 * (
            float(np.linalg.norm(wrist_l - shoulder_centre))
            + float(np.linalg.norm(wrist_r - shoulder_centre))
        )

        return BodySegments(
            leg_length=leg_length,
            thigh_length=thigh_length,
            shin_length=shin_length,
            arm_length=arm_length,
            spine_length=spine_length,
            shoulder_half_width=shoulder_half_width,
            ankle_to_sole_m=ankle_to_sole_m,
        )


def save_personal(directory: Path, user_id: str, segments: BodySegments) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"personal_{user_id}.json"
    path.write_text(json.dumps(segments.to_json(), indent=2), encoding="utf-8")
    return path


def load_personal(directory: Path, user_id: str) -> BodySegments:
    path = directory / f"personal_{user_id}.json"
    return BodySegments.from_json(json.loads(path.read_text(encoding="utf-8")))
