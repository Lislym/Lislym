"""VRChat OSC tracker output.

VRChat's "OSC Trackers" feature accepts up to 8 trackers, each with a
position and rotation channel. Addresses follow the pattern documented
at https://docs.vrchat.com/docs/osc-trackers :

* ``/tracking/trackers/{1..8}/position`` — three floats (x, y, z) in
  metres in SteamVR/Unity coordinates (Y-up, right-handed).
* ``/tracking/trackers/{1..8}/rotation`` — three floats (pitch, yaw,
  roll) in degrees, applied in YXZ order.

For the 3-marker setup we route:

* tracker 1 → hip
* tracker 2 → left foot
* tracker 3 → right foot

This module only depends on ``python-osc`` for the actual UDP send;
the rest is plain dataclasses so :class:`TrackerSnapshot` is unit
testable without a network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackerSnapshot:
    """A single tracker's instantaneous pose in SteamVR coordinates."""

    index: int  # 1..8
    position_xyz_m: np.ndarray
    rotation_pyr_deg: np.ndarray  # pitch, yaw, roll

    def __post_init__(self) -> None:
        if not 1 <= self.index <= 8:
            raise ValueError("tracker index must be in [1, 8]")
        if self.position_xyz_m.shape != (3,):
            raise ValueError("position must be a 3-vector")
        if self.rotation_pyr_deg.shape != (3,):
            raise ValueError("rotation must be a 3-vector (pitch, yaw, roll)")

    def position_address(self) -> str:
        return f"/tracking/trackers/{self.index}/position"

    def rotation_address(self) -> str:
        return f"/tracking/trackers/{self.index}/rotation"


def steamvr_from_world(position_world: np.ndarray) -> np.ndarray:
    """Convert our world coordinates (Z-up) to SteamVR/Unity (Y-up).

    Our calibration uses Z-up, +X right, +Y forward (right-handed). VRChat
    consumes Y-up, +X right, +Z forward (right-handed). The mapping is
    just a 90° rotation about +X: ``(x, y, z) → (x, z, -y)``.
    """
    x, y, z = position_world
    return np.array([x, z, -y], dtype=np.float64)


def euler_yaw_to_pyr(yaw_radians: float) -> np.ndarray:
    """Convert a single-axis (world up) yaw into VRChat's PYR triple in degrees.

    For trackers we don't compute pitch/roll from the markers — those
    come from the surrounding chain. Hip yaw is the dominant rotation
    component for a hip tracker; the feet trackers get yaw aligned with
    walking direction and zero pitch/roll, which lets VRChat's IK keep
    the avatar's foot rotation natural.
    """
    return np.array([0.0, np.degrees(yaw_radians), 0.0], dtype=np.float64)


def build_tracker_snapshots(
    *,
    hip_world: np.ndarray,
    hip_yaw_radians: float,
    foot_left_world: np.ndarray,
    foot_left_yaw_radians: float,
    foot_right_world: np.ndarray,
    foot_right_yaw_radians: float,
) -> list[TrackerSnapshot]:
    """Build the three OSC tracker snapshots from world-frame inputs."""
    return [
        TrackerSnapshot(
            index=1,
            position_xyz_m=steamvr_from_world(hip_world),
            rotation_pyr_deg=euler_yaw_to_pyr(hip_yaw_radians),
        ),
        TrackerSnapshot(
            index=2,
            position_xyz_m=steamvr_from_world(foot_left_world),
            rotation_pyr_deg=euler_yaw_to_pyr(foot_left_yaw_radians),
        ),
        TrackerSnapshot(
            index=3,
            position_xyz_m=steamvr_from_world(foot_right_world),
            rotation_pyr_deg=euler_yaw_to_pyr(foot_right_yaw_radians),
        ),
    ]


class OSCTrackerSender:
    """Thin wrapper over python-osc that sends tracker snapshots."""

    def __init__(self, host: str = "127.0.0.1", port: int = 9000) -> None:
        # Lazy import so the rest of the package can be unit tested without
        # python-osc installed.
        from pythonosc.udp_client import SimpleUDPClient

        self._client = SimpleUDPClient(host, port)

    def send(self, snapshots: list[TrackerSnapshot]) -> None:
        for snap in snapshots:
            self._client.send_message(snap.position_address(), snap.position_xyz_m.tolist())
            self._client.send_message(snap.rotation_address(), snap.rotation_pyr_deg.tolist())
