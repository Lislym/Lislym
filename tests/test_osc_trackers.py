"""Tests for VRChat OSC tracker output (network-free).

The actual UDP send is exercised by ``OSCTrackerSender``; here we verify
the address scheme, coordinate conversion, and snapshot construction
without spinning up a socket.
"""

from __future__ import annotations

import numpy as np
import pytest

from lislym.output.osc_trackers import (
    TrackerSnapshot,
    build_tracker_snapshots,
    euler_yaw_to_pyr,
    steamvr_from_world,
)


def test_steamvr_from_world_swaps_axes() -> None:
    np.testing.assert_allclose(steamvr_from_world(np.array([1.0, 2.0, 3.0])), [1.0, 3.0, -2.0])


def test_euler_yaw_to_pyr_in_degrees() -> None:
    np.testing.assert_allclose(
        euler_yaw_to_pyr(np.deg2rad(45.0)), [0.0, 45.0, 0.0], atol=1e-9
    )


def test_tracker_snapshot_address_scheme() -> None:
    snap = TrackerSnapshot(
        index=2,
        position_xyz_m=np.array([0.0, 1.0, 0.0]),
        rotation_pyr_deg=np.array([0.0, 90.0, 0.0]),
    )
    assert snap.position_address() == "/tracking/trackers/2/position"
    assert snap.rotation_address() == "/tracking/trackers/2/rotation"


def test_tracker_snapshot_rejects_invalid_index() -> None:
    with pytest.raises(ValueError):
        TrackerSnapshot(
            index=0,
            position_xyz_m=np.zeros(3),
            rotation_pyr_deg=np.zeros(3),
        )
    with pytest.raises(ValueError):
        TrackerSnapshot(
            index=9,
            position_xyz_m=np.zeros(3),
            rotation_pyr_deg=np.zeros(3),
        )


def test_build_tracker_snapshots_routes_three_indices() -> None:
    snaps = build_tracker_snapshots(
        hip_world=np.array([0.0, 0.0, 1.0]),
        hip_yaw_radians=0.0,
        foot_left_world=np.array([-0.1, 0.0, 0.0]),
        foot_left_yaw_radians=0.0,
        foot_right_world=np.array([0.1, 0.0, 0.0]),
        foot_right_yaw_radians=0.0,
    )
    indices = [s.index for s in snaps]
    assert indices == [1, 2, 3]
    # Y-up conversion: world z=1.0 → SteamVR y=1.0
    assert snaps[0].position_xyz_m[1] == pytest.approx(1.0)
