"""Tests for ZUPT + foot contact + floor snap."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.calibration.floor import FloorPlane
from lislym.postprocess.zupt import FootContactState, update_foot_contact


def _floor() -> FloorPlane:
    return FloorPlane(normal=np.array([0.0, 0.0, 1.0]), offset=0.0)


def test_standing_still_enters_contact_after_dwell() -> None:
    state = FootContactState(contact_dwell_frames=3, ankle_to_sole_m=0.06)
    floor = _floor()
    pos = np.array([0.1, 0.0, 0.06])

    # Five quiet frames → contact entered after 3.
    contacts = []
    for k in range(5):
        result = update_foot_contact(state, pos, timestamp=k * 0.05, floor=floor)
        contacts.append(result.in_contact)
    assert contacts == [False, False, True, True, True]


def test_contact_anchor_lifted_to_ankle_to_sole_offset() -> None:
    state = FootContactState(contact_dwell_frames=2, ankle_to_sole_m=0.06)
    floor = _floor()
    # Marker is essentially still during quiet stance (sub-mm jitter).
    positions = [
        np.array([0.1, 0.0, 0.0598]),
        np.array([0.1, 0.0, 0.0602]),
        np.array([0.1, 0.0, 0.0599]),
        np.array([0.1, 0.0, 0.0601]),
    ]
    for k, pos in enumerate(positions):
        result = update_foot_contact(state, pos, timestamp=k * 0.05, floor=floor)
    assert result.in_contact is True
    # While snapped, the foot reports the anchor — z exactly at the offset.
    assert result.snapped_position[2] == pytest.approx(0.06, abs=1e-3)


def test_velocity_above_release_breaks_contact() -> None:
    state = FootContactState(contact_dwell_frames=2, release_velocity_m_s=0.20)
    floor = _floor()
    # Establish contact.
    for k in range(3):
        update_foot_contact(state, np.array([0.1, 0.0, 0.06]), timestamp=k * 0.05, floor=floor)
    assert state.in_contact

    # Step out: large jump → velocity above release threshold.
    moving = np.array([0.30, 0.0, 0.10])
    result = update_foot_contact(state, moving, timestamp=0.20, floor=floor)
    assert result.in_contact is False
    np.testing.assert_allclose(result.snapped_position, moving)


def test_walking_signal_alternates_contact() -> None:
    """Synthetic gait cycle: a foot is still for 100 ms then moves for 200 ms."""
    rng = np.random.default_rng(0)
    floor = _floor()
    state = FootContactState(contact_dwell_frames=2)
    contacts: list[bool] = []
    for cycle in range(3):
        # Stance: 5 quiet frames at z=0.06.
        for k in range(5):
            t = cycle * 0.30 + k * 0.02
            pos = np.array([cycle * 0.4, 0.0, 0.06]) + rng.normal(0, 0.0005, 3)
            result = update_foot_contact(state, pos, timestamp=t, floor=floor)
            contacts.append(result.in_contact)
        # Swing: 5 frames moving and rising.
        for k in range(5):
            t = cycle * 0.30 + 0.10 + k * 0.04
            pos = np.array([cycle * 0.4 + 0.1 + 0.05 * k, 0.0, 0.10 + 0.02 * k])
            result = update_foot_contact(state, pos, timestamp=t, floor=floor)
            contacts.append(result.in_contact)
    # Each gait cycle should produce at least 2 stance frames in contact.
    assert sum(contacts[0:5]) >= 2
    assert sum(contacts[5:10]) == 0  # swing frames must NOT be in contact


def test_floor_penetration_pushes_back_when_not_in_contact() -> None:
    """A marker reading below the floor (impossible physically) is lifted up."""
    state = FootContactState(contact_dwell_frames=999)  # never enter contact
    floor = _floor()
    raw = np.array([0.0, 0.0, -0.02])
    result = update_foot_contact(state, raw, timestamp=0.0, floor=floor)
    assert result.snapped_position[2] >= 0.0


def test_zupt_jitter_under_two_mm_during_contact() -> None:
    """During contact the snapped position should not move at all."""
    state = FootContactState(contact_dwell_frames=2)
    floor = _floor()
    rng = np.random.default_rng(0)
    snapshots: list[np.ndarray] = []
    for k in range(20):
        # 2 mm random walk inside the contact thresholds.
        noise = rng.normal(0, 0.0005, 3)
        result = update_foot_contact(state, np.array([0.0, 0.0, 0.060]) + noise, timestamp=k * 0.05, floor=floor)
        if result.in_contact:
            snapshots.append(result.snapped_position)
    arr = np.stack(snapshots)
    spread = np.std(arr, axis=0)
    assert np.max(spread) < 1e-6
