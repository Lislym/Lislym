"""Zero-velocity update (ZUPT) and foot-contact / floor snap.

Plan §C-3 + §E-2: when a foot is stationary, lock its position completely.
This trades a tiny bit of latency at heel strike for a dramatic reduction
in jitter — the same trick inertial pedestrians use.

The detector watches the magnitude of the ankle velocity and the ankle's
height above the floor. When both are below their thresholds for a small
number of consecutive frames, it declares contact. While contact is
active, the foot's reported position is the *anchor* recorded at contact
onset (with z snapped to the floor + ankle-to-sole offset), and the
filter's velocity is forced to zero, suppressing drift.

When the foot lifts (velocity exceeds release threshold OR height grows
beyond the lift threshold), contact ends. The anchor is discarded and
position passes through unmodified again.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from lislym.calibration.floor import FloorPlane


@dataclass
class FootContactState:
    """Per-foot state machine for ZUPT.

    Attributes
    ----------
    velocity_threshold_m_s:
        Magnitude of velocity below which the foot is "potentially still".
        0.05 m/s is a good starting point for ankle-mounted markers.
    height_threshold_m:
        Maximum height above the floor at which contact can be declared.
        Allow ~3 cm to absorb sole thickness and per-step variation.
    contact_dwell_frames:
        Consecutive frames meeting both thresholds required before
        declaring contact (debounce against transient zero-crossings).
    release_velocity_m_s:
        Hysteresis: velocity above this releases contact.
    release_height_m:
        Hysteresis: height above this releases contact.
    ankle_to_sole_m:
        Vertical offset from the ankle marker centre to the sole. The
        snapped foot rests at ``floor_z - ankle_to_sole_m`` ... no, we
        actually want to express the *ankle* position when the sole is
        on the floor, i.e. ankle_z = floor_z + ankle_to_sole_m. Stored
        positive.
    """

    velocity_threshold_m_s: float = 0.05
    height_threshold_m: float = 0.04
    contact_dwell_frames: int = 3
    release_velocity_m_s: float = 0.20
    release_height_m: float = 0.08
    ankle_to_sole_m: float = 0.06

    in_contact: bool = False
    anchor: np.ndarray | None = None
    candidate_run: int = 0
    history: deque[tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=8))


def _velocity_estimate(history: deque[tuple[float, np.ndarray]]) -> float:
    """Speed in m/s from the most recent two samples; 0 if not enough data."""
    if len(history) < 2:
        return 0.0
    t1, p1 = history[-1]
    t0, p0 = history[-2]
    dt = max(t1 - t0, 1e-6)
    return float(np.linalg.norm(p1 - p0) / dt)


@dataclass
class FootContactResult:
    """Outcome of one update step for a single foot."""

    snapped_position: np.ndarray
    in_contact: bool
    velocity_m_s: float


def update_foot_contact(
    state: FootContactState,
    raw_position: np.ndarray,
    timestamp: float,
    floor: FloorPlane,
) -> FootContactResult:
    """Apply ZUPT + floor snap to a single ankle observation.

    Mutates ``state`` in place. Returns the position the rest of the
    pipeline should consume — that is, the anchor while in contact and
    the raw position otherwise.
    """
    state.history.append((timestamp, raw_position.copy()))
    velocity = _velocity_estimate(state.history)
    raw_height = floor.height_above(raw_position)
    # Compare against the *sole* height: subtract the expected ankle-to-sole
    # offset so a marker resting at the right height for a planted foot
    # registers as "on the floor" with no remainder.
    sole_height = raw_height - state.ankle_to_sole_m

    if state.in_contact:
        # Hysteresis-based release.
        if velocity > state.release_velocity_m_s or sole_height > state.release_height_m:
            state.in_contact = False
            state.anchor = None
            state.candidate_run = 0
            return FootContactResult(snapped_position=raw_position, in_contact=False, velocity_m_s=velocity)
        # Stay snapped to the anchor.
        assert state.anchor is not None
        return FootContactResult(snapped_position=state.anchor.copy(), in_contact=True, velocity_m_s=velocity)

    # Not in contact yet — check for entry.
    if velocity < state.velocity_threshold_m_s and abs(sole_height) < state.height_threshold_m:
        state.candidate_run += 1
    else:
        state.candidate_run = 0

    if state.candidate_run >= state.contact_dwell_frames:
        state.in_contact = True
        # Snap the anchor onto the floor: project the raw position onto the
        # floor plane and lift by the ankle-to-sole offset along the floor
        # normal so the ankle marker sits where it should.
        projected = floor.project(raw_position)
        anchor = projected + state.ankle_to_sole_m * floor.normal
        state.anchor = anchor
        state.candidate_run = 0
        return FootContactResult(snapped_position=anchor.copy(), in_contact=True, velocity_m_s=velocity)

    # Not in contact and not yet a candidate; just enforce that the foot
    # never goes below the floor.
    if raw_height < 0:
        snapped = raw_position - raw_height * floor.normal
        return FootContactResult(snapped_position=snapped, in_contact=False, velocity_m_s=velocity)
    return FootContactResult(snapped_position=raw_position, in_contact=False, velocity_m_s=velocity)
