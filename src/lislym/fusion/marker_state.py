"""Per-marker state estimator with constant-acceleration EKF.

Plan §C-4 / C-5 require: when a marker drops to K=0 cameras, predict its
position from recent dynamics; when observations come back, re-associate
the right identity (color × spatial gate × kinematic plausibility) and
return without a visible jump.

This module owns three small primitives that the per-frame pipeline glues
together:

* :class:`MarkerState` — a 9-dimensional CA Kalman filter (position +
  velocity + acceleration). Constant-velocity is too tight for human
  ankles which decelerate sharply at heel-strike; constant-acceleration
  with reasonable process noise tracks both well.
* :class:`MarkerTracker` — owns one ``MarkerState`` per known marker
  identity, plus the lost-tracking timer and gating radius schedule.
* :func:`reassign_markers` — chooses the best identity assignment for a
  set of fresh detections using a Hungarian solve with cost terms for
  spatial gate, kinematic feasibility, and color match.

The EKF here is a *linear* Kalman filter — the dynamics and observation
models are linear — but we keep the "EKF" naming used in the plan to make
cross-references easier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


_STATE_DIM = 9   # [px, py, pz, vx, vy, vz, ax, ay, az]
_MEAS_DIM = 3


def _F(dt: float) -> np.ndarray:
    """Constant-acceleration transition matrix for ``dt`` seconds."""
    half = 0.5 * dt * dt
    F = np.eye(_STATE_DIM)
    F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
    F[0, 6] = half; F[1, 7] = half; F[2, 8] = half
    F[3, 6] = dt; F[4, 7] = dt; F[5, 8] = dt
    return F


def _Q(dt: float, sigma_jerk: float) -> np.ndarray:
    """Process noise covariance from a continuous white-jerk model.

    The block structure is the textbook CA discretisation. ``sigma_jerk``
    is in m/s^3; for human limbs 30 m/s^3 is a reasonable starting point.
    """
    q = sigma_jerk ** 2
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt3 * dt
    dt5 = dt4 * dt
    block = np.array(
        [
            [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
            [dt4 / 8.0,  dt3 / 3.0, dt2 / 2.0],
            [dt3 / 6.0,  dt2 / 2.0, dt],
        ]
    )
    Q = np.zeros((_STATE_DIM, _STATE_DIM))
    for axis in range(3):
        idx = (axis, axis + 3, axis + 6)
        for i, ii in enumerate(idx):
            for j, jj in enumerate(idx):
                Q[ii, jj] = block[i, j] * q
    return Q


_H = np.zeros((_MEAS_DIM, _STATE_DIM))
_H[0, 0] = 1.0; _H[1, 1] = 1.0; _H[2, 2] = 1.0


@dataclass
class MarkerState:
    """Kalman filter for one marker's 3D position.

    Attributes
    ----------
    name:
        Identity (e.g. ``"hip"``, ``"ankle_left"``).
    color:
        Color channel that uniquely identifies the marker visually. Used
        by :func:`reassign_markers` for the color-match gate.
    """

    name: str
    color: str
    x: np.ndarray = field(default_factory=lambda: np.zeros(_STATE_DIM))
    P: np.ndarray = field(default_factory=lambda: np.eye(_STATE_DIM) * 10.0)
    last_observation_time: float | None = None
    initialised: bool = False

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    def predict(self, dt: float, *, sigma_jerk: float = 30.0) -> None:
        if dt <= 0:
            return
        F = _F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + _Q(dt, sigma_jerk)

    def update(self, z: np.ndarray, R: np.ndarray, observation_time: float) -> None:
        """Apply a measurement ``z`` (3-vector world position) with covariance ``R``."""
        if not self.initialised:
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P = np.eye(_STATE_DIM)
            self.P[:3, :3] = R
            self.P[3:, 3:] *= 1.0
            self.initialised = True
            self.last_observation_time = observation_time
            return

        y = z - _H @ self.x
        S = _H @ self.P @ _H.T + R
        K = self.P @ _H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I_KH = np.eye(_STATE_DIM) - K @ _H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T
        self.last_observation_time = observation_time

    def position_uncertainty(self) -> float:
        """1-sigma position uncertainty, in metres, averaged over axes."""
        return float(np.sqrt(np.trace(self.P[:3, :3]) / 3.0))


@dataclass
class GatingSchedule:
    """How the spatial gate radius grows while a marker is unobserved.

    The radius starts at ``initial_m`` and grows linearly with the time the
    marker has been unobserved at ``growth_m_per_s``, capped at
    ``max_m``. Wide enough to forgive a quick step out of view, narrow
    enough to reject an unrelated colored object on the other side of
    the room.
    """

    initial_m: float = 0.5
    growth_m_per_s: float = 0.5
    max_m: float = 2.0

    def radius(self, seconds_since_last_obs: float) -> float:
        return float(min(self.max_m, self.initial_m + self.growth_m_per_s * max(0.0, seconds_since_last_obs)))


@dataclass
class MarkerTracker:
    """A bag of :class:`MarkerState` objects keyed by name."""

    states: dict[str, MarkerState] = field(default_factory=dict)
    lost_threshold_s: float = 0.5
    gating: GatingSchedule = field(default_factory=GatingSchedule)

    def register(self, name: str, color: str) -> None:
        if name in self.states:
            return
        self.states[name] = MarkerState(name=name, color=color)

    def predict_all(self, t_now: float, dt: float, sigma_jerk: float = 30.0) -> None:
        for s in self.states.values():
            s.predict(dt, sigma_jerk=sigma_jerk)

    def update(self, name: str, position: np.ndarray, covariance: np.ndarray, t: float) -> None:
        self.states[name].update(position, covariance, t)

    def is_lost(self, name: str, t_now: float) -> bool:
        s = self.states[name]
        if not s.initialised or s.last_observation_time is None:
            return True
        return (t_now - s.last_observation_time) > self.lost_threshold_s

    def gate_radius(self, name: str, t_now: float) -> float:
        s = self.states[name]
        last = s.last_observation_time if s.last_observation_time is not None else t_now
        return self.gating.radius(t_now - last)


@dataclass(frozen=True)
class CandidateDetection:
    """A 3D detection that has not yet been matched to an identity."""

    position: np.ndarray
    color: str


def reassign_markers(
    tracker: MarkerTracker,
    detections: list[CandidateDetection],
    *,
    bone_constraints: dict[tuple[str, str], tuple[float, float]] | None = None,
    t_now: float = 0.0,
) -> dict[str, int]:
    """Assign each tracked marker name to at most one candidate detection.

    Parameters
    ----------
    tracker:
        Current state of every known marker.
    detections:
        Newly produced 3D candidates from triangulation; identity is
        unknown — only the dominant color is.
    bone_constraints:
        Optional ``(name_a, name_b) → (min_m, max_m)`` distances. When two
        names are both being assigned in the same call, configurations
        that violate these distance bounds are penalised. The most useful
        constraints in our 3-marker setup are
        ``("hip", "ankle_left")`` and ``("hip", "ankle_right")``.
    t_now:
        Current time, used to compute the per-marker gating radius.

    Returns
    -------
    Dict mapping marker name → index in ``detections`` for the chosen pair.
    Markers that find no acceptable candidate are absent from the dict.
    """
    if not detections or not tracker.states:
        return {}

    names = list(tracker.states.keys())
    n_markers = len(names)
    n_dets = len(detections)
    # Cost matrix; we minimise. ``np.inf`` rules out an assignment.
    INF = 1e9
    cost = np.full((n_markers, n_dets), INF, dtype=np.float64)

    for i, name in enumerate(names):
        state = tracker.states[name]
        gate = tracker.gate_radius(name, t_now)
        predicted = state.position if state.initialised else None
        for j, det in enumerate(detections):
            if det.color != state.color:
                continue
            if predicted is not None:
                dist = float(np.linalg.norm(det.position - predicted))
                if dist > gate:
                    continue
                cost[i, j] = dist
            else:
                # Unseen so far: any color-matched detection is a candidate.
                cost[i, j] = 0.0

    if bone_constraints:
        # Penalise pairs that violate kinematic distance bounds.
        for (a, b), (lo, hi) in bone_constraints.items():
            if a not in names or b not in names:
                continue
            ia, ib = names.index(a), names.index(b)
            for ja in range(n_dets):
                if cost[ia, ja] >= INF:
                    continue
                for jb in range(n_dets):
                    if jb == ja or cost[ib, jb] >= INF:
                        continue
                    d = float(np.linalg.norm(detections[ja].position - detections[jb].position))
                    if d < lo or d > hi:
                        cost[ia, ja] += 1e3  # heavy but finite penalty
                        cost[ib, jb] += 1e3

    # Hungarian assignment via scipy if available, else a tiny exact solver
    # for the up-to-3 markers we have in practice.
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        out: dict[str, int] = {}
        for i, j in zip(rows.tolist(), cols.tolist(), strict=True):
            if cost[i, j] >= INF:
                continue
            out[names[i]] = j
        return out
    except ImportError:  # pragma: no cover — scipy is a hard dep
        return _brute_force_assignment(cost, names, INF)


def _brute_force_assignment(
    cost: np.ndarray, names: list[str], inf: float
) -> dict[str, int]:
    from itertools import permutations

    n_markers, n_dets = cost.shape
    best_total = float("inf")
    best: dict[str, int] = {}
    indices: Iterable[tuple[int, ...]]
    if n_dets >= n_markers:
        indices = permutations(range(n_dets), n_markers)
    else:
        indices = permutations(range(n_dets) + (-1,) * (n_markers - n_dets), n_markers)
    for perm in indices:
        total = 0.0
        ok = True
        for i, j in enumerate(perm):
            if j < 0 or cost[i, j] >= inf:
                ok = False
                break
            total += float(cost[i, j])
        if ok and total < best_total:
            best_total = total
            best = {names[i]: int(perm[i]) for i in range(n_markers)}
    return best
