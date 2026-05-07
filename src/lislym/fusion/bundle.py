"""Sliding-window bundle adjustment for jitter reduction.

Plan §C-2. Triangulation alone gives crisp 3D positions but they jitter
frame-to-frame because every observation is independent. Sliding-window
bundle adjustment minimises the joint cost over the last ``W`` frames:

* per-frame re-projection error against every camera that saw the marker
* a smoothness penalty that discourages large frame-to-frame jumps in
  position and velocity

The window slides one frame at a time: when frame ``k+1`` arrives, we
push out the oldest, append the newest, and re-solve. This is much
cheaper than a global BA over the whole session and runs comfortably at
60 Hz for a window of 18 frames (= 0.3 s) and a single 3D point.

The smoother is implemented for one marker at a time. Run it three
times (hip, ankle_left, ankle_right) per frame.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from lislym.calibration.camera_model import CameraModel


@dataclass
class FrameObservations:
    """One frame's worth of 2D observations for a single marker."""

    timestamp: float
    pixel_observations: dict[int, tuple[float, float]]  # camera_index → (px, py)
    pixel_residuals: dict[int, float]  # camera_index → 1-sigma px


@dataclass
class SlidingWindowSmoother:
    """Sliding-window BA over the last ``window_size`` frames.

    The optimisation variables are the 3D positions ``P_0 ... P_{W-1}``,
    one per frame. Constants are the camera projection matrices and the
    pixel observations; their re-projection error is the data term. The
    second-order finite difference of consecutive 3D positions is the
    smoothness term, weighted by ``smoothness_weight``.
    """

    cameras: list[CameraModel]
    window_size: int = 18
    smoothness_weight: float = 1.0
    max_iterations: int = 20
    _frames: deque[FrameObservations] = None  # set in __post_init__
    _last_estimate: np.ndarray | None = None  # (W, 3) seed for next solve

    def __post_init__(self) -> None:
        self._frames = deque(maxlen=self.window_size)

    def push(self, obs: FrameObservations) -> None:
        self._frames.append(obs)

    def _project(self, cam: CameraModel, point: np.ndarray) -> np.ndarray:
        Xc = cam.R @ point + cam.t
        if Xc[2] <= 1e-9:
            return np.array([np.nan, np.nan])
        p = cam.K @ Xc
        return p[:2] / p[2]

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        n = len(self._frames)
        positions = x.reshape(n, 3)
        out: list[float] = []
        for i, obs in enumerate(self._frames):
            for cam_idx, pix in obs.pixel_observations.items():
                cam = self.cameras[cam_idx]
                proj = self._project(cam, positions[i])
                if np.isnan(proj).any():
                    out.extend([1e3, 1e3])
                    continue
                w = 1.0 / max(obs.pixel_residuals.get(cam_idx, 1.0), 1e-3)
                out.extend((w * (proj - np.array(pix))).tolist())
        # Smoothness: penalise second differences in 3D position.
        for i in range(1, n - 1):
            diff = positions[i - 1] - 2.0 * positions[i] + positions[i + 1]
            out.extend((self.smoothness_weight * diff).tolist())
        return np.asarray(out, dtype=np.float64)

    def solve(self, seed_positions: list[np.ndarray] | None = None) -> np.ndarray | None:
        """Run the BA over the current window.

        Returns the smoothed 3D position for every frame in the window
        (shape ``(W, 3)``), or ``None`` if there are no frames yet.
        """
        n = len(self._frames)
        if n == 0:
            return None
        if seed_positions is None or len(seed_positions) != n:
            # Use the previous solve's last value, then fill with the most
            # recent observation's per-camera centroid as a rough seed.
            if self._last_estimate is not None and len(self._last_estimate) >= 1:
                last = self._last_estimate[-1]
            else:
                last = np.zeros(3)
            seed_positions = [last for _ in range(n)]

        x0 = np.stack(seed_positions).reshape(-1)
        result = least_squares(
            self._residuals,
            x0,
            method="lm",
            max_nfev=self.max_iterations * len(x0) + 10,
            xtol=1e-9,
            ftol=1e-9,
        )
        positions = result.x.reshape(n, 3)
        self._last_estimate = positions.copy()
        return positions

    def latest_smoothed(self) -> np.ndarray | None:
        """Return the most recent frame's smoothed position, if any."""
        if self._last_estimate is None or len(self._last_estimate) == 0:
            return None
        return self._last_estimate[-1].copy()
