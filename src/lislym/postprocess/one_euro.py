"""One Euro Filter — speed-adaptive low-pass for low-latency output.

Reference: Casiez et al., "1€ Filter: A Simple Speed-Based Low-Pass
Filter for Noisy Input in Interactive Systems" (CHI 2012). The cutoff
frequency adapts to the signal's velocity: when the value is moving
fast, cutoff is high (low latency, more noise tolerated); when nearly
still, cutoff is low (max smoothing).

Plan §E-1 specifies that this filter applies to MediaPipe-derived joints
(knee, elbow, upper body) but *not* to the marker-derived hip and
ankle positions, which are already trustworthy by the time they reach
the IK and would lose their hard-edge stability if filtered again.

Vectorised over the components of the input value, so a single
:class:`OneEuroFilter` works for either a scalar or a 3-vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OneEuroFilter:
    """A speed-adaptive low-pass filter.

    Attributes
    ----------
    min_cutoff:
        Cutoff frequency at zero velocity, in Hz. Smaller values = more
        smoothing at rest. 1.0 Hz is a sensible default for body tracking
        at 60 Hz update rate.
    beta:
        Speed coefficient. Higher values reduce lag during fast motion
        but pass more noise. 0.0 reproduces a plain exponential filter.
    d_cutoff:
        Cutoff for the velocity estimator itself. 1.0 Hz works for most
        signals.
    """

    min_cutoff: float = 1.0
    beta: float = 0.05
    d_cutoff: float = 1.0
    _x_prev: np.ndarray | None = None
    _dx_prev: np.ndarray | None = None
    _t_prev: float | None = None

    def reset(self) -> None:
        self._x_prev = None
        self._dx_prev = None
        self._t_prev = None

    def filter(self, x: np.ndarray, t: float) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        if self._t_prev is None:
            self._x_prev = x_arr.copy()
            self._dx_prev = np.zeros_like(x_arr)
            self._t_prev = t
            return x_arr.copy()

        dt = max(t - self._t_prev, 1e-6)
        # Estimate the derivative and filter it.
        dx = (x_arr - self._x_prev) / dt
        a_d = _alpha(dt, self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev

        # Adapt cutoff based on derivative magnitude.
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(dx_hat))
        a = _alpha(dt, cutoff)
        x_hat = a * x_arr + (1.0 - a) * self._x_prev

        self._x_prev = x_hat
        self._dx_prev = dx_hat
        self._t_prev = t
        return x_hat.copy()


def _alpha(dt: float, cutoff: float) -> float:
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return float(1.0 / (1.0 + tau / dt))
