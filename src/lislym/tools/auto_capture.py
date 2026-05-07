"""Frame-quality scoring for auto-capture calibration tools.

Both the intrinsic (ChArUco) and extrinsic (wand) tools follow the same
pattern: keep a sliding pool of accepted frames, score every incoming
detection, and auto-replace the lowest-quality frame when a better
diverse candidate arrives. This module owns the scoring helpers so the
two tools share one implementation.

Two quality dimensions matter:

* **Sharpness** — variance of the Laplacian on the detected region.
  Motion-blurred frames have low values and produce poor calibration.
* **Diversity** — how far the new frame's geometric signature is from
  the closest already-saved frame. Without diversity, calibrating on
  100 nearly-identical frames does not constrain distortion or focal
  length any better than calibrating on 5.

The signature is a small numpy vector built from on-image observables
(centroid, bounding-box size). It does not require the intrinsics
(important for the intrinsic step where the intrinsics don't exist
yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


def laplacian_sharpness(gray_image: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> float:
    """Variance-of-Laplacian sharpness, optionally on a bbox-cropped patch."""
    if bbox is not None:
        x, y, w, h = bbox
        x = max(x, 0)
        y = max(y, 0)
        w = min(w, gray_image.shape[1] - x)
        h = min(h, gray_image.shape[0] - y)
        if w <= 0 or h <= 0:
            return 0.0
        gray_image = gray_image[y : y + h, x : x + w]
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())


def points_bbox(points_xy: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box ``(x, y, w, h)`` for ``(N, 2)`` pixel points."""
    pts = points_xy.reshape(-1, 2)
    x_min = int(np.floor(pts[:, 0].min()))
    y_min = int(np.floor(pts[:, 1].min()))
    x_max = int(np.ceil(pts[:, 0].max()))
    y_max = int(np.ceil(pts[:, 1].max()))
    return x_min, y_min, max(x_max - x_min, 1), max(y_max - y_min, 1)


def pose_signature(points_xy: np.ndarray) -> np.ndarray:
    """A 5-D descriptor capturing where on the image and at what apparent
    scale / aspect a marker pattern was observed.

    Components: ``[cx, cy, bbox_w, bbox_h, aspect]``. Distance in this
    space is a cheap proxy for "did the user actually move the board
    between these two frames" without needing intrinsics.
    """
    pts = points_xy.reshape(-1, 2).astype(np.float64)
    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    spread = pts.max(axis=0) - pts.min(axis=0)
    w = float(spread[0])
    h = float(spread[1])
    aspect = w / max(h, 1.0)
    return np.array([cx, cy, w, h, aspect * 100.0])  # scale aspect so it has px-comparable magnitude


def diversity_distance(
    candidate: np.ndarray, accepted: list[np.ndarray]
) -> float:
    """Distance from ``candidate`` to its nearest neighbour in ``accepted``.

    Returns ``+inf`` when there are no accepted frames yet, so the very
    first observation always wins.
    """
    if not accepted:
        return float("inf")
    diffs = np.stack(accepted) - candidate[None, :]
    return float(np.min(np.linalg.norm(diffs, axis=1)))


@dataclass
class FrameCandidate:
    """One scored, fully-detected frame waiting to be accepted or replaced."""

    image: np.ndarray
    signature: np.ndarray
    sharpness: float
    extra: dict = field(default_factory=dict)

    def quality(self) -> float:
        """Combined score used to pick which accepted frame to evict.

        Sharpness alone is a sensible fallback; diversity is checked
        separately at acceptance time.
        """
        return self.sharpness


@dataclass
class AutoCaptureBuffer:
    """Bounded buffer of best-quality, diverse frames.

    Acceptance policy:

    1. Reject if sharpness < ``min_sharpness``.
    2. If the buffer isn't full *and* the candidate is far enough from
       every accepted frame (``signature`` distance ≥ ``min_diversity``),
       accept it.
    3. If the buffer is full but the candidate beats the worst-scoring
       frame on quality *and* still satisfies diversity against the
       remaining set, replace the worst.
    """

    target_size: int
    min_sharpness: float = 30.0
    min_diversity: float = 60.0  # pixels in the 5-D signature space
    accepted: list[FrameCandidate] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.accepted)

    def is_full(self) -> bool:
        return len(self.accepted) >= self.target_size

    def offer(self, candidate: FrameCandidate) -> str:
        """Try to insert ``candidate``. Returns one of:

        * ``"accepted"`` — newly added to the buffer.
        * ``"replaced"`` — buffer was full; a worse frame was evicted.
        * ``"rejected_sharpness"`` — failed the sharpness threshold.
        * ``"rejected_diversity"`` — too similar to an existing frame.
        """
        if candidate.sharpness < self.min_sharpness:
            return "rejected_sharpness"

        signatures = [c.signature for c in self.accepted]
        diversity = diversity_distance(candidate.signature, signatures)
        if diversity < self.min_diversity:
            return "rejected_diversity"

        if not self.is_full():
            self.accepted.append(candidate)
            return "accepted"

        # Buffer full: replace the lowest-quality frame if the candidate
        # would still keep diversity intact after the swap.
        worst_idx = int(np.argmin([c.quality() for c in self.accepted]))
        if candidate.quality() <= self.accepted[worst_idx].quality():
            return "rejected_diversity"

        # Re-check diversity excluding the candidate-replaced slot.
        remaining = [c.signature for i, c in enumerate(self.accepted) if i != worst_idx]
        if diversity_distance(candidate.signature, remaining) < self.min_diversity:
            return "rejected_diversity"

        self.accepted[worst_idx] = candidate
        return "replaced"
