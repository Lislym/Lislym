"""Sub-pixel color-sphere detection.

Pipeline per frame:

1. Convert BGR → HSV.
2. For each registered marker color, build a binary mask using HSV thresholds
   with hue wrap-around.
3. Morphological open/close to remove speckle noise and close small holes.
4. Find connected components and keep ones that match the expected size /
   circularity of a sphere projected at the configured working distances.
5. Refine the centre to sub-pixel precision using image moments computed on
   the mask region weighted by the V (brightness) channel — equivalent to a
   centroid of intensity over the silhouette, which is what gives the
   0.1–0.3 px accuracy quoted for circular blobs of ~20–30 px diameter.

The returned :class:`Detection` carries the centre, an estimated radius, and
a residual proxy that downstream triangulation uses as the observation
weight (Section B-3 of the plan).

This module deliberately depends only on numpy + opencv-contrib so it can run
on any OpenCV install without MediaPipe.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class HSVRange:
    """Inclusive HSV range with optional hue wrap.

    OpenCV uses H in [0, 179]. ``h_low`` may be greater than ``h_high`` to
    represent a wrap-around range (e.g. red spans 170–179 and 0–9).
    """

    h_low: int
    h_high: int
    s_low: int
    s_high: int
    v_low: int
    v_high: int

    def mask(self, hsv: np.ndarray) -> np.ndarray:
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]
        sv = cv2.bitwise_and(
            cv2.inRange(s, self.s_low, self.s_high),
            cv2.inRange(v, self.v_low, self.v_high),
        )
        h_chan = hsv[:, :, 0]
        if self.h_low <= self.h_high:
            h = cv2.inRange(h_chan, self.h_low, self.h_high)
        else:
            h = cv2.bitwise_or(
                cv2.inRange(h_chan, self.h_low, 179),
                cv2.inRange(h_chan, 0, self.h_high),
            )
        return cv2.bitwise_and(h, sv)


@dataclass(frozen=True)
class MarkerSpec:
    """A registered marker.

    Parameters
    ----------
    name:
        Stable identifier such as ``"hip_front"``, ``"ankle_left"``.
    color:
        Display label for logs (e.g. ``"red"``).
    hsv_range:
        Threshold for HSV masking.
    diameter_m:
        Physical diameter of the sphere in metres.
    """

    name: str
    color: str
    hsv_range: HSVRange
    diameter_m: float = 0.05


@dataclass(frozen=True)
class Detection:
    """A single 2D detection in image coordinates.

    Attributes
    ----------
    marker:
        Name of the matched :class:`MarkerSpec`.
    cx, cy:
        Sub-pixel centre.
    radius_px:
        Fitted radius (pixels). Useful for debug overlay and as a quick
        self-consistency check vs. expected projected size.
    confidence:
        In [0, 1]. Higher is better. Combines mask coverage, circularity, and
        moment residual; a reasonable observation weight for triangulation is
        ``confidence ** 2``.
    pixel_residual:
        Estimated 1-sigma localisation error in pixels. Inverse-variance
        weight is ``1 / pixel_residual ** 2``.
    """

    marker: str
    cx: float
    cy: float
    radius_px: float
    confidence: float
    pixel_residual: float


def _morphology(mask: np.ndarray, open_k: int = 3, close_k: int = 5) -> np.ndarray:
    if open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    if close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def _subpixel_centroid(
    value_channel: np.ndarray, mask: np.ndarray, bbox: tuple[int, int, int, int]
) -> tuple[float, float, float]:
    """Compute intensity-weighted centroid inside a bounding box.

    The mask localises which pixels are part of the marker; the V channel
    weights them so darker shadow edges contribute less than the bright
    interior. The result is a sub-pixel (cx, cy) plus a normalised residual
    (mean absolute deviation in px) that we expose as a localisation error.
    """
    x, y, w, h = bbox
    sub_mask = mask[y : y + h, x : x + w] > 0
    sub_val = value_channel[y : y + h, x : x + w].astype(np.float32)
    weights = sub_val * sub_mask
    total = float(weights.sum())
    if total <= 0.0:
        return float(x + w / 2.0), float(y + h / 2.0), float(max(w, h))

    ys, xs = np.indices((h, w), dtype=np.float32)
    cx = float((weights * xs).sum() / total) + x
    cy = float((weights * ys).sum() / total) + y

    dx = xs + x - cx
    dy = ys + y - cy
    r = np.sqrt(dx * dx + dy * dy)
    mean_r = float((weights * r).sum() / total)
    var_r = float((weights * (r - mean_r) ** 2).sum() / total)
    residual_px = float(np.sqrt(max(var_r, 1e-9)) / max(np.sqrt(total), 1.0))
    return cx, cy, residual_px


def detect_spheres(
    bgr: np.ndarray,
    markers: list[MarkerSpec],
    *,
    min_radius_px: float = 4.0,
    max_radius_px: float = 80.0,
    min_circularity: float = 0.55,
) -> list[Detection]:
    """Detect every registered marker in ``bgr``.

    Returns at most one detection per marker (the largest blob that passes
    the circularity / size gates). Markers that are not visible or fail the
    gates are simply absent from the result list.
    """
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("expected a BGR image with 3 channels")

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    out: list[Detection] = []
    for spec in markers:
        mask = spec.hsv_range.mask(hsv)
        mask = _morphology(mask)

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        best: Detection | None = None
        best_score: float = -1.0
        for label in range(1, n_labels):
            x, y, w, h, area = stats[label]
            if area < np.pi * min_radius_px ** 2:
                continue
            if area > np.pi * max_radius_px ** 2:
                continue

            equiv_radius = float(np.sqrt(area / np.pi))
            bbox_radius = 0.5 * float(max(w, h))
            circularity = equiv_radius / bbox_radius if bbox_radius > 0 else 0.0
            if circularity < min_circularity:
                continue

            component_mask = np.where(labels == label, np.uint8(255), np.uint8(0))
            cx, cy, residual = _subpixel_centroid(value, component_mask, (x, y, w, h))

            mask_fill = float(area) / float(max(w * h, 1))
            confidence = float(np.clip(circularity * mask_fill, 0.0, 1.0))
            score = circularity * np.log1p(area)

            if score > best_score:
                best_score = score
                best = Detection(
                    marker=spec.name,
                    cx=cx,
                    cy=cy,
                    radius_px=equiv_radius,
                    confidence=confidence,
                    pixel_residual=max(residual, 0.05),
                )

        if best is not None:
            out.append(best)
    return out


DEFAULT_HSV_RANGES: dict[str, HSVRange] = {
    "red":   HSVRange(h_low=170, h_high=10, s_low=120, s_high=255, v_low=80, v_high=255),
    "green": HSVRange(h_low=40,  h_high=85, s_low=100, s_high=255, v_low=60, v_high=255),
    "blue":  HSVRange(h_low=95,  h_high=130, s_low=120, s_high=255, v_low=60, v_high=255),
}


def default_markers() -> list[MarkerSpec]:
    """The 3-marker default registry: red hip, green left ankle, blue right ankle."""
    return [
        MarkerSpec("hip",         "red",   DEFAULT_HSV_RANGES["red"]),
        MarkerSpec("ankle_left",  "green", DEFAULT_HSV_RANGES["green"]),
        MarkerSpec("ankle_right", "blue",  DEFAULT_HSV_RANGES["blue"]),
    ]
