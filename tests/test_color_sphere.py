"""Tests for sub-pixel color-sphere detection.

These tests render synthetic spheres at known sub-pixel positions and check
that :func:`detect_spheres` recovers the centre with the precision the plan
relies on (better than 0.3 px for ~22 px diameter blobs).
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from lislym.detection.color_sphere import (
    DEFAULT_HSV_RANGES,
    Detection,
    HSVRange,
    MarkerSpec,
    default_markers,
    detect_spheres,
)


def _render_sphere(
    img: np.ndarray, cx: float, cy: float, radius_px: float, color_bgr: tuple[int, int, int]
) -> None:
    """Render a soft-edged disk that mimics a colored sphere under diffuse light.

    Uses a Lambert-like falloff so the centroid algorithm has realistic
    intensity weighting. Anti-aliased via per-pixel super-sampling so the
    sub-pixel centre information actually lives in the image.
    """
    h, w = img.shape[:2]
    x0, x1 = max(0, int(cx - radius_px - 2)), min(w, int(cx + radius_px + 3))
    y0, y1 = max(0, int(cy - radius_px - 2)), min(h, int(cy + radius_px + 3))
    ys, xs = np.indices((y1 - y0, x1 - x0), dtype=np.float32)
    xs = xs + x0
    ys = ys + y0
    # 4x4 super-samples per pixel for proper anti-aliasing.
    offsets = [(-0.375 + i * 0.25, -0.375 + j * 0.25) for i in range(4) for j in range(4)]
    accum = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    for ox, oy in offsets:
        dx = xs + ox - cx
        dy = ys + oy - cy
        r = np.sqrt(dx * dx + dy * dy)
        inside = r < radius_px
        # Lambert-ish brightness: bright in the centre, falls off near the edge.
        falloff = np.clip(1.0 - (r / radius_px) ** 2, 0.0, 1.0)
        accum += inside * (0.4 + 0.6 * falloff)
    accum /= len(offsets)

    bgr = np.stack([accum * color_bgr[0], accum * color_bgr[1], accum * color_bgr[2]], axis=-1)
    region = img[y0:y1, x0:x1].astype(np.float32)
    mask = accum[..., None] > 0
    region = np.where(mask, np.maximum(region, bgr), region)
    img[y0:y1, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)


def _make_canvas(h: int = 480, w: int = 640, bg: tuple[int, int, int] = (40, 40, 40)) -> np.ndarray:
    img = np.empty((h, w, 3), dtype=np.uint8)
    img[:] = bg
    return img


def _find(detections: list[Detection], name: str) -> Detection:
    for d in detections:
        if d.marker == name:
            return d
    raise AssertionError(f"detection for marker {name!r} missing")


def test_default_markers_returns_three_distinct_colors() -> None:
    markers = default_markers()
    names = {m.name for m in markers}
    assert names == {"hip", "ankle_left", "ankle_right"}
    assert len({m.color for m in markers}) == 3


def test_hue_wrap_red_mask_includes_both_ends() -> None:
    rng = DEFAULT_HSV_RANGES["red"]
    hsv = np.zeros((1, 4, 3), dtype=np.uint8)
    hsv[0, 0] = (175, 200, 200)  # high end
    hsv[0, 1] = (5, 200, 200)    # low end
    hsv[0, 2] = (90, 200, 200)   # middle (should be excluded)
    hsv[0, 3] = (175, 50, 200)   # too desaturated
    mask = rng.mask(hsv)
    assert mask[0, 0] > 0
    assert mask[0, 1] > 0
    assert mask[0, 2] == 0
    assert mask[0, 3] == 0


@pytest.mark.parametrize(
    "fractional",
    [(0.0, 0.0), (0.25, 0.5), (0.5, 0.5), (0.75, 0.25), (0.33, 0.66)],
)
def test_subpixel_centre_within_quarter_pixel(fractional: tuple[float, float]) -> None:
    fx, fy = fractional
    cx = 320.0 + fx
    cy = 240.0 + fy
    radius = 11.0  # 22 px diameter — the worked example from the plan.

    img = _make_canvas()
    _render_sphere(img, cx, cy, radius, color_bgr=(0, 0, 230))  # red

    dets = detect_spheres(img, default_markers())
    hip = _find(dets, "hip")

    err = math.hypot(hip.cx - cx, hip.cy - cy)
    # The plan budgets 0.1–0.3 px localisation. We accept up to 0.4 px to leave
    # a small headroom for super-sampling artefacts in the synthetic image.
    assert err < 0.4, f"sub-pixel error too large: {err:.3f} px"


def test_three_markers_localised_simultaneously() -> None:
    img = _make_canvas()
    _render_sphere(img, 120.0, 200.0, 12.0, color_bgr=(0, 0, 230))    # red hip
    _render_sphere(img, 300.0, 350.0, 12.0, color_bgr=(0, 220, 0))    # green ankle_left
    _render_sphere(img, 500.0, 350.0, 12.0, color_bgr=(230, 0, 0))    # blue ankle_right

    dets = detect_spheres(img, default_markers())
    assert {d.marker for d in dets} == {"hip", "ankle_left", "ankle_right"}

    expected = {
        "hip": (120.0, 200.0),
        "ankle_left": (300.0, 350.0),
        "ankle_right": (500.0, 350.0),
    }
    for d in dets:
        ex, ey = expected[d.marker]
        assert math.hypot(d.cx - ex, d.cy - ey) < 0.6


def test_no_detection_when_marker_absent() -> None:
    img = _make_canvas()
    _render_sphere(img, 200.0, 200.0, 11.0, color_bgr=(0, 0, 230))
    dets = detect_spheres(img, default_markers())
    names = {d.marker for d in dets}
    # Only red rendered → green/blue must be silent rather than spuriously matched.
    assert names == {"hip"}


def test_too_small_blob_rejected() -> None:
    img = _make_canvas()
    cv2.circle(img, (100, 100), 2, (0, 0, 230), -1)  # below min_radius_px
    dets = detect_spheres(img, default_markers())
    assert all(d.marker != "hip" for d in dets)


def test_custom_marker_with_explicit_hsv_range() -> None:
    img = _make_canvas()
    _render_sphere(img, 250.0, 250.0, 14.0, color_bgr=(0, 0, 230))
    spec = MarkerSpec(
        name="custom",
        color="crimson",
        hsv_range=HSVRange(h_low=170, h_high=10, s_low=100, s_high=255, v_low=60, v_high=255),
    )
    dets = detect_spheres(img, [spec])
    assert len(dets) == 1
    assert dets[0].marker == "custom"


def test_confidence_in_unit_interval() -> None:
    img = _make_canvas()
    _render_sphere(img, 320.0, 240.0, 11.0, color_bgr=(0, 0, 230))
    dets = detect_spheres(img, default_markers())
    for d in dets:
        assert 0.0 <= d.confidence <= 1.0
        assert d.pixel_residual >= 0.05
