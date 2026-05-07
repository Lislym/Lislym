"""Generate a print-ready A4 floor ArUco marker.

Run::

    python -m lislym.tools.generate_floor_aruco --out ./prints/floor_aruco_a4.png

A single ArUco marker is rendered centred on an A4 canvas. The default
marker is **150 mm** square (well above the 100 mm we recommend in the
plan, taking advantage of the spare A4 area to give the cameras a
larger target). Print at 100 % scale, tape it flat on the floor with
masking tape, and run the floor calibration via
``lislym.calibration.floor.floor_plane_from_apriltag``.

Defaults:

* ``DICT_4X4_50`` (matches the dictionary used by the rest of the project)
* marker id 0 (free to override)
* 150 mm marker side at 300 DPI
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0

# Map common dictionary names to the OpenCV constants. Using the same name
# space as ``cv2.aruco`` itself; we expose only those that fit our 4x4 / 5x5
# defaults — anything more exotic and the operator can pass --dict-id.
_DICT_BY_NAME = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A4 single-ArUco marker generator (floor target)")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--marker-id", type=int, default=0)
    parser.add_argument("--marker-mm", type=float, default=150.0)
    parser.add_argument("--dict-name", default="DICT_4X4_50", choices=list(_DICT_BY_NAME))
    parser.add_argument("--dpi", type=float, default=300.0)
    parser.add_argument("--margin-mm", type=float, default=10.0)
    return parser


def _mm_to_px(mm: float, dpi: float) -> int:
    return int(round(mm * dpi / 25.4))


def generate_floor_aruco_a4(
    *,
    marker_id: int = 0,
    marker_mm: float = 150.0,
    dict_id: int = cv2.aruco.DICT_4X4_50,
    dpi: float = 300.0,
    margin_mm: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render a single ArUco marker centred on an A4 canvas."""
    if marker_mm + 2 * margin_mm > min(A4_WIDTH_MM, A4_HEIGHT_MM):
        raise ValueError(
            f"marker {marker_mm:.0f} mm + margins {margin_mm:.0f} mm exceeds A4 short side"
        )

    canvas_w = _mm_to_px(A4_WIDTH_MM, dpi)
    canvas_h = _mm_to_px(A4_HEIGHT_MM, dpi)
    marker_px = _mm_to_px(marker_mm, dpi)

    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px, borderBits=1)

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    x0 = (canvas_w - marker_px) // 2
    y0 = (canvas_h - marker_px) // 2
    canvas[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker_img

    # Crosshair and corner ticks: makes it easy to align the marker squarely
    # to the room's reference axes when taping it down.
    tick = _mm_to_px(5.0, dpi)
    half = marker_px // 2
    cx = canvas_w // 2
    cy = canvas_h // 2
    cv2.line(canvas, (cx - half - tick, cy), (cx - half, cy), 0, max(1, int(dpi / 300.0)))
    cv2.line(canvas, (cx + half, cy), (cx + half + tick, cy), 0, max(1, int(dpi / 300.0)))
    cv2.line(canvas, (cx, cy - half - tick), (cx, cy - half), 0, max(1, int(dpi / 300.0)))
    cv2.line(canvas, (cx, cy + half), (cx, cy + half + tick), 0, max(1, int(dpi / 300.0)))

    footer = (
        f"Lislym Floor ArUco  id={marker_id}  size={marker_mm:.0f}mm  "
        f"dict=DICT_{dict_id}  DPI={dpi:.0f}  print at 100%"
    )
    cv2.putText(
        canvas,
        footer,
        (_mm_to_px(margin_mm, dpi), canvas_h - _mm_to_px(margin_mm * 0.5, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, dpi / 600.0),
        0,
        max(1, int(dpi / 200.0)),
        cv2.LINE_AA,
    )

    return canvas, {
        "marker_mm": marker_mm,
        "marker_px": marker_px,
        "canvas_px": (canvas_w, canvas_h),
        "dict_id": dict_id,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    image, info = generate_floor_aruco_a4(
        marker_id=args.marker_id,
        marker_mm=args.marker_mm,
        dict_id=_DICT_BY_NAME[args.dict_name],
        dpi=args.dpi,
        margin_mm=args.margin_mm,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)

    print(f"wrote {out_path} ({info['canvas_px'][0]} × {info['canvas_px'][1]} px @ {args.dpi} DPI)")
    print(f"marker id={args.marker_id}, side={info['marker_mm']:.0f} mm, dict={args.dict_name}")
    print("PRINT AT 100 % — disable any auto-fit/scale-to-page setting in the print dialog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
