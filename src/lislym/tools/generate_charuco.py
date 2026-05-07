"""Generate a print-ready A4 ChArUco board PNG.

Run::

    python -m lislym.tools.generate_charuco --out ./prints/charuco_a4.png

The output image has the right pixel dimensions for a 300 DPI A4 print.
Open it in any image viewer or office suite, choose "Print at 100 %"
(critical — disable any auto-fit/scale-to-page option), and verify the
square edge length with a ruler before using the board for calibration.

Defaults match what the rest of the pipeline expects:

* 6 squares × 9 squares, 30 mm square edge, 22 mm marker edge
* DICT_4X4_50 ArUco dictionary

If you change the geometry on the print, pass the same numbers to
``lislym-calib-intrinsic`` so the calibration solver knows the real
dimensions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from lislym.calibration.intrinsic import CharucoBoardSpec


# ISO A4: 210 mm × 297 mm.
A4_WIDTH_MM = 210.0
A4_HEIGHT_MM = 297.0


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="A4 ChArUco board generator")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--dpi", type=float, default=300.0, help="Render DPI; print at 100% scale")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="White border around the pattern")
    return parser


def _mm_to_px(mm: float, dpi: float) -> int:
    return int(round(mm * dpi / 25.4))


def generate_charuco_a4(
    spec: CharucoBoardSpec,
    *,
    dpi: float = 300.0,
    margin_mm: float = 10.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """Render the ChArUco board centred on an A4 canvas.

    Returns the image and a dict with the actual physical sizes used
    (so the caller can sanity-check vs. the expected millimetre values).
    """
    pattern_w_mm = spec.squares_x * spec.square_length_m * 1000.0
    pattern_h_mm = spec.squares_y * spec.square_length_m * 1000.0
    if pattern_w_mm + 2 * margin_mm > A4_WIDTH_MM:
        raise ValueError(
            f"pattern width {pattern_w_mm:.0f} mm + margins exceeds A4 width {A4_WIDTH_MM} mm"
        )
    if pattern_h_mm + 2 * margin_mm > A4_HEIGHT_MM:
        raise ValueError(
            f"pattern height {pattern_h_mm:.0f} mm + margins exceeds A4 height {A4_HEIGHT_MM} mm"
        )

    canvas_w = _mm_to_px(A4_WIDTH_MM, dpi)
    canvas_h = _mm_to_px(A4_HEIGHT_MM, dpi)
    pattern_w = _mm_to_px(pattern_w_mm, dpi)
    pattern_h = _mm_to_px(pattern_h_mm, dpi)

    board, _ = spec.make()
    board_img = board.generateImage((pattern_w, pattern_h), marginSize=0, borderBits=1)

    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    x0 = (canvas_w - pattern_w) // 2
    y0 = (canvas_h - pattern_h) // 2
    canvas[y0 : y0 + pattern_h, x0 : x0 + pattern_w] = board_img

    # Footer: print the spec so the operator can verify after printing.
    footer = (
        f"Lislym ChArUco  squares={spec.squares_x}x{spec.squares_y}  "
        f"square={spec.square_length_m * 1000:.1f}mm  marker={spec.marker_length_m * 1000:.1f}mm  "
        f"DPI={dpi:.0f}  print at 100%"
    )
    cv2.putText(
        canvas,
        footer,
        (x0, canvas_h - _mm_to_px(margin_mm * 0.5, dpi)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, dpi / 600.0),
        0,
        max(1, int(dpi / 200.0)),
        cv2.LINE_AA,
    )

    return canvas, {
        "pattern_width_mm": pattern_w_mm,
        "pattern_height_mm": pattern_h_mm,
        "square_mm": spec.square_length_m * 1000.0,
        "marker_mm": spec.marker_length_m * 1000.0,
        "canvas_px": (canvas_w, canvas_h),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    spec = CharucoBoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_mm / 1000.0,
        marker_length_m=args.marker_mm / 1000.0,
    )
    image, info = generate_charuco_a4(spec, dpi=args.dpi, margin_mm=args.margin_mm)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)

    print(f"wrote {out_path} ({info['canvas_px'][0]} × {info['canvas_px'][1]} px @ {args.dpi} DPI)")
    print(f"pattern dimensions: {info['pattern_width_mm']:.1f} × {info['pattern_height_mm']:.1f} mm")
    print(f"square edge: {info['square_mm']:.1f} mm, marker edge: {info['marker_mm']:.1f} mm")
    print("PRINT AT 100 % — disable any auto-fit/scale-to-page setting in the print dialog.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
