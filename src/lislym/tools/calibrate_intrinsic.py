"""CLI: capture ChArUco frames from a camera and run intrinsic calibration.

Usage::

    lislym-calib-intrinsic --camera 0 --name iphone14 --frames 80 \\
        --calib-dir ./calib

The tool opens the requested camera (an integer device index, a file
path, or a stream URL), shows a live preview annotated with detected
ChArUco corners, and lets the operator save the current frame to the
calibration buffer by pressing space. ``q`` quits, ``c`` runs
calibration on the buffered frames and writes the result to
``intrinsic_<name>.json``.

Designed to run on the development laptop; the user holds the A4
ChArUco board in front of each camera in turn and collects 60–100
diverse views.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from lislym.calibration.intrinsic import (
    CharucoBoardSpec,
    calibrate_intrinsic_from_frames,
    detect_charuco,
)
from lislym.calibration.store import save_intrinsics


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Intrinsic ChArUco calibration")
    parser.add_argument("--camera", required=True, help="Device index or path/URL")
    parser.add_argument("--name", required=True, help="Camera name (e.g. iphone14)")
    parser.add_argument("--calib-dir", default="./calib", help="Where to write the JSON")
    parser.add_argument("--frames", type=int, default=80, help="Target number of saved frames")
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser


def _open_capture(source: str, width: int, height: int) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    spec = CharucoBoardSpec(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_mm / 1000.0,
        marker_length_m=args.marker_mm / 1000.0,
    )

    cap = _open_capture(args.camera, args.width, args.height)
    if not cap.isOpened():
        print(f"could not open camera {args.camera!r}")
        return 1

    saved_frames: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            obs = detect_charuco(frame, spec, min_corners=8)
            preview = frame.copy()
            if obs is not None:
                cv2.aruco.drawDetectedCornersCharuco(preview, obs.corners, obs.ids)
            cv2.putText(
                preview,
                f"saved {len(saved_frames)}/{args.frames}  [SPACE]=save  [c]=calibrate  [q]=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("intrinsic", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if key == ord(" ") and obs is not None:
                saved_frames.append(frame.copy())
                print(f"frame {len(saved_frames)} saved")
            if key == ord("c") and len(saved_frames) >= 10:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if len(saved_frames) < 10:
        print("too few frames captured")
        return 1
    intrinsics = calibrate_intrinsic_from_frames(args.name, saved_frames, spec)
    print(f"intrinsic RMS: {intrinsics.reprojection_rms_px:.3f} px")
    print(f"image size: {intrinsics.image_size}")
    print(f"K =\n{intrinsics.K}")
    print(f"dist = {intrinsics.dist}")
    out_path = save_intrinsics(Path(args.calib_dir), intrinsics)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
