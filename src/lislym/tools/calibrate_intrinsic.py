"""CLI: auto-capture ChArUco frames and run intrinsic calibration.

Usage::

    lislym-calib-intrinsic --camera 0 --name iphone14 --frames 80 \\
        --calib-dir ./calib

Workflow:

1. The tool opens the camera and shows a live preview that highlights
   detected ChArUco corners.
2. Press ``s`` to **start** auto-capture. The buffer fills automatically:
   the tool keeps the sharpest frames whose poses are spread across the
   image — moving the board around, tilting it, getting close and far
   are all rewarded; holding it still in one place is not.
3. When ``--frames`` accepted frames are buffered, the tool pauses for
   a final review (press ``c`` to confirm, or ``r`` to reset and keep
   capturing).
4. ``c`` (confirm) runs ``cv2.calibrateCamera`` on the buffer and writes
   ``intrinsic_<name>.json``.
5. ``q`` quits at any time without saving.

The auto-capture mode means the operator never has to time button
presses — they just hold the board and rotate it while the tool picks
the best frames in the background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from lislym.calibration.intrinsic import (
    CharucoBoardSpec,
    calibrate_intrinsic_from_frames,
    detect_charuco,
)
from lislym.calibration.store import save_intrinsics
from lislym.tools.auto_capture import (
    AutoCaptureBuffer,
    FrameCandidate,
    laplacian_sharpness,
    points_bbox,
    pose_signature,
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-capture intrinsic ChArUco calibration")
    parser.add_argument("--camera", required=True, help="Device index or path/URL")
    parser.add_argument("--name", required=True, help="Camera name (e.g. iphone14)")
    parser.add_argument("--calib-dir", default="./calib", help="Where to write the JSON")
    parser.add_argument("--frames", type=int, default=80, help="Target frames in buffer")
    parser.add_argument("--squares-x", type=int, default=6)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--min-sharpness",
        type=float,
        default=80.0,
        help="Reject frames whose Laplacian variance falls below this (motion blur guard)",
    )
    parser.add_argument(
        "--min-diversity-px",
        type=float,
        default=80.0,
        help="Minimum L2 distance in the 5-D pose signature for a frame to count as new",
    )
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


def _draw_status(image: np.ndarray, lines: list[str]) -> None:
    y = 30
    for line in lines:
        cv2.putText(
            image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        y += 24


def _coverage_heatmap(image: np.ndarray, accepted_signatures: list[np.ndarray]) -> None:
    """Sketch where on the image the accepted frames' boards appeared."""
    for sig in accepted_signatures:
        cx, cy, _, _, _ = sig
        cv2.circle(image, (int(cx), int(cy)), 4, (255, 200, 0), -1)


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

    buf = AutoCaptureBuffer(
        target_size=args.frames,
        min_sharpness=args.min_sharpness,
        min_diversity=args.min_diversity_px,
    )
    state = "idle"   # idle → capturing → review → done
    last_offer_status: str = ""
    last_full_announced = False

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            obs = detect_charuco(frame, spec, min_corners=8)
            preview = frame.copy()
            if obs is not None:
                cv2.aruco.drawDetectedCornersCharuco(preview, obs.corners, obs.ids)

            if state == "capturing" and obs is not None:
                bbox = points_bbox(obs.corners)
                sharp = laplacian_sharpness(gray, bbox)
                sig = pose_signature(obs.corners)
                cand = FrameCandidate(image=frame.copy(), signature=sig, sharpness=sharp)
                last_offer_status = buf.offer(cand)
                if buf.is_full() and not last_full_announced:
                    state = "review"
                    last_full_announced = True

            instructions = {
                "idle": "[s]=start auto-capture  [q]=quit",
                "capturing": "auto-capturing… move the board around  [q]=quit",
                "review": "[c]=confirm + calibrate  [r]=reset and keep capturing  [q]=quit",
            }
            _draw_status(
                preview,
                [
                    f"camera={args.name}  state={state}  buffered={len(buf)}/{args.frames}",
                    f"last={last_offer_status or '-'}  sharpness>={args.min_sharpness:.0f}",
                    instructions[state],
                ],
            )
            _coverage_heatmap(preview, [c.signature for c in buf.accepted])
            cv2.imshow("intrinsic", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if state == "idle" and key == ord("s"):
                state = "capturing"
            elif state == "review":
                if key == ord("r"):
                    state = "capturing"
                    last_full_announced = False
                elif key == ord("c"):
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not buf.is_full():
        print(f"only {len(buf)}/{args.frames} frames captured — calibrating anyway")

    frames = [c.image for c in buf.accepted]
    intrinsics = calibrate_intrinsic_from_frames(args.name, frames, spec)
    print(f"intrinsic RMS: {intrinsics.reprojection_rms_px:.3f} px (target ≤ 0.5)")
    print(f"image size: {intrinsics.image_size}")
    print(f"K =\n{intrinsics.K}")
    print(f"dist = {intrinsics.dist}")
    out_path = save_intrinsics(Path(args.calib_dir), intrinsics)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
