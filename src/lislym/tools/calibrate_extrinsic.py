"""CLI: collect synchronised wand frames from N cameras and run extrinsic calibration.

Usage::

    lislym-calib-extrinsic --intrinsic-dir ./calib --session demo \\
        --cameras "iphone14:0,ipad:1,inspiron:2"

The colon-separated camera list maps a name to a capture source. The
tool reads from all cameras in parallel, detects three colored balls
(red/green/blue, by default) per frame, and waits for the user to press
``space`` to capture a frame. Once enough frames are accumulated
(``--frames``, default 60), it runs the wand bundle adjustment and
writes ``extrinsic_session_<session>.json``.

The tool needs at least two cameras and at least 6 frames where every
camera sees all three balls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from lislym.calibration.extrinsic import (
    WandObservation,
    WandSpec,
    calibrate_extrinsic_wand,
)
from lislym.calibration.store import (
    load_intrinsics,
    save_extrinsics,
)
from lislym.detection.color_sphere import default_markers, detect_spheres


def _parse_cameras(spec: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise SystemExit(f"camera spec must be name:source, got {token!r}")
        name, source = token.split(":", 1)
        out.append((name.strip(), source.strip()))
    return out


def _open_capture(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    return cap


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wand-based extrinsic calibration")
    parser.add_argument("--intrinsic-dir", required=True)
    parser.add_argument("--session", required=True, help="Session id for the output file")
    parser.add_argument("--cameras", required=True, help='"name1:src1,name2:src2,..."')
    parser.add_argument("--calib-dir", default="./calib", help="Where to write the extrinsic")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--wand-length-mm", type=float, default=500.0)
    parser.add_argument("--wand-mid-mm", type=float, default=250.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    cameras = _parse_cameras(args.cameras)
    if len(cameras) < 2:
        raise SystemExit("need at least two cameras")

    intrinsic_dir = Path(args.intrinsic_dir)
    intrinsics = [load_intrinsics(intrinsic_dir, name) for name, _ in cameras]

    captures = [_open_capture(src) for _, src in cameras]
    if not all(cap.isOpened() for cap in captures):
        for c in captures:
            c.release()
        raise SystemExit("at least one camera failed to open")

    markers = default_markers()
    observations: list[list[WandObservation]] = [[] for _ in cameras]
    frame_index = 0

    try:
        while frame_index < args.frames:
            preview_panels: list[np.ndarray] = []
            current_per_cam: list[np.ndarray | None] = [None] * len(cameras)
            for cam_idx, cap in enumerate(captures):
                ok, frame = cap.read()
                if not ok or frame is None:
                    preview_panels.append(np.zeros((360, 640, 3), dtype=np.uint8))
                    continue
                dets = detect_spheres(frame, markers)
                points = np.full((3, 2), np.nan)
                ordering = {"hip": 0, "ankle_left": 1, "ankle_right": 2}
                for d in dets:
                    if d.marker in ordering:
                        points[ordering[d.marker]] = (d.cx, d.cy)
                        cv2.circle(frame, (int(d.cx), int(d.cy)), 8, (0, 255, 255), 2)
                current_per_cam[cam_idx] = points
                preview_panels.append(cv2.resize(frame, (640, 360)))

            grid = np.vstack(preview_panels) if preview_panels else np.zeros((360, 640, 3), np.uint8)
            cv2.putText(
                grid,
                f"saved {frame_index}/{args.frames}  [SPACE]=save  [q]=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imshow("extrinsic", grid)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                # Only save if every camera detected all three balls.
                if any(p is None or np.isnan(p).any() for p in current_per_cam):
                    print("frame skipped: some camera missed at least one ball")
                    continue
                for cam_idx, points in enumerate(current_per_cam):
                    observations[cam_idx].append(
                        WandObservation(
                            camera_index=cam_idx,
                            frame_index=frame_index,
                            points=points,
                        )
                    )
                frame_index += 1
                print(f"frame {frame_index} saved")
    finally:
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()

    wand = WandSpec(
        length_m=args.wand_length_mm / 1000.0,
        mid_offset_m=args.wand_mid_mm / 1000.0,
    )
    session = calibrate_extrinsic_wand(
        intrinsics, observations, wand, session_id=args.session
    )
    print(f"bundle adjustment RMS: {session.bundle_adjustment_rms_px:.3f} px")
    out = save_extrinsics(Path(args.calib_dir), session, overwrite=args.overwrite)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
