"""Runtime entry point: live tracking + OSC output.

Usage::

    lislym --intrinsic-dir ./calib --session demo --user alice \\
        --cameras "iphone14:0,ipad:1,inspiron:2" \\
        --osc-host 127.0.0.1 --osc-port 9000

Loads calibration artifacts from ``./calib`` (intrinsics per camera,
the locked extrinsic session, the personal segment lengths), opens
cameras, runs the pipeline at the cameras' shared frame rate, and
streams OSC trackers to VRChat.

This is the runtime that the operator launches once everything else is
set up. CTRL-C cleanly stops capture.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import cv2
import numpy as np

from lislym.calibration.floor import FloorPlane
from lislym.calibration.hmd_align import FrameAlignment
from lislym.calibration.personal import load_personal
from lislym.calibration.store import (
    assemble_cameras,
    load_extrinsics,
    load_intrinsics,
)
from lislym.capture.camera import CameraConfig, MultiCameraReader
from lislym.detection.color_sphere import default_markers, detect_spheres
from lislym.output.osc_trackers import OSCTrackerSender
from lislym.pipeline import FrameDetections, Pipeline


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lislym runtime")
    parser.add_argument("--intrinsic-dir", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--calib-dir", default="./calib")
    parser.add_argument(
        "--cameras",
        required=True,
        help='"name1:src1,name2:src2,..." matching the calibration session',
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--osc-host", default="127.0.0.1")
    parser.add_argument("--osc-port", type=int, default=9000)
    parser.add_argument(
        "--floor-z",
        type=float,
        default=0.0,
        help="Floor height (m) along the world up axis if no AprilTag calib was run",
    )
    return parser


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


def _to_source(s: str) -> int | str:
    try:
        return int(s)
    except ValueError:
        return s


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    intrinsic_dir = Path(args.intrinsic_dir)
    calib_dir = Path(args.calib_dir)
    camera_specs = _parse_cameras(args.cameras)
    intrinsics = {name: load_intrinsics(intrinsic_dir, name) for name, _ in camera_specs}
    extrinsics = load_extrinsics(calib_dir, args.session)
    cameras = assemble_cameras(intrinsics, extrinsics)
    segments = load_personal(calib_dir, args.user)

    pipeline = Pipeline(
        cameras=cameras,
        segments=segments,
        floor=FloorPlane(normal=np.array([0.0, 0.0, 1.0]), offset=-args.floor_z),
        hmd_alignment=FrameAlignment(yaw_radians=0.0, translation=np.zeros(3)),
        use_bundle_smoother=True,
    )
    sender = OSCTrackerSender(args.osc_host, args.osc_port)
    markers = default_markers()

    capture_configs = [
        CameraConfig(name=name, source=_to_source(src), width=args.width, height=args.height, fps=args.fps)
        for name, src in camera_specs
    ]

    stop = {"flag": False}

    def _handle(_signo, _frame) -> None:
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print(f"streaming OSC trackers to {args.osc_host}:{args.osc_port}")
    print(f"cameras: {[c.name for c in cameras]}")

    with MultiCameraReader(capture_configs) as reader:
        frame_count = 0
        last_log = time.monotonic()
        while not stop["flag"]:
            bundle = reader.read(timeout=0.5)
            if not bundle.frames:
                continue

            detections_per_camera = []
            for cam in cameras:
                frame = bundle.frames.get(cam.name)
                if frame is None:
                    detections_per_camera.append([])
                else:
                    detections_per_camera.append(detect_spheres(frame.image, markers))

            # No HMD/controller integration yet — placeholder values keep the
            # pipeline running so the operator can see hip+ankle tracking
            # before SteamVR is wired in.
            placeholder_hmd = np.array([0.0, 0.0, 1.65])
            placeholder_left = np.array([-0.3, 0.0, 1.30])
            placeholder_right = np.array([0.3, 0.0, 1.30])

            frame_dets = FrameDetections(
                timestamp=bundle.target_time,
                detections_per_camera=detections_per_camera,
                hmd_position_world=placeholder_hmd,
                hmd_yaw_radians=0.0,
                controller_left_world=placeholder_left,
                controller_right_world=placeholder_right,
            )
            out = pipeline.process_frame(frame_dets)
            sender.send(out.tracker_snapshots)
            frame_count += 1
            now = time.monotonic()
            if now - last_log > 1.0:
                fps = frame_count / (now - last_log)
                print(
                    f"fps={fps:.1f}  hip=({out.hip_world[0]:+.3f}, {out.hip_world[1]:+.3f}, {out.hip_world[2]:+.3f}) "
                    f"contact_L={out.in_contact_left} contact_R={out.in_contact_right} "
                    f"modes={out.triangulation_modes}"
                )
                frame_count = 0
                last_log = now

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
