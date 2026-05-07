"""CLI: auto-capture wand frames and run extrinsic calibration.

Usage::

    lislym-calib-extrinsic --intrinsic-dir ./calib --session demo \\
        --cameras "iphone14:0,ipad:1,inspiron:2"

Workflow:

1. Tool opens every camera in the list and shows a tiled live preview.
2. Press ``s`` to **start** auto-capture. Whenever every camera sees
   all three colored balls in the same frame, the tool checks whether
   the wand is at a *new region of the capture volume* (using a per-
   camera 2D centroid signature). Diverse, sharp configurations are
   added to the buffer; clusters of similar shots are not.
3. When ``--frames`` accepted frames are buffered, the tool pauses for
   a final review.
4. ``c`` runs the wand bundle adjustment and writes
   ``extrinsic_session_<session>.json`` (locked, content-hash protected).
5. ``r`` resets to keep capturing; ``q`` quits without saving.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
from lislym.tools.auto_capture import (
    AutoCaptureBuffer,
    FrameCandidate,
    laplacian_sharpness,
    points_bbox,
    pose_signature,
)


@dataclass
class WandFrameRecord:
    """One auto-captured frame's worth of per-camera 2D wand observations."""

    frame_index: int
    points_per_camera: list[np.ndarray]  # length = n_cameras, each (3, 2)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto-capture wand-based extrinsic calibration")
    parser.add_argument("--intrinsic-dir", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--cameras", required=True, help='"name1:src1,name2:src2,..."')
    parser.add_argument("--calib-dir", default="./calib")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--wand-length-mm", type=float, default=500.0)
    parser.add_argument("--wand-mid-mm", type=float, default=250.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-sharpness", type=float, default=60.0)
    parser.add_argument(
        "--min-diversity-px",
        type=float,
        default=120.0,
        help="Minimum L2 distance in the per-frame multi-camera signature",
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


def _open_capture(source: str) -> cv2.VideoCapture:
    try:
        idx = int(source)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(source)
    return cap


def _per_camera_wand_points(frame_bgr: np.ndarray) -> np.ndarray | None:
    """Detect red/green/blue spheres in the order (red, green, blue).

    Returns ``(3, 2)`` pixel array, or ``None`` if any of the three
    colors went undetected.
    """
    markers = default_markers()
    dets = detect_spheres(frame_bgr, markers)
    by_marker = {d.marker: (d.cx, d.cy) for d in dets}
    if not all(name in by_marker for name in ("hip", "ankle_left", "ankle_right")):
        return None
    return np.array(
        [
            by_marker["hip"],
            by_marker["ankle_left"],
            by_marker["ankle_right"],
        ],
        dtype=np.float64,
    )


def _draw_status(image: np.ndarray, lines: list[str]) -> np.ndarray:
    pad = np.full((24 * (len(lines) + 1), image.shape[1], 3), 30, dtype=np.uint8)
    y = 24
    for line in lines:
        cv2.putText(pad, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y += 24
    return np.vstack([pad, image])


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
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

    buf: AutoCaptureBuffer = AutoCaptureBuffer(
        target_size=args.frames,
        min_sharpness=args.min_sharpness,
        min_diversity=args.min_diversity_px,
    )
    records: list[WandFrameRecord] = []
    state = "idle"  # idle / capturing / review / done
    last_status = ""
    full_announced = False

    try:
        while True:
            grabbed: list[np.ndarray | None] = []
            for cap in captures:
                ok, frm = cap.read()
                grabbed.append(frm if ok else None)

            per_camera_points: list[np.ndarray | None] = []
            preview_panels: list[np.ndarray] = []
            for frame in grabbed:
                if frame is None:
                    preview_panels.append(np.zeros((360, 640, 3), dtype=np.uint8))
                    per_camera_points.append(None)
                    continue
                pts = _per_camera_wand_points(frame)
                preview = frame.copy()
                if pts is not None:
                    for k, color in enumerate([(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                        cv2.circle(preview, (int(pts[k, 0]), int(pts[k, 1])), 8, color, 2)
                preview_panels.append(cv2.resize(preview, (640, 360)))
                per_camera_points.append(pts)

            all_visible = all(pts is not None for pts in per_camera_points)
            if state == "capturing" and all_visible:
                # Build a single combined signature across all cameras so the
                # diversity check considers the wand's *spatial* placement
                # (every camera's pixel centroid must move).
                stacked_pts = np.concatenate(per_camera_points, axis=0)  # (3*N_cams, 2)
                sig = pose_signature(stacked_pts)
                # Sharpness from the bbox of the wand region in camera 0.
                gray0 = cv2.cvtColor(grabbed[0], cv2.COLOR_BGR2GRAY)
                bbox0 = points_bbox(per_camera_points[0])
                sharp = laplacian_sharpness(gray0, bbox0)
                cand = FrameCandidate(
                    image=np.zeros((1, 1, 3), dtype=np.uint8),  # we don't keep the image
                    signature=sig,
                    sharpness=sharp,
                    extra={"points": per_camera_points},
                )
                last_status = buf.offer(cand)
                if buf.is_full() and not full_announced:
                    state = "review"
                    full_announced = True

            grid = np.vstack(preview_panels) if preview_panels else np.zeros((360, 640, 3), np.uint8)
            instructions = {
                "idle": "[s]=start auto-capture  [q]=quit",
                "capturing": "auto-capturing — move the wand around the volume  [q]=quit",
                "review": "[c]=confirm + calibrate  [r]=reset and keep capturing  [q]=quit",
            }
            grid = _draw_status(
                grid,
                [
                    f"session={args.session}  state={state}  buffered={len(buf)}/{args.frames}",
                    f"all-cameras-see-wand={all_visible}  last={last_status or '-'}",
                    instructions[state],
                ],
            )
            cv2.imshow("extrinsic", grid)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                return 0
            if state == "idle" and key == ord("s"):
                state = "capturing"
            elif state == "review":
                if key == ord("r"):
                    state = "capturing"
                    full_announced = False
                elif key == ord("c"):
                    break
    finally:
        for cap in captures:
            cap.release()
        cv2.destroyAllWindows()

    if not buf.is_full():
        print(f"only {len(buf)}/{args.frames} frames captured — calibrating anyway")

    # Materialise WandObservation per camera.
    observations: list[list[WandObservation]] = [[] for _ in cameras]
    for frame_idx, candidate in enumerate(buf.accepted):
        per_camera_pts = candidate.extra["points"]
        for cam_idx, pts in enumerate(per_camera_pts):
            observations[cam_idx].append(
                WandObservation(
                    camera_index=cam_idx,
                    frame_index=frame_idx,
                    points=pts.astype(np.float64),
                )
            )

    wand = WandSpec(
        length_m=args.wand_length_mm / 1000.0,
        mid_offset_m=args.wand_mid_mm / 1000.0,
    )
    session = calibrate_extrinsic_wand(intrinsics, observations, wand, session_id=args.session)
    print(f"bundle adjustment RMS: {session.bundle_adjustment_rms_px:.3f} px (target ≤ 0.5)")
    out = save_extrinsics(Path(args.calib_dir), session, overwrite=args.overwrite)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
