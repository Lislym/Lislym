"""Intrinsic calibration via a ChArUco board (A4-printable).

Uses OpenCV's ChArUco detection so the marker corners give a robust
identification of which board cells are visible, enabling correct
calibration even when the board is partially occluded — important when the
user holds the A4 board by hand and only part of it falls in the camera's
field of view.

The default board geometry is sized to fit on a single A4 sheet at 96 DPI:

* Squares: 6 columns × 9 rows
* Square edge: 30 mm
* Marker edge inside each square: 22 mm (large enough to be detected at 1 m)

That yields a 180 × 270 mm pattern, comfortably inside A4's 210 × 297 mm
printable area. Override these in :class:`CharucoBoardSpec` for a custom
print.

The output of :func:`calibrate_intrinsic_from_frames` is an
:class:`~lislym.calibration.store.Intrinsics` that the store module can
serialise without further conversion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from lislym.calibration.store import Intrinsics


@dataclass(frozen=True)
class CharucoBoardSpec:
    """Geometry of the printed ChArUco board."""

    squares_x: int = 6
    squares_y: int = 9
    square_length_m: float = 0.030
    marker_length_m: float = 0.022
    aruco_dict_id: int = cv2.aruco.DICT_4X4_50

    def make(self) -> tuple[cv2.aruco.CharucoBoard, cv2.aruco.Dictionary]:
        """Return the OpenCV objects describing this board."""
        aruco_dict = cv2.aruco.getPredefinedDictionary(self.aruco_dict_id)
        board = cv2.aruco.CharucoBoard(
            (self.squares_x, self.squares_y),
            self.square_length_m,
            self.marker_length_m,
            aruco_dict,
        )
        return board, aruco_dict


@dataclass(frozen=True)
class CharucoObservation:
    """One image's worth of detected ChArUco corners."""

    corners: np.ndarray  # (N, 1, 2) float32
    ids: np.ndarray  # (N, 1) int32


def detect_charuco(
    image_bgr: np.ndarray,
    spec: CharucoBoardSpec,
    *,
    min_corners: int = 8,
) -> CharucoObservation | None:
    """Detect ChArUco corners in one frame.

    Returns ``None`` if fewer than ``min_corners`` corners are recovered;
    the calibration loop should silently skip such frames rather than
    poison the optimisation.
    """
    board, aruco_dict = spec.make()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    marker_corners, marker_ids, _ = detector.detectMarkers(gray)
    if marker_ids is None or len(marker_ids) == 0:
        return None

    refiner = cv2.aruco.CharucoDetector(board)
    ch_corners, ch_ids, _, _ = refiner.detectBoard(gray)
    if ch_ids is None or len(ch_ids) < min_corners:
        return None
    return CharucoObservation(corners=ch_corners, ids=ch_ids)


def calibrate_intrinsic_from_frames(
    camera_name: str,
    frames: list[np.ndarray],
    spec: CharucoBoardSpec,
    *,
    min_corners: int = 8,
) -> Intrinsics:
    """Run intrinsic calibration on a list of BGR frames.

    Frames that don't contain enough corners are silently skipped. The
    returned :class:`Intrinsics` carries the achieved RMS so the caller can
    abort if it's too high (the plan budgets ``< 0.5 px``).

    Raises ``RuntimeError`` if fewer than 5 usable frames remain — that's
    the lower bound for a stable intrinsic solve in OpenCV.
    """
    if not frames:
        raise RuntimeError("need at least one frame")

    board, _ = spec.make()
    image_size: tuple[int, int] | None = None
    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []

    for frame in frames:
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
        elif (frame.shape[1], frame.shape[0]) != image_size:
            raise RuntimeError("all frames must share the same resolution")

        obs = detect_charuco(frame, spec, min_corners=min_corners)
        if obs is None:
            continue
        op, ip = board.matchImagePoints(obs.corners, obs.ids)
        if op is None or ip is None or len(op) < min_corners:
            continue
        obj_points.append(op)
        img_points.append(ip)

    if len(obj_points) < 5:
        raise RuntimeError(
            f"only {len(obj_points)} usable frames — need at least 5 for a stable solve"
        )

    assert image_size is not None
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_points,
        img_points,
        image_size,
        None,
        None,
        flags=cv2.CALIB_RATIONAL_MODEL,
    )

    return Intrinsics(
        camera_name=camera_name,
        K=np.asarray(K, dtype=np.float64),
        dist=np.asarray(dist, dtype=np.float64).reshape(-1),
        image_size=image_size,
        reprojection_rms_px=float(rms),
        captured_frames=len(obj_points),
    )


def calibrate_intrinsic_from_directory(
    camera_name: str,
    directory: Path,
    spec: CharucoBoardSpec,
    *,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png"),
) -> Intrinsics:
    """Convenience: load every image in ``directory`` and calibrate."""
    paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in extensions)
    if not paths:
        raise RuntimeError(f"no calibration images found in {directory}")
    frames = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        frames.append(img)
    return calibrate_intrinsic_from_frames(camera_name, frames, spec)
