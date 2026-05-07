"""Tests for ChArUco / floor-ArUco print generators.

We don't print anything — just verify that the generated images have the
correct pixel dimensions for the requested DPI, that the embedded
markers can be detected back by OpenCV (round-trip), and that the
boards match the spec the calibration solver expects.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from lislym.calibration.intrinsic import CharucoBoardSpec, detect_charuco
from lislym.tools.generate_charuco import generate_charuco_a4
from lislym.tools.generate_floor_aruco import generate_floor_aruco_a4


def test_charuco_a4_has_correct_pixel_size_for_300_dpi() -> None:
    spec = CharucoBoardSpec()
    image, info = generate_charuco_a4(spec, dpi=300.0)
    # A4 at 300 DPI: 210 mm × 297 mm → 2480 × 3508 px (rounded).
    assert info["canvas_px"] == (2480, 3508)
    assert image.shape == (3508, 2480)


def test_charuco_a4_pattern_dimensions_match_spec() -> None:
    spec = CharucoBoardSpec(squares_x=6, squares_y=9, square_length_m=0.030, marker_length_m=0.022)
    _, info = generate_charuco_a4(spec, dpi=300.0)
    assert info["pattern_width_mm"] == pytest.approx(180.0)
    assert info["pattern_height_mm"] == pytest.approx(270.0)


def test_charuco_a4_round_trip_detection() -> None:
    """OpenCV must be able to detect the same board it generated."""
    spec = CharucoBoardSpec()
    gray, _ = generate_charuco_a4(spec, dpi=300.0)
    # detect_charuco expects BGR; lift to 3 channels.
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    obs = detect_charuco(bgr, spec, min_corners=8)
    assert obs is not None
    # We expect a near-complete corner set since the print is pristine.
    assert len(obs.ids) >= (spec.squares_x - 1) * (spec.squares_y - 1) * 0.8


def test_charuco_a4_rejects_pattern_larger_than_paper() -> None:
    # 30 mm squares × 8 across = 240 mm > 210 mm A4 width.
    spec = CharucoBoardSpec(squares_x=8, squares_y=10, square_length_m=0.030, marker_length_m=0.022)
    with pytest.raises(ValueError):
        generate_charuco_a4(spec, dpi=300.0)


def test_floor_aruco_a4_round_trip_detection() -> None:
    image, info = generate_floor_aruco_a4(marker_id=0, marker_mm=150.0, dpi=300.0)
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    aruco_dict = cv2.aruco.getPredefinedDictionary(info["dict_id"])
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(bgr)
    assert ids is not None
    assert 0 in ids.flatten().tolist()
    assert len(corners) == 1


def test_floor_aruco_a4_rejects_oversized_marker() -> None:
    with pytest.raises(ValueError):
        generate_floor_aruco_a4(marker_mm=220.0, dpi=300.0)


def test_charuco_main_writes_file(tmp_path: Path) -> None:
    from lislym.tools.generate_charuco import main as charuco_main

    out = tmp_path / "board.png"
    rc = charuco_main(["--out", str(out), "--dpi", "150"])
    assert rc == 0
    assert out.exists()
    image = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    # 150 DPI → A4 = 1240 × 1754 px.
    assert image.shape == (1754, 1240)


def test_floor_aruco_main_writes_file(tmp_path: Path) -> None:
    from lislym.tools.generate_floor_aruco import main as floor_main

    out = tmp_path / "floor.png"
    rc = floor_main(["--out", str(out), "--dpi", "150", "--marker-id", "7"])
    assert rc == 0
    assert out.exists()
    image = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert image.shape == (1754, 1240)
