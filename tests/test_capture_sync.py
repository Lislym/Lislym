"""Tests for the timestamp alignment logic used by :class:`MultiCameraReader`."""

from __future__ import annotations

import numpy as np

from lislym.capture.camera import Frame, align_frames


def _frame(name: str, ts: float) -> Frame:
    return Frame(camera_name=name, timestamp=ts, frame_index=0, image=np.zeros((1, 1, 3), np.uint8))


def test_align_frames_keeps_all_when_within_skew() -> None:
    frames = {
        "a": _frame("a", 100.000),
        "b": _frame("b", 100.020),
        "c": _frame("c", 100.030),
    }
    target, kept, skew = align_frames(frames, max_skew=0.040)
    assert set(kept) == {"a", "b", "c"}
    assert target == 100.020  # median of three
    assert skew <= 0.040


def test_align_frames_drops_stale_camera() -> None:
    frames = {
        "iphone": _frame("iphone", 100.000),
        "ipad": _frame("ipad", 100.005),
        "inspiron": _frame("inspiron", 100.200),  # 195 ms behind
    }
    target, kept, skew = align_frames(frames, max_skew=0.040)
    assert set(kept) == {"iphone", "ipad"}
    assert "inspiron" not in kept
    assert skew <= 0.040


def test_align_frames_empty_returns_empty() -> None:
    target, kept, skew = align_frames({}, max_skew=0.040)
    assert kept == {}
    assert skew == 0.0
    assert target > 0


def test_align_frames_two_cameras_picks_higher_stamp_as_target() -> None:
    # With even count, the upper median is chosen by ``sorted[len//2]``.
    frames = {
        "a": _frame("a", 50.000),
        "b": _frame("b", 50.010),
    }
    target, kept, _ = align_frames(frames, max_skew=0.040)
    assert target == 50.010
    assert set(kept) == {"a", "b"}
