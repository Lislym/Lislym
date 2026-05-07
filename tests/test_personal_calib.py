"""Tests for personal calibration (T-pose bone-length estimation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lislym.calibration.personal import (
    BodySegments,
    TPoseSamples,
    load_personal,
    save_personal,
)


def _ideal_tpose_frame(rng: np.random.Generator, jitter: float = 0.0) -> dict[str, np.ndarray]:
    """Generate one synchronised T-pose frame for a 1.75 m tall user."""
    base = {
        "hip": np.array([0.0, 0.0, 0.95]),
        "ankle_left": np.array([-0.10, 0.0, 0.06]),
        "ankle_right": np.array([0.10, 0.0, 0.06]),
        "hmd": np.array([0.0, -0.05, 1.65]),
        # T-pose: arms straight out at shoulder height (z ≈ 1.40), 0.65 m away.
        "wrist_left": np.array([-0.65, 0.0, 1.40]),
        "wrist_right": np.array([0.65, 0.0, 1.40]),
    }
    if jitter > 0:
        base = {k: v + rng.normal(0, jitter, 3) for k, v in base.items()}
    return base


def test_solve_returns_reasonable_segment_lengths() -> None:
    rng = np.random.default_rng(0)
    samples = TPoseSamples()
    for _ in range(30):
        frame = _ideal_tpose_frame(rng, jitter=0.001)
        samples.add_frame(**frame)

    seg = samples.solve(head_offset_m=0.10, ankle_to_sole_m=0.06)
    # Hip (0.95) → ankle (0.06): 0.89 m, plus the small lateral offset.
    assert seg.leg_length == pytest.approx(np.hypot(0.10, 0.89), abs=0.01)
    assert seg.thigh_length == pytest.approx(seg.shin_length, abs=1e-9)
    # Spine: hip z=0.95, neck z = hmd z (1.65) - 0.10 = 1.55, distance 0.60 (with small lateral).
    assert seg.spine_length == pytest.approx(np.linalg.norm([0, 0.05, 0.60]), abs=0.02)
    # Arms ≈ 0.65 m each.
    assert seg.arm_length == pytest.approx(0.65, abs=0.02)
    assert seg.shoulder_half_width > 0.10
    assert seg.shoulder_half_width < 0.30
    assert seg.ankle_to_sole_m == 0.06


def test_solve_requires_minimum_samples() -> None:
    samples = TPoseSamples()
    rng = np.random.default_rng(0)
    for _ in range(3):
        samples.add_frame(**_ideal_tpose_frame(rng))
    with pytest.raises(RuntimeError, match="at least 5"):
        samples.solve()


def test_personal_roundtrip_persists_segments(tmp_path: Path) -> None:
    seg = BodySegments(
        leg_length=0.90,
        thigh_length=0.45,
        shin_length=0.45,
        arm_length=0.60,
        spine_length=0.55,
        shoulder_half_width=0.18,
        ankle_to_sole_m=0.06,
    )
    save_personal(tmp_path, "alice", seg)
    loaded = load_personal(tmp_path, "alice")
    assert loaded == seg


def test_solve_robust_to_outlier_frames() -> None:
    """Median-based aggregation should reject a single big outlier."""
    rng = np.random.default_rng(2)
    samples = TPoseSamples()
    for k in range(20):
        frame = _ideal_tpose_frame(rng, jitter=0.001)
        if k == 5:
            # Inject a 50 cm outlier on the ankle in one frame.
            frame["ankle_left"] = frame["ankle_left"] + np.array([0.5, 0.0, 0.0])
        samples.add_frame(**frame)
    seg = samples.solve()
    # Median over 20 frames must be barely affected by one outlier.
    expected = np.hypot(0.10, 0.89)
    assert seg.leg_length == pytest.approx(expected, abs=0.02)
