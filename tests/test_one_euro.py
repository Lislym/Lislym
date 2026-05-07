"""Tests for the One Euro Filter."""

from __future__ import annotations

import numpy as np
import pytest

from lislym.postprocess.one_euro import OneEuroFilter


def test_first_call_passes_input_through() -> None:
    f = OneEuroFilter()
    out = f.filter(np.array([1.0, 2.0, 3.0]), t=0.0)
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_static_input_does_not_drift() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.05)
    last = None
    for k in range(100):
        last = f.filter(np.array([0.5, 0.5, 0.5]), t=k / 60.0)
    np.testing.assert_allclose(last, [0.5, 0.5, 0.5], atol=1e-9)


def test_filter_attenuates_high_frequency_noise() -> None:
    rng = np.random.default_rng(0)
    f = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    signal = []
    filtered = []
    for k in range(120):
        x = np.array([0.0, 0.0, 0.0]) + rng.normal(0, 0.05, 3)
        signal.append(x)
        filtered.append(f.filter(x, t=k / 60.0))
    s = np.std(np.stack(signal), axis=0)
    fout = np.std(np.stack(filtered)[20:], axis=0)
    assert np.all(fout < s * 0.6), "filter must reduce noise variance"


def test_filter_tracks_fast_motion_with_low_lag() -> None:
    f = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    # Step jump after 10 frames; check we reach the new value within 5 frames.
    out = []
    for k in range(40):
        x = np.array([0.0, 0.0, 0.0]) if k < 10 else np.array([1.0, 0.0, 0.0])
        out.append(f.filter(x, t=k / 60.0))
    assert out[15][0] > 0.7, "must track step within 5 frames at beta=0.5"


def test_reset_clears_state() -> None:
    f = OneEuroFilter()
    f.filter(np.array([5.0, 5.0, 5.0]), t=0.0)
    f.reset()
    out = f.filter(np.array([1.0, 1.0, 1.0]), t=0.0)
    np.testing.assert_allclose(out, [1.0, 1.0, 1.0])
