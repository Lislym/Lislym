"""Tests for calibration persistence (plan section C-6).

The critical invariant under test: extrinsic calibration files, once
written, should not silently change at runtime when markers go out of
view. This is enforced through three mechanisms:

* ``ExtrinsicSession.locked`` flag.
* ``save_extrinsics`` refuses to overwrite an existing file by default.
* The on-disk content hash detects post-hoc edits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lislym.calibration.camera_model import CameraModel
from lislym.calibration.store import (
    ExtrinsicEntry,
    ExtrinsicSession,
    Intrinsics,
    assemble_cameras,
    load_extrinsics,
    load_intrinsics,
    save_extrinsics,
    save_intrinsics,
    utc_now_isoformat,
)


def _intrinsics(name: str = "cam0") -> Intrinsics:
    return Intrinsics(
        camera_name=name,
        K=np.array([[900.0, 0, 640], [0, 900, 360], [0, 0, 1]]),
        dist=np.array([0.01, -0.02, 0.0, 0.0, 0.001]),
        image_size=(1280, 720),
        reprojection_rms_px=0.27,
        captured_frames=80,
    )


def _session(session_id: str = "demo") -> ExtrinsicSession:
    return ExtrinsicSession(
        session_id=session_id,
        created_at=utc_now_isoformat(),
        cameras=(
            ExtrinsicEntry("cam0", R=np.eye(3), t=np.zeros(3)),
            ExtrinsicEntry("cam1", R=np.eye(3), t=np.array([1.0, 0.0, 0.0])),
        ),
        bundle_adjustment_rms_px=0.41,
        notes="ワンドキャリブ 30s",
    )


def test_intrinsics_roundtrip(tmp_path: Path) -> None:
    intr = _intrinsics()
    save_intrinsics(tmp_path, intr)
    loaded = load_intrinsics(tmp_path, "cam0")
    assert loaded.camera_name == intr.camera_name
    np.testing.assert_allclose(loaded.K, intr.K)
    np.testing.assert_allclose(loaded.dist, intr.dist)
    assert loaded.image_size == intr.image_size
    assert loaded.reprojection_rms_px == pytest.approx(intr.reprojection_rms_px)


def test_extrinsics_roundtrip_preserves_pose(tmp_path: Path) -> None:
    session = _session()
    save_extrinsics(tmp_path, session)
    loaded = load_extrinsics(tmp_path, "demo")
    assert loaded.session_id == session.session_id
    for original, restored in zip(session.cameras, loaded.cameras, strict=True):
        np.testing.assert_allclose(original.R, restored.R)
        np.testing.assert_allclose(original.t, restored.t)


def test_extrinsics_refuses_overwrite_by_default(tmp_path: Path) -> None:
    session = _session()
    save_extrinsics(tmp_path, session)
    with pytest.raises(FileExistsError):
        save_extrinsics(tmp_path, session)
    save_extrinsics(tmp_path, session, overwrite=True)  # explicit opt-in is fine


def test_extrinsics_detects_post_hoc_edit(tmp_path: Path) -> None:
    session = _session()
    path = save_extrinsics(tmp_path, session)
    raw = json.loads(path.read_text())
    raw["cameras"][1]["t"] = [99.0, 0.0, 0.0]  # tamper with the file
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="content_hash"):
        load_extrinsics(tmp_path, "demo")


def test_extrinsics_hash_stable_across_dict_order(tmp_path: Path) -> None:
    session = _session()
    save_extrinsics(tmp_path, session)
    loaded_a = load_extrinsics(tmp_path, "demo")
    save_extrinsics(tmp_path, session, overwrite=True)
    loaded_b = load_extrinsics(tmp_path, "demo")
    assert loaded_a.content_hash == loaded_b.content_hash


def test_assemble_cameras_uses_intrinsics_and_extrinsics() -> None:
    intrinsics = {"cam0": _intrinsics("cam0"), "cam1": _intrinsics("cam1")}
    session = _session()
    cams = assemble_cameras(intrinsics, session)
    assert [c.name for c in cams] == ["cam0", "cam1"]
    assert isinstance(cams[0], CameraModel)
    np.testing.assert_allclose(cams[1].t, [1.0, 0.0, 0.0])


def test_assemble_cameras_raises_on_missing_intrinsics() -> None:
    session = _session()
    with pytest.raises(KeyError):
        assemble_cameras({"cam0": _intrinsics("cam0")}, session)
