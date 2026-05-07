"""On-disk calibration store.

Implements the persistence contract from plan section C-6:

* ``intrinsic_<cam_id>.json`` — per-camera intrinsics, written once.
* ``extrinsic_session_<id>.json`` — per-session extrinsics, treated as
  read-only at runtime.
* ``personal_<user_id>.json`` — per-user bone lengths and marker offsets.

Critically, :func:`load_extrinsics` rejects writes that would touch a file
with the read-only marker, and :func:`save_extrinsics` refuses to overwrite
unless ``overwrite=True`` is passed explicitly. This guards the
"calibration must not drift just because a marker left view" invariant.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from lislym.calibration.camera_model import CameraModel


@dataclass(frozen=True)
class Intrinsics:
    """Per-camera intrinsic calibration result."""

    camera_name: str
    K: np.ndarray
    dist: np.ndarray
    image_size: tuple[int, int]
    reprojection_rms_px: float
    captured_frames: int

    def to_json(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "K": self.K.tolist(),
            "dist": self.dist.reshape(-1).tolist(),
            "image_size": list(self.image_size),
            "reprojection_rms_px": float(self.reprojection_rms_px),
            "captured_frames": int(self.captured_frames),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Intrinsics":
        return cls(
            camera_name=data["camera_name"],
            K=np.array(data["K"], dtype=np.float64),
            dist=np.array(data["dist"], dtype=np.float64),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            reprojection_rms_px=float(data["reprojection_rms_px"]),
            captured_frames=int(data["captured_frames"]),
        )


@dataclass(frozen=True)
class ExtrinsicEntry:
    """One camera's pose in a session-level extrinsic calibration."""

    camera_name: str
    R: np.ndarray
    t: np.ndarray

    def to_json(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "R": self.R.tolist(),
            "t": self.t.reshape(-1).tolist(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ExtrinsicEntry":
        return cls(
            camera_name=data["camera_name"],
            R=np.array(data["R"], dtype=np.float64),
            t=np.array(data["t"], dtype=np.float64),
        )


@dataclass(frozen=True)
class ExtrinsicSession:
    """A frozen, read-only session-level extrinsic calibration."""

    session_id: str
    created_at: str
    cameras: tuple[ExtrinsicEntry, ...]
    bundle_adjustment_rms_px: float
    notes: str = ""
    locked: bool = True
    content_hash: str = field(default="")

    def to_json(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "cameras": [c.to_json() for c in self.cameras],
            "bundle_adjustment_rms_px": float(self.bundle_adjustment_rms_px),
            "notes": self.notes,
            "locked": bool(self.locked),
        }
        body["content_hash"] = _hash_payload(body)
        return body

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ExtrinsicSession":
        recorded = data.get("content_hash", "")
        verify = dict(data)
        verify.pop("content_hash", None)
        actual = _hash_payload(verify)
        if recorded and recorded != actual:
            raise ValueError(
                f"extrinsic content_hash mismatch — file may have been edited: "
                f"recorded={recorded[:12]}…, actual={actual[:12]}…"
            )
        return cls(
            session_id=data["session_id"],
            created_at=data["created_at"],
            cameras=tuple(ExtrinsicEntry.from_json(c) for c in data["cameras"]),
            bundle_adjustment_rms_px=float(data["bundle_adjustment_rms_px"]),
            notes=str(data.get("notes", "")),
            locked=bool(data.get("locked", True)),
            content_hash=actual,
        )


def _hash_payload(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def save_intrinsics(directory: Path, intrinsics: Intrinsics) -> Path:
    path = directory / f"intrinsic_{intrinsics.camera_name}.json"
    _atomic_write(path, intrinsics.to_json())
    return path


def load_intrinsics(directory: Path, camera_name: str) -> Intrinsics:
    path = directory / f"intrinsic_{camera_name}.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Intrinsics.from_json(data)


def save_extrinsics(
    directory: Path,
    session: ExtrinsicSession,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist an extrinsic session to disk.

    Unless ``overwrite=True``, raises ``FileExistsError`` if the file is
    already present. Combined with the in-memory ``locked`` flag this
    enforces the C-6 contract that runtime never silently mutates extrinsics.
    """
    path = directory / f"extrinsic_session_{session.session_id}.json"
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite locked extrinsic session at {path}: pass overwrite=True"
        )
    _atomic_write(path, session.to_json())
    return path


def load_extrinsics(directory: Path, session_id: str) -> ExtrinsicSession:
    path = directory / f"extrinsic_session_{session_id}.json"
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return ExtrinsicSession.from_json(data)


def assemble_cameras(
    intrinsics: dict[str, Intrinsics], extrinsics: ExtrinsicSession
) -> list[CameraModel]:
    """Combine per-camera intrinsics with a session's extrinsics.

    The intrinsics dict maps camera name → :class:`Intrinsics`. The
    extrinsics session lists every camera's world-to-camera pose; the
    resulting list is in the order of ``extrinsics.cameras``.
    """
    cameras: list[CameraModel] = []
    for entry in extrinsics.cameras:
        if entry.camera_name not in intrinsics:
            raise KeyError(f"intrinsics missing for camera {entry.camera_name!r}")
        intr = intrinsics[entry.camera_name]
        cameras.append(
            CameraModel(
                name=entry.camera_name,
                K=intr.K,
                dist=intr.dist,
                R=entry.R,
                t=entry.t,
                image_size=intr.image_size,
            )
        )
    return cameras


def utc_now_isoformat() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
