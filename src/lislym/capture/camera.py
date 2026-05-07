"""Camera capture workers and PC-side timestamp synchronisation.

Each ``CameraWorker`` runs an OpenCV ``VideoCapture`` in its own thread,
reads frames at the highest rate the device delivers, and pushes the most
recent frame plus a monotonic PC timestamp into a single-slot queue. The
sync layer (``MultiCameraReader``) polls all workers, finds the closest
timestamp to a target time, and returns one frame per camera that share a
common reference time.

Why one slot instead of a full queue: Iriun / EpocCam / built-in webcams
all run independent clocks. Buffering old frames inflates latency without
helping anyone — for tracking we always want the freshest sample.

The development build runs the iPhone 14, iPad Air 5, and Inspiron 15 camera
each through this same path. iPhone/iPad come in as virtual cameras (Iriun)
which OpenCV treats as ordinary device indices on Windows.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from queue import Empty, Queue

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraConfig:
    """Configuration for a single capture device.

    Attributes
    ----------
    name:
        Stable identifier (e.g. ``"iphone14"``, ``"ipad_air5"``, ``"inspiron"``).
    source:
        Anything ``cv2.VideoCapture`` accepts: an integer device index, a
        file path, or a stream URL. On Windows, Iriun virtual cameras
        appear as integer indices; the built-in webcam is usually 0.
    width, height:
        Requested capture resolution. The driver may pick the closest
        supported mode.
    fps:
        Requested frame rate. The bottleneck device (Inspiron at ~30 fps)
        sets the system's effective rate.
    backend:
        OpenCV capture backend hint. ``cv2.CAP_DSHOW`` works well for Iriun
        on Windows; ``cv2.CAP_ANY`` lets OpenCV decide.
    fourcc:
        Optional FourCC like ``"MJPG"`` to force a high-throughput codec
        on USB webcams that default to NV12 / YUY2 with frame drops.
    """

    name: str
    source: int | str
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    backend: int = cv2.CAP_ANY
    fourcc: str | None = "MJPG"


@dataclass
class Frame:
    """One captured frame paired with a PC monotonic timestamp."""

    camera_name: str
    timestamp: float  # seconds, ``time.monotonic()`` reference
    frame_index: int
    image: np.ndarray  # BGR, ``(H, W, 3)`` uint8


class CameraWorker:
    """Background thread that keeps the latest frame from one camera."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._slot: Queue[Frame] = Queue(maxsize=1)
        self._frame_index = 0
        self._error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("worker already started")
        self._cap = cv2.VideoCapture(self.config.source, self.config.backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera {self.config.source!r}")
        if self.config.fourcc:
            fourcc = cv2.VideoWriter_fourcc(*self.config.fourcc)
            self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name=f"cam-{self.config.name}", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        try:
            while not self._stop.is_set():
                ok, image = self._cap.read()
                ts = time.monotonic()
                if not ok or image is None:
                    # The driver dropped a frame; keep going. Don't sleep
                    # long enough to mask a real disconnect — a quick yield
                    # is enough.
                    time.sleep(0.001)
                    continue
                self._frame_index += 1
                frame = Frame(
                    camera_name=self.config.name,
                    timestamp=ts,
                    frame_index=self._frame_index,
                    image=image,
                )
                # Replace whatever was in the slot so the consumer always
                # sees the freshest frame.
                if self._slot.full():
                    try:
                        self._slot.get_nowait()
                    except Empty:
                        pass
                self._slot.put(frame)
        except BaseException as exc:  # capture and surface in main thread
            self._error = exc

    def latest(self, timeout: float = 0.5) -> Frame | None:
        """Block briefly for the freshest frame; ``None`` if none arrived in time."""
        if self._error is not None:
            raise self._error
        try:
            frame = self._slot.get(timeout=timeout)
        except Empty:
            return None
        return frame

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def align_frames(
    frames: dict[str, "Frame"], max_skew: float
) -> tuple[float, dict[str, "Frame"], float]:
    """Pick the median timestamp and drop frames that drift too far from it.

    Pure function pulled out of :class:`MultiCameraReader` so it can be unit
    tested without spinning up real cameras. Returns the chosen target
    time, the kept frames, and the maximum residual skew.
    """
    if not frames:
        return time.monotonic(), {}, 0.0
    stamps = sorted(f.timestamp for f in frames.values())
    target = stamps[len(stamps) // 2]
    kept = {n: f for n, f in frames.items() if abs(f.timestamp - target) <= max_skew}
    skew = max((abs(f.timestamp - target) for f in kept.values()), default=0.0)
    return target, kept, skew


@dataclass(frozen=True)
class SynchronisedFrames:
    """One frame per camera, all close to the same target timestamp.

    The ``target_time`` is the reference instant we tried to align to;
    individual ``Frame.timestamp`` values may differ by up to
    :attr:`MultiCameraReader.max_skew`.
    """

    target_time: float
    frames: dict[str, Frame] = field(default_factory=dict)
    skew_seconds: float = 0.0

    @property
    def cameras_present(self) -> list[str]:
        return list(self.frames.keys())


class MultiCameraReader:
    """Aggregates several ``CameraWorker``s into time-aligned bundles.

    Because the workers run independent clocks, exact alignment is
    impossible. We accept the freshest frame from each camera that lies
    within :attr:`max_skew` of the bundle's target time. Cameras whose
    latest frame is too stale are simply omitted, which downstream
    triangulation handles via the K-adaptive fallback (Phase 3.5).
    """

    def __init__(self, configs: list[CameraConfig], *, max_skew: float = 0.040) -> None:
        if len({c.name for c in configs}) != len(configs):
            raise ValueError("camera names must be unique")
        self.workers = [CameraWorker(cfg) for cfg in configs]
        self.max_skew = max_skew

    def __enter__(self) -> "MultiCameraReader":
        for w in self.workers:
            w.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for w in self.workers:
            w.stop()

    def read(self, timeout: float = 0.5) -> SynchronisedFrames:
        """Return a bundle where all frames are within ``max_skew`` of one another.

        The strategy is: pull the latest frame from every worker, then pick
        the median timestamp as the target. Drop any frame whose stamp is
        more than ``max_skew`` from that target.
        """
        latest: dict[str, Frame] = {}
        for w in self.workers:
            frame = w.latest(timeout=timeout)
            if frame is not None:
                latest[w.config.name] = frame

        if not latest:
            return SynchronisedFrames(target_time=time.monotonic())

        target, kept, skew = align_frames(latest, self.max_skew)
        return SynchronisedFrames(target_time=target, frames=kept, skew_seconds=skew)
