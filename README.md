# Lislym

Multi-camera color-marker full-body tracker for VRChat — a Vive Tracker / SlimeVR
alternative built around 3 markers (hip + both ankles), an HMD, and two
controllers.

> 日本語版は [README.ja.md](./README.ja.md) を参照してください。

The minimum viable system runs on three cameras, three colored spheres,
ChArUco-based calibration that you can put away after setup, and a Windows PC.
Target accuracy: 10–15 mm RMSE on hip / ankle positions, comparable to a Vive
Tracker FBT setup at a fraction of the cost.

See `/root/.claude/plans/media-pipe-pose-vive-tracker-3d-breezy-alpaca.md` for
the full design plan.

## Status

Early development. Implementation order:

1. **Phase 0** — capture from iPhone/iPad/laptop via Iriun virtual cameras with
   timestamp-aligned frames.
2. **Phase 1** — sub-pixel color-sphere detection per camera.
3. **Phase 2** — multi-view triangulation with weighted DLT + RANSAC.
4. **Phase 3.5** — occlusion-robust fallback (K=1 ray + kinematics, EKF, identity
   re-association on view return) and persistent calibration.
5. **Phase 4–7** — floor / ZUPT / IK / OSC / accuracy tuning.

## Repository layout

```
src/lislym/
  calibration/   intrinsic, extrinsic, floor, personal, hmd alignment
  capture/       camera workers, timestamp sync
  detection/     color sphere sub-pixel finder, MediaPipe auxiliary
  fusion/        triangulation, sliding-window bundle adjustment, EKF
  postprocess/   ZUPT, foot contact, floor snap, One Euro filter
  ik/            two-bone leg IK, upper-body IK, hierarchical solver
  output/        VRChat OSC tracker emitter
  tools/         CLI entry points (calibration utilities, benchmarks)
  main.py        runtime entry point
tests/           unit tests
```

## Install (development)

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
pytest
```

## Hardware verification workflow

The codebase ships with synthetic-data tests that cover every algorithm
end-to-end. Once they pass on your machine, run through this sequence on
the real cameras (iPhone 14 + iPad Air 5 + Inspiron 15 7000 via Iriun
Webcam, or any other 3-camera rig):

1. **Iriun pre-flight** — install Iriun Webcam on each phone/tablet and
   on the PC. Confirm three webcams show up in `Device Manager` →
   `Cameras` and pick the device indices (Inspiron usually 0; iPhone and
   iPad take the next two depending on connect order).
2. **Print the calibration targets** — generate A4 print-ready PNGs:

   ```bash
   lislym-print-charuco --out ./prints/charuco_a4.png --dpi 300
   lislym-print-floor-aruco --out ./prints/floor_aruco_a4.png --dpi 300
   ```

   Print at 100 % scale (disable any "fit to page" option) and verify
   the square edge with a ruler before proceeding.
3. **Per-camera intrinsic calibration** (auto-capture). For each camera in turn:

   ```bash
   lislym-calib-intrinsic --camera 0 --name inspiron --frames 80 --calib-dir ./calib
   lislym-calib-intrinsic --camera 1 --name iphone14 --frames 80 --calib-dir ./calib
   lislym-calib-intrinsic --camera 2 --name ipad --frames 80 --calib-dir ./calib
   ```

   Press `s` to start. Move the board around — sharp, diverse poses
   are auto-saved; blurred or near-duplicate frames are skipped. When
   80 frames are buffered the tool pauses; press `c` to confirm and
   calibrate (target RMS ≤ 0.5 px) or `r` to keep capturing.
4. **Wand extrinsic calibration** (auto-capture). Build the 50 cm wand
   with red / green / blue 5 cm spheres at 0, 25, 50 cm. With all
   cameras running:

   ```bash
   lislym-calib-extrinsic \
     --intrinsic-dir ./calib --session phase0 \
     --cameras "inspiron:0,iphone14:1,ipad:2" --frames 60
   ```

   Press `s` to start, then slowly wave the wand through the capture
   volume. Frames are auto-saved when all three balls are visible in
   every camera *and* the wand has moved to a new region. Press `c`
   to run the bundle adjustment.
5. **Personal calibration** — there is no dedicated CLI yet, but the
   `lislym.calibration.personal.TPoseSamples` API is straightforward:
   stand T-pose for ~3 s, drive `add_frame(...)` from your tracker, and
   call `solve()` → `save_personal(...)`.
6. **Floor plane** — the simplest path is to feed three or more 3D
   ankle positions while the user stands quietly into
   `fit_plane_from_points`. AprilTag-based recovery is also wired up
   in `lislym.calibration.floor.floor_plane_from_apriltag` if you tape
   a tag to the floor.
7. **Runtime** — once all four artefacts (`intrinsic_*.json`,
   `extrinsic_session_phase0.json`, `personal_<user>.json`, floor
   plane) exist:

   ```bash
   lislym \
     --intrinsic-dir ./calib --session phase0 --user alice \
     --cameras "inspiron:0,iphone14:1,ipad:2" \
     --osc-host 127.0.0.1 --osc-port 9000
   ```

   You should see live FPS, hip world coordinates, foot-contact flags,
   and triangulation modes printed once per second.
8. **VRChat verification** — open VRChat with OSC enabled, set 3
   trackers in the Settings → OSC menu, and check that the avatar's
   hip and feet follow the markers when you walk in front of the
   cameras.

## Marker-method alternatives

A separate analysis covers alternatives to the all-colored-sphere setup.
See [`docs/marker_alternatives.md`](docs/marker_alternatives.md). TL;DR:

- **Hybrid (hip = ArUco, ankles = color spheres)** — strongly recommended
  v1.5. Solves the hip-yaw weakness in the current plan with ~4–5 days
  of work. Implement in parallel with v1 hardware verification.
- **Full ArUco multi-marker bands** — v2 candidate. Eliminates identity
  confusion entirely, gives 6-DoF per body part. Pursue only if v1.5
  still has issues.
- **Colored paper bands** — not worth implementing.

### Recommended marker specs

Full reasoning in `docs/marker_alternatives.md` §2.8–2.10.

| Part | Recommendation |
|---|---|
| Hip ArUco | **8×8 cm**, DICT_4X4_50, **1 marker** on belt front |
| Floor ArUco | 15×15 cm, DICT_4X4_50, taped once |
| Ankle balls | φ5 cm, **3 per ankle** placed **front / outer / back** (inner side is occluded by the contralateral leg) |

The hip needs only one marker because the user faces the play area in
VRChat and the 4-camera ring guarantees at least one front view; rare
back-facing moments are covered by EKF prediction.
