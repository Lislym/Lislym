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
2. **Per-camera intrinsic calibration** — print the 6×9 ChArUco board on
   A4 (square=30 mm, marker=22 mm). For each camera in turn:

   ```bash
   lislym-calib-intrinsic --camera 0 --name inspiron --frames 80 --calib-dir ./calib
   lislym-calib-intrinsic --camera 1 --name iphone14 --frames 80 --calib-dir ./calib
   lislym-calib-intrinsic --camera 2 --name ipad --frames 80 --calib-dir ./calib
   ```

   Aim for ≤ 0.5 px RMS. The script prints it after pressing `c`.
3. **Wand extrinsic calibration** — assemble the 50 cm wand with red /
   green / blue 5 cm spheres at 0, 25, 50 cm. With all three cameras
   running:

   ```bash
   lislym-calib-extrinsic \
     --intrinsic-dir ./calib --session phase0 \
     --cameras "inspiron:0,iphone14:1,ipad:2" --frames 60
   ```

   Wave the wand slowly through the capture volume and press `space`
   each time all three cameras see all three balls.
4. **Personal calibration** — there is no dedicated CLI yet, but the
   `lislym.calibration.personal.TPoseSamples` API is straightforward:
   stand T-pose for ~3 s, drive `add_frame(...)` from your tracker, and
   call `solve()` → `save_personal(...)`.
5. **Floor plane** — the simplest path is to feed three or more 3D
   ankle positions while the user stands quietly into
   `fit_plane_from_points`. AprilTag-based recovery is also wired up
   in `lislym.calibration.floor.floor_plane_from_apriltag` if you tape
   a tag to the floor.
6. **Runtime** — once all four artefacts (`intrinsic_*.json`,
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
7. **VRChat verification** — open VRChat with OSC enabled, set 3
   trackers in the Settings → OSC menu, and check that the avatar's
   hip and feet follow the markers when you walk in front of the
   cameras.
