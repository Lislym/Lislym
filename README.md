# Lislym

Multi-camera color-marker full-body tracker for VRChat — a Vive Tracker / SlimeVR
alternative built around 3 markers (hip + both ankles), an HMD, and two
controllers.

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
