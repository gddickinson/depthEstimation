# Depth Estimation & 3D Body Tracking — Roadmap

## Current State
A collection of ~20 Python scripts (4,376 total lines) for depth estimation and body tracking using webcam, Kinect, YOLO, dlib, and MediaPipe. The project works but is organized as many standalone scripts with duplicated code (e.g., `depthEstimation.py`, `depthEstimation_working.py`, `depthEstimation_copy.py`). There is no package structure — just flat files with significant copy-paste between variants. Includes C++/Cython Kinect interface code. Multiple YOLO model weights are stored directly in the repo.

## Short-term Improvements
- [x] Consolidate duplicate scripts: merge `depthEstimation.py`, `depthEstimation_working.py`, and `depthEstimation_copy.py` into one configurable module
- [ ] Create a proper package structure: `src/depth_estimation/{capture.py, detectors.py, skeleton.py, visualization.py, kinect/}`
- [ ] Extract shared logic (YOLO loading, depth colormap rendering, camera setup) into a `common.py` utility module
- [x] Add `.gitignore` to exclude `.pt` model weights, `.dat` files, `__pycache__`, `*.so`, and video outputs
- [x] Add a `requirements.txt` with pinned versions
- [ ] Add error handling for missing Kinect hardware — currently crashes without a sensor
- [x] Remove `Thumbs.db` and other non-code artifacts from the project

## Feature Enhancements
- [ ] Add a unified CLI entry point that selects mode (webcam-only, kinect-skeleton, kinect-full) via flags
- [ ] Add recording/playback capability — save depth+RGB streams to disk and replay without hardware
- [ ] Upgrade YOLO models from v8 to v11 for better pose estimation accuracy
- [ ] Add multi-person 3D skeleton tracking (currently seems single-person focused)
- [ ] Add FPS counter and performance profiling overlay to all visualization modes
- [ ] Implement depth-based gesture recognition using the skeleton data from `skeletonPositionEstimation.py`
- [ ] Add camera calibration support for more accurate depth-to-3D projection

## Long-term Vision
- [ ] Replace the scattered scripts with a pipeline architecture: CaptureNode -> DetectionNode -> TrackingNode -> VisualizationNode
- [ ] Add support for Intel RealSense and Azure Kinect alongside Xbox 360 Kinect
- [ ] Build a PyQt5 unified dashboard that lets users switch between all modes in one window
- [ ] Add WebSocket streaming for remote 3D skeleton visualization in a browser
- [ ] Implement temporal smoothing across all tracking modes using Kalman filters (currently only in `skeletonPositionEstimation.py`)
- [ ] Add BVH motion capture export from skeleton tracking data

## Technical Debt
- [ ] `skeletonPositionEstimation.py` (427 lines) mixes Kalman filter logic, rendering, and Kinect capture — split into separate concerns
- [ ] `depthEstimation_kinect.py` (291 lines) has hardcoded paths and magic numbers for skeleton connections
- [x] The Cython build (`setup.py`) has no error handling for missing libfreenect
- [x] `slamMapping.py` is experimental and incomplete — either develop it or mark it clearly as WIP
- [x] `animateFBXskeleton.py` and `skull.py` seem tangential — consider moving to an `experiments/` folder
- [x] No tests of any kind — at minimum add smoke tests for the non-hardware modules
