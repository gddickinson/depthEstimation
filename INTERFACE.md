# Depth Estimation — Interface Map

## Project Structure

### Core Scripts
- **depthEstimation.py** — Main webcam-based depth estimation with YOLO segmentation, dlib face landmarks, and wireframe overlay
- **depthEstimation_kinect.py** — Kinect-based depth estimation with YOLO pose, dlib face, MediaPipe hands, PyQt5 GUI
- **skeletonPositionEstimation.py** — Kinect skeleton tracking with Kalman filter smoothing, face/hand tracking, PyQt5 GUI
- **skeleton3D.py** — 3D skeleton visualization
- **skeletonTest.py** — Skeleton tracking test script
- **skeletonAnimation.py** — Skeleton animation playback

- **depth_utils.py** — cv2/numpy helpers for depthEstimation.py: box format conversion (YOLO xyxy <-> KCF xywh),
  `person_box_label`, `edge_aware_filter` (bilateral, sigma_color 0.1 for the 0-1 model output); tests/test_depth_utils.py

### Hardware Interface
- **kinect_wrapper.pyx** — Cython wrapper for libfreenect (Kinect capture): `PyKinectCapture.init/get_frames/close`
- **kinect_capture.cpp / .h** — C++ Kinect capture implementation
- **setup.py** — Build script for Cython extension (auto-finds libfreenect/libusb)
- **kinectStream/** — Kinect streaming test scripts and build variants

### Depth maths (kinect_depth/, numpy only, no hardware)
- **kinect_depth/processing.py** — `FRAME_W/FRAME_H` (640x480), `FOCAL_PX` (525), `depth_to_points` (used by slamMapping.py),
  `normalize_minmax` (cv2.normalize NORM_MINMAX), `apply_colormap` / `load_colormaps`, `bilateral_filter` / `bilateral_kernel`
  (cv2.bilateralFilter weights), `depth_stats`
- **kinect_depth/scenes.py** — `SCENES`, `render_scene` (ray-cast synthetic depth frames), `scene_list`
- **kinect_depth/colormap_luts.json** — OpenCV Jet/Viridis tables (written by scripts/make_colormap_luts.py)

### Web interface (web/, run_web.py on 127.0.0.1:8155)
- **web/app.py** — FastAPI: `/api/info`, `/api/scenes`, `/api/colormaps` (exported as STATIC_ENDPOINTS), live
  `/api/device/status|connect|disconnect|frame`
- **web/device.py** — `KinectSource`: lazy kinect_wrapper import, lock, frame timeout, status reasons
- **web/static/** — index.html, core.js + style.css (shared kit), main.js (router), data.js, render.js (canvas drawing,
  `CloudView`), pages home/view/cloud/filter/live/about.js, static_api.js (static site),
  **depth_engine.js** / **scene_engine.js** (JS ports of kinect_depth) + depth_engine.test.mjs (node, vs fixture)
- **scripts/** — build_static.py (Pages export), screenshots.py (Playwright smoke test), make_engine_fixture.py,
  make_colormap_luts.py
- **tests/** — test_web.py (API + static build + node), test_processing.py, test_engine_fixture.py,
  test_kinect_wrapper.py (no-device lifecycle; skipped if the extension is not built), fixtures/engine_fixture.json

### Test & Utility Scripts
- **test_smoke.py** — Smoke tests for non-hardware modules (8 tests)
- **test_kinect.py** — Kinect hardware tests
- **cameraTest.py** — Webcam test
- **kinectTest.py** — Basic Kinect test
- **quickTest.py** — Quick functionality test
- **videoTest.py** — Video file test
- **yoloTest.py** — YOLO model test
- **yolo_segmentation_test.py** — YOLO segmentation test

### Experimental (experiments/)
- **animateFBXskeleton.py** — FBX skeleton animation loading
- **skull.py** — Skull model rendering
- **skull_from_obj.py** — Skull model from OBJ file
- **create_skull_model.py** — Skull model generation

### Archived (_archive/)
- Duplicate/backup versions of scripts (depthEstimation_copy.py, *_working.py, etc.)
- Thumbs.db

### Other
- **slamMapping.py** — EXPERIMENTAL/WIP SLAM mapping with Kinect

## Key Constants (shared across scripts)
- `SKELETON_CONNECTIONS` — YOLO pose keypoint connections (18 pairs)
- `FACIAL_LANDMARKS_CONNECTORS` — dlib 68-point face landmark connections
- `HAND_CONNECTIONS` — MediaPipe hand landmark connections (20 pairs)
- `frameWidth=640, frameHeight=480` — Default capture resolution
