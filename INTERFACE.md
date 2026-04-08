# Depth Estimation — Interface Map

## Project Structure

### Core Scripts
- **depthEstimation.py** — Main webcam-based depth estimation with YOLO segmentation, dlib face landmarks, and wireframe overlay
- **depthEstimation_kinect.py** — Kinect-based depth estimation with YOLO pose, dlib face, MediaPipe hands, PyQt5 GUI
- **skeletonPositionEstimation.py** — Kinect skeleton tracking with Kalman filter smoothing, face/hand tracking, PyQt5 GUI
- **skeleton3D.py** — 3D skeleton visualization
- **skeletonTest.py** — Skeleton tracking test script
- **skeletonAnimation.py** — Skeleton animation playback

### Hardware Interface
- **kinect_wrapper.pyx** — Cython wrapper for libfreenect (Kinect capture)
- **kinect_capture.cpp / .h** — C++ Kinect capture implementation
- **setup.py** — Build script for Cython extension (auto-finds libfreenect/libusb)
- **kinectStream/** — Kinect streaming test scripts and build variants

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
