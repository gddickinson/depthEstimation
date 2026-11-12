# Session log

## 2026-11-12 — web interface (branch `web-interface`, not pushed)
- Added the Kinect Depth Lab web interface: FastAPI app (`web/`, `run_web.py`, port 8155) and static GitHub Pages site
  (depth map, point cloud, bilateral filter, live Kinect dashboard, about).
- Factored the hardware-free depth maths into `kinect_depth/` (processing + synthetic scenes); ported to JS
  (`web/static/depth_engine.js`, `scene_engine.js`), node test against `tests/fixtures/engine_fixture.json`.
- No recorded depth frames exist in the repo; the pages use synthetic ray-cast scenes (labelled as such).
- Bugs fixed (separate commits, regression tests first):
  - Kinect wrapper: process aborted when a capture whose init() failed was freed (join of a never-started thread);
    get_frames() hung after a failed init(); `close()` (called by every GUI's closeEvent) did not exist. Rebuilt
    `kinect_wrapper.cpython-311-darwin.so`.
  - slamMapping.py: point-cloud x grid built with np.repeat (wrong column per pixel) and depth-0 pixels kept as points.
  - README: untrained monocular model described as producing depth; unused midas_model.pb; moved scripts' paths.
- Open for George: depthEstimation.py prints "Avg Depth: <bbox width in px>m" (bbox width labelled as metres);
  its bilateral filter (sigma_color 100) runs on 0-1 model output, where it is a plain Gaussian blur.
