// About page: what the project is, how to run the desktop scripts and the web app, what the web pages port, credits.
import { app, esc, route } from './core.js';
import { loadAll } from './data.js';

route('about', async () => {
  const { info } = await loadAll();
  app().innerHTML = `
  <h1>About</h1>
  <p class="lead">depthEstimation is George Dickinson's collection of Python scripts for depth sensing and body tracking:
    a custom C++/Cython interface to the Kinect Xbox 360 (libfreenect) that hands RGB and depth frames to Python, PyQt5 /
    pyqtgraph viewers that combine them with YOLOv8 pose, dlib face landmarks and MediaPipe hands, Kalman-smoothed 3D
    skeletons, and an experimental SLAM-like point-cloud mapper.</p>
  <h2>What the web pages show</h2>
  <ul>
    <li><b>Depth map</b>: <code>cv2.normalize(..., NORM_MINMAX)</code> and <code>cv2.applyColorMap</code> (Jet in quickTest.py
      and test_kinect.py, Viridis in depthEstimation.py), reproduced from OpenCV's own tables.</li>
    <li><b>Point cloud</b>: the pinhole back-projection of slamMapping.py, f = ${esc(info.intrinsics.f_px)} px, principal point
      (${esc(info.intrinsics.cx)}, ${esc(info.intrinsics.cy)}).</li>
    <li><b>Filter</b>: <code>cv2.bilateralFilter</code> with depthEstimation.py's settings (d = ${esc(info.filter.d)}, σ = ${esc(info.filter.sigma_color)}).</li>
    <li><b>Live Kinect</b>: the local app's dashboard for a connected device.</li>
  </ul>
  <p>The maths is in <code>kinect_depth/</code> (numpy, no hardware) and ported to JavaScript; a node test checks the JS
    against the Python on every scene. The scenes are synthetic: ray-cast planes, boxes and spheres with illustrative
    noise and shadow bands, not recordings and not a calibrated Kinect noise model. YOLO, dlib and MediaPipe tracking need
    the desktop scripts and are not part of the web pages.</p>
  <h2>Run the desktop scripts</h2>
  <pre>brew install libfreenect
pip install -r requirements.txt
python setup.py build_ext --inplace     # builds kinect_wrapper
python depthEstimation_kinect.py        # Kinect + 3D skeleton (PyQt5)</pre>
  <h2>Run the web app</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py                       # http://127.0.0.1:8155</pre>
  <p>Without a Kinect (or without the built extension) the Live page says why and the other pages work as on this site.</p>
  <h2>Credits</h2>
  <p>Code by George Dickinson, built with substantial help from Claude (Anthropic). Kinect access through
    <a href="https://github.com/OpenKinect/libfreenect">libfreenect</a>; colour maps from OpenCV. MIT licence.</p>`;
});
