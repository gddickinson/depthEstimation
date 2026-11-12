// Home page: hero, real counts from the API, a rendered preview and links to the pages.
import { app, esc, route } from './core.js';
import { loadAll, sceneDepth } from './data.js';
import { paintDepth } from './render.js';

route('home', async () => {
  const { info, scenes, luts } = await loadAll();
  const fr = info.frame, k = info.intrinsics;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Kinect Xbox 360 · depth in millimetres</p>
    <h1 class="glow">Kinect Depth Lab</h1>
    <p class="lead">The depth side of <b>depthEstimation</b>: a C++/Cython interface streams RGB and depth from a Kinect
      Xbox 360 into Python. This site runs the repo's depth processing in your browser: OpenCV colour maps,
      back-projection to a point cloud with the focal length used in the code, and the bilateral filter, on
      synthetic depth scenes. With a Kinect plugged in, the local app adds a live dashboard.</p>
    <a class="btn" href="#/view">Explore a depth map</a><a class="btn alt" href="#/cloud">Point cloud</a><a class="btn alt" href="#/live">Live Kinect</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${fr.width}×${fr.height}</b><span>depth and RGB frame (px)</span></div>
    <div class="tile"><b>${k.f_px}</b><span>focal length (px), ${esc(k.source)}</span></div>
    <div class="tile"><b>${info.colormaps.length - 1}</b><span>OpenCV colour maps used in the scripts</span></div>
    <div class="tile"><b>d=${info.filter.d}</b><span>bilateral filter window, σ ${info.filter.sigma_color}/${info.filter.sigma_space}</span></div>
    <div class="tile"><b>${scenes.length}</b><span>synthetic depth scenes</span></div>
  </div>
  <div class="two">
    <div class="stage preview"><canvas id="pv" class="pix" aria-label="Room scene depth map, jet colour map" role="img"></canvas></div>
    <div>
      <h2 style="margin-top:0">What is here</h2>
      <div class="grid">
        <a class="card" href="#/view"><b>Depth map</b><small>min-max stretch + colour map</small><span>The normalisation and colour maps of quickTest.py and depthEstimation.py, with a millimetre read-out under the pointer.</span></a>
        <a class="card" href="#/cloud"><b>Point cloud</b><small>pinhole back-projection</small><span>Every pixel lifted to 3D with f = ${k.f_px} px, as slamMapping.py does. Drag to orbit.</span></a>
        <a class="card" href="#/filter"><b>Bilateral filter</b><small>edge-aware smoothing</small><span>The filter from depthEstimation.py on noisy steps, with a row profile against the noise-free truth.</span></a>
        <a class="card" href="#/live"><b>Live Kinect</b><small>local app + device</small><span>RGB, depth and point cloud streamed from a connected Kinect through kinect_wrapper.</span></a>
      </div>
    </div>
  </div>
  <p class="small mute">Scenes are synthetic (ray-cast planes, boxes and spheres with illustrative noise): the repo has no
    recorded depth frames. ${scenes.map(s => esc(s.title)).join(' · ')}.</p>`;
  const room = scenes.find(s => s.name === 'room') || scenes[0];
  const d = sceneDepth(room);
  paintDepth(document.getElementById('pv'), d, fr.width, fr.height, { lut: luts.jet, validOnly: true });
});
