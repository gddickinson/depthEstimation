// Live Kinect dashboard (local app only): RGB, colourised depth and the back-projected point cloud of frames streamed
// by kinect_wrapper.PyKinectCapture through /api/device/frame, with Python's frame statistics. The static site and a
// server without a Kinect explain what is missing instead.
import { $, api, app, esc, isStatic, needsServer, onLeave, route } from './core.js';
import { CMAP_NAMES, loadAll, lutFor, mm } from './data.js';
import { depthToPoints, probe } from './depth_engine.js';
import { CloudView, colorbar, paintDepth, paintRGB, pixelAt, seg } from './render.js';

const b64bytes = s => Uint8Array.from(atob(s), c => c.charCodeAt(0));
function decodeDepth(s) {
  const b = b64bytes(s), dv = new DataView(b.buffer), out = new Uint16Array(b.length / 2);
  for (let i = 0; i < out.length; i++) out[i] = dv.getUint16(2 * i, true);
  return out;
}

const INTRO = `<h1>Live Kinect</h1>
  <p class="lead">With a Kinect Xbox 360 on USB, the local app reads frames through the repo's Cython wrapper
    (<code>kinect_wrapper.PyKinectCapture</code>, libfreenect, 640×480 RGB and depth in mm) and shows the RGB image,
    the colourised depth and the point cloud side by side, as the PyQt viewers do.</p>`;

route('live', async () => {
  if (isStatic()) {
    app().innerHTML = `${INTRO}${needsServer('The live Kinect dashboard')}
      <p>The depth maths it uses runs here on synthetic scenes: <a href="#/view">depth map</a>,
      <a href="#/cloud">point cloud</a>, <a href="#/filter">bilateral filter</a>.</p>`;
    return;
  }
  const { luts } = await loadAll();
  const st = { cmap: 'jet', validOnly: true, step: 2, paused: false, color: 'rgb' };
  app().innerHTML = `${INTRO}
  <div class="panel"><span class="state" id="state"></span> <span class="mute" id="reason"></span><br>
    <button id="con">Connect</button><button class="alt" id="dis">Disconnect</button><button class="alt" id="pause">Pause</button></div>
  <div id="views" hidden>
    <div class="row" id="ctl"></div>
    <div class="two">
      <figure class="stage" style="margin:0"><canvas id="rgb" role="img" aria-label="Live RGB frame"></canvas></figure>
      <figure class="stage" style="margin:0"><canvas id="dep" class="pix" role="img" aria-label="Live depth frame"></canvas><div class="hud" id="hud">point at the depth image</div></figure>
    </div>
    <div id="bar"></div>
    <div class="two">
      <div class="stage"><canvas id="cloud" role="img" aria-label="Live point cloud"></canvas></div>
      <div><h3>Frame (computed in Python)</h3><dl class="kv" id="stats"></dl></div>
    </div>
  </div>
  <p class="small mute" id="hint"></p>`;
  const ctl = $('#ctl');
  seg(ctl, 'Colour map', Object.entries(CMAP_NAMES).map(([k, t]) => [k, t.split(' (')[0]]), st.cmap, v => { st.cmap = v; });
  seg(ctl, 'Stretch range', [['valid', 'Readings only'], ['frame', 'Whole frame']], 'valid', v => { st.validOnly = v === 'valid'; });
  seg(ctl, 'Resolution', [['1', '640×480'], ['2', '320×240'], ['4', '160×120']], String(st.step), v => { st.step = +v; });
  seg(ctl, 'Cloud colour', [['rgb', 'RGB'], ['depth', 'Depth']], st.color, v => { st.color = v; });
  const cloud = new CloudView($('#cloud'), { w: 480, h: 360 });
  let alive = true, frame = null, t0 = performance.now(), n = 0;
  onLeave(() => { alive = false; });

  function show(s) {
    $('#state').textContent = s.state; $('#state').className = 'state ' + s.state; $('#reason').textContent = s.reason;
    $('#views').hidden = !s.streaming;
    $('#hint').innerHTML = s.streaming ? '' : 'No live frames. The synthetic scenes work without a device: <a href="#/view">depth map</a>, <a href="#/cloud">point cloud</a>, <a href="#/filter">filter</a>.';
  }
  async function loop() {
    while (alive) {
      if (st.paused) { await new Promise(r => setTimeout(r, 200)); continue; }
      const f = await api(`/device/frame?step=${st.step}`, { fresh: true }).catch(e => ({ ok: false, state: 'unavailable', reason: e.message }));
      if (!alive) return;
      if (!f.ok) { show({ ...f, streaming: false }); return; }
      draw(f); n++;
    }
  }
  function draw(f) {
    const w = f.width, h = f.height, rgb = b64bytes(f.rgb), depth = decodeDepth(f.depth), lut = lutFor(luts, st.cmap);
    frame = { w, h, depth, f: f.f_px };
    paintRGB($('#rgb'), rgb, w, h);
    const gray = paintDepth($('#dep'), depth, w, h, { lut, validOnly: st.validOnly, holesBlack: st.validOnly });
    $('#bar').innerHTML = colorbar(lut, gray.lo, gray.hi);
    const { xyz, index } = depthToPoints(depth, w, h, f.f_px, Math.max(1, 4 / f.step));
    const col = new Uint8Array(index.length * 3);
    index.forEach((i, p) => col.set(st.color === 'rgb' ? rgb.subarray(3 * i, 3 * i + 3) : (lut ? lut[gray[i]] : [gray[i], gray[i], gray[i]]), 3 * p));
    cloud.setPoints(xyz, col); cloud.draw();
    const s = f.stats, fps = n / ((performance.now() - t0) / 1000);
    $('#stats').innerHTML = `<dt>frames</dt><dd>${f.frames}</dd><dt>display rate</dt><dd>${fps.toFixed(1)} /s</dd>
      <dt>with a reading</dt><dd>${(100 * s.valid_fraction).toFixed(1)} %</dd><dt>nearest</dt><dd>${mm(s.min_mm)}</dd>
      <dt>median</dt><dd>${mm(s.median_mm)}</dd><dt>farthest</dt><dd>${mm(s.max_mm)}</dd>
      <dt>focal length</dt><dd>${esc(f.f_px)} px at this resolution</dd>`;
  }
  $('#dep').addEventListener('pointermove', e => {
    if (!frame) return;
    const [u, v] = pixelAt($('#dep'), e, frame.w, frame.h), p = probe(frame.depth, frame.w, frame.h, u, v, frame.f);
    $('#hud').textContent = p.z > 0 ? `z ${p.z} mm  x ${Math.round(p.x)}  y ${Math.round(p.y)}` : 'no reading';
  });
  $('#con').addEventListener('click', async () => {
    $('#con').disabled = true;
    const s = await api('/device/connect', { body: {} }); show(s); $('#con').disabled = false;
    if (s.streaming) { t0 = performance.now(); n = 0; loop(); }
  });
  $('#dis').addEventListener('click', async () => { show(await api('/device/disconnect', { body: {} })); });
  $('#pause').addEventListener('click', () => { st.paused = !st.paused; $('#pause').textContent = st.paused ? 'Resume' : 'Pause'; });
  const s = await api('/device/status', { fresh: true });
  show(s);
  if (s.streaming) loop();
});
