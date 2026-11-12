// Depth map page: a synthetic scene through cv2.normalize (min-max) and an OpenCV colour map, with a read-out of the
// depth and back-projected position under the pointer and frame statistics (checked against Python's).
import { $, app, esc, route, slider } from './core.js';
import { CMAP_NAMES, loadAll, lutFor, mm, sceneDepth } from './data.js';
import { depthStats, probe } from './depth_engine.js';
import { colorbar, paintDepth, pixelAt, seg } from './render.js';

route('view', async (args, query) => {
  const { info, scenes, luts, byName } = await loadAll();
  const W = info.frame.width, H = info.frame.height, F = info.intrinsics.f_px;
  const st = { scene: byName[query.scene] ? query.scene : 'room', cmap: 'jet', validOnly: false, noise: null, shadow: true };
  app().innerHTML = `
  <h1>Depth map</h1>
  <p class="lead">A depth frame is 640×480 millimetres (0 = no reading). The scripts stretch it to 0–255 with
    <code>cv2.normalize(..., NORM_MINMAX)</code> and colour it with <code>cv2.applyColorMap</code>.</p>
  <div class="side">
    <div class="panel" id="ctl">
      <label for="sc">Scene</label>
      <select id="sc">${scenes.map(s => `<option value="${esc(s.name)}">${esc(s.title)}</option>`).join('')}</select>
      <p class="small mute" id="about"></p>
    </div>
    <div>
      <div class="stage"><canvas id="cv" class="pix" role="img" aria-label="Colourised depth map"></canvas><div class="hud" id="hud">point at the image</div></div>
      <div id="bar"></div>
      <div class="two">
        <div><h3>Frame statistics</h3><dl class="kv" id="stats"></dl></div>
        <div><h3>Notes</h3><p class="small" id="note"></p></div>
      </div>
    </div>
  </div>`;
  const ctl = $('#ctl'), sel = $('#sc'), cv = $('#cv');
  sel.value = st.scene;
  seg(ctl, 'Colour map', Object.entries(CMAP_NAMES).map(([k, t]) => [k, t.split(' (')[0]]), st.cmap, v => { st.cmap = v; draw(); });
  seg(ctl, 'Stretch range', [['frame', 'Whole frame'], ['valid', 'Readings only']], 'frame', v => { st.validOnly = v === 'valid'; draw(); });
  const noise = slider(ctl, { id: 'nz', label: 'Noise scale', min: 0, max: 12, step: 0.5, value: byName[st.scene].noise_mm, unit: ' mm at 1 m' }, v => { st.noise = v; draw(); });
  const sh = document.createElement('label');
  sh.innerHTML = '<input type="checkbox" id="shd" checked> shadow bands (no reading beside near edges)';
  ctl.appendChild(sh);
  $('#shd').addEventListener('change', e => { st.shadow = e.target.checked; draw(); });
  sel.addEventListener('change', () => {
    st.scene = sel.value; st.noise = null; noise.value = byName[st.scene].noise_mm; noise.dispatchEvent(new Event('input')); st.noise = null; draw();
  });

  let depth = null;
  function draw() {
    const sc = byName[st.scene];
    const opts = { noise_mm: st.noise ?? sc.noise_mm, shadow_px: st.shadow ? sc.shadow_px : 0 };
    depth = sceneDepth(sc, opts);
    const gray = paintDepth(cv, depth, W, H, { lut: lutFor(luts, st.cmap), validOnly: st.validOnly, holesBlack: st.validOnly });
    $('#about').textContent = sc.about;
    $('#bar').innerHTML = colorbar(lutFor(luts, st.cmap), gray.lo, gray.hi);
    const s = depthStats(depth, W, H);
    const isDefault = opts.noise_mm === sc.noise_mm && opts.shadow_px === sc.shadow_px;
    const agree = ['min_mm', 'median_mm', 'max_mm', 'valid_fraction'].every(k => Math.abs(s[k] - sc.stats[k]) < 1e-9);
    $('#stats').innerHTML = `<dt>with a reading</dt><dd>${(100 * s.valid_fraction).toFixed(1)} %</dd>
      <dt>nearest</dt><dd>${mm(s.min_mm)}</dd><dt>median</dt><dd>${mm(s.median_mm)}</dd><dt>farthest</dt><dd>${mm(s.max_mm)}</dd>
      <dt>Python check</dt><dd>${isDefault ? (agree ? '<span class="ok">matches kinect_depth</span>' : '<span class="err">differs</span>') : 'default settings only'}</dd>`;
    $('#note').innerHTML = st.validOnly
      ? 'Range taken over pixels with a reading; holes are drawn black. This is an option of this page, not of the scripts.'
      : s.valid_fraction < 1
        ? `As in the scripts, the 0-valued holes set the bottom of the range (0 mm), so the holes take the bottom colour of the map and the scene (${mm(s.min_mm)} and beyond) is squeezed into the upper part of the scale.`
        : 'No holes in this frame, so the whole-frame range is just the scene\'s nearest to farthest depth.';
  }
  cv.addEventListener('pointermove', e => {
    if (!depth) return;
    const [u, v] = pixelAt(cv, e, W, H), p = probe(depth, W, H, u, v, F);
    $('#hud').textContent = p.z > 0 ? `(${u}, ${v})  z ${p.z} mm  x ${Math.round(p.x)}  y ${Math.round(p.y)}` : `(${u}, ${v})  no reading`;
  });
  draw();
});
