// Point-cloud page: the depth frame back-projected through the pinhole model (f = 525 px, principal point at the image
// centre, slamMapping.py) and drawn in an orbiting perspective view, coloured by depth.
import { $, animate, app, esc, route } from './core.js';
import { CMAP_NAMES, loadAll, lutFor, sceneDepth } from './data.js';
import { depthToPoints, normalizeMinmax } from './depth_engine.js';
import { CloudView, seg } from './render.js';

route('cloud', async (args, query) => {
  const { info, scenes, luts, byName } = await loadAll();
  const W = info.frame.width, H = info.frame.height, F = info.intrinsics.f_px;
  const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const st = { scene: byName[query.scene] ? query.scene : 'room', cmap: 'jet', stride: 2, spin: !still };
  app().innerHTML = `
  <h1>Point cloud</h1>
  <p class="lead">Each pixel (u, v) with depth z becomes a 3D point in millimetres, camera frame (x right, y down):</p>
  <p><span class="eq">x = (u − ${info.intrinsics.cx}) · z / ${F}</span> <span class="eq">y = (v − ${info.intrinsics.cy}) · z / ${F}</span>
    <span class="eq">z = depth</span></p>
  <div class="side">
    <div class="panel" id="ctl">
      <label for="sc">Scene</label>
      <select id="sc">${scenes.map(s => `<option value="${esc(s.name)}">${esc(s.title)}</option>`).join('')}</select>
      <label><input type="checkbox" id="spin" ${st.spin ? 'checked' : ''}> auto-rotate</label>
      <dl class="kv" id="cnt"></dl>
      <p class="small mute">Drag (or focus and use the arrow keys) to orbit; wheel or +/− to zoom. Pixels without a
        reading are dropped, the fix made to slamMapping.py in this branch.</p>
    </div>
    <div class="stage"><canvas id="cv" role="img" aria-label="Orbiting 3D point cloud of the depth scene"></canvas></div>
  </div>`;
  const ctl = $('#ctl'), sel = $('#sc');
  sel.value = st.scene;
  seg(ctl, 'Colour by depth', Object.entries(CMAP_NAMES).map(([k, t]) => [k, t.split(' (')[0]]), st.cmap, v => { st.cmap = v; build(); });
  seg(ctl, 'Every n-th pixel', [['1', '1'], ['2', '2'], ['4', '4']], String(st.stride), v => { st.stride = +v; build(); });
  sel.addEventListener('change', () => { st.scene = sel.value; build(); });
  $('#spin').addEventListener('change', e => { st.spin = e.target.checked; });
  const view = new CloudView($('#cv'), { w: 720, h: 540 });

  function build() {
    const depth = sceneDepth(byName[st.scene]);
    const { xyz, index } = depthToPoints(depth, W, H, F, st.stride);
    const gray = normalizeMinmax(depth, true), lut = lutFor(luts, st.cmap), rgb = new Uint8Array(index.length * 3);
    index.forEach((i, p) => { const c = lut ? lut[gray[i]] : [gray[i], gray[i], gray[i]]; rgb.set(c, 3 * p); });
    view.setPoints(xyz, rgb); view.draw();
    const all = depthToPoints(depth, W, H, F, 1).index.length;
    $('#cnt').innerHTML = `<dt>points</dt><dd>${all.toLocaleString()} of ${(W * H).toLocaleString()}</dd><dt>drawn</dt><dd>${index.length.toLocaleString()}</dd>`;
  }
  build();
  let last = 0;
  animate(t => { if (st.spin && t - last > 40) { last = t; view.yaw += 0.006; view.draw(); } });
});
