// Filter page: the edge-aware bilateral filter of depthEstimation.py (cv2.bilateralFilter, d=7, sigmas 100) on a
// noisy synthetic scene, compared with the noise-free scene along one image row.
import { $, app, esc, route, slider } from './core.js';
import { loadAll, mm, sceneDepth } from './data.js';
import { bilateralFilter } from './depth_engine.js';
import { lineChart, paintDepth, seg } from './render.js';

route('filter', async (args, query) => {
  const { info, scenes, luts, byName } = await loadAll();
  const W = info.frame.width, H = info.frame.height, def = info.filter;
  const st = { scene: byName[query.scene] ? query.scene : 'steps', d: def.d, sc: def.sigma_color, ss: def.sigma_space, noise: null, row: 240 };
  app().innerHTML = `
  <h1>Bilateral filter</h1>
  <p class="lead">Each pixel becomes a weighted mean of its neighbours in a disc of diameter d. The weight falls with
    distance (σ<sub>space</sub>, pixels) and with depth difference (σ<sub>colour</sub>, here millimetres), so noise is
    smoothed but steps between objects are kept. Defaults: d = ${def.d}, σ = ${def.sigma_color} (depthEstimation.py's original call).</p>
  <p><span class="eq">w = exp(−|q−p|² / 2σ<sub>s</sub>²) · exp(−(D<sub>q</sub>−D<sub>p</sub>)² / 2σ<sub>c</sub>²)</span></p>
  <div class="side">
    <div class="panel" id="ctl">
      <label for="sc">Scene</label>
      <select id="sc">${scenes.map(s => `<option value="${esc(s.name)}">${esc(s.title)}</option>`).join('')}</select>
    </div>
    <div>
      <div class="pair">
        <figure class="stage"><canvas id="a" class="pix" role="img" aria-label="Noisy depth"></canvas><figcaption>&nbsp;input (noisy)</figcaption></figure>
        <figure class="stage"><canvas id="b" class="pix" role="img" aria-label="Filtered depth"></canvas><figcaption>&nbsp;filtered</figcaption></figure>
      </div>
      <h3>Row profile</h3>
      <canvas id="ch" class="chart" role="img" aria-label="Depth along one image row: truth, input and filtered"></canvas>
      <p class="legend"><i style="background:#ffffff"></i>noise-free scene<i style="background:#ff8a3d"></i>input<i style="background:#38e1ff"></i>filtered</p>
      <dl class="kv" id="err"></dl>
      <p class="small mute">depthEstimation.py runs the filter on its monocular model's output, a sigmoid in 0–1, so it
        now uses σ<sub>colour</sub> = 0.1 there (σ = 100 on that range weighed every neighbour equally: a plain Gaussian
        blur). On Kinect millimetres, σ<sub>colour</sub> = 100 keeps edges larger than a few hundred mm.</p>
    </div>
  </div>`;
  const ctl = $('#ctl'), sel = $('#sc');
  sel.value = st.scene;
  slider(ctl, { id: 'd', label: 'Diameter d', min: 3, max: 11, step: 2, value: st.d, unit: ' px' }, v => { st.d = v; run(); });
  slider(ctl, { id: 'sc', label: 'σ colour', min: 5, max: 400, step: 5, value: st.sc, unit: ' mm' }, v => { st.sc = v; run(); });
  slider(ctl, { id: 'ss', label: 'σ space', min: 1, max: 100, step: 1, value: st.ss, unit: ' px' }, v => { st.ss = v; run(); });
  const nz = slider(ctl, { id: 'nz', label: 'Noise scale', min: 0, max: 20, step: 1, value: byName[st.scene].noise_mm, unit: ' mm at 1 m' }, v => { st.noise = v; run(); });
  slider(ctl, { id: 'row', label: 'Profile row', min: 0, max: H - 1, value: st.row }, v => { st.row = v; plot(); });
  seg(ctl, 'Colour map', [['jet', 'Jet'], ['viridis', 'Viridis']], 'jet', v => { st.cmap = v; run(); });
  sel.addEventListener('change', () => { st.scene = sel.value; st.noise = null; nz.value = byName[st.scene].noise_mm; run(); });

  let noisy, truth, out, timer = 0;
  function run() { clearTimeout(timer); timer = setTimeout(compute, 60); }
  function compute() {
    const sc = byName[st.scene], noise = st.noise ?? sc.noise_mm;
    noisy = sceneDepth(sc, { noise_mm: noise }); truth = sceneDepth(sc, { noise_mm: 0 });
    out = bilateralFilter(Float64Array.from(noisy), W, H, st.d, st.sc, st.ss);
    const lut = luts[st.cmap || 'jet'];
    paintDepth($('#a'), noisy, W, H, { lut, validOnly: true });
    paintDepth($('#b'), Uint16Array.from(out, (v, i) => (noisy[i] > 0 ? Math.round(v) : 0)), W, H, { lut, validOnly: true });
    let e0 = 0, e1 = 0, n = 0;
    for (let i = 0; i < noisy.length; i++) if (noisy[i] > 0 && truth[i] > 0) { e0 += (noisy[i] - truth[i]) ** 2; e1 += (out[i] - truth[i]) ** 2; n++; }
    $('#err').innerHTML = `<dt>RMS error, input</dt><dd>${mm(Math.sqrt(e0 / n))}</dd><dt>RMS error, filtered</dt><dd>${mm(Math.sqrt(e1 / n))}</dd>
      <dt>note</dt><dd class="small">over pixels with a reading; holes (0) take part in the filter as in OpenCV</dd>`;
    plot();
  }
  function plot() {
    if (!out) return;
    const r = st.row * W, sl = (a, f = v => v) => Array.from(a.slice(r, r + W), f);
    lineChart($('#ch'), [
      { values: sl(noisy), color: '#ff8a3d', width: 1 },
      { values: sl(out).map((v, i) => (noisy[r + i] > 0 ? v : null)), color: '#38e1ff', width: 2 },
      { values: sl(truth), color: '#ffffff', width: 1 },
    ]);
  }
  compute();
});
