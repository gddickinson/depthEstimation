// Node test: the JS engines reproduce kinect_depth (Python) on the values in tests/fixtures/engine_fixture.json.
// Run: node web/static/depth_engine.test.mjs   (regenerate the fixture with scripts/make_engine_fixture.py)
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { bilateralFilter, colorize, depthStats, depthToPoints, normalizeMinmax, rint } from './depth_engine.js';
import { hash32, renderScene } from './scene_engine.js';

const here = new URL('.', import.meta.url);
const fx = JSON.parse(readFileSync(new URL('../../tests/fixtures/engine_fixture.json', here)));
const luts = JSON.parse(readFileSync(new URL('../../kinect_depth/colormap_luts.json', here)));
const every = (a, k) => Array.from(a).filter((_, i) => i % k === 0);
const close = (a, b, tol, what) => {
  assert.equal(a.length, b.length, what + ' length');
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= tol * Math.max(1, Math.abs(b[i])), `${what}[${i}] ${a[i]} vs ${b[i]}`);
};
let n = 0;
const test = (name, fn) => { fn(); n++; console.log('ok', name); };

test('rint rounds half to even', () => assert.deepEqual([0.5, 1.5, 2.5, 2.4, 2.6].map(rint), [0, 2, 2, 2, 3]));
test('hash32 is a 32-bit unsigned hash', () => { const h = hash32(12345); assert.ok(h >= 0 && h < 2 ** 32 && Number.isInteger(h)); });

const W = 640, H = 480, renders = {};
for (const [name, want] of Object.entries(fx.scenes)) {
  test(`scene ${name} matches Python`, () => {
    const d = renders[name] = renderScene(want.def);
    let sum = 0; for (const v of d) sum += v;
    const s = every(d, fx.sample_every), mism = s.filter((v, i) => v !== want.samples[i]).length;
    assert.ok(mism <= 1 && Math.abs(sum - want.sum) <= 2, `${name}: ${mism} mismatches, sum ${sum} vs ${want.sum}`);
    const st = depthStats(d, W, H);
    for (const k of Object.keys(want.stats)) close([st[k]], [want.stats[k]], 1e-6, `${name} stats.${k}`);
  });
}
test('seeded noise variant matches Python', () => {
  const d = renderScene(fx.scenes.room.def, { noise_mm: 8, shadow_px: 3, seed: 7 });
  let sum = 0; for (const v of d) sum += v;
  assert.ok(Math.abs(sum - fx.room_seed7.sum) <= 2);
  close(every(d, fx.sample_every), fx.room_seed7.samples, 0, 'seed7');
});
test('min-max normalisation matches Python (both modes)', () => {
  close(every(normalizeMinmax(renders.room), fx.sample_every), fx.normalize.all, 0, 'all');
  close(every(normalizeMinmax(renders.room, true), fx.sample_every), fx.normalize.valid, 0, 'valid');
});
test('colormap uses the OpenCV table', () => {
  const g = Uint8Array.from([0, 128, 255]), px = colorize(g, luts.jet);
  assert.deepEqual(Array.from(px.slice(4, 7)), luts.jet[128]);
  assert.deepEqual(Array.from(colorize(g, null).slice(8, 11)), [255, 255, 255]);
});
test('point cloud matches Python', () => {
  const { xyz, index } = depthToPoints(renders.room, W, H);
  assert.equal(index.length, fx.points.n);
  assert.equal(index.reduce((a, b) => a + b, 0), fx.points.valid_sum);
  const sum = [0, 0, 0]; for (let i = 0; i < xyz.length; i++) sum[i % 3] += xyz[i];
  close(sum, fx.points.sum, 1e-9, 'sum');
  const ev = []; for (let i = 0; i < index.length; i += fx.sample_every) ev.push(...xyz.slice(3 * i, 3 * i + 3));
  close(ev, fx.points.every.flat(), 1e-12, 'every');
});
test('bilateral filter matches Python', () => {
  const [v0, v1, u0, u1] = fx.crop, w = u1 - u0, h = v1 - v0, crop = new Float64Array(w * h);
  for (let v = 0; v < h; v++) for (let u = 0; u < w; u++) crop[v * w + u] = renders.steps[(v + v0) * W + u + u0];
  close(every(bilateralFilter(crop, w, h, 7, 100, 100), 7), fx.bilateral.out, 1e-9, 'd7');
  close(every(bilateralFilter(crop, w, h, 9, 30, 3), 7), fx.bilateral.out_d9_s30_s3, 1e-9, 'd9');
});
console.log(`${n} tests passed`);
