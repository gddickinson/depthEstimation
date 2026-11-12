// Synthetic depth scenes, ported from kinect_depth/scenes.py: ray-cast planes, boxes and spheres through the Kinect
// pinhole model (f = 525 px, principal point at the image centre). Scene definitions come from /api/scenes.
// Noise and shadow bands are illustrative, not a calibrated sensor model (see scenes.py).

import { FOCAL_PX, FRAME_H, FRAME_W } from './depth_engine.js';

// lowbias32 integer hash (same bit operations as scenes._hash32).
export function hash32(x) {
  x >>>= 0;
  x = (x ^ (x >>> 16)) >>> 0;
  x = Math.imul(x, 0x7feb352d) >>> 0;
  x = (x ^ (x >>> 15)) >>> 0;
  x = Math.imul(x, 0x846ca68b) >>> 0;
  return (x ^ (x >>> 16)) >>> 0;
}

// Depth z of the first hit of ray (dx, dy, 1) with one primitive, or Infinity.
export function hit(item, dx, dy) {
  if (item.kind === 'plane') {
    const p = item.point, n = item.normal;
    const den = n[0] * dx + n[1] * dy + n[2], num = n[0] * p[0] + n[1] * p[1] + n[2] * p[2];
    const t = num / den;
    return den !== 0 && t > 0 ? t : Infinity;
  }
  if (item.kind === 'sphere') {
    const c = item.center, r = item.radius;
    const a = dx * dx + dy * dy + 1.0, b = -2.0 * (dx * c[0] + dy * c[1] + c[2]);
    const cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - r * r, disc = b * b - 4.0 * a * cc;
    const t = (-b - Math.sqrt(Math.max(disc, 0.0))) / (2.0 * a);
    return disc >= 0 && t > 0 ? t : Infinity;
  }
  const lo = item.min, hi = item.max, dir = [dx, dy, 1];
  let tn = -Infinity, tf = Infinity;
  for (let k = 0; k < 3; k++) {
    let t1, t2;
    if (k === 2) { t1 = lo[2]; t2 = hi[2]; }
    else if (dir[k] === 0) { const inside = lo[k] <= 0 && 0 <= hi[k]; t1 = inside ? -Infinity : Infinity; t2 = Infinity; }
    else { t1 = lo[k] / dir[k]; t2 = hi[k] / dir[k]; }
    tn = Math.max(tn, Math.min(t1, t2));
    tf = Math.min(tf, Math.max(t1, t2));
  }
  return tn <= tf && tn > 0 ? tn : Infinity;
}

// Uint16Array depth image (mm, 0 = no reading); opts default to the scene's own noise_mm / shadow_px.
export function renderScene(scene, { noise_mm = scene.noise_mm, shadow_px = scene.shadow_px, seed = 1,
  w = FRAME_W, h = FRAME_H, f = FOCAL_PX } = {}) {
  const n = w * h, z = new Float64Array(n);
  for (let v = 0; v < h; v++) {
    const dy = (v - h / 2) / f;
    for (let u = 0; u < w; u++) {
      const dx = (u - w / 2) / f;
      let best = Infinity;
      for (const it of scene.items) best = Math.min(best, hit(it, dx, dy));
      z[v * w + u] = Number.isFinite(best) ? best : 0;
    }
  }
  const hole = new Uint8Array(n);
  for (let v = 0; v < h; v++) {
    for (let u = 0; u < w; u++) {
      const i = v * w + u;
      for (let s = 1; s <= shadow_px && s <= u; s++) {
        const l = z[i - s];
        if (l > 0 && l < z[i] - 150) { hole[i] = 1; break; }
      }
    }
  }
  const base = Math.imul(seed, 2654435761 | 0) >>> 0, out = new Uint16Array(n);
  for (let i = 0; i < n; i++) {
    let zi = z[i];
    if (noise_mm > 0 && zi > 0) {
      const idx = i * 3 + base;
      let g = 0;
      for (let k = 0; k < 3; k++) g += hash32((idx + k) % 4294967296) / 4294967296.0;
      zi = zi + noise_mm * ((zi / 1000.0) * (zi / 1000.0)) * (g - 1.5) * 2.0;
    }
    out[i] = hole[i] ? 0 : Math.min(65535, Math.max(0, Math.floor(zi + 0.5)));
  }
  return out;
}
