// Depth-frame maths, ported from kinect_depth/processing.py (pure functions, no DOM; node-tested against Python in
// depth_engine.test.mjs). Depth frames are flat row-major arrays of millimetres, 0 = no reading.

export const FRAME_W = 640, FRAME_H = 480, FOCAL_PX = 525;

// numpy.rint / C lrint: round half to even.
export function rint(x) {
  const f = Math.floor(x), d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

// cv2.normalize(depth, None, 0, 255, NORM_MINMAX, CV_8U); validOnly takes the range over depth > 0 (holes stay 0).
export function normalizeMinmax(depth, validOnly = false) {
  const n = depth.length, out = new Uint8Array(n);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    const v = depth[i];
    if (validOnly && !(v > 0)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo === Infinity) { out.lo = out.hi = 0; return out; }
  const scale = hi - lo > Number.EPSILON ? 255 / (hi - lo) : 0, shift = 0 - lo * scale;
  for (let i = 0; i < n; i++) {
    const v = depth[i];
    if (validOnly && !(v > 0)) continue;
    out[i] = Math.min(255, Math.max(0, rint(v * scale + shift)));
  }
  out.lo = lo; out.hi = hi;  // range used, for colour-bar labels
  return out;
}

// RGBA pixels of cv2.applyColorMap(gray, lut) (lut: 256 [r,g,b] rows, or null for grey). Pixels where hole[i] is
// true are painted holeRGB (display choice: the scripts colour holes like the nearest depth).
export function colorize(gray, lut, rgba = new Uint8ClampedArray(gray.length * 4), hole = null, holeRGB = [0, 0, 0]) {
  for (let i = 0, j = 0; i < gray.length; i++, j += 4) {
    const c = hole && hole[i] ? holeRGB : lut ? lut[gray[i]] : [gray[i], gray[i], gray[i]];
    rgba[j] = c[0]; rgba[j + 1] = c[1]; rgba[j + 2] = c[2]; rgba[j + 3] = 255;
  }
  return rgba;
}

// Pinhole back-projection (slamMapping.py, fixed): x = (u - w/2) z / f, y = (v - h/2) z / f for pixels with z > 0.
// Returns { xyz: Float64Array(3N), index: Int32Array(N) } with index = flat pixel index of each point. stride > 1
// samples every stride-th pixel in both directions (for drawing).
export function depthToPoints(depth, w, h, f = FOCAL_PX, stride = 1) {
  const xyz = [], index = [];
  for (let v = 0; v < h; v += stride) {
    for (let u = 0; u < w; u += stride) {
      const i = v * w + u, z = depth[i];
      if (!(z > 0) || !Number.isFinite(z)) continue;
      xyz.push((u - w / 2) * z / f, (v - h / 2) * z / f, z);
      index.push(i);
    }
  }
  return { xyz: Float64Array.from(xyz), index: Int32Array.from(index) };
}

// Window of cv2.bilateralFilter: disc of radius d // 2 (or round(1.5 sigmaSpace) if d <= 0), spatial weights.
export function bilateralKernel(d = 7, sigmaSpace = 100) {
  let radius = d > 0 ? Math.floor(d / 2) : Math.round(sigmaSpace * 1.5);
  radius = Math.max(radius, 1);
  const offs = [], wts = [];
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const r2 = dy * dy + dx * dx;
      if (Math.sqrt(r2) > radius) continue;
      offs.push([dy, dx]); wts.push(Math.exp(-0.5 * r2 / (sigmaSpace * sigmaSpace)));
    }
  }
  return { radius, offs, wts };
}

const reflect101 = (i, n) => { while (i < 0 || i >= n) i = i < 0 ? -i : 2 * n - 2 - i; return i; };

// Bilateral filter with mirrored borders (BORDER_REFLECT_101), as kinect_depth.processing.bilateral_filter.
export function bilateralFilter(img, w, h, d = 7, sigmaColor = 100, sigmaSpace = 100) {
  const { offs, wts } = bilateralKernel(d, sigmaSpace), out = new Float64Array(w * h);
  const gc = -0.5 / (sigmaColor * sigmaColor), k = offs.length;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const c = img[y * w + x];
      let num = 0, den = 0;
      for (let o = 0; o < k; o++) {
        const yy = reflect101(y + offs[o][0], h), xx = reflect101(x + offs[o][1], w), q = img[yy * w + xx];
        const wgt = wts[o] * Math.exp((q - c) * (q - c) * gc);
        num += wgt * q; den += wgt;
      }
      out[y * w + x] = num / den;
    }
  }
  return out;
}

// kinect_depth.processing.depth_stats: valid fraction and min / median / max / mean of pixels with a reading.
export function depthStats(depth, w, h) {
  const v = Float64Array.from(depth.filter(x => x > 0)).sort();
  const out = { width: w, height: h, valid_fraction: v.length / depth.length };
  if (v.length) {
    const m = v.length >> 1, sum = v.reduce((a, b) => a + b, 0);
    Object.assign(out, { min_mm: v[0], max_mm: v[v.length - 1], mean_mm: sum / v.length,
      median_mm: v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2 });
  }
  return out;
}

// Depth at pixel (u, v) and its back-projected point, for hover read-outs.
export function probe(depth, w, h, u, v, f = FOCAL_PX) {
  const z = depth[v * w + u];
  return { u, v, z, x: z > 0 ? (u - w / 2) * z / f : null, y: z > 0 ? (v - h / 2) * z / f : null };
}
