// Canvas drawing for the depth pages: colourised depth images, colour bars, an orbiting point-cloud view and a
// simple line chart. All the maths lives in depth_engine.js; this module only draws.
import { esc } from './core.js';
import { colorize, normalizeMinmax } from './depth_engine.js';

// Draw a depth frame with min-max normalisation and a colour map. Returns the gray image (with .lo/.hi range).
export function paintDepth(canvas, depth, w, h, { lut = null, validOnly = false, holesBlack = true } = {}) {
  canvas.width = w; canvas.height = h;
  const gray = normalizeMinmax(depth, validOnly), ctx = canvas.getContext('2d'), img = ctx.createImageData(w, h);
  const hole = holesBlack ? Uint8Array.from(depth, v => (v > 0 ? 0 : 1)) : null;
  colorize(gray, lut, img.data, hole);
  ctx.putImageData(img, 0, 0);
  return gray;
}

// Draw an interleaved RGB byte frame.
export function paintRGB(canvas, rgb, w, h) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d'), img = ctx.createImageData(w, h);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) { img.data[j] = rgb[3 * i]; img.data[j + 1] = rgb[3 * i + 1]; img.data[j + 2] = rgb[3 * i + 2]; img.data[j + 3] = 255; }
  ctx.putImageData(img, 0, 0);
}

export function colorbar(lut, lo, hi, label = 'mm') {
  const stops = lut ? [0, 32, 64, 96, 128, 160, 192, 224, 255].map(i => `rgb(${lut[i].join(',')}) ${(i / 255 * 100).toFixed(1)}%`).join(',')
    : '#000 0%, #fff 100%';
  return `<div class="cbar" style="background:linear-gradient(90deg,${stops})" role="img" aria-label="colour scale ${esc(lo)} to ${esc(hi)} ${esc(label)}"></div>
    <div class="cbar-l"><span>${Math.round(lo)} ${esc(label)}</span><span>${Math.round(hi)} ${esc(label)}</span></div>`;
}

// Orbiting perspective view of a point cloud (camera-frame mm, y down). setPoints(xyz, rgb) then draw().
export class CloudView {
  constructor(canvas, { w = 640, h = 480 } = {}) {
    this.c = canvas; this.c.width = w; this.c.height = h; this.ctx = canvas.getContext('2d');
    this.yaw = -0.5; this.pitch = 0.25; this.zoom = 1; this.xyz = new Float64Array(0); this.rgb = new Uint8Array(0);
    let drag = null;
    canvas.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', e => {
      if (!drag) return;
      this.yaw += (e.clientX - drag[0]) * 0.01; this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + (e.clientY - drag[1]) * 0.01));
      drag = [e.clientX, e.clientY]; this.draw();
    });
    canvas.addEventListener('pointerup', () => { drag = null; });
    canvas.addEventListener('wheel', e => { e.preventDefault(); this.zoom = Math.max(0.3, Math.min(4, this.zoom * (e.deltaY > 0 ? 0.9 : 1.1))); this.draw(); }, { passive: false });
    canvas.tabIndex = 0;
    canvas.addEventListener('keydown', e => {
      const k = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, -0.1], ArrowDown: [0, 0.1] }[e.key];
      if (k) { e.preventDefault(); this.yaw += k[0]; this.pitch = Math.max(-1.4, Math.min(1.4, this.pitch + k[1])); this.draw(); }
      if (e.key === '+' || e.key === '-') { this.zoom *= e.key === '+' ? 1.1 : 0.9; this.draw(); }
    });
  }

  setPoints(xyz, rgb) {
    this.xyz = xyz; this.rgb = rgb;
    const n = xyz.length / 3, c = [0, 0, 0];
    for (let i = 0; i < xyz.length; i++) c[i % 3] += xyz[i];
    this.center = n ? c.map(v => v / n) : [0, 0, 2000];
    let r = 1;
    for (let i = 0; i < xyz.length; i += 3) r = Math.max(r, Math.hypot(xyz[i] - this.center[0], xyz[i + 1] - this.center[1], xyz[i + 2] - this.center[2]));
    this.radius = r;
  }

  draw() {
    const { ctx, c } = this, W = c.width, H = c.height, img = ctx.createImageData(W, H), zb = new Float32Array(W * H).fill(Infinity);
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const dist = this.radius * 2.6 / this.zoom, f = Math.min(W, H) * 1.25, [mx, my, mz] = this.center || [0, 0, 0];
    for (let j = 3; j < img.data.length; j += 4) img.data[j] = 255;
    for (let i = 0, p = 0; i < this.xyz.length; i += 3, p++) {
      const x = this.xyz[i] - mx, y = this.xyz[i + 1] - my, z = this.xyz[i + 2] - mz;
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;           // yaw about the vertical (y) axis
      const y2 = cp * y - sp * z1, z2 = sp * y + cp * z1 + dist;     // pitch about x, then move in front of the eye
      if (z2 <= 1) continue;
      const u = Math.round(W / 2 + f * x1 / z2), v = Math.round(H / 2 + f * y2 / z2);
      for (let dv = 0; dv < 2; dv++) for (let du = 0; du < 2; du++) {
        const uu = u + du, vv = v + dv;
        if (uu < 0 || vv < 0 || uu >= W || vv >= H) continue;
        const k = vv * W + uu;
        if (z2 >= zb[k]) continue;
        zb[k] = z2; img.data[4 * k] = this.rgb[3 * p]; img.data[4 * k + 1] = this.rgb[3 * p + 1]; img.data[4 * k + 2] = this.rgb[3 * p + 2];
      }
    }
    ctx.putImageData(img, 0, 0);
  }
}

// Line chart of several series [{ values, color, label }] sharing an x index; null values break the line.
export function lineChart(canvas, series, { ylabel = 'mm' } = {}) {
  const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth || 600, H = canvas.clientHeight || 200;
  canvas.width = W * dpr; canvas.height = H * dpr;
  const ctx = canvas.getContext('2d'); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, W, H);
  const all = series.flatMap(s => Array.from(s.values).filter(v => v != null && v > 0));
  if (!all.length) return;
  let lo = Math.min(...all), hi = Math.max(...all); const pad = (hi - lo) * 0.05 || 10; lo -= pad; hi += pad;
  const L = 52, R = 8, T = 8, B = 20, n = series[0].values.length;
  const X = i => L + (W - L - R) * i / Math.max(1, n - 1), Y = v => T + (H - T - B) * (1 - (v - lo) / (hi - lo));
  ctx.strokeStyle = '#212c4c'; ctx.fillStyle = '#94a3c6'; ctx.font = '11px ui-monospace, Menlo, monospace';
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(L, y); ctx.lineTo(W - R, y); ctx.stroke(); ctx.fillText(Math.round(v), 4, y + 4);
  }
  ctx.fillText(ylabel, 4, H - 6); ctx.fillText('pixel column', W - 90, H - 6);
  for (const s of series) {
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width || 1.5; ctx.beginPath(); let pen = false;
    s.values.forEach((v, i) => { if (v == null || !(v > 0)) { pen = false; return; } pen ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)); pen = true; });
    ctx.stroke();
  }
}

// Segmented button group; onPick(value) on click. Returns the container element.
export function seg(host, label, options, value, onPick) {
  const wrap = document.createElement('div');
  wrap.className = 'ctl';
  wrap.innerHTML = `<label>${esc(label)}</label><div class="seg" role="group" aria-label="${esc(label)}">${options.map(([v, t]) =>
    `<button type="button" data-v="${esc(v)}" class="${v === value ? 'on' : ''}" aria-pressed="${v === value}">${esc(t)}</button>`).join('')}</div>`;
  wrap.querySelectorAll('button').forEach(b => b.addEventListener('click', () => {
    wrap.querySelectorAll('button').forEach(x => { x.classList.toggle('on', x === b); x.setAttribute('aria-pressed', x === b); });
    onPick(b.dataset.v);
  }));
  host.appendChild(wrap);
  return wrap;
}

// Pixel coordinates in a canvas drawn at w x h from a pointer event.
export function pixelAt(canvas, e, w, h) {
  const r = canvas.getBoundingClientRect();
  return [Math.min(w - 1, Math.max(0, Math.floor((e.clientX - r.left) / r.width * w))),
    Math.min(h - 1, Math.max(0, Math.floor((e.clientY - r.top) / r.height * h)))];
}
