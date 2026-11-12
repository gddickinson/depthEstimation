// Shared page data: scene definitions and colour maps from the API, rendered scenes cached by their parameters.
import { api } from './core.js';
import { renderScene } from './scene_engine.js';

export async function loadAll() {
  const [info, scenes, luts] = await Promise.all([api('/info'), api('/scenes'), api('/colormaps')]);
  return { info, scenes, luts, byName: Object.fromEntries(scenes.map(s => [s.name, s])) };
}

const cache = new Map();
export function sceneDepth(scene, opts = {}) {
  const key = JSON.stringify([scene.name, opts]);
  if (!cache.has(key)) { if (cache.size > 12) cache.clear(); cache.set(key, renderScene(scene, opts)); }
  return cache.get(key);
}

export const lutFor = (luts, name) => (name === 'gray' ? null : luts[name]);
export const CMAP_NAMES = { jet: 'Jet (quickTest.py, test_kinect.py)', viridis: 'Viridis (depthEstimation.py)', gray: 'Grey (pyqtgraph views)' };
export const mm = v => (v == null ? '-' : `${Math.round(v)} mm`);
