// Static-site data layer: answers api() calls from the JSON files exported by scripts/build_static.py.
// /info, /scenes and /colormaps were computed by the Python server at build time. The /device/* routes talk to a
// Kinect on the machine running the server and are not available here (the Live page says so).
const files = new Map();
async function file(rel) {
  if (!files.has(rel)) files.set(rel, fetch('api/' + rel + '.json').then(r => { if (!r.ok) throw new Error(`${r.status} ${rel}`); return r.json(); }));
  return files.get(rel);
}

export async function staticApi(path, body) {
  const rel = path.split('?')[0].replace(/^\/+/, '');
  if (body || rel.startsWith('device')) throw new Error('The live Kinect needs the local app (Python server).');
  return file(rel);
}
