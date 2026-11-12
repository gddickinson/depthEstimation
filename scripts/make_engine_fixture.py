"""Write tests/fixtures/engine_fixture.json: values computed by kinect_depth (Python) that the JS engines must match.

web/static/depth_engine.test.mjs recomputes everything in JS and compares; tests/test_engine_fixture.py checks the
fixture is still what the Python code produces. Run after changing kinect_depth: python scripts/make_engine_fixture.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kinect_depth import processing as P  # noqa: E402
from kinect_depth import scenes as S  # noqa: E402

OUT = ROOT / "tests" / "fixtures" / "engine_fixture.json"
SAMPLE = 997  # every SAMPLE-th pixel is stored
CROP = (200, 260, 150, 250)  # rows v0:v1, cols u0:u1 of the 'steps' scene for the bilateral filter


def build():
    fx = {"scenes": {}, "sample_every": SAMPLE, "crop": CROP}
    for name in S.SCENES:
        d = S.render_scene(name)
        flat = d.ravel()
        fx["scenes"][name] = {"def": S.SCENES[name], "sum": int(flat.astype(np.int64).sum()),
                              "samples": flat[::SAMPLE].tolist(), "stats": P.depth_stats(d)}
    seeded = S.render_scene("room", noise_mm=8.0, shadow_px=3, seed=7)
    fx["room_seed7"] = {"sum": int(seeded.astype(np.int64).sum()), "samples": seeded.ravel()[::SAMPLE].tolist()}
    room = S.render_scene("room")
    fx["normalize"] = {"all": P.normalize_minmax(room).ravel()[::SAMPLE].tolist(),
                       "valid": P.normalize_minmax(room, valid_only=True).ravel()[::SAMPLE].tolist()}
    pts, valid = P.depth_to_points(room)
    fx["points"] = {"n": int(len(pts)), "sum": pts.sum(axis=0).tolist(), "every": pts[::SAMPLE].tolist(),
                    "valid_sum": int(np.flatnonzero(valid).astype(np.int64).sum())}
    v0, v1, u0, u1 = CROP
    steps = S.render_scene("steps")[v0:v1, u0:u1].astype(np.float64)
    fx["bilateral"] = {"d": 7, "sigma_color": 100.0, "sigma_space": 100.0,
                       "out": P.bilateral_filter(steps, 7, 100.0, 100.0).ravel()[::7].tolist(),
                       "out_d9_s30_s3": P.bilateral_filter(steps, 9, 30.0, 3.0).ravel()[::7].tolist()}
    return fx


def main():
    OUT.write_text(json.dumps(build(), separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
