"""FastAPI web app for depthEstimation: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8155, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The
/api/device/* routes need a Kinect and are local only; the static site shows a "needs the local app" notice instead.
"""
import base64
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from kinect_depth import processing as P
from kinect_depth import scenes as S
from web.device import KinectSource

STATIC = Path(__file__).resolve().parent / "static"

source = KinectSource()


@asynccontextmanager
async def lifespan(_app):
    yield
    source.close()  # release the Kinect when the server stops


app = FastAPI(title="Kinect Depth Lab", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC), name="static")

# depthEstimation.py, apply_edge_aware_filter: cv2.bilateralFilter(depth, d=7, sigmaColor=100, sigmaSpace=100)
FILTER_DEFAULTS = {"d": 7, "sigma_color": 100.0, "sigma_space": 100.0}


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Frame format, intrinsics and filter settings taken from the repo's code, for the home and about pages."""
    luts = P.load_colormaps()
    return {
        "title": "Kinect Depth Lab",
        "frame": {"width": P.FRAME_W, "height": P.FRAME_H, "depth_unit": "mm", "no_reading": 0,
                  "mode": "FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_MM / FREENECT_VIDEO_RGB"},
        "intrinsics": {"f_px": P.FOCAL_PX, "cx": P.FRAME_W / 2, "cy": P.FRAME_H / 2, "source": "slamMapping.py"},
        "filter": FILTER_DEFAULTS,
        "colormaps": sorted(luts) + ["gray"],
        "scenes": len(S.SCENES),
    }


@app.get("/api/scenes")
def scenes():
    """Synthetic scene definitions (rendered in the browser by scene_engine.js) with Python-computed statistics."""
    out = []
    for item in S.scene_list():
        sc = S.SCENES[item["name"]]
        depth = S.render_scene(item["name"])
        out.append({**item, "items": sc["items"], "stats": P.depth_stats(depth),
                    "points": int((depth > 0).sum())})
    return out


@app.get("/api/colormaps")
def colormaps():
    """OpenCV colour maps as 256 x RGB tables (kinect_depth/colormap_luts.json)."""
    return {k: v.tolist() for k, v in P.load_colormaps().items()}


@app.get("/api/device/status")
def device_status():
    return source.status()


@app.post("/api/device/connect")
def device_connect():
    return source.connect()


@app.post("/api/device/disconnect")
def device_disconnect():
    source.close()
    return source.status()


@app.get("/api/device/frame")
def device_frame(step: int = Query(1, ge=1, le=8)):
    """Next Kinect frame: RGB (uint8) and depth (uint16 little-endian, mm) as base64, plus Python frame statistics.

    step > 1 keeps every step-th pixel in both directions (smaller transfers); the point cloud then uses f / step.
    """
    got = source.frame()
    if got is None:
        return {"ok": False, **source.status()}
    rgb, depth = got
    rgb, depth = np.ascontiguousarray(rgb[::step, ::step]), np.ascontiguousarray(depth[::step, ::step])
    return {
        "ok": True, "width": int(depth.shape[1]), "height": int(depth.shape[0]), "step": step,
        "f_px": P.FOCAL_PX / step, "frames": source.frames,
        "rgb": base64.b64encode(rgb.astype(np.uint8).tobytes()).decode(),
        "depth": base64.b64encode(depth.astype("<u2").tobytes()).decode(),
        "stats": P.depth_stats(depth),
    }


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/scenes", "/colormaps"]
