"""Synthetic depth scenes for exploring the depth maths without a Kinect (mirrored in web/static/scene_engine.js).

Each scene is a few planes, axis-aligned boxes and spheres in the camera frame (mm; x right, y down, z forward). Every
pixel's ray through the pinhole model of kinect_depth.processing (f = 525 px, principal point at the image centre) is
intersected with them and the nearest hit's z is the depth, as the Kinect's FREENECT_DEPTH_MM stream reports it.
Two optional, purely illustrative effects imitate what real frames look like; they are not a calibrated sensor model:
    noise_mm   depth noise whose spread grows with distance squared (noise_mm * (z / 1000 mm)^2, deterministic hash)
    shadow_px  pixels up to shadow_px right of a step of more than 150 mm towards the camera get no reading (0),
               like the shadow bands beside foreground objects in Kinect frames
"""
import numpy as np

from .processing import FOCAL_PX, FRAME_H, FRAME_W

SCENES = {
    "room": {
        "title": "Room corner",
        "about": "Floor, back and left walls, a box on the floor and a ball in front.",
        "noise_mm": 2.0, "shadow_px": 6,
        "items": [
            {"kind": "plane", "point": [0, 800, 0], "normal": [0, -1, 0]},
            {"kind": "plane", "point": [0, 0, 3500], "normal": [0, 0, -1]},
            {"kind": "plane", "point": [-1800, 0, 0], "normal": [1, 0, 0]},
            {"kind": "box", "min": [-900, 300, 2000], "max": [-300, 800, 2500]},
            {"kind": "sphere", "center": [450, 350, 1600], "radius": 300},
        ],
    },
    "steps": {
        "title": "Depth steps",
        "about": "Four flat panels at 1.2, 1.8, 2.4 and 3.0 m in front of a wall: sharp edges for the filter.",
        "noise_mm": 4.0, "shadow_px": 6,
        "items": [
            {"kind": "plane", "point": [0, 0, 3600], "normal": [0, 0, -1]},
            {"kind": "box", "min": [-760, -420, 1200], "max": [-366, 420, 1300]},
            {"kind": "box", "min": [-549, -630, 1800], "max": [0, 630, 1900]},
            {"kind": "box", "min": [0, -840, 2400], "max": [731, 840, 2500]},
            {"kind": "box", "min": [914, -1050, 3000], "max": [1900, 1050, 3100]},
        ],
    },
    "ramp": {
        "title": "Tilted wall",
        "about": "One plane turned about the vertical axis: depth rises smoothly from left to right.",
        "noise_mm": 0.0, "shadow_px": 0,
        "items": [{"kind": "plane", "point": [0, 0, 2200], "normal": [0.5, 0, -1]}],
    },
    "spheres": {
        "title": "Spheres",
        "about": "Five balls at different distances in front of a wall.",
        "noise_mm": 1.0, "shadow_px": 6,
        "items": [
            {"kind": "plane", "point": [0, 0, 4000], "normal": [0, 0, -1]},
            {"kind": "sphere", "center": [-700, -250, 1500], "radius": 250},
            {"kind": "sphere", "center": [0, 0, 2000], "radius": 450},
            {"kind": "sphere", "center": [800, 300, 1300], "radius": 200},
            {"kind": "sphere", "center": [-900, 600, 2800], "radius": 400},
            {"kind": "sphere", "center": [1100, -500, 3200], "radius": 500},
        ],
    },
}


def _hash32(x):
    """lowbias32 integer hash on uint32 arrays (same bit operations as hash32 in scene_engine.js)."""
    x = x.astype(np.uint32)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


def _hit(item, dx, dy):
    """Depth z of the first intersection of the rays (dx, dy, 1) with one primitive (inf where missed)."""
    inf = np.full(dx.shape, np.inf)
    if item["kind"] == "plane":
        p, n = item["point"], item["normal"]
        den = n[0] * dx + n[1] * dy + n[2]
        num = n[0] * p[0] + n[1] * p[1] + n[2] * p[2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = num / den
        return np.where((den != 0) & (t > 0), t, inf)
    if item["kind"] == "sphere":
        c, r = item["center"], item["radius"]
        a = dx * dx + dy * dy + 1.0
        b = -2.0 * (dx * c[0] + dy * c[1] + c[2])
        cc = c[0] * c[0] + c[1] * c[1] + c[2] * c[2] - r * r
        disc = b * b - 4.0 * a * cc
        t = (-b - np.sqrt(np.maximum(disc, 0.0))) / (2.0 * a)
        return np.where((disc >= 0) & (t > 0), t, inf)
    lo, hi = item["min"], item["max"]  # box: slab method, ray direction z component is 1
    tn = np.full(dx.shape, -np.inf)
    tf = np.full(dx.shape, np.inf)
    for dcomp, k in ((dx, 0), (dy, 1), (None, 2)):
        if dcomp is None:
            t1, t2 = np.full(dx.shape, float(lo[2])), np.full(dx.shape, float(hi[2]))
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                t1, t2 = lo[k] / dcomp, hi[k] / dcomp
            par = dcomp == 0  # parallel to the slab: inside it or never
            inside = (lo[k] <= 0) & (0 <= hi[k])
            t1 = np.where(par, -np.inf if inside else np.inf, t1)
            t2 = np.where(par, np.inf, t2)
        tn = np.maximum(tn, np.minimum(t1, t2))
        tf = np.minimum(tf, np.maximum(t1, t2))
    return np.where((tn <= tf) & (tn > 0), tn, inf)


def render_scene(name, noise_mm=None, shadow_px=None, seed=1, w=FRAME_W, h=FRAME_H, f=FOCAL_PX):
    """uint16 depth image (mm, 0 = no reading) of a named scene; noise / shadow default to the scene's own values."""
    sc = SCENES[name]
    noise_mm = sc["noise_mm"] if noise_mm is None else noise_mm
    shadow_px = sc["shadow_px"] if shadow_px is None else shadow_px
    v, u = np.mgrid[0:h, 0:w]
    dx = (u - w / 2) / f
    dy = (v - h / 2) / f
    z = np.full((h, w), np.inf)
    for item in sc["items"]:
        z = np.minimum(z, _hit(item, dx, dy))
    z = np.where(np.isfinite(z), z, 0.0)
    hole = np.zeros(z.shape, bool)
    for s in range(1, shadow_px + 1):  # shadow bands from the noise-free depth
        hole[:, s:] |= (z[:, :-s] > 0) & (z[:, :-s] < z[:, s:] - 150)
    if noise_mm > 0:
        idx = np.arange(w * h, dtype=np.uint64).reshape(h, w) * 3 + ((seed * 2654435761) % 2**32)
        g = sum(_hash32((idx + k) % 2**32) / 4294967296.0 for k in range(3))
        z = np.where(z > 0, z + noise_mm * ((z / 1000.0) * (z / 1000.0)) * (g - 1.5) * 2.0, 0.0)
    out = np.clip(np.floor(z + 0.5), 0, 65535)
    out[hole] = 0
    return out.astype(np.uint16)


def scene_list():
    """Scene names, titles and parameters for the web app."""
    return [{"name": k, "title": v["title"], "about": v["about"], "noise_mm": v["noise_mm"],
             "shadow_px": v["shadow_px"], "items": len(v["items"])} for k, v in SCENES.items()]
