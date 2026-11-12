"""Depth-frame processing used by the Kinect scripts, factored out so it can be tested and reused without hardware.

The Kinect wrapper (kinect_capture.cpp) streams 640 x 480 depth in millimetres (FREENECT_DEPTH_MM, 0 = no reading).
"""
import numpy as np

FRAME_W = 640  # kinect_capture.cpp: FREENECT_RESOLUTION_MEDIUM
FRAME_H = 480
FOCAL_PX = 525.0  # slamMapping.py: focal length in pixels; the principal point is the image centre (w/2, h/2)


def depth_to_points(depth, f=FOCAL_PX):
    """Pinhole back-projection of a depth image (mm) to (x, y, z) points in mm, camera frame (x right, y down).

    Pixel (u, v) with depth z maps to x = (u - w/2) z / f, y = (v - h/2) z / f, the model used by slamMapping.py.
    Returns (points, valid): points is (N, 3) for the pixels with a reading (finite and z > 0) in row-major order, and
    valid is the flat boolean mask of those pixels (use it to pick matching colours from the RGB frame).
    """
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    x = (u - w / 2) * z / f
    y = (v - h / 2) * z / f
    xyz = np.dstack((x, y, z)).reshape((-1, 3))
    valid = np.all(np.isfinite(xyz), axis=1) & (xyz[:, 2] > 0)
    return xyz[valid], valid


def normalize_minmax(depth, valid_only=False):
    """Stretch a depth frame to 0..255 uint8, as cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U) does
    in quickTest.py and test_kinect.py (scale = 255 / (max - min), value = round(depth * scale - min * scale)).

    With valid_only=True the range is taken over pixels with a reading (depth > 0) and those without stay 0; the
    scripts themselves use the whole frame, so holes (0) set the minimum.
    """
    d = np.asarray(depth, dtype=np.float64)
    mask = d > 0 if valid_only else np.ones(d.shape, bool)
    if not mask.any():
        return np.zeros(d.shape, np.uint8)
    lo, hi = d[mask].min(), d[mask].max()
    scale = 255.0 / (hi - lo) if hi - lo > np.finfo(float).eps else 0.0
    shift = 0.0 - lo * scale
    out = np.clip(np.rint(d * scale + shift), 0, 255).astype(np.uint8)
    out[~mask] = 0
    return out


def load_colormaps():
    """{'jet': (256, 3) uint8 RGB, 'viridis': ...} from colormap_luts.json (tabulated from cv2.applyColorMap)."""
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).resolve().parent / "colormap_luts.json").read_text())
    return {k: np.array(v, dtype=np.uint8) for k, v in raw.items() if isinstance(v, list)}


_LUTS = None


def apply_colormap(gray, name="jet"):
    """RGB version of cv2.applyColorMap(gray, COLORMAP_<NAME>) for a uint8 image ('jet', 'viridis' or 'gray')."""
    global _LUTS
    gray = np.asarray(gray, dtype=np.uint8)
    if name == "gray":
        return np.repeat(gray[..., None], 3, axis=-1)
    if _LUTS is None:
        _LUTS = load_colormaps()
    return _LUTS[name][gray]


def bilateral_kernel(d=7, sigma_space=100.0):
    """Offsets (dy, dx) and spatial weights of cv2.bilateralFilter's disc-shaped window (radius d // 2)."""
    radius = d // 2 if d > 0 else int(round(sigma_space * 1.5))
    radius = max(radius, 1)
    offs, wts = [], []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            r2 = dy * dy + dx * dx
            if np.sqrt(r2) > radius:
                continue
            offs.append((dy, dx))
            wts.append(np.exp(-0.5 * r2 / (sigma_space * sigma_space)))
    return radius, offs, wts


def bilateral_filter(img, d=7, sigma_color=100.0, sigma_space=100.0):
    """Edge-aware smoothing with the weights cv2.bilateralFilter uses (depthEstimation.py: d=7, sigmas 100).

    out(p) = sum_q w_s(q-p) w_c(I_q - I_p) I_q / sum_q w_s w_c over a disc of radius d // 2, with
    w_s = exp(-|q-p|^2 / 2 sigma_space^2), w_c = exp(-(I_q - I_p)^2 / 2 sigma_color^2) and mirrored borders
    (BORDER_REFLECT_101). OpenCV evaluates w_c from a binned table for float images, so it agrees to a small fraction
    of sigma_color rather than bit for bit.
    """
    a = np.asarray(img, dtype=np.float64)
    radius, offs, wts = bilateral_kernel(d, sigma_space)
    pad = np.pad(a, radius, mode="reflect")
    h, w = a.shape
    num = np.zeros_like(a)
    den = np.zeros_like(a)
    gc = -0.5 / (sigma_color * sigma_color)
    for (dy, dx), ws in zip(offs, wts):
        q = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w]
        wgt = ws * np.exp((q - a) ** 2 * gc)
        num += wgt * q
        den += wgt
    return num / den


def depth_stats(depth):
    """Summary of one depth frame (mm): size, fraction of pixels with a reading, and min / median / max of those."""
    d = np.asarray(depth)
    v = d[d > 0].astype(np.float64)
    out = {"width": int(d.shape[1]), "height": int(d.shape[0]), "valid_fraction": float(v.size / d.size)}
    if v.size:
        out.update(min_mm=float(v.min()), median_mm=float(np.median(v)), max_mm=float(v.max()), mean_mm=float(v.mean()))
    return out
