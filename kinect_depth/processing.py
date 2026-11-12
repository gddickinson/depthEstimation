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
