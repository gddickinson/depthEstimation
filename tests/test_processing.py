"""Tests for kinect_depth.processing (pure numpy, no hardware)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kinect_depth import processing as P  # noqa: E402


def test_point_cloud_back_projects_each_pixel_through_its_own_column():
    """Regression: slamMapping built the column index with np.repeat, so x did not follow the pixel column."""
    depth = np.full((4, 6), 1000, dtype=np.uint16)
    pts, valid = P.depth_to_points(depth, f=500.0)
    pts = pts.reshape(4, 6, 3)
    u = np.arange(6)
    for v in range(4):
        np.testing.assert_allclose(pts[v, :, 0], (u - 3) * 1000 / 500.0)
        np.testing.assert_allclose(pts[v, :, 1], (v - 2) * 1000 / 500.0)


def test_point_cloud_drops_pixels_without_a_reading():
    """Regression: depth 0 (no reading in FREENECT_DEPTH_MM) was kept as a point at the camera origin."""
    depth = np.array([[0, 1200], [800, 0]], dtype=np.uint16)
    pts, valid = P.depth_to_points(depth, f=525.0)
    assert len(pts) == 2 and valid.tolist() == [False, True, True, False]
    assert (pts[:, 2] > 0).all()
