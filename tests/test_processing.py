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


def test_normalize_matches_cv2_formula():
    depth = np.array([[0, 1000], [2000, 4000]], dtype=np.uint16)
    assert P.normalize_minmax(depth).tolist() == [[0, 64], [128, 255]]
    assert P.normalize_minmax(depth, valid_only=True).tolist() == [[0, 0], [85, 255]]
    assert P.normalize_minmax(np.full((2, 2), 7)).tolist() == [[0, 0], [0, 0]]


def test_colormaps_and_filter_agree_with_opencv():
    cv2 = __import__("pytest").importorskip("cv2")
    rng = np.random.default_rng(0)
    d = (rng.random((48, 64)) * 4000).astype(np.uint16)
    d[d < 300] = 0
    g = P.normalize_minmax(d)
    assert np.abs(g.astype(int) - cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)).max() <= 1
    for name, code in (("jet", cv2.COLORMAP_JET), ("viridis", cv2.COLORMAP_VIRIDIS)):
        assert (P.apply_colormap(g, name) == cv2.applyColorMap(g, code)[..., ::-1]).all()
    img = np.full((48, 64), 1000.0)
    img[:, 32:] = 2500
    img += rng.normal(0, 20, img.shape)
    ref = cv2.bilateralFilter(img.astype(np.float32), 7, 100, 100)
    assert np.abs(P.bilateral_filter(img, 7, 100, 100) - ref).max() < 0.05


def test_bilateral_keeps_a_step_and_smooths_noise():
    rng = np.random.default_rng(1)
    clean = np.where(np.arange(64)[None, :] < 32, 1000.0, 2000.0).repeat(32, 0)
    noisy = clean + rng.normal(0, 10, clean.shape)
    out = P.bilateral_filter(noisy, 7, 100, 100)
    assert np.abs(out - clean).std() < 0.6 * np.abs(noisy - clean).std()
    assert abs(out[:, 31].mean() - 1000) < 5 and abs(out[:, 32].mean() - 2000) < 5


def test_scenes_render_hits_and_holes():
    from kinect_depth import scenes as S
    ramp = S.render_scene("ramp")
    assert ramp.shape == (480, 640) and (ramp > 0).all()
    assert ramp[240, 0] < ramp[240, 639]  # turned plane: nearer on the left
    clean = S.render_scene("steps", noise_mm=0, shadow_px=0)
    assert clean[240, 100] == 1200 and (clean[240, :] > 0).all()
    assert (S.render_scene("steps")[240] == 0).any()  # shadow bands beside the near panels
    assert (S.render_scene("room", seed=2) != S.render_scene("room", seed=3)).any()
