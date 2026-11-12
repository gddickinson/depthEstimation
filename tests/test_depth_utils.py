"""Regression tests for depthEstimation.py's person read-out, tracker box format and edge-aware filter."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
cv2 = pytest.importorskip("cv2")
import depth_utils as U  # noqa: E402

SRC = (Path(__file__).resolve().parent.parent / "depthEstimation.py").read_text()


def test_box_formats_round_trip():
    assert U.xyxy_to_xywh((10, 20, 110, 220)) == (10, 20, 100, 200)
    assert U.xywh_to_xyxy((10, 20, 100, 200)) == (10, 20, 110, 220)


def test_person_label_reports_pixels_not_metres():
    """Was 'Avg Depth: <x2 - x1>m': a box width in pixels labelled as metres."""
    assert U.person_box_label((10, 20, 110, 220)) == "Box width: 100 px"
    assert U.person_box_label(None) is None
    assert "Avg Depth" not in SRC and "person_box_label(" in SRC


def test_tracker_gets_xywh_and_returns_xyxy():
    """KCF takes (x, y, w, h); it was given YOLO's (x1, y1, x2, y2)."""
    assert "xyxy_to_xywh(bbox)" in SRC and "xywh_to_xyxy(" in SRC


def test_filter_keeps_edges_of_a_0_1_depth_map():
    """sigma_color = 100 on 0-1 data blurred a 0.3 -> 0.7 step like a Gaussian; 0.1 keeps it."""
    rng = np.random.default_rng(0)
    step = np.where(np.arange(64)[None, :] < 32, 0.3, 0.7).repeat(32, 0) + rng.normal(0, 0.01, (32, 64))
    old = cv2.bilateralFilter(step.astype(np.float32), 7, 100, 100)
    new = U.edge_aware_filter(step)
    assert old[:, 31].mean() > 0.4  # old setting smeared the step
    assert abs(new[:, 31].mean() - 0.3) < 0.02 and abs(new[:, 32].mean() - 0.7) < 0.02
    assert np.abs(new - np.where(np.arange(64) < 32, 0.3, 0.7)).std() < np.abs(step - np.where(np.arange(64) < 32, 0.3, 0.7)).std()
    assert "edge_aware_filter(refined_depth_map)" in SRC and "sigma_color = 100" not in SRC
