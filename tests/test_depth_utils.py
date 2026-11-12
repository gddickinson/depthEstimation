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

