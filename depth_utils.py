"""Small, model-free helpers for depthEstimation.py (cv2 + numpy only, so they can be tested without TensorFlow/YOLO).

- xyxy_to_xywh / xywh_to_xyxy: YOLO gives boxes as (x1, y1, x2, y2); OpenCV's KCF tracker takes and returns (x, y, w, h).
- person_box_label: the on-screen person read-out. The monocular depth model's output is a unitless 0-1 map (and its
  decoder is untrained), so no metric depth exists; the read-out reports the box width in pixels.
- edge_aware_filter: bilateral filter with sigma_color scaled to the 0-1 range of the model output.
"""
import cv2
import numpy as np

# The depth map is a sigmoid output in 0-1: sigma_color must be on that scale for the filter to keep edges
# (sigma_color = 100 made every neighbour weigh the same, i.e. a plain Gaussian blur).
FILTER_D = 7
FILTER_SIGMA_COLOR = 0.1
FILTER_SIGMA_SPACE = 100


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    return (x1, y1, x2 - x1, y2 - y1)


def xywh_to_xyxy(box):
    x, y, w, h = (float(v) for v in box[:4])
    return (x, y, x + w, y + h)


def person_box_label(box_xyxy):
    """Read-out for a person box (x1, y1, x2, y2), or None without a box."""
    if box_xyxy is None:
        return None
    return f"Box width: {box_xyxy[2] - box_xyxy[0]:.0f} px"


def edge_aware_filter(depth_map, d=FILTER_D, sigma_color=FILTER_SIGMA_COLOR, sigma_space=FILTER_SIGMA_SPACE):
    return cv2.bilateralFilter(np.asarray(depth_map, dtype=np.float32), d, sigma_color, sigma_space)
