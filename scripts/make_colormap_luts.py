"""Write kinect_depth/colormap_luts.json: OpenCV's colour maps used by the Kinect scripts, as 256-entry RGB tables.

quickTest.py / test_kinect.py colour depth with cv2.COLORMAP_JET and depthEstimation.py with cv2.COLORMAP_VIRIDIS.
Tabulating them once lets the web app and kinect_depth.processing reproduce cv2.applyColorMap exactly without OpenCV.

Run (needs opencv-python): python scripts/make_colormap_luts.py
"""
import json
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "kinect_depth" / "colormap_luts.json"
MAPS = {"jet": cv2.COLORMAP_JET, "viridis": cv2.COLORMAP_VIRIDIS}


def main():
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    luts = {"source": f"cv2.applyColorMap, OpenCV {cv2.__version__}", "order": "RGB"}
    for name, code in MAPS.items():
        bgr = cv2.applyColorMap(ramp, code)[0]
        luts[name] = bgr[:, ::-1].astype(int).tolist()
    OUT.write_text(json.dumps(luts, separators=(",", ":")) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
