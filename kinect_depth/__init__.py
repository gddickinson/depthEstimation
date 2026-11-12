"""Hardware-free depth maths for the Kinect Xbox 360 depth stream (no Qt, no libfreenect, numpy only).

Contents:
    processing.py      frame constants and intrinsics, min-max normalisation, colour maps, depth -> point cloud,
                       bilateral filter, frame statistics
    scenes.py          synthetic depth scenes (ray-cast planes, boxes and spheres) used when no Kinect is attached
    colormap_luts.json OpenCV's COLORMAP_JET and COLORMAP_VIRIDIS lookup tables (scripts/make_colormap_luts.py)
"""
