"""Web interface for depthEstimation (FastAPI + plain ES modules, no build step).

Contents:
    app.py     FastAPI app: /api/info, /api/scenes, /api/colormaps (exported to the static site) and the live
               Kinect endpoints /api/device/status, /api/device/connect, /api/device/frame (local app only)
    device.py  KinectSource: thread-safe access to kinect_wrapper.PyKinectCapture with timeouts and a clear
               status when the extension is not built or no Kinect is attached
    static/    front end (index.html, core.js, pages, depth_engine.js / scene_engine.js and their node tests)
"""
