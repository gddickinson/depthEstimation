"""Start the Kinect Depth Lab web app on http://127.0.0.1:8155 (local only).

Run: python run_web.py [--port 8155]
Needs: pip install -r requirements-web.txt. The Live page also needs the kinect_wrapper extension built for this Python
(python setup.py build_ext --inplace, with libfreenect) and a Kinect on USB; everything else works without them.
"""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8155)
    port = ap.parse_args().port
    print(f"Kinect Depth Lab web app: http://127.0.0.1:{port}")
    uvicorn.run("web.app:app", host="127.0.0.1", port=port)
