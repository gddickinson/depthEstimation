"""Regression tests for the compiled Kinect wrapper without a Kinect attached.

Before the fix, creating a PyKinectCapture whose init() failed (no device) aborted the whole process when it was
garbage-collected (the destructor joined a thread that was never started), get_frames() after a failed init() blocked
forever, and close(), which every GUI calls from closeEvent, did not exist. Skipped when kinect_wrapper has not been
built for this Python (python setup.py build_ext --inplace).
"""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

PROBE = """
import kinect_wrapper
c = kinect_wrapper.PyKinectCapture()
ok = c.init()
if not ok:
    assert c.get_frames() == (None, None)
c.close()
c.close()
del c
print("clean exit")
"""


def _wrapper_importable():
    r = subprocess.run([sys.executable, "-c", "import kinect_wrapper"], cwd=ROOT, capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not _wrapper_importable(), reason="kinect_wrapper not built for this Python")
def test_capture_lifecycle_without_device_exits_cleanly():
    r = subprocess.run([sys.executable, "-c", PROBE], cwd=ROOT, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
    assert "clean exit" in r.stdout
