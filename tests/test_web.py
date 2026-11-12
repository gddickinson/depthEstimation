"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests ---------------------------------------------------------------------------------
import numpy as np  # noqa: E402

from kinect_depth import processing as P  # noqa: E402
from kinect_depth import scenes as S  # noqa: E402
from web import app as webapp  # noqa: E402
from web.device import KinectSource  # noqa: E402


def test_info_reports_the_code_constants():
    j = client.get("/api/info").json()
    assert j["frame"]["width"] == 640 and j["intrinsics"]["f_px"] == 525.0 and j["filter"]["d"] == 7


def test_scenes_carry_definitions_and_python_stats():
    j = client.get("/api/scenes").json()
    assert {s["name"] for s in j} == set(S.SCENES)
    room = next(s for s in j if s["name"] == "room")
    assert room["stats"] == P.depth_stats(S.render_scene("room")) and room["items"] == S.SCENES["room"]["items"]


def test_colormaps_are_256_rgb_rows():
    j = client.get("/api/colormaps").json()
    assert set(j) == {"jet", "viridis"} and all(len(v) == 256 and len(v[0]) == 3 for v in j.values())


class FakeCapture:
    """Stands in for kinect_wrapper.PyKinectCapture: serves a synthetic scene as the depth stream."""
    def __init__(self, ok=True):
        self.ok, self.closed = ok, 0

    def init(self):
        return self.ok

    def get_frames(self):
        depth = S.render_scene("room")
        rgb = np.zeros((480, 640, 3), np.uint8)
        rgb[..., 1] = 200
        return rgb, depth

    def close(self):
        self.closed += 1


def _with_source(monkeypatch, factory):
    src = KinectSource(factory=factory, timeout=2)
    monkeypatch.setattr(webapp, "source", src)
    return src


def test_device_frame_streams_rgb_and_depth(monkeypatch):
    _with_source(monkeypatch, FakeCapture)
    assert client.get("/api/device/status").json()["state"] == "idle"
    assert client.post("/api/device/connect").json()["streaming"]
    j = client.get("/api/device/frame?step=2").json()
    assert j["ok"] and (j["width"], j["height"], j["f_px"]) == (320, 240, 262.5)
    depth = np.frombuffer(__import__("base64").b64decode(j["depth"]), "<u2").reshape(240, 320)
    assert (depth == S.render_scene("room")[::2, ::2]).all()
    assert j["stats"] == P.depth_stats(depth)
    assert client.post("/api/device/disconnect").json()["state"] == "idle"


def test_device_without_kinect_says_why(monkeypatch):
    src = _with_source(monkeypatch, lambda: FakeCapture(ok=False))
    j = client.post("/api/device/connect").json()
    assert j["state"] == "no-device" and not j["streaming"]
    assert client.get("/api/device/frame").json()["ok"] is False
    assert src.cap is None


def test_device_without_extension_says_why(monkeypatch):
    def missing():
        raise ImportError("No module named 'kinect_wrapper'")
    _with_source(monkeypatch, missing)
    j = client.post("/api/device/connect").json()
    assert j["state"] == "unavailable" and "setup.py build_ext" in j["reason"]


def test_device_stall_times_out(monkeypatch):
    import time

    class Stuck(FakeCapture):
        def get_frames(self):
            time.sleep(1.5)
            return super().get_frames()
    src = _with_source(monkeypatch, Stuck)
    src.timeout = 0.2
    client.post("/api/device/connect")
    j = client.get("/api/device/frame").json()
    assert j["ok"] is False and j["state"] == "stalled"
