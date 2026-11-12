"""The JS-vs-Python fixture (tests/fixtures/engine_fixture.json) is still what kinect_depth produces.

Numbers are compared with a tolerance because libm can differ in the last digit between macOS and Linux.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import make_engine_fixture  # noqa: E402


def _close(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _close(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, (list, tuple)) and a and isinstance(a[0], (list, tuple, dict)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _close(x, y, f"{path}[{i}]")
    elif isinstance(a, (list, tuple, int, float)):
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), rtol=1e-9, atol=1, err_msg=path)
    else:
        assert a == b, path


def test_fixture_is_current():
    stored = json.loads((ROOT / "tests" / "fixtures" / "engine_fixture.json").read_text())
    fresh = json.loads(json.dumps(make_engine_fixture.build()))
    _close(fresh, stored)
