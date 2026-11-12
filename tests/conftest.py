import json
import os
import subprocess
import sys

import pytest

from tests import _guard

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
macos = pytest.mark.skipif(sys.platform != "darwin", reason="needs macOS")


@pytest.fixture(autouse=True)
def no_spawn(request, monkeypatch):
    """Every test runs with process creation blocked, unless marked allow_subprocess."""
    if request.node.get_closest_marker("allow_subprocess") is None:
        _guard.install(monkeypatch.setattr)


def run_child(code):
    """Run `code` in a fresh Python with the guard installed first; return its last stdout line as JSON."""
    prelude = f"import sys; sys.path.insert(0, {ROOT!r}); from tests import _guard; _guard.install()\n"
    p = subprocess.run([sys.executable, "-c", prelude + code], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout.strip().splitlines()[-1])


def _has_agx():
    if sys.platform != "darwin":
        return False
    from metalmeter._bind import iokit
    with iokit.matching("IOAccelerator") as accs:
        return any((iokit.class_name(a) or "").startswith("AGX") for a in accs)


HAS_AGX = _has_agx()
needs_gpu = pytest.mark.skipif(not HAS_AGX, reason="needs an Apple Silicon GPU (AGXAccelerator)")
