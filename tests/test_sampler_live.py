"""The sampler on real hardware: a GPU load in a child process (short form of accuracy check 3)."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest

from metalmeter._bind import metal
from metalmeter.sampler import Sampler
from metalmeter.types import OK, PROCESS_EXITED
from tests.conftest import macos, needs_gpu

LOADS = pathlib.Path(__file__).resolve().parent.parent / "validation" / "loads.py"
pytestmark = [macos, needs_gpu, pytest.mark.allow_subprocess,
              pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="needs torch (dev only)")]


def test_gpu_load_in_a_child_shows_on_the_tree_not_on_the_parent_alone():
    load = subprocess.Popen([sys.executable, str(LOADS), "torch-matmul", "--seconds", "5"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        pid = json.loads(load.stdout.readline())["ready"]
        alone, tree, direct = Sampler(os.getpid()), Sampler(os.getpid(), children=True), Sampler(pid)
        for s in (alone, tree, direct):
            s.sample()
        time.sleep(3)
        a, t, d = alone.sample(), tree.sample(), direct.sample()
    finally:
        load.wait(timeout=30)
    assert a.status == t.status == d.status == OK
    assert a.gpu_util_pct < 2                                   # isolation: the parent does no GPU work
    assert t.gpu_util_pct > 50 and t.children_count >= 1 and t.children_gpu_util_pct > 50
    assert d.gpu_util_pct > 50 and d.cpu_util_pct is not None and d.mem_footprint_bytes > 0
    assert d.gpu_mem_bytes is None                              # not our process: no API gives it
    assert metal._device is None                                # and we never created an MTLDevice
    assert direct.sample().status == PROCESS_EXITED             # the load has finished
