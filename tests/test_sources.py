"""Sources against the real macOS APIs: shapes, PID filtering, never raising."""
import errno
import os
import subprocess
import sys
import time
import warnings

import pytest

from metalmeter import _platform, parse
from metalmeter._bind import iokit, metal
from metalmeter.sources import ane, gpu_clients, perf_stats, process, system
from tests.conftest import macos, needs_gpu

pytestmark = macos


@pytest.fixture(autouse=True)
def fresh_warnings():
    _platform._warned.clear()


@needs_gpu
def test_gpu_clients_read_all_and_filtered():
    raw = gpu_clients.read()
    assert raw["accelerators"] >= 1 and isinstance(raw["consistent"], bool) and raw["t_mono_ns"] > 0
    pids = {parse.parse_creator(c["creator"])[0] for c in raw["clients"]}
    assert pids
    one = sorted(pids)[0]
    only = gpu_clients.read([one])
    assert only["clients"] and {parse.parse_creator(c["creator"])[0] for c in only["clients"]} == {one}
    assert gpu_clients.read([]) ["clients"] == []
    assert parse.gpu_clients(raw)["pids"]


def test_own_memory_without_a_gpu_client_creates_no_device():
    assert gpu_clients.own_memory(False) is None
    assert metal._device is None


@needs_gpu
def test_perf_stats():
    out = parse.perf_stats(perf_stats.read())
    assert 0 <= out["sys_gpu_util_pct"] <= 100
    assert all(v is not None for v in out.values())
    assert parse.device_info(system.read_static(), perf_stats.read_static())["gpu_cores"] > 0


def test_process_read():
    own = process.read(os.getpid())
    assert own["errno"] == 0 and parse.rusage(own, process.timebase())["cpu_ns"] > 0
    if os.geteuid() != 0:
        assert process.read(1) == {"pid": 1, "rusage": None, "errno": errno.EPERM}
    assert process.read(999_999)["errno"] == errno.ESRCH
    assert process.descendants(os.getpid()) == []


@pytest.mark.allow_subprocess
def test_descendants_include_grandchildren():
    code = ("import subprocess, sys, time\n"
            "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3)'])\n"
            "print(g.pid, flush=True); time.sleep(3)\n")
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    try:
        grandchild = int(child.stdout.readline())
        time.sleep(0.1)
        assert process.descendants(os.getpid())[:2] == [child.pid, grandchild]
    finally:
        child.kill()
        child.wait()


@pytest.mark.skipif(not os.uname().machine == "arm64", reason="ANE needs Apple Silicon")
def test_ane_read():
    raw = ane.read()
    assert raw["providers"] and isinstance(raw["clients"], list)
    assert ane.read([os.getpid()])["clients"] == []


def test_system_read():
    out = parse.system(system.read())
    assert out["thermal_state"] in parse.THERMAL_STATES and out["mem_pressure"] in (1, 2, 4)
    assert out["swap_total_bytes"] is not None
    static = system.read_static()
    assert static["memsize"] > 0 and static["os_version"] and static["vmm_present"] in (0, 1)


def test_a_failing_binding_gives_none_and_warns_once(monkeypatch):
    def boom(*a):
        raise RuntimeError("registry exploded")
    monkeypatch.setattr(iokit, "prop", boom)
    with pytest.warns(RuntimeWarning, match="gpu_clients.read failed"):
        assert gpu_clients.read() is None
    with pytest.warns(RuntimeWarning, match="ane.read failed"):  # each source warns under its own key
        assert ane.read() is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a second failure must not warn again
        assert gpu_clients.read() is None
        assert ane.read() is None
