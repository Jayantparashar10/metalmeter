"""Bindings against the real macOS APIs. Runs on any Mac; GPU tests skip without AGXAccelerator."""
import ctypes
import errno
import os
import platform
import re
import subprocess
import sys
import time

import pytest

from metalmeter._bind import cf, iokit, libproc, objc, sysctl
from tests.conftest import macos, needs_gpu, run_child

pytestmark = macos
ARM64_NATIVE = platform.machine() == "arm64"


# libproc
def test_rusage_layout_matches_sdk():
    s = libproc.RUsageInfoV6
    assert libproc.LAYOUT_OK
    assert ctypes.sizeof(s) == 464
    assert s.ri_energy_nj.offset == 336 and s.ri_neural_footprint.offset == 368
    assert s.ri_runnable_time.offset + 8 == 296  # end of rusage_info_v4


def test_rusage_flavor_is_v6_on_macos_13_or_later():
    major = int((sysctl.string("kern.osproductversion") or "0").split(".")[0])
    assert libproc.FLAVOR == (libproc.RUSAGE_INFO_V6 if major >= 13 else libproc.RUSAGE_INFO_V4)


def test_own_rusage():
    r, err = libproc.pid_rusage(os.getpid())
    assert err == 0 and r is not None
    assert r["flavor"] == libproc.FLAVOR
    assert len(r) == 1 + len(libproc.V6_FIELDS if r["flavor"] == 6 else libproc.V4_FIELDS)
    assert r["ri_phys_footprint"] > 0 and r["ri_proc_start_abstime"] > 0
    assert r["ri_user_time"] > 0


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read every process")
def test_rusage_of_another_users_process_is_eperm():
    assert libproc.pid_rusage(1) == (None, errno.EPERM)  # launchd runs as root


def test_rusage_of_missing_pid_is_esrch():
    assert libproc.pid_rusage(999_999) == (None, errno.ESRCH)


def test_timebase():
    numer, denom = libproc.TIMEBASE
    assert numer > 0 and denom > 0
    if ARM64_NATIVE:
        assert libproc.TIMEBASE == (125, 3)
        assert libproc.ticks_to_ns(3) == 125


@pytest.mark.allow_subprocess
def test_child_pids():
    kids = [subprocess.Popen([sys.executable, "-c", "import time; time.sleep(2)"]) for _ in range(3)]
    try:
        time.sleep(0.2)
        found = libproc.child_pids(os.getpid())
        assert {k.pid for k in kids} <= set(found)
        assert libproc.child_pids(kids[0].pid) == []
    finally:
        for k in kids:
            k.kill()
            k.wait()
    assert not {k.pid for k in kids} & set(libproc.child_pids(os.getpid()))


# sysctl
def test_sysctl_values():
    assert sysctl.integer("hw.memsize", ctypes.c_uint64) > 0
    assert sysctl.integer("kern.memorystatus_vm_pressure_level") in (1, 2, 4)
    assert sysctl.string("kern.osproductversion")
    assert sysctl.string("machdep.cpu.brand_string")
    swap = sysctl.swap_usage()
    assert swap is not None and swap.xsu_total >= swap.xsu_used


def test_sysctl_wrong_size_or_missing_name_is_none():
    assert sysctl.integer("hw.memsize", ctypes.c_int) is None  # kernel value is 8 bytes
    assert sysctl.integer("metalmeter.no.such.name") is None
    assert sysctl.string("metalmeter.no.such.name") is None


# IOKit and CoreFoundation
def test_iokit_platform_expert_device():
    with iokit.matching("IOPlatformExpertDevice") as objs:
        assert len(objs) == 1
        e = objs[0]
        assert iokit.class_name(e) == "IOPlatformExpertDevice"
        assert iokit.entry_id(e) > 0
        assert isinstance(iokit.prop(e, "model"), bytes)          # CFData
        assert isinstance(iokit.prop(e, "IOPlatformUUID"), str)   # CFString (value not shown)
        assert iokit.prop(e, "metalmeter-no-such-key") is None


def test_iokit_unknown_class_is_empty():
    with iokit.matching("MetalmeterNoSuchClass") as objs:
        assert objs == []


def test_cf_owned_releases_once_and_tolerates_null():
    with cf.owned(cf.create_string("metalmeter")) as s:
        assert cf.to_py(s) == "metalmeter"
    with cf.owned(None) as s:
        assert s is None


@needs_gpu
def test_gpu_registry_keys():
    with iokit.matching("IOAccelerator") as accs:
        agx = [a for a in accs if (iokit.class_name(a) or "").startswith("AGX")]
        assert agx
        a = agx[0]
        assert iokit.prop(a, "gpu-core-count") > 0
        perf = iokit.prop(a, "PerformanceStatistics")
        assert isinstance(perf, dict) and isinstance(perf["Device Utilization %"], int)
        with iokit.children(a) as (kids, consistent):
            assert isinstance(consistent, bool)
            creators = [iokit.prop(k, "IOUserClientCreator") for k in kids]
            assert any(isinstance(c, str) and re.match(r"pid \d+, ", c) for c in creators)
            usages = [iokit.prop(k, "AppUsage") for k in kids]
            assert any(isinstance(u, list) for u in usages)


# Objective-C, Foundation and Metal
def test_thermal_state():
    assert objc.thermal_state() in (0, 1, 2, 3)


METAL_CHILD = """
import json, os
from metalmeter._bind import iokit, metal
def own_gpu_clients():
    n = 0
    with iokit.matching("IOAccelerator") as accs:
        for a in accs:
            with iokit.children(a) as (kids, _):
                for k in kids:
                    c = iokit.prop(k, "IOUserClientCreator")
                    n += isinstance(c, str) and c.startswith(f"pid {os.getpid()},")
    return n
before = own_gpu_clients()
dev = metal.device()
print(json.dumps({"clients_before": before, "clients_after": own_gpu_clients(), "same": metal.device() == dev,
                  "dev": bool(dev), "alloc": metal.current_allocated_size(dev),
                  "limit": metal.recommended_max_working_set_size(dev)}))
"""


@needs_gpu
@pytest.mark.allow_subprocess
def test_metal_device_is_created_only_on_first_use_and_reads_own_memory():
    r = run_child(METAL_CHILD)
    assert r["clients_before"] == 0 and r["clients_after"] >= 1  # importing opened no GPU client
    assert r["dev"] and r["same"]
    assert isinstance(r["alloc"], int) and r["alloc"] >= 0
    assert r["limit"] > 0
