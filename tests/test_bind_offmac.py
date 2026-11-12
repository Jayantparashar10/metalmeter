"""Off macOS (Linux, Windows) every binding imports and returns None or empty, never raises."""
import pytest

from tests.conftest import run_child

CHILD = """
import ctypes, json, sys
sys.platform = "linux"   # before any metalmeter import
import metalmeter
from metalmeter._bind import _ffi, cf, iokit, libproc, metal, objc, sysctl
with iokit.matching("IOAccelerator") as accs:
    pass
r = {
    "registry": len(_ffi.REGISTRY),
    "available": [cf.AVAILABLE, iokit.AVAILABLE, libproc.AVAILABLE, sysctl.AVAILABLE, objc.AVAILABLE],
    "to_py": cf.to_py(123), "create_string": cf.create_string("x"),
    "matching": accs, "prop": iokit.prop(1, "x"), "entry_id": iokit.entry_id(1), "class_name": iokit.class_name(1),
    "rusage": libproc.pid_rusage(1), "children": libproc.child_pids(1), "timebase": libproc.TIMEBASE,
    "sysctl_int": sysctl.integer("hw.memsize"), "sysctl_str": sysctl.string("kern.osversion"),
    "swap": sysctl.swap_usage(), "thermal": objc.thermal_state(), "msg": objc.msg(ctypes.c_long),
    "cls": objc.cls("NSObject"), "device": metal.device(),
}
print(json.dumps(r))
"""


@pytest.mark.allow_subprocess
def test_bindings_import_and_return_nothing_off_macos():
    r = run_child(CHILD)
    assert r.pop("registry") == 0
    assert r.pop("available") == [False] * 5
    assert r.pop("matching") == []
    assert r.pop("rusage") == [None, 0]
    assert r.pop("timebase") == [1, 1]
    assert all(v is None for v in r.values()), r
