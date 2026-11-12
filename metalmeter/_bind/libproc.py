"""libproc and Mach time: per-process resource usage, child PIDs, and the Mach clock."""
from __future__ import annotations

import ctypes as C
import os
from typing import Optional

from .._platform import warn_once
from ._ffi import LIBSYSTEM, bind, load

u32, u64, vp = C.c_uint32, C.c_uint64, C.c_void_p
_libsystem = load(LIBSYSTEM)

# Field order of struct rusage_info_v6 in <sys/resource.h>. The first 35 fields are rusage_info_v4.
V4_FIELDS = (
    "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint "
    "proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups "
    "child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
    "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility "
    "cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time "
    "serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy "
    "serviced_energy interval_max_phys_footprint runnable_time"
).split()
V6_FIELDS = V4_FIELDS + (
    "flags user_ptime system_ptime pinstructions pcycles energy_nj penergy_nj secure_time_in_system "
    "secure_ptime_in_system neural_footprint lifetime_max_neural_footprint interval_max_neural_footprint "
    "conclave_footprint page_wait_time_mach page_cache_hits"
).split()


class RUsageInfoV6(C.Structure):
    _fields_ = ([("ri_uuid", C.c_uint8 * 16)] + [("ri_" + n, u64) for n in V6_FIELDS]
                + [("ri_reserved", u64 * 6)])


RUSAGE_INFO_V4 = 4
RUSAGE_INFO_V6 = 6

# Expected layout, checked with offsetof() against the macOS 27 SDK. If the ctypes definition
# ever disagrees, the rusage fields are switched off rather than read from the wrong offsets.
EXPECTED_LAYOUT = {"size": 464, "ri_phys_footprint": 72, "ri_proc_start_abstime": 80,
                   "ri_runnable_time": 288, "ri_energy_nj": 336, "ri_neural_footprint": 368}


def _layout_ok() -> bool:
    got = {"size": C.sizeof(RUsageInfoV6)}
    got.update({k: getattr(RUsageInfoV6, k).offset for k in EXPECTED_LAYOUT if k != "size"})
    return got == EXPECTED_LAYOUT


LAYOUT_OK = _layout_ok()

proc_pid_rusage = bind(_libsystem, "proc_pid_rusage", C.c_int, C.c_int, C.c_int, vp)
proc_listchildpids = bind(_libsystem, "proc_listchildpids", C.c_int, C.c_int, vp, C.c_int)


class MachTimebaseInfo(C.Structure):
    _fields_ = [("numer", u32), ("denom", u32)]


mach_timebase_info = bind(_libsystem, "mach_timebase_info", C.c_int, C.POINTER(MachTimebaseInfo))
mach_absolute_time = bind(_libsystem, "mach_absolute_time", u64)


def abstime() -> Optional[int]:
    """Return the current Mach absolute time, the clock used by ri_proc_start_abstime.

    It counts Mach ticks and stops while the Mac sleeps.
    """
    return int(mach_absolute_time()) if mach_absolute_time is not None else None


def _probe_flavor() -> Optional[int]:
    """Return the newest rusage flavor the kernel accepts, probed on our own PID."""
    if proc_pid_rusage is None or not LAYOUT_OK:
        return None
    buf = RUsageInfoV6()
    for flavor in (RUSAGE_INFO_V6, RUSAGE_INFO_V4):
        if proc_pid_rusage(os.getpid(), flavor, C.byref(buf)) == 0:
            return flavor
    return None


FLAVOR = _probe_flavor()
AVAILABLE = FLAVOR is not None


def pid_rusage(pid: int) -> tuple[Optional[dict[str, int]], int]:
    """Read the resource usage of `pid`.

    Returns:
        ``({"flavor": n, "ri_<field>": value, ...}, 0)`` on success, or ``(None, errno)``:
        EPERM for another user's process, ESRCH if the process does not exist. CPU times
        are in Mach ticks, as the kernel reports them; convert them with `ticks_to_ns`.
    """
    if not LAYOUT_OK:
        warn_once("rusage-layout", "rusage_info_v6 layout does not match; process fields are off")
        return None, 0
    if FLAVOR is None:
        return None, 0
    buf = RUsageInfoV6()
    if proc_pid_rusage(pid, FLAVOR, C.byref(buf)) != 0:
        return None, C.get_errno()
    names = V6_FIELDS if FLAVOR == RUSAGE_INFO_V6 else V4_FIELDS
    out = {"flavor": FLAVOR}
    out.update({"ri_" + n: getattr(buf, "ri_" + n) for n in names})
    return out, 0


def child_pids(pid: int) -> Optional[list[int]]:
    """Return the PIDs of the direct children of `pid`, or None if they cannot be listed."""
    if proc_listchildpids is None:
        return None
    capacity = 256
    while capacity <= 65536:
        buf = (C.c_int * capacity)()
        n = proc_listchildpids(pid, buf, C.sizeof(buf))  # returns a PID count, not bytes
        if n < 0:
            return None
        if n < capacity:
            return [p for p in buf[:n] if p > 0]
        capacity *= 4  # a full buffer may have cut the list short, so ask again with more room
    return None


def _timebase() -> tuple[int, int]:
    tb = MachTimebaseInfo()
    if mach_timebase_info is None or mach_timebase_info(C.byref(tb)) != 0 or tb.denom == 0:
        return 1, 1
    return tb.numer, tb.denom


TIMEBASE = _timebase()


def ticks_to_ns(ticks: int) -> int:
    """Convert Mach ticks to nanoseconds (125/3 on Apple Silicon, 1/1 on Intel)."""
    return ticks * TIMEBASE[0] // TIMEBASE[1]
