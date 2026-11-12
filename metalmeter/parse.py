"""Turn raw source output into metric fields.

Every function here is pure. It takes the dicts and lists that the sources return (or that a
fixture file holds) and never calls into macOS, so it runs, and is tested, on any platform.

A value that is missing or has an unexpected type becomes None. When a value that should be
present is absent, which usually means a macOS update renamed a registry key, the parser
calls the optional ``warn(key, detail)`` callback and leaves the reporting to the caller.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Union

Warn = Optional[Callable[[str, str], None]]
Number = Union[int, float]

MAX_PID = 2**31 - 1
_U64 = 2**64
THERMAL_STATES = ("nominal", "fair", "serious", "critical")
PRESSURE_LEVELS = (1, 2, 4)  # normal, warn, critical (kern.memorystatus_vm_pressure_level)

# PerformanceStatistics keys. "GPU Activity(%)" is the name used by Intel and AMD drivers.
UTIL_KEYS = ("Device Utilization %", "GPU Activity(%)")
PERF_FIELDS = {
    "sys_renderer_pct": "Renderer Utilization %",
    "sys_tiler_pct": "Tiler Utilization %",
    "sys_gpu_mem_in_use_bytes": "In use system memory",
    "sys_gpu_mem_alloc_bytes": "Alloc system memory",
}


def _no_warn(key: str, detail: str) -> None:
    pass


def _uint(v: Any) -> Optional[int]:
    """Return `v` if it is an int in [0, 2**64), else None. A bool does not count as an int."""
    return v if type(v) is int and 0 <= v < _U64 else None


def _num(v: Any) -> Optional[Number]:
    """Return `v` if it is a finite, non-negative int or float, else None."""
    if type(v) is int:
        return v if v >= 0 else None
    if type(v) is float:
        return v if math.isfinite(v) and v >= 0 else None
    return None


def parse_creator(s: Any) -> Optional[tuple[int, str]]:
    """Split an IOUserClientCreator string into PID and process name.

    ``"pid 410, WindowServer"`` gives ``(410, "WindowServer")``. Only ASCII digits between
    ``"pid "`` and the first comma are accepted; anything else gives None.
    """
    if not isinstance(s, str) or not s.startswith("pid "):
        return None
    digits, _, name = s[4:].partition(",")
    if not (digits.isascii() and digits.isdigit() and len(digits) <= 10):  # int() rejects > 4300 digits
        return None
    pid = int(digits)
    return (pid, name.strip()) if 0 < pid <= MAX_PID else None


def app_usage_ns(usage: Any, warn: Warn = None) -> Optional[int]:
    """Sum ``accumulatedGPUTime`` over the queues in one client's AppUsage array.

    Returns None if `usage` is not a list or if any entry lacks a valid value, so a client
    is never reported with a partial sum.
    """
    if not isinstance(usage, list):
        return None
    total = 0
    for q in usage:
        ns = _uint(q.get("accumulatedGPUTime")) if isinstance(q, dict) else None
        if ns is None:
            (warn or _no_warn)("accumulatedGPUTime", "AppUsage entry without a valid accumulatedGPUTime")
            return None
        total += ns
    return total


def gpu_clients(raw: Any, warn: Warn = None) -> Optional[dict[str, Any]]:
    """Parse the output of ``sources.gpu_clients.read()``.

    Returns:
        ``{"consistent": bool, "t_mono_ns": int | None, "no_usage": bool,
        "pids": {pid: {client_entry_id: gpu_ns | None}}}``, or None if `raw` is not a
        GPU client reading.

    Clients stay separate, keyed by registry entry ID, so the sampler can tell a closed
    client from a counter that went backwards. ``no_usage`` is True when there are clients
    but none of them has AppUsage. A client briefly loses AppUsage while its process exits,
    so a single such reading is normal; the sampler warns only when it persists.
    """
    if not isinstance(raw, dict) or not isinstance(raw.get("clients"), list):
        return None
    pids: dict[int, dict[int, Optional[int]]] = {}
    seen = with_usage = 0
    for c in raw["clients"]:
        if not isinstance(c, dict):
            continue
        p, eid = parse_creator(c.get("creator")), _uint(c.get("entry_id"))
        if p is None or eid is None:
            continue
        seen += 1
        with_usage += c.get("app_usage") is not None
        pids.setdefault(p[0], {})[eid] = app_usage_ns(c.get("app_usage"), warn)
    return {"consistent": raw.get("consistent") is True, "t_mono_ns": _uint(raw.get("t_mono_ns")),
            "no_usage": bool(seen) and not with_usage, "pids": pids}


def perf_stats(raw: Any, warn: Warn = None) -> dict[str, Optional[Number]]:
    """Parse the output of ``sources.perf_stats.read()`` into the ``sys_*`` GPU fields.

    Every field is None unless exactly one accelerator reports statistics. With more than
    one there is no right way to combine them, so the parser warns and returns no values.
    Missing keys only warn for Apple GPUs (class names starting with ``AGX``); other
    drivers do not have all of them.
    """
    w = warn or _no_warn
    out: dict[str, Optional[Number]] = {"sys_gpu_util_pct": None, **{f: None for f in PERF_FIELDS}}
    accs = [a for a in raw if isinstance(a, dict) and isinstance(a.get("perf"), dict)] if isinstance(raw, list) else []
    if len(accs) != 1:
        if len(accs) > 1:
            w("PerformanceStatistics", f"{len(accs)} accelerators report stats; system GPU fields not combined")
        return out
    acc, perf = accs[0], accs[0]["perf"]
    agx = isinstance(acc.get("class"), str) and acc["class"].startswith("AGX")
    util = next((perf[k] for k in UTIL_KEYS if k in perf), None)
    out["sys_gpu_util_pct"] = _num(util)
    if out["sys_gpu_util_pct"] is None and agx:
        w(UTIL_KEYS[0], "missing or not a number")
    for field, key in PERF_FIELDS.items():
        out[field] = _num(perf.get(key))
        if out[field] is None and agx:
            w(key, "missing or not a number")
    return out


def _timebase(tb: Any) -> Optional[tuple[int, int]]:
    if isinstance(tb, (list, tuple)) and len(tb) == 2 and all(type(x) is int and x > 0 for x in tb):
        return tb[0], tb[1]
    return None


def rusage(raw: Any, timebase: Any) -> Optional[dict[str, Optional[int]]]:
    """Parse the output of ``sources.process.read(pid)``.

    Args:
        raw: the process reading, with raw ``ri_*`` fields.
        timebase: ``(numer, denom)`` from mach_timebase_info, used to convert CPU time from
            Mach ticks to nanoseconds (125/3 on Apple Silicon, 1/1 on Intel).

    Returns:
        ``flavor``, ``start_abstime``, ``exit_abstime``, ``cpu_ns``, ``mem_footprint_bytes``,
        ``energy_nj`` and ``ane_mem_bytes``, or None if the read failed (``raw["errno"]``
        says why). ``exit_abstime`` is non-zero once the process has exited, even while it
        is still a zombie that its parent has not reaped. With rusage v4 (macOS 12) energy
        and Neural Engine memory are None.
    """
    ru = raw.get("rusage") if isinstance(raw, dict) else None
    if not isinstance(ru, dict) or type(ru.get("flavor")) is not int or ru["flavor"] not in (4, 6):
        return None
    tb = _timebase(timebase)
    user, system = _uint(ru.get("ri_user_time")), _uint(ru.get("ri_system_time"))
    v6 = ru["flavor"] == 6
    return {
        "flavor": ru["flavor"],
        "start_abstime": _uint(ru.get("ri_proc_start_abstime")),
        "exit_abstime": _uint(ru.get("ri_proc_exit_abstime")),
        "cpu_ns": None if tb is None or user is None or system is None else (user + system) * tb[0] // tb[1],
        "mem_footprint_bytes": _uint(ru.get("ri_phys_footprint")),
        "energy_nj": _uint(ru.get("ri_energy_nj")) if v6 else None,       # CPU only; GPU energy is not included
        "ane_mem_bytes": _uint(ru.get("ri_neural_footprint")) if v6 else None,
    }


def ane_pids(raw: Any) -> Optional[set[int]]:
    """Return the PIDs with an open Neural Engine connection, or None if the Mac has no ANE."""
    if not isinstance(raw, dict) or not isinstance(raw.get("clients"), list):
        return None
    out = set()
    for c in raw["clients"]:
        p = parse_creator(c.get("creator")) if isinstance(c, dict) else None
        if p is not None:
            out.add(p[0])
    return out


def system(raw: Any) -> dict[str, Any]:
    """Parse ``sources.system.read()``: memory pressure level, swap bytes and thermal state name."""
    r = raw if isinstance(raw, dict) else {}
    pressure, thermal = r.get("mem_pressure"), r.get("thermal_state")
    return {
        "mem_pressure": pressure if type(pressure) is int and pressure in PRESSURE_LEVELS else None,
        "swap_used_bytes": _uint(r.get("swap_used_bytes")),
        "swap_total_bytes": _uint(r.get("swap_total_bytes")),
        "thermal_state": THERMAL_STATES[thermal] if type(thermal) is int and 0 <= thermal < 4 else None,
    }


def _str(v: Any) -> Optional[str]:
    return v if isinstance(v, str) and v else None


def _flag(v: Any) -> Optional[bool]:
    return v == 1 if type(v) is int else None


def device_info(static_system: Any, static_accelerators: Any) -> dict[str, Any]:
    """Combine ``system.read_static()`` and ``perf_stats.read_static()`` into device facts."""
    s = static_system if isinstance(static_system, dict) else {}
    accs = static_accelerators if isinstance(static_accelerators, list) else []
    agx = [a for a in accs if isinstance(a, dict) and isinstance(a.get("class"), str) and a["class"].startswith("AGX")]
    cores = _uint(agx[0].get("gpu_core_count")) if len(agx) == 1 else None
    return {
        "chip": _str(s.get("cpu_brand")),
        "gpu_class": agx[0]["class"] if len(agx) == 1 else None,
        "gpu_cores": cores,
        "memory_bytes": _uint(s.get("memsize")),
        "macos": _str(s.get("os_version")),
        "macos_build": _str(s.get("os_build")),
        "apple_silicon": _flag(s.get("arm64")),
        "rosetta": _flag(s.get("translated")) is True,  # the sysctl does not exist where Rosetta is absent
        "vm": _flag(s.get("vmm_present")),
    }
