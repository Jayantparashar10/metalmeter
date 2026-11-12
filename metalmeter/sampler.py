"""Turn successive readings of a process into Samples with rates.

This is the only stateful part of metalmeter. ``read_live`` does all the I/O and returns a
plain reading dict: the fixture format plus ``tree``, ``tree_ok``, ``abstime``, ``t_wall``,
``own_pid`` and ``macos_build``. ``Sampler`` only does arithmetic on readings, so tests can
drive it with recorded or hand-written ones.

Counter rules:

* A rate needs two readings. The first Sample has no rates (None, not 0).
* A counter that goes backwards makes that Sample's rate None, never negative.
* A process is identified by its PID and start time (``ri_proc_start_abstime``). If the
  watched PID's start time changes, the watched process has exited and its PID was reused.
  In the child tree, a new start time is simply a new child.

GPU time is tracked per GPU client (registry entry ID). A client that closes drops out of
the delta instead of making the process total go down.
"""
from __future__ import annotations

import errno
import os
import threading
import time
from typing import Any, Callable, Iterable, Optional

from . import parse
from ._bind import libproc
from ._platform import warn_once
from .sources import ane, gpu_clients, perf_stats, process, system
from .types import OK, PARTIAL, PROCESS_EXITED, UNSUPPORTED, DeviceInfo, Sample, Snapshot

MIN_INTERVAL_NS = 100_000_000  # below 0.1 s the counters are mostly noise
NO_USAGE_READINGS = 3  # clients without AppUsage this many readings in a row suggests a renamed key
Reading = dict[str, Any]
_static_build: list[Optional[str]] = []


def _macos_build() -> Optional[str]:
    if not _static_build:
        static = system.read_static()
        _static_build.append(static.get("os_build") if static else None)
    return _static_build[0]


def read_live(pid: int, children: bool) -> Reading:
    """Read every source once for `pid`, and for its descendants if `children` is True."""
    tree, tree_ok = [pid], True
    if children:
        kids = process.descendants(pid)
        tree_ok = kids is not None
        tree += kids or []
    g = gpu_clients.read(tree)
    t_mono = g["t_mono_ns"] if g else time.monotonic_ns()
    abstime = libproc.abstime()
    own = os.getpid()
    own_is_client = pid == own and g is not None and any(
        (parse.parse_creator(c.get("creator")) or (0,))[0] == own for c in g["clients"])
    return {
        "t_mono_ns": t_mono, "t_wall": time.time(), "abstime": abstime, "tree": tree, "tree_ok": tree_ok,
        "timebase": list(process.timebase()), "own_pid": own, "macos_build": _macos_build(),
        "gpu_clients": g, "perf_stats": perf_stats.read(), "ane": ane.read(tree), "system": system.read(),
        "process": {p: process.read(p) for p in tree},
        "own_memory": gpu_clients.own_memory(own_is_client),  # creates an MTLDevice only if we use the GPU
    }


def _delta(old: Optional[int], new: Optional[int]) -> Optional[int]:
    return None if old is None or new is None or new < old else new - old


def _sum(values: Iterable[Optional[int]]) -> Optional[int]:
    total = 0
    for v in values:
        if v is None:
            return None
        total += v
    return total


def _rate(delta: Optional[int], dt_ns: Optional[int], scale: float) -> Optional[float]:
    return None if delta is None or not dt_ns else delta / dt_ns * scale


def _common(r: Reading, pid: int, warn: Callable[[str, str], None]) -> dict[str, Any]:
    """Fields that need no earlier reading: system context, and GPU memory when `pid` is us."""
    gpu_present = r.get("gpu_clients") is not None
    perf = parse.perf_stats(r.get("perf_stats"), warn if gpu_present else None)  # stay quiet on VMs
    own = r.get("own_memory") if isinstance(r.get("own_memory"), dict) and r.get("own_pid") == pid else {}
    return {
        **perf, **{k: v for k, v in parse.system(r.get("system")).items() if k != "swap_total_bytes"},
        "gpu_mem_bytes": parse._uint(own.get("current_allocated_size")),
        "gpu_mem_limit_bytes": parse._uint(own.get("recommended_max_working_set_size")),
    }


def _process_raw(r: Reading, pid: int) -> Any:
    procs = r.get("process") if isinstance(r.get("process"), dict) else {}
    return procs.get(pid, procs.get(str(pid)))  # JSON fixtures have string keys


def _errno(raw: Any) -> int:
    e = raw.get("errno") if isinstance(raw, dict) else None
    return e if type(e) is int else 0


def _reading_time(r: Reading, gpu: Optional[dict[str, Any]]) -> Optional[int]:
    """The reading's own timestamp, else the GPU walk's (fixtures carry only the latter)."""
    t = parse._uint(r.get("t_mono_ns"))
    return gpu["t_mono_ns"] if t is None and gpu is not None else t


def _tree(r: Reading, pid: int) -> list[int]:
    tree = r.get("tree")
    if not isinstance(tree, list) or not tree or tree[0] != pid:
        return [pid]
    return [p for p in tree if type(p) is int]


class Sampler:
    """Produce Samples for one process, or for a process and all its descendants.

    Args:
        pid: the process to watch.
        children: if True, every metric covers the whole process tree and the
            ``children_*`` fields hold the descendants' share.
        reader: function that returns a reading; defaults to ``read_live``.
        clock: monotonic clock in nanoseconds, used for the minimum interval.

    The Sampler is thread-safe.
    """

    def __init__(self, pid: int, children: bool = False,
                 reader: Optional[Callable[[int, bool], Reading]] = None,
                 clock: Callable[[], int] = time.monotonic_ns) -> None:
        self.pid, self.children = pid, children
        self._read = reader or read_live
        self._clock = clock
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Drop every baseline, as after a fork. The next Sample has no rates."""
        self._prev: Optional[dict[str, Any]] = None
        self._gpu_ns = 0
        self._energy_nj = 0
        self._no_usage = 0
        self._last: Optional[Sample] = None

    def _warn(self, key: str, detail: str, build: Optional[str] = None) -> None:
        warn_once(f"key:{key}", f"{detail} (key {key!r}, macOS build {build or 'unknown'}); field is None")

    def sample(self) -> Sample:
        """Take a reading and return a Sample.

        A call less than 0.1 s after the previous reading returns the previous Sample.
        Once the watched process has exited, every call returns the exit Sample.
        """
        with self._lock:
            last = self._last
            if last is not None and (last.status == PROCESS_EXITED or
                                     self._clock() - last.t_mono_ns < MIN_INTERVAL_NS):
                return last
            self._last = self._update(self._read(self.pid, self.children))
            return self._last

    def _update(self, r: Reading) -> Sample:
        build = r.get("macos_build") if isinstance(r.get("macos_build"), str) else None

        def warn(key: str, detail: str) -> None:
            self._warn(key, detail, build)

        gpu = parse.gpu_clients(r.get("gpu_clients"), warn)
        self._no_usage = self._no_usage + 1 if gpu is not None and gpu["no_usage"] else 0
        if self._no_usage >= NO_USAGE_READINGS:
            warn("AppUsage", f"GPU clients without AppUsage in {NO_USAGE_READINGS} readings in a row")
        t = _reading_time(r, gpu)
        t = self._clock() if t is None else t
        t_wall = r.get("t_wall") if isinstance(r.get("t_wall"), float) else time.time()
        base = {"t_wall": t_wall, "t_mono_ns": t, "pid": self.pid, **_common(r, self.pid, warn)}
        tree = _tree(r, self.pid) if self.children else [self.pid]
        tree_set = set(tree)
        prev = self._prev
        timebase = r.get("timebase")

        # CPU time and energy, per process.
        root_raw = _process_raw(r, self.pid)
        fields = {p: parse.rusage(_process_raw(r, p), timebase) for p in tree}
        root = fields[self.pid]
        root_start = root["start_abstime"] if root else None
        old_root = prev["procs"].get(self.pid) if prev else None
        if (_errno(root_raw) == errno.ESRCH or (root and root["exit_abstime"])
                or (old_root and root_start is not None and old_root[0] != root_start)):
            return Sample(status=PROCESS_EXITED, **base)  # gone, a zombie, or its PID was reused

        procs_now: dict[int, tuple[Optional[int], Optional[int], Optional[int]]] = {}
        d_cpu: dict[int, Optional[int]] = {}
        d_energy: dict[int, Optional[int]] = {}
        new_procs: set[int] = set()
        for p, f in fields.items():
            if f is None:
                continue
            cur = (f["start_abstime"], f["cpu_ns"], f["energy_nj"])
            procs_now[p] = cur
            if prev is None:
                continue
            old = prev["procs"].get(p)
            if old is not None and old[0] == cur[0]:
                d_cpu[p], d_energy[p] = _delta(old[1], cur[1]), _delta(old[2], cur[2])
            elif cur[0] is not None and prev["abstime"] is not None and cur[0] >= prev["abstime"]:
                new_procs.add(p)  # started since the last reading, so all of its usage is new
                d_cpu[p], d_energy[p] = cur[1], cur[2]
            # Otherwise this is an older process seen for the first time: a baseline only.

        # GPU time, per client.
        clients_now: dict[int, tuple[int, Optional[int]]] = {}
        if gpu is not None:
            for p, by_client in gpu["pids"].items():
                if p in tree_set:
                    for eid, ns in by_client.items():
                        clients_now[eid] = (p, ns)
        d_gpu: dict[int, int] = {}
        gpu_known = prev is not None and gpu is not None and prev["gpu_present"]
        if gpu_known and prev is not None:
            known_pids = prev["tree"] | {self.pid}
            for eid, (p, ns) in clients_now.items():
                old_c = prev["clients"].get(eid)
                if ns is None:
                    gpu_known = False
                    continue
                if old_c is not None and old_c[0] == p:
                    if old_c[1] is None:
                        d = 0  # earlier readings of this client were invalid; this one is its baseline
                    elif ns < old_c[1]:
                        gpu_known = False  # the client's own counter went backwards
                        continue
                    else:
                        d = ns - old_c[1]
                elif p in new_procs or p in known_pids:
                    d = ns  # the client was opened since the last reading
                else:
                    d = 0   # an older process seen for the first time: a baseline only
                d_gpu[p] = d_gpu.get(p, 0) + d

        # Baselines for the next reading. After an invalidated registry walk, clients the walk
        # missed keep their old baselines; otherwise a returning client would look new and its
        # whole lifetime would be counted as one interval.
        old_clients = prev["clients"] if prev else {}
        keep_clients = dict(old_clients) if gpu is not None and not gpu["consistent"] else {}
        for eid, (p, ns) in clients_now.items():
            if ns is not None:
                keep_clients[eid] = (p, ns)
            else:  # keep the old baseline, or record the client as seen without one
                keep_clients[eid] = old_clients.get(eid, (p, None))
        self._prev = {"t_mono_ns": t, "abstime": parse._uint(r.get("abstime")), "procs": procs_now,
                      "clients": keep_clients, "tree": tree_set, "gpu_present": gpu is not None}

        # Rates.
        dt = t - prev["t_mono_ns"] if prev else None
        dt = dt if dt and dt > 0 else None
        kids = [p for p in tree if p != self.pid]
        gpu_total = _sum(d_gpu.values()) if gpu_known else None
        gpu_kids = sum(d_gpu.get(p, 0) for p in kids) if gpu_known else None
        if gpu_total is not None and dt is not None:
            self._gpu_ns += gpu_total
        cpu_total = _sum(d_cpu.values()) if root is not None and self.pid in d_cpu else None
        energy_total = _sum(d_energy.values()) if root is not None and self.pid in d_energy else None
        if energy_total is not None and dt is not None:
            self._energy_nj += energy_total
        raw_pct = _rate(gpu_total, dt, 100)
        kids_rusage = [fields[p] for p in kids if fields[p] is not None]
        has_energy = root is not None and root["energy_nj"] is not None

        status = OK
        if gpu is None:
            status = UNSUPPORTED
        elif not gpu["consistent"] or r.get("tree_ok") is False or len(kids_rusage) < len(kids) or (
                root is None and _errno(root_raw) not in (0, errno.EPERM)):
            status = PARTIAL

        return Sample(
            status=status, interval_s=dt / 1e9 if dt else None, **base,
            gpu_time_ns=self._gpu_ns if gpu is not None else None,
            gpu_client_count=len(clients_now) if gpu is not None else None,
            gpu_util_raw=raw_pct, gpu_util_pct=None if raw_pct is None else min(raw_pct, 100.0),
            cpu_util_pct=_rate(cpu_total, dt, 100),
            mem_footprint_bytes=_sum(f["mem_footprint_bytes"] for f in [root, *kids_rusage]) if root else None,
            energy_j=self._energy_nj / 1e9 if has_energy else None,
            power_w=_rate(energy_total, dt, 1),  # nJ / ns = W
            ane_mem_bytes=_sum(f["ane_mem_bytes"] for f in [root, *kids_rusage]) if root else None,
            ane_connected=_ane_connected(r, tree_set),
            children_count=len(kids) if self.children else None,
            children_gpu_util_pct=(None if gpu_kids is None or dt is None else min(gpu_kids / dt * 100, 100.0))
            if self.children else None,
            children_cpu_util_pct=_rate(_sum(d_cpu.get(p, 0) for p in kids), dt, 100)
            if self.children and cpu_total is not None else None,
            children_mem_footprint_bytes=_sum(f["mem_footprint_bytes"] for f in kids_rusage)
            if self.children else None,
            children_power_w=_rate(_sum(d_energy.get(p, 0) for p in kids), dt, 1)
            if self.children and energy_total is not None else None,
        )


def _ane_connected(r: Reading, pids: set[int]) -> Optional[bool]:
    connected = parse.ane_pids(r.get("ane"))
    return None if connected is None else bool(connected & pids)


def snapshot_from(r: Reading, pid: int) -> Snapshot:
    """Build a Snapshot of `pid`'s raw cumulative counters from one reading."""
    def warn(key: str, detail: str) -> None:
        warn_once(f"key:{key}", f"{detail} (key {key!r}); field is None")

    gpu = parse.gpu_clients(r.get("gpu_clients"), warn)
    raw = _process_raw(r, pid)
    f = parse.rusage(raw, r.get("timebase"))
    common = _common(r, pid, warn)
    t_mono = _reading_time(r, gpu)
    if gpu is None:
        status = UNSUPPORTED
    elif _errno(raw) == errno.ESRCH or (f and f["exit_abstime"]):
        status = PROCESS_EXITED
    else:
        status = OK if gpu["consistent"] else PARTIAL
    return Snapshot(
        t_wall=r.get("t_wall") if isinstance(r.get("t_wall"), float) else time.time(),
        t_mono_ns=time.monotonic_ns() if t_mono is None else t_mono, pid=pid, status=status,
        gpu_time_ns=_sum(gpu["pids"].get(pid, {}).values()) if gpu else None,
        cpu_time_ns=f["cpu_ns"] if f else None, energy_nj=f["energy_nj"] if f else None,
        mem_footprint_bytes=f["mem_footprint_bytes"] if f else None, ane_mem_bytes=f["ane_mem_bytes"] if f else None,
        ane_connected=_ane_connected(r, {pid}), **common,
    )


def snapshot(pid: Optional[int] = None) -> Snapshot:
    """Read every source once and return the raw counters of `pid` (default: this process)."""
    pid = os.getpid() if pid is None else pid
    return snapshot_from(read_live(pid, False), pid)


_device: list[Optional[DeviceInfo]] = []


def device_info() -> Optional[DeviceInfo]:
    """Return static facts about this Mac, read once. None off macOS."""
    if not _device:
        static = system.read_static()
        _device.append(DeviceInfo(**parse.device_info(static, perf_stats.read_static())) if static else None)
    return _device[0]


def is_supported() -> bool:
    """True on Apple Silicon with an Apple GPU, the case metalmeter is built and tested for."""
    info = device_info()
    return bool(info and info.apple_silicon and info.gpu_class)
