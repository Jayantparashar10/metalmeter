"""Monitor: sample a process in the background or on demand, keep a history, summarise it.

Monitors that watch the same (pid, children) share one engine: one Sampler, one history
buffer and at most one sampling thread, which runs at the shortest interval any of them
asked for. A Monitor's ``on_sample`` callback runs on a thread of its own and receives at
most one Sample at a time, so a slow callback skips Samples (counted in ``dropped``)
instead of delaying the sampling or building a queue.

After a fork, inherited Monitors stop in the child: they have no thread and ``read()``
returns the last Sample seen before the fork. Monitors created in the child work normally.
"""
from __future__ import annotations

import atexit
import math
import os
import threading
import time
import weakref
from collections import deque
from typing import Any, Callable, Optional

from ._platform import warn_once
from .sampler import Sampler, device_info
from .types import PROCESS_EXITED, FieldStats, Sample, WindowStats

MIN_INTERVAL_S = 0.1
MAX_INTERVAL_S = 60.0
JOIN_TIMEOUT_S = 2.0

# Fields summarised by Monitor.stats(). Cumulative and bookkeeping fields are left out.
STATS_FIELDS = (
    "gpu_util_pct", "gpu_util_raw", "gpu_mem_bytes", "cpu_util_pct", "mem_footprint_bytes", "power_w",
    "ane_mem_bytes", "children_gpu_util_pct", "children_cpu_util_pct", "children_mem_footprint_bytes",
    "children_power_w", "sys_gpu_util_pct", "sys_renderer_pct", "sys_tiler_pct", "sys_gpu_mem_in_use_bytes",
    "sys_gpu_mem_alloc_bytes", "swap_used_bytes", "mem_pressure",
)

_registry_lock = threading.Lock()
_engines: dict[tuple[int, bool], _Engine] = {}
_monitors: weakref.WeakSet[Monitor] = weakref.WeakSet()
_shutting_down = False


class _Callback:
    """Runs one Monitor's on_sample on its own thread, one Sample at a time."""

    def __init__(self, fn: Callable[[Sample], Any]) -> None:
        self.fn = fn
        self.dropped = 0
        self.errors = 0
        self._pending: Optional[Sample] = None
        self._busy = False
        self._closed = False
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="metalmeter-callback", daemon=True)
        self._thread.start()

    def offer(self, s: Sample) -> None:
        with self._lock:
            if self._closed:
                return
            if self._busy or self._pending is not None:
                self.dropped += 1
                return
            self._pending = s
        self._ready.set()

    def _run(self) -> None:
        while True:
            self._ready.wait()
            with self._lock:
                if self._closed:
                    return
                s, self._pending, self._busy = self._pending, None, True
                self._ready.clear()
            try:
                if s is not None:
                    self.fn(s)
            except Exception as e:  # the user's callback must not stop sampling
                self.errors += 1
                warn_once(f"callback:{id(self)}", f"on_sample raised {e!r}; sampling continues")
            finally:
                with self._lock:
                    self._busy = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._ready.set()

    def after_fork(self) -> None:
        self._lock = threading.Lock()
        self._closed = True  # its thread does not exist in the child


class _Engine:
    """The Sampler, history and sampling thread shared by Monitors of one (pid, children)."""

    def __init__(self, pid: int, children: bool) -> None:
        self.key = (pid, children)
        self.sampler = Sampler(pid, children)
        self.lock = threading.Lock()
        self.history: deque[Sample] = deque(maxlen=1)
        self.monitors: list[Monitor] = []
        self.thread: Optional[threading.Thread] = None
        self.stop = threading.Event()
        self.wake = threading.Event()  # set when the interval changes or on stop
        self.finished = False   # the watched process exited
        self.forked = False     # inherited by a forked child: inert

    def interval(self) -> float:
        return min((m.interval for m in self.monitors if m.background), default=1.0)

    def latest(self) -> Optional[Sample]:
        with self.lock:
            return self.history[-1] if self.history else None

    def samples(self) -> list[Sample]:
        with self.lock:
            return list(self.history)

    def take(self) -> Optional[Sample]:
        """Take one Sample, add it to the history and hand it to the Monitors' callbacks."""
        try:
            s = self.sampler.sample()
        except Exception as e:  # a bug in metalmeter must not reach the training loop
            warn_once("monitor:sample", f"sampling failed ({e!r}); Samples are missing")
            return None
        with self.lock:
            new = not self.history or self.history[-1] is not s
            if new:
                self.history.append(s)
            monitors = list(self.monitors)
        if new:
            for m in monitors:
                m._deliver(s)
        if s.status == PROCESS_EXITED:
            self.finished = True
        return s

    def run(self, stop: threading.Event) -> None:
        while not stop.is_set() and not _shutting_down and not self.finished:
            started = time.monotonic_ns()
            s = self.take()
            # Count the interval from the Sample's own timestamp, which is set partway through
            # the reading. Counting from the start of the reading lets jitter bring two
            # timestamps just under 0.1 s apart, and the sampler then returns the previous
            # Sample, losing that cycle.
            base_ns = s.t_mono_ns if s is not None else started
            # Wait out the interval, rechecking it when woken: a Monitor with a shorter
            # interval may have joined, or the last background Monitor may have stopped.
            while not stop.is_set():
                left = (base_ns - time.monotonic_ns()) / 1e9 + self.interval()
                if left <= 0:
                    break
                if self.wake.wait(left):
                    self.wake.clear()

    def halt(self) -> None:
        self.stop.set()
        self.wake.set()

    def start_thread(self) -> None:
        if self.finished or self.forked or (self.thread is not None and self.thread.is_alive()):
            return
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, args=(self.stop,), name="metalmeter-sampler", daemon=True)
        self.thread.start()

    def resize(self) -> None:
        # Only grow: a stopped Monitor can still call stats() on what it collected.
        size = max((m.history_size for m in self.monitors), default=1)
        with self.lock:
            if (self.history.maxlen or 0) < size:
                self.history = deque(self.history, maxlen=size)

    def after_fork(self) -> None:
        self.forked = True
        self.thread = None
        self.lock = threading.Lock()
        self.sampler._lock = threading.Lock()


def _attach(m: Monitor) -> _Engine:
    with _registry_lock:
        engine = _engines.get(m._key)
        if engine is None:
            engine = _engines[m._key] = _Engine(*m._key)
        engine.monitors.append(m)
        engine.resize()
        if m.background:
            engine.start_thread()
            engine.wake.set()  # a running thread picks up a shorter interval now
        return engine


def _detach(m: Monitor) -> None:
    thread = None
    with _registry_lock:
        engine = m._engine
        if m in engine.monitors:
            engine.monitors.remove(m)
        if not any(x.background for x in engine.monitors):
            engine.halt()
            thread = engine.thread
        if not engine.monitors and _engines.get(engine.key) is engine:
            del _engines[engine.key]
    if thread is not None and thread is not threading.current_thread():
        thread.join(JOIN_TIMEOUT_S)


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def window_stats(samples: list[Sample], window_s: float) -> WindowStats:
    """Summarise the Samples taken in the last `window_s` seconds (by monotonic time)."""
    if not samples:
        return WindowStats(window_s=0.0, samples=0)
    cut = samples[-1].t_mono_ns - window_s * 1e9
    w = [s for s in samples if s.t_mono_ns >= cut]
    first, last = w[0], w[-1]
    elapsed_ns = last.t_mono_ns - first.t_mono_ns
    fields = {}
    for name in STATS_FIELDS:
        values = [float(v) for s in w if (v := getattr(s, name)) is not None]
        if values:
            fields[name] = FieldStats(mean=sum(values) / len(values), max=max(values), p95=_p95(values))
    # Means from cumulative counters: exact even when a long command buffer lands in one Sample.
    if elapsed_ns > 0 and first.gpu_time_ns is not None and last.gpu_time_ns is not None:
        raw = (last.gpu_time_ns - first.gpu_time_ns) / elapsed_ns * 100
        for name, mean in (("gpu_util_raw", raw), ("gpu_util_pct", min(raw, 100.0))):
            old = fields.get(name, FieldStats())
            fields[name] = FieldStats(mean=mean, max=old.max, p95=old.p95)
    if elapsed_ns > 0 and first.energy_j is not None and last.energy_j is not None:
        old = fields.get("power_w", FieldStats())
        fields["power_w"] = FieldStats(mean=(last.energy_j - first.energy_j) / (elapsed_ns / 1e9),
                                       max=old.max, p95=old.p95)
    return WindowStats(window_s=elapsed_ns / 1e9, samples=len(w), fields=fields, warnings=_warnings(w))


def _warnings(w: list[Sample]) -> tuple[str, ...]:
    out = []
    pressure = max((s.mem_pressure for s in w if s.mem_pressure is not None), default=None)
    if pressure == 4:
        out.append("memory pressure reached critical")
    elif pressure == 2:
        out.append("memory pressure reached warning")
    swaps = [s.swap_used_bytes for s in w if s.swap_used_bytes is not None]
    if len(swaps) >= 2 and swaps[-1] > swaps[0]:
        out.append(f"swap use grew by {(swaps[-1] - swaps[0]) / 2**20:.0f} MiB")
    thermal = {s.thermal_state for s in w}
    for state in ("critical", "serious"):
        if state in thermal:
            out.append(f"thermal state reached {state}")
            break
    return tuple(out)


def _environment_warnings() -> None:
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        warn_once("root", "running as root is not needed: metalmeter only reads")
    info = device_info()
    if info is not None and info.rosetta:
        warn_once("rosetta", "this Python runs as x86_64 under Rosetta; an arm64 Python is recommended")


class Monitor:
    """Watch one process, and optionally its descendants, and keep a history of Samples.

    The Monitor starts when it is created. Use it as a context manager, or call ``stop()``.

    Args:
        pid: the process to watch; the current process by default.
        interval: seconds between Samples in the background, clamped to 0.1 .. 60.
        children: include every descendant of `pid` (see ``Sample.children_*``).
        background: sample on a thread. If False there is no thread, and ``read()`` takes a
            Sample when called.
        on_sample: called with each new Sample, on a separate thread. While it runs, newer
            Samples are skipped and counted in ``dropped``.
        history: number of Samples kept for ``read()`` and ``stats()``.

    Methods never raise into the caller: failures become a warning and a missing value.
    """

    def __init__(self, pid: Optional[int] = None, interval: float = 1.0, children: bool = False,
                 background: bool = True, on_sample: Optional[Callable[[Sample], Any]] = None,
                 history: int = 600) -> None:
        self.pid = os.getpid() if pid is None else int(pid)
        self.interval = min(max(float(interval), MIN_INTERVAL_S), MAX_INTERVAL_S)
        self.children = bool(children)
        self.background = bool(background)
        self.history_size = max(1, int(history))
        self._key = (self.pid, self.children)
        self._callback = _Callback(on_sample) if on_sample is not None else None
        self._last_delivered: Optional[int] = None
        self._stopped = False
        _environment_warnings()
        self._engine = _attach(self)
        _monitors.add(self)

    @property
    def running(self) -> bool:
        """True while the Monitor can take new Samples."""
        e = self._engine
        if self._stopped or e.forked or e.finished:
            return False
        return not self.background or (e.thread is not None and e.thread.is_alive())

    @property
    def dropped(self) -> int:
        """Samples skipped because on_sample was still busy with an earlier one."""
        return self._callback.dropped if self._callback else 0

    @property
    def callback_errors(self) -> int:
        """Times on_sample raised."""
        return self._callback.errors if self._callback else 0

    def read(self) -> Optional[Sample]:
        """Return the newest Sample. With ``background=False``, take one first.

        A call within 0.1 s of the previous reading returns that Sample. None only if no
        Sample could be taken.
        """
        try:
            e = self._engine
            if self._stopped or e.forked or e.finished:
                return e.latest()
            if not self.background or e.latest() is None:
                return e.take() or e.latest()
            return e.latest()
        except Exception as exc:
            warn_once("monitor:read", f"read() failed ({exc!r})")
            return None

    def stats(self, window: float = 60.0) -> WindowStats:
        """Summarise the Samples of the last `window` seconds: mean, max and p95 per field."""
        try:
            return window_stats(self._engine.samples(), float(window))
        except Exception as exc:
            warn_once("monitor:stats", f"stats() failed ({exc!r})")
            return WindowStats(window_s=0.0, samples=0)

    def stop(self) -> None:
        """Stop this Monitor. The sampling thread ends when no background Monitor needs it."""
        if self._stopped:
            return
        self._stopped = True
        if self._callback is not None:
            self._callback.close()
        if not self._engine.forked:
            _detach(self)

    def __enter__(self) -> Monitor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def __repr__(self) -> str:
        state = "running" if self.running else "stopped"
        return (f"<metalmeter.Monitor pid={self.pid} interval={self.interval}s children={self.children} "
                f"background={self.background} {state} samples={len(self._engine.history)}>")

    def _deliver(self, s: Sample) -> None:
        """Hand a new Sample to on_sample, at most once per this Monitor's interval."""
        if self._callback is None or self._stopped:
            return
        due = self._last_delivered is None or s.t_mono_ns - self._last_delivered >= self.interval * 0.95e9
        if due or not self.background:
            self._last_delivered = s.t_mono_ns
            self._callback.offer(s)


def stop_all() -> None:
    """Stop every Monitor in this process, for example before re-running a notebook cell."""
    for m in list(_monitors):
        m.stop()


def _after_fork_in_child() -> None:
    global _registry_lock
    _registry_lock = threading.Lock()
    for engine in _engines.values():
        engine.after_fork()
    _engines.clear()
    for m in list(_monitors):
        if m._callback is not None:
            m._callback.after_fork()


def _shutdown() -> None:
    """Stop sampling threads at interpreter exit, before finalisation starts."""
    global _shutting_down
    _shutting_down = True
    with _registry_lock:
        engines = list(_engines.values())
    for m in list(_monitors):
        if m._callback is not None:
            m._callback.close()
    for engine in engines:
        engine.halt()
    for engine in engines:
        if engine.thread is not None:
            engine.thread.join(JOIN_TIMEOUT_S)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork_in_child)
atexit.register(_shutdown)
