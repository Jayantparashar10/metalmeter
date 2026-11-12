"""Monitor tests: threads, sharing, lifecycle, fork, interpreter exit, overhead, window stats.

These run against the live sources. Off macOS every Sample is "unsupported", which is enough
to exercise the threading and lifecycle code.
"""
import multiprocessing
import os
import resource
import subprocess
import sys
import threading
import time
import warnings

import pytest

import metalmeter as mm
from metalmeter import _platform
from metalmeter.monitor import _Callback, window_stats
from metalmeter.types import PROCESS_EXITED, Sample
from tests.conftest import ROOT, macos, needs_gpu

S = 1_000_000_000


def sampler_threads():
    return [t for t in threading.enumerate() if t.name == "metalmeter-sampler" and t.is_alive()]


def wait_for(condition, timeout=5.0):
    """Poll until `condition()` is true. CI machines can be slow, so tests wait for an outcome
    with a generous deadline instead of sleeping a fixed time and counting."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.02)
    return True


@pytest.fixture(autouse=True)
def clean():
    _platform._warned.clear()
    yield
    mm.stop_all()
    deadline = time.monotonic() + 3
    while sampler_threads() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert not sampler_threads(), "a sampler thread outlived its Monitors"


# Construction and reading
def test_defaults_and_interval_clamp():
    with mm.Monitor() as m:
        assert m.pid == os.getpid() and m.interval == 1.0 and m.running
    assert mm.Monitor(interval=0.001, background=False).interval == 0.1
    assert mm.Monitor(interval=10_000, background=False).interval == 60.0


def test_background_sampling_fills_the_history_at_the_interval():
    with mm.Monitor(interval=0.1) as m:
        assert wait_for(lambda: len(m._engine.history) >= 5)
        samples = list(m._engine.history)
    # Samples are never closer than the interval, and none is lost to the sampler's 0.1 s guard.
    assert all(b.t_mono_ns - a.t_mono_ns >= 100_000_000 for a, b in zip(samples, samples[1:]))


def test_read_right_after_start_returns_a_sample():
    with mm.Monitor(interval=5) as m:
        assert isinstance(m.read(), Sample)


def test_without_background_there_is_no_thread_and_read_samples():
    with mm.Monitor(background=False) as m:
        assert sampler_threads() == []
        a = m.read()
        assert m.read() is a          # within 0.1 s: the same Sample
        time.sleep(0.12)
        b = m.read()
        assert b is not a and b.interval_s is not None


def test_history_is_bounded():
    seen = []
    with mm.Monitor(interval=0.1, history=3, on_sample=seen.append) as m:
        assert wait_for(lambda: len(seen) >= 5)
        assert len(m._engine.history) == 3


def test_stats_still_work_after_stop():
    with mm.Monitor(interval=0.1) as m:
        assert wait_for(lambda: len(m._engine.history) >= 3)
    w = m.stats(window=60)
    assert w.samples >= 3 and not m.running


# Lifecycle: shared samplers, stop_all, repr
def test_monitors_of_the_same_process_share_one_sampler_and_thread():
    a = mm.Monitor(interval=0.2)
    b = mm.Monitor(interval=0.2)
    c = mm.Monitor(interval=0.2, children=True)
    assert a._engine is b._engine and a._engine is not c._engine
    assert len(sampler_threads()) == 2  # one for (pid, False), one for (pid, True)
    a.stop()
    time.sleep(0.3)
    assert b.running and len(sampler_threads()) == 2
    b.stop()
    c.stop()


def test_the_shortest_interval_drives_sampling_and_callbacks_keep_their_own_pace():
    slow_calls, fast_calls = [], []
    slow = mm.Monitor(interval=1.0, on_sample=slow_calls.append)
    assert wait_for(lambda: slow_calls)  # the thread is now inside its 1 s wait
    fast = mm.Monitor(interval=0.1, on_sample=fast_calls.append)
    # Without waking the thread, the first fast Sample would wait out the rest of the 1 s.
    assert wait_for(lambda: len(fast_calls) >= 2, timeout=0.6)
    assert wait_for(lambda: len(slow_calls) >= 2)
    slow.stop()
    fast.stop()
    # Each Monitor's callback keeps its own pace (5% slack), however fast the thread samples.
    assert all(b.t_mono_ns - a.t_mono_ns >= 0.95e9 for a, b in zip(slow_calls, slow_calls[1:]))


def test_stop_all_and_repr():
    ms = [mm.Monitor(interval=0.5) for _ in range(3)]
    assert "running" in repr(ms[0])
    mm.stop_all()
    assert not any(m.running for m in ms) and "stopped" in repr(ms[0])
    ms[0].stop()  # stopping twice is harmless


# Threads: callbacks
def test_a_raising_callback_warns_once_and_sampling_continues():
    def bad(s):
        raise ValueError("boom")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with mm.Monitor(interval=0.1, on_sample=bad) as m:
            assert wait_for(lambda: m.callback_errors >= 3)  # sampling went on after the first error
    assert len([w for w in caught if "on_sample raised" in str(w.message)]) == 1


def test_a_busy_callback_drops_samples_instead_of_queueing():
    # Driven by hand, without timing: the callback blocks until released.
    release, started, calls = threading.Event(), threading.Event(), []

    def blocked(s):
        calls.append(s)
        started.set()
        release.wait(5)
    cb = _Callback(blocked)
    s1, s2, s3 = (sample(t) for t in (1, 2, 3))
    cb.offer(s1)
    assert started.wait(5)
    cb.offer(s2)                        # the callback is busy: dropped, not queued
    cb.offer(s3)
    assert cb.dropped == 2
    release.set()
    assert wait_for(lambda: not cb._busy)
    cb.offer(s3)                        # free again: delivered
    assert wait_for(lambda: len(calls) == 2) and calls == [s1, s3]
    cb.close()


def test_concurrent_reads_and_stats_while_sampling():
    errors = []

    def hammer(m):
        try:
            for _ in range(200):
                m.read()
                m.stats(window=5)
        except Exception as e:  # pragma: no cover - the failure is the point
            errors.append(e)
    with mm.Monitor(interval=0.1) as m:
        threads = [threading.Thread(target=hammer, args=(m,)) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert errors == []


# The watched process exits (detected through libproc, so macOS only)
@macos
@pytest.mark.allow_subprocess
def test_the_thread_stops_when_the_watched_process_exits():
    seen = []
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.4)"])
    m = mm.Monitor(pid=child.pid, interval=0.1, on_sample=seen.append)
    child.wait()
    deadline = time.monotonic() + 3
    while m.running and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not m.running and m.read().status == PROCESS_EXITED
    time.sleep(0.1)
    assert seen and seen[-1].status == PROCESS_EXITED
    m.stop()


@macos
@pytest.mark.allow_subprocess
def test_an_unreaped_child_is_seen_as_exited():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        m = mm.Monitor(pid=child.pid, interval=0.1)
        deadline = time.monotonic() + 5
        while m.running and time.monotonic() < deadline:  # the child is a zombie: not waited for yet
            time.sleep(0.05)
        assert not m.running and m.read().status == PROCESS_EXITED
        m.stop()
    finally:
        child.wait()


# Fork. Python 3.12+ warns that forking a multi-threaded process may deadlock the child;
# these tests create exactly that situation on purpose.
FORK_WARNING = "ignore:This process .* is multi-threaded:DeprecationWarning"


def _wait(pid, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.02)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    return "hung"


@pytest.mark.allow_subprocess
@pytest.mark.filterwarnings(FORK_WARNING)
def test_fork_after_start_leaves_an_inert_monitor_and_no_thread_in_the_child():
    m = mm.Monitor(interval=0.1)
    time.sleep(0.25)
    before = m.read()
    pid = os.fork()
    if pid == 0:  # child: report through the exit code only
        code = 0
        try:
            if sampler_threads() or m.running or m.read() is not before:
                code = 2
            child_mon = mm.Monitor(interval=0.1)   # a new Monitor in the child works
            if not wait_for(lambda: len(child_mon._engine.history) >= 2) or not child_mon.running:
                code = 3
            child_mon.stop()
            m.stop()
        except BaseException:
            code = 4
        os._exit(code)
    assert _wait(pid) == 0
    assert m.running  # the parent is unaffected
    m.stop()


def _mp_child(q):
    q.put((len(sampler_threads()), mm.Monitor(background=False).read() is not None))


@pytest.mark.allow_subprocess
@pytest.mark.filterwarnings(FORK_WARNING)
def test_multiprocessing_fork_does_not_crash_the_child():
    m = mm.Monitor(interval=0.1)
    time.sleep(0.2)
    ctx = multiprocessing.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_mp_child, args=(q,))
    p.start()
    p.join(10)
    assert p.exitcode == 0 and q.get(timeout=5) == (0, True)
    m.stop()


# Interpreter exit with sampling running
EXIT_SCRIPT = """
import sys, time
sys.path.insert(0, {root!r})
import metalmeter as mm
m = mm.Monitor(interval=0.1, on_sample=lambda s: time.sleep({callback_s}))
time.sleep(0.35)
"""


@pytest.mark.allow_subprocess
@pytest.mark.parametrize("callback_s", [0, 5])
def test_interpreter_exits_cleanly_with_a_monitor_running(callback_s):
    t = time.monotonic()
    p = subprocess.run([sys.executable, "-c", EXIT_SCRIPT.format(root=ROOT, callback_s=callback_s)],
                       capture_output=True, text=True, timeout=20)
    assert p.returncode == 0 and p.stderr == ""
    assert time.monotonic() - t < 4  # not held up by the sampler or by a callback in flight


# Overhead. Budgets for a real Mac: on a shared CI virtual machine the numbers say little.
# validation/validate.py checks them on hardware before each release.
@needs_gpu
def test_overhead_at_one_second_is_under_one_percent_of_a_core():
    with mm.Monitor(interval=1.0):
        time.sleep(0.2)
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t0 = time.monotonic()
        time.sleep(5)
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        elapsed = time.monotonic() - t0
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    assert cpu / elapsed < 0.01, f"{cpu / elapsed:.2%} of one core"


@needs_gpu
def test_one_read_takes_under_5_ms():
    # Sleeping between reads, as real use does, lets macOS move the thread to a slower core,
    # which makes a read several times dearer than back to back. The budget is for the
    # typical read; the slowest one gets a looser bound so the test is not flaky.
    with mm.Monitor(background=False) as m:
        m.read()
        times = []
        for _ in range(15):
            time.sleep(0.11)
            t = time.perf_counter()
            m.read()
            times.append(time.perf_counter() - t)
    times.sort()
    assert times[len(times) // 2] < 0.005, f"median read {times[len(times) // 2] * 1000:.2f} ms"
    assert times[-1] < 0.010, f"slowest read {times[-1] * 1000:.2f} ms"


# Window statistics (pure)
def sample(t, **kw):
    return Sample(t_wall=float(t), t_mono_ns=int(t * S), pid=1, status="ok", **kw)


def test_gpu_mean_comes_from_cumulative_time():
    gpu = [0, 0, 0, 3 * S, 3 * S]
    pct = [None, 0.0, 0.0, 100.0, 0.0]
    w = window_stats([sample(t, gpu_time_ns=g, gpu_util_pct=p) for t, (g, p) in enumerate(zip(gpu, pct))], 60)
    assert w.samples == 5 and w.window_s == 4
    assert w.fields["gpu_util_pct"].mean == pytest.approx(75)   # not 25, the mean of the shown values
    assert w.fields["gpu_util_pct"].max == 100 and w.fields["gpu_util_raw"].mean == pytest.approx(75)


def test_power_mean_comes_from_cumulative_energy():
    w = window_stats([sample(0, energy_j=0.0, power_w=None), sample(2, energy_j=6.0, power_w=9.0)], 60)
    assert w.fields["power_w"].mean == pytest.approx(3.0) and w.fields["power_w"].max == 9.0


def test_window_keeps_only_recent_samples_and_p95():
    samples = [sample(t, cpu_util_pct=float(t)) for t in range(100)]
    w = window_stats(samples, window_s=19)
    assert w.samples == 20 and w.fields["cpu_util_pct"].mean == pytest.approx(89.5)
    assert w.fields["cpu_util_pct"].p95 == 98 and w.fields["cpu_util_pct"].max == 99


def test_warnings_for_pressure_swap_and_thermal_state():
    w = window_stats([sample(0, mem_pressure=1, swap_used_bytes=0, thermal_state="nominal"),
                      sample(1, mem_pressure=2, swap_used_bytes=512 * 2**20, thermal_state="serious")], 60)
    assert w.warnings == ("memory pressure reached warning", "swap use grew by 512 MiB",
                          "thermal state reached serious")


def test_empty_window_and_to_dict():
    assert window_stats([], 60).samples == 0
    d = window_stats([sample(0, cpu_util_pct=5.0), sample(1, cpu_util_pct=7.0)], 60).to_dict(prefix="sys/")
    assert d == {"sys/cpu_util_pct/mean": 6.0, "sys/cpu_util_pct/max": 7.0, "sys/cpu_util_pct/p95": 7.0}
