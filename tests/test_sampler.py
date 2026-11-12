"""Sampler tests for GPU clients, GPU time, processes and process trees.

The readings are hand-made dicts, so these tests run on any OS. Times are given in seconds
and converted to nanoseconds.
"""
import copy
import errno
import json
import pathlib
import warnings

import pytest

from metalmeter import _platform
from metalmeter.sampler import Sampler, snapshot_from
from metalmeter.types import OK, PARTIAL, PROCESS_EXITED, UNSUPPORTED

S = 1_000_000_000
ROOT, CHILD, OTHER = 10, 11, 12


@pytest.fixture(autouse=True)
def fresh_warnings():
    _platform._warned.clear()


def proc(start=1, cpu=0, energy=0, foot=1000, ane=0, err=0, flavor=6, exit=0):
    if err:
        return {"pid": 0, "rusage": None, "errno": err}
    return {"pid": 0, "errno": 0, "rusage": {
        "flavor": flavor, "ri_user_time": cpu, "ri_system_time": 0, "ri_phys_footprint": foot,
        "ri_proc_exit_abstime": exit,
        "ri_proc_start_abstime": start, "ri_energy_nj": energy, "ri_neural_footprint": ane}}


def reading(t, clients=(), procs=None, tree=None, consistent=True, abstime=None, gpu=True, ane_pids=(),
            perf=None, own_pid=None, own_memory=None, timebase=(1, 1)):
    """clients: (pid, entry_id, [ns per queue] or a raw AppUsage value, creator name)."""
    procs = {ROOT: proc()} if procs is None else procs
    rows = []
    for c in clients:
        pid, eid, usage = c[:3]
        name = c[3] if len(c) > 3 else "python"
        app_usage = ([{"API": "Metal", "accumulatedGPUTime": int(ns)} for ns in usage]  # the registry gives ints
                     if isinstance(usage, list) else usage)
        rows.append({"accelerator": 1, "entry_id": eid, "creator": f"pid {pid}, {name}", "app_usage": app_usage})
    return {
        "t_mono_ns": int(t * S), "t_wall": 1.0e9 + t, "abstime": abstime if abstime is not None else int(t * S),
        "tree": tree or [ROOT], "tree_ok": True, "timebase": list(timebase), "own_pid": own_pid,
        "macos_build": "26A428",
        "gpu_clients": {"t_mono_ns": int(t * S), "consistent": consistent, "accelerators": 1, "clients": rows}
        if gpu else None,
        "perf_stats": perf if perf is not None else [{"class": "AGXAcceleratorG14G", "perf": {
            "Device Utilization %": 50, "Renderer Utilization %": 40, "Tiler Utilization %": 30,
            "In use system memory": 1, "Alloc system memory": 2}}],
        "ane": {"providers": ["H1xANELoadBalancer"], "clients": [{"creator": f"pid {p}, x"} for p in ane_pids]},
        "system": {"mem_pressure": 1, "swap_used_bytes": 5, "swap_total_bytes": 9, "thermal_state": 0},
        "process": procs, "own_memory": own_memory,
    }


class Feed:
    """A reader that returns readings in order; its clock is the next reading's time."""

    def __init__(self, readings):
        self.readings, self.calls = list(readings), 0

    def __call__(self, pid, children):
        self.calls += 1
        return self.readings.pop(0)

    def clock(self):
        if not self.readings:
            return 10**15
        r = self.readings[0]
        return r["t_mono_ns"] if "t_mono_ns" in r else r["gpu_clients"]["t_mono_ns"]


def run(readings, pid=ROOT, children=False):
    feed = Feed(readings)
    s = Sampler(pid, children, reader=feed, clock=feed.clock)
    return [s.sample() for _ in range(len(readings))]


def gpu(seq_ns, t0=0):
    """One client (entry 100) of ROOT whose single queue reads seq_ns[i] at t = i seconds."""
    return [reading(t0 + i, clients=[(ROOT, 100, [ns])]) for i, ns in enumerate(seq_ns)]


# Counter rules
def test_first_sample_has_no_rates_but_measured_values():
    s = run([reading(0, clients=[(ROOT, 100, [5 * S])])])[0]
    assert s.status == OK and s.interval_s is None
    assert s.gpu_util_pct is None and s.cpu_util_pct is None and s.power_w is None
    assert s.gpu_time_ns == 0 and s.mem_footprint_bytes == 1000 and s.sys_gpu_util_pct == 50


def test_backwards_counter_gives_none_then_rates_resume():
    out = run(gpu([4 * S, 3 * S, 3.5 * S]))
    assert out[1].gpu_util_pct is None and out[1].gpu_util_raw is None
    assert out[2].gpu_util_pct == pytest.approx(50)


# GPU clients
def test_queue_that_closes_drops_out_of_the_delta():
    out = run([reading(0, clients=[(ROOT, 100, [5 * S]), (ROOT, 101, [2 * S])]),
               reading(1, clients=[(ROOT, 100, [5.5 * S])])])  # client 101 closed
    assert out[1].gpu_util_pct == pytest.approx(50)


def test_client_whose_own_total_goes_down_gives_none():
    out = run([reading(0, clients=[(ROOT, 100, [5 * S, 1 * S])]),
               reading(1, clients=[(ROOT, 100, [5 * S])])])  # one of its queues vanished
    assert out[1].gpu_util_pct is None


def test_invalidated_walk_is_partial_and_a_returning_client_is_not_a_spike():
    out = run([reading(0, clients=[(ROOT, 100, [1 * S]), (ROOT, 101, [1 * S])]),
               reading(1, clients=[(ROOT, 100, [1.5 * S])], consistent=False),   # 101 missed by the walk
               reading(2, clients=[(ROOT, 100, [2 * S]), (ROOT, 101, [2 * S])])])
    assert out[1].status == PARTIAL and out[1].gpu_util_pct == pytest.approx(50)
    assert out[2].status == OK and out[2].gpu_util_raw == pytest.approx(150)   # not 250: no lifetime spike


def test_many_clients_and_queues_are_summed():
    out = run([reading(0, clients=[(ROOT, 100, [0, 0]), (ROOT, 101, [0]), (ROOT, 102, [])]),
               reading(1, clients=[(ROOT, 100, [0.1 * S, 0.2 * S]), (ROOT, 101, [0.3 * S]), (ROOT, 102, [])])])
    assert out[1].gpu_util_pct == pytest.approx(60)


def test_client_opened_during_the_interval_counts_from_zero():
    out = run([reading(0, clients=[(ROOT, 100, [0])]),
               reading(1, clients=[(ROOT, 100, [0.2 * S]), (ROOT, 101, [0.3 * S])])])
    assert out[1].gpu_util_pct == pytest.approx(50)


def test_bad_creators_are_skipped_and_odd_names_still_parse():
    r0 = reading(0, clients=[(ROOT, 100, [0], "my, app, 名前")])
    r1 = reading(1, clients=[(ROOT, 100, [0.4 * S], "my, app, 名前")])
    r1["gpu_clients"]["clients"].append({"accelerator": 1, "entry_id": 900, "creator": "garbage",
                                         "app_usage": [{"accumulatedGPUTime": 9 * S}]})
    r1["gpu_clients"]["clients"].append({"accelerator": 1, "entry_id": 901, "creator": None, "app_usage": None})
    assert run([r0, r1])[1].gpu_util_pct == pytest.approx(40)


def test_clients_on_two_accelerators_are_summed():
    r0, r1 = reading(0, clients=[(ROOT, 100, [0]), (ROOT, 300, [0])]), \
        reading(1, clients=[(ROOT, 100, [0.25 * S]), (ROOT, 300, [0.25 * S])])
    r0["gpu_clients"]["clients"][1]["accelerator"] = r1["gpu_clients"]["clients"][1]["accelerator"] = 2
    assert run([r0, r1])[1].gpu_util_pct == pytest.approx(50)


def test_invalid_appusage_gives_none_and_keeps_the_baseline():
    out = run([reading(0, clients=[(ROOT, 100, [1 * S])]),
               reading(1, clients=[(ROOT, 100, "not a list")]),
               reading(2, clients=[(ROOT, 100, [2 * S])])])
    assert out[1].gpu_util_pct is None
    assert out[2].gpu_util_pct == pytest.approx(100)  # the 1 s gained since the last valid value lands here
    assert out[2].gpu_time_ns == 1 * S                 # nothing lost from the cumulative total


def test_client_first_seen_invalid_is_a_baseline_when_it_becomes_valid():
    out = run([reading(0, clients=[(ROOT, 100, "bad")]),
               reading(1, clients=[(ROOT, 100, "bad")]),
               reading(2, clients=[(ROOT, 100, [50 * S])]),
               reading(3, clients=[(ROOT, 100, [50.5 * S])])])
    assert out[2].gpu_util_pct == 0      # not 5000%: its lifetime total is only a baseline
    assert out[3].gpu_util_pct == pytest.approx(50)


# GPU time
def test_long_command_buffer_clamps_shown_pct_keeps_raw_and_window_is_exact():
    out = run(gpu([0, 0, 0, 3 * S, 3 * S]))  # one 3 s buffer completes between t=2 and t=3
    assert [s.gpu_util_pct for s in out] == [None, 0, 0, 100, 0]
    assert out[3].gpu_util_raw == pytest.approx(300)
    window = (out[-1].gpu_time_ns - out[0].gpu_time_ns) / (out[-1].t_mono_ns - out[0].t_mono_ns) * 100
    assert window == pytest.approx(75)  # 3 s of GPU over 4 s


def test_two_queues_at_once_raw_above_100_shown_at_most_100():
    out = run([reading(0, clients=[(ROOT, 100, [0, 0])]), reading(1, clients=[(ROOT, 100, [1 * S, 1 * S])])])
    assert out[1].gpu_util_raw == pytest.approx(200) and out[1].gpu_util_pct == 100


def test_readings_closer_than_0_1_s_return_the_last_sample():
    now = [0]
    feed = Feed([reading(0), reading(0.2)])
    s = Sampler(ROOT, reader=feed, clock=lambda: now[0])
    a = s.sample()
    now[0] = int(0.05 * S)
    assert s.sample() is a and feed.calls == 1   # no new reading
    now[0] = int(0.2 * S)
    assert s.sample() is not a and feed.calls == 2


# Processes
def test_watched_pid_reused_means_the_watched_process_exited():
    feed = Feed([reading(0, procs={ROOT: proc(start=1)}), reading(1, procs={ROOT: proc(start=999)})])
    s = Sampler(ROOT, reader=feed, clock=feed.clock)
    s.sample()
    gone = s.sample()
    assert gone.status == PROCESS_EXITED and gone.cpu_util_pct is None and gone.gpu_time_ns is None
    assert s.sample() is gone and feed.calls == 2  # sticky: no more readings


def test_watched_process_exits_gives_one_last_sample_with_system_context():
    out = run([reading(0), reading(1, procs={ROOT: proc(err=errno.ESRCH)})])
    assert out[1].status == PROCESS_EXITED
    assert out[1].thermal_state == "nominal" and out[1].sys_gpu_util_pct == 50


def test_a_zombie_counts_as_exited():
    # An exited process that its parent has not reaped still answers proc_pid_rusage;
    # only its exit time shows that it is gone.
    out = run([reading(0), reading(1, procs={ROOT: proc(exit=12345)})])
    assert out[1].status == PROCESS_EXITED


def test_another_users_process_has_gpu_time_but_no_rusage_fields():
    eperm = {ROOT: proc(err=errno.EPERM)}
    out = run([reading(0, clients=[(ROOT, 100, [0])], procs=eperm),
               reading(1, clients=[(ROOT, 100, [0.7 * S])], procs=eperm)])
    s = out[1]
    assert s.status == OK and s.gpu_util_pct == pytest.approx(70)
    assert s.cpu_util_pct is None and s.mem_footprint_bytes is None and s.energy_j is None and s.power_w is None


def test_cpu_time_in_mach_ticks_is_converted():
    tb = (125, 3)
    out = run([reading(0, procs={ROOT: proc(cpu=0)}, timebase=tb),
               reading(1, procs={ROOT: proc(cpu=24_000_000)}, timebase=tb)])  # 24e6 ticks = 1e9 ns
    assert out[1].cpu_util_pct == pytest.approx(100)


def test_energy_and_power():
    out = run([reading(0, procs={ROOT: proc(energy=0)}), reading(2, procs={ROOT: proc(energy=4 * S)})])
    assert out[1].power_w == pytest.approx(2.0) and out[1].energy_j == pytest.approx(4.0)


def test_macos_12_has_no_energy_and_no_ane_memory():
    v4 = {ROOT: proc(flavor=4)}
    s = run([reading(0, procs=v4), reading(1, procs=v4)])[1]
    assert s.power_w is None and s.energy_j is None and s.ane_mem_bytes is None and s.cpu_util_pct == 0


# Process tree
def test_tree_children_join_and_leave_and_split_self_from_children():
    readings = [
        reading(0, procs={ROOT: proc()}, tree=[ROOT], abstime=1000),
        # CHILD started (start 1500) after the last reading (abstime 1000): all of it is new.
        reading(1, tree=[ROOT, CHILD], abstime=2000,
                procs={ROOT: proc(cpu=int(0.2 * S)), CHILD: proc(start=1500, cpu=int(0.5 * S), foot=500)},
                clients=[(CHILD, 200, [int(0.3 * S)])]),
        # OTHER is older (start 500) but first seen now: baseline only, it adds nothing yet.
        reading(2, tree=[ROOT, CHILD, OTHER], abstime=3000,
                procs={ROOT: proc(cpu=int(0.2 * S)), CHILD: proc(start=1500, cpu=int(1.5 * S), foot=500),
                       OTHER: proc(start=500, cpu=50 * S, foot=7)},
                clients=[(CHILD, 200, [int(1.3 * S)]), (OTHER, 300, [40 * S])]),
        # CHILD exited: it leaves the tree and the delta.
        reading(3, tree=[ROOT, OTHER], abstime=4000,
                procs={ROOT: proc(cpu=int(0.2 * S)), OTHER: proc(start=500, cpu=int(50.25 * S), foot=7)},
                clients=[(OTHER, 300, [int(40.5 * S)])]),
    ]
    out = run(readings, children=True)
    one, two, three = out[1], out[2], out[3]
    assert one.children_count == 1 and one.cpu_util_pct == pytest.approx(70)
    assert one.children_cpu_util_pct == pytest.approx(50) and one.children_gpu_util_pct == pytest.approx(30)
    assert one.gpu_util_pct == pytest.approx(30) and one.mem_footprint_bytes == 1500
    assert one.children_mem_footprint_bytes == 500
    assert two.cpu_util_pct == pytest.approx(100) and two.gpu_util_pct == pytest.approx(100)  # OTHER: baseline
    assert three.children_count == 1 and three.cpu_util_pct == pytest.approx(25)
    assert three.gpu_util_pct == pytest.approx(50)


def test_parent_alone_reads_zero_while_its_child_does_the_gpu_work():
    tree = [ROOT, CHILD]
    procs = {ROOT: proc(), CHILD: proc(start=1)}
    r = [reading(0, tree=tree, procs=procs, clients=[(CHILD, 200, [0])]),
         reading(1, tree=tree, procs=procs, clients=[(CHILD, 200, [int(0.9 * S)])])]
    alone = run(copy.deepcopy(r), children=False)[1]
    whole = run(r, children=True)[1]
    assert alone.gpu_util_pct == 0 and alone.children_count is None
    assert whole.gpu_util_pct == pytest.approx(90)


def test_child_pid_reused_by_a_new_process_counts_from_its_start():
    tree = [ROOT, CHILD]
    out = run([reading(0, tree=tree, abstime=100, procs={ROOT: proc(), CHILD: proc(start=50, cpu=9 * S)}),
               reading(1, tree=tree, abstime=200, procs={ROOT: proc(), CHILD: proc(start=150, cpu=int(0.4 * S))})],
              children=True)
    assert out[1].cpu_util_pct == pytest.approx(40)  # not negative, not None


def test_failed_tree_listing_is_partial():
    r = [reading(0), reading(1)]
    r[1]["tree_ok"] = False
    assert run(r, children=True)[1].status == PARTIAL


# Platform, memory, warnings and output
def test_vm_or_ci_runner_is_unsupported_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = run([reading(0, gpu=False, perf=[]), reading(1, gpu=False, perf=[])])
    assert {s.status for s in out} == {UNSUPPORTED}
    assert out[1].gpu_util_pct is None and out[1].gpu_time_ns is None and out[1].sys_gpu_util_pct is None
    assert out[1].cpu_util_pct == 0  # process fields still work


def test_gpu_memory_only_for_our_own_pid():
    mem = {"current_allocated_size": 2**30, "recommended_max_working_set_size": 12 * 2**30}
    mine = run([reading(0, own_pid=ROOT, own_memory=mem)])[0]
    other = run([reading(0, own_pid=999, own_memory=mem)])[0]
    assert mine.gpu_mem_bytes == 2**30 and mine.gpu_mem_limit_bytes == 12 * 2**30
    assert other.gpu_mem_bytes is None and other.gpu_mem_limit_bytes is None


def test_ane_connected_and_memory():
    s = run([reading(0, procs={ROOT: proc(ane=4096)}, ane_pids=[ROOT])])[0]
    assert s.ane_connected is True and s.ane_mem_bytes == 4096
    assert run([reading(0)])[0].ane_connected is False


def test_renamed_key_warns_once_with_key_and_build():
    stats = {"Device Utilisation %": 50, "Renderer Utilization %": 40, "Tiler Utilization %": 30,
             "In use system memory": 1, "Alloc system memory": 2}  # one key renamed by a "macOS update"
    perf = [{"class": "AGXAcceleratorG14G", "perf": stats}]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = run([reading(0, perf=perf), reading(1, perf=perf), reading(2, perf=perf)])
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1 and "'Device Utilization %'" in messages[0] and "26A428" in messages[0]
    assert out[1].sys_gpu_util_pct is None and out[1].sys_renderer_pct == 40


def _no_usage(t):
    r = reading(t, clients=[(ROOT, 100, [0])])
    del r["gpu_clients"]["clients"][0]["app_usage"]
    return r


def test_a_closing_client_without_appusage_is_not_a_warning():
    # Observed on macOS 27: a client loses AppUsage for a moment while its process exits.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = run([reading(0, clients=[(ROOT, 100, [0])]), _no_usage(1), _no_usage(2), reading(3)])
    assert out[1].gpu_util_pct is None  # that interval cannot be measured


def test_appusage_missing_for_three_readings_warns_once():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run([_no_usage(t) for t in range(6)])
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1 and "'AppUsage'" in messages[0] and "26A428" in messages[0]


def test_to_dict_is_flat_prefixed_and_leaves_out_none():
    d = run([reading(0)])[0].to_dict(prefix="sys/")
    assert d["sys/status"] == OK and d["sys/mem_pressure"] == 1
    assert "sys/gpu_util_pct" not in d and all(k.startswith("sys/") for k in d)


def test_reset_starts_a_new_baseline():
    feed = Feed(gpu([0, 1 * S, 2 * S]))
    s = Sampler(ROOT, reader=feed, clock=feed.clock)
    s.sample()
    s.sample()
    s.reset()
    assert s.sample().gpu_util_pct is None


# Recorded fixtures as readings
FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "m2-macos27.0-torch-matmul.json"


def test_a_recorded_fixture_works_as_a_reading():
    r0 = json.loads(FIXTURE.read_text())
    pid = r0["pid"]
    r1 = copy.deepcopy(r0)
    r1["gpu_clients"]["t_mono_ns"] += S
    mine = next(c for c in r1["gpu_clients"]["clients"] if c["creator"].startswith(f"pid {pid},") and c["app_usage"])
    mine["app_usage"][0]["accumulatedGPUTime"] += S // 2
    r1["process"][str(pid)]["rusage"]["ri_user_time"] += 24_000_000  # 1 s of CPU in Mach ticks
    out = run([r0, r1], pid=pid)
    assert out[0].status == OK and out[0].gpu_time_ns == 0
    assert out[1].gpu_util_pct == pytest.approx(50) and out[1].cpu_util_pct == pytest.approx(100)
    snap = snapshot_from(r0, pid)
    assert snap.gpu_time_ns > 0 and snap.cpu_time_ns > 0 and snap.sys_gpu_util_pct is not None
