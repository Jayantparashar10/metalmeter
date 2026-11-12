"""Parser tests. Pure: fixtures and hand-made dicts only, so they run on Linux too."""
import ast
import copy
import json
import pathlib

import pytest

from metalmeter import parse

FIXTURES = sorted((pathlib.Path(__file__).parent / "fixtures").glob("*.json"))
LOADED = {f.stem: json.load(open(f)) for f in FIXTURES}


def by_load(name):
    return next(d for d in LOADED.values() if d["load"] == name)


class Warnings:
    def __init__(self):
        self.keys = []

    def __call__(self, key, detail):
        self.keys.append(key)


def parse_all(d, warn=None):
    """Every parser over one fixture: what the sampler will call."""
    pid = str(d.get("pid")) if isinstance(d, dict) else None
    proc = d.get("process", {}).get(pid) if isinstance(d.get("process"), dict) else None
    static = d.get("static") if isinstance(d.get("static"), dict) else {}
    return {
        "gpu": parse.gpu_clients(d.get("gpu_clients"), warn),
        "perf": parse.perf_stats(d.get("perf_stats"), warn),
        "rusage": parse.rusage(proc, d.get("timebase")),
        "ane": parse.ane_pids(d.get("ane")),
        "system": parse.system(d.get("system")),
        "device": parse.device_info(static.get("system"), static.get("accelerators")),
    }


# The parser module stays pure
def test_parse_imports_only_the_standard_library():
    tree = ast.parse(pathlib.Path(parse.__file__).read_text())
    imported = {n.module if isinstance(n, ast.ImportFrom) else a.name
                for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                for a in (n.names if isinstance(n, ast.Import) else [n])}
    assert imported <= {"__future__", "math", "typing"}, imported


def test_fixtures_present():
    assert {d["load"] for d in LOADED.values()} >= {"idle", "torch-matmul", "mlx-matmul", "coreml-ane"}


# Recorded fixtures give the expected fields
@pytest.mark.parametrize("name", sorted(LOADED))
def test_fixture_parses_without_warnings(name):
    w = Warnings()
    out = parse_all(LOADED[name], w)
    assert w.keys == []
    assert out["gpu"]["consistent"] is True and out["gpu"]["t_mono_ns"] > 0
    assert 0 <= out["perf"]["sys_gpu_util_pct"] <= 100
    assert out["perf"]["sys_gpu_mem_alloc_bytes"] > 0
    r = out["rusage"]
    assert r["flavor"] == 6 and r["cpu_ns"] > 0 and r["mem_footprint_bytes"] > 0 and r["start_abstime"] > 0
    assert r["energy_nj"] is not None and r["ane_mem_bytes"] is not None
    assert out["system"]["thermal_state"] in parse.THERMAL_STATES and out["system"]["mem_pressure"] in (1, 2, 4)
    dev = out["device"]
    assert dev["apple_silicon"] is True and dev["rosetta"] is False and dev["vm"] is False
    assert dev["gpu_cores"] > 0 and dev["gpu_class"].startswith("AGX") and dev["macos"]


@pytest.mark.parametrize("load", ["torch-matmul", "mlx-matmul"])
def test_gpu_loads_have_gpu_time_on_their_own_pid(load):
    d = by_load(load)
    clients = parse.gpu_clients(d["gpu_clients"])["pids"][d["pid"]]
    assert clients and all(ns is not None for ns in clients.values())
    assert sum(clients.values()) > 0
    assert d["pid"] not in parse.ane_pids(d["ane"])
    assert parse.perf_stats(d["perf_stats"])["sys_gpu_util_pct"] > 50


def test_idle_load_has_no_gpu_client_and_measured_zero_ane_memory():
    d = by_load("idle")
    assert d["pid"] not in parse.gpu_clients(d["gpu_clients"])["pids"]
    assert parse.rusage(d["process"][str(d["pid"])], d["timebase"])["ane_mem_bytes"] == 0


def test_coreml_load_shows_ane_on_its_own_pid():
    d = by_load("coreml-ane")
    assert d["pid"] in parse.ane_pids(d["ane"])
    assert parse.rusage(d["process"][str(d["pid"])], d["timebase"])["ane_mem_bytes"] > 0


def test_other_processes_names_are_hashed_in_fixtures():
    for d in LOADED.values():
        for c in d["gpu_clients"]["clients"] + d["ane"]["clients"]:
            pid, name = parse.parse_creator(c["creator"])
            assert pid == d["pid"] or name.startswith("h:"), c["creator"]


# Mutated fixtures never raise
MUTATIONS = [None, "x", b"\x00", [], {}, -1, 2**70, 1.5, float("nan"), True]


def _paths(node, path=()):
    yield path
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _paths(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node[:3]):  # first few clients/queues cover every shape
            yield from _paths(v, path + (i,))


def _get(node, path):
    for p in path:
        node = node[p]
    return node


@pytest.mark.parametrize("name", sorted(LOADED))
def test_every_key_deleted_or_retyped_never_raises(name):
    d = copy.deepcopy(LOADED[name])
    paths = [p for p in _paths(d) if p]
    assert len(paths) > 100
    for path in paths:
        parent, key = _get(d, path[:-1]), path[-1]
        original = parent[key]
        for m in MUTATIONS + ["<delete>"]:
            if m == "<delete>":
                if isinstance(parent, dict):
                    del parent[key]
                else:
                    continue
            else:
                parent[key] = m
            try:
                out = parse_all(d, Warnings())
            except Exception as e:  # pragma: no cover - the failure message is the point
                pytest.fail(f"{name}: {path} = {m!r} raised {e!r}")
            assert set(out["perf"]) == {"sys_gpu_util_pct", *parse.PERF_FIELDS}
            parent[key] = original
        if isinstance(parent, dict):
            parent[key] = original  # restore order-insensitive content after delete


@pytest.mark.parametrize("bad", MUTATIONS)
def test_whole_inputs_of_the_wrong_type_never_raise(bad):
    parse_all({"gpu_clients": bad, "perf_stats": bad, "process": bad, "ane": bad, "system": bad,
               "static": bad, "timebase": bad, "pid": bad})


# GPU client edge cases
@pytest.mark.parametrize("creator, expected", [
    ("pid 410, WindowServer", (410, "WindowServer")),
    ("pid 12, name, with, commas", (12, "name, with, commas")),
    ("pid 7, 名前 プロセス", (7, "名前 プロセス")),
    ("pid 7", (7, "")),
    ("pid  12, two spaces", None),
    ("pid -1, x", None),
    ("pid 0, kernel", None),
    ("pid ١٢, arabic digits", None),
    ("pid 99999999999, too big", None),
    ("pid " + "9" * 5000 + ", huge", None),
    ("PID 1, x", None),
    ("", None),
    (None, None),
    (b"pid 1, x", None),
])
def test_parse_creator(creator, expected):
    assert parse.parse_creator(creator) == expected


def client(pid, eid, usage):
    return {"accelerator": 1, "entry_id": eid, "creator": f"pid {pid}, p", "app_usage": usage}


def q(ns):
    return {"API": "Metal", "accumulatedGPUTime": ns}


def test_many_clients_and_queues_are_kept_per_client():
    raw = {"t_mono_ns": 5, "consistent": True, "clients": [
        client(10, 100, [q(1), q(2)]), client(10, 101, [q(4)]), client(10, 102, []), client(20, 200, [q(8)])]}
    out = parse.gpu_clients(raw)
    assert out["pids"] == {10: {100: 3, 101: 4, 102: 0}, 20: {200: 8}}


def test_two_accelerators_keep_both_clients():
    raw = {"consistent": True, "clients": [client(10, 100, [q(5)]), dict(client(10, 300, [q(7)]), accelerator=2)]}
    assert parse.gpu_clients(raw)["pids"][10] == {100: 5, 300: 7}


def test_client_without_creator_or_entry_id_is_skipped():
    raw = {"consistent": True, "clients": [{"entry_id": 1, "creator": None, "app_usage": [q(5)]},
                                           {"creator": "pid 3, x", "app_usage": [q(5)]},
                                           "not a dict"]}
    assert parse.gpu_clients(raw)["pids"] == {}


def test_inconsistent_walk_is_reported():
    assert parse.gpu_clients({"consistent": False, "clients": []})["consistent"] is False
    assert parse.gpu_clients({"clients": []})["consistent"] is False


@pytest.mark.parametrize("bad_queue", [{"API": "Metal"}, q(-1), q(True), q(1.5), q(b"\x01"), q(None), "x"])
def test_invalid_queue_makes_the_client_unknown_and_warns(bad_queue):
    w = Warnings()
    out = parse.gpu_clients({"consistent": True, "clients": [client(10, 100, [q(5), bad_queue])]}, w)
    assert out["pids"][10][100] is None  # never a partial sum
    assert w.keys == ["accumulatedGPUTime"]


def test_appusage_missing_on_every_client_is_flagged_not_warned():
    # A closing client loses AppUsage for a moment, so one reading is not proof of a renamed key;
    # the sampler warns only when it lasts.
    w = Warnings()
    raw = {"consistent": True,
           "clients": [{"entry_id": 1, "creator": "pid 1, a"}, {"entry_id": 2, "creator": "pid 2, b"}]}
    out = parse.gpu_clients(raw, w)
    assert out["pids"] == {1: {1: None}, 2: {2: None}} and out["no_usage"] is True and w.keys == []
    assert parse.gpu_clients({"consistent": True, "clients": [client(1, 1, [q(1)])]})["no_usage"] is False
    assert parse.gpu_clients({"consistent": True, "clients": []})["no_usage"] is False


# PerformanceStatistics
AGX_PERF = {"Device Utilization %": 40, "Renderer Utilization %": 30, "Tiler Utilization %": 20,
            "In use system memory": 100, "Alloc system memory": 200}


def test_perf_stats_reads_values_as_given():
    out = parse.perf_stats([{"class": "AGXAcceleratorG14G", "perf": AGX_PERF}])
    assert out == {"sys_gpu_util_pct": 40, "sys_renderer_pct": 30, "sys_tiler_pct": 20,
                   "sys_gpu_mem_in_use_bytes": 100, "sys_gpu_mem_alloc_bytes": 200}


def test_renamed_key_gives_none_and_warns_with_the_key_name():
    w = Warnings()
    perf = {k: v for k, v in AGX_PERF.items() if k != "Device Utilization %"}
    perf["Device Utilisation %"] = 40
    out = parse.perf_stats([{"class": "AGXAcceleratorG14G", "perf": perf}], w)
    assert out["sys_gpu_util_pct"] is None and w.keys == ["Device Utilization %"]


def test_new_value_type_gives_none_and_warns():
    w = Warnings()
    perf = dict(AGX_PERF, **{"In use system memory": b"\x01"})  # CFNumber became CFData
    out = parse.perf_stats([{"class": "AGXAcceleratorG14G", "perf": perf}], w)
    assert out["sys_gpu_mem_in_use_bytes"] is None and w.keys == ["In use system memory"]


def test_intel_fallback_key_and_no_warnings_for_missing_agx_keys():
    w = Warnings()
    out = parse.perf_stats([{"class": "IntelAccelerator", "perf": {"GPU Activity(%)": 12}}], w)
    assert out["sys_gpu_util_pct"] == 12 and out["sys_renderer_pct"] is None and w.keys == []


def test_several_accelerators_are_not_combined():
    w = Warnings()
    out = parse.perf_stats([{"class": "AGXA", "perf": AGX_PERF}, {"class": "AGXB", "perf": AGX_PERF}], w)
    assert set(out.values()) == {None} and w.keys == ["PerformanceStatistics"]


def test_accelerator_without_stats_is_ignored():
    out = parse.perf_stats([{"class": "AGXAcceleratorG14G", "perf": AGX_PERF}, {"class": "Other", "perf": None}])
    assert out["sys_gpu_util_pct"] == 40


def test_no_accelerator_gives_all_none():
    assert set(parse.perf_stats(None).values()) == {None}


# Processes
def raw_rusage(flavor=6, **over):
    ru = {"flavor": flavor, "ri_user_time": 300, "ri_system_time": 3, "ri_phys_footprint": 4096,
          "ri_proc_start_abstime": 77, "ri_energy_nj": 5, "ri_neural_footprint": 6}
    ru.update(over)
    return {"pid": 1, "rusage": ru, "errno": 0}


def test_cpu_time_is_converted_from_mach_ticks():
    r = parse.rusage(raw_rusage(), [125, 3])
    assert r["cpu_ns"] == 303 * 125 // 3 and r["mem_footprint_bytes"] == 4096 and r["start_abstime"] == 77


def test_intel_timebase_is_one_to_one():
    assert parse.rusage(raw_rusage(), [1, 1])["cpu_ns"] == 303


@pytest.mark.parametrize("tb", [None, [0, 1], [1, 0], [1], "125/3", [1.0, 1]])
def test_bad_timebase_gives_no_cpu_time(tb):
    r = parse.rusage(raw_rusage(), tb)
    assert r["cpu_ns"] is None and r["mem_footprint_bytes"] == 4096


def test_macos_12_v4_has_no_energy_and_no_ane_memory():
    r = parse.rusage(raw_rusage(flavor=4), [125, 3])
    assert r["energy_nj"] is None and r["ane_mem_bytes"] is None and r["cpu_ns"] is not None


def test_another_users_process_has_no_rusage():
    assert parse.rusage({"pid": 1, "rusage": None, "errno": 1}, [125, 3]) is None


@pytest.mark.parametrize("flavor", [5, 4.0, "6", None, True])
def test_unknown_flavor_is_rejected(flavor):
    assert parse.rusage(raw_rusage(flavor=flavor), [125, 3]) is None


# ANE, system, device
def test_ane_pids():
    raw = {"providers": ["H1xANELoadBalancer"], "clients": [{"creator": "pid 5, a"}, {"creator": None}, {}]}
    assert parse.ane_pids(raw) == {5}
    assert parse.ane_pids({"providers": ["H11ANEIn"], "clients": []}) == set()
    assert parse.ane_pids(None) is None  # no ANE (Intel, VM): None, not empty


@pytest.mark.parametrize("raw, expected", [
    ({"mem_pressure": 4, "swap_used_bytes": 1, "swap_total_bytes": 2, "thermal_state": 2},
     {"mem_pressure": 4, "swap_used_bytes": 1, "swap_total_bytes": 2, "thermal_state": "serious"}),
    ({"mem_pressure": 3, "thermal_state": 9}, {"mem_pressure": None, "swap_used_bytes": None,
                                               "swap_total_bytes": None, "thermal_state": None}),
    ({"mem_pressure": True, "thermal_state": True}, {"mem_pressure": None, "swap_used_bytes": None,
                                                     "swap_total_bytes": None, "thermal_state": None}),
])
def test_system(raw, expected):
    assert parse.system(raw) == expected


def test_device_info_for_a_vm_and_rosetta():
    vm = parse.device_info({"cpu_brand": "Apple M2 (Virtual)", "arm64": 1, "vmm_present": 1}, [])
    assert vm["vm"] is True and vm["gpu_cores"] is None and vm["gpu_class"] is None
    rosetta = parse.device_info({"arm64": 1, "translated": 1}, None)
    assert rosetta["rosetta"] is True and rosetta["apple_silicon"] is True


def test_device_info_intel():
    intel = parse.device_info({"cpu_brand": "Intel(R) Core(TM) i7", "arm64": None}, [{"class": "IntelAccelerator"}])
    assert intel["apple_silicon"] is None and intel["gpu_class"] is None and intel["rosetta"] is False
