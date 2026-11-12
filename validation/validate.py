"""Hardware validation: the accuracy checks, run through metalmeter's public API on a real Mac.

    python validation/validate.py                      # checks 1-8, about 6 minutes
    python validation/validate.py --only 4 6           # chosen checks
    python validation/validate.py --activity-monitor   # check 1 also asks for Activity Monitor's % GPU
    caffeinate -i python validation/validate.py soak --hours 8     # check 9, with the Mac kept awake

Each load runs in its own process and is watched from here, the way a user would watch a
training script. Results are merged into validation/results/<chip>-macos<version>-<build>.json,
so runs of single checks add up. Checks 1-6 need torch with MPS; check 1 also uses mlx, and
check 6 coremltools. A missing framework skips that part. Nothing here needs root.
"""
import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import metalmeter as mm  # noqa: E402
from metalmeter import parse  # noqa: E402
from metalmeter.sources import ane  # noqa: E402

LOADS = os.path.join(HERE, "loads.py")
RESULTS = os.path.join(HERE, "results")
READY_TIMEOUT_S = 180   # a Core ML conversion or a cold torch import can take a while
WARM_S = 3              # let a load settle before measuring it
GIB = 2**30


class Skip(Exception):
    pass


class Load:
    """A test load in its own process, started and waited on until it is ready."""

    def __init__(self, name, *args, needs=("torch",), seconds=7200):
        for module in needs:
            if importlib.util.find_spec(module) is None:
                raise Skip(f"{module} is not installed")
        self.err = tempfile.TemporaryFile("w+")
        self.p = subprocess.Popen([sys.executable, LOADS, name, "--seconds", str(seconds), *args],
                                  stdout=subprocess.PIPE, stderr=self.err, text=True)
        timer = threading.Timer(READY_TIMEOUT_S, self.p.kill)
        timer.start()
        try:
            line = self.p.stdout.readline()
        finally:
            timer.cancel()
        if not line:
            self.p.wait()
            self.err.seek(0)
            tail = self.err.read().strip().splitlines()
            raise Skip(f"{name} did not start: {tail[-1] if tail else 'no output'}")
        self.info = json.loads(line)
        self.pid = self.info["ready"]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.p.terminate()  # SIGTERM, so the spawn load can stop its child
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()
            self.p.wait()
        self.err.close()


def watch(targets, seconds, interval=0.5):
    """Watch several (pid, children) targets at once for `seconds`; return their WindowStats."""
    monitors = [mm.Monitor(pid=pid, children=children, interval=interval) for pid, children in targets]
    time.sleep(seconds)
    out = [m.stats(window=seconds + 5) for m in monitors]
    for m in monitors:
        m.stop()
    return out


def mean(w, field, digits=1):
    f = w.fields.get(field)
    return None if f is None or f.mean is None else round(f.mean, digits)


def result(check, name, expect, status, measured, note=None):
    return {"check": check, "name": name, "expect": expect, "status": status, "measured": measured,
            "note": note, "date": datetime.date.today().isoformat()}


def overall(rows, ok):
    """pass/fail over the parts that ran; skipped if none did."""
    ran = [r for r in rows.values() if "skipped" not in r]
    if not ran:
        return "skipped"
    return "pass" if all(ok(r) for r in ran) else "fail"


# 1. Own-process GPU %
def check_1(args):
    rows = {}
    for load, module in (("torch-matmul", "torch"), ("mlx-matmul", "mlx")):
        try:
            with Load(load, needs=(module,)) as L:
                time.sleep(WARM_S)
                m = mm.Monitor(pid=L.pid, interval=0.5)
                t0 = time.monotonic()
                am = None
                if args.activity_monitor:
                    am = ask_float(f"  Activity Monitor, CPU tab: what is % GPU for PID {L.pid} ({load})? ")
                time.sleep(max(0.0, 30 - (time.monotonic() - t0)))
                w = m.stats(window=60)
                m.stop()
            rows[load] = {"gpu_util_pct": mean(w, "gpu_util_pct"), "gpu_util_raw": mean(w, "gpu_util_raw"),
                          "sys_gpu_util_pct": mean(w, "sys_gpu_util_pct"), "activity_monitor_pct": am,
                          "seconds": round(w.window_s, 1)}
        except Skip as e:
            rows[load] = {"skipped": str(e)}

    def ok(r):
        close = r["activity_monitor_pct"] is None or abs(r["gpu_util_pct"] - r["activity_monitor_pct"]) <= 10
        return r["gpu_util_pct"] is not None and r["gpu_util_pct"] > 80 and close
    note = None if args.activity_monitor else "Activity Monitor comparison not done (run with --activity-monitor)"
    return result(1, "own-process GPU %", "mean gpu_util_pct > 80 over 30 s; within 10 points of Activity Monitor",
                  overall(rows, ok), rows, note)


def ask_float(prompt):
    while True:
        text = input(prompt).strip().rstrip("%")
        try:
            return float(text)
        except ValueError:
            print("  type a number, for example 85.3")


# 2. Isolation
def check_2(args):
    try:
        with Load("torch-matmul") as a, Load("idle", needs=()) as b, Load("torch-idle-client") as c:
            time.sleep(WARM_S)
            wa, wb, wc = watch([(a.pid, False), (b.pid, False), (c.pid, False)], 20)
    except Skip as e:
        return result(2, "isolation", "idle process B < 2% while A runs a load", "skipped", {}, str(e))
    measured = {"load_a_gpu_util_pct": mean(wa, "gpu_util_pct"),
                "idle_b_gpu_util_pct": mean(wb, "gpu_util_pct", 2),
                "idle_gpu_client_c_gpu_util_pct": mean(wc, "gpu_util_pct", 2)}
    ok = all(v is not None and v < 2 for v in (measured["idle_b_gpu_util_pct"],
                                                measured["idle_gpu_client_c_gpu_util_pct"]))
    return result(2, "isolation", "idle process B < 2% while A runs a load", "pass" if ok else "fail", measured,
                  "B never touches the GPU; C opened the GPU once and then sleeps")


# 3. Children
def check_3(args):
    try:
        with Load("spawn", "--child", "torch-matmul") as L:
            time.sleep(WARM_S)
            alone, tree = watch([(L.pid, False), (L.pid, True)], 20)
            os.kill(L.info["child"], signal.SIGKILL)
    except Skip as e:
        return result(3, "children", "parent alone 0%, children=True > 80%", "skipped", {}, str(e))
    measured = {"parent_alone_gpu_util_pct": mean(alone, "gpu_util_pct", 2),
                "tree_gpu_util_pct": mean(tree, "gpu_util_pct"),
                "tree_children_gpu_util_pct": mean(tree, "children_gpu_util_pct")}
    ok = measured["parent_alone_gpu_util_pct"] == 0 and (measured["tree_gpu_util_pct"] or 0) > 80
    return result(3, "children", "parent alone 0%, children=True > 80%", "pass" if ok else "fail", measured)


# 4. GPU memory
def check_4(args):
    try:
        with Load("torch-alloc", "--gib", "1") as L:
            info = L.info
    except Skip as e:
        return result(4, "GPU memory", "1 GiB allocation shows as 1 GiB +-5%", "skipped", {}, str(e))
    before, after, driver = info["mm_before"], info["mm_after"], info["driver_after"]
    grew = None if before is None or after is None else after - before
    measured = {"gpu_mem_bytes_before": before, "gpu_mem_bytes_after": after,
                "growth_gib": None if grew is None else round(grew / GIB, 4),
                "torch_driver_allocated_after": driver, "gpu_mem_limit_bytes": info["limit"]}
    ok = (grew is not None and abs(grew - GIB) <= 0.05 * GIB and abs(after - driver) <= 0.01 * driver)
    return result(4, "GPU memory", "gpu_mem_bytes grows by 1 GiB +-5% and matches torch.mps.driver_allocated_memory()",
                  "pass" if ok else "fail", measured, "measured inside the load: GPU memory is own-process only")


# 5. Energy
def check_5(args):
    rows = {}
    for load, needs in (("idle", ()), ("cpu-busy", ()), ("torch-bigkernel", ("torch",))):
        try:
            with Load(load, needs=needs) as L:
                time.sleep(WARM_S)
                (w,) = watch([(L.pid, False)], 20)
            rows[load] = {"power_w": mean(w, "power_w", 3), "cpu_util_pct": mean(w, "cpu_util_pct"),
                          "sys_gpu_util_pct": mean(w, "sys_gpu_util_pct")}
        except Skip as e:
            rows[load] = {"skipped": str(e)}
    gpu, cpu = rows["torch-bigkernel"], rows["cpu-busy"]
    if "skipped" in gpu or "skipped" in cpu or None in (gpu["power_w"], cpu["power_w"]):
        status, note = "skipped", "needs the cpu-busy and torch-bigkernel loads"
    elif (gpu["sys_gpu_util_pct"] or 0) > 80 and gpu["power_w"] < 0.25 * cpu["power_w"]:
        status, note = "recorded", "GPU energy is not included: a GPU-bound load reads close to idle"
    else:
        status, note = "recorded", "inconclusive: the GPU load did not read near idle"
    return result(5, "energy", "record whether ri_energy_nj includes GPU energy", status, rows, note)


# 6. Neural Engine
def check_6(args):
    try:
        with Load("coreml-ane", needs=("coremltools", "numpy")) as L:
            time.sleep(WARM_S)
            m = mm.Monitor(pid=L.pid, interval=0.5)
            time.sleep(10)
            s = m.read()
            m.stop()
            raw = ane.read()
            names = {p[1] for c in (raw or {}).get("clients", []) if (p := parse.parse_creator(c.get("creator")))}
            prefs = L.info.get("preferred_devices")
    except Skip as e:
        return result(6, "Neural Engine", "record where ANE use shows", "skipped", {}, str(e))
    measured = {"script_ane_connected": s.ane_connected, "script_ane_mem_bytes": s.ane_mem_bytes,
                "aned_has_client": "aned" in names, "preferred_devices": prefs}
    shows = bool(s.ane_connected and s.ane_mem_bytes)
    return result(6, "Neural Engine", "record where ANE use shows (script PID or aned)", "recorded", measured,
                  "shows on the script's PID" if shows else "does not show on the script's PID")


# 7. Fork
FORK_CODE = """
import json, multiprocessing as mp, sys, time
sys.path.insert(0, {root!r})
import metalmeter as mm

def child(q):
    s = mm.Monitor(background=False).read()
    q.put([m.running, s.status if s else None])

if __name__ == "__main__":
    m = mm.Monitor(interval=0.1)
    time.sleep(0.5)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=child, args=(q,))
    p.start()
    p.join(30)
    got = q.get(timeout=5) if p.exitcode == 0 else None
    print(json.dumps({{"exitcode": p.exitcode, "child": got, "parent_running": m.running}}))
    m.stop()
"""


def check_7(args):
    p = subprocess.run([sys.executable, "-W", "ignore::DeprecationWarning", "-c", FORK_CODE.format(root=ROOT)],
                       capture_output=True, text=True, timeout=90)
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return result(7, "fork", "no crash in a multiprocessing fork child", "fail",
                      {"returncode": p.returncode, "stderr": p.stderr.strip()[-500:]})
    child = d["child"] or [None, None]
    measured = {"child_exitcode": d["exitcode"], "inherited_monitor_running_in_child": child[0],
                "new_monitor_in_child_status": child[1], "parent_still_running": d["parent_running"]}
    ok = d["exitcode"] == 0 and child[0] is False and child[1] == "ok" and d["parent_running"]
    return result(7, "fork", "no crash in a multiprocessing fork child", "pass" if ok else "fail", measured)


# 8. Overhead
OVERHEAD_CODE = """
import json, resource, statistics, sys, time
sys.path.insert(0, {root!r})
import metalmeter as mm

m = mm.Monitor(interval=1.0)
time.sleep(2)
r0, t0 = resource.getrusage(resource.RUSAGE_SELF), time.monotonic()
time.sleep(30)
r1, elapsed = resource.getrusage(resource.RUSAGE_SELF), time.monotonic() - t0
m.stop()
cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
r = mm.Monitor(background=False)
r.read()
times = []
for _ in range(40):
    time.sleep(0.5)
    t = time.perf_counter()
    r.read()
    times.append((time.perf_counter() - t) * 1000)
times.sort()
print(json.dumps({{"cpu_pct_of_one_core": round(cpu / elapsed * 100, 3), "read_ms_median": round(times[20], 2),
                  "read_ms_p95": round(times[37], 2), "read_ms_max": round(times[-1], 2)}}))
"""


def check_8(args):
    p = subprocess.run([sys.executable, "-c", OVERHEAD_CODE.format(root=ROOT)], capture_output=True, text=True,
                       timeout=180)
    try:
        d = json.loads(p.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return result(8, "overhead", "< 1% of one core at 1 s; one read() < 5 ms", "fail",
                      {"returncode": p.returncode, "stderr": p.stderr.strip()[-500:]})
    ok = d["cpu_pct_of_one_core"] < 1 and d["read_ms_median"] < 5
    note = None if d["read_ms_max"] < 5 else f"the slowest of 40 reads took {d['read_ms_max']} ms"
    return result(8, "overhead", "< 1% of one core at 1 s; one read() < 5 ms (median of 40, 0.5 s apart)",
                  "pass" if ok else "fail", d, note)


# 9. Soak
def soak(args):
    """Watch this process and a light GPU load at 0.1 s for hours; log this process's footprint."""
    os.makedirs(RESULTS, exist_ok=True)
    log_path = os.path.join(RESULTS, f"{tag()}-soak.jsonl")
    seconds = args.hours * 3600
    with Load("torch-light", seconds=seconds + 600) as L, open(log_path, "w") as log:
        own = mm.Monitor(interval=0.1)
        load = mm.Monitor(pid=L.pid, interval=0.1)
        t0, wall0 = time.monotonic(), time.time()
        rows = []
        while time.monotonic() - t0 < seconds:
            time.sleep(min(60, max(1.0, seconds - (time.monotonic() - t0))))
            s, g = own.read(), load.read()
            row = {"t_s": round(time.monotonic() - t0), "t_wall": round(time.time(), 1),
                   "footprint_bytes": s.mem_footprint_bytes if s else None,
                   "load_gpu_util_pct": None if g is None or g.gpu_util_pct is None else round(g.gpu_util_pct, 1),
                   "own_cpu_pct": None if s is None or s.cpu_util_pct is None else round(s.cpu_util_pct, 2)}
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            log.flush()
        own.stop()
        load.stop()
    warm_s = min(600, seconds / 4)  # history buffers and caches fill up first
    warm = [r for r in rows if r["t_s"] >= warm_s and r["footprint_bytes"] is not None]
    measured = {"hours": args.hours, "wall_hours": round((time.time() - wall0) / 3600, 2), "log": log_path}
    if len(warm) < 3:
        return result(9, "soak", "footprint flat at 0.1 s", "fail", measured, "too few readings")
    xs, ys = [r["t_s"] / 3600 for r in warm], [r["footprint_bytes"] / 2**20 for r in warm]
    slope = statistics.linear_regression(xs, ys).slope if sys.version_info >= (3, 10) else None
    growth = ys[-1] - ys[0]
    cpu = [r["own_cpu_pct"] for r in rows if r["own_cpu_pct"] is not None]
    measured.update({"footprint_start_mib": round(ys[0], 2), "footprint_end_mib": round(ys[-1], 2),
                     "footprint_max_mib": round(max(ys), 2), "growth_mib": round(growth, 2),
                     "slope_mib_per_hour": None if slope is None else round(slope, 3),
                     "own_cpu_pct_mean": round(statistics.mean(cpu), 2) if cpu else None})
    ok = growth < 2 and (slope is None or slope < 0.5)
    return result(9, "soak", "footprint flat at 0.1 s: growth after warm-up < 2 MiB, slope < 0.5 MiB/h",
                  "pass" if ok else "fail", measured, f"first {warm_s / 60:.0f} minutes excluded as warm-up")


CHECKS = {1: check_1, 2: check_2, 3: check_3, 4: check_4, 5: check_5, 6: check_6, 7: check_7, 8: check_8}


def tag():
    d = mm.device_info()
    chip = (d.chip or "unknown").replace("Apple ", "").replace(" ", "").lower() if d else "unknown"
    return f"{chip}-macos{d.macos if d else '?'}-{d.macos_build if d else '?'}"


def versions():
    out = {"python": f"{platform.python_version()} {platform.machine()}", "metalmeter": mm.__version__}
    for pkg in ("torch", "mlx", "coremltools"):
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def save(results):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"{tag()}.json")
    try:
        with open(path) as f:
            doc = json.load(f)
    except (OSError, ValueError):
        doc = {"checks": {}}
    d = mm.device_info()
    doc.update({"device": d.to_dict() if d else None, "versions": versions()})
    for r in results:
        doc["checks"][str(r["check"])] = r
    doc["checks"] = dict(sorted(doc["checks"].items(), key=lambda kv: int(kv[0])))
    with open(path, "w") as f:
        json.dump(doc, f, indent=1)
    return path


def show(r):
    m = r["measured"]
    print(f"check {r['check']} {r['name']}: {r['status'].upper()}")
    for k, v in m.items():
        print(f"    {k}: {v}")
    if r["note"]:
        print(f"    note: {r['note']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", choices=("checks", "soak"), default="checks")
    ap.add_argument("--only", type=int, nargs="+", choices=sorted(CHECKS))
    ap.add_argument("--activity-monitor", action="store_true")
    ap.add_argument("--hours", type=float, default=8.0)
    args = ap.parse_args()
    if not mm.is_supported():
        print("validate: this Mac has no Apple GPU that metalmeter can read", file=sys.stderr)
        return 1
    print(f"metalmeter validation on {tag()}  ({versions()})")
    results = []
    if args.command == "soak":
        print(f"soak: {args.hours} h at 0.1 s; progress is logged every minute")
        results.append(soak(args))
        show(results[-1])
    else:
        for n in args.only or sorted(CHECKS):
            results.append(CHECKS[n](args))
            show(results[-1])
    print("results:", os.path.relpath(save(results), ROOT))
    return 0 if all(r["status"] != "fail" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
