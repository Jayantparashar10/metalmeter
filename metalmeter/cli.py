"""The ``metalmeter`` command.

    metalmeter                          system GPU, memory and thermal state, every second
    metalmeter --pid 1234               one process
    metalmeter --pid 1234 --children --json --interval 0.5
    metalmeter info                     this Mac, and which sources can be read
    metalmeter dump > mac.json          one raw reading, for bug reports and test fixtures
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any, Optional

from . import __version__, parse
from .monitor import Monitor
from .sampler import device_info, is_supported
from .sources import ane, gpu_clients, perf_stats, process, system
from .types import PROCESS_EXITED, UNSUPPORTED, Sample

PRESSURE_NAMES = {1: "normal", 2: "warning", 4: "critical"}


def _size(v: Optional[float]) -> str:
    if v is None:
        return "-"
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if v >= scale:
            return f"{v / scale:.1f} {unit}"
    return f"{v:.0f} B"


def _pct(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.1f}%"


def system_line(s: Sample) -> str:
    return (f"GPU {_pct(s.sys_gpu_util_pct)}  renderer {_pct(s.sys_renderer_pct)}  tiler {_pct(s.sys_tiler_pct)}  "
            f"GPU memory {_size(s.sys_gpu_mem_in_use_bytes)} in use, {_size(s.sys_gpu_mem_alloc_bytes)} allocated  "
            f"pressure {PRESSURE_NAMES.get(s.mem_pressure, '-')}  "
            f"swap {_size(s.swap_used_bytes)}  thermal {s.thermal_state or '-'}")


def process_line(s: Sample) -> str:
    gpu = "no GPU clients" if s.gpu_client_count == 0 else f"GPU {_pct(s.gpu_util_pct)}"
    parts = [f"pid {s.pid}", gpu, f"CPU {_pct(s.cpu_util_pct)}", f"memory {_size(s.mem_footprint_bytes)}",
             f"CPU power {'-' if s.power_w is None else f'{s.power_w:.2f} W'}",
             f"ANE memory {_size(s.ane_mem_bytes)}"]
    if s.gpu_mem_bytes is not None:
        parts.insert(3, f"GPU memory {_size(s.gpu_mem_bytes)}")
    if s.children_count is not None:
        parts.append(f"children {s.children_count} (GPU {_pct(s.children_gpu_util_pct)})")
    parts.append(f"| system GPU {_pct(s.sys_gpu_util_pct)}, thermal {s.thermal_state or '-'}")
    return "  ".join(parts)


def watch(pid: Optional[int], children: bool, interval: float, as_json: bool, count: Optional[int]) -> int:
    mon = Monitor(pid=pid, interval=interval, children=children, background=False)
    try:
        first = mon.read()
        if first is not None and first.status == PROCESS_EXITED:
            print(f"metalmeter: no process with pid {mon.pid}", file=sys.stderr)
            return 1
        if first is not None and first.status == UNSUPPORTED:
            print("metalmeter: no Apple GPU found; GPU fields are empty", file=sys.stderr)
        shown = 0
        while count is None or shown < count:
            time.sleep(mon.interval)
            s = mon.read()
            if s is None:
                continue
            if as_json:
                print(json.dumps(s.to_dict()), flush=True)
            elif s.status != PROCESS_EXITED:
                print(process_line(s) if pid is not None else system_line(s), flush=True)
            if s.status == PROCESS_EXITED:
                if not as_json:
                    print(f"process {s.pid} exited", flush=True)
                return 0
            shown += 1
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        mon.stop()


def _redact(rows: Optional[list[dict[str, Any]]], keep: set[int]) -> None:
    for r in rows or []:
        p = parse.parse_creator(r.get("creator"))
        if p and p[0] not in keep:
            r["creator"] = f"pid {p[0]}, h:{hashlib.sha256(p[1].encode()).hexdigest()[:12]}"


def dump_reading(pid: int, load: Optional[str] = None) -> dict[str, Any]:
    """One raw reading of every source, in the test fixture format (format 1).

    Every GPU and Neural Engine client is included, but only `pid` keeps its process name;
    the others are replaced by a hash.
    """
    g, a = gpu_clients.read(), ane.read()
    _redact(g["clients"] if g else None, {pid})
    _redact(a["clients"] if a else None, {pid})
    return {
        "format": 1, "captured": datetime.date.today().isoformat(), "load": load, "pid": pid,
        "metalmeter": __version__, "timebase": list(process.timebase()),
        "static": {"system": system.read_static(), "accelerators": perf_stats.read_static()},
        "gpu_clients": g, "perf_stats": perf_stats.read(), "ane": a, "system": system.read(),
        "process": {str(pid): process.read(pid)},
    }


def _json_default(v: Any) -> Any:
    if isinstance(v, bytes):
        return {"bytes": v.hex()}
    raise TypeError(f"cannot serialise {type(v).__name__}")


def info() -> dict[str, Any]:
    """This Mac and the state of each source."""
    g, perf, a, sysr = gpu_clients.read(), perf_stats.read(), ane.read(), system.read()
    own = process.read(os.getpid())
    dev = device_info()
    return {
        "metalmeter": __version__,
        "python": f"{platform.python_version()} {platform.machine()}",
        "supported": is_supported(),
        "device": dev.to_dict() if dev else None,
        "sources": {
            "gpu_clients": None if g is None else {"clients": len(g["clients"]), "consistent": g["consistent"]},
            "gpu_statistics": None if perf is None else [p["class"] for p in perf if p.get("perf") is not None],
            "process_counters": None if own is None else {"rusage_flavor": (own["rusage"] or {}).get("flavor"),
                                                          "errno": own["errno"]},
            "neural_engine": None if a is None else a["providers"],
            "system": sysr is not None,
        },
    }


def print_info(d: dict[str, Any]) -> None:
    dev, src = d["device"] or {}, d["sources"]
    cores = f"{dev['gpu_cores']}-core GPU" if dev.get("gpu_cores") else "no Apple GPU"
    print(f"metalmeter {d['metalmeter']}, Python {d['python']}")
    print(f"chip          {dev.get('chip', '-')} ({cores}, {dev.get('gpu_class') or '-'})")
    print(f"memory        {_size(dev.get('memory_bytes'))}")
    print(f"macOS         {dev.get('macos', '-')} ({dev.get('macos_build', '-')})")
    print(f"supported     {'yes' if d['supported'] else 'no'}"
          f"{'  (Rosetta)' if dev.get('rosetta') else ''}{'  (virtual machine)' if dev.get('vm') else ''}")
    g, pc = src["gpu_clients"], src["process_counters"]
    rows = [
        ("GPU clients", "unavailable" if g is None else
         f"ok, {g['clients']} clients{'' if g['consistent'] else ', registry changed during the walk'}"),
        ("GPU statistics", "unavailable" if not src["gpu_statistics"] else f"ok ({', '.join(src['gpu_statistics'])})"),
        ("process counters", "unavailable" if pc is None or pc["rusage_flavor"] is None else
         f"ok (rusage v{pc['rusage_flavor']})"),
        ("Neural Engine", "not present" if src["neural_engine"] is None else f"ok ({', '.join(src['neural_engine'])})"),
        ("system", "ok" if src["system"] else "unavailable"),
        ("own GPU memory", "read once this process uses the GPU"),
    ]
    print("sources")
    for name, state in rows:
        print(f"  {name:17s} {state}")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="metalmeter", description="Apple Silicon GPU and Neural Engine use, "
                                 "per process. Without a command, prints a reading every interval.")
    ap.add_argument("--version", action="version", version=f"metalmeter {__version__}")
    ap.add_argument("--pid", type=int, help="process to watch (default: system view)")
    ap.add_argument("--children", action="store_true", help="include the process's descendants")
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between readings (0.1 to 60)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--count", type=int, help="stop after this many readings")
    sub = ap.add_subparsers(dest="command")
    i = sub.add_parser("info", help="describe this Mac and which sources can be read")
    # SUPPRESS keeps "metalmeter --json info" working: the subcommand's default does not overwrite it.
    i.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="print JSON instead of text")
    d = sub.add_parser("dump", help="print one raw reading as JSON (other processes' names are hashed)")
    d.add_argument("--pid", dest="dump_pid", type=int, help="process to include (default: this process)")
    args = ap.parse_args(argv)

    if args.command == "info":
        d_info = info()
        if args.json:
            print(json.dumps(d_info, indent=1))
        else:
            print_info(d_info)
        return 0
    if args.command == "dump":
        pid = args.dump_pid if args.dump_pid is not None else os.getpid()
        print(json.dumps(dump_reading(pid), indent=1, default=_json_default))
        return 0
    if args.children and args.pid is None:
        ap.error("--children needs --pid")
    return watch(args.pid, args.children, args.interval, args.json, args.count)


if __name__ == "__main__":
    raise SystemExit(main())
