"""GPU clients of every IOAccelerator (owning PID and GPU time), and this process's GPU memory."""
from __future__ import annotations

import time
from typing import Any, Iterable, Optional

from .. import parse
from .._bind import iokit, metal
from . import never_raises

# IOUserClientCreator by registry entry ID. A client's creator never changes, and reading the
# entry ID costs well under half as much as reading the property, so each walk reads the
# property only for clients it has not seen before. Rebuilt on every walk, so it holds only
# clients that still exist.
_creators: dict[int, Optional[str]] = {}


def _creator(entry: int, eid: Optional[int], seen: dict[int, Optional[str]]) -> Optional[str]:
    if eid is None:
        return iokit.prop(entry, "IOUserClientCreator")
    creator = _creators[eid] if eid in _creators else iokit.prop(entry, "IOUserClientCreator")
    seen[eid] = creator
    return creator


@never_raises
def read(pids: Optional[Iterable[int]] = None) -> Optional[dict[str, Any]]:
    """Read the GPU clients of every IOAccelerator.

    Args:
        pids: if given, only these processes' clients are read. That is cheaper, and other
            processes' data is never collected. If None, every client is returned.

    Returns:
        ``{"t_mono_ns", "consistent", "accelerators", "clients": [{"accelerator",
        "entry_id", "creator", "app_usage"}]}``, or None without IOKit or without an
        accelerator (virtual machines, CI runners). ``t_mono_ns`` is taken right after the
        walk because the GPU counters carry no timestamp of their own.
    """
    global _creators
    if not iokit.AVAILABLE:
        return None
    want = None if pids is None else set(pids)
    clients: list[dict[str, Any]] = []
    seen: dict[int, Optional[str]] = {}
    consistent, n_accel = True, 0
    with iokit.matching("IOAccelerator") as accelerators:
        for acc in accelerators:
            n_accel += 1
            acc_id = iokit.entry_id(acc)
            with iokit.children(acc) as (kids, ok):
                consistent = consistent and ok
                for k in kids:
                    eid = iokit.entry_id(k)
                    creator = _creator(k, eid, seen)
                    if creator is None:
                        continue  # a child that is not a user client
                    if want is not None:
                        p = parse.parse_creator(creator)
                        if p is None or p[0] not in want:
                            continue
                    clients.append({"accelerator": acc_id, "entry_id": eid, "creator": creator,
                                    "app_usage": iokit.prop(k, "AppUsage")})
        t = time.monotonic_ns()
    _creators = seen
    if n_accel == 0:
        return None
    return {"t_mono_ns": t, "consistent": consistent, "accelerators": n_accel, "clients": clients}


@never_raises
def own_memory(already_gpu_client: bool) -> Optional[dict[str, Optional[int]]]:
    """Read this process's GPU memory from its MTLDevice.

    Pass True only if this process already has a GPU client. With False, no MTLDevice is
    created and None is returned, so a CPU-only process never opens a GPU client.
    """
    if not already_gpu_client:
        return None
    dev = metal.device()
    if dev is None:
        return None
    return {"current_allocated_size": metal.current_allocated_size(dev),
            "recommended_max_working_set_size": metal.recommended_max_working_set_size(dev)}
