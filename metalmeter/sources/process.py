"""Resource usage of one process, and the PIDs of its descendants."""
from __future__ import annotations

from typing import Any, Optional

from .._bind import libproc
from . import never_raises

MAX_TREE = 4096  # bounds the walk if a process forks without limit


@never_raises
def read(pid: int) -> Optional[dict[str, Any]]:
    """Return ``{"pid", "rusage", "errno"}`` for `pid`, or None off macOS.

    ``rusage`` holds the raw ``ri_*`` fields, or None when the read fails. ``errno`` is
    EPERM for another user's process and ESRCH for a process that no longer exists.
    """
    if not libproc.AVAILABLE:
        return None
    ru, err = libproc.pid_rusage(pid)
    return {"pid": pid, "rusage": ru, "errno": err}


@never_raises
def descendants(pid: int) -> Optional[list[int]]:
    """Return every descendant of `pid`, breadth first, or None if its children cannot be listed."""
    first = libproc.child_pids(pid)
    if first is None:
        return None
    seen, queue, out = {pid}, list(first), []
    while queue and len(out) < MAX_TREE:
        p = queue.pop(0)
        if p in seen:
            continue
        seen.add(p)
        out.append(p)
        queue.extend(libproc.child_pids(p) or [])
    return out


def timebase() -> tuple[int, int]:
    return libproc.TIMEBASE
