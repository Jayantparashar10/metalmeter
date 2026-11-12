"""System-wide GPU statistics from each IOAccelerator's PerformanceStatistics property."""
from __future__ import annotations

from typing import Any, Optional

from .._bind import iokit
from . import never_raises


def _each(keys: tuple[tuple[str, str], ...]) -> Optional[list[dict[str, Any]]]:
    """One row per accelerator with the given (field, registry property) pairs."""
    if not iokit.AVAILABLE:
        return None
    out = []
    with iokit.matching("IOAccelerator") as accelerators:
        for acc in accelerators:
            row = {"entry_id": iokit.entry_id(acc), "class": iokit.class_name(acc)}
            row.update({k: iokit.prop(acc, prop) for k, prop in keys})
            out.append(row)
    return out or None


@never_raises
def read() -> Optional[list[dict[str, Any]]]:
    """Return ``[{"entry_id", "class", "perf"}]``, one per accelerator, or None if there is none."""
    return _each((("perf", "PerformanceStatistics"),))


@never_raises
def read_static() -> Optional[list[dict[str, Any]]]:
    """Return ``[{"entry_id", "class", "gpu_core_count"}]``, one per accelerator."""
    return _each((("gpu_core_count", "gpu-core-count"),))
