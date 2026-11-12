"""System context (memory pressure, swap, thermal state) and static facts about the Mac."""
from __future__ import annotations

import ctypes
from typing import Any, Optional

from .._bind import objc, sysctl
from . import never_raises


@never_raises
def read() -> Optional[dict[str, Any]]:
    """Return memory pressure, swap use and thermal state (0 to 3), or None off macOS."""
    if not sysctl.AVAILABLE:
        return None
    swap = sysctl.swap_usage()
    return {
        "mem_pressure": sysctl.integer("kern.memorystatus_vm_pressure_level"),
        "swap_used_bytes": swap.xsu_used if swap else None,
        "swap_total_bytes": swap.xsu_total if swap else None,
        "thermal_state": objc.thermal_state(),
    }


@never_raises
def read_static() -> Optional[dict[str, Any]]:
    """Return raw sysctl values that describe the Mac, or None off macOS.

    ``arm64`` is hw.optional.arm64 (1 on Apple Silicon, also under Rosetta), ``translated``
    is sysctl.proc_translated (1 under Rosetta), and ``vmm_present`` is kern.hv_vmm_present
    (1 in a virtual machine).
    """
    if not sysctl.AVAILABLE:
        return None
    return {
        "memsize": sysctl.integer("hw.memsize", ctypes.c_uint64),
        "cpu_brand": sysctl.string("machdep.cpu.brand_string"),
        "os_version": sysctl.string("kern.osproductversion"),
        "os_build": sysctl.string("kern.osversion"),
        "arm64": sysctl.integer("hw.optional.arm64"),
        "translated": sysctl.integer("sysctl.proc_translated"),
        "vmm_present": sysctl.integer("kern.hv_vmm_present"),
    }
