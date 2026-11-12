"""Result types. Every metric is Optional: None means "not available", 0 means "measured zero"."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional

OK = "ok"
PARTIAL = "partial"
UNSUPPORTED = "unsupported"
PROCESS_EXITED = "process_exited"


def _flat(obj: Any, prefix: str) -> dict[str, Any]:
    return {prefix + f.name: v for f in dataclasses.fields(obj) if (v := getattr(obj, f.name)) is not None}


@dataclass(frozen=True)
class Snapshot:
    """Raw cumulative counters for one process at one instant. No rates."""
    t_wall: float
    t_mono_ns: int
    pid: int
    status: str
    gpu_time_ns: Optional[int] = None        # sum over the process's current GPU clients and queues
    cpu_time_ns: Optional[int] = None
    energy_nj: Optional[int] = None          # CPU only; GPU energy is not included
    mem_footprint_bytes: Optional[int] = None
    ane_mem_bytes: Optional[int] = None
    ane_connected: Optional[bool] = None
    gpu_mem_bytes: Optional[int] = None      # own process only
    gpu_mem_limit_bytes: Optional[int] = None
    sys_gpu_util_pct: Optional[float] = None
    sys_renderer_pct: Optional[float] = None
    sys_tiler_pct: Optional[float] = None
    sys_gpu_mem_in_use_bytes: Optional[int] = None
    sys_gpu_mem_alloc_bytes: Optional[int] = None
    mem_pressure: Optional[int] = None
    swap_used_bytes: Optional[int] = None
    thermal_state: Optional[str] = None

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flat dict; None fields are left out."""
        return _flat(self, prefix)


@dataclass(frozen=True)
class DeviceInfo:
    """Static facts about the Mac. Fields are None when they could not be read."""
    chip: Optional[str] = None               # e.g. "Apple M2"
    gpu_class: Optional[str] = None          # IOAccelerator class, e.g. "AGXAcceleratorG14G"
    gpu_cores: Optional[int] = None
    memory_bytes: Optional[int] = None       # unified memory shared by CPU and GPU
    macos: Optional[str] = None
    macos_build: Optional[str] = None
    apple_silicon: Optional[bool] = None
    rosetta: bool = False                    # this Python runs as x86_64 under Rosetta
    vm: Optional[bool] = None

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flat dict; None fields are left out."""
        return _flat(self, prefix)


@dataclass(frozen=True)
class FieldStats:
    """Summary of one field over a window. Each value is None if the field had no values."""
    mean: Optional[float] = None
    max: Optional[float] = None
    p95: Optional[float] = None


@dataclass(frozen=True)
class WindowStats:
    """Summary of the Samples in a time window.

    For ``gpu_util_pct`` and ``power_w`` the mean is computed from the cumulative counters
    (change over the window divided by its length), so long command buffers that land in a
    single Sample do not distort it.
    """
    window_s: float                          # time actually covered, first to last Sample
    samples: int
    fields: dict[str, FieldStats] = dataclasses.field(default_factory=dict)
    warnings: tuple[str, ...] = ()           # memory pressure, swap growth, thermal throttling

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flat dict such as ``{"gpu_util_pct/mean": 87.0}``; None values are left out."""
        out: dict[str, Any] = {}
        for name, s in self.fields.items():
            for stat in ("mean", "max", "p95"):
                v = getattr(s, stat)
                if v is not None:
                    out[f"{prefix}{name}/{stat}"] = v
        return out


@dataclass(frozen=True)
class Sample:
    """One reading of a process, with rates over the `interval_s` since the previous one.

    For a Sampler created with ``children=True`` the process fields cover the whole tree and
    the ``children_*`` fields hold the descendants' share.

    ``gpu_time_ns`` and ``energy_j`` count from the Sampler's first reading. The mean over any
    window is therefore ``(last - first) / elapsed``, however bursty single readings are.
    """
    t_wall: float
    t_mono_ns: int
    pid: int
    status: str
    interval_s: Optional[float] = None
    gpu_time_ns: Optional[int] = None
    gpu_client_count: Optional[int] = None   # 0 means the process has never opened the GPU
    gpu_util_pct: Optional[float] = None     # at most 100
    gpu_util_raw: Optional[float] = None     # true ratio; above 100 with overlapping queues or bursts
    gpu_mem_bytes: Optional[int] = None      # own process only
    gpu_mem_limit_bytes: Optional[int] = None
    cpu_util_pct: Optional[float] = None     # 100 = one core
    mem_footprint_bytes: Optional[int] = None
    energy_j: Optional[float] = None         # CPU only; GPU energy is not included
    power_w: Optional[float] = None
    ane_mem_bytes: Optional[int] = None
    ane_connected: Optional[bool] = None
    children_count: Optional[int] = None
    children_gpu_util_pct: Optional[float] = None
    children_cpu_util_pct: Optional[float] = None
    children_mem_footprint_bytes: Optional[int] = None
    children_power_w: Optional[float] = None
    sys_gpu_util_pct: Optional[float] = None
    sys_renderer_pct: Optional[float] = None
    sys_tiler_pct: Optional[float] = None
    sys_gpu_mem_in_use_bytes: Optional[int] = None
    sys_gpu_mem_alloc_bytes: Optional[int] = None
    mem_pressure: Optional[int] = None
    swap_used_bytes: Optional[int] = None
    thermal_state: Optional[str] = None

    def to_dict(self, prefix: str = "") -> dict[str, Any]:
        """Flat dict; None fields are left out."""
        return _flat(self, prefix)
