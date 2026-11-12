"""metalmeter: per-process Apple Silicon GPU and Neural Engine metrics. Read-only, no sudo.

    import metalmeter as mm

    with mm.Monitor() as mon:          # this process, sampled every second in the background
        ...
        s = mon.read()                 # newest Sample
        w = mon.stats(window=60)       # mean, max and p95 per field
"""
from .monitor import Monitor, stop_all
from .sampler import device_info, is_supported, snapshot
from .types import DeviceInfo, FieldStats, Sample, Snapshot, WindowStats

__version__ = "0.1.0"

__all__ = [
    "DeviceInfo", "FieldStats", "Monitor", "Sample", "Snapshot", "WindowStats",
    "device_info", "is_supported", "snapshot", "stop_all", "__version__",
]
