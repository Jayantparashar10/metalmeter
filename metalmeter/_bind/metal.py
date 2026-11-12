"""Metal: this process's own GPU memory (MTLDevice.currentAllocatedSize).

Creating an MTLDevice opens a GPU client for the process. In a CPU-only process that is
wasted work, and Objective-C state set up before a fork can crash the child, so the device
is created on first use and callers should ask for it only when the process already has a
GPU client. On macOS 27, loading CoreFoundation already maps the Metal library into the
process; what matters is that no device is created.
"""
from __future__ import annotations

import ctypes as C
import os
import threading
from typing import Any, Optional

from . import objc
from ._ffi import METAL, bind, load

vp = C.c_void_p
_lock = threading.Lock()
_create_device: Any = None
_device: Optional[int] = None  # owned, kept for the life of the process, never released


def _reinit_after_fork() -> None:
    # The parent's device must not be used from the child, and its lock may be held.
    global _lock, _device
    _lock = threading.Lock()
    _device = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_after_fork)

_send_ulong = objc.msg(C.c_ulong)  # NSUInteger currentAllocatedSize
_send_u64 = objc.msg(C.c_uint64)   # uint64_t recommendedMaxWorkingSetSize


def _load() -> Any:
    """Bind MTLCreateSystemDefaultDevice on first call. None if Metal is unavailable."""
    global _create_device
    if _create_device is None:
        _create_device = bind(load(METAL), "MTLCreateSystemDefaultDevice", vp) or False
    return _create_device or None


def device() -> Optional[int]:
    """Return the system default MTLDevice, creating it on the first call. None without a GPU."""
    global _device
    with _lock:
        if _device is None:
            create = _load()
            _device = (create() or 0) if create is not None else 0
    return _device or None


def current_allocated_size(dev: int) -> Optional[int]:
    """Bytes allocated on `dev` by this process. Other processes' allocations are not included."""
    s = objc.sel("currentAllocatedSize")
    return int(_send_ulong(dev, s)) if dev and s and _send_ulong else None


def recommended_max_working_set_size(dev: int) -> Optional[int]:
    s = objc.sel("recommendedMaxWorkingSetSize")
    return int(_send_u64(dev, s)) if dev and s and _send_u64 else None
