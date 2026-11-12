"""sysctlbyname reads with sized buffers and checked lengths."""
from __future__ import annotations

import ctypes as C
from typing import Any, Optional

from ._ffi import LIBSYSTEM, bind, load

_libsystem = load(LIBSYSTEM)
sysctlbyname = bind(_libsystem, "sysctlbyname", C.c_int, C.c_char_p, C.c_void_p, C.POINTER(C.c_size_t),
                    C.c_void_p, C.c_size_t)

AVAILABLE = sysctlbyname is not None


class XswUsage(C.Structure):
    """struct xsw_usage from <sys/sysctl.h> (vm.swapusage)."""
    _fields_ = [("xsu_total", C.c_uint64), ("xsu_avail", C.c_uint64), ("xsu_used", C.c_uint64),
                ("xsu_pagesize", C.c_uint32), ("xsu_encrypted", C.c_int)]


def value(name: str, ctype: Any) -> Any:
    """Read a fixed-size value as `ctype`. None if the name is unknown or the size differs."""
    if not AVAILABLE:
        return None
    v = ctype()
    size = C.c_size_t(C.sizeof(v))
    if sysctlbyname(name.encode(), C.byref(v), C.byref(size), None, 0) != 0 or size.value != C.sizeof(v):
        return None
    return v


def integer(name: str, ctype: Any = C.c_int) -> Optional[int]:
    v = value(name, ctype)
    return None if v is None else int(v.value)


def string(name: str) -> Optional[str]:
    """Read a string value: one call for its size, a second to read it."""
    if not AVAILABLE:
        return None
    size = C.c_size_t(0)
    if sysctlbyname(name.encode(), None, C.byref(size), None, 0) != 0 or size.value == 0:
        return None
    buf = C.create_string_buffer(size.value)
    if sysctlbyname(name.encode(), buf, C.byref(size), None, 0) != 0:
        return None  # the value can grow between the two calls
    return buf.value.decode("utf-8", "replace")


def swap_usage() -> Optional[XswUsage]:
    return value("vm.swapusage", XswUsage)
