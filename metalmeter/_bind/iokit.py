"""IOKit registry access: find services, walk their children, read properties.

Only registry properties are read. Nothing here opens a connection to a driver or sends it
a command. Every io_object_t obtained is released: `matching` and `children` are context
managers that release their objects on exit.
"""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from . import cf
from ._ffi import IOKIT, bind, load

u32, u64, vp = C.c_uint32, C.c_uint64, C.c_void_p
_iokit = load(IOKIT)

kIOMainPortDefault = 0
IO_SERVICE_PLANE = b"IOService"

IOServiceMatching = bind(_iokit, "IOServiceMatching", vp, C.c_char_p)
IOServiceGetMatchingServices = bind(_iokit, "IOServiceGetMatchingServices", C.c_int, u32, vp, C.POINTER(u32))
IOIteratorNext = bind(_iokit, "IOIteratorNext", u32, u32)
IOIteratorIsValid = bind(_iokit, "IOIteratorIsValid", C.c_int, u32)
IORegistryEntryGetChildIterator = bind(_iokit, "IORegistryEntryGetChildIterator", C.c_int, u32, C.c_char_p,
                                       C.POINTER(u32))
IORegistryEntryCreateCFProperty = bind(_iokit, "IORegistryEntryCreateCFProperty", vp, u32, vp, vp, u32)
IORegistryEntryGetRegistryEntryID = bind(_iokit, "IORegistryEntryGetRegistryEntryID", C.c_int, u32, C.POINTER(u64))
IOObjectGetClass = bind(_iokit, "IOObjectGetClass", C.c_int, u32, C.c_char_p)
IOObjectRelease = bind(_iokit, "IOObjectRelease", C.c_int, u32)

AVAILABLE = _iokit is not None and cf.AVAILABLE

_keys: dict[str, int] = {}  # one CFString per property name, created on first use and never released


def _drain(iterator: int) -> list[int]:
    out = []
    while True:
        obj = IOIteratorNext(iterator)
        if not obj:
            return out
        out.append(obj)


def _release_all(objs: list[int]) -> None:
    for o in objs:
        IOObjectRelease(o)


@contextmanager
def matching(class_name: str) -> Iterator[list[int]]:
    """Yield the registered services of `class_name` or a subclass; empty off macOS."""
    objs: list[int] = []
    if AVAILABLE:
        it = u32()
        match = IOServiceMatching(class_name.encode())
        # IOServiceGetMatchingServices consumes `match`, so it is not released here.
        if match and IOServiceGetMatchingServices(kIOMainPortDefault, match, C.byref(it)) == 0:
            objs = _drain(it.value)
            IOObjectRelease(it.value)
    try:
        yield objs
    finally:
        _release_all(objs)


@contextmanager
def children(entry: int, tries: int = 3) -> Iterator[tuple[list[int], bool]]:
    """Yield the children of `entry` in the IOService plane, and whether the walk was consistent.

    If the registry changes during the walk the iterator becomes invalid and children may
    have been skipped. The walk is then repeated, up to `tries` times, and the last one kept.
    """
    kids: list[int] = []
    consistent = False
    if AVAILABLE:
        for attempt in range(tries):
            it = u32()
            if IORegistryEntryGetChildIterator(entry, IO_SERVICE_PLANE, C.byref(it)) != 0:
                break
            kids = _drain(it.value)
            consistent = bool(IOIteratorIsValid(it.value))
            IOObjectRelease(it.value)
            if consistent or attempt == tries - 1:
                break
            _release_all(kids)
            kids = []
    try:
        yield kids, consistent
    finally:
        _release_all(kids)


def prop(entry: int, key: str) -> Any:
    """Read one registry property of `entry`. None if it is missing or of an unsupported type."""
    if not AVAILABLE:
        return None
    k = _keys.get(key)
    if k is None:
        k = cf.create_string(key)
        if k is None:
            return None
        k = _keys.setdefault(key, k)
    with cf.owned(IORegistryEntryCreateCFProperty(entry, k, None, 0)) as ref:
        return cf.to_py(ref)


def entry_id(entry: int) -> Optional[int]:
    if not AVAILABLE:
        return None
    i = u64()
    return i.value if IORegistryEntryGetRegistryEntryID(entry, C.byref(i)) == 0 else None


def class_name(entry: int) -> Optional[str]:
    if not AVAILABLE:
        return None
    buf = C.create_string_buffer(128)  # io_name_t
    return buf.value.decode("utf-8", "replace") if IOObjectGetClass(entry, buf) == 0 else None
