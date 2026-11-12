"""CoreFoundation: convert CF objects to Python values.

Ownership follows the CoreFoundation rules. A reference returned by a Create or Copy
function belongs to the caller and must be released exactly once, which `owned` does. A
reference returned by a Get function is borrowed and must not be released. `to_py` only
reads borrowed references.
"""
from __future__ import annotations

import ctypes as C
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Union

from ._ffi import CORE_FOUNDATION, bind, load

vp = C.c_void_p
_cf = load(CORE_FOUNDATION)

kCFStringEncodingUTF8 = 0x08000100
kCFNumberSInt64Type = 4
kCFNumberDoubleType = 13

CFRelease = bind(_cf, "CFRelease", None, vp)
CFGetTypeID = bind(_cf, "CFGetTypeID", C.c_ulong, vp)
CFStringCreateWithCString = bind(_cf, "CFStringCreateWithCString", vp, vp, C.c_char_p, C.c_uint32)
CFStringGetLength = bind(_cf, "CFStringGetLength", C.c_long, vp)
CFStringGetMaximumSizeForEncoding = bind(_cf, "CFStringGetMaximumSizeForEncoding", C.c_long, C.c_long, C.c_uint32)
CFStringGetCString = bind(_cf, "CFStringGetCString", C.c_bool, vp, C.c_char_p, C.c_long, C.c_uint32)
CFNumberIsFloatType = bind(_cf, "CFNumberIsFloatType", C.c_bool, vp)
CFNumberGetValue = bind(_cf, "CFNumberGetValue", C.c_bool, vp, C.c_long, vp)
CFBooleanGetValue = bind(_cf, "CFBooleanGetValue", C.c_bool, vp)
CFArrayGetCount = bind(_cf, "CFArrayGetCount", C.c_long, vp)
CFArrayGetValueAtIndex = bind(_cf, "CFArrayGetValueAtIndex", vp, vp, C.c_long)
CFDictionaryGetCount = bind(_cf, "CFDictionaryGetCount", C.c_long, vp)
CFDictionaryGetKeysAndValues = bind(_cf, "CFDictionaryGetKeysAndValues", None, vp, vp, vp)
CFDataGetLength = bind(_cf, "CFDataGetLength", C.c_long, vp)
CFDataGetBytePtr = bind(_cf, "CFDataGetBytePtr", vp, vp)

_TYPE_ID_FUNCS = {n: bind(_cf, f"CF{n}GetTypeID", C.c_ulong) for n in
                  ("String", "Number", "Boolean", "Array", "Dictionary", "Data")}
TYPE_IDS = {n: f() for n, f in _TYPE_ID_FUNCS.items() if f is not None}

AVAILABLE = _cf is not None and len(TYPE_IDS) == len(_TYPE_ID_FUNCS)


@contextmanager
def owned(ref: Optional[int]) -> Iterator[Optional[int]]:
    """Hold a reference from a Create or Copy call and release it once on exit. NULL is allowed."""
    try:
        yield ref
    finally:
        if ref:
            CFRelease(ref)


def create_string(s: str) -> Optional[int]:
    """Create a CFString that the caller must release. None off macOS."""
    if not AVAILABLE:
        return None
    return CFStringCreateWithCString(None, s.encode("utf-8"), kCFStringEncodingUTF8) or None


def to_py(ref: Optional[int]) -> Any:
    """Convert a borrowed CF reference to str, int, float, bool, bytes, list or dict.

    Other CF types, and values that fail to convert, give None.
    """
    if not ref or not AVAILABLE:
        return None
    t = CFGetTypeID(ref)
    if t == TYPE_IDS["String"]:
        return _string(ref)
    if t == TYPE_IDS["Number"]:
        return _number(ref)
    if t == TYPE_IDS["Boolean"]:
        return bool(CFBooleanGetValue(ref))
    if t == TYPE_IDS["Array"]:
        return [to_py(CFArrayGetValueAtIndex(ref, i)) for i in range(CFArrayGetCount(ref))]
    if t == TYPE_IDS["Dictionary"]:
        return _dictionary(ref)
    if t == TYPE_IDS["Data"]:
        return C.string_at(CFDataGetBytePtr(ref), CFDataGetLength(ref))
    return None


def _string(ref: int) -> Optional[str]:
    size = CFStringGetMaximumSizeForEncoding(CFStringGetLength(ref), kCFStringEncodingUTF8) + 1
    buf = C.create_string_buffer(size)
    if not CFStringGetCString(ref, buf, size, kCFStringEncodingUTF8):
        return None
    return buf.value.decode("utf-8", "replace")


def _number(ref: int) -> Union[int, float, None]:
    if CFNumberIsFloatType(ref):
        d = C.c_double()
        return d.value if CFNumberGetValue(ref, kCFNumberDoubleType, C.byref(d)) else None
    i = C.c_int64()
    return i.value if CFNumberGetValue(ref, kCFNumberSInt64Type, C.byref(i)) else None


def _dictionary(ref: int) -> dict[Any, Any]:
    n = CFDictionaryGetCount(ref)
    keys, values = (vp * n)(), (vp * n)()
    CFDictionaryGetKeysAndValues(ref, keys, values)
    return {to_py(k): to_py(v) for k, v in zip(keys, values)}
