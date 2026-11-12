"""Library loading and the registry of every foreign function the package binds."""
from __future__ import annotations

import ctypes
import sys
from typing import Any, Optional

# Every library metalmeter loads. All are public system libraries; load() refuses any other path.
CORE_FOUNDATION = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"
IOKIT = "/System/Library/Frameworks/IOKit.framework/IOKit"
FOUNDATION = "/System/Library/Frameworks/Foundation.framework/Foundation"
METAL = "/System/Library/Frameworks/Metal.framework/Metal"
LIBSYSTEM = "/usr/lib/libSystem.B.dylib"
LIBOBJC = "/usr/lib/libobjc.A.dylib"
ALLOWED = (CORE_FOUNDATION, IOKIT, FOUNDATION, METAL, LIBSYSTEM, LIBOBJC)

# (name, function, restype, argtypes) for every bound function. Tests check that each entry
# has its argtypes and restype set: a wrong signature on arm64 corrupts memory silently.
REGISTRY: list[tuple[str, Any, Any, tuple[Any, ...]]] = []


def load(path: str) -> Optional[ctypes.CDLL]:
    """Load an allowed library; None off macOS or if it cannot be loaded."""
    if path not in ALLOWED:
        raise ValueError(f"metalmeter may not load {path}")
    if sys.platform != "darwin":
        return None
    try:
        return ctypes.CDLL(path, use_errno=True)
    except OSError:
        return None


def bind(lib: Optional[ctypes.CDLL], name: str, restype: Any, *argtypes: Any) -> Any:
    """Look up `name` and set its restype and argtypes. None if the library or symbol is missing."""
    if lib is None:
        return None
    try:
        f = getattr(lib, name)
    except AttributeError:
        return None
    f.restype = restype
    f.argtypes = list(argtypes)
    REGISTRY.append((name, f, restype, argtypes))
    return f


def register(name: str, f: Any, restype: Any, argtypes: tuple[Any, ...]) -> Any:
    """Record a function created another way, such as a typed objc_msgSend cast."""
    REGISTRY.append((name, f, restype, argtypes))
    return f
