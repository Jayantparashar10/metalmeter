"""Objective-C runtime: classes, selectors, and typed objc_msgSend casts.

On arm64, objc_msgSend must be called through a pointer cast to the exact signature
of the method. `msg(restype, *argtypes)` makes one such cast per signature.
"""
from __future__ import annotations

import ctypes as C
import os
import threading
from typing import Any, Optional

from ._ffi import FOUNDATION, LIBOBJC, bind, load, register

vp = C.c_void_p
_objc = load(LIBOBJC)
_foundation = load(FOUNDATION)  # loading Foundation registers NSProcessInfo with the runtime

objc_getClass = bind(_objc, "objc_getClass", vp, C.c_char_p)
sel_registerName = bind(_objc, "sel_registerName", vp, C.c_char_p)
_MSG_SEND = C.cast(_objc.objc_msgSend, vp).value if _objc is not None else None

AVAILABLE = _MSG_SEND is not None and objc_getClass is not None and sel_registerName is not None

_casts: dict[tuple[Any, ...], Any] = {}
_lock = threading.Lock()


def _reinit_after_fork() -> None:
    global _lock  # a lock held by another thread at fork time would stay held in the child
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_after_fork)


def msg(restype: Any, *argtypes: Any) -> Any:
    """Return objc_msgSend cast to ``restype (*)(id, SEL, *argtypes)``. None off macOS.

    Casts are cached, so each signature is created and registered once.
    """
    if not AVAILABLE:
        return None
    key = (restype, *argtypes)
    with _lock:
        f = _casts.get(key)
        if f is None:
            f = C.CFUNCTYPE(restype, vp, vp, *argtypes)(_MSG_SEND)
            _casts[key] = register(f"objc_msgSend{key}", f, restype, (vp, vp, *argtypes))
    return f


def cls(name: str) -> Optional[int]:
    return (objc_getClass(name.encode()) or None) if AVAILABLE else None


def sel(name: str) -> Optional[int]:
    return (sel_registerName(name.encode()) or None) if AVAILABLE else None


_send_id = msg(vp)
_send_long = msg(C.c_long)


def thermal_state() -> Optional[int]:
    """[[NSProcessInfo processInfo] thermalState]: 0 nominal, 1 fair, 2 serious, 3 critical."""
    if _foundation is None or _send_id is None:
        return None
    process_info_class, s_info, s_state = cls("NSProcessInfo"), sel("processInfo"), sel("thermalState")
    if not (process_info_class and s_info and s_state):
        return None
    info = _send_id(process_info_class, s_info)  # a shared instance we do not own
    return int(_send_long(info, s_state)) if info else None
