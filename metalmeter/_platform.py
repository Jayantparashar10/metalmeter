"""Platform checks and the warn-once helper."""
from __future__ import annotations

import os
import sys
import threading
import warnings

_warned: set[str] = set()
_lock = threading.Lock()


def _reinit_after_fork() -> None:
    # Another thread may have held the lock at the moment of the fork; in the child it
    # would stay locked forever.
    global _lock
    _lock = threading.Lock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reinit_after_fork)


def is_macos() -> bool:
    return sys.platform == "darwin"


def warn_once(key: str, message: str) -> None:
    """Emit one RuntimeWarning per key for the life of the process."""
    with _lock:
        if key in _warned:
            return
        _warned.add(key)
    warnings.warn(f"metalmeter: {message}", RuntimeWarning, stacklevel=3)
