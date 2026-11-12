"""Sources read one kind of data from macOS and return plain dicts or lists, or None.

Readings run inside the user's training loop, so a source must not raise: every public
source function is wrapped in `never_raises`.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, TypeVar

from .._platform import warn_once

F = TypeVar("F", bound=Callable[..., Any])


def never_raises(fn: F) -> F:
    """Make `fn` return None instead of raising, warning once per wrapped function."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # deliberately broad: this includes bugs in metalmeter itself
            name = f"{fn.__module__.rsplit('.', 1)[-1]}.{fn.__name__}"
            warn_once(f"source:{name}", f"{name} failed ({e!r}); its fields are None")
            return None
    return wrapper  # type: ignore[return-value]
