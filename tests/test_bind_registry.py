"""Every foreign function is bound with argtypes and restype, through the registry."""
import ctypes
import importlib
import pkgutil
import sys

import pytest

import metalmeter._bind as bind_pkg
from metalmeter._bind import _ffi
from tests.conftest import macos, needs_gpu, run_child

MODULES = [importlib.import_module(f"metalmeter._bind.{m.name}") for m in pkgutil.iter_modules(bind_pkg.__path__)]


def check_registry(registry):
    for name, f, restype, argtypes in registry:
        assert f.argtypes is not None, name
        assert list(f.argtypes) == list(argtypes), name
        assert f.restype is restype or f.restype == restype, name


def test_every_registered_function_has_argtypes_and_restype():
    check_registry(_ffi.REGISTRY)
    if sys.platform == "darwin":
        assert len(_ffi.REGISTRY) >= 35


def test_no_foreign_function_is_bound_outside_the_registry():
    registered = {id(f) for _, f, _, _ in _ffi.REGISTRY}
    for mod in MODULES:
        for attr, v in vars(mod).items():
            if isinstance(v, ctypes._CFuncPtr):
                assert id(v) in registered, f"{mod.__name__}.{attr} was not bound with bind()"


def test_load_refuses_libraries_outside_the_allowlist():
    with pytest.raises(ValueError):
        _ffi.load("/usr/lib/libIOReport.dylib")
    with pytest.raises(ValueError):
        _ffi.load("/System/Library/PrivateFrameworks/IOReport.framework/IOReport")


@macos
@needs_gpu
@pytest.mark.allow_subprocess
def test_metal_bindings_are_registered_too():
    # Metal loads lazily, so check it in a child to keep this process free of Metal.
    r = run_child(
        "import json\n"
        "from metalmeter._bind import _ffi, metal\n"
        "assert metal._load() is not None\n"
        "from tests.test_bind_registry import check_registry\n"
        "check_registry(_ffi.REGISTRY)\n"
        "print(json.dumps(sorted(n for n, *_ in _ffi.REGISTRY if 'Metal' in n or 'MTL' in n)))\n"
    )
    assert r == ["MTLCreateSystemDefaultDevice"]
