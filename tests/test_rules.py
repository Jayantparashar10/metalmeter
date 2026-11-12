"""Security rules, checked on the package source: no process creation, no network, no
environment changes, no driver connections, and no libraries outside the allowlist."""
import ast
import pathlib
import subprocess

import pytest

from tests._guard import SpawnBlocked

PKG = pathlib.Path(__file__).resolve().parent.parent / "metalmeter"
FILES = sorted(PKG.rglob("*.py"))

FORBIDDEN_MODULES = {"subprocess", "socket", "ssl", "urllib", "http", "pty", "ftplib", "smtplib", "xmlrpc",
                     "multiprocessing"}
FORBIDDEN_OS = {"system", "popen", "fork", "forkpty", "putenv", "unsetenv", "environ"}
FORBIDDEN_TEXT = ["IOServiceOpen", "IOConnectCall", "IOReport", "PrivateFrameworks", "powermetrics", "sudo",
                  "OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "setuid", "seteuid"]


def _trees():
    return [(f, ast.parse(f.read_text(), str(f))) for f in FILES]


def test_package_files_found():
    assert any(f.name == "_ffi.py" for f in FILES)


def _code_words(tree):
    """Identifiers and string literals, without docstrings (prose such as "no sudo" is fine)."""
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                  and n.body and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            yield n.id
        elif isinstance(n, ast.Attribute):
            yield n.attr
        elif isinstance(n, ast.Constant) and isinstance(n.value, (str, bytes)) and id(n) not in docstrings:
            yield n.value.decode("latin-1") if isinstance(n.value, bytes) else n.value


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(PKG)))
def test_no_forbidden_names_or_strings_in_code(path):
    for w in _code_words(ast.parse(path.read_text())):
        for word in FORBIDDEN_TEXT:
            assert word not in w, f"{word} in {path.name}: {w!r}"


def test_no_process_network_or_environment_modules():
    for f, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for n in names:
                assert n.split(".")[0] not in FORBIDDEN_MODULES, f"{f.name} imports {n}"


def test_no_os_process_or_environment_calls():
    for f, tree in _trees():
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "os":
                a = node.attr
                assert a not in FORBIDDEN_OS and not a.startswith(("exec", "spawn", "posix_spawn")), \
                    f"{f.name} uses os.{a}"


def test_libraries_are_loaded_only_through_the_allowlist():
    for f, tree in _trees():
        if f.name == "_ffi.py":
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in ("CDLL", "cdll", "LoadLibrary", "PyDLL", "pydll"), f"{f.name}: {node.attr}"
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "load":
                assert all(isinstance(a, ast.Name) for a in node.args), f"{f.name}: load() with a literal path"


def test_guard_really_blocks_process_creation():
    with pytest.raises(SpawnBlocked):
        subprocess.run(["true"])
