"""Make every way of starting a process fail, so tests catch metalmeter doing it."""
import _posixsubprocess
import os
import subprocess

NAMES = [n for n in dir(os) if n.startswith(("exec", "spawn", "posix_spawn"))] + ["system", "popen", "fork", "forkpty"]


class SpawnBlocked(AssertionError):
    pass


def _blocked(name):
    def f(*args, **kwargs):
        raise SpawnBlocked(f"metalmeter must not start processes ({name})")
    return f


def install(setattr_=setattr):
    """Replace the process-creation functions. Pass pytest's monkeypatch.setattr to undo it after a test."""
    for n in NAMES:
        if hasattr(os, n):
            setattr_(os, n, _blocked(f"os.{n}"))
    setattr_(subprocess, "Popen", _blocked("subprocess.Popen"))
    setattr_(_posixsubprocess, "fork_exec", _blocked("_posixsubprocess.fork_exec"))
