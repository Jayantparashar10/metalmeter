"""The metalmeter command: live views, info and dump."""
import json
import os
import subprocess
import sys

import pytest

import metalmeter as mm
from metalmeter import cli, parse
from metalmeter.sampler import Sampler
from tests.conftest import ROOT, macos, needs_gpu


@pytest.fixture(autouse=True)
def clean():
    yield
    mm.stop_all()


def run(capsys, *argv):
    code = cli.main(list(argv))
    out = capsys.readouterr()
    return code, out.out, out.err


@macos
def test_info(capsys):
    code, out, _ = run(capsys, "info")
    assert code == 0 and "supported" in out and "process counters  ok (rusage v6)" in out


@macos
@pytest.mark.parametrize("argv", [("--json", "info"), ("info", "--json")])
def test_info_json(capsys, argv):
    code, out, _ = run(capsys, *argv)
    d = json.loads(out)
    assert code == 0 and d["device"]["chip"] and set(d["sources"]) == {
        "gpu_clients", "gpu_statistics", "process_counters", "neural_engine", "system"}


@macos
@needs_gpu
def test_system_view_prints_one_line_per_reading(capsys):
    code, out, _ = run(capsys, "--count", "2", "--interval", "0.1")
    lines = out.strip().splitlines()
    assert code == 0 and len(lines) == 2 and all(line.startswith("GPU ") for line in lines)


def test_process_view_json(capsys):
    code, out, _ = run(capsys, "--pid", str(os.getpid()), "--json", "--count", "2", "--interval", "0.1")
    rows = [json.loads(line) for line in out.strip().splitlines()]
    assert code == 0 and len(rows) == 2 and all(r["pid"] == os.getpid() for r in rows)
    assert all("interval_s" in r for r in rows)


@macos
@needs_gpu  # without a GPU the line shows "GPU -" (not available) instead
def test_a_process_without_gpu_clients_says_so(capsys):
    code, out, _ = run(capsys, "--pid", str(os.getpid()), "--count", "1", "--interval", "0.1")
    assert code == 0 and "no GPU clients" in out


@macos  # process exit is detected through libproc
def test_missing_process_is_an_error(capsys):
    code, _, err = run(capsys, "--pid", "999999", "--count", "1")
    assert code == 1 and "no process with pid 999999" in err


@macos
@pytest.mark.allow_subprocess
def test_exiting_process_ends_the_view(capsys):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.4)"])
    code, out, _ = run(capsys, "--pid", str(child.pid), "--interval", "0.1")
    child.wait()
    assert code == 0 and out.strip().endswith(f"process {child.pid} exited")


def test_children_needs_a_pid(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--children"])


@macos
@needs_gpu
def test_dump_is_a_fixture_with_other_names_hashed(capsys):
    code, out, _ = run(capsys, "dump")
    d = json.loads(out)
    assert code == 0 and d["format"] == 1 and d["pid"] == os.getpid()
    for c in d["gpu_clients"]["clients"] + d["ane"]["clients"]:
        pid, name = parse.parse_creator(c["creator"])
        assert pid == os.getpid() or name.startswith("h:")
    # A dump is a valid reading: the sampler accepts it.
    s = Sampler(d["pid"], reader=lambda pid, children: d).sample()
    assert s.status == "ok" and s.sys_gpu_util_pct is not None


def test_bytes_are_serialised_in_dumps():
    assert json.loads(json.dumps({"v": b"\x01\xff"}, default=cli._json_default)) == {"v": {"bytes": "01ff"}}


@pytest.mark.allow_subprocess
def test_python_dash_m_and_version():
    p = subprocess.run([sys.executable, "-m", "metalmeter", "--version"], capture_output=True, text=True,
                       cwd=ROOT, timeout=30)
    assert p.returncode == 0 and p.stdout.strip() == f"metalmeter {mm.__version__}"
