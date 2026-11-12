"""Dev tool: record a fixture while each test load runs. Not a test; needs a Mac with a GPU.

    python tests/fixtures/capture.py              # idle, torch-matmul, mlx-matmul, coreml-ane

A fixture is the output of ``metalmeter dump`` for the load's PID (format 1). Other
processes' names are replaced by hashes, so fixtures can be shared.
"""
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from metalmeter.cli import dump_reading  # noqa: E402
from metalmeter.sources import system  # noqa: E402

LOADS = ("idle", "torch-matmul", "mlx-matmul", "coreml-ane")


def main():
    static = system.read_static()
    chip = (static["cpu_brand"] or "unknown").replace("Apple ", "").replace(" ", "").lower()
    tag = f"{chip}-macos{static['os_version']}"
    for name in LOADS:
        p = subprocess.Popen([sys.executable, os.path.join(ROOT, "validation", "loads.py"), name, "--seconds", "8"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        pid = next(json.loads(line)["ready"] for line in p.stdout if line.startswith('{"ready"'))
        time.sleep(2)  # let the load reach a steady state
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{tag}-{name}.json")
        with open(path, "w") as f:
            json.dump(dump_reading(pid, load=name), f, indent=1)
        p.stdout.read()
        p.wait()
        print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
