"""Test loads for the hardware checks. Each prints one JSON line {"ready": pid, ...}, then works.

    python validation/loads.py torch-matmul --seconds 30
    python validation/loads.py spawn --child torch-matmul --seconds 30   # the load runs in a child

A load runs for --seconds or until it is killed. Dev tool: needs torch, mlx or coremltools,
depending on the load. Nothing here needs root.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ready(**kw):
    print(json.dumps({"ready": os.getpid(), **kw}), flush=True)


def until(seconds):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        yield


def torch_matmul(a):
    """Back-to-back 4096x4096 matmuls: the GPU stays busy with short command buffers."""
    import torch
    x = torch.randn(4096, 4096, device="mps")
    torch.mps.synchronize()
    ready()
    for _ in until(a.seconds):
        x @ x
        torch.mps.synchronize()


def torch_light(a):
    """A small matmul every 50 ms: keeps a GPU client active for a long soak without heating the Mac."""
    import torch
    x = torch.randn(1024, 1024, device="mps")
    torch.mps.synchronize()
    ready()
    for _ in until(a.seconds):
        x @ x
        torch.mps.synchronize()
        time.sleep(0.05)


def torch_bigkernel(a):
    """One huge matmul per command buffer. The CPU only waits, so it is close to idle."""
    import torch
    x = torch.randn(a.n, a.n, device="mps")
    torch.mps.synchronize()
    ready(n=a.n)
    for _ in until(a.seconds):
        x @ x
        torch.mps.synchronize()


def torch_idle_client(a):
    """Opens the GPU once, then does nothing: a GPU client with no work."""
    import torch
    torch.zeros(1, device="mps")
    torch.mps.synchronize()
    ready()
    for _ in until(a.seconds):
        time.sleep(0.2)


def mlx_matmul(a):
    import mlx.core as mx
    x = mx.random.normal((4096, 4096))
    mx.eval(x)
    ready()
    for _ in until(a.seconds):
        mx.eval(x @ x)


def torch_alloc(a):
    """Allocate `--gib` on the GPU and report this process's own GPU memory before and after.

    metalmeter reads GPU memory only for its own process, so the check runs in here.
    """
    import torch
    sys.path.insert(0, ROOT)
    import metalmeter as mm
    torch.zeros(1, device="mps")
    torch.mps.synchronize()  # this process is now a GPU client
    mon = mm.Monitor(background=False)
    before = mon.read()
    driver_before = torch.mps.driver_allocated_memory()
    t = torch.ones(int(a.gib * 2**30) // 4, device="mps")  # float32
    torch.mps.synchronize()
    time.sleep(0.15)  # the Monitor returns the last Sample within 0.1 s of a reading
    after = mon.read()
    ready(gib=a.gib, mm_before=before.gpu_mem_bytes, mm_after=after.gpu_mem_bytes,
          driver_before=driver_before, driver_after=torch.mps.driver_allocated_memory(),
          limit=after.gpu_mem_limit_bytes)
    mon.stop()
    for _ in until(a.seconds):
        time.sleep(0.2)
    del t


def coreml_ane(a):
    """A stack of convolutions compiled for CPU and Neural Engine, run in a loop."""
    import coremltools as ct
    import numpy as np
    from coremltools.converters.mil import Builder as mb

    c = 256
    w = np.random.rand(c, c, 3, 3).astype(np.float32) * 0.01

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, c, 64, 64))])
    def prog(x):
        for _ in range(8):
            x = mb.relu(x=mb.conv(x=x, weight=w, pad_type="same"))
        return x

    model = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
                       compute_units=ct.ComputeUnit.CPU_AND_NE)
    inp = {"x": np.random.rand(1, c, 64, 64).astype(np.float32)}
    model.predict(inp)
    ready(preferred_devices=_compute_plan(ct, model))
    for _ in until(a.seconds):
        model.predict(inp)


def _compute_plan(ct, model):
    """Count ops per preferred device from Core ML's public compute plan; None if unavailable."""
    try:
        from coremltools.models.compute_plan import MLComputePlan
        plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(),
                                            compute_units=ct.ComputeUnit.CPU_AND_NE)
        out = {}
        for op in plan.model_structure.program.functions["main"].block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is not None:
                k = type(u.preferred_compute_device).__name__
                out[k] = out.get(k, 0) + 1
        return out
    except Exception:
        return None


def cpu_busy(a):
    ready()
    for _ in until(a.seconds):
        pass


def idle(a):
    ready()
    for _ in until(a.seconds):
        time.sleep(0.2)


def spawn(a):
    """Run `--child` in a child process and stay idle; the ready line names the child."""
    c = subprocess.Popen([sys.executable, os.path.abspath(__file__), a.child, "--seconds", str(a.seconds)],
                         stdout=subprocess.PIPE, text=True)
    child = json.loads(c.stdout.readline())["ready"]
    # By default SIGTERM ends Python without running `finally`, which would leave the child running.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    ready(child=child)
    try:
        c.wait()
    finally:
        c.kill()


LOADS = {"torch-matmul": torch_matmul, "torch-light": torch_light, "torch-bigkernel": torch_bigkernel,
         "torch-idle-client": torch_idle_client, "mlx-matmul": mlx_matmul, "torch-alloc": torch_alloc,
         "coreml-ane": coreml_ane, "cpu-busy": cpu_busy, "idle": idle, "spawn": spawn}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("load", choices=LOADS)
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--gib", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=12288)
    ap.add_argument("--child", default="torch-matmul", choices=LOADS)
    a = ap.parse_args()
    LOADS[a.load](a)
