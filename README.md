# metalmeter

metalmeter tells a Python program how much of an Apple Silicon Mac's GPU and Neural Engine it is using: GPU time, GPU memory, CPU, memory footprint and energy, for one process and optionally its child processes. You call it from inside a training loop, it needs no `sudo`, it has no dependencies, and it works with any framework that runs on Metal or Core ML (PyTorch MPS, MLX, TensorFlow-Metal, JAX-Metal, Core ML).

```python
import metalmeter as mm

with mm.Monitor() as mon:
    train()
    print(mon.read().gpu_util_pct)   # e.g. 88.8
```

Tools like macmon, asitop, mactop and Activity Monitor show the whole machine, and some of them need `sudo powermetrics`. metalmeter answers a narrower question from inside your own script: how busy is the GPU *with my work*, and how much memory does my process hold?

- [Requirements](#requirements)
- [Install](#install)
- [Quick start](#quick-start)
- [Examples](#examples)
- [Command line](#command-line)
- [What it measures](#what-it-measures)
- [Accuracy in brief](#accuracy-in-brief)
- [Security and privacy](#security-and-privacy)
- [Project status](#project-status)

Full reference: [docs/api.md](docs/api.md). Measured accuracy and limits: [docs/accuracy.md](docs/accuracy.md). Working on metalmeter itself: [CONTRIBUTING.md](CONTRIBUTING.md).

## Requirements

- A Mac with Apple Silicon (M1 or later).
- macOS 13 or later. macOS 12 works without the energy and Neural Engine memory fields.
- CPython 3.9 or later, running natively as arm64 (not under Rosetta).

On other systems metalmeter still imports and never raises. `mm.is_supported()` returns False and the GPU fields are None, so code that uses it runs unchanged on Linux CI.

## Install

metalmeter is not on PyPI yet. Install it from GitHub:

```sh
python3 -m pip install git+https://github.com/Jayantparashar10/metalmeter.git
```

This installs the `metalmeter` package and the `metalmeter` command. There is nothing to compile.

## Quick start

```python
import metalmeter as mm

with mm.Monitor() as mon:                  # this process, sampled once a second on a background thread
    for step in range(num_steps):
        train_step()
        if step % 100 == 0:
            s = mon.read()                 # the newest Sample
            print(step, s.gpu_util_pct, s.gpu_mem_bytes, s.cpu_util_pct)

    w = mon.stats(window=60)               # summary of the last 60 seconds
    print(w.fields["gpu_util_pct"].mean, w.warnings)
```

Two things to know from the start:

- A field is `None` when it is not available and `0` when it was measured as zero. Rates such as `gpu_util_pct` need two readings, so they are `None` in the first second of a Monitor's life.
- `gpu_util_pct` is the share of wall-clock time the GPU spent on this process's work, capped at 100. `cpu_util_pct` uses 100 for one fully busy core, like Activity Monitor.

## Examples

### Log to your experiment tracker

`Sample.to_dict()` returns a flat dict and leaves out fields that are None. It also has a few non-numeric fields (`status`, `thermal_state`, `ane_connected`), so filter them out before handing the dict to a logger that accepts only numbers:

```python
import metalmeter as mm

def numbers(d):
    return {k: v for k, v in d.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}

with mm.Monitor() as mon:
    for step, batch in enumerate(loader):
        loss = train_step(batch)
        if step % 50 == 0:
            metrics = numbers(mon.read().to_dict(prefix="mac/"))
            wandb.log({"loss": loss, **metrics}, step=step)
            # TensorBoard: for k, v in metrics.items(): writer.add_scalar(k, v, step)
```

### Summarise each epoch

`stats(window)` gives the mean, maximum and 95th percentile of every numeric field over the last `window` seconds, plus warnings about memory pressure, swap growth and thermal throttling:

```python
with mm.Monitor() as mon:
    for epoch in range(epochs):
        t0 = time.monotonic()
        train_one_epoch()
        w = mon.stats(window=time.monotonic() - t0)
        gpu = w.fields["gpu_util_pct"]
        print(f"epoch {epoch}: GPU mean {gpu.mean:.1f}%, p95 {gpu.p95:.1f}%")
        for warning in w.warnings:
            print("warning:", warning)          # e.g. "memory pressure reached warning"
        writer.add_scalars("gpu", w.to_dict(), epoch)   # keys like "gpu_util_pct/mean"
```

The mean of `gpu_util_pct` in a window comes from the total GPU time, so it stays exact even when single Samples are uneven.

### Get every Sample through a callback

`on_sample` is called with each new Sample on a metalmeter thread, so the training loop never waits for it:

```python
def send(s):
    dashboard.post(s.to_dict())            # slow network calls are fine here

with mm.Monitor(interval=2.0, on_sample=send) as mon:
    train()
print("skipped while the callback was busy:", mon.dropped)
```

If the callback is still running when the next Sample arrives, that Sample is skipped and counted in `mon.dropped`. Nothing queues up, and an exception in the callback is reported once as a warning without stopping the Monitor.

### Include DataLoader workers and other child processes

With `children=True` the main fields cover the process and all of its descendants, and the `children_*` fields show the descendants' share. Use it when the GPU work happens in a subprocess, or to count DataLoader workers' CPU time:

```python
with mm.Monitor(children=True) as mon:
    run_training_in_workers()
    s = mon.read()
    print(s.gpu_util_pct, s.children_count, s.children_gpu_util_pct, s.children_cpu_util_pct)
```

### Watch another process

Give a PID to watch a process other than your own, for example a training script from a notebook:

```python
with mm.Monitor(pid=12345) as mon:
    time.sleep(10)
    print(mon.stats(window=10).fields["gpu_util_pct"].mean)
```

GPU time works for any process. `gpu_mem_bytes` is only available for metalmeter's own process, because macOS has no API for another process's GPU memory. CPU, memory and energy are None for processes that belong to another user.

### Sample without a background thread

With `background=False` there is no thread: each `read()` takes a reading, and the rate covers the time since the previous call.

```python
mon = mm.Monitor(background=False)
for step in range(num_steps):
    train_step()
    if step % 100 == 0:
        print(mon.read().gpu_util_pct)     # GPU use since the previous read()
mon.stop()
```

Reads closer together than 0.1 s return the previous Sample.

### Check the machine, or take one reading

```python
if mm.is_supported():                      # Apple Silicon with an Apple GPU
    info = mm.device_info()
    print(info.chip, info.gpu_cores, info.memory_bytes // 2**30)   # Apple M2 8 16

snap = mm.snapshot()                       # raw cumulative counters right now, no rates
print(snap.gpu_time_ns, snap.mem_footprint_bytes, snap.sys_gpu_util_pct)
```

### In a notebook

Re-running a cell can leave old Monitors running. `mm.stop_all()` stops every Monitor in the process, and `repr(mon)` shows whether one is still running:

```python
mm.stop_all()
mon = mm.Monitor()
mon          # <metalmeter.Monitor pid=4242 interval=1.0s children=False background=True running samples=3>
```

## Command line

```text
metalmeter                                   system GPU, memory and thermal state, once a second
metalmeter --pid 1234                        one process
metalmeter --pid 1234 --children --json      the process tree, one JSON object per line
metalmeter --interval 0.5 --count 20         twice a second, 20 readings
metalmeter info                              this Mac, and which data sources can be read
metalmeter dump > dump.json                  one raw reading, for bug reports
```

`python3 -m metalmeter` works the same way. A process view stops when the process exits. Example output:

```text
$ metalmeter --pid 23005
pid 23005  GPU 70.5%  CPU 1.1%  memory 1.2 GiB  CPU power 0.00 W  ANE memory 0 B  | system GPU 100.0%, thermal nominal
```

The command line watches another process, so it cannot show that process's GPU memory; use `Monitor` inside the script for `gpu_mem_bytes`.

## What it measures

| Field | Meaning |
| --- | --- |
| `gpu_util_pct` | share of time the GPU spent on this process's work, 0 to 100 |
| `gpu_time_ns` | GPU time used since the Monitor started |
| `gpu_mem_bytes` | GPU memory allocated by this process (own process only) |
| `cpu_util_pct` | CPU use, 100 = one core |
| `mem_footprint_bytes` | memory footprint, the "Memory" column in Activity Monitor |
| `power_w`, `energy_j` | CPU power and energy of the process (GPU energy is not included) |
| `ane_connected`, `ane_mem_bytes` | whether the process uses the Neural Engine, and the memory it holds there |
| `children_*` | the descendants' share, with `children=True` |
| `sys_gpu_util_pct`, `sys_gpu_mem_in_use_bytes` | the whole GPU, for context |
| `mem_pressure`, `swap_used_bytes`, `thermal_state` | system memory pressure, swap and thermal state |

Every field, with units and sources, is in [docs/api.md](docs/api.md#sample).

## Accuracy in brief

Measured on an Apple M2 with macOS 27; details in [docs/accuracy.md](docs/accuracy.md).

- A PyTorch MPS matmul loop reads 88.8% and an MLX one 95.1%, while the whole GPU reads 99%. The per-process figure is usually 5 to 30 points below the system figure.
- GPU time is counted when a command buffer finishes, so command buffers longer than about a second are undercounted, and single Samples can jump between 0 and 100%. Window means from `stats()` are exact over the window.
- An idle process reads 0.0% while another process keeps the GPU busy.
- `gpu_mem_bytes` matches `torch.mps.driver_allocated_memory()` to the byte.
- `power_w` is CPU power only: a GPU-bound process reads close to 0 W.
- macOS has no busy counter for the Neural Engine, so there is no Neural Engine utilization percentage.
- A Monitor at the default 1 s interval uses about 0.4% of one CPU core.

## Security and privacy

metalmeter only reads data that macOS makes available to every user: GPU client entries in the IORegistry and per-process counters from libproc. It never asks for `sudo` or calls `powermetrics`, starts no subprocesses, opens no connection to a driver, loads only public system frameworks, uses no network, and changes no environment variables. The test suite enforces each of these.

Samples contain only the watched process (and its children) and system totals, never other processes' names. `metalmeter dump` lists every GPU client on the Mac for bug reports, but replaces other processes' names with hashes.

## Project status

Version 0.1.0.dev0, not yet released on PyPI. It has been tested on an Apple M2 (8-core GPU) with macOS 27.0, on CPython 3.9 and 3.13. Results from other chips are very welcome: [CONTRIBUTING.md](CONTRIBUTING.md#check-another-mac) explains how to run the checks and send them in.

Planned: an optional integration for [Visdom](https://github.com/fossasia/visdom) (`visdom[mac]`) that plots these metrics next to training curves.

## License

MIT. See [LICENSE](LICENSE).
