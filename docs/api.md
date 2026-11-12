# API reference

Everything public is available from the top-level package:

```python
import metalmeter as mm
```

| Name | What it is |
| --- | --- |
| [`Monitor`](#monitor) | watches one process (and optionally its children) and keeps a history of Samples |
| [`Sample`](#sample) | one reading with rates, returned by `Monitor.read()` |
| [`WindowStats`, `FieldStats`](#windowstats) | a summary of recent Samples, returned by `Monitor.stats()` |
| [`snapshot()`, `Snapshot`](#snapshot) | raw cumulative counters of a process at one moment |
| [`device_info()`, `DeviceInfo`](#device_info) | facts about the Mac |
| [`is_supported()`](#is_supported) | whether this Mac can be measured |
| [`stop_all()`](#stop_all) | stop every Monitor in the process |
| [Command line](#command-line) | the `metalmeter` command |

No function or method raises into your code. When something cannot be read, the value is None and metalmeter emits one `RuntimeWarning` for that problem (not one per Sample).

## Monitor

```python
mm.Monitor(pid=None, interval=1.0, children=False, background=True, on_sample=None, history=600)
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `pid` | this process | the process to watch |
| `interval` | `1.0` | seconds between background Samples, clamped to 0.1 to 60 |
| `children` | `False` | include every descendant of `pid`; see [Process trees](#process-trees) |
| `background` | `True` | sample on a background thread. With `False` there is no thread and `read()` takes a Sample when called |
| `on_sample` | `None` | a function called with each new Sample, on a separate thread; see [Callbacks](#callbacks) |
| `history` | `600` | how many Samples to keep for `read()` and `stats()` (10 minutes at 1 s) |

A Monitor starts sampling as soon as it is created.

| Method or property | Returns |
| --- | --- |
| `read()` | the newest `Sample`, or None if none could be taken. With `background=False` it takes a reading first |
| `stats(window=60.0)` | a `WindowStats` over the Samples of the last `window` seconds |
| `stop()` | stops the Monitor. Calling it twice is harmless |
| `running` | True while the Monitor can take new Samples |
| `dropped` | Samples skipped because `on_sample` was still busy |
| `callback_errors` | how many times `on_sample` raised |
| `pid`, `interval`, `children`, `background` | the arguments, after clamping |

`Monitor` is a context manager: leaving the `with` block calls `stop()`. A stopped Monitor keeps its history, so `read()` and `stats()` still work afterwards:

```python
with mm.Monitor() as mon:
    train()
print(mon.stats(window=600).fields["gpu_util_pct"].mean)   # after the block
```

### Timing

The first Sample is a baseline, so its rates (`gpu_util_pct`, `cpu_util_pct`, `power_w` and the like) are None. Each later Sample's rates cover the time since the one before it, which is `Sample.interval_s`. Calls to `read()` less than 0.1 s apart return the same Sample, because shorter intervals mostly measure noise.

### Callbacks

`on_sample` runs on a thread of its own, and only one call runs at a time. If it is still busy when a new Sample arrives, that Sample is skipped and counted in `dropped`. Sampling is never delayed and Samples never pile up in a queue, so a slow callback (for example one that sends data over the network) cannot slow down training or grow memory.

If the callback raises, metalmeter counts it in `callback_errors`, warns once, and keeps sampling.

```python
def on_sample(s):
    print(f"{s.gpu_util_pct}% GPU, {s.mem_footprint_bytes / 2**30:.1f} GiB")

mon = mm.Monitor(interval=5, on_sample=on_sample)
```

### Several Monitors

Monitors that watch the same process with the same `children` setting share one reader, one history and one thread. The thread runs at the shortest interval any of them asked for, and each Monitor's `on_sample` is still called at its own interval. Creating a second Monitor for the same process therefore costs almost nothing, and both see the same numbers.

### Process trees

With `children=True`, the main fields (`gpu_util_pct`, `cpu_util_pct`, `mem_footprint_bytes`, `power_w` and so on) cover the watched process and every descendant. The `children_*` fields show the descendants' share, and `children_count` how many there are. Children that start or exit while the Monitor runs are picked up or dropped at the next reading; a child that exits does not make the totals go down.

### When the process exits

When the watched process exits, the next Sample has `status == "process_exited"` and the Monitor stops. A process that has exited but has not yet been collected by its parent (a zombie) counts as exited. If the PID is reused by a new process, metalmeter notices from the process start time and reports the old one as exited instead of mixing the two.

### fork and multiprocessing

If your program forks after a Monitor started (for example `multiprocessing` with the `fork` start method, or DataLoader workers on some setups), the Monitor is stopped in the child: the child has no sampling thread, and `read()` there returns the last Sample from before the fork. A new Monitor created in the child works normally.

Python 3.12 and later warn that forking a process that has threads can deadlock the child. Where you can, use the `spawn` start method. metalmeter never opens a Metal device in a process that does not already use the GPU, so it does not add the Objective-C state that makes forked children crash.

### Interpreter exit

Background threads are stopped when the interpreter exits. Exit is not held up by a callback that is still running.

### Cost

On an Apple M2, a Monitor at the default 1 s interval uses about 0.4% of one CPU core. One reading takes about 0.3 ms when readings follow each other closely and 2 to 4.5 ms with a pause between them, because macOS moves the idle thread to an efficiency core. Intervals well below 1 s are meant for short diagnostic runs: two Monitors at 0.1 s use about 5% of a core.

## Sample

A frozen dataclass with these fields. Every field except the first five can be None.

| Field | Unit | Scope | Meaning |
| --- | --- | --- | --- |
| `t_wall` | Unix seconds | | wall-clock time of the reading, for labels |
| `t_mono_ns` | ns | | monotonic time of the reading; use this for differences |
| `pid` | | | the watched process |
| `status` | | | see [Status](#status) |
| `interval_s` | s | | time since the previous Sample; None for the first |
| `gpu_time_ns` | ns, cumulative | process | GPU time used since the Monitor's first reading |
| `gpu_client_count` | count | process | GPU connections the process has open; 0 means it has never used the GPU |
| `gpu_util_pct` | %, 0 to 100 | process | GPU time divided by elapsed time |
| `gpu_util_raw` | % | process | the same ratio without the cap. Above 100 when several queues overlap or a long command buffer finishes inside one interval |
| `gpu_mem_bytes` | bytes | own process only | GPU memory allocated by this process |
| `gpu_mem_limit_bytes` | bytes | own process only | the GPU memory Metal recommends this process stays under |
| `cpu_util_pct` | %, 100 = one core | process | user plus system CPU time divided by elapsed time |
| `mem_footprint_bytes` | bytes | process | memory footprint, as in Activity Monitor's "Memory" column |
| `energy_j` | J, cumulative | process | CPU energy since the Monitor's first reading. GPU energy is not included |
| `power_w` | W | process | CPU power. GPU power is not included |
| `ane_mem_bytes` | bytes | process | memory the process holds on the Neural Engine |
| `ane_connected` | bool | process | whether the process has a Neural Engine connection |
| `children_count` | count | descendants | number of descendants (`children=True` only) |
| `children_gpu_util_pct`, `children_cpu_util_pct`, `children_mem_footprint_bytes`, `children_power_w` | as above | descendants | the descendants' share of the main fields |
| `sys_gpu_util_pct` | % | system | how busy the whole GPU is |
| `sys_renderer_pct`, `sys_tiler_pct` | % | system | the GPU's renderer and tiler utilization |
| `sys_gpu_mem_in_use_bytes`, `sys_gpu_mem_alloc_bytes` | bytes | system | GPU memory in use and allocated, all processes |
| `mem_pressure` | 1, 2 or 4 | system | memory pressure: normal, warning, critical |
| `swap_used_bytes` | bytes | system | swap in use |
| `thermal_state` | text | system | `"nominal"`, `"fair"`, `"serious"` or `"critical"` |

`gpu_mem_bytes` and `gpu_mem_limit_bytes` are only filled in when the Monitor watches its own process and that process already uses the GPU. macOS offers no way to read another process's GPU memory, and metalmeter does not open a GPU connection for a process that has none.

For a process owned by another user, macOS refuses to give out CPU time, memory and energy, so those fields are None. GPU time is still available.

### Status

| Status | Meaning |
| --- | --- |
| `"ok"` | every source was read |
| `"partial"` | some sources could not be read, for example because the GPU registry kept changing during the reading or a child process vanished mid-reading. The fields that could be read are filled in |
| `"unsupported"` | no Apple GPU was found: a virtual machine, a CI runner, or not macOS. GPU fields are None; on macOS the process and system fields are still read |
| `"process_exited"` | the process has exited, or its PID now belongs to another process. The Monitor stops |

### to_dict

```python
s.to_dict(prefix="")
```

Returns a flat dict of the fields, leaving out those that are None. `prefix` is added to every key:

```python
>>> mon.read().to_dict(prefix="mac/")
{'mac/t_wall': 1794465348.83, 'mac/t_mono_ns': 317838406781500, 'mac/pid': 22953, 'mac/status': 'ok',
 'mac/interval_s': 0.205, 'mac/gpu_time_ns': 0, 'mac/gpu_client_count': 0, 'mac/gpu_util_pct': 0.0,
 'mac/cpu_util_pct': 0.92, 'mac/mem_footprint_bytes': 12665600, 'mac/power_w': 0.0016, ...,
 'mac/mem_pressure': 1, 'mac/swap_used_bytes': 693895168, 'mac/thermal_state': 'nominal'}
```

`status`, `thermal_state` and `ane_connected` are not numbers. Filter them out for loggers that take only numbers:

```python
metrics = {k: v for k, v in s.to_dict().items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
```

## WindowStats

```python
w = mon.stats(window=60.0)
```

| Attribute | Meaning |
| --- | --- |
| `window_s` | the time actually covered, from the first to the last Sample in the window |
| `samples` | number of Samples in the window |
| `fields` | dict from field name to a `FieldStats(mean, max, p95)` |
| `warnings` | tuple of short messages; empty when nothing happened |

`fields` has an entry for each rate and size field that had values in the window: the `gpu_*`, `cpu_*`, `mem_*`, `power_w`, `ane_mem_bytes`, `children_*`, `sys_*`, `swap_used_bytes` and `mem_pressure` fields.

The means of `gpu_util_pct`, `gpu_util_raw` and `power_w` come from the cumulative counters, (last − first) / elapsed time, so they are exact even when single Samples are uneven. The max and p95 come from the individual Samples.

`warnings` can contain:

- `"memory pressure reached warning"` or `"memory pressure reached critical"`
- `"swap use grew by N MiB"`
- `"thermal state reached serious"` or `"thermal state reached critical"`, which usually means the Mac is slowing itself down

`w.to_dict(prefix="")` flattens it to keys like `"gpu_util_pct/mean"`, `"gpu_util_pct/max"` and `"gpu_util_pct/p95"`:

```python
w = mon.stats(window=60)
print(w.samples, w.window_s)                       # for example: 60 59.0
print(w.fields["gpu_util_pct"])                    # for example: FieldStats(mean=88.8, max=96.1, p95=94.3)
writer.add_scalars("mac", w.to_dict(), epoch)
```

## snapshot

```python
mm.snapshot(pid=None) -> Snapshot
```

Reads the raw cumulative counters of a process (this one by default) once, with no Monitor and no rates. Use it when you want totals or will compute differences yourself.

| Field | Meaning |
| --- | --- |
| `t_wall`, `t_mono_ns`, `pid`, `status` | as in `Sample` |
| `gpu_time_ns` | total GPU time of the process's open GPU connections |
| `cpu_time_ns` | total CPU time since the process started |
| `energy_nj` | total CPU energy since the process started, in nanojoules |
| `mem_footprint_bytes`, `ane_mem_bytes`, `ane_connected`, `gpu_mem_bytes`, `gpu_mem_limit_bytes` | as in `Sample` |
| `sys_*`, `mem_pressure`, `swap_used_bytes`, `thermal_state` | as in `Sample` |

```python
a = mm.snapshot()
work()
b = mm.snapshot()
print("GPU seconds:", (b.gpu_time_ns - a.gpu_time_ns) / 1e9)
```

`Snapshot.to_dict(prefix="")` works like `Sample.to_dict`.

## device_info

```python
mm.device_info() -> DeviceInfo | None
```

Static facts about the Mac, read once and cached. None only if nothing could be read.

| Field | Example |
| --- | --- |
| `chip` | `"Apple M2"` |
| `gpu_class` | `"AGXAcceleratorG14G"` |
| `gpu_cores` | `8` |
| `memory_bytes` | `17179869184` (unified memory shared by CPU and GPU) |
| `macos`, `macos_build` | `"27.0"`, `"26A428"` |
| `apple_silicon` | `True` |
| `rosetta` | `False`; True if this Python runs as x86_64 under Rosetta |
| `vm` | `False`; True inside a virtual machine |

## is_supported

```python
mm.is_supported() -> bool
```

True on an Apple Silicon Mac with an Apple GPU. Use it to skip GPU reporting on other machines; the rest of the API works anyway and returns `status="unsupported"`.

## stop_all

```python
mm.stop_all()
```

Stops every Monitor in the process. Useful in notebooks, where re-running a cell can leave old Monitors behind.

## Command line

```text
metalmeter [--pid PID] [--children] [--interval SECONDS] [--json] [--count N]
metalmeter info [--json]
metalmeter dump [--pid PID]
python3 -m metalmeter ...
```

| Option | Meaning |
| --- | --- |
| no `--pid` | show the system view: GPU, GPU memory, memory pressure, swap and thermal state |
| `--pid PID` | show one process |
| `--children` | include the process's descendants (needs `--pid`) |
| `--interval SECONDS` | time between readings, 0.1 to 60, default 1 |
| `--json` | print one `Sample.to_dict()` JSON object per line instead of text |
| `--count N` | stop after N readings |

A process view stops when the process exits, and `--pid` with a PID that does not exist exits with status 1.

`metalmeter info` describes the Mac and whether each data source can be read (`metalmeter info --json` for JSON):

```text
$ metalmeter info
metalmeter 0.1.0.dev0, Python 3.13.0 arm64
chip          Apple M2 (8-core GPU, AGXAcceleratorG14G)
memory        16.0 GiB
macOS         27.0 (26A428)
supported     yes
sources
  GPU clients       ok, 53 clients
  GPU statistics    ok (AGXAcceleratorG14G)
  process counters  ok (rusage v6)
  Neural Engine     ok (H1xANELoadBalancer, H11ANEIn)
  system            ok
  own GPU memory    read once this process uses the GPU
```

`metalmeter dump` prints one raw reading of every source as JSON, for bug reports. It includes every GPU and Neural Engine client on the Mac, but every process name except the dumped process's own is replaced by a hash. By default the dumped process is the `metalmeter` command itself; use `dump --pid N` for another.
