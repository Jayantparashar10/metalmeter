# Accuracy and limits

All figures on this page were measured on an Apple M2 (8-core GPU, 16 GB) with macOS 27.0 (build 26A428). Other chips may differ; see [Not yet checked](#not-yet-checked).

## Validation checks

`validation/validate.py` runs these checks through the public API. Each test load runs in its own process and is watched from outside, the same way a Monitor watches a training script.

| Check | Expected | Apple M2, macOS 27.0 |
| --- | --- | --- |
| Own-process GPU % | mean above 80% over 30 s | PyTorch MPS 88.8%, MLX 95.1% (whole GPU 99%) |
| Isolation | an idle process reads below 2% while another runs a load | 0.0% with no GPU connection, 0.0% with an idle one |
| Children | parent alone 0%, `children=True` above 80% | 0.0% and 91.4% |
| GPU memory | a 1 GiB allocation adds 1 GiB ± 5% and matches PyTorch | +1.0000 GiB; 1,082,540,032 bytes in both |
| Energy | record whether GPU energy is included | not included: 0.00 W at 99.8% GPU, 4.52 W for one busy core |
| Neural Engine | record where Neural Engine use shows | on the script's own process (connected, 6,651,904 bytes) |
| Fork | no crash in a `multiprocessing` fork child | child exits cleanly; the inherited Monitor is stopped; a new one works |
| Overhead | under 1% of one core at 1 s; a read under 5 ms | 0.36% of a core; reads 0.5 s apart: median 4.05 ms, max 4.42 ms |
| Soak | memory footprint flat over 8 hours at 0.1 s | in progress |

## Per-process GPU time

GPU time comes from the GPU driver's own accounting, one counter per command queue of each process. It is isolated between processes: work in one process never shows up in another, and an idle process reads exactly 0 while another keeps the GPU at 99%.

The driver adds to the counter when a command buffer finishes, not while it runs. With the short command buffers that training loops usually submit, this is invisible at a 1 s interval. With long ones, a single Sample can read 0% and the next one more than 100%. That is why `gpu_util_pct` is capped at 100, why `gpu_util_raw` keeps the uncapped value, and why the window means from `Monitor.stats()` are computed from total GPU time, which is exact over the window.

The per-process figure reads below the whole-GPU figure by an amount that depends on the workload. Each row is a 10 s window of back-to-back single matmul kernels in PyTorch MPS, measured three times:

| Matrix size | Time per kernel | Per-process % | Whole GPU % |
| --- | --- | --- | --- |
| 2048 | 0.007 s | 87 to 94 | 93 to 96 |
| 4096 | 0.057 s | 83 to 90 | 98 to 99 |
| 6144 | 0.21 s | 67 to 82 | 99 |
| 8192 | 0.53 s | 90 to 91 | 99.3 to 99.5 |
| 10240 | 1.0 s | 83 to 90 | 99.5 |
| 12288 | 2.5 to 3.0 s | 44 to 76 | 99.7 |

For command buffers up to about 1 s, the per-process figure is 5 to 30 points below the whole-GPU figure. Command buffers that run for several seconds are undercounted: each one adds less GPU time than its wall-clock length. The cause is not known, and it is not settled which of the two figures is closer to the truth. A comparison with Activity Monitor's "% GPU" column is still to be done.

In practice: use `gpu_util_pct` to compare runs, spot an idle GPU (data loading stalls, CPU bottlenecks) and see which process does the work. Do not read it as an exact share of the GPU's capacity.

## GPU memory

`gpu_mem_bytes` is the process's own Metal allocation. After a 1 GiB allocation in PyTorch MPS it grew by exactly 1,073,741,824 bytes and matched `torch.mps.driver_allocated_memory()` to the byte, while another process's value did not change.

It is only available for metalmeter's own process. macOS has no public way to read another process's GPU memory. It is also only read once the process already uses the GPU: metalmeter does not open a Metal device in a process that has none, because that would waste GPU resources and can make a later `fork` crash.

## Energy

The per-process energy counter does not include GPU energy:

| Load | CPU % | GPU % | `power_w` |
| --- | --- | --- | --- |
| Idle | 0.02 | 0 | 0.00 |
| CPU busy loop on one core | 99.3 | 0 | 5.07 |
| PyTorch MPS matmul | 0.6 | 89.8 | 0.00 |
| MLX matmul | 0.6 | 98.4 | 0.00 |

Read `energy_j` and `power_w` as the process's CPU energy and power.

## Neural Engine

macOS has no busy counter for the Neural Engine, so metalmeter reports two per-process signals, `ane_connected` and `ane_mem_bytes`, and no utilization percentage.

With a Core ML model whose operations all run on the Neural Engine, the script's own process was connected in every Sample and held 6,651,904 bytes of Neural Engine memory. The `aned` system daemon also holds a connection on its behalf; its counters belong to another user, so they cannot be read without root.

A connection does not mean the Neural Engine is working: several system processes hold connections while idle. Training frameworks run on the GPU, not the Neural Engine, so `ane_mem_bytes` is normally 0 during training.

## Cost

A Monitor at the default 1 s interval uses 0.36% of one CPU core. One reading takes about 0.3 ms when readings follow each other closely. With a pause between readings, as in normal use, macOS moves the sampling thread to an efficiency core and a reading takes 2 to 4.5 ms. Intervals well below 1 s are for short diagnostic runs: two Monitors at 0.1 s use about 5% of a core.

## Not measured

- GPU or CPU power in watts, and GPU frequency. These need IOReport, a private macOS framework.
- Neural Engine utilization or time per process. macOS does not attribute Neural Engine work to processes.
- Another process's GPU memory.
- Per-kernel or per-layer timing. Use `torch.mps.profiler` or Xcode Instruments.
- CPU time, memory and energy of another user's processes: macOS refuses those reads without root, so the fields are None. GPU time is still available.

## Not yet checked

- Agreement with Activity Monitor's "% GPU" column.
- Chips other than the M2, Ultra chips in particular. If more than one GPU reports statistics, the `sys_*` GPU fields are None and a warning is raised rather than combining them.
- macOS 12, where metalmeter falls back to an older counter format without energy and Neural Engine memory.
- Free-threaded Python (3.13t and later).

If you have a different Mac, [CONTRIBUTING.md](../CONTRIBUTING.md#check-another-mac) explains how to run the checks and share the results.
