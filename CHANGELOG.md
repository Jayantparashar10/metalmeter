# Changelog

## 0.1.0

The first release.

- `Monitor` watches one process, and optionally its descendants, on a background thread or on demand (`background=False`). It keeps a history of Samples, summarises a time window with `stats()` (mean, max and p95 per field, plus warnings for memory pressure, swap growth and thermal throttling), and can call an `on_sample` function on its own thread without ever delaying sampling.
- Per process: GPU time and utilization, the process's own GPU memory, CPU utilization, memory footprint, CPU energy and power, and Neural Engine connection and memory. System-wide: GPU utilization and memory, memory pressure, swap and thermal state.
- `snapshot()`, `device_info()`, `is_supported()` and `stop_all()`.
- The `metalmeter` command: a live system or process view (text or JSON), `metalmeter info` and `metalmeter dump`.
- Safe in forked children and at interpreter exit; never raises into the caller.
- Pure Python with no dependencies. Needs no root, starts no subprocesses and reads only public macOS data.
- Tested on an Apple M2 with macOS 27.0, on CPython 3.9 and 3.13. Measured accuracy is in [docs/accuracy.md](https://github.com/Jayantparashar10/metalmeter/blob/main/docs/accuracy.md).
