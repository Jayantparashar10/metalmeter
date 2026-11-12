# Contributing to metalmeter

Bug reports, results from other Macs and pull requests are all welcome. This page covers the development setup, how metalmeter works inside, and the rules the code follows.

## Check another Mac

The most useful contribution right now is a run of the hardware checks on a chip other than the M2 (M1, M3, M4, M5, and any Pro, Max or Ultra). None of it needs root.

```sh
git clone https://github.com/Jayantparashar10/metalmeter.git
cd metalmeter
python3 -m pip install torch mlx coremltools numpy

python3 validation/validate.py --activity-monitor             # about 6 minutes
caffeinate -i python3 validation/validate.py soak --hours 8   # optional; keeps the Mac awake while it runs
python3 -m metalmeter dump > dump.json
```

Close other GPU-heavy apps first and keep the Mac plugged in. With `--activity-monitor`, the first check pauses and asks you to type the "% GPU" value Activity Monitor shows for the test process (CPU tab; add the column with View > Columns > % GPU if it is hidden).

Then open an issue and attach `validation/results/<chip>-macos<version>-<build>.json` and `dump.json`. The dump lists the GPU clients on your Mac, but other processes' names are replaced by hashes.

## Development setup

```sh
git clone https://github.com/Jayantparashar10/metalmeter.git
cd metalmeter
python3 -m pip install -e .
python3 -m pip install pytest ruff
python3 -m pip install torch mlx coremltools numpy   # only for the live tests, fixtures and validation/
```

None of these are runtime dependencies: the package itself uses only the standard library.

```sh
python3 -m pytest -q                       # the whole suite, about 25 s
ruff check metalmeter tests validation     # lint; rules are pinned in pyproject.toml
python3 tests/fixtures/capture.py          # record new fixtures (needs a Mac with a GPU and torch)
python3 validation/validate.py             # hardware checks (needs a Mac with a GPU and torch)
```

macOS-only tests skip on other systems, tests that need a GPU skip in virtual machines, and the live tests skip without PyTorch. The parser and sampler tests run on Linux from recorded readings.

## Layout

```text
metalmeter/
  _bind/        ctypes bindings: CoreFoundation, IOKit, libproc, sysctl, Objective-C, Metal
  sources/      readers that return plain dicts or lists, or None; they never raise
  parse.py      pure functions from those dicts to fields; no macOS imports
  sampler.py    pairs of readings to Samples with rates; snapshot, device_info
  monitor.py    Monitor: threads, shared samplers, history, window statistics, fork handling
  cli.py        the metalmeter command
  types.py      Sample, Snapshot, DeviceInfo, WindowStats
tests/          pytest suite; tests/fixtures/ holds recorded readings
validation/     hardware checks and their results per Mac
docs/           API reference and accuracy notes
```

## How it works

Each reading calls every source once and produces a plain dict, in the same format as the fixture files and `metalmeter dump`. The sampler only does arithmetic on these dicts, and the parsers are pure functions, which is why most of the tests run on Linux from recorded readings.

**GPU time.** The GPU driver keeps an IORegistry entry for every GPU connection (user client) a process opens. They are found by walking the children of every `IOAccelerator`; matching on the client class does not find them. Each client's `IOUserClientCreator` property (`"pid 410, WindowServer"`) names the owning process, and its `AppUsage` array holds one `accumulatedGPUTime` per command queue. If the registry changes during a walk, the walk is repeated, up to three times. A client's creator never changes, so it is read once and remembered by registry entry ID.

**Process counters.** `proc_pid_rusage` gives CPU time (in Mach ticks, converted with `mach_timebase_info`), memory footprint, energy, Neural Engine memory, and the process start and exit times. metalmeter asks for the v6 layout and falls back to v4 on macOS 12.

**GPU memory.** `MTLDevice.currentAllocatedSize`, read only for metalmeter's own process and only after that process already has a GPU client.

**System figures.** The accelerator's `PerformanceStatistics` dictionary, `kern.memorystatus_vm_pressure_level` and `vm.swapusage` through sysctl, and `NSProcessInfo.thermalState`.

Rates follow three rules:

1. A rate needs two readings, so the first Sample has none.
2. A counter that goes backwards gives None for that Sample, never a negative value or a spike.
3. A process is its PID plus its start time. A new start time on the watched PID means the watched process has exited; in a process tree it means a new child.

GPU time is tracked per client. When a client closes, it drops out of the next difference instead of making the process total go down. A client that a disturbed walk missed keeps its previous baseline, so it does not look new when it comes back. A child process that started after the previous reading counts in full; an older process seen for the first time only sets a baseline.

## Rules the code follows

These are enforced by tests where possible. Please keep them in pull requests.

- **Read only, no root.** No `sudo`, no `powermetrics`, no setuid. If something can only be measured as root, document the gap instead.
- **No subprocesses in the package.** No `ioreg`, `ps`, `sysctl` command or shell. Every test runs with process creation blocked; a test that needs a child Python is marked `allow_subprocess` and uses `run_child()` from `tests/conftest.py`.
- **No driver connections.** Only `IORegistryEntryCreateCFProperty`; never `IOServiceOpen` or `IOConnectCall*`.
- **Public frameworks only:** CoreFoundation, IOKit, Foundation, Metal, libSystem and libobjc. No network, and no changes to environment variables.
- **No runtime dependencies and no compiled code.** The wheel is `py3-none-any`.
- **Privacy.** Return only the watched process, its children and system totals. Never put other processes' names in a Sample; `dump` hashes them.
- **Never raise into the caller, never guess.** A value that cannot be read is None plus one warning. None means "not available"; 0 means "measured zero".
- **Only the sampler holds state.** Sources return plain dicts or None; parsers are pure.
- **Don't create an `MTLDevice`** unless the process already has a GPU client.

Rules for `metalmeter/_bind/`:

- Bind every foreign function with `_ffi.bind(lib, name, restype, *argtypes)`, and load libraries only with `_ffi.load()` and one of the path constants in `_ffi.py`. A test fails if a foreign function is missing from the registry.
- Call `objc_msgSend` only through a cast made by `objc.msg(restype, *argtypes)`. On arm64 a mismatched signature is undefined behaviour.
- Release every CoreFoundation object from a Create or Copy call exactly once (`cf.owned`), and never release one from a Get call.
- Check pointers for NULL, and CF objects with `CFGetTypeID`, before reading them.
- Every binding module must import on any OS; off macOS its functions are None.

Style:

- Keep `Optional[...]` in annotations rather than `X | None`, so Python 3.9 still works.
- Comments and docstrings explain why, not what. Docstrings have a summary line, then details, with `Args:` and `Returns:` where the shape is not obvious.

## Fixtures

Fixtures are named `<chip>-macos<version>-<load>.json`. Each one is a single reading taken while a load ran, in the format `metalmeter dump` writes, so it can be passed to the parsers or fed to a sampler through its `reader` argument. The mutation tests delete or retype every key in every fixture and check that nothing raises.
