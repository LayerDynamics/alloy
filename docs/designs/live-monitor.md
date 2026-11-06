# Live system and runtime monitor

## Outcome

`alloy monitor` provides a continuously refreshed, read-only terminal view
while Alloy runs. It combines system-wide measurements from the local host
with runtime ownership measurements from the existing local daemon socket.
It also works when the daemon is offline so host utilization remains visible.

The Apple-Silicon display covers CPU, GPU and Apple Neural Engine (ANE), with
one shared physical-memory capacity. Alloy continues using the existing
unified CPU/Metal execution path. Observing ANE system activity introduces
no new inference backend and does not attribute other applications' work to
Alloy.

This observability slice supports M9 investigation. It does not replace the
representative two-large-model acceptance or mark M9 complete.

## Terminal layout

The interactive view uses colored, bordered system, accelerator and runtime
panels. Wide terminals place system and hardware measurements side by side;
80×24 terminals stack the panels and retain connection state, admission,
pressure, cancellation and exit controls. Full panels require 60 columns and
22 terminal rows (one row is reserved by the live loop). Smaller windows keep
CPU, RAM and GPU bars plus recent CPU/GPU history in a compact view, with a
border when space permits. Very short windows select pressure, cancellation
and freshness labels before optional readings instead of truncating the last
lines. Resizing takes effect on the next
sampling frame. `q` and Ctrl+C restore the previous terminal input mode,
cursor and screen. The renderer clears old row tails when text shrinks without
blanking the whole screen between frames.

Usage bars have a fixed 0–100% scale for CPU, GPU active residency, non-free
RAM, swap and reservations. A reservation exceeding a reduced budget fills
its bar while retaining the actual byte counts. System RAM bars do not infer
pressure or change color to claim an emergency. Only the daemon's pressure
classification drives its pressure indicator.

At most 60 recent frames are retained in memory for the history graphs.
Each source has independent timestamps: failed, repeated or more than
five-second-old readings become gaps, never fabricated zeros. `·` indicates
a gap; `▁` indicates the bottom of the scale, including reported zero.
Host changes reset history, and a changed daemon PID clears only the old
reservation history. Graphs represent successive samples, not an interpolated
time axis; the header shows the retained observation span.

Estimated CPU/GPU/ANE power bars share a labeled scale for the current sample.
The ANE history has its own power scale printed on the graph's line. Neither
view converts ANE power into an occupancy percentage. Hardware freshness and
collection errors remain visible, including in the compact view. Daemon
unavailability does not imply that the process is absent, and cached runtime
values remain explicitly stale.

Color is enabled only for the interactive terminal and can be disabled with
`NO_COLOR=1`. `TERM=dumb` and redirected text produce one plain snapshot.
The JSON capture schema is unchanged and never includes terminal escapes or
the local rendering history. Names and errors are sanitized before display;
layout uses terminal cell widths for CJK, emoji and combining characters.

## Execution defaults

A native Apple-Silicon `alloyd` build already selects the shared CPU/Metal
path by default; `--metal` repeats that default. CPU work includes artifact
loading, tokenization, request scheduling and token selection. Transformer
operations run through an explicit Metal stream, with one model weight set,
shared KV storage and one admission ledger. The current backend chooses a
transformer stream; it does not automatically split transformer operations
across CPU and Metal streams.

`--cpu` explicitly selects MLX CPU transformer operations for portability or
testing, and CPU is the default for other build targets. Requested Metal
execution fails if the device is unavailable; it never silently falls back
to CPU. See the [daemon defaults](../../crates/alloyd/src/main.rs) and
[backend stream selection](../../crates/engine/src/backends/qwen.rs).

The monitor connects to the selected daemon socket. It does not start the
daemon or change its placement, and system GPU activity alone does not
identify which execution path that daemon uses.

When disconnected, the monitor shows a local readiness action: guided
`alloy start`, configuration/model repair through `alloy setup`, or socket/API
diagnosis. These are suggestions, never implicit mutations. Setup advice uses
bounded model preflight and is cached for ten seconds. Saved configuration is
reloaded during these checks; an explicit `--socket` always takes precedence.
If the saved socket changes, the old reader is closed and the next poll uses
the new socket. See [the startup contract](startup.md). Advice is display-only
and does not add fields to the monitor JSON schema.

## Measurements and their meaning

| Section | Measurements | Source and limits |
| --- | --- | --- |
| Host | CPU utilization across logical cores, load averages, installed/free/non-free RAM, uptime | Node OS counters; CPU is a delta between observations; non-free RAM includes reclaimable pages and is not Activity Monitor's app-memory total |
| macOS memory | Wired and compressed memory, swap usage | Bounded `vm_stat` and `sysctl` reads; missing probes remain unavailable |
| macOS GPU | Driver-reported device utilization, per-device identity and counters | Unprivileged IORegistry snapshot through `ioreg`; sampled by default, independently of native power collection; driver-dependent availability |
| Apple Silicon hardware | CPU/GPU/ANE estimated component power, GPU active residency, CPU/GPU frequency-state averages and thermal pressure | Bundled unprivileged native sampler; IOReport counters, IORegistry frequency tables and Foundation thermal state; component estimates are not whole-machine power |
| Alloy | Reserved/available bytes, current budget, reservation count, admission pause, request/cancellation state | Read-only daemon metadata, without dispatching a native-worker command |
| Residency and pressure | Last published model metadata, pressure classification and ceiling breakdown | Explicit observation age and errors; ownership survives cancellation until actual cleanup |

The default GPU bar uses `PerformanceStatistics` device utilization from
`/usr/sbin/ioreg -a -r -d 1 -c IOAccelerator`. The class filter includes
subclasses and the depth limit omits client children. The bounded plist reader
preserves per-device identity and accepts only finite percentages from 0 to
100. `Device Utilization %` takes precedence over the alternate
`GPU Activity(%)` key. Renderer and tiler counters are not combined into a
device total. With multiple devices, the summary is explicitly the busiest
reporting GPU; it is never a sum or average. These driver-defined counters
are best-effort rather than a stable telemetry contract across every driver.
See [Apple's ioreg manual](https://github.com/apple-oss-distributions/IOKitTools/blob/main/ioreg.tproj/ioreg.8)
and the [Stats GPU reader](https://github.com/exelban/stats/blob/master/Modules/GPU/reader.swift).

ANE power is measured in watts/milliwatts, not percent occupancy. A missing
ANE field is unavailable, not zero. GPU active residency is the fraction of
the sampling interval in an active state, not a count of saturated GPU cores.
Native power uses actual monotonic sample duration and explicitly supported
energy units. A measured zero retains its source; a missing channel, invalid
delta or unknown unit stays unavailable. Valid CLPC energy counters take
precedence over legacy Energy Model readings, with supported PMP ANE counters
as an independent fallback. These sources must never be summed as duplicate
measurements of the same component.

Frequency values are residency-weighted averages of active frequency states,
using the machine's IORegistry DVFS tables. Inactive states are excluded from
that average; hardware with no active residency is labeled idle. Table/state
mismatches remain unavailable. On M4, these reported
CPU states can represent requested frequency rather than the throttled hardware
clock; the display therefore labels them as state averages. Thermal pressure
comes independently from Foundation's public process thermal-state API.
The retained legacy plist parser still accepts both observed GPU `freq_hz`
encodings for old captures; it is not used to launch a power sampler.
System RAM, allocator activity, daemon footprint and reserved bytes overlap;
they must not be summed as independent CPU and GPU memory pools.

## Collection and ownership

The TypeScript client samples the host and calls the new `daemon.metrics`
JSON-RPC method using `api_version: 1`. The response has `schema_version: 1`,
the daemon process ID, a wall-clock response timestamp, live admission and
execution metadata, and separately aged residency and memory-guard data.
The existing `daemon.status` and planned session-history `daemon.stats`
contracts remain unchanged.

The runtime publishes only non-owning residency metadata at worker
boundaries. Reading metrics never acquires a model lease, reserves memory,
changes pressure policy, cancels a request or waits for an executing native
operation. Cancellation-requested remains distinct from cleanup completion.
Pressure failures retain the last good observation with its real age and an
error instead of making old data appear fresh.

Host and daemon observations are independent and not one atomic sample.
The client serializes its polling, bounds command/RPC waits, and displays
connection and sampling failures. Terminal shutdown restores input/cursor
state and stops sampling. GPU registry probes and native power samples have
bounded lifetime and are both settled before collector shutdown completes.

Monitor JSON adds an independent `host.gpu` source with its own status,
timestamp, age, utilization and device entries. The existing
`host.accelerators` fields retain their units and now identify the native
source as `ioreport`, including per-component power provenance and warnings.
Readers still
accept older version-1 captures without `host.gpu`; their GPU line remains
explicitly labeled active residency. Changing measurement sources clears
the GPU trend rather than joining different definitions. GPU and ANE history
deduplicate their own source timestamps; failed power sampling cannot erase
a fresh registry reading or make a stale GPU reading current.

## Unprivileged macOS hardware collection

The ordinary `monitor` command samples all supported hardware fields without
administrator access. No path invokes `sudo`, requests a password, installs a
privileged service, or falls back to Apple's `powermetrics` command. The old
`--powermetrics` argument is accepted only as a compatibility alias; it cannot
select a different permission level or disable the default collector.

`npm --prefix ts/cli run build` compiles the bundled Objective-C source at
`ts/cli/native/macos-metrics.m` into `ts/cli/dist/native/alloy-metrics` on Apple
Silicon, using the installed Xcode Command Line Tools. Compilation is bounded,
publishes the executable atomically, and fails the build on error. Other
platforms skip the native build. The runtime resolves this executable relative
to the installed CLI, not the current directory, and never compiles, downloads
or installs tools. A missing executable is an explicit build error in the
sample rather than an authorization request.

Each native invocation collects one finite sampling window and exits. The
client validates its JSON before publication and stops its owned child during
shutdown. Subscription discovery, supported units, real elapsed duration and
per-component source metadata distinguish actual idle zero from unsupported
measurements. Partial warnings stay separate from fatal collection errors so
one missing counter does not suppress other valid readings or history.

The sampler uses private IOReport channels through the system library and
public IORegistry/Foundation APIs. Channel availability and DVFS layouts can
change with OS or hardware versions; unsupported data stays null. Native API
references include the pinned [macmon source](https://github.com/vladkens/macmon/tree/880f641ed67628ae92713928cd6806afa3c51d04)
and [Stats declarations](https://github.com/exelban/stats/blob/482da327d4b19d54a55dfc34b2a316f359e68f41/Modules/Sensors/bridge.h).
These are references for the bundled sampler, not separately installed runtime
applications.

## Capture and acceptance

`--once --json` emits one machine-readable snapshot. `--json` without
`--once` emits newline-delimited snapshots for local recording. Each capture
includes data source/freshness and failures so an absent or stale probe cannot
masquerade as a successful zero reading. Monitoring does not send these host
samples through anonymous telemetry.

The capture's `alloy.sample_age_ms` is the age of the retained daemon response
at frame publication. Within `alloy.metrics`, residency and guard ages retain
their meaning at that daemon response's `sampled_at_unix_ms`; add the outer
response age to obtain their age at capture. The terminal performs that
addition. Host, GPU, power and outer daemon ages are recalculated at frame
publication after any local readiness checks, so time spent preparing advice
does not make old samples look fresh. A failed host poll marks retained
GPU and power data stale.

Required checks for this feature:

1. System CPU/memory/swap remain visible when the daemon is absent; CPU deltas
   handle warm-up and invalid counters without fabricated utilization.
2. Hardware parsing distinguishes real zero power from unavailable fields,
   preserves GPU ratio units, and rejects malformed/overlarge samples.
   GPU registry parsing preserves zero utilization and device identities,
   rejects invalid percentages, and never invokes a privileged helper.
   The Apple-Silicon CI host must provide a real GPU utilization reading
   while the test process is unprivileged. Native telemetry failures must retain
   independent GPU readings and history, without requesting authorization.
   Default host sampling and the compatibility flag must both use only the
   colocated unprivileged helper for power, frequency and thermal data.
3. Live daemon metadata remains readable while the native worker is blocked;
   cancellation leaves reservations and the execution slot intact until cleanup.
4. Stale observations, lost connections and unsupported counters remain visible;
   reconnects do not retain an old daemon's values as current.
5. One-shot JSON is valid, continuous polling does not overlap, and terminal
   exit/resize restores a usable terminal without leaving an unbounded sampler.
6. Mac hardware collection is validated separately from fixture/parser tests.
   The Apple-Silicon lane must compile and run the bundled helper as a normal
   user, obtaining real CPU/GPU/ANE energy counters, frequency-state readings
   and thermal data. Missing or fabricated values cannot pass that acceptance
   check. Linux runs the portable parser, ownership and integration checks.
