# Parallel CI verification

The CI workflow runs Linux foundation, reproducibility, engine/native, and
core/acceptance checks as separate jobs. The existing `linux-arm64` check
succeeds only if all four jobs succeed. Reproducibility still performs two
clean builds; it does not reuse a build from another shard. Linux W4 and W5
collectors retain their own candidate identifiers and artifacts.

On macOS, `mac-core` runs the approval/storage, native, and W4/W5 checks and
`mac-installed-and-real-model` runs the installed Homebrew and real-model
regressions. The existing `mac-metal` check requires both. Both jobs use the
same self-hosted macOS ARM64 runner labels and check out the exact PR head
(or push/dispatch SHA), with full history and recursive submodules. W3/W4/W5
records and installed/model evidence all use that same Mac candidate.
Fork pull requests retain the existing restriction against running on these
self-hosted machines.

Two *separate physical* Apple Silicon Macs are required to run these macOS
jobs concurrently. Each eligible runner must satisfy all prerequisites for
either job: the pinned Rust and Node toolchains, Xcode/Metal, Homebrew, and the
configured local model artifacts, including the M9 models. Runner workspaces,
model-memory resources, and Homebrew installations must be isolated. Registering
multiple runner processes on one Mac does not provide that isolation. With one
eligible runner, GitHub queues the macOS jobs and runs them serially.
For pull requests, the complete approval/storage collector runs once inside
`mac-core`. It supplies six named W3 gates, Rust provenance vectors, full source
sections, workspace check/Clippy, serial workspace all-target tests, workspace
doctests, strict workspace rustdoc, and parallel harness tests. One additional
parallel all-target invocation covers tool-registry, inference-infra, qwen3,
engine, and alloyd, preserving their previous test schedules. MLX remains serial.
W4/W5 collectors and the installed/model job remain required. A lightweight
`approval-storage-macos-arm64` aggregate succeeds only if core and its mandatory
approval evidence upload succeeded; failure, cancellation and skip fail closed.
The separate workflow retains Linux PR evidence and a renamed standalone Mac
collector for manual/feature-branch runs. Its Mac job no longer duplicates the
entire workspace suite on every PR. This removes a competing third Mac job
without removing its qualification commands or evidence.

Cooperating macOS CI shell steps also take an advisory lock at
`/tmp/alloy-ci-mac-host.lock`, shared by runner users on that host. The
approval/storage collector, including standalone runs, uses the same lock. It serializes locked steps if
multiple runner agents happen to share a Mac; separate machines have separate
locks. This coordinates trusted jobs, not other host processes, an administrator
changing the lock file, or conflicting runner workspaces. Do not rely on it to
turn one Mac into two independent machines.
Every successful step must stop and join the resource-using descendants it
starts before returning. A detached child that closes inherited descriptors
can survive a successful step, and the lock alone cannot detect it. Treat
physical runner exclusivity and each script's cleanup contract as prerequisites
for accepting timing and resource measurements.

The lock retains an unclean lease marker if a step fails or its wrapper is
interrupted before the step has been successfully reaped. A signal arriving
after successful reap, during the final clean-record write, can leave the lease
clean; the successful-step descendant cleanup contract still applies.
Later locked steps fail rather than assume detached work has stopped. After
checking that no CI work or descendants remain on that host, an operator must
reset the marker while holding the same exclusive file lock. Do not delete or
replace the lock file while jobs may be running. Older branches or unrelated
workflows that do not use this lock are not coordinated by it.

W4 and W5 uploads fail if evidence is absent. Their artifacts include the
checked-out SHA, pull-request head SHA, and checked-out tree SHA. Linux PR
checkouts can be GitHub's synthetic merge commit; the Mac lanes use the exact
PR head. Compare these identities and the logs
before attributing evidence to a submitted head.

The goal is to reduce execution time, not to skip checks. Ten minutes is a
target rather than a guaranteed deadline: cold native builds, runner capacity,
queue times, model loading, and artifact uploads can exceed it. Measure the
complete run and each shard on the candidate before claiming a duration.

Normal MLX builds now exclude the optional `bindgen` build dependency. Bindgen's
proc-macro feature unification previously changed the build-script fingerprint
between Cargo package selections and caused repeated native CMake/Metal builds
inside one workflow. Checked-in bindings remain the normal build input;
`scripts/bindgen_check.sh` explicitly enables `regenerate-bindings` for the
pinned Linux regeneration gate. This removes that source of redundant work
without dropping verification or adding competing runners to the same Mac.
Complete candidate workflow timing is still required before claiming the
ten-minute target.

Mac checkout cleanup retains only the root `target/` Cargo build cache. Checkout
still selects the exact candidate, resets tracked files, and removes every other
untracked/ignored source path before qualification. Previous W4, W5, and Phase 1
evidence under `target/` is removed before any gate can upload artifacts.
Uploads from `target/` require successful cleanup, including when checkout or
the host lease fails; later gate failures still retain current-run evidence. A
regression test executes those
cleanup commands against stale tracked, ignored, untracked and nested-target
fixtures. This avoids deleting native build outputs between the two jobs and
between runs. Cargo's normal input/toolchain/dependency checks remain enabled;
no stale-binary or skip-build option is used. The Homebrew acceptance continues
to build a separate fresh installation and verify its candidate manifest, with
the original build tree made unavailable during installed testing. Cache misses
and toolchain/vendor changes still require rebuilds and can exceed the timing
target.
