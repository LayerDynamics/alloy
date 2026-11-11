# Native macOS ownership and broker recovery

Shell commands and trusted plugins execute directly on the host Mac. The broker
runs as a per-host user launchd job. Each invocation runs in its own resource
coalition, with an independent cleaner, target owner and revocation holder.
Ordinary `fork`, `exec` and `posix_spawn`, including new sessions and process
groups, remain available while the invocation is running.

The existing deny-default, single-process plugin sandbox retains its separate
launch path. Its policy denies descendant creation; applying another Seatbelt
profile over that sandbox would fail. The plugin host selects this path from its
host-owned `Sandbox::MacOs` policy, rather than from a model-supplied argument.

## Broker lifecycle and authentication

The host submits an in-memory job definition through user-domain ServiceManagement.
Registration creates no administrative child in the embedding process. Competing
`waitpid` callers and automatic child reaping cannot consume the broker's status.
The target owner exclusively retains its child's wait status.

Job labels contain the UID, host PID, kernel process unique identity and a random
suffix. The owner-only registry is `/private/tmp/alloy-brokers-<uid>`. A shared
`flock` serializes host registration and stale retirement. Inconclusive identity
observations retain the registration and its capacity charge.

The broker uses `KeepAlive=false`, `RunAtLoad=true`, and an on-demand Mach service
with `ResetAtClose=true`. A crash leaves an inactive job until another request
activates it. A fresh instance supplies a new random 128-bit generation.
Authentication checks the kernel audit token, effective UID, process identity and
code-signing designated requirement against the expected helper. Requests use
private reply ports; stale generations are rejected.

No possibly-started command is automatically replayed. Losing a request or reply
publishes `STOP` in the independent shared control channel. A late owner cannot
advance through `CREATE` or `RUN` after that cancellation. Already-running
invocations retain their control channels across broker loss. An interrupted
launch reports failure while subsequent commands can reconnect.

## Descendant containment

A new per-invocation launchd job starts alone in a distinct resource coalition.
Its trusted gate applies a Seatbelt profile with a randomly named extension
controlling `process-fork`. The gate consumes the extension before creating the
independent revocation holder and executing the workload. It erases the token,
class string and profile buffer before workload execution.

The inherited grant allows normal subprocess creation. Cancellation explicitly
releases it; process exit alone does not revoke inherited grants. Both trusted
holders can release the same inherited lease, so the surviving holder can revoke
after the other dies. The profile denies reminting that class, task-port access
to other processes, signals except to self/children, launchd job creation,
AppleEvents, hard-link creation, and reads/writes in Alloy's recovery registry. These controls protect the holders from workload interference.
The holders use a separate session so launchd's cleanup of a dead cleaner's
process group does not kill both revocation authorities.

After revocation the cleaner enumerates resource-coalition members. It brackets
each membership observation with stable kernel process identity, then uses
`proc_signal_with_audittoken` to signal that exact identity. Process-group and
session IDs are not ownership evidence. Double-forked, reparented, `setsid` and
`setpgid` descendants remain members of the same resource coalition.

The resource coalition's started-minus-exited task count includes adopted tasks
before they become visible as processes. Cleanup requires successful revocation,
a complete census and a final task count of one after both trusted holders exit.
That remaining task is the cleaner itself. A truncated census, unknown ownership,
failed signal or incomplete drain does not publish successful completion.

The quiescence argument is based on the tested kernel's implementation: child
adoption is synchronous with fork/spawn, and a parent task remains counted until
its last terminating thread exits. Thus a kernel thread creating a child is
represented by the parent or adopted child during teardown. This is an inference
from Apple's implementation, supported by synchronized creation-at-cancellation
tests; it is not a documented public coalition teardown API.

If the cleaner dies, the embedding host retains the coalition identity and
revocation acknowledgement. A bounded recovery attempt drains that coalition to
zero tasks and reports the interrupted operation as an error. `try_wait` starts
one background recovery worker instead of blocking on that drain. If a holder is
paused beyond an attempt's deadline, the charged slot and its single watcher
remain alive even after the caller drops its control. New revocation or kernel
absence evidence permits recovery to continue without replaying a command.

This is descendant-process lifetime control, not isolation of a trusted program's
entire user account, filesystem or network authority. The separate declared-I/O
plugin sandbox supplies those restrictions for untrusted plugins.

## Durable recovery and bounds

Each invocation reserves a record before `SMJobSubmit`. It holds the shared
registry lock through the submission call. The service atomically replaces and
syncs that record with its kernel coalition and cleaner identity before creating
a workload. The record includes `kern.bootsessionuuid`: a stale coalition number
from a different boot is never used to signal current processes.

On this kernel coalition IDs are monotonically allocated 64-bit values, without
reuse before counter wrap. This implementation assumption is distinct from a
public API guarantee. Stable per-process identity checks are still required for
every signal.

The independent holder publishes revocation in the durable record. A new host can
therefore finish cleanup after both the previous host and cleaner disappear.
An already-empty or reaped coalition needs no revocation marker. Job removal and
record retirement require positive no-workload/cleanup evidence.

A submission marker is synced before entering ServiceManagement. If the submitter
dies inside that RPC, a momentarily absent job is not proof of cancellation.
An unresolved record remains charged until a service publishes completion or a
known registration can be explicitly retired. If no positive resolution ever
arrives, that registration capacity remains unavailable, including across host
restarts. Time alone never deletes the record or authorizes a signal.

Bounds are 64 retained host registrations per user and 64 invocation records per
host directory, including uncertain submissions. A host also has at most 64
outstanding ownership slots and at most one recovery worker per charged slot.
Reconnection permits three handshakes per ten-second window. Handshakes and
launch replies have a five-second internal bound, additionally subject to caller
cancellation. Recovery attempts are bounded; an unresolved watcher retains
ownership and capacity while waiting for new evidence. After revocation is known,
transient cleanup failures retry bounded attempts with a one-second backoff.

Trusted plugin I/O still comes from the configured fileport helper. The existing
pipe ends travel with shared state over authenticated Mach messages into the
cohort. Helpers are validated before any cached broker connection is created,
so failed helper messages neither run a plugin nor leak host rights/descriptors.
Only the intended standard streams reach the workload.

## Compatibility and verification

Live qualification is on macOS 26.4 arm64, build 25E246, Darwin 25.4.0,
`xnu-12377.101.15~1`. A deployment-target build for an older macOS version is not
live qualification of that version. Missing native interfaces fail closed.

`SMJobSubmit`, `SMJobRemove` and `SMJobCopyDictionary` are public legacy APIs,
deprecated since macOS 10.10. The sandbox extension APIs, custom Seatbelt policy
and coalition queries are private/deprecated interfaces. Their behavior and ABI
must be requalified when macOS changes; API presence alone is insufficient.
Unbundled CLI/embedding hosts cannot assume the app-bundle packaging required by
`SMAppService`. No root service, VM or system-wide signal policy is installed.

Relevant primary implementation sources for the qualified kernel:

- [Resource coalition accounting and IDs](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.101.15/osfmk/kern/coalition.c)
- [Fork and spawn child creation](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.101.15/bsd/kern/kern_fork.c)
- [Task creation and inherited coalition adoption](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.101.15/osfmk/kern/task.c)
- [Last-thread coalition removal](https://github.com/apple-oss-distributions/xnu/blob/xnu-12377.101.15/osfmk/kern/thread.c)
- [Sandbox extension SPI declarations](https://github.com/WebKit/WebKit/blob/main/Source/WTF/wtf/spi/darwin/SandboxSPI.h)

Run the native regressions on an ordinary logged-in Mac:

```sh
cargo test --locked -p alloy-process-owner --all-targets -- --nocapture --test-threads=1
cargo test --locked -p tool-registry --test shell_execution -- --nocapture
cargo test --locked -p harness-core --test plugin_hooks -- --nocapture --test-threads=1
```

The acceptance collector discovers every broker/containment case explicitly.
Fault tests cover broker loss before/during launch, execution and cancellation;
real owner/revoker/cleaner death; host loss across durable startup publication;
and delayed recovery after caller drop. The concurrent-creation test holds cleanup
after actual revocation until fork and both spawn variants report `EPERM`, then
independently checks owned PID disappearance. Debug fault hooks and barriers are
omitted from release helper code.
