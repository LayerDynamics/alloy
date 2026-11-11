# Native macOS shell broker recovery

The shell broker runs as a per-host user launchd job. A broker crash interrupts
the affected launch but permits the next command to connect to a fresh instance.
Workloads execute directly on the host Mac with ordinary macOS subprocess APIs.
This mechanism does not establish containment of deliberately detached descendants.

## Lifecycle and identity

The embedding host submits an in-memory job definition through the public
ServiceManagement user-domain compatibility APIs. Administrative operations create
no child processes in the embedding host, so unrelated `waitpid` callers and
automatic child reaping cannot consume their status. The native broker and its
supervisors own their own child wait statuses.

Each job label includes the user ID, embedding PID, kernel process unique ID, and
a random suffix. The owner-only registry is `/private/tmp/alloy-brokers-<uid>`.
Its cross-process lock serializes registration publication and stale retirement.
An inconclusive owner-identity observation retains the record and charges the
registration budget. Only a proven dead owner permits retirement. Checking whether
a job exists reads launchd job metadata; it does not look up or activate a retiring
Mach service. The diagnostic plist is not reread to submit a job.

The job has `KeepAlive=false`, `RunAtLoad=true`, and an on-demand Mach service with
`ResetAtClose=true`. After its host dies, the broker drains its supervisors and
exits. The inactive job is collected by a subsequent host under the registry lock.
A crashed broker remains inactive until requested, rather than restarting in an
unconditional crash loop. The broker receives an empty environment and supervisors
retain the normal launchd bootstrap namespace needed by macOS frameworks.

The connection handshake checks the peer's kernel audit token, effective user,
stable process identity, and code-signing designated requirement against the
expected helper. The broker authenticates the exact embedding host identity.
Every broker instance generates a random 128-bit generation. Requests and replies
carry that generation, and each launch has a private reply port.

## Failure and cancellation

No workload launch is automatically replayed. If a request or reply is lost,
the host publishes `STOP` to its independent shared control channel before
retiring it. A delayed supervisor cannot proceed through `CREATE` or `RUN` after
that cancellation. The failed operation returns its interruption; a subsequent
operation may reconnect. A stale-generation response is rejected.

Already-running supervisors keep their status and cancellation channels when the
broker dies. Their cleanup continues independently. Signals use stable kernel
identity, or a leader whose wait status is retained by its sole supervisor, rather
than an unowned numeric PID. Completion is reread after observed supervisor death
so a concurrent final status publication is preserved.

Reconnection permits three handshakes per ten-second window. A handshake or
launch reply has a five-second internal bound, additionally limited by caller
cancellation/deadline. There are at most 64 retained registrations per user and
64 outstanding shell ownership records per embedding host across broker
generations. The broker also limits its own concurrent supervisors to 64.

A confirmed pre-send failure or an authenticated no-child error reply releases
its slot. An ambiguous launch retains its mapping and capacity charge until
completion or stable identity evidence proves its supervisor is gone. Elapsed
time alone never releases that charge. In the exceptional case where no owner
identity or completion can ever be recovered, capacity remains unavailable for
that host lifetime. This fails closed rather than creating unbounded supervisors.

## Compatibility and verification

`SMJobSubmit`, `SMJobRemove`, and `SMJobCopyDictionary` are public legacy APIs,
deprecated since macOS 10.10. They remain available in the current SDK and were
live-tested on macOS 26.4 arm64. Symbols are resolved at runtime; missing APIs fail
closed. This is not a promise of support on future macOS releases. An unbundled
embedding process cannot assume the app-bundle packaging required by
`SMAppService`; replacing this compatibility layer needs a separately qualified
packaging design. Apple documents the legacy APIs in the
[ServiceManagement reference](https://developer.apple.com/documentation/servicemanagement).

Run the native recovery regressions on an ordinary logged-in Mac:

```sh
cargo test --locked -p alloy-process-owner --all-targets -- --nocapture --test-threads=1
cargo test --locked -p tool-registry --test shell_execution -- --nocapture
cargo test --locked -p harness-core --test plugin_hooks -- --nocapture --test-threads=1
```

The recovery suite injects real broker death before launch, after sending a launch,
during execution, and during cancellation. It checks next-command success, actual
marker counts, surviving unrelated work, OS-level disappearance of owned PIDs,
generation rejection, peer authentication, concurrent host retirement, restart
budget, and capacity across generations. A real stopped pre-identity process
checks that timeout cannot erase an outstanding ownership charge. Eighty
confirmed pre-send cancellations check that known no-spawn requests do not leak
capacity. These fault hooks are compiled only in debug builds.

## Remaining containment requirement

Process groups remain the cleanup boundary for trusted descendants. A descendant
that deliberately calls `setsid`, changes its group, or uses equivalent
`posix_spawn` attributes can escape that boundary. launchd's group cleanup and
broker recovery do not change this fact.

A native sandbox probe on macOS 26.4 denied the `setsid` and `setpgid` syscalls but
still allowed `POSIX_SPAWN_SETSID` and `POSIX_SPAWN_SETPGROUP`; independent PID,
session, and group observations confirmed the escape. Apple's
[XNU spawn implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_exec.c)
performs these operations internally. Denying those two syscall entry points is
therefore insufficient. No verified policy preserving normal macOS subprocess
APIs has been established. Execution must remain directly on the host, so a VM
backend is outside the accepted scope. Hard containment remains an open
requirement; this recovery implementation must not be presented as satisfying it.
