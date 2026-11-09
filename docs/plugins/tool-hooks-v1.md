# Tool hooks V1 — W5 candidate

Status: candidate under review. Acceptance requires Linux and macOS results and
an independent security decision bound to the submitted commit; prior green
checks do not validate later changes.

Transport: one JSON-RPC 2.0 request/response per newline, UTF-8 JSON, strictly
matching random u64 IDs in the exactly representable JSON integer range 0 through
2^53-1, no notifications, no batching, no retries. The ID is serialized after the payload;
plugins must echo it unchanged. A response prepared before the challenge is
revealed has a 2^-53 chance of guessing it in one attempt, assuming secure OS
randomness. Repeated independent attempts increase the aggregate probability.
Entropy failure stops the child; there is no predictable fallback. initialize declares
protocol_version=1, exact plugin_id/version and the supported subset of
 tool.before_execute and tool.after_execute. No unknown hooks or duplicate
capabilities. Unknown envelope/response fields fail the exchange. No environment
is inherited and no shell is used. Executable and cwd are absolute daemon inputs.
Pending stdout is rejected before request encoding and again before writing.
Separate pipes cannot independently prove the time of a write; the challenge
binds an accepted response probabilistically to the current request rather than
claiming a deterministic observation of the child's write time.

Before params: protocol_version and invocation (invocation/session/call IDs,
exact registered definition, schema_hash, validated arguments, approval snapshot,
deadline_unix_ms). Reply: decision allow/veto, optional reason and annotations.
After params: same invocation plus original canonical result and duration_ms.
Reply: annotations only. Keys must begin with the exact plugin ID plus a dot.
All messages, reasons, stderr and annotations have configured byte ceilings.

Host calls borrow invocation/result immutably. Plugin responses have no path to
change identity, arguments, registration, effect, policy, approval, artifact bytes,
or original result. Before failure/veto stops later before hooks. After failures
produce receipts/warnings; the host has no executor callback and cannot replay.
Registration is all-or-nothing; failed registration drops already-owned children.
A transport/protocol failure poisons and terminates that child, with no restart.
The owner kills its process group, waits the direct child, and closes its pipes.
No reader/writer worker threads are detached. Captured stderr remains available
through diagnostics while the host is alive; persist it externally as needed.

W6 boundary (NOT implemented here): derive the snapshot from W3 validated and
approved binding; run before hooks before durable tool.started; consume the
existing single-use start/execution receipt only after allow. Run after hooks once
for an attempted execution, never for a veto. Persist ordered hook receipts with
the original canonical result. Release store/session/native locks before IPC.
W6 must enforce exactly-once hook dispatch and receipt persistence. Host APIs by
themselves do not supply durable authority or duplicate-invocation fencing.

Mac policy: explicit deny-default sandbox-exec profile, only daemon-selected
canonical file resources, no network grant. Never silently fall back to trusted.
Trusted is explicit and offers no sandbox claim. Linux does not implement the
Mac policy and rejects it. Runtime reads must be explicitly declared; the Mac
positive/negative test is mandatory, not ignored/skipped on a Mac runner.

On macOS the host needs the `alloy-plugin-pipe-helper` executable. Production
packages install it next to the daemon. For a source build, use
`cargo build --locked -p alloyd -p harness-core`; library embedders may instead
provide its absolute path to `PluginHost::start_with_helper`. A missing or invalid
helper fails registration. The helper creates pipe endpoints in a single-threaded
process and transfers them through a private Mach port; the host checks the
kernel-supplied sender identity and imports close-on-exec descriptors.

The operator may keep a user-owned, writable installation. That model trusts the
owner and unsandboxed processes running with the same privileges not to change
the plugin, helper, runtime resources, or paths while the host uses them; the
sandbox does not defend against such host-side changes. Less-trusted plugins
must not receive write access to executable or runtime resources. Registration
rejects writable trees containing a configured executable or helper, and rejects
existing hard-linked regular files in those trees (up to 16,384 entries); the
profile denies creating new hard links. This is a point-in-time check, not inode
pinning or a defense against concurrent changes by trusted host processes.

See `w5-security-review.md` for the tightened V1 restrictions and review record.
Remaining deployment/security assumptions must be confirmed by the reviewer: Seatbelt platform/runtime compatibility; profile
rights minimization; executable identity/TOCTOU; descriptor inheritance; process
escape via new sessions/groups; duplicate annotation-key handling and strict JSON
edge cases; hostile resource exhaustion. Trusted mode does not contain malicious descendants. Sandboxed V1 denies fork
and restricts exec to the configured executable; tests exercise that boundary. Do not advertise containment of untrusted code or
close P2-11 until these are reviewed and the necessary hardening is implemented.

Tests launch an actual compiled Rust fixture child. They demonstrate host IPC and
caller-side zero-dispatch after veto, NOT a deployed tool-loop or W3 integration.
Follow-up tests independently check ESRCH/ECHILD for child cleanup and run
allowed/denied I/O, network and fork probes inside the actual sandboxed host.
No real-model or full Phase 2 completion is claimed by this patch.
