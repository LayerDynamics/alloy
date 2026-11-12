# W5 security boundary and review record

Implementation review checklist; this file is NOT a claim of independent approval.

The process-group probes and lifecycle limitations below record the original W5
implementation. The current native Mac broker and descendant-lifetime backend
supersedes those Mac trusted-descendant/supervisor limits; see
[native ownership and its qualification boundary](macos-broker-recovery.md).
Historical negative evidence is retained here to explain the change.

## V1 restrictions

Less-trusted macOS plugins use deny-default Seatbelt with no network grant.
The profile denies process-fork, preventing creation of a descendant that could
escape the host-owned group. The actual host command additionally denies exec
outside its canonical configured program. V1 sandboxed plugins therefore cannot
use helper processes. Trusted local plugins remain explicitly trusted; this is
not a Linux sandbox or containment guarantee for malicious trusted code.

The fork restriction does not confine the original plugin to its initial
process group. A native macOS probe using the W5-style deny-default profile
observed a sandboxed group leader join its parent's group with `setpgid` and
then create a new session with `setsid`. Both test children were reaped.
The host now launches macOS plugins as session leaders using
`POSIX_SPAWN_SETSID`, and the real sandboxed-host test attempts the same
operations. Candidate-bound tests and independent review are still required.
This does not by itself solve competing reapers or trusted descendants.
Trusted macOS plugins also start in a new session without a controlling
terminal; plugins must use their configured standard streams rather than rely
on the daemon's terminal or job-control session.

Only declared existing file resources are granted content access. Exact ancestor
metadata rules cover traversal of original and canonical path spellings, without
blanket metadata access. Startup grants remain the root-directory data probe and
the three named sysctls observed on the tested Mac: `security.mac.lockdown_mode_state`, `kern.bootargs`, and `hw.pagesize_compat`. Do not add unrelated grants to
silence log noise. Read/write trees must actually be directories.

The executable/cwd are canonicalized before launch, and the executable must be a
regular file. The operator selected a user-owned writable installation. The owner
and unsandboxed same-privilege processes are trusted not to modify the helper,
plugin, runtime files, their aliases or parent paths while the host uses them.
Canonicalization does not pin an inode across replacement, and this module does
not provide supply-chain verification or protection against adversarial changes
by those trusted host processes. Do not grant plugins write access to their code
or runtime resources.

Descriptor sanitization is independent of allocation limits. Linux's isolated
supervisor uses close_range to exclude inherited non-stdio descriptors from the
plugin; failure rejects spawn, with no incomplete limit-based fallback. Linux
also requires atomic CLONE_PIDFD, waitid(P_PIDFD), and readable /proc process
group metadata as described in tool-hooks-v1.md. macOS uses POSIX_SPAWN_CLOEXEC_DEFAULT with explicit
stdio duplication, source-end closes, working directory, and a private group.
Only stdin/stdout/stderr are deliberately inherited; no environment is inherited.
An isolated real-host regression retains FD >=128, lowers NOFILE to 64, and verifies
that the executed plugin cannot observe the descriptor on either platform.

On Linux the host owns a kernel-created pidfd for the supervisor and communicates
through a private control channel. The supervisor owns its target and keeps the
target unreaped while signaling and checking its group. Concurrent daemon reapers
or SIGCHLD auto-reaping cannot cause the host to signal a recycled numeric PID.
Cleanup failures and deferred-waiter exhaustion are reported as unestablished;
they do not authorize numeric fallback. The lifecycle and exceptional resource
limits in tool-hooks-v1.md are part of this supported contract.

On macOS, a dedicated `alloy-plugin-exec-helper` supervisor owns each helper or
plugin as its direct child. It resets SIGCHLD to its default disposition, starts
the target suspended in a new session, publishes its stable kernel identity,
and resumes it only after the host commits the launch. Configured code never
runs inside the supervisor. A daemon-wide `waitpid(-1)` reaper therefore cannot
reap the target. The supervisor observes exit with `waitid(WNOWAIT)`, kills the
private process group and target while the unreaped leader pins their numeric
IDs, verifies that no live group members remain, then reaps the target. Trusted
plugins may launch subprocesses; cleanup covers descendants remaining in that
session's original process group. Trusted descendants that deliberately detach
are outside this containment contract.

The host sends cancellation through shared atomic state, not a numeric PID.
Cancellation is monotonic and survives host unmapping. The supervisor also
checks the host's stable identity and cleans up when the host disappears.
Fallback direct-target signals compare `PROC_PIDUNIQIDENTIFIERINFO` uniqueid and
use `proc_signal_with_audittoken`; the kernel validates PID plus exec version.
Exec-version changes are retried only while the stable uniqueid matches. There
is no daemon raw PID/group signal fallback. These Darwin interfaces are checked
at runtime; identity lookup failure rejects launch. The native regression runs
an active competing reaper, SIGCHLD auto-reaping, exec transitions, stale
identity signals against a live sentinel, and trusted descendant cleanup after
both stop and natural leader exit. It models stale identity deterministically;
it does not claim to force actual kernel PID wraparound.

Cleanup waits are bounded. A caller timeout reports cleanup as unestablished
and leaves the supervisor alive to finish; it does not destroy the process
that owns group cleanup. This is not an OS OOM prevention mechanism.

Remaining lifecycle limits must not be represented as accepted W5 guarantees:

- A bootstrap wedged before publishing its identity cannot be safely signaled
  by PID. Cancellation prevents it from subsequently starting configured code.
- Unexpected supervisor death before target identity publication can leave a
  suspended target. After publication, the host can signal the exact target,
  but this does not establish cleanup of arbitrary trusted descendants.
Host-owned mode exclusively owns supervisor exit statuses. Embedders with a
process-wide child reaper or SIGCHLD auto-reaping must use
`PluginHost::start_with_external_reaper`: this mode never waits for supervisor
PIDs and leaves those statuses to the registered reaper. Plugin completion and
errors still come from shared memory. This explicit mode closes the separate
numeric-wait/PID-reuse exit-status race; mixing an uncoordinated reaper with
host-owned mode is unsupported for the pipe-allocation helper. In the current
native Mac integration test the external reaper collects exactly that one host
child status. The trusted workload supervisor is owned by launchd, so counting
it as a second embedding-host child would contradict the current ownership model.


Research references: Apple's [process identity and audited signal implementation](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/proc_info.c),
[identity ABI](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/sys/proc_info_private.h),
[signal/group behavior](https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_sig.c),
and [libproc return conventions](https://github.com/apple-oss-distributions/xnu/blob/main/libsyscall/wrappers/libproc/libproc.c).
The tested host is macOS 26.4 (25E246); this is not qualification of every macOS release.

## Protocol boundary

Bounded raw lines are decoded through duplicate-key-rejecting recursive JSON
validation before typed deserialization. Escaped-equivalent duplicates and
nested annotation duplicates are rejected. Typed envelopes reject unknown fields.
Request encoding borrows the caller's immutable inputs rather than building a
second Value tree. Configured message limits have explicit implementation caps.
Before failure stops later before hooks; after failure is warning-only and cannot
replace the original result or replay an executor. Stopped processes never retry.

W6 derives/persists authority and invokes hooks once. This module supplies no
executable authority; constructing a snapshot cannot approve or start a tool.
The W6 integration must not hold store/session/native locks across hook IPC.

## Evidence and approval

Independent review of candidate 253c24c found that macOS pipe creation used
`pipe` followed by `fcntl(F_SETFD)`, exposing endpoints to unrelated concurrent
exec. The current implementation creates them in a separate single-threaded
helper, transfers them over a private Mach port with a kernel audit-token sender
check, and imports close-on-exec file descriptors. The isolated concurrency test
includes an intentionally inheritable positive control. The changed transport
still requires candidate-bound CI and an independent review decision.

Executable pathname checks do not pin an inode against replacement. Registration
rejects a write tree containing any configured executable or either helper, scans
write trees for existing hard-linked regular files, and rejects trees exceeding
16,384 entries. Seatbelt denies creation of new hard links. This conservative
scan is point-in-time and cannot stop a trusted same-UID host process from racing
it; the operator remains responsible for runtime resources, mount aliases and
the selected trust boundary. Clearing owner write permission is insufficient
against the owner, who can restore it. Native tests now
exercise fork, direct posix_spawn and the standard library's process-spawn path;
they do not prove that every possible process-creation interface is denied.
Another native test gives a sandboxed plugin the real path of a published
artifact outside its grants, observes a denied write and rejected result
replacement, and verifies the original bytes through the artifact store. This
does not protect an artifact tree the operator has explicitly granted writable.

- [ ] P2-10 and protocol P2-11 pass on Linux for the submitted candidate.
- [ ] P2-10, protocol P2-11, and native sandbox tests pass on macOS.
- [ ] Actual sandboxed host proves allowed read, denied undeclared read/write,
      denied loopback connection, denied fork, and independent child reaping.
- [ ] Reviewer confirms executable deployment assumptions and grants are adequate
      for the supported plugin configuration; no claim of arbitrary-code isolation.
- [ ] Candidate/run/artifact identities and logs are linked on #32/PR #100.
- [ ] Reviewer records name, date, candidate SHA, and explicit acceptance decision.

A shell script or passing test must not check off reviewer approval automatically.
The earlier Mac run proved the prior candidate's canonical cat-file fixture only;
it does not establish any of the new host/process hardening assertions.

Implementation ceilings: 64 configured plugins, 8 MiB per message, 1 MiB retained
diagnostics per plugin; per-plugin message/annotation/reason limits may tighten
these bounds. These bounds do not claim whole-process RSS or kernel-memory caps.
The actual Mac launch rejects a configured executable inside a declared writable
tree. The operator must also prevent untrusted replacement of its parent paths.
