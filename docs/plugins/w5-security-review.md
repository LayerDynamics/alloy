# W5 security boundary and review record

Implementation review checklist; this file is NOT a claim of independent approval.

## V1 restrictions

Less-trusted macOS plugins use deny-default Seatbelt with no network grant.
The profile denies process-fork, preventing creation of a descendant that could
escape the host-owned group. The actual host command additionally denies exec
outside its canonical configured program. V1 sandboxed plugins therefore cannot
use helper processes. Trusted local plugins remain explicitly trusted; this is
not a Linux sandbox or containment guarantee for malicious trusted code.

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

Descriptor sanitization is independent of allocation limits. Linux uses
close_range(3, UINT_MAX, CLOSE_RANGE_CLOEXEC) in the child; failure rejects spawn,
with no incomplete limit-based fallback. This requires the supported Linux image
and a kernel/libc providing close_range with CLOEXEC support. Rust's exec-error
pipe remains usable until exec. macOS uses POSIX_SPAWN_CLOEXEC_DEFAULT with explicit
stdio duplication, source-end closes, working directory, and a private group.
Only stdin/stdout/stderr are deliberately inherited; no environment is inherited.
An isolated real-host regression retains FD >=128, lowers NOFILE to 64, and verifies
that the executed plugin cannot observe the descriptor on either platform.

The owner keeps a direct child handle unreaped until stop, sends SIGKILL to its
private process group and direct child, retries interrupted wait, and reports
cleanup errors in bounded retained diagnostics. Tests independently observe ESRCH
and ECHILD for exact child PIDs after timeout, cancellation, invalid responses,
failed registration, and drop. This is not an OS OOM prevention mechanism.

Independent review found that an unrelated process-wide child reaper can consume
an owned child before cleanup; its PID can then be reused before a SIGKILL. This
applies to helper and plugin cleanup. The tests assume exclusive reaping and do
not establish safety with a competing reaper or SIGCHLD auto-reaping. Acceptance
is blocked until ownership is enforced or an identity-safe cleanup mechanism is
implemented and tested. Do not embed this host in a process with competing child
reapers and treat the warning as a complete fix.

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
rejects a write tree containing any configured executable or the helper, scans
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
- [ ] Candidate/run/artifact identities and logs are linked on #32/PR #46.
- [ ] Reviewer records name, date, candidate SHA, and explicit acceptance decision.

A shell script or passing test must not check off reviewer approval automatically.
The earlier Mac run proved the prior candidate's canonical cat-file fixture only;
it does not establish any of the new host/process hardening assertions.

Implementation ceilings: 64 configured plugins, 8 MiB per message, 1 MiB retained
diagnostics per plugin; per-plugin message/annotation/reason limits may tighten
these bounds. These bounds do not claim whole-process RSS or kernel-memory caps.
The actual Mac launch rejects a configured executable inside a declared writable
tree. The operator must also prevent untrusted replacement of its parent paths.
