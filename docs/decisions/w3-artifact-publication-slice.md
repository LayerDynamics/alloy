# W3 bounded artifact storage slice

Baseline: 2e1d34de416135d1a72d143cabbe6432348144de. Contract: alloy.w3/1.
This is an implementation candidate, not an assertion of passing Linux/Mac evidence.

## Ownership and initialization

Initialize ArtifactStore::open(&mut store, explicit_root, limits) before placing
SessionStore in live daemon/session mutexes. This performs startup reconciliation
under exclusive daemon ownership but without an open SQLite transaction across
file reads, hashing, deletion or directory synchronization. The explicit root's
parent must already exist. The daemon owns that parent/root. Managed directories
and entries reject symlinks; this is not protection against malicious same-user
concurrent replacement of trusted directories. Supported target is local Unix
storage (Linux/macOS), not network filesystems or Windows.

SessionStore holds the existing OS sidecar guard in Arc. ArtifactStore clones,
writers, pending publications, and lookup/verified read handles retain a lease.
Dropping SessionStore cannot release daemon ownership until these values drop.
There is no replacement lock design and no per-process global mutex.

A synchronized root store-ID marker and a synchronized database-adjacent root
binding pin one canonical root to one database identity. Partial/mismatched marker
initialization fails closed; it is not silently repaired. No schema/DDL version
change is introduced: this uses the previously reserved tool_artifacts table with
the frozen metadata fields. Changing configured root is intentionally rejected.
Binding/marker files and directories are fixed control metadata outside payload
file/byte accounting, as are SQLite/WAL and ownership sidecars.

## Lock-free streaming handoff

1. Under a short store access, begin validates a current Started invocation and
   reserves payload, file entries and one of at most eight writers.
2. Release store/session locks. Stream through ArtifactWriter using a 64 KiB
   copy buffer. Any input/write error permanently invalidates that writer.
3. Still outside store/session locks, finish synchronizes the complete private
   temporary file, publishes using a no-overwrite hard link, synchronizes the
   objects directory, removes the temporary name and synchronizes tmp.
4. Reacquire the store only for PendingArtifact::commit. Revalidate current
   Started session/call/epoch, then insert metadata and artifact.created in one
   short SQLite transaction. Return a reference only on successful commit.

The temporary and object directories are siblings beneath the configured root.
Two names briefly exist during hard-link publication. Admission conservatively
reserves two file entries and twice the requested capped payload bytes, then
settles to one entry and actual length on commit. Actual bytes are not additionally
charged on top of those reservations. This can reject admission near quota even
when the final single-name object would fit; the hard upper bounds are unchanged.
Per-object bound is min(requested maximum, bound invocation output limit,
configured object limit). Overflow is an error, never a truncated valid artifact.
No referenced artifact is evicted to make room.

Metadata retains the exact frozen fields, with preview_bytes holding either UTF8
text or base64 according to preview_encoding. At most 4096 source bytes are kept;
only an incomplete trailing UTF8 codepoint may be trimmed. Binary previews are
at most 5464 base64 characters. The SHA-256 and length cover the full accepted data.
Random references use 128 bits from /dev/urandom. Collision publication fails
without overwriting or returning a reference; callers may make a new attempt.

## Lookup and reconciliation

lookup performs only ownership/metadata checks. Release the store mutex before
UnverifiedArtifact::verify, which reads/hashes the complete object in fixed-size
chunks and exposes VerifiedArtifact only after full validation. Objects are
immutable within the trusted daemon-owned root; no defense against malicious
same-user modification after verification is claimed. Reads require BOTH the
requesting session and originating call association, not reference possession.

Reopen inventory is bounded and validates all managed filenames/types/identities
and all committed object lengths/digests before deleting any leftover. Unknown
entries, external aliases, missing/corrupt committed objects, or unavailable
metadata proof fail closed with no cleanup deletion. A proven temporary/object
hard-link pair is recognized so a process exit between link and unlink can be
recovered. Proven unreferenced managed objects and temporary files are removed;
the affected directories are synchronized before writer admission.

A failure after publication poisons the live artifact state and retains conservative
reservations. It never blindly deletes the published object: the metadata commit
may have succeeded. Drop all leases and reopen to consult committed state, retain
registered objects and remove proven orphans. This is conservative failure handling,
not automatic effect retry or atomicity across SQLite and the filesystem.

## Tests and limits of evidence

The integration suite adds P2-08 oversized_tool_result_artifact_roundtrip through
the real policy/claim/start coordinator, plus empty/exact/overflow payloads,
16 MiB acceptance, preview encodings, ownership/ref checks, integrity failures,
quotas, failed input, SQLite event rollback, retained ownership leases, cleanup
proofs, root binding and symlink rejection.

Unit tests cover no-overwrite collision, eight concurrent streamers, bounded copy
buffers, early/late failure injection and a fresh subprocess exiting without Drop
at six publication/commit boundaries. The after-commit case models an uncertain
caller result: the reference is not returned but committed bytes must survive.
These subprocess tests exercise process exits on the running filesystem, not
power-loss or arbitrary hardware guarantees. Run on both Linux and macOS.

## Boundaries

No changes to tool-result wire variants/error categories, W2 slab/parser ownership,
plugins, builtin execution, live RPC/CLI, or the canonical tool loop. This slice
registers artifacts. Step B now connects committed artifact-backed ToolResult to
complete_tool; see w3-artifact-completion-slice.md for provenance checks and tests.
Existing contract JSON status is not rewritten to claim acceptance.
