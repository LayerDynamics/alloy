# Tiered native KV storage (M10)

The engine can persist actual Qwen3 KV tensors to SSD and import a verified
prefix into a newly loaded model instance. SQLite session recovery and KV
recovery are independent: SQLite restores conversation events; the SSD tier
avoids repeating some transformer prefill. Missing, incompatible, or corrupt
disk snapshots fall back to prefill from the recovered conversation. A native
backend failure propagates through the ordinary request cleanup path.

## Configuration

`alloyd` enables the SSD tier by default. Its directory is `kv-cache` beside the
selected session database, normally `~/.alloy/kv-cache`. It keeps unleased idle
native KV within 25% of the current shared admission ceiling.

| Option | Behavior |
| --- | --- |
| `--paged-ssd-cache-dir PATH` | Use this directory for immutable cold snapshots. |
| `--hot-cache-max-size SIZE` | Cap retained native KV reservations. Accept bytes, `KiB`, `MiB`, `GiB`, or an integer percentage from `0%` through `100%`. The default is `25%`. |
| `--hot-cache-only` | Disable SSD reads and writes; keep the bounded RAM cache. |
| `--hot-cache-max-size 0` | Persist completed state, then release unleased idle native KV while retaining eligible model weights. Useful for restart acceptance. |

The engine API exposes `TieredKvConfig` for embedded runtimes. A runtime that
does not call `configure_tiered_kv` retains its previous retention policy.
The default SSD capacity is 4 GiB and the catalog contains at most 64 complete
snapshots; the oldest published snapshots are removed when room is needed.
These defaults are configurable through the engine API.

## Native ownership and admission

All export, import, persistence, and eviction runs on the owning native worker.
No monitoring call acquires native arrays or waits on its command queue.

The SSD owner admits a 64 MiB transfer allowance before serving requests and
holds it until that owner is destroyed. It covers one 32-token block, host
copies, serialization, and bounded metadata. Its bound is four times the
block's float32 payload plus 8 MiB. The existing KV reservation separately
covers native resident buffers and their transactional update overlap. A
layout that cannot fit the transfer bound produces a diagnostic cache miss;
it does not allocate an unadmitted larger transfer.

Keeping this allowance admitted lets the worker spill while admissions are
paused. `transfer_reserved_bytes` therefore remains nonzero between requests;
it is an owned allowance, not a signal that an I/O operation is still running.
The admission ledger includes it throughout the daemon's lifetime.

The hot cap uses actual owned cache reservations, including the conservative
native update allowance. Active execution, pending startup, and leased
canonical publications remain protected. They may temporarily exceed the
idle cap; their reservations remain visible. A model residency pin retains
weights but does not pin an unrelated, unleased KV prefix. Eviction removes
native arrays before releasing their guards. Pressure reclamation tries idle
KV before unloading model weights and reobserves pressure before the next
destructive step or reopening admission.

Emergency cancellation continues through the scheduler's exact-request
control path. A transfer is observed at supported block/native boundaries.
Cancellation does not free tensors beneath a native operation, release a
pending grant early, or permit a replacement request to acquire the slot
before cleanup.

## Storage and compatibility

Each complete snapshot contains a bounded `manifest.json` and ordered
`00000000.safetensors`, `00000001.safetensors`, … blocks. Every block stores
keys and values for each layer with layout
`[batch, kv_heads, block_tokens, head_dim]`. Source tensors are evaluated and
read back by MLX; the snapshot does not contain simulated tensors or only
token IDs. The final block may be shorter than 32 tokens.

The compatibility digest binds the verified artifact content manifest,
artifact and profile identities, tokenizer, template, renderer, family
implementation version, engine API version, and cache representation. This
private storage digest is distinct from the unchanged public B.1 BLAKE3
session-state identity. It excludes process-local model identity and artifact
directory names, allowing restore into a new verified instance. That instance
still owns and validates every imported native array.

The manifest records the executed token prefix, layer layout, ordered block
positions, byte lengths, and SHA-256 checksums. Each safetensors header also
binds its compatibility digest and exact token-prefix hash. The reader checks
the full selected block before native import, including dtype, dimensions,
layer count, bounds, finite values, and checksums. A cropped final block uses
the correct per-batch/per-head strides.

The cold directory has one storage owner, enforced by an OS file lock. A
second live owner fails configuration explicitly. After a crash releases the
lock, the next owner removes abandoned temporary writes before scanning.
The writer uses a private temporary directory and mode-0600 files, syncs all
blocks and the manifest, then atomically renames the directory into the
published namespace. Startup ignores incomplete temporary directories.
Published files are immutable. The implementation does not share mutable
native storage between snapshots or processes.

Restore finds the longest compatible exact prefix and imports blocks
transactionally. It leaves at least one input token for real prefill so that
final-position logits are newly computed. It never fabricates or persists
logits as a substitute for transformer work. A corrupt later block resets
the partially imported cache before full prefill. An invalidated native cache
is discarded through the ordinary failure cleanup path. I/O, quota, and
compatibility failures are optional-cache failures, reported in metrics.

This first implementation supports full float32 KV. It does not implement
multi-session page sharing, rotating/quantized KV persistence, compression,
background SSD workers, or cross-process ownership. Those extensions must
preserve the same admission and identity rules.

## Evidence and observability

`daemon.metrics.kv_cache` exposes `enabled`, `disk_enabled`, effective hot cap,
hot/protected/transfer reservation bytes, cold entry count, successful spill
and restore counts, actual restored token count, hot evictions, and the last
cache error. Values are published at native-worker boundaries; they are not
an instruction to release memory.

Focused tests use real MLX arrays and Qwen3 execution to verify native block
contents, continuation logits after restart, partial-block cropping,
corruption rollback, identity isolation, hot-only behavior, lease protection,
and cancellation with cleanup ownership. On Apple Silicon the runtime tests
use the production unified CPU/Metal path; native block tests additionally
exercise both devices and cross-device transfer.

The real-model restart driver forces a zero hot cap, verifies a completed SSD
spill, terminates its own daemon, starts the binary built from the same
checkout, checks recovered SQLite events, and requires positive native KV
restore counters during resumed generation. It retains raw stream records,
metrics, process logs, commit/binary provenance, and artifact identities.
See [the exit review](../acceptance/phase-1-exit-review.md) for recorded runs;
tooling or fixture success alone is not real-model acceptance.
