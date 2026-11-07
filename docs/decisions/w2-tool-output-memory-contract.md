# W2 tool-output storage accounting: implementation decision pending

Status: design constraints only. This document is not an implementation or a W2 sign-off.
Baseline: afe9bfa48e254e2c29e3306aff9708e661ba628d.

## Existing defect

The draft proposal charge estimates serde_json::Value/map retention with a fixed
per-entry allowance. Qwen output admission currently includes the text queue,
one pending text delta, and String headers. It does not establish a complete
parser/JSON/held-proposal/queue allocation bound. The rendering patch does not
repair that defect.

## Required representation change

Do not replace the estimate with another multiplier. A defensible exact contract
must define what it counts: requested host allocation layouts and owned payload
storage, not allocator rounding, RSS, GPU working sets, or OS OOM prevention.
Allocator overhead stays a separately calibrated allowance.

The preferred implementation is a pre-admitted generation-owned byte/node arena:

- Fixed parser pending-call bytes plus bounded delimiter/UTF-8 carry.
- Fixed-size, bounded JSON validation stack and node/duplicate-key storage.
- Canonical validated proposal bytes and fixed descriptor slots in that arena.
- Bounded text and descriptor queues with known element layouts.
- Explicit lifetime transfer of arena ownership through worker replies, held
  proposals, queued events, and consumer-owned proposal handles.
- No ordinary serde_json::Map/Value construction before the corresponding
  allocation is covered. Decoding into Value at a consumer is a separately
  budgeted ownership transition, not a hidden part of dequeueing.

Reserve the checked sum of simultaneously live layouts before allocation.
Reject arithmetic overflow, node/depth/key/byte/call limits before committing
storage. A closing envelope, validation scratch, conversion result and old/new
buffers may overlap; account for that peak rather than only final payload size.

This changes the current Map-based proposal API and parser implementation and
requires coordinated decoder/RPC/registry consumers. It must not be presented as
a one-line change to proposal_charge.

## Required tests before marking accounting complete

Exact fit and one-byte/one-node-over refusal; oversized String/Vec capacity;
deep/nested JSON; duplicate keys; fragmented delimiters; multiple calls; parser
failure after a prior call; publication backpressure; cancellation; consumer drop;
failed allocation and unwind. For every case compare the live ownership ledger
with actual requested layout sizes, including scratch and conversion overlap,
and verify every owner releases its charge exactly once.

## Cache/render work is independent

The companion rendering patch binds schemas to render hashes, retained RAM
compatibility, retained metadata, and SSD cache identity. It deliberately leaves
Appendix B.1 canonical session-state framing unchanged. A canonical state ID is
not by itself proof of runtime cache compatibility.
