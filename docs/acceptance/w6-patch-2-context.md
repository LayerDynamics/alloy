# W6 patch 2 — canonical tool-context projection

## Candidate and scope

Inspected base: `1aa0de5a65d61fb043e88891a77ca4dfd3eddbb2` on
`feat/33-canonical-tool-loop` (W6 patch 1). This is partial W6, not P2-12,
P2-13, P2-14, W7 real-model acceptance or Phase 2 sign-off. Keep #33 open.

## Recorded file-placement decision

Implement the context projection as private child module
`crates/harness-core/src/context/tool_history.rs`, declared by `context.rs`.
It extends the existing public `ContextBuilder` with `build_tool_snapshot`;
there is no duplicate registry, dispatcher, root crate or new dependency.
`tests/canonical_tool_context.rs` is an auto-discovered harness-core integration target.
The manifest has no `autotests = false`; exact discovery is checked by the runner.

## Implemented boundary

- Copy exact model-visible ID/version/description/input schema from validated
  `ToolRegistry::definitions`; retain the engine's canonical schema fingerprint.
- Read complete, gapless, session-bound `SessionStore::list_events` history.
  The caller must source this history from its own store, not from client input.
- Keep text before the first request in the base messages and all calls,
  results and later assistant/user text in a single ordered historical tail.
  W3's additional `tool.requested` Claimed transition does not create another call.
- Preserve historical argument/result content; do not resolve old arguments
  against a possibly changed current schema or execute a historical call.
- Reject orphan/duplicate result links, mixed session identity, unsupported
  contract/state, malformed result envelopes and unresolved calls. A pending
  approval or execution is not permission to run another generation.
- Project inline JSON without changing its value category or is_error flag.
  Pass artifact envelopes as opaque reference descriptions, without file access.
  W3/W4 remain responsible for committed artifact provenance and byte verification.
- Convert W3 terminal markers into canonical structured ToolError content:
  Rejected/Denied/Expired are policy errors; Withdrawn/OutcomeUnknown are execution
  errors. Preserve the complete original marker under details, including the
  unknown-outcome flag. This changes no stored or RPC payload contract.
- Enforce explicit tool count/history-block/serialized metadata limits, retaining
  existing renderer/tokenizer/input-message/token/admission checks. This is not
  a hard bound on transient serde/template allocations. No context limit is raised.
- Make the legacy text builder reject tool lifecycle events rather than silently
  omitting them. Pure text-only snapshots keep their existing behavior.

## Integration status and next boundary

The deployed supervisor remains text-only. It does NOT call build_tool_snapshot
or dispatch tools in this patch; it now fails closed if handed durable tool
history. This patch prepares the model-context side for the tool-aware loop.

Still pending: generation proposal collection/finalization and bounded rounds,
daemon call IDs, W3 admission/approval coordination, W5 receipt persistence,
W4 receipt-owned dispatch, one-terminal user-turn handling, responsive approval
RPC and actual daemon configuration, CLI controls, integrated P2-12/13/14,
Linux/Mac candidate evidence, and W7 acceptance. Do not mark the whole canonical
context checklist item complete before the real loop uses it on every round.

## Test evidence boundaries

Seven named tests cover exact ordered projection and independent HF re-encoding;
schema/version identity changes despite equal template/token output; fifteen
corrupt-history vectors; pending-call rejection and real W3 restart-to-unknown;
denial/error/reference presentation; context/render limits and reserved syntax;
and unchanged legacy text rendering. A counter asserts reconstruction does not
execute the registered fixture implementation.

The tests use real SQLite/W3 transitions and the Rust template renderer/HF
tokenizer, but supply fixture results at the store-completion boundary. No actual
builtin execution or trained model is claimed. The artifact-envelope read-side
case is explicitly synthetic; it is NOT artifact durability/provenance acceptance.

Required local checks: exact discovery and execution of all seven tests; the
existing engine tool_rendering and W6 patch-1 boundary tests; locked workspace
all-target check/test/Clippy, doc tests, strict rustdoc, formatting and diff checks.
The runner records actual source-file hashes, base, platform and tool versions.
Mac local results are not Linux/Mac GitHub CI evidence. A feature-branch push
alone is not a CI run. Record final committed candidate/PR evidence separately.
