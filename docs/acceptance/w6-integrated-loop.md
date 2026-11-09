# W6 integrated supervisor, control RPC and CLI

This resumes issue #33 from branch `feat/33-canonical-tool-loop`, base `588e5756f42d0b4cc9aa400def96b3a2b7ca8550`. The controlling scope remains [the W6 checklist](w6-checklist.md) and [the Phase 2 execution plan](../specs/PHASE-2-EXECUTION-PLAN.md). W7 trained-model acceptance is separate.

## Implementation

The configured daemon calls `Supervisor::run_tool_turn`, which renders exact canonical tool context for each bounded generation, admits proposals through the existing registry/W3 policy, waits for exact bound approval without holding store/session/native locks, claims once, calls ordered W5 before hooks, commits W3 start, runs the unchanged W4 builtin adapter, persists its original result/artifact, calls after hooks, and regenerates from durable linked history. Text-only startup remains available without a tool configuration. See [configuration and ordinary CLI use](../tools/configuration.md).

Both legacy and owned slab proposals remain untrusted until successful engine completion. Call IDs bind the durable turn, generation and ordinal. Registry/schema rejection is recorded as linked protocol history with no executable intent. Denial/veto/cancellation preserve the W3 terminal state and do not create a start. External execution retains only its receipt and cancellation ownership. An operation watchdog enforces the turn deadline in addition to the bound W4 limits.

The service reserves one turn slot per session. `session.approve` requires exactly three parameters and a real boolean. Duplicate JSON keys, including escaped duplicates, are rejected by the actual NDJSON decoder before authority is evaluated. Wrong-session, stale and duplicate decisions fail at the W3 store boundary. Pause targets the operation token and exact generation ID. Status, metrics and unrelated sessions use independent socket handlers.

The CLI uses a separate control socket and the real readline stream for decisions. It preserves queued chat messages without allowing queued messages to answer an approval. EOF/noninteractive input cancels. Disconnect/terminal settlement aborts an outstanding prompt. Startup forwards an explicit `--tools-config` argument literally and does not change a running daemon's configuration.

## Additive history and wire contracts

W6 adds `tool.requested {transition:"HookReceipts",call_id,receipts}` records carrying ordered plugin identity, version, invocation identity, hook, bounded annotations and diagnostic. These are audit data, not a fresh call or result. The context builder validates the call link and omits receipts from model context. The existing B.4 event vocabulary and schema-v2 database remain unchanged. Older tool-context readers reject this unknown transition rather than silently misrender it.

Canonical protocol rejections append `tool.requested` with `state:"ProtocolRejected"` and the complete daemon-bound request, then one linked `tool.completed` with a protocol result. They never produce an executable `tool_intents` entry or start receipt.

B.6 `tool_call_started`, `tool_result {call_id,is_error,summary}`, `approval_requested {call_id,tool_id,summary}`, and terminal events are emitted from the coordinator. Result/approval bodies also carry structured details for the CLI; `tool_requested` is an additive informational notification. Intermediate engine terminals are withheld. A successful terminal carries aggregate token usage and the final round's finish reason. After-hook audit receipts are appended after the original result is durably committed, so interruption between those operations may leave a committed result without after-hook receipts; restart never reruns the effect or invents missing receipts.

## Acceptance evidence

Current commands, test counts, hashes, platform identities, diagnostic failures and unresolved gates are retained under `work/w6-resume-20261108/` and the `work/w6-*.log` files. Reports distinguish earlier diagnostic runs from checks against the final manifest. The GitHub issue stays open pending the final candidate/review relationship; this document does not assert merge or trained-model operation.

The focused tests are `canonical_tool_loop_persists_results`, `approval_rpc_remains_responsive`, and `cli_renders_approval_and_tool_result`. Additional integration cases exercise a real approved write, zero-effect denial/invalid calls/call limits, actual plugin veto/annotation, cancelled approval, artifact provenance, and a subsequent turn after reopening without replay. The real CLI/socket case exercises yes, no and EOF through a separate control connection.

## Final local review corrections

The local review directly corrected four integration boundaries: shared plugin-host lock waits now observe cancellation/deadlines, total turn text bytes are checked before forwarding each delta across rounds, approval/result previews keep identity and outcome visible independently of large argument/hook data and escape C1/bidi terminal controls, and configured tool notifications enforce their escaped-envelope byte ceiling before any write. Tool-mode socket writes receive bounded defaults when no explicit transport policy is supplied.

After-hook host contention records ordered identity-bound failure receipts without replacing the already persisted execution result or blocking on another session beyond cancellation/deadline. New regression tests exercise both host-wait stages, zero effects before start, preserved actual results after start, UTF-8 byte boundaries, oversized approval arguments/hook annotations and exact/over-limit notification frames. Review is local self-review; no independent review is claimed.
