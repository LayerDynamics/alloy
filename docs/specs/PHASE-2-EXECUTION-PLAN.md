# Phase 2 execution plan — tools and policy

**Status:** Requirements review completed 2026-11-06 against the current SPEC-1 Phase 2 scope and exit criteria. Phase 1 was accepted at merged main `a4a17df` in [issue #9](https://github.com/LayerDynamics/alloy/issues/9); its entry gate is satisfied. This document is the implementation plan, not a claim that Phase 2 code exists. The first [Homebrew source installation](../designs/homebrew.md) is a separate distribution slice before this tool-loop work.

**Source of scope:** [SPEC-1](SPEC-1-mlx-agent-harness.md.md), §4.1 **Phase 2: Tools and policy**, FR-20/21/24, §3.2 tool registry and plugin host, §3.4 builtin tool set, and Appendices B.2/B.4/B.6. FR-25, FR-26, FR-28, and FR-30a constrain the existing session, rendering, cache, and CLI interfaces when tools are added. The work below is an implementation decomposition of those requirements, not a new product roadmap.

## 1. Goal and entry gate

A supported Qwen3 model must request a builtin tool through a parsed, validated call; the daemon must apply policy and approval, execute the authorized call, persist its result, and feed the result back into the model. An out-of-process plugin must receive the two V1 tool hooks and be able to veto execution or annotate the result.

The [Phase 1 exit review](../acceptance/phase-1-exit-review.md) records the completed entry gate: successful native jobs, real-model streaming and restart evidence, memory-parity acceptance, and the required post-freeze reproducible builds. Its final tested source tree matches merged main `a4a17df`. New Phase 2 changes must retain these regression gates; an unavailable model or an unexecuted hardware gate is not a waiver.

Phase 2 retains the existing vendored mlx-c engine, pinned corpus identities, exact rendering/counting path, append-only SQLite history, local JSON-RPC transport, memory guard, and SSD KV behavior. Their regression gates continue to run on every candidate.

## 2. Scope and boundaries

| Source requirement | Phase 2 deliverable |
| --- | --- |
| FR-20; Phase 2 goal | Canonical content blocks and a bounded generate → parse → validate → authorize → execute → persist → generate loop. Raw model text never selects an executable action. |
| FR-21; B.2 | One registry definition/result contract with schema validation, origin, effect class, approval requirement, timeout, and output-size limit. Builtin, plugin, and future MCP registrations use the same contract. |
| §3.4 builtin set | `fs.read`, `fs.list`, `fs.write`, `shell.run`, and `http.fetch`, using the default effects/approvals listed below. |
| §2.2 security; B.6 | Daemon-owned read/write/destructive policy and approval state; `session.approve` and streamed tool/approval events. The CLI renders decisions and returns the user's answer. |
| Phase 2 scope | Cancellation while generating, awaiting approval, running a tool, or waiting for a hook; bounded output collection with a plain-file artifact store. |
| FR-24; §3.2 plugin host | Versioned out-of-process `tool.before_execute` and `tool.after_execute`, explicit ordering and mutation rules, supervised lifetime, and macOS restrictions for less-trusted plugins. |
| FR-25; B.4 | Durable tool intent, approval decision, execution outcome, and artifact references, with a forward migration if payload schemas change. |

MCP transports/discovery and skills remain Phase 4 work. Transactional compaction and the second family remain Phase 3 work. Automations, general plugin lifecycle hooks, speculative decoding, additional model families, TUI work, and benchmark expansion are not Phase 2 deliverables. Registry origin types must leave room for MCP without implementing its client now. Adding tool schemas/results to the existing exact context renderer is required for this loop; it does not pull the whole Phase 3 context project forward.

## 3. Contracts to freeze in W1

### Canonical calls, registry definitions, and results

Use the B.2 definition fields: `id`, semantic `version`, `description`, `inputSchema`, `origin`, and `policy { effect, approval, timeout_ms, max_output_bytes }`. Reject duplicate IDs and invalid schemas when registering. The registry holds the executable implementation; a generated name, origin, version, or path cannot create a registration.

A complete parsed call contains a daemon-scoped `call_id`, a registered `tool_id`, and one JSON argument object. Associate each call with its session, turn, and generation. The Qwen3 parser handles arbitrary streaming chunk boundaries and emits complete structured calls only after the model family's closing delimiter. Incomplete JSON, duplicate/ambiguous call boundaries, unknown tool names, invalid arguments, and malformed mixed text/call output produce a protocol result and zero executions. The parser's fixtures cover Qwen3 text and Coder syntax supported by the current family; adding another family is outside this phase.

A result uses `call_id`, either inline `content` or an `artifact_ref`, and `is_error`. Error kinds remain the SPEC's `protocol`, `execution`, `timeout`, and `policy`. Cancellation uses `execution` with a structured cancellation reason; a new wire error enum must not appear without a documented contract revision. Keep tool results as tool-result context blocks linked to their call IDs, not untrusted instructions inserted into the system role.

Pass the exact registered schemas and structured tool history through the existing Rust template renderer and HF tokenizer before every generation. Include the tool-schema identity in the context/cache identity. A changed schema or rendered prompt must prevent incompatible KV reuse. No hand-estimated token counts or artificial context-limit increase is acceptable.

### Builtins and default policy

| Tool | Default effect / approval | Bounded implementation contract |
| --- | --- | --- |
| `fs.read` | read / never | Policy-approved path; bounded byte range; text or artifact result. Resolve traversal and symlinks against the allowed roots. |
| `fs.list` | read / never | Policy-approved directory; deterministic ordering and bounded entry count/output. |
| `fs.write` | write / policy | Validated path and contents; policy-approved destination; result identifies the written artifact/path without widening allowed roots. |
| `shell.run` | destructive / always | Validated executable, argument array, and working directory; approval binds the exact invocation. Spawn without implicit shell interpolation. Capture stdout/stderr under a deadline, byte limit, and cancellation handle. |
| `http.fetch` | read / policy | Bounded GET/HEAD request, destination policy applied to redirects, deadline and response-size limit. No listener or credential-management subsystem. |

Read/write/destructive effects and the approval defaults above come from SPEC-1 §2.2 and §3.4. Detailed input schemas, path scope, and local policy configuration format are W1 implementation decisions. Freeze them in fixtures before wiring model output to execution. A plugin or model cannot lower an effect class, extend a path/network scope, or grant approval.

### Tool lifecycle and approval

1. Parse the complete model call, resolve its canonical registry entry, validate its arguments, and append `tool.requested` before an action can run.
2. Evaluate daemon policy. For a required approval, persist the pending intent and emit `approval_requested`. An approval is bound to the session, call ID, registered version, canonical arguments, and policy revision. Changed, expired, duplicate, wrong-session, or already-completed approvals cannot authorize a new action.
3. Accept `session.approve { session_id, call_id, allow }` on a control connection while generation/tool work is paused. Append the decision using the existing `tool.approved` kind with an explicit `allow` value. Denial produces a policy result with zero execution.
4. Invoke the configured before hooks on the approved, immutable invocation. A veto stops it before `tool.started`; neither the tool nor later before hooks run for that invocation.
5. Append `tool.started` and execute once, with a deadline, cancellation, and bounded output handling. Store oversized output as an artifact, then call the after hooks for an attempted execution. Append one `tool.completed` containing the original result and hook annotations; append/reference `artifact.created` when applicable.
6. Append canonical assistant/tool context blocks and re-render/count before the next generation. Continue until a final assistant response, cancellation, policy stop, error, or configured turn/tool-round limit. Emit one terminal outcome for the user turn. Model-generation completion inside a tool round does not falsely finish the user turn.

Never hold the SQLite/store lock, a session-map lock, or the native execution slot across a human approval wait, hook IPC, or external tool execution. Retain enough ownership to cancel the exact current operation. Pausing a session cancels the active generation/tool, withdraws its pending approval, and preserves its durable history. Historical recovery may show a tool that started without a recorded completion as having an unknown outcome; do not automatically rerun that effect. Full automation retry/idempotency semantics remain Phase 6 work.

### Minimal plugin interface, version 1

The plugin host is a supervised Rust component using bounded JSON-RPC messages over the child process's stdin/stdout. Each handshake declares the protocol version, plugin ID/version, and the supported subset of the two hooks. Stderr is retained as diagnostics. A version mismatch fails registration before any call is offered to that plugin.

| Hook | Immutable request payload | Allowed response |
| --- | --- | --- |
| `tool.before_execute` | Protocol version, invocation/session/call IDs; canonical tool ID/version/origin and effect; schema hash; validated arguments; bound approval decision and deadline. | `allow` or `veto`, bounded reason, and bounded namespaced annotations. |
| `tool.after_execute` | The same invocation identity, execution outcome/duration, and inline result or artifact reference. | Bounded namespaced annotations only. |

Hooks execute serially in the explicit configured plugin order. Before hooks run after argument validation and required approval, before any tool execution. After hooks run once for an attempted execution, including a tool error, timeout, or cancellation; a call vetoed before execution has no after-execute hook. Each receipt records plugin ID/version and invocation ID so order and duplicate responses are auditable.

No hook may change arguments, tool identity, policy, approval, artifact bytes, or the original tool result. Before-hook timeout, crash, malformed reply, or protocol mismatch fails closed as a policy result. After-hook failure becomes a recorded warning/receipt alongside the actual result; it never reruns the tool or converts an executed effect into a claimed non-execution. All plugin calls have message limits, deadlines, and cancellation; a broken child is terminated and reaped by its owner. Less-trusted plugin processes run under an explicit macOS sandbox profile limited to their declared local resources. The sandbox boundary and its positive/negative tests must be reviewed before plugin acceptance passes; merely spawning a process does not satisfy §3.2.

The full session/context/generation/compaction/automation hook surface remains deferred, as required by FR-24.

## 4. Exact work packages and dependencies

`new` means the file is planned and does not yet claim to exist. Each package lands with its named tests in §5. Revisit file placement through a recorded design change if inspection reveals a better boundary; do not silently expand the package.

| Work | Dependencies | Exact production/document files |
| --- | --- | --- |
| W1 — contracts and registry | Phase 1 exit sign-off | `Cargo.toml`, `Cargo.lock`; **new** `crates/tool-registry/Cargo.toml`, `crates/tool-registry/src/lib.rs`, `crates/tool-registry/src/definition.rs`, `crates/tool-registry/src/call.rs`, `crates/tool-registry/src/result.rs`, `crates/tool-registry/src/registry.rs`, `crates/tool-registry/src/validation.rs`; `docs/specs/PHASE-2-EXECUTION-PLAN.md`. |
| W2 — Qwen3 structured parsing and rendering | W1 | **new** `crates/families/qwen3/src/tool_calls.rs`, `crates/families/qwen3/fixture/tool_calls.json`; `crates/families/qwen3/src/lib.rs`, `crates/families/qwen3/src/template.rs`, `crates/families/qwen3/src/decode_impl.rs`, `crates/engine/src/family.rs`, `crates/engine/src/types.rs`, `crates/engine/src/generation.rs`, `crates/engine/src/rendering.rs`, `crates/engine/src/cache_identity.rs`. |
| W3 — policy, durable approvals, artifact store | W1 | `crates/harness-core/Cargo.toml`, `crates/harness-core/src/lib.rs`, `crates/harness-core/src/error.rs`, `crates/harness-core/src/store.rs`; **new** `crates/harness-core/src/policy.rs`, `crates/harness-core/src/approval.rs`, `crates/harness-core/src/artifact.rs`. Preserve older event rows and add a forward migration if B.4 payload/schema interpretation changes. |
| W4 — builtin execution and cancellation | W1, W3 | **new** `crates/tool-registry/src/execution.rs`, `crates/tool-registry/src/builtins/mod.rs`, `crates/tool-registry/src/builtins/fs.rs`, `crates/tool-registry/src/builtins/shell.rs`, `crates/tool-registry/src/builtins/http.rs`; extend W1 registry exports. |
| W5 — two-hook plugin host | W1, W3 | **new** `crates/harness-core/src/plugin_host/mod.rs`, `crates/harness-core/src/plugin_host/protocol.rs`, `crates/harness-core/src/plugin_host/process.rs`, `crates/harness-core/src/plugin_host/sandbox.rs`, `docs/plugins/tool-hooks-v1.md`; extend harness-core exports/dependencies. |
| W6 — canonical loop and local client | W2, W3, W4, W5 | `crates/harness-core/src/supervisor.rs`, `crates/harness-core/src/context.rs`, `crates/alloyd/src/service.rs`, `crates/alloyd/src/rpc.rs`, `crates/alloyd/src/main.rs`, `crates/alloyd/src/server.rs`, `ts/cli/src/chat.ts`, `ts/cli/src/rpc-client.ts`, `ts/cli/src/render.ts`, `ts/cli/src/args.ts`; **new** `ts/cli/src/approval.ts`. No policy decisions move into TypeScript. |
| W7 — acceptance and review | W6; every named gate passing | `.github/workflows/ci.yaml`, `ts/cli/scripts/real-model-acceptance.mjs`; **new** `ts/cli/scripts/phase2_tool_loop_real_model.sh`, `docs/acceptance/phase-2-exit-review.md`; update `README.md` only after verified behavior is available. |

W2 and W3 can proceed independently once W1 freezes the contract. W4 and W5 can proceed independently after W3. W6 depends on all four behavior components; W7 follows integration. Dependency manifests change only for packages actually introduced. Keep JSON Schema, process, and network dependencies pinned in `Cargo.lock` and inspect their runtime footprint.

## 5. Named tests and evidence

All tests below are planned additions. A fixture/mocked test proves orchestration or parsing; real-model tests prove the complete deployed path. Every rejection test must verify an execution counter, filesystem marker, or owned-process observation showing that no unauthorized effect ran.

| ID / exact test name | Exact test file | Required assertion / lane |
| --- | --- | --- |
| P2-01 `registry_schema_normalization` | **new** `crates/tool-registry/tests/registry.rs` | All origins use B.2; duplicate IDs and invalid registration schemas rejected. Linux + Mac. |
| P2-02 `tool_arguments_rejected_before_execution` | **new** `crates/tool-registry/tests/validation.rs` | Missing/extra/type-invalid arguments and unknown IDs execute zero calls. Linux + Mac. |
| P2-03 `qwen3_tool_call_chunk_boundaries` | **new** `crates/families/qwen3/tests/tool_calls.rs` | Same parsed blocks across every byte/chunk split, escaped JSON, mixed text, and multiple calls. Linux + Mac. |
| P2-04 `qwen3_malformed_tool_calls_never_execute` | **new** `crates/families/qwen3/tests/tool_calls.rs` | Truncated/invalid/ambiguous calls are protocol errors; no raw-text execution fallback. Linux + Mac. |
| P2-05 `tool_context_exact_count_and_identity` | **new** `crates/engine/tests/tool_rendering.rs` | Schema, arguments, and linked results appear in the exact rendered prompt/count; schema changes invalidate cache compatibility. Linux + Mac. |
| P2-06 `destructive_effect_requires_bound_approval` | **new** `crates/harness-core/tests/tool_policy.rs` | Shell cannot start before approval; denial, changed arguments, wrong-session, stale and duplicate decisions cause zero extra executions. Linux + Mac. |
| P2-07 `builtin_scope_and_output_limits` | **new** `crates/tool-registry/tests/builtins.rs` | Allowlisted reads/listing/writes; traversal/symlink/redirect policy; bounded shell and HTTP output/deadlines. Local fixtures only. Linux + Mac. |
| P2-08 `oversized_tool_result_artifact_roundtrip` | **new** `crates/harness-core/tests/tool_artifacts.rs` | Full output bytes, length, digest, reference and bounded preview persist/reopen; append-only artifact event; interrupted writes never become valid references. Linux + Mac. |
| P2-09 `tool_cancel_releases_owned_resources` | **new** `crates/harness-core/tests/tool_cancellation.rs` | Pause during generation, approval, hook and process execution reaches a terminal outcome, reaps owned children, and cannot affect another session. Linux + Mac. |
| P2-10 `plugin_tool_hooks_order_veto_annotation` | **new** `crates/harness-core/tests/plugin_hooks.rs`, **new** `crates/harness-core/tests/fixtures/tool_hook_plugin.rs` | Actual child process observes both hooks in order; veto prevents execution; annotation cannot mutate result or arguments. Linux + Mac. |
| P2-11 `plugin_protocol_timeout_and_sandbox` | **new** `crates/harness-core/tests/plugin_hooks.rs`, **new** `crates/harness-core/tests/plugin_sandbox.rs` | Mismatched/malformed/hung children fail as specified; macOS sandbox permits allowed resources and denies an undeclared resource. Protocol cases Linux + Mac; sandbox cases Mac. |
| P2-12 `canonical_tool_loop_persists_results` | **new** `crates/harness-core/tests/tool_loop.rs` | Scripted engine requests builtin, consumes result, emits final text; event ordering/call IDs preserved; loop limit is bounded; no duplicate execution after reopen. Linux + Mac. |
| P2-13 `approval_rpc_remains_responsive` | **new** `crates/alloyd/tests/tool_rpc.rs` | A second connection can approve/deny/pause while a send waits; exact B.6 ordering; one terminal turn outcome; unrelated status/metrics remain responsive. Linux + Mac. |
| P2-14 `cli_renders_approval_and_tool_result` | **new** `ts/cli/test/approval.test.ts`; `ts/cli/test/chat.test.ts`, `ts/cli/test/rpc-client.test.ts` | CLI displays tool/approval summary, sends explicit user decision, handles denial/cancel and noninteractive input without inventing approval. Node 26.13.0 Linux + Mac. |
| P2-15 `phase2_tool_loop_real_model` | **new** `ts/cli/scripts/phase2_tool_loop_real_model.sh` | Candidate-built real daemon and pinned Qwen3: read fixture, request shell, observe approval before marker file can exist, approve, consume result, produce final answer; loaded plugin receives both hooks. Retain RPC/events/output and denied-call evidence. Mac. |
| P2-16 `tool_roundtrip_overhead_under_50ms` | **new** `crates/harness-core/tests/tool_latency.rs` | SPEC §2.2 harness round-trip target with mocked inference; exclude human approval wait and client transport; record sample count/distribution and timing boundaries. Mac acceptance lane. |

Run each Rust package's `--all-targets` and `--doc` checks, plus the existing engine rustdoc gate. Explicit package commands after implementation are `cargo test --locked -p tool-registry --all-targets`, `cargo test --locked -p qwen3 --all-targets`, `cargo test --locked -p engine --all-targets`, `cargo test --locked -p harness-core --all-targets`, and `cargo test --locked -p alloyd --all-targets`. CLI verification remains `cd ts/cli && npm ci && npm test` on Node 26.13.0. W7 adds the new real-model script to the automatic Mac lane, with `upload-artifact@v4` under `always()` and the same candidate/artifact provenance rules as Phase 1 tests 13 and 14.

## 6. Exit review

Phase 2 is complete only when the candidate commit has passing evidence for all three SPEC Phase 2 exit statements:

1. A real model calls a builtin file-read and a policy-governed shell tool, and destructive execution waits for explicit bound approval (P2-06/12/13/14/15).
2. Malformed tool calls and invalid arguments are rejected and never execute (P2-02/03/04/07).
3. A real out-of-process plugin receives before/after events, can veto or annotate, and respects the documented version, ordering, limits, and mutation rules (P2-10/11/15).

Cancellation, durable artifacts/history, exact rendering, latency, and the existing Phase 1 gates are supporting invariants, not substitutes for those exit statements. The review records the candidate SHA, current corpus pin, CI run/artifact URLs, named-test outcomes, and any remaining defect. Failed or unrun required cases keep the phase open.
