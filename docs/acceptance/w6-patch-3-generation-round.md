# W6 patch 3 — bounded canonical generation round

Inspected parent: 21ffcf6571f2aecb661258b25d8b74bf801871f2,
feat/33-canonical-tool-loop, after W6 patch 2. Keep #33 open.

## Recorded placement and API

A private supervisor/tool_round.rs child module, declared and re-exported from
supervisor.rs, extends the existing Supervisor with generate_tool_round. Public
request/config/outcome/owner types live at harness_core::supervisor. The existing
crate root, registry, engine types and dependency manifests are unchanged.
Tests join the existing supervisor_turn integration target through an included
module, reusing only its real fixture renderer/tokenizer and telemetry interface.
The runner explicitly formats/checks the .inc source as well as the crate tree.

## What this patch wires

The opt-in method reads the current active session and latest open run.started,
uses the patch-1 durable session/turn/round tracing identity, constructs canonical
context through patch 2, invokes the actual engine render/count and generate APIs,
and collects exactly one successful engine round. No store lock crosses engine
polling, rendering, on_started, on_event or cancellation. A final open-turn check
prevents publication after pause/resume, completion or a newer durable turn.

The caller must already own the daemon's one-active-turn slot, create the user /
run.started records, advance the serial round without ID reuse, and keep the
original absolute deadline. This method checks a supplied round is below an
explicit cap, but does NOT allocate that slot or implement a whole bounded loop.

Only text and runtime warnings reach on_event. Neither proposals nor engine
GenerationCompleted/RequestFailed are forwarded as user-turn notifications.
The outcome is a generation result, not a user-turn success or durable result.
The eventual coordinator owns user-turn persistence and exactly-one terminal.

## Proposal ownership and trust

Legacy proposal trees move without cloning. Slab proposals keep the original
FrozenParserAllocation and its real admission reservation through the returned
outcome. Slab argument JSON is borrowed, never decoded/materialized here.
The legacy serialized-byte check uses a counting writer rather than a second
JSON String/Value. Names and arguments remain untrusted; model-supplied approved
and call_id fields stay ordinary argument data. Unknown names are deliberately
not resolved/authorized here. Later W3 admission must bind daemon call IDs,
validate schema and policy, persist requests and reject invalid proposals with
canonical protocol results before any effect. This method creates no permit.

Explicit positive ceilings cover round index, generated tokens, accepted text,
warning bytes, total events, complete proposal count, proposal payload bytes,
whole retained slab reservation and existing context metadata. Existing engine
render/token/admission limits still apply. These ceilings are not new memory
reservations, RSS limits or a hard bound on all transient library allocations.

EOF/failure/cancellation/raw deltas/mixed or duplicate batch delivery, text after
proposals, empty successful output, and output-limited proposals fail closed.
No fallback raw-text execution exists. The collector trusts the engine contract
that no events follow a successful terminal, drops its stream at that terminal,
and never attempts to drain an unbounded post-terminal stream.

## Deadlines and cleanup limits

The absolute deadline is checked around cooperative engine polls and callbacks,
not renewed per event. Pending waits recheck the session at most every 20 ms
when store access and engine polling cooperate. Synchronous/blocking engine code,
a contended mutex or an arbitrary callback cannot be forcibly preempted by this
API; a deadline-crossing callback cannot subsequently publish proposals.
On an ordinary error, streams/proposals drop and the exact expected request ID
gets a bounded best-effort cancellation request. The primary failure is preserved;
cleanup cancellation failure is not proof of cleanup success. A foreign handle ID
is never used as a cancellation target. This is NOT proof of native/process
termination or W4/W5 child reaping. Panics rely on normal stream/owner unwinding;
this patch does not establish a panic-recovery protocol.

## Tests and acceptance limits

Nine named tests use the actual supervisor, SQLite history, existing canonical
context builder, existing Rust renderer/HF tokenizer and actual admitted parser
slabs. They verify no store lock crosses callbacks/engine calls, no engine terminal
or proposal escapes the callback, distinct round IDs, original slab reservation
lifetime, release on EOF/failure/cancellation/limits/callback error/deadline/pause,
foreign-ID handling and zero fixture executions / no unexpected durable events.
Scripted inference is not trained-model evidence. Unknown-name preservation is
an untrusted-data test, not successful registry validation or an authorized call.

Still pending: proposal binding/registry validation with protocol outcomes,
W3 durable admission/approval coordination, W5 hook receipt persistence, W4
receipt-owned execution, repeated generate/result rounds, durable user-turn
terminal/error handling, responsive session.approve/configuration wiring, CLI,
P2-12/13/14, Linux/Mac CI evidence and W7 real-model/exit acceptance.
The deployed daemon still selects the legacy text-only run_turn_streaming path;
this new API is not silently enabled by startup or client defaults.
