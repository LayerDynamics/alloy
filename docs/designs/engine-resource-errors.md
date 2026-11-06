# Engine resource errors

## Outcome and acceptance

Callers distinguish execution contention, protected ownership, and memory
refusal without parsing a display message or querying the native worker.
The same captured cause reaches immediate RPC failures, terminal stream
events, and the CLI. No handler releases ownership or retries a request.

Required acceptance:

1. A pinned or leased resource reports its exact artifact, model instance,
   protection type, and positive observed handle count. Leased state also
   reports its canonical state ID. Retaining the error retains no protection.
2. An occupied scheduler/native slot reports `busy` with the conflicting
   request ID when known. Idle protected resources report
   `protected_resources`; merely being resident does not imply protection.
3. Memory refusal reports a typed reason and bytes from the actual constraint.
   An admission pause may be reported even when numerical capacity suffices.
   A host preflight refusal describes planned demand before native loading.
4. Unrelated leases survive reclamation of an eligible model. Lease counts
   remain a subset of the total observed state count; an empty pool has no
   protected owners. Model pins permit state-only reclamation of idle KV.
5. Immediate and streamed RPC failures preserve the same structured cause.
   The CLI retains RPC code/data, prints one terminal failure, and continues
   to understand responses from an older daemon or an unknown error kind.
6. Frozen 4B/14B acceptance continues to prove protected refusal, successful
   execution after unpinning, destruction before replacement loading, and
   an empty shutdown ledger. Synthetic reservation exhaustion is recorded
   separately from actual native-memory observations.

## Error categories

| Engine kind | Meaning | Caller action |
| --- | --- | --- |
| `busy` | Active/pending execution or a scheduler slot prevents this operation. | Wait for the identified request to finish, or explicitly cancel that request; requested cancellation does not imply completed cleanup. |
| `protected_resources` | The requested access, mutation, replacement, or reclamation conflicts with specific residency pins or state leases. | Arrange for the owning caller to release the relevant handles, then submit a new attempt. An explicit pin need not expire merely because time passes. |
| `memory_limit_exceeded` | A typed memory constraint prevents the operation. | Use the reason and captured byte counts below; do not infer physical OOM from this category. |
| `allocation_failed` | A fallible allocation actually failed. | Preserve this failure; do not describe it as a successful preflight rejection. |

`ProtectedResource::ModelPin` identifies one exact loaded model and a positive
pin-handle count. `ProtectedResource::SessionStateLease` identifies one exact
model and canonical publication with a positive lease-handle count. A blocker
list is scoped to the failing operation. For example, a model pin does not
block releasing that model's idle KV, and a lease on model A does not prevent
evicting unrelated eligible model C.

Protection is the reported cause only when it blocks a permitted operation.
A disabled reclamation policy or an exhausted action budget is not evidence
that an incidental pin caused the refusal. If an eligible candidate remains
but the bounded policy cannot take another action, the refusal retains the
capacity/policy cause instead of assigning blame to an unrelated owner.

Counts are observations at an ownership boundary, not reservations and not
authorizations. Reading a weak count must not upgrade the weak reference.
Final drops can happen concurrently, so diagnostics can immediately become
stale. Destructive operations still revalidate native ownership. Error
handling does not call `model_residency` or enqueue a native-worker command.

Each `ModelInstanceId` has a monotonically assigned process-local diagnostic
number, rendered as `model-N`. Clones share it; replacement loads never reuse
it in the same process. It is not a pointer, a durable identity across daemon
restarts, or a release/unpin capability. Canonical state identity remains the
existing digest and is not replaced by this diagnostic token.

## Memory reasons

Every `MemoryLimitExceeded` carries `operation`, `required_bytes`,
`available_bytes`, and a typed `MemoryLimitReason` captured at the origin.

| Reason | Byte interpretation and handling |
| --- | --- |
| `reservation_budget` | Required grant versus remaining ledger capacity; `budget_bytes` also reports the whole budget. A request exceeding that whole budget cannot fit merely by evicting other residents. |
| `advisory_estimate` | A fail-fast estimate exceeds reported available capacity. The advisory interface does not expose the complete budget, so none is invented. A successful estimate is not a reservation. |
| `allocator_ceiling` | Native planned/projected demand versus the applicable allocator/admit-line ceiling. `ceiling_bytes` preserves that domain even when the reservation ledger has space. |
| `admissions_paused` | The enforcer currently prohibits admission. Available ledger bytes may exceed the request; wait for a fresh acceptable observation, not merely a numerical budget change. |
| `buffer_limit` | A particular load, cache, output, or other buffer allowance was exceeded. This can occur while checking actual loaded buffers; it is not automatically a pre-load refusal. |
| `host_preflight` | Planned startup exceeds the safe ceiling from a fresh host observation. Includes artifact ID and the static/dynamic/Metal/reserve breakdown and binding component. Native model loading has not begun. |

Unconditional pressure-policy rejections use their own domain directly:
`RequestExceedsAdmissionBudget` uses reservation demand and ledger budget;
`RequestExceedsAllocatorCeiling` uses native demand and allocator ceiling.
Those decisions can precede the first `PressureAssessment`, so its absence
must not silently substitute ledger bytes for allocator bytes.

## RPC and CLI contract

The daemon keeps JSON-RPC API version 1, code `-32000`, and outer
`error.data.kind = "engine_failed"`. It adds `error.data.engine_error`, encoded
by `EngineError::diagnostic()`. A streamed `request_failed` event includes the
same `engine_error` beside its existing `message` and `partial_usage`.

```json
{
  "kind": "protected_resources",
  "message": "generation_startup blocked by protected resources",
  "operation": "generation_startup",
  "resources": [
    {
      "protection": "model_pin",
      "artifact_id": "qwen3-4b-4bit",
      "instance_id": "model-7",
      "handles": 1
    }
  ]
}
```

The harness retains native `EngineError` values in `EngineFailure`; synthetic
harness/transport failures retain their separate string error. The TypeScript
RPC error retains `code` and `data` instead of converting the entire result to
an unstructured `Error`. Human guidance uses typed fields. Unknown error kinds
or an older server's missing diagnostic preserve the original failure message.
No error is converted into successful output, an automatic retry, or CPU-only
execution. A streamed terminal failure followed by its final RPC rejection is
reported once.

The M9 acceptance driver's host check writes this same structured diagnostic
to its `acceptance_failed` evidence record. That preflight runs before native
loading and does not itself send an RPC. Foreground daemon boot failures
before a listener exists remain startup diagnostics on stderr; the additive
RPC contract applies to errors returned by a running daemon.

Resource metrics remain observations of the running daemon. A refused turn
does not make a healthy daemon disconnected, and the monitor does not invent
a last rejection from current pressure or reuse the guard's probe-error field.

## Source and compatibility

The definitions live in `crates/engine/src/error.rs`; pressure and native
ownership classify failures at their origins. Rust constructors of
`MemoryLimitExceeded` now supply the typed reason explicitly. Wire changes
are additive to the existing version-1 envelope. Clients must use stable
diagnostic fields rather than the wording of `message`.

The prior `ProtectedResources -> Busy` behavior was established before this
PR. This change deliberately replaces that overloaded classification while
preserving pin/lease lifetime and reclamation semantics. Historical acceptance
records retain their original classification and commit identity.
