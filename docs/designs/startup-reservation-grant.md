# Design: Atomic startup reservation grants in the native runtime

Status: landed on `4-engine-core` (`6c750f3`)  
Date: 2026-04-09  
Branch: `4-engine-core`  
Refs: #4

Approval notes:

- History capacity is part of the grant; unused remainder is released when
  prior history is reused.
- `reserve_output_after_reclaim` evolves into a reclaim-then-grant hook
  (`admit_startup_after_reclaim`).
- Land with the held admission assertion fix.
- Open questions resolved: history in grant; evolve reclaim helper; implement
  as written.
- Policy integration tests updated so mid-startup snapshots expect the full
  pending grant (not the old output-only / peak-only totals).

## Product thesis

The engine must admit a generation's **incremental** memory needs as one
atomic decision, then hand each reserved slice to the real resource that
owns the corresponding native allocation for its full lifetime. Startup
must not temporarily free and reacquire the same bytes, and it must not
double-count by reserving a grant and then calling `reserve` again for
the same requirement.

## Current state (verified in tree)

### Admission API (`crates/engine/src/admission.rs`)

`ReservationGrant` already exists:

- `AdmissionController::reserve_grant(operation, &[components])` admits
  the sum under one lock as a single `active_reservations` entry.
- `ReservationGrant::take(operation, bytes)` transfers ownership into a
  `MemoryReservation` without changing `reserved_bytes`.
- Partial `take` increments `active_reservations` (complete `take`
  replaces the grant's entry). This accounting is already correct on
  `HEAD` (`face2d2`).
- Drop/`release` frees only the grant's unclaimed remainder.

Unstaged correction (held for this design's landing commit): the split
test in `grant_can_be_split_into_loading_peak_then_resident_weights`
must expect reserved totals that still include the grant remainder
(`120` / `100` / `100`), not the earlier incorrect `100` / `80` / `80`.

### Runtime reserves today (`crates/engine/src/runtime.rs`)

| Stage | Operation label | When | Lifetime after success |
| --- | --- | --- | --- |
| `admit_output` | `generation output buffers` | After pressure reclaim | `Arc` held by `NativeSession` / `BudgetedStream` until stream drop |
| `prepare` | `generation workspaces` | Always when `temporary + headroom > 0` | `Active::_workspace` until `finish` |
| `prepare` | `model loading peak` | Only when no reusable resident model | Resized to resident; lives in `Resident::weights` across requests |
| `prepare` | `model KV buffers` | Only when no matching resident cache | `ResidentCache::reservation` while cache retained |
| `prepare` | `executed token history` | When `retains_contents` and prior history is missing/too small | `StateCandidate::history` → retained candidate / published state |
| `finish` | `retained final-position logits` | Optional; failure discards retention | Independent idle reservation; **not** part of the startup ledger forecast |

`pending` is currently `Option<RequestId>` only. There is nowhere to hold
an unpartitioned grant between `AdmitOutput` and `Prepare`.

`pressure_demand` already forecasts the incremental ledger:

```text
weights_peak (0 if reusable model)
+ cache      (0 if matching cache)
+ history    (0 if retention does not keep contents)
+ temporary + headroom + output
```

Public `estimate_memory` reports resident weights plus loader overlap in
`temporary_bytes`. The **reservation** path uses peak bytes then
`try_resize` down to resident. The grant must follow the reservation
path (peak), not the estimate breakdown.

## Goal

Replace the multi-step `reserve(...)` sequence for the incremental
startup ledger with:

1. One `reserve_grant` after successful startup pressure.
2. `take` into each real owner.
3. Release of any unused grant remainder when prepare finishes or when
   pending startup is abandoned.

Do **not** wrap existing `reserve` calls with an outer grant of the same
amounts. That double-counts and rejects valid work.

## Non-goals

- Changing pressure policy, reclaim actions, or MLX allocator ceilings.
- Putting retained-logits reservation into the startup grant (finish-time
  optional path stays independent, matching today's forecast).
- Deduplicating already-resident weight/cache bytes inside
  `reserve_grant` itself; callers pass incremental components only.
- Changing scheduler permit ownership or generation streaming APIs.

## Proposed design

### 1. Pending startup owns the grant remainder

```rust
struct PendingStartup {
    request_id: RequestId,
    /// Unpartitioned remainder after output `take`.
    grant: ReservationGrant,
}

// NativeState
pending: Option<PendingStartup>,
```

Identity checks that today compare `pending.as_ref() == Some(request)`
become `pending.as_ref().map(|p| &p.request_id) == Some(request)`.

Dropping `PendingStartup` (via `finish`, failed prepare cleanup, or
worker teardown) drops the grant and releases only unclaimed bytes.
Already-taken reservations stay with their owners.

### 2. Acquire the grant only after pressure, from post-reclaim residency

In `admit_output`, keep the existing order:

1. Verify artifact / cancel checks.
2. Reclaim MLX cache / unpublish idle state as today.
3. Run the reclaim closure (compatible resident trimming +
   `apply_startup_pressure`).
4. **Then** compute incremental components from **current** residency
   (same rules as `pressure_demand`).
5. `reserve_grant("generation startup", &components)`.
6. `take("generation output buffers", output_bytes)` → `Arc`.
7. Store `PendingStartup { request_id, grant: remainder }`.

If step 5 or 6 fails, no pending entry is published and any grant is
dropped before returning the error.

Shared helper (name illustrative):

```rust
fn incremental_startup_components(
    plan: &RuntimePlan,
    retention: RetentionPolicy,
    residency: RuntimeResidency, // reusable_model + reusable_cache_buffer_bytes / same_cache
) -> Result<Vec<u64>, EngineError>
```

Component order (stable for tests and debugging):

1. `plan.output_bytes`
2. `temporary_bytes + headroom_bytes` (workspace)
3. `max_load_peak_bytes()` if `!reusable_model`, else `0`
4. `cache.max_cache_bytes()` if cache must be allocated, else `0`
5. history capacity bytes if `retains_contents(retention)`, else `0`

Zeros are allowed in the slice; `reserve_grant` already accepts them as
long as the total is positive. If the total would be zero (degenerate
plan), keep today's behavior: reject or skip according to existing
output/workspace invariants rather than inventing a new empty-grant
path.

`pressure_demand.additional_reservation_bytes` must remain equal to the
sum of these components so pressure evaluation and grant sizing cannot
drift.

### 3. Partition in `prepare` without calling `admission.reserve` for grant-covered items

`prepare` requires `pending.request_id` to match the control request,
then:

| Need | Action |
| --- | --- |
| Workspace `> 0` | `grant.take("generation workspaces", workspace_bytes)` → `Active::_workspace` |
| Load model | `grant.take("model loading peak", peak)` → load → `try_resize(resident)` → `Resident::weights` |
| Allocate cache | `grant.take("model KV buffers", cache_bytes)` → allocate → `ResidentCache` |
| New token history | `grant.take("executed token history", history_bytes)` → build `TokenHistory` from that reservation |
| Reused prior history | Do **not** take history; leave those bytes on the grant |

At the end of successful prepare, before clearing pending:

- `grant.release()` (or drop pending after moving taken pieces) so any
  unused forecast (especially unused history when reusing a candidate)
  returns to the ledger.
- Set `pending = None` and install `Active`.

Refactor helpers:

- `TokenHistory::from_reservation(reservation, capacity)` for the grant
  path; keep or thin-wrap today's `new(admission, capacity)` for tests.
- Do not add a second reservation inside those helpers when a grant
  slice is supplied.

### 4. Lifetimes and abandonment

```text
AdmitOutput success
  ├─ output Arc ─────────────────────────────► session / stream drop
  └─ PendingStartup.grant remainder
        │
        ├─ abandon / finish(pending only) ───► Drop grant (release remainder)
        │                                       Drop output via session cleanup
        │
        └─ prepare
              ├─ take → workspace ───────────► Active until finish
              ├─ take → weights (+ resize) ──► Resident across requests
              ├─ take → KV ──────────────────► ResidentCache while retained
              ├─ take → history (if new) ────► candidate → retained state
              └─ release unused remainder
```

Invariants:

- Native allocation still happens **after** the matching reservation
  exists (unchanged).
- Resize of loading peak releases overlap bytes from the
  `MemoryReservation`, not from the grant (grant already transferred
  the peak amount).
- `finish` optional logits reservation stays a separate
  `admission.reserve` / soft-fail path.
- Undelivered `AdmitOutput` reply already calls `finish(request)`; with
  pending grant ownership that path releases the remainder.

### 5. Failure and rollback matrix

| Failure | Accounting outcome |
| --- | --- |
| Pressure rejects | No grant; ledger unchanged by grant path |
| `reserve_grant` fails | No pending; ledger unchanged |
| `take(output)` fails | Grant dropped (full release); no pending |
| Consumer abandons admission future | `abandoned` flag + later `finish` / session drop releases pending grant and output |
| `prepare` fails after some `take`s | Local `MemoryReservation`s drop on unwind; pending cleared or finished so grant remainder releases; existing `invalidate_cache_for` behavior retained |
| `prepare` succeeds with unused history forecast | Explicit `grant.release()` returns unused bytes before active execution |
| Mid-request cancel after prepare | Workspace drops on `finish`; resident weights/KV follow retention policy |

### 6. What must not change

- Output escaping startup as `Arc<MemoryReservation>` on the session/stream.
- Weight reservation outliving the request when `retain_model` is set.
- KV and history outliving the request when `retains_contents` holds.
- Pressure reclaim still running **before** the grant is acquired, so
  the grant sees post-reclaim availability and residency.
- Single-threaded native worker assumptions: residency cannot change
  under a pending grant between admit and prepare on another thread.

## Files to change (implementation phase)

| File | Change |
| --- | --- |
| `crates/engine/src/admission.rs` | Land the held test assertion fix (`120`/`100`/`100`). No API change required for runtime wiring. |
| `crates/engine/src/runtime.rs` | `PendingStartup`; grant acquire in `admit_output`; partition in `prepare`; `TokenHistory` from reservation; finish/pending identity updates; unit tests. |
| `crates/engine/src/lib.rs` | Only if exports/docs need updating (likely unchanged). |

No new public crate API is required unless we choose to export
`ReservationGrant` for integration tests outside the engine crate.
Internal unit tests in `runtime.rs` / `admission.rs` are enough for the
first landing.

## Test plan

### Already covered (admission)

- Atomic grant admit, partition, partial drop, overpartition, concurrent
  grants, poison fail-closed, peak→resize→remainder transfer (after
  assertion fix).

### Add / extend (runtime)

1. **Grant rejects when incremental total does not fit** after pressure
   (mirror `unrelated_reservations_are_not_reclaimable` but through
   grant-sized components).
2. **Abandon pending startup** releases grant remainder and leaves no
   stuck `active_reservations` once output/`PendingStartup` are dropped.
3. **Prepare rollback** after a successful output take: failing load or
   cache allocate releases taken slices + remainder; budget returns to
   the pre-prepare reserved baseline (output may still be held by the
   test harness until dropped).
4. **Reusable model/cache**: grant components omit weights/KV; prepare
   does not `take` them; resident guards unchanged.
5. **History reuse**: grant includes history forecast; prepare reuses
   prior `TokenHistory`; unused history bytes released at prepare end.
6. **Loading peak resize**: grant takes peak; after load, reservation
   bytes equal resident; ledger reflects resize.
7. **Retained ownership**: after successful generate/finish with
   retention enabled, weights/KV/history reservations remain; logits
   still independently reserved or discarded.

Prefer focused `NativeState` unit tests where possible (same style as
`pending_startup_can_finish_without_prefill` and
`reserve_output_after_reclaim` tests). Add a backend-level test only if
unit hooks cannot express grant + reclaim ordering.

## Implementation sequence (after approval)

1. Land admission assertion fix in the same commit as runtime wiring (or
   as the first commit of a two-commit stack on `4-engine-core`).
2. Introduce `PendingStartup` and update pending identity/`finish`
   handling with a grant-less test fixture helper if needed.
3. Add `incremental_startup_components` and switch `admit_output` to
   grant + output `take` (keep `reserve_output_after_reclaim` as a thin
   wrapper or replace call sites carefully so existing reclaim tests
   still express the reclaim-then-admit ordering).
4. Switch `prepare` workspace/weights/KV/history to `take`.
5. Add runtime unit tests from the plan above.
6. Run `cargo fmt`, `cargo test --locked` for `engine` lib + relevant
   integration tests, `git diff --check`.
7. Commit and push only after explicit approval of the staged diff.

## Open questions for approval

1. **History in the grant:** Recommended yes — include when
   `retains_contents`, release unused remainder when prior history is
   reused. Alternative: leave history as a prepare-time `reserve` (weaker
   atomicity, smaller grant).
2. **Replace vs wrap `reserve_output_after_reclaim`:** Recommended
   evolve it into reclaim-then-grant (or reclaim-then-caller-grant) so
   existing reclaim unit tests keep a clear hook. Avoid a parallel path
   that still `reserve`s output beside a grant.
3. **Commit shape:** Single commit ("engine: add atomic startup
   reservation grants" with runtime wiring + assertion fix) vs two
   commits (assertion fix, then runtime). Recommendation: one commit if
   both land together; the earlier `face2d2` message already describes
   the admission half.

## Explicit non-approach

Do not acquire `reserve_grant(sum)` and then call the existing
`admission.reserve(...)` helpers for output/workspace/weights/KV/history.
That double-counts and is rejected by this design.
