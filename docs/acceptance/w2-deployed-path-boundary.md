# W2 deployed-path acceptance boundary

Status: code/test boundary record. No trained-model daemon tool-loop acceptance
run is claimed by adding this document or by passing the unit/integration gates.
The collector records actual per-layer results against one clean candidate SHA.

## Distinct layers of evidence

| Layer | Collected gate | What it proves | What it does not prove |
| --- | --- | --- | --- |
| Qwen output decoder | `qwen-decoder-hooks` | Actual detokenization/parser construction and slab-owner transfer with fixture tokens | A trained model selected the calls |
| Native worker/session and completion | `runtime-slab-handoff`, `slab-delivery` | Real ownership/gate code exercised with controlled inference and failure injection | Trained-model inference or builtin execution |
| Shared grammar corpus | `parser-shared-fixture-parity` | Common-fixture semantic text/call projection and rejection behavior | Every Qwen3/Coder protocol or arbitrary JSON input |
| Service dispatch/socket codec | `rpc-borrowed-transport`, `rpc-notification-dispatch`, `rpc-write-failure-ownership` | Borrowed serialization, Unix-socket fixtures, preflight limits, failure cleanup | The deployed tool supervisor accepts/authorizes calls |
| Actual binary configuration | `daemon-transport-options`, `daemon-transport-cli` | Paired option validation and application to the actual service-construction helper | A model/daemon was started by these option tests |
| Candidate-built trained-model daemon tool loop | NOT COLLECTED by these gates | Requires separate deployed evidence | Cannot be inferred by combining the other rows |

The report's `acceptance_boundary` section represents these distinctions. Passing
all lower layers never sets the trained-model layer to passed.

## Deployment transport configuration

The actual daemon binary now accepts:

- `--rpc-max-notification-bytes N`: positive decimal byte ceiling for the entire
  escaped slab notification, including its newline.
- `--rpc-write-timeout-ms N`: positive decimal per-write socket timeout.

Supply both or neither. No implicit transport policy is selected. Missing pairs,
zero, malformed/out-of-range values, and duplicate flags are rejected before
artifact loading. The real service constructor applies a supplied policy using
`with_rpc_transport_limits`. Startup reports configured values, not a claim that
tools have been enabled.

With neither flag, legacy text behavior remains unchanged and slab notification
transport remains unconfigured/fail-closed. These flags do not enable engine tool
parsing, change effects/approvals, or remove the current supervisor rejection of
tool proposals. A per-write timeout is not a full turn/human-approval deadline.

## What a separate deployed run must retain

- Full candidate SHA; clean build/source provenance; binary identity; pinned
  model/artifact identities and relevant configuration.
- The actual daemon launch command including explicit transport policy.
- Observed RPC/event ordering, terminal outcome, and durable history where
  applicable; distinguish application test fixtures from model-produced calls.
- Zero-effect evidence for malformed/rejected generations and ownership release
  evidence for cancel/drop cases.
- The applicable policy/approval/tool-aware-supervisor prerequisites. A text-only
  supervisor's correct rejection is not a successful tool-loop run.

Do not fabricate those prerequisites or waive them to make this document green.
A real deployed run requires the appropriate subsequent work packages and actual
model/hardware execution. Unrun evidence stays pending, not implemented-by-doc.

## Other still-independent checks

Host-reference schema-cache tests and SSD-key unit tests are not an MLX retained
cache/SSD-restore schema-change acceptance run. Global process RSS, allocator
rounding, all RPC request/response paths, and OS OOM prevention remain outside
the selected parser-slab/borrowed-batch accounting contract. Final W2 sign-off
requires reviewer reconciliation with the plan and the recorded candidate.
