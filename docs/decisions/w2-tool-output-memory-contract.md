# W2 tool-output storage accounting: landed implementation, acceptance pending

Status: slab storage, runtime handoff, decoder integration, and borrowed batch
transport are implemented in the checkpoints below. This is not a W2 sign-off
and is not a claim of a hard process-memory or OS OOM bound.

Implementation checkpoints:

- `2009b0eddce81150deabbf52be697e7d7ff997ed`: admitted slab-backed Qwen decoding
  and ownership-bearing batch delivery.
- `50d262b75e9b7637984299086ca01cd9a7c06bd2`: strict rustdoc correction.
- `362aeef902c921f9f47a8c8153df777e708d7814`: borrowed slab-notification transport,
  exact escaped-size preflight, and connection write-failure protection.

These SHAs identify implementation history, not test results for a later
candidate. Run the collector after committing this update to bind actual results
to that candidate. No successful run or native acceptance is asserted here.

## Counted storage and ownership

`crates/engine/src/parser_allocation.rs` computes one backing `Layout` with
checked array sizes, region alignment, inter-region padding, and final padding.
The configured regions cover delimiter/UTF-8 carry, a pending call body, decoded
string/key bytes, nesting records, duplicate-key descriptors, flat JSON nodes,
retained proposal payload, and proposal descriptors. Capacity never grows.

The native runtime includes that slab in output admission, partitions its guard
from the same atomic startup grant, retains it in pending startup, and transfers
it into the selected decoder. It does not reserve those bytes a second time.
Qwen planning also lists its separately allocated decoder owner layout.

Successful JSON validation on the slab path creates no argument Map/Value tree.
Tool names are decoded; argument JSON retains its validated original text bytes.
This is not JSON canonicalization. Validated calls commit into the retained slab
region; source scratch and committed payload overlap in lifetime within the
single admitted backing allocation.

A frozen allocation owns both the slab and its exclusive reservation. Scratch
remains retained and charged alongside payload until the final frozen owner
drops. Deallocation precedes release of the reservation. This design does not
claim to free scratch early or reduce the charge while scratch is still resident.

The worker/session flush path transfers a frozen batch to the generation driver.
Publication occurs only after protocol completion, usage reconciliation, session
cleanup, and scheduler terminal arbitration succeed. Failure/cancellation before
that boundary discards the batch. Views cannot outlive the owner, and delivery
of a batch does not authorize an effect. Consumers still require successful
terminal observation plus registry, argument, policy, and approval checks.

The generic legacy Map-proposal path and its heuristic charge remain for
compatibility. They are not the representation used by selected slab-backed Qwen
generation and must not be cited as exact accounting for arbitrary backends.

## Parser placement and compatibility

The admitted parser lives in `crates/engine/src/slab_parser.rs`, behind the Qwen
adapter. The legacy family parser remains in
`crates/families/qwen3/src/tool_calls.rs`. This records the implementation's file
placement change from the original W2 file table; it does not add an engine
dependency to the family crate or make the generic generation driver parse Qwen
syntax. Supported grammar/fixture parity remains a review item.

The legacy parser accepts arbitrary byte fragments. The slab parser receives
valid UTF-8 strings after incremental detokenization. Evidence records those
layers separately; a UTF-8-boundary test is not renamed into an arbitrary-byte
parser test. The current slab parser rejects alternate function/parameter
spellings. Supported Qwen3/Coder artifact and syntax scope must be reviewed.

## RPC transport boundary

`crates/alloyd/src/rpc/slab_transport.rs` borrows frozen proposal views. It measures
the complete escaped notification plus newline before any destination write,
then serializes the same immutable views directly. The selected batch path does
not allocate a call Value vector, reparse arguments, or retain an encoded line.
The existing validated-JSON-text wire format is preserved.

The service dispatch helper routes slab batches to that writer. Byte limits and
socket per-write timeouts are explicit. Failed/zero-progress/flush-failed
connections refuse subsequent writes, preventing an appended fatal response from
corrupting a partial NDJSON frame. Socket timeout is per write, not a whole-turn
or human-approval deadline. Transport setup does not enable tool execution.

The compatibility `wire_event_body` helper can still materialize JSON values.
Legacy non-batch response encoding, incoming-request buffering, kernel/socket
buffers, serializer stack state, and control-plane diagnostic allocations are
outside this selected no-intermediate-batch-buffer claim. Deployment still needs
review of explicit transport-policy configuration and tool-aware supervision.

## Evidence collection

`scripts/w2_evidence.py` schema version 2:

1. Requires a clean candidate HEAD containing the implementation checkpoints.
2. Records candidate SHA, relevant source blob IDs, toolchain logs, commands,
   log SHA-256 digests, return codes, timeouts, and observed test names.
3. Lists each selected test target first and rejects empty discovery, missing
   critical tests, duplicate names, and unexpectedly small suites.
4. Runs tests with stable pretty output and compares all named passing results
   to discovery. Zero tests, ignored/failed tests, missing/duplicate results,
   multiple summaries, or summary count mismatches fail that gate.
5. Checks clean HEAD before and after every subprocess and stops on detected
   drift. This is not an immutable worktree or adversarial concurrent-change proof.
6. Runs collector self-tests, workspace checks/Clippy/regressions, and strict
   engine/alloyd rustdoc. Compile/doc command success is labeled separately from
   named test evidence; zero doctests is not substituted for parser coverage.
7. Writes evidence outside the checkout, preserves individual failure logs,
   and does not overwrite an existing evidence directory.

P2-03 records both legacy byte-fragment and admitted-slab UTF-8-fragment tests.
P2-04's primary current-path evidence is
`generation::w2_slab_delivery_tests::malformed_later_slab_call_publishes_zero_effects`;
the earlier named compatibility-session test remains supplementary. P2-05 stays
`tool_context_exact_count_and_identity` against the shared renderer/tokenizer.

Supporting gates include the 20 allocation/layout/ownership tests, output-plan
and runtime handoff tests, 12 slab parser tests, 8 slab delivery tests, Qwen
construction tests, schema key/predicate tests, host-reference cache tests,
borrowed transport, real dispatch helper, and write-failure ownership tests.
Test floors are drift detectors, not a claim that a count alone proves behavior.

## Evidence distinctions that remain required

- Host-reference schema-change tests execute the host cache, not MLX retained
  KV or an SSD restore. Disk-key unit tests do not prove restore behavior.
- Scripted-inference worker tests and direct decoder token tests are not a
  trained-model daemon tool-loop acceptance run.
- Unix-socket and service-helper tests do not remove the text-only supervisor's
  fail-closed restriction or establish approved builtin execution.
- Supported Qwen3/Coder syntax/fixture review and a final candidate-specific W2
  review remain pending. No missing implementation is inferred merely because
  this evidence has not been recorded.

The report exposes `recorded_tests_passed`, `quality_checks_passed`, and
`collection_passed` separately. `w2_complete` remains false until independent
acceptance review; the collector does not self-certify the work package.

## Cache/render identity

Tool schemas/history are rendered and counted through the real pinned tokenizer.
Schema fingerprints are checked in render identity, retained compatibility,
metadata, and SSD keys. Appendix B.1 canonical session-state framing is unchanged.
A canonical state ID alone remains insufficient proof of cache reusability.
