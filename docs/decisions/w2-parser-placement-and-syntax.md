# W2 parser placement and supported-syntax boundary

Status: implementation placement is recorded here; shared-candidate test results
are collected by `scripts/w2_evidence.py`. This document is not a trained-model
or full tool-loop acceptance record.

## Decision

Keep reservation ownership and the admitted parser in the engine integration
layer. The bounded implementation is `crates/engine/src/slab_parser.rs`, selected
by `crates/engine/src/backends/qwen/output.rs`. The legacy public parser remains
at `crates/families/qwen3/src/tool_calls.rs` for compatibility.

This amends the earlier W2 file-placement table; it does not add an engine
dependency to qwen3, move model-family dispatch into the generic generation
loop, or imply that the old parser has become allocation-bounded. Qwen-specific
syntax remains behind the Qwen adapter. Slab ownership is transferred through
family-neutral decoder/session output hooks to the completion gate.

## Shared supported scope

Both streaming parsers are tested against the SAME checked-in corpus:
`crates/families/qwen3/fixture/tool_calls.json`.

Supported call syntax is a JSON object inside `<tool_call>` / `</tool_call>`:
exactly a string `name` and object `arguments`, with no model-supplied identity
or authority fields. Shared fixtures cover ordinary/empty output, mixed text,
ordered calls, Unicode, JSON escaping, invalid envelope shape, duplicate keys,
truncation, and a malformed later call.

The test suite in `crates/engine/src/slab_parser/parity.rs` compares semantic
projections: concatenated assistant text and the ordered calls/argument values.
It does NOT claim identical callback/block publication schedules. The slab path
holds one allocation-owning call batch until successful finalization; earlier
legacy streaming APIs can expose tentative per-call values. Call IDs, approval,
registered implementations, and execution permission are not inferred by either
parser.

The legacy parser accepts byte fragments. The slab parser accepts valid UTF-8
strings after incremental detokenization. Shared tests use every UTF-8 boundary
and one Unicode scalar at a time; existing legacy/detokenization tests cover the
separate byte layer. These scopes must not be conflated.

## Explicit non-support

`unsupported_coder_syntax` already exists in the family corpus as a rejection
fixture. `<function=...>` / `<parameter=...>` markup is NOT enabled by this
change. The new parity test requires both parsers to reject that fixture.

A model/artifact requiring a different Qwen3/Coder protocol must not be advertised
as supported merely because these tests pass. Adding such a protocol needs a
separate grammar/fixture/artifact review and negative zero-effect evidence.
This records current scope; it is not an assertion about every Qwen-branded model.

## Resource-policy differences

Slab capacities are explicit and can reject an otherwise syntactically valid
fixture. Tests deliberately lower call capacity and require refusal without
falling back to the legacy parser. Grammar parity is asserted only where both
resource policies admit the shared fixture input. Numeric representation outside
the shared fixture domain, all possible Unicode cases, exhaustive JSON grammar,
and arbitrary model outputs are not certified by a small corpus.

## Evidence boundary

`parser-shared-fixture-parity` is candidate-bound test evidence. A passing result
establishes the assertions in this corpus; artifact scope and final W2 review
remain independent. Parser tests do not execute a builtin or start a model.
