# W3 dependency, freeze and fixture provenance

Status: reproducible verification procedure; no execution result is asserted here.

## Historical facts

The frozen manifest records starting_commit
58d7db53d8fb6f549cddba990dd9e83521989d92. The canonical fixtures first appear in
68b7cf0c446c12883a065687ef316de9b8b9c01a. The policy commit
a578f175c504e54719abb164bb1d638f328cfc5d promotes sha2 = "=0.10.9" from a
dev-dependency to a runtime dependency and adds the existing workspace
 tool-registry path dependency. Its Cargo.lock patch adds only tool-registry to
harness-core's dependency list. No registry version/checksum changes occur in
that patch. Later candidate changes must be audited separately.

## Audit procedure

Run the attached verify-w3-provenance.py against this checkout. Python 3.11+
(tomllib) and the repository Rust toolchain are required. The verifier:

- Uses the frozen starting commit as baseline and requires ancestor relationship.
- Compares all lock package records, including registry versions, sources,
  checksums and dependencies, allowing only the harness-core tool-registry edge.
- Audits each intervening committed Cargo.lock change as well as the working copy,
  so a temporary version change reverted at HEAD is not hidden.
- Confirms sha2's exact manifest pin and existing locked version.
- Confirms the frozen manifest and normative Markdown remain byte-identical to
  their first committed versions and checks the normative SHA-256.
- Runs exact freeze and canonical fixture tests, rejecting zero-test success.
- Runs a new Rust reproduction test that emits all computed canonical strings,
  SHA-256 digests and the domain-separated binding digest, then compares them
  against unchanged committed vectors. Python never encodes the canonical bytes.
- Writes logs, source hashes, platform/toolchain, candidate/dirty state, audit
  results and a Rust-generated vector report outside the checkout.

No fixtures, lockfile, normative record or manifest are regenerated in place.
The frozen design-time status remains historical and is not an acceptance flag.
A failed comparison is a stop/review event, not permission to update expected
hashes, accept dependency churn or overwrite golden values automatically.

## Scope of the provenance claim

Current vectors are reproduced by Rust from the committed input_json under the
pinned implementation and compared byte-for-byte and digest-for-digest. This
establishes reproducibility, not the undocumented original creation command.
Do not retrospectively claim that an absent historical generator was run.
The record does not prove an independent canonicalization implementation; the
production encoder is used and existing golden values remain the comparison.

A dirty-tree result is local evidence. Use --candidate only on a clean committed
checkout for SHA-bound evidence. Keep Linux/macOS acceptance, full regressions,
API review and W3 sign-off separate. The checker never changes checklist boxes
or signs acceptance automatically.
