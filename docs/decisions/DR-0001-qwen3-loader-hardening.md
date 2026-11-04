# DR-0001 amendment: Qwen3 loader and rendering hardening

Status: proposed; Rust regression tests and CI must pass before acceptance.

Supersedes the loader-default description in the earlier Qwen3 review amendment.
Default raw checkpoint limit: 16 MiB, cumulatively counting config.json,
model.safetensors.index.json, and all loaded shard bytes. Default decoded-weight
limit: 16 MiB. These two payload representations can coexist; JSON parsing,
index maps, allocator overhead and application memory are additional. These
are payload limits, not a process RSS or available-system-memory guarantee.
Large trusted checkpoints require explicit LoadOptions overrides.

Checkpoint errors now distinguish I/O (operation, path, ErrorKind and message),
format parsing (JSON versus safetensors), and validation. The old Checkpoint
variant is retained for compatibility, but the loader no longer emits it.

The supplied path must exist and be a regular file or directory; canonicalize
that path before choosing its root. File input never falls back to siblings.
Indexed shard inventories must exactly match weight_map in both directions.
Present numeric architecture fields must be numeric. Only SiLU activation is
accepted when hidden_act is declared. Affine group sizes are inferred from
validated scale/bias shapes when metadata is absent; declared grouping must agree.

Tied model construction rejects a separate output head, including disk heads.
The renderer rejects special-token syntax beginning with <| and reserved
thinking/tool delimiters in all message bodies. This prevents delimiter-based
turn forgery; it does not solve semantic prompt injection.

Owned cache prefill resets on empty input, capacity/length failures, and decode
failures. A cache belonging to another model is rejected without mutation.
No additional Rust dependencies or lockfile changes are required by this patch.
