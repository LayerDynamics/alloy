# DR-0001 amendment: Qwen3 review dependencies and execution scope

Status: proposed; requires local compilation, lockfile resolution, and CI evidence.

- Pin minijinja =2.12.0 at workspace scope for chat rendering.
- Pin safetensors =0.4.3, half =2.4.1, sha2 =0.10.9, and serde_json =1.0.145.
- Regenerate and commit Cargo.lock using Cargo before pushing.
- Rendering supports a deliberately versioned text-only, non-thinking template,
  with an assistant generation prompt and a SHA-256 identity over exact bytes.
  This subset is not the entire upstream tool/thinking template. Unsupported
  tool roles return errors rather than pretending to support tool conversations.
- Keep the source Qwen3-4B tied-embedding profile: the published source config
  currently declares tie_word_embeddings=true. The exact pinned artifact must
  be checked locally before changing that setting. Untied weights are supported
  with a separate, shape-validated lm_head and the loader checks config agreement.
- Full-logits prefill has a default 64 MiB output budget covering f32 result rows,
  row headers, and a single f64 projection row. Large callers should use bounded
  cached prefill, which returns only final-position logits.
- The output budget is not a process-wide memory guarantee. Weights, activations,
  allocator overhead, and other processes still consume memory.
- Cached decoding is host-side f64 reference execution, not MLX acceleration.
  Registration remains conservative until cached-path validation is accepted.
- The safetensors reader supports single files, indexed shards, floating weights,
  and MLX affine four-bit matrices. Large checkpoints require explicit raw-file
  and decoded-weight limits via LoadOptions; defaults are 512 MiB for each.
- Real pinned quantized artifact loading and token conformance remain unverified.
- Misplaced duplicate family tests are removed; active tests live under qwen3/tests.
