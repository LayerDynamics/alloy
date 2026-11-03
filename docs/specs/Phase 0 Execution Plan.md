# Phase 0 Execution Plan — Decision Spike

**Spec:** SPEC-1 v1.9 (§4.1, Phase 0) · **Duration:** week 1 of M1, 2026-11-03 → 2026-11-10 (fallback gate at 1.5 weeks, per Q2) · **Owner:** solo

This plan lists the exact file set, named test cases, and the mapping to the Phase 0 exit criteria. It is committed to the repo at the start of Phase 0; the Phase 1 execution plan is written at Phase 0 exit (exit criterion 4).

## Exit criteria → deliverables

| # | Exit criterion (SPEC-1 §4.1) | Delivered by |
|---|------------------------------|--------------|
| 1 | Bindgen layer builds against the vendored pin with a passing coverage inventory (or fallback exercised and recorded) | `crates/mlx-ffi/` + `vendor/` + `PINS.md`; tests `coverage_inventory_complete`, `vendored_build_reproducible`; outcome (own layer or mlx-c-rust fallback) recorded in `docs/decisions/DR-0001` |
| 2 | Names reserved | Work items W1 (GitHub, crates.io `alloyd`, npm `alloyml`, PyPI `alloy-convert`, Homebrew tap `alloy`); verified in W1 checklist |
| 3 | `bench/corpus/` published in-repo | `bench/corpus/` tree below; validated by `corpus_*` tests |
| 4 | Phase 1 execution plan written | `docs/specs/PHASE-1-EXECUTION-PLAN.md` (same file-set + named-tests format) |

## File set (exact — everything Phase 0 creates)

```text
repo/
├── README.md                  # stub: name, pitch, status, license, repo map
├── LICENSE                    # Apache-2.0 (full text)
├── .gitignore
├── rust-toolchain.toml        # pinned Rust channel
├── PINS.md                    # vendored mlx-c pin: v0.8.x commit, tree hash, MLX v0.32.4
├── .github/workflows/ci.yml   # Linux lane (every PR): vendored build (CPU MLX, §4.2) + FFI tests +
│                              #   inventory gate (the §3.10 engine-mocked fast lane begins in Phase 1,
│                              #   when there is an engine to mock); Mac lane (self-hosted, on
│                              #   merge-to-main and release tags per §3.10): Metal/unified tests
├── scripts/
│   ├── vendor.sh              # updates the vendored pin, rewrites PINS.md + mlx-c.sha256
│   └── coverage_inventory.sh  # regenerates inventory, diffs against checked-in copy (CI gate)
├── vendor/
│   ├── mlx-c/                 # vendored mlx-c sources, pinned commit, checked in
│   └── mlx-c.sha256           # hash of the vendored tree (PINS.md companion)
├── crates/mlx-ffi/            # the Q2 bindgen spike
│   ├── Cargo.toml
│   ├── build.rs               # CMake build of vendored sources + bindgen invocation
│   ├── wrapper.h               # #include of every public mlx-c header
│   ├── src/
│   │   ├── lib.rs             # RAII layer: arrays, devices, streams, closures, mlx_compile
│   │   └── bindings.rs        # bindgen output, checked in (regenerated on pin bumps)
│   ├── tests/
│   │   ├── raii.rs             # named tests 1-3, 5
│   │   ├── compile.rs          # named test 4
│   │   ├── unified.rs          # named test 6 (Mac lane only)
│   │   └── ops.rs              # named test 7
│   └── coverage/
│       └── inventory.txt       # generated symbol→declaration map, checked in
├── bench/corpus/              # the Q6 frozen corpus
│   ├── README.md              # freeze record, run instructions, validation results
│   ├── manifest.json          # machine-readable definition: tiers (36 GB hardware + 8 GB simulated), models, scenarios, metrics (decode tokens/s, TTFT p50/p95, peak memory, agent-loop overhead)
│   ├── models.json            # Qwen3-4B/14B 4-bit, Llama-3.1-8B 4/8-bit, Gemma-3-12B 4-bit
│   │                          #   (mlx-community refs + artifact hashes; same model+quant on
│   │                          #    llama.cpp/Ollama/oMLX for comparison runs)
│   ├── scenarios/
│   │   ├── code-qa.json       # 50 questions over the fixed repo snapshot
│   │   ├── agentic-loops.json # 20 multi-turn tool-loop tasks
│   │   ├── long-context.json  # 8k and 32k documents
│   │   └── prefill-heavy.json # large-prompt scenarios
│   ├── fixtures/
│   │   ├── repo-snapshot.sha256   # pinned public repo @ fixed commit + tree hash
│   │   └── documents/            # the 8k/32k documents, checked in, + .sha256
│   └── validate.sh            # runs the corpus_* named tests (Linux CI, structural only)
└── docs/
    ├── specs/
    │   ├── SPEC-1-mlx-agent-harness.md   # v1.9
    │   ├── PHASE-0-EXECUTION-PLAN.md     # this document
    │   └── PHASE-1-EXECUTION-PLAN.md    # written at Phase 0 exit
    └── decisions/
        └── DR-0001-mlx-c-pin-and-bindgen-outcome.md
```

`docs/decisions/` (DR-numbered, ironbark precedent) is a deliberate one-directory extension of the §4.1 layout; DR-0001 records the vendored pin, the minijinja/Node LTS pins (Phase 0 scope item c), the own-layer-vs-fallback outcome, and the name-reservation results.

## Named tests

| # | Test (name as written in code) | Proves | Lane |
|---|--------------------------------|--------|------|
| 1 | `array_raii_roundtrip` | mlx_array create/dispose through RAII guards, no leaks (ASan/leaks) | Linux + Mac |
| 2 | `device_and_stream_lifecycle` | device + stream creation, synchronize, default-selection | Linux + Mac |
| 3 | `closure_smoke` | mlx_closure create/call through the FFI | Linux + Mac |
| 4 | `mlx_compile_smoke` | a trivial graph compiles and evaluates via `mlx_compile` | Linux + Mac |
| 5 | `lazy_eval_and_synchronize` | lazy evaluation + stream sync semantics hold through FFI | Linux + Mac |
| 6 | `cpu_gpu_views_alias_memory` | CPU/GPU views of one array alias the same memory (zero-copy, FR-17) | Mac only |
| 7 | `quantized_matmul_smoke` | quantized-matmul op surface is reachable through the FFI | Linux + Mac |
| 8 | `coverage_inventory_complete` | every public symbol in vendored mlx-c headers has a Rust declaration; diff is empty | Linux CI |
| 9 | `coverage_drift_rejected` | a symbol removed from the inventory fails the gate (the gate actually gates) | Linux CI |
| 10 | `bindgen_output_stable` | regenerating bindgen reproduces the checked-in `bindings.rs` byte-for-byte | Linux CI |
| 11 | `vendored_build_reproducible` | two clean builds of the vendored pin produce identical library hashes | Linux CI |
| 12 | `pins_hash_matches` | PINS.md recorded hash equals the hash of the vendored tree | Linux CI |
| 13 | `corpus_manifest_validates` | `manifest.json` parses and conforms to its schema | Linux CI |
| 14 | `scenario_counts_match_freeze` | exactly 50 code-QA questions, 20 agentic tasks, 8k+32k documents, prefill-heavy set present (Q6 freeze) | Linux CI |
| 15 | `corpus_models_pinned` | every model entry carries an mlx-community ref + recorded artifact hash | Linux CI |
| 16 | `fixtures_hashed` | repo snapshot and documents hashed, hashes recorded in the manifest | Linux CI |

Model-download verification (that the pinned mlx-community artifacts resolve and hash-match) is a one-time local Mac run recorded in `bench/corpus/README.md` — not a CI test, since CI never downloads models.

## Work items

| ID | Item | Detail | Budget |
|----|------|--------|--------|
| W1 | Names + repo | Create the public GitHub repo (open-source from day one, §4.3); reserve crates.io `alloyd`, npm `alloyml`, PyPI `alloy-convert` via placeholder `0.0.0` publications; create the `homebrew-alloy` tap repo with a stub `Formula/alloy.rb`; config dir `~/.alloy/` documented in README | Day 1 |
| W2 | Pins | Record in DR-0001: vendored mlx-c v0.8.x commit + hash (MLX v0.32.4), minijinja version (FR-9), Rust channel, and the Node LTS line current at execution (minimum 22 — verify the active LTS, expected Node 26; Node 22 EOLs 2027-04-30) | Day 1 |
| W3 | Bindgen spike | `vendor/` + `crates/mlx-ffi/` per the file set; FFI surface = arrays, devices, streams, closures, `mlx_compile` (§3.2); tests 1-12 | Days 1-7 |
| W4 | Corpus | `bench/corpus/` per the file set; tests 13-16; local artifact-resolution run recorded in the corpus README | Days 2-4 (parallel) |
| W5 | CI | `.github/workflows/ci.yml`: Linux lane on every PR (vendored CPU-MLX build per §4.2, all Linux-runnable tests, inventory gate — the §3.10 engine-mocked fast lane begins in Phase 1); Mac lane on merge-to-main and release tags per §3.10 (test 6) | Day 5 |
| W6 | Close-out | DR-0001 finalized; `PHASE-1-EXECUTION-PLAN.md` written (exact file set + named tests for the vertical slice, reviewed against the Phase 1 exit criteria) | Days 7-8 (within the week; the 1.5-week fallback budget covers overrun if W3 slips) |

**Fallback gate (Q2):** if W3 exceeds ~1.5 weeks of build work without tests 1-8 green, stop — adopt/fork mlx-c-rust instead, record the decision and the boundary it plugs into (the infra-crate API isolates family crates from the binding layer either way) in DR-0001, and do not re-open the decision.

## Success review

Phase 0 is done when the exit-criteria table above is fully green: CI runs tests 1-16 with the inventory gate enforcing coverage, names are reserved, `bench/corpus/` is frozen in-repo, and PHASE-1-EXECUTION-PLAN.md exists. Anything not on the file list above is out of Phase 0 scope.
