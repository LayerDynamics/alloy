# Design: oMLX-method memory parity (native engine)

Status: approved for implementation (`memory-omlx-parity`)  
Date: 2026-04-10  
Refs: SPEC-1 FR-17 / FR-18 / FR-19 / FR-31; user requirement that oMLX memory methods are REQUIRED  
Source of truth for numbers: [jundot/omlx](https://github.com/jundot/omlx) `omlx/process_memory_enforcer.py` (main), verified 2026-04-10

## Product thesis

Alloy preserves unified-memory headroom the way oMLX does: tiered reserves for the Mac, a live hard ceiling, soft/hard watermarks with Metal-pool drain before eviction, quantized-resident weights, multi-model pin/TTL/LRU, and hot RAM + cold SSD KV blocks. The existing RAII `AdmissionController` remains the **inner** ownership ledger; its budget is the **outer** live hard ceiling.

## Non-goals

- Depending on the oMLX process as the primary engine.
- Copying oMLX Python into the tree.
- Replacing the grant/`take`/`try_resize` startup ownership model.

## Methods (M1–M10)

| ID | Method |
| --- | --- |
| M1 | Memory-guard tiers `safe` / `balanced` / `aggressive` / `custom` |
| M2 | Live hard ceiling = `min(static, dynamic, metal_cap + cpu_footprint)` |
| M3 | Soft / hard watermarks + pool-drain grace before destructive reclaim |
| M4 | Usage = `max(MLX active, phys_footprint − unreleased_graphics)` (+ committed-model floor for load admit) |
| M5 | Prefill/generation admit against headroom line; reject naming binding component |
| M6 | Adaptive prefill chunk shrink to a configured floor (default 256 tokens) |
| M7 | `mlx_set_wired_limit` only when `iogpu.wired_limit_mb > 0`; else sysctl guidance |
| M8 | Affine4 (and family quants) stay resident; matmul via `mlx_quantized_matmul` |
| M9 | Multi-model pin / per-model TTL / LRU under soft/hard pressure |
| M10 | Hot RAM + cold SSD safetensors KV blocks (restore across daemon restart) |

## Tier reserves (M1)

Memory each tier leaves for the OS and other apps: `(share of RAM, min, max)`.

| Tier | Share | Min | Max | Other-apps active reclaim ratio |
| --- | --- | --- | --- | --- |
| `safe` | 0.20 | 6 GiB | 16 GiB | 0.0 |
| `balanced` | 0.08 | 3 GiB | 8 GiB | 0.25 |
| `aggressive` | 0.02 | 1.5 GiB | 4 GiB | 0.5 |
| `custom` | fixed 2 GiB static reserve | — | — | user-pinned ceiling (no vm_stat dynamic) |

```text
tier_reserve = clamp(total_ram × share, min, max)   # custom → 2 GiB
static_ceiling = total_ram − tier_reserve
```

## Hard ceiling (M2)

```text
dynamic (non-custom) =
    alloy_phys
  + free + inactive + active_file_cache
  + reclaim_ratio × other_apps_active
  − tier_reserve
  + release_credits   # briefly credit Metal clears until vm_stat catches up

metal_cap_raw =
    iogpu.wired_limit_mb × 1 MiB   if sysctl > 0
  else MTL recommendedMaxWorkingSetSize

metal_cap_effective = metal_cap_raw + cpu_footprint
  where cpu_footprint = max(0, phys_footprint − graphics_footprint)

hard_ceiling = min(non-zero among {static, dynamic, metal_cap_effective})
  custom: dynamic := user_ceiling_gb (still min'd with static + metal)
  guard disabled: hard_ceiling := 0  # callers treat as “no limit”
```

Errors and status MUST name the **binding** component (`static` | `dynamic` | `metal_cap`).

## Watermarks and prefill margins (M3 / M5 / M6)

Fractions of `hard_ceiling`:

| Tier | soft | hard | prefill headroom | prefill abort margin |
| --- | --- | --- | --- | --- |
| safe | 0.85 | 0.95 | 0.90 | 0.90 |
| balanced | 0.90 | 0.95 | 0.92 | 0.93 |
| aggressive | 0.95 | 0.98 | 0.97 | 0.97 |
| custom | 0.90 | 0.97 | 0.95 | 0.95 |

```text
soft_bytes = floor(hard_ceiling × soft)
hard_bytes = floor(hard_ceiling × hard)
admit_line = min(hard_ceiling, hard_bytes, floor(hard_ceiling × prefill_headroom))
abort_limit = min(static_ceiling, metal_cap_effective)   # NOT the jittery dynamic alone
```

### Enforcement ladder (poll ~1s; recompute ceiling each tick)

1. **ok** (`usage < soft`): ensure admissions unpaused.
2. Prefer **non-destructive MLX/Metal pool drain** for a bounded grace (several polls) before destructive action.
3. **soft**: pause new admissions; LRU-unload **non-pinned** idle models; in-flight continue.
4. **hard**: LRU + abort sole non-pinned in-flight / in-progress loads; idle reclaim if no victim.
5. **emergency**: usage ≥ `abort_limit` (+2 GiB margin or consecutive over-ceiling polls) may abort pinned work.
6. Prefill: predict peak; if over admit line → shrink chunks down to `prefill_min_chunk_tokens` (default 256) or reject/abort with typed error.

## Usage signal (M4)

```text
usage = max(mlx_active, phys_footprint − unreleased_graphics_bytes) + external_process_bytes
```

During active generation, sample MLX counters on the executor thread and cache them for the enforcer (do not call Metal from the background poll thread).

Load admission MUST also floor against a **committed model-memory accumulator** so settled Metal pools cannot under-report and double-admit a second large model.

## Wired limit (M7)

Alloy already wraps `mlx_set_wired_limit` as `mlx_ffi::memory::set_wired_limit_bytes`.

- If `iogpu.wired_limit_mb > 0`: raise process wired limit to `min(desired, sysctl_cap)`, desired = static ceiling suggestion floored to whole MiB and ≤ 95% of RAM.
- If sysctl unset (`0`): **do not** call `set_wired_limit`; leave Apple’s default; surface `sudo sysctl iogpu.wired_limit_mb=N` in daemon status / doctor.
- `set_memory_limit` / `set_cache_limit` remain configured from the engine plan as graph-eval guidelines (MLX docs: not a hard OS ceiling).

## Inner admission ledger

`AdmissionController::new(hard_ceiling)` (or resize budget when the enforcer refreshes). Startup continues to use `reserve_grant` → `take` → `try_resize` per `docs/designs/startup-reservation-grant.md`. Components are incremental; peak then resident for loads. The ledger never replaces live OS/Metal observation.

## Quantized-resident weights (M8)

Stop Affine4 → host `Vec<f32>` → MLX f32 for **projection** residency. Keep packed U32 weights + scales/biases as MLX arrays; execute with `mlx_quantized_matmul`. Embeddings and `lm_head` may arrive Affine4 on disk but are dequantized **once** at MLX upload into resident float32 (embedding lookup / vocab projection need dense rows). Host loader charge estimates all matrices as Affine4 (group 64 by default); MLX `max_resident_bytes` estimates projections as Affine4 and embed/`lm_head` as float32, plus a small decode/workspace allowance — never a full float32 duplicate of the projection stack.

## Multi-model pool (M9)

Engine-owned pool:

- Load / unload
- Pin (skip LRU)
- Per-model idle TTL
- Soft/hard LRU of unpinned idle models
- Committed-byte accumulator for pre-load admit

## Tiered KV (M10)

- Fixed-size blocks, prefix hash / CoW as needed for correctness
- **Hot** tier in RAM (configurable max size)
- **Cold** tier: safetensors files under `--paged-ssd-cache-dir`, scanned at startup for restore
- Spill cold under pressure **before** model unload when possible
- `hot_cache_only` disables SSD writer while keeping hot RAM cache

## alloyd surface

| Flag | Meaning |
| --- | --- |
| `--memory-guard {safe\|balanced\|aggressive\|custom}` | Tier (default `balanced`) |
| `--memory-guard-gb N` | Custom ceiling (GiB); used when tier is `custom` |
| `--admission-budget-bytes N` | Test/override only; when set, acts as custom pinned ceiling path |
| `--paged-ssd-cache-dir PATH` | Cold KV directory |
| `--hot-cache-max-size SIZE` | Hot tier cap (bytes or `%` of ceiling) |

## PR stack

| PR | Scope |
| --- | --- |
| PR0 | This design + SPEC-1 / PHASE-1 amendments |
| PR1 | M1–M7 live ceiling + enforcer + alloyd boot — **landed in tree** (`memory_guard`, `memory_enforcer`, alloyd `--memory-guard*`, 1s poll, pool-drain grace, admission pause). Multi-model LRU unload still PR4. |
| PR2 | M8 quantized-resident Qwen3 — **landed in tree** (`MatrixWeight::Affine4`, `mlx_quantized_matmul`, host vs MLX resident bounds, Affine4 wiring checkpoints). |
| PR3 | M5–M6 prefill admit line + chunking — **in tree** (`prefill_admit`, enforcer `admit_line_bytes`, `FamilyRuntime::with_memory_enforcer`, alloyd chunk 1024 → floor 256). |
| PR4 | M9 multi-model pool |
| PR5 | M10 SSD KV |
| PR6 | W6 E2E 13–14 resume |

## Acceptance matrix

| Check | Evidence |
| --- | --- |
| Ceiling honesty | Status shows static/dynamic/metal + binding |
| Quantized admit | Qwen3-4B cold load under default `balanced` on ≤36 GB Mac |
| Soft before hard | Pool drain and/or LRU non-pinned before abort |
| Binding errors | Over-budget names component + raise recipe |
| Wired policy | Raised iff sysctl set; else doctor/sysctl warning |
| Multi-model | Second large model evicts unpinned idle; accumulator prevents double-admit |
| SSD KV | Spill → restore after daemon restart for matching prefix |
| W6 E2E | Tests 13–14 green on this stack |

## Relationship to Appendix D

Appendix D RAM-tier **profile defaults** (weights/KV class tables) remain for model profiles and benchmarks. They are **not** the live controller formula. The live controller is this document’s oMLX-method ceiling.
