# SPEC-1: Alloy — MLX Agent Harness

> An open-source, Apple-Silicon-native agent harness with a pure-Rust MLX engine — models, tools, MCP, skills, plugins, context management with compaction, and automations, running across all unified-memory Macs with all inference and data processing on-device (the only always-on network egress is required anonymous telemetry, FR-35; the only other outbound traffic is user-initiated — model downloads and user-configured integrations). Rust daemon (harness + engine, one process); TypeScript CLI/TUI as thin clients.

**Naming (decided, v1.6):** the project is **alloy** — binaries `alloy` (TS CLI) and `alloyd` (Rust daemon), npm package `alloyml`, Homebrew tap formula `alloy`, config dir `~/.alloy/` (bare `alloy` is taken on crates.io and npm, hence the suffixed names).

**Date:** 2026-11-03 (revised same day — v1.1: pure-Rust mlx-c engine; v1.2: TypeScript clients + Apple RAM tiers; v1.3: consistency fixes; v1.4: Goldfish round-1 confirmed fixes, 4 isolated reviewers; v1.5: Goldfish round-2 structural fixes; v1.6: Phase 0 questions resolved — project named "alloy", own-bindgen decision, benchmark corpus adopted; v1.7: all remaining open questions resolved — conversion via uv, telemetry required, speculative decoding promoted to SHOULD, Kokoro-82M TTS; v1.8: post-decision consistency pass — 9 fixes; v1.9: Goldfish round-3 fixes — all 4 lenses, 26 findings applied)
**Author:** User (with Perplexity Computer)
**Status:** Draft
**Version:** 1.9

## Revision History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | 2026-11-03 | Initial spec: Rust control plane + Python MLX-LM inference worker |
| 1.1 | 2026-11-03 | Runtime changed to pure-Rust mlx-c engine (per-family model crates, vendored mlx-c, single process); IPC gateway component removed; Python demoted to optional build-time conversion CLI; architecture, phases, risks, and appendices revised accordingly |
| 1.2 | 2026-11-03 | User-facing layer opened to TypeScript: CLI and TUI are thin TS clients over the Rust daemon's local API; harness and engine remain pure Rust; distribution, security, and API contracts updated |
| 1.3 | 2026-11-03 | Pre-implementation consistency review fixes: Phase 1 scope includes daemon API + TS CLI + basic prefix cache; R8 panic-boundary wording corrected (native MLX crashes remain fatal); LoRA added as FR-34 COULD; timeline constraint restated as ~7 months (v1.0 by end of May 2027); NFR/approval IPC scoping, oMLX network-client wording, SessionState backend wording, "single binary" → daemon binary, test-matrix 36 GB alignment, R1 contingency FR-4 note, `alloy convert` delegation note, B.5 trait versioning tied to B.1 |
| 1.4 | 2026-11-03 | Goldfish review round 1 (4 isolated reviewers; structural lens pending next round): timeline remnants fixed (§1.5, §4.3); FR-4 split into FR-4a text MUST / FR-4b multimodal SHOULD; FR-5 oMLX demoted to SHOULD with trigger + new R10 upstream risk; §1.6 Alternatives Considered added; Phase 0 decision spike added (resolves Q1/Q2, pins FR-9 renderer and TS runtime); B.6 contract gaps closed (skills/mcp/automation methods, tool_call_delta, sessions.resume); Appendix D 48 GB arithmetic fixed + interpolation rule; compaction summarizer contract, builtin tool set, and SKILL.md convention specified; plugin hooks trimmed to minimal V1 set; KV persistence dropped from V1; TUI demoted to SHOULD; benchmark scenario ex-ante criteria; security notes (same-user UDS vector, at-rest encryption) |
| 1.5 | 2026-11-03 | Goldfish review round 2 (structural lens, 14 findings — 4 Major, 10 Nit, all applied): §1.7 Goals (V1) added as the counterpart to §2.4 Non-Goals; `alloy stats` added to B.6 + §3.4 CLI surface; §1.6 oMLX-contingency reference aligned with R9/R1; FR-24 plugin host scheduled into Phase 2; FR-30 split into FR-30a (CLI MUST) / FR-30b (TUI + SDK SHOULD); §3.2 event names aligned to snake_case contracts; §3.4 render_and_count signature aligned to B.1 (tokenizer_id); M5→M7 edge added to dependency graph; Appendix C panic row updated with the R8 native-crash caveat; Appendices promoted to numbered H2 (§9); bold pseudo-headings promoted to real headings; Appendix D formula moved to a code fence; binary-size operator unified (≤ 150 MB); review-process residue stripped from normative text |
| 1.6 | 2026-11-03 | Phase 0 questions resolved ahead of the spike: (Q1) project named **alloy** — binaries `alloy` (TS CLI) / `alloyd` (Rust daemon), crates.io `alloyd` and npm `alloyml` verified free (bare `alloy` is taken on both crates.io and npm; Homebrew tap formula `alloy` under the project account); config dir `~/.alloy/`; (Q2) binding strategy decided: **own thin bindgen layer over vendored mlx-c** (ironbark DR-0002 precedent — maximum control of the full stack; fallback: adopt/fork mlx-c-rust if own-layer maintenance proves too heavy), R3/R4 updated; mlx-c reference updated to v0.8.x / MLX v0.32.4; (Q6) benchmark corpus adopted ex-ante (36 GB hardware + 8 GB simulated; Qwen3-4B/14B, Llama-3.1-8B, Gemma-3-12B; four scenario classes) — Phase 0 now validates and publishes these decisions rather than making them; Q3/Q4/Q5/Q7 remain open |
| 1.7 | 2026-11-03 | All remaining open questions resolved: (Q3) conversion CLI ships as an optional **uv** tool (`uvx alloy-convert`), mlx-community artifacts stay the primary zero-conversion path; (Q4) telemetry is **required and always-on** (not opt-in) — new FR-35, anonymous usage events (version, hardware tier, session/feature counts) to the project endpoint; privacy policy + published event schema ship with v1.0 (Phase 8 scope); §2.3 constraint reversed accordingly; (Q5) speculative decoding **committed to shipping**: FR-12 promoted COULD → SHOULD, scheduled into Phase 5 scope + exit criteria (conformance suite passes with drafting enabled); (Q7) **Kokoro-82M** is the V1 TTS family (mlx-community Kokoro-82M-bf16/8bit/4bit artifacts; native non-Python MLX implementation proven by kokoro-swift; misaki G2P phonemization in the family crate; M7 conformance gate unchanged). All §8 questions are now resolved — the spec has zero open decisions |
| 1.8 | 2026-11-03 | Post-decision consistency pass (9 findings, all applied): §2.3 binding-layer constraint updated to the resolved Q2 decision (was stale "open question Q2"); intro blockquote + §1.3 reworded — "entirely on-device"/"fully local" no longer contradicts required telemetry (FR-35) — the only egress is named explicitly; §3.7 security notes now enumerate the telemetry client as the only always-on outbound connection; §3.9 observability ties metrics to the FR-35 event schema; §3.10 + §3.4 CLI comment: conversion distribution unified to uv (`uvx alloy-convert`), pipx references removed; descope ladder gained speculative decoding (FR-12 SHOULD, slot 2 — cut before user-facing capability); FR-4b + Phase 7 + glossary now name Kokoro-82M per Q7 |
| 1.9 | 2026-11-03 | Goldfish review round 3 (all 4 lenses; 26 unique findings, all applied): telemetry client scheduled into Phase 1 scope + exit criteria, with the privacy policy + event schema published before the first tagged 0.x pre-release emits and Phase 8 finalizing (FR-35 phase-mapped); no-opt-out decision recorded in FR-35 + new R11 risk row (backlash + endpoint operation); telemetry field lists aligned to §3.9, offline drop + flush + failure-isolation semantics and stated purpose added; intro blockquote + §1.3 egress claim qualified to "always-on"; Node pin made "current LTS at Phase 0, minimum 22" (Node 22 EOLs 2027-04-30) with a Phase 8 re-check; §1.6 + FR-4b descope order aligned to the actual ladder; TUI made a V1 SHOULD cuttable to v1.1 (FR-30b/§3.2 "v1.1 target" removed); FR-4b multimodal artifacts pinned (Qwen2.5-VL, Whisper large-v3-turbo) with per-modality conformance criteria; FR-12 draft-model pairing rule pinned (profile field, Qwen3-0.6B drafting Qwen3-4B default); FR-29 job schema + V1 trigger vocabulary scheduled into Phase 6; npm daemon-bootstrap mechanism pinned (§3.10); B.4 enum extended (session.created, run.started, skill.loaded) + sessions-table note; `alloy doctor` scheduled into Phase 5; Phase 5 exit carries the FR-12 SHOULD qualifier; §8 renamed "Decisions (Resolved Questions)"; Appendix B labels promoted to headings; Revision History promoted to H2; PyPI `alloy-convert` added to Phase 0 reservations; "Sec." → "§"; FR-26/27 cite fixed; Phase 0 note v1.6/v1.7; RAM phrasing fixed (FR-18, Appendix D); single-binary claims reworded (§1.1, §1.4); R2 contingency linked to §6.1; authoring residue removed (Appendix C) |

---

## 1. Background

### 1.1 Problem Statement

Local agent harnesses on macOS today force a choice between three insufficient options:

1. **Harnesses with weak local runtimes** (e.g., generic OpenAI-API clients pointed at Ollama) — they get GGUF quantization and llama.cpp's Metal backend, but not MLX's unified-memory optimizations, tiered caching, or the fastest Apple-Silicon inference path.
2. **MLX servers without a harness** (e.g., oMLX, mlx-serve) — excellent inference, but they expose an API and stop there: no agent loop, no durable sessions, no skills, no tool policy, no compaction, no automations.
3. **Full agent harnesses that are not Mac-first** (e.g., Pi) — rich extensibility, but built on TypeScript abstractions, no built-in permission system, and no unified-memory awareness.

No open-source project combines a proper agent harness (tools, MCP, skills, plugins, context, automations) with an MLX-native inference plane that is explicitly optimized for Apple unified memory — and ships as one native daemon binary with no Python runtime (the TS CLI ships separately via npm).

### 1.2 Current State

- **MLX / MLX-LM** (Apple, Apache-2.0): MLX is the reference array framework for Apple Silicon; MLX-LM is its Python model library (loading, conversion, quantization, generation, caches). Foundation layer; Python-only at runtime.
- **mlx-c** (Apple, Apache-2.0): the official C API for MLX — wraps the C++ core (arrays, devices, streams, closures, `mlx_compile`, Metal) and is explicitly designed as a bridge for binding other languages. MLX C v0.8.x tracks MLX v0.32.4.
- **Rust MLX bindings** (community): several crates wrap mlx-c — InftyAI/mlx (layered safe API, MLX-Swift-style), mlx-rs, and mlx-c-rust (binds every public symbol of mlx-c). A viable binding layer exists off-the-shelf.
- **Pure-Rust LLM runtimes are proven**: mlex is a from-scratch Rust runtime on mlx-c with no Python — vendored and statically linked MLX, loads HF-style model directories, chat, streaming, tool calling, multimodal input, prompt caching. OminiX-MLX extends the same pattern with shared inference-infrastructure crates (KV cache, RoPE, attention) plus one crate per model family. This precedent de-risks the core engine design.
- **oMLX** (Apache-2.0): MLX-native inference server with continuous batching, paged tiered KV cache (RAM hot tier + SSD cold tier, prefix sharing, copy-on-write), multi-model serving (LLMs, VLMs, embeddings, rerankers, OCR), memory enforcement, OpenAI/Anthropic-compatible APIs. Remains relevant as an optional backend.
- **llama.cpp / Ollama**: broad model support on Apple Silicon via Metal/Accelerate/NEON, but GGUF-based, not MLX-native.
- **Pi**: an open agent-harness reference with skills, extensions, SDK, RPC, and automation modes — but TypeScript-based, and explicitly ships no built-in permission system restricting filesystem, process, network, or credential access.

What is missing is the composition: a Rust-native harness and inference engine in one process — with a reusable conversion/quantization pipeline — where the model zoo is implemented as per-family Rust crates on mlx-c.

### 1.3 Target Users

- **Primary:** developers and technical users on Apple-Silicon Macs (8 GB to 128 GB+ unified memory) who want a local-first, extensible agent harness for coding, research, and automation — all inference and data processing on-device; the only always-on egress is required anonymous telemetry (FR-35), and the only other outbound traffic is user-initiated (model downloads, user-configured MCP).
- **Secondary:** contributors who extend the system with model-family crates, skills, plugins, and MCP servers.
- **Not targeted (V1):** Windows/Linux users, server fleet operators, enterprise multi-tenant deployments.

### 1.4 Motivation

- Apple Silicon's unified memory is uniquely suited to large-model local inference, and MLX is the fastest path on that hardware — but the agent ecosystem around it is thin.
- The mlx-c C API plus proven pure-Rust runtimes (mlex, OminiX-MLX) make a single-daemon-binary, no-Python architecture feasible now — a materially better product (simpler distribution, tighter memory control, one process to debug) than the hybrid design originally considered.
- The author's own workflows (daily local coding/research agent use) are the immediate driver; an open-source release makes the work reusable by the community.

### 1.5 Assumptions

- Solo developer, ~7 months to v1 (v1.0 by end of May 2027), working primarily solo with community contributions post-release.
- Development machine is the author's 36 GB unified-memory MacBook (a standard Apple RAM configuration, not a power-of-two size) — the 36 GB tier is a first-class benchmark and CI target, not an edge case.
- MLX and mlx-c continue to be actively maintained by Apple; the project pins mlx-c commits and vendors them into the build (mlex PINS.md pattern).
- Model artifacts from Hugging Face (safetensors, mlx-community pre-converted directories) remain the dominant distribution format; the HF-style model directory (config.json + safetensors shards + tokenizer files) is the canonical artifact format the engine loads.
- Python is never required at runtime. Conversion from raw checkpoints uses an optional build-time Python CLI (wrapping mlx-lm tooling), or users consume mlx-community artifacts directly.
- The oMLX optional backend runs as an external process (user-installed); communication is local HTTP — it is the only inference-plane component outside the main process (TS clients, plugins, and MCP servers are separate processes by design).

### 1.6 Alternatives Considered

Each viable architecture was evaluated and rejected for specific reasons; the raw material appears in §1.1-1.2 and the revision history, and is consolidated here as decisions:

| Alternative | Decision | Rejection rationale |
|-------------|----------|---------------------|
| Adopt or extend mlex / OminiX-MLX as the engine | Rejected; adopted as pattern precedent | Their goals differ: mlex is a Node-first general-purpose runtime with npm distribution; OminiX-MLX is a standalone ecosystem. This project needs the engine embedded in a Rust daemon with policy integration and daemon-owned session state. We adopt their proven patterns (vendored mlx-c builds via build.rs, per-family crates over a shared infra crate) rather than their products. |
| Harness over oMLX only (no native engine) | Rejected as V1 primary; recorded as fallback sequencing | Loses in-process unified-memory control (FR-17), prefix-reuse keyed to the harness's own context builder (FR-28), and integrated admission control; puts an HTTP boundary in the agent loop and makes the product dependent on an external project's API (R10). It remains the R9 contingency if the native engine stalls — the harness value stands alone over oMLX; R1's descope ladder cuts the oMLX adapter and speculative decoding (engine-plane optimizations) first, then FR-4b multimodal scope, before any MUST item is dropped. |
| Hybrid Python MLX-LM worker (v1.0 design) | Rejected (v1.1 revision) | Two runtimes, an IPC gateway in the hot path, and a Python environment for users; conversion tooling retained Python-only at build time, where it is harmless. |
| llama.cpp / GGUF as primary inference | Rejected | Not MLX-native: forfeits the unified-memory feature set (FR-17) and the mlx-community artifact ecosystem; GGUF conversion moves away from original higher-precision sources. Remains a possible optional backend post-v1 (FR-33). |
| TypeScript harness (Pi-style) for the whole system | Rejected | TS in the runtime path gives up in-process memory control and adds a heavier runtime; Pi-class harnesses also ship no built-in permission system. TypeScript is retained where it is strong: thin user-facing clients (FR-30a). |
| Core ML execution | Excluded by decision | Excluded from the start of this spec (v1.0); not MLX-native, not required by any target workload. |

Sequencing note (why the native engine precedes harness polish in V1): the engine is the differentiating core and carries the highest technical risk (R3/R4); building it first means every later phase runs on the real substrate, and the harness-over-oMLX fallback remains available throughout — the riskiest component gets the most calendar room, not the least.

### 1.7 Goals (V1)

What V1 delivers — the consolidated counterpart to §2.4's Non-Goals:

- **Native MLX engine (MUST):** a pure-Rust, mlx-c-based inference engine in-process with the harness — FR-4a text families (Qwen3 text + Coder, Llama 3.x, Gemma 3), KV-cache management with prefix reuse, admission control against unified-memory budgets, and the FR-17/18/19 unified-memory feature set across all Apple RAM tiers (8–128 GB; 36 GB first-class).
- **Agent harness (MUST):** the canonical tool loop with structured tool-call parsing, argument validation, and a policy engine with approval gating; MCP client + skills; context engine with exact token counting and transactional compaction (FR-26/FR-27); durable, resumable sessions (event log); automations with crash recovery; plugin host with the minimal V1 hook set (FR-24).
- **User surface (MUST):** a TypeScript CLI (FR-30a) over the daemon's local JSON-RPC API (B.6); `alloy doctor`, benchmark/validation reports; required always-on telemetry (FR-35) with a privacy policy and published event schema.
- **V1 SHOULD surface:** FR-4b multimodal families (Qwen-VL + OCR, Whisper STT, TTS: Kokoro-82M per Q7 — with recorded deferrals per the TTS acceptance rule); speculative decoding with draft models (FR-12, committed); the oMLX adapter (FR-5); the published TS SDK and the TUI (FR-30b); embeddings/reranking first cut (FR-10).
- **Quality bar:** ≥ 70% of published benchmark scenarios meet-or-exceed vs. llama.cpp/Ollama with oMLX in every table; model profiles are only "supported" when their conformance suites pass (FR-8).

---

## 2. Requirements

### 2.1 Functional Requirements

**Model runtime (pure-Rust mlx-c engine)**

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-1 | MUST | The system MUST load and run supported model families from HF-style model directories (config.json + safetensors + tokenizer files), local or downloaded from Hugging Face, with streaming generation — entirely within the Rust process via vendored mlx-c. |
| FR-2 | MUST | The engine MUST be a pure-Rust runtime on mlx-c: MLX/mlx-c sources vendored and statically linked via the build system, no Python interpreter, no system MLX installation, and no other runtime process required. |
| FR-3 | MUST | Model architectures MUST be implemented as per-family Rust crates over a shared inference-infrastructure crate (KV cache, RoPE, attention, quantized matmuls), with a registry that maps `architectures` from config.json to an implementation. |
| FR-4a | MUST | V1 MUST implement the text families: Qwen 3 (text + Coder), Llama 3.x, and Gemma 3 (text) — this is the MUST floor. |
| FR-4b | SHOULD | V1 SHOULD implement the multimodal families: VLM (vision + OCR), STT, and TTS — planned for V1; under the R1 contingency they are cut after the oMLX adapter and speculative decoding (descope ladder slot 3; slip to v1.1 is a SHOULD-level deferral, not a MUST violation). Pinned artifacts (v1.9; revisable only in the Phase 7 execution plan, recorded as a decision): VLM = Qwen2.5-VL (mlx-community, 3B/7B class), STT = Whisper large-v3-turbo, TTS = Kokoro-82M (decision, v1.7 — Q7). Multimodal conformance criteria (M7 gate): VLM — fixture image-QA and OCR accuracy on a fixed image set; STT — word error rate on a fixed audio fixture set; TTS — misaki G2P phonemization correctness plus audio-output sanity (duration, sample rate, no clipping). TTS acceptance: Kokoro-82M passing its conformance criteria by M7; if it fails to qualify, TTS is recorded as deferred to v1.1. |
| FR-5 | SHOULD | The system SHOULD ship an oMLX adapter (users opt in per model profile), including its tiered KV caching and continuous batching, via its local OpenAI/Anthropic-compatible API. Promotion trigger: promote to MUST if native-engine family coverage blocks target users, or when large-context workloads need oMLX's SSD-tiered KV caching. The capabilities contract makes late addition a transport change, not a redesign. Compatibility policy: the adapter targets oMLX's OpenAI/Anthropic-compatible API (its stable surface), tested against a pinned oMLX release per release cycle. |
| FR-6 | MUST | The system MUST provide an optional build-time conversion CLI (Python, wrapping mlx-lm tooling) that converts raw checkpoints into MLX-format HF-style directories with configurable quantization recipes (bit width, group size, tensor exclusions), recording the recipe hash in the artifact manifest. The runtime MUST NOT depend on this CLI. |
| FR-7 | MUST | The system MUST load mlx-community pre-converted artifacts directly, with no conversion step. |
| FR-8 | MUST | The system MUST validate model artifacts before marking a model profile usable, producing a machine-readable validation report. Numerical spot-check pass criteria: max absolute logit deviation < 2e-2 (relative < 1e-2) on a fixed prompt set against reference vectors captured from eager-mode execution. Agent-conformance pass criteria: 100% of the profile's must-pass fixtures (tool-call formats, stop conditions, long-context, compaction quality). |
| FR-9 | MUST | Chat templates MUST be rendered in Rust with minijinja (decision recorded at Phase 0 — cache-identity-critical, so the renderer is pinned, not "or equivalent") with template-hash-based cache compatibility checks; tokenization MUST use the official HF tokenizers Rust crate. |
| FR-10 | SHOULD | The system SHOULD support embeddings/reranking models for retrieval-augmented context building. |
| FR-11 | SHOULD | The system SHOULD preserve quantization capabilities exposed by mlx-c (quantized matmuls, quantized KV where supported per family). |
| FR-12 | SHOULD | The system SHOULD support speculative decoding with draft models in v1.0 (promoted from COULD, v1.7 — committed to shipping), gated behind per-model conformance tests: a family is draft-enabled only when its conformance suite passes with drafting enabled. Draft pairing (pinned v1.9): the draft model is declared in the model profile (`draft: { artifact, max_draft_tokens }` — a profile field, admission-reserved per FR-31) and MUST share the target family's tokenizer; default V1 pairing: Qwen3-0.6B drafting Qwen3-4B (mlx-community artifacts) — the family that must ship draft-enabled per the Phase 5 exit criterion. |
| FR-13 | WONT | The system WILL NOT use Core ML in any version. |
| FR-14 | WONT | The system WILL NOT generate images or video (image scope is understanding, description, and content extraction only). |
| FR-15 | WONT | The system WILL NOT support training or fine-tuning in V1. |
| FR-16 | WONT | The system WILL NOT run a Python interpreter at inference time. |
| FR-17 | MUST | The engine MUST exploit MLX unified-memory features as the primary optimization path: shared CPU/GPU arrays with zero-copy weight residency (no host-device duplication of loaded weights), Metal memory-limit and cache-limit control, MLX allocator cache reclamation under pressure, and lazy/compiled graphs to bound temporary allocations. |
| FR-18 | MUST | The admission controller MUST work in actual installed RAM units, not power-of-two classes — Apple's RAM sizes are not uniformly power-of-two (8, 16, 18, 24, 32, 36, 48, 64, 96, and 128 GB configurations), and budgets MUST derive from the detected physical memory and measured wired-memory headroom. |
| FR-19 | MUST | The engine MUST expose and respect per-process Metal memory limits and wired-limit guidance (with `alloy doctor` reporting both), and MUST reclaim MLX allocator caches before evicting models under pressure. |

**Agent harness**

| ID | Priority | Requirement |
|----|----------|-------------|
| FR-20 | MUST | The system MUST implement an agent loop that renders model output into structured content blocks (text, tool calls, tool results) and executes tools from a canonical registry — never from raw unparsed model output. |
| FR-21 | MUST | The system MUST provide a tool registry where built-in tools, plugin tools, and MCP tools are normalized to one schema with per-tool policy (effect class, approval requirement, timeout, output-size limit). |
| FR-22 | MUST | The system MUST implement MCP client support: server lifecycle, tool discovery, schema normalization, timeouts, cancellation, and result-size limits. |
| FR-23 | MUST | The system MUST support skills loaded from a directory convention (SKILL.md + references + scripts), discovered by name/description and loaded on demand. |
| FR-24 | MUST | The system MUST support out-of-process plugins with a versioned lifecycle hook interface. V1 ships a minimal hook set — `tool.before_execute` and `tool.after_execute` only; the full hook surface (session, context, generation, compaction, automation events) is deferred until real plugin demand exists (post-v1). |
| FR-25 | MUST | The system MUST maintain durable, append-only session event history (messages, tool calls, artifacts, compaction checkpoints) in SQLite. |
| FR-26 | MUST | The system MUST render and token-count the exact model-visible context (template, tool schemas, skills, history) before each request, and admit requests only when input + output reserve + safety margin fits the validated context. |
| FR-27 | MUST | The system MUST implement compaction as a transactional operation producing structured checkpoints (goal, constraints, decisions, open tasks, artifact references), preserving original event history. |
| FR-28 | MUST | The system MUST implement KV-cache prefix reuse keyed on exact token-prefix compatibility, and MUST invalidate cache reuse when compaction or schema changes alter the rendered prompt. |
| FR-29 | MUST | The system MUST provide durable automations (schedule and event triggers) that create agent runs under explicit permission scopes, with overlap policy, deadlines, retries, and approval pauses. |
| FR-30a | MUST | The system MUST expose a CLI as a thin client (TypeScript) over the Rust daemon's local API; the daemon runs harness and engine in one process. |
| FR-30b | SHOULD | The system SHOULD ship a TUI in V1 (cuttable to v1.1 per the descope ladder — the B.6 API makes late addition cheap) and publish the TS SDK (it is the JSON-RPC client library, published as a byproduct). |
| FR-31 | MUST | The system MUST implement an admission controller that estimates and reserves memory (weights, KV state, prefill/decode temporaries, draft models, headroom) before loading a model or admitting a session, with graceful degradation (evict, reduce chunk size, reject with explanation). |
| FR-32 | SHOULD | The system SHOULD support multiple concurrent sessions sharing one loaded model, with isolated per-session state. |
| FR-33 | COULD | The system COULD support a GGUF/llama.cpp fallback backend in a later minor version (not V1-blocking). |
| FR-34 | COULD | The system COULD support inference-time LoRA adapters applied over a base model artifact (justifies the `lora_id` term in the B.1 cache-identity key; not V1-blocking). If unimplemented in V1, `lora_id` is null in the key. |
| FR-35 | MUST | The daemon MUST emit telemetry — anonymous usage events (version, hardware tier, session/feature counts, plus the aggregate §3.9 performance dimensions: tokens/s, TTFT, cache hit rate, tool latencies, memory reservation vs. actual) reported to the project endpoint over HTTPS/TLS — always-on with no opt-in flow and no opt-out in V1 (decision, v1.7; revisit post-v1 with adoption data — R11). Purpose: aggregate development guidance (feature usage, hardware mix, performance) — nothing more. Client behavior (pinned v1.9): events are counted in memory and flushed at least once per 24 h; when the endpoint is unreachable they are dropped — no on-disk queueing or buffering (offline Macs lose counts, never session data); telemetry failures are logged and never affect sessions, tools, or the agent loop. Scheduling: the emission client ships in Phase 1; the privacy policy and event schema are published before the first tagged 0.x pre-release emits; Phase 8 finalizes both. |

### 2.2 Non-Functional Requirements

#### Performance

| Metric | Target | Measurement |
|--------|--------|--------------|
| Inference throughput | Meet or exceed llama.cpp/Ollama (Metal) tokens/s on the same model and hardware in ≥ 70% of the project's published benchmark scenarios; oMLX included in every comparison table as the MLX reference point | Self-hosted Mac benchmark matrix against the Phase 0-frozen scenario corpus (§4.2), operational runbook per §4.4, published with each release |
| Time to first token (p50) | Within 2x of oMLX on the default engine for the same model | Benchmark matrix |
| Agent-loop overhead | Harness overhead (non-model) < 50 ms per tool-call round trip within the daemon (no IPC in the agent loop). Excludes approval waits (human latency, and approval round-trips traverse the TS client by design) and client→daemon transport | Integration test with mocked inference |
| Context render + count | < 150 ms for a 32k-token rendered context | Integration test |
| Cold model load | < 30 s for a 7B-class quantized model from local disk | Benchmark matrix |
| Daemon binary size | ≤ 150 MB release daemon binary (MLX Metal kernels dominate; acceptable); the TS CLI ships separately via npm | Release artifact check |
| Weight residency | Loaded weights exist once in unified memory (zero-copy between CPU/GPU views); no host-side duplicate | Engine memory instrumentation test |
| Pressure reclamation | Under simulated memory pressure, allocator-cache reclaim completes < 100 ms and avoids model eviction when headroom allows | Chaos test on 8 GB-tier profile |
| Memory safety | No OOM crashes under admission control; over-budget requests rejected with explanation | Chaos test on 8 GB-tier and 36 GB-tier profiles |

The "beat alternatives" bar is deliberately scoped: meet-or-exceed on a majority of benchmarked scenarios, honestly published, rather than a universal claim. Scenarios where MLX or the engine loses are documented, not hidden.

#### Reliability

| Metric | Target |
|--------|--------|
| Session durability | No event loss on process crash; sessions resume from durable history |
| Automation durability | Jobs survive daemon restarts; no duplicate external side effects (idempotency keys where supported) |
| Cache correctness | Zero silent wrong-context generations from stale KV reuse — token-prefix compatibility is proven, not assumed |
| Conversion reproducibility | Same source + recipe + pinned deps ⇒ identical artifact hash |

#### Security & Compliance

- Single-user, local-first: no network listeners enabled by default (oMLX backend is opt-in and local); plugins and MCP servers are the only trusted-but-sandboxed external processes.
- Tool policy engine: read/write/destructive effect classes; approval required for destructive effects by default.
- MCP server annotations are treated as untrusted descriptions; only local policy authorizes execution.
- Audit log of every tool intent, approval decision, and execution result.
- Secrets: OS keychain integration for any credentials (Hugging Face tokens, MCP server auth); never in plaintext config.
- Telemetry is required and always-on (no opt-in flow): the daemon reports anonymous usage events (version, hardware tier, session/feature counts, plus aggregate §3.9 performance dimensions) to the project endpoint (FR-35); the emission client ships in Phase 1 and the privacy policy + published event schema precede the first tagged 0.x pre-release that emits.
- License compliance: Apache-2.0 project; MLX/mlx-c (Apache-2.0) and all bundled dependencies license-compatible; vendored sources pinned by commit and hash (PINS.md pattern).

#### Scalability

- Target scale: one Mac, 1-16 concurrent agent sessions, 1-4 loaded models depending on tier. Not a multi-tenant server in V1.
- All Apple RAM configurations supported (budgets derived from detected RAM, never power-of-two assumptions): 8, 16, 18, 24, 32, 36, 48, 64, 96, 128 GB+ (see Appendix D capacity model).

### 2.3 Constraints

- **Languages:** Rust for the runtime path — harness and engine are one process (`alloyd`). TypeScript is acceptable and expected for user-facing clients (CLI, TUI, SDK bindings), which talk to the daemon over a local API and contain no harness logic, policy, or inference code. Python appears only in the optional build-time conversion CLI, never at runtime.
- **MLX access:** via Apple's official mlx-c C API, vendored and statically linked (build.rs + CMake, mlex pattern). Binding layer: own thin bindgen layer over the vendored pin (resolved v1.6, Q2 — ironbark DR-0002 precedent; coverage inventory fails CI on drift; fallback: adopt/fork mlx-c-rust).
- **Platform:** macOS 15+ on Apple Silicon (M1-M5). Linux support is a non-goal for V1 (mlx-c itself is portable; revisit post-v1).
- **Inference:** MLX-native. Core ML excluded entirely. llama.cpp/GGUF only as an optional later backend.
- **License:** Apache-2.0.
- **Timeline:** v1.0 by end of May 2027 (~7 months from the 2026-11-03 start; the original "6 months" framing was optimistic once M8 landed on 2027-05-31 — restated honestly per the consistency review).
- **Team:** solo developer; architecture must be contributor-friendly (clear crate boundaries, documented engine API).
- **Storage:** SQLite + plain artifact files; no external database services.

### 2.4 Explicit Non-Goals (V1)

- No Core ML, no image/video generation, no training/fine-tuning.
- No Python interpreter at runtime; no system MLX installation requirement.
- No Windows/Linux support.
- No multi-tenant serving, authentication for remote users, or horizontal scaling.
- No cloud-provider fallback models in V1 (may be a plugin later).
- No universal "run any architecture" promise: any source can be inspected; only implemented families execute (broad architecture coverage is oMLX's optional-backend role or a post-v1 mlx-c port). Compatibility is a first-class reported result, not an assumption.

---

## 3. Architecture

### 3.1 System Overview

One Rust daemon (`alloyd`) — harness and engine share the address space; the inference gateway IPC layer of v1.0 is gone. TypeScript CLI/TUI clients connect over a local JSON-RPC API on a Unix domain socket. The oMLX optional backend is the sole external inference process.

```text
┌──────────────────────────────────────────────────────────────────────┐
│                     macOS (Apple Silicon)                            │
│                                                                      │
│  ┌──────────────────────────┐        ┌──────────────────────────┐     │
│  │ TypeScript clients       │        │ Rust daemon (alloyd)      │     │
│  │                          │ local  │                          │     │
│  │  CLI / TUI / TS SDK      │───────▶│  Agent Supervisor         │     │
│  │  (thin: no policy, no    │ UDS/   │        │                 │     │
│  │   harness logic, no       │ JSON-  │        ├▶ Tool registry   │     │
│  │   inference)              │ RPC    │        ├▶ MCP client      │     │
│  └──────────────────────────┘        │        ├▶ Skills/plugins │     │
│                                      │        ├▶ Policy         │     │
│  ┌───────────────┐                    │        └▶ Automations    │     │
│  │ SQLite +     │◀──── artifact ──── │        │                 │     │
│  │ artifacts/   │      references    │  Context & Sessions       │     │
│  └───────────────┘                    │  (history, builder,       │     │
│                                      │   budgets, compaction)   │     │
│                                      │        │                 │     │
│                                      │  ┌─────▼──────────────┐   │     │
│                                      │  │ Engine Core (Rust) │   │     │
│                                      │  │ scheduler, cache,  │   │     │
│                                      │  │ admission, registry │  │     │
│                                      │  │ oMLX adapter (HTTP) │  │     │
│                                      │  └───┬─────┬──────────┘   │     │
│                                      │      │     │              │     │
│                                      │  ┌───▼──┐ ┌▼────────────┐│     │
│                                      │  │Family│ │Infra crate  ││     │
│                                      │  │crates│ │KV/RoPE/attn ││     │
│                                      │  └───┬──┘ └──────┬──────┘│     │
│                                      │      └──┬───────┘        │     │
│                                      │      ┌──▼─────────────┐   │     │
│                                      │      │ mlx-c FFI      │   │     │
│                                      │      │ (vendored)     │   │     │
│                                      │      └───┬────────────┘   │     │
│                                      │          ▼                │     │
│                                      │   MLX (statically linked) │     │
│                                      │          ▼                │     │
│                                      │   Metal GPU + unified mem │     │
│                                      └──────────────────────────┘     │
│                                                                      │
│  ┌───────────────┐   external, opt-in backend                         │
│  │ oMLX          │◀── local HTTP                                     │
│  └───────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 Component Design

#### Component: TypeScript Clients (CLI / TUI / TS SDK)
- **Responsibility:** Thin user-facing surface — argument parsing, terminal rendering, interactive flows, and SDK bindings. No policy decisions, no harness logic, no inference; every capability question goes to the daemon. The CLI is the V1 client surface; the TUI is a V1 SHOULD, cuttable to v1.1 per the descope ladder.
- **Technology:** TypeScript on the Node.js LTS current at Phase 0 (minimum 22 — Node 22 reaches EOL 2027-04-30, one month before v1.0; the pin is re-confirmed in Phase 8 and bumped if the pinned line is near EOL; decision recorded at Phase 0; Bun compatibility is not a V1 target); a JSON-RPC client library over the daemon's Unix domain socket; the same client library is published as the public TS SDK.
- **Interfaces:** Consumes the daemon local API (Appendix B.6); renders streaming events; renders approval prompts (approval decisions execute daemon-side).
- **Dependencies:** alloyd local API.

#### Component: Agent Supervisor (Rust)
- **Responsibility:** Own the agent loop — render model output into structured blocks, dispatch tool calls through the registry, enforce policy, and coordinate all harness subsystems.
- **Technology:** Rust; tokio async runtime.
- **Interfaces:** Receives structured requests from the daemon's local API (CLI/TUI/SDK) and automation engine; emits streaming events (text_delta, tool_call_started / tool_call_delta, generation_completed, runtime_warning, request_failed); calls the engine core's async API directly (in-process).
- **Dependencies:** Tool registry, context engine, policy engine, engine core.

#### Component: Tool Registry (Rust)
- **Responsibility:** Normalize builtin/plugin/MCP tools into one schema with policy metadata; validate arguments, gate approvals, execute, and store results as artifacts when oversized.
- **Technology:** Rust; JSON Schema validation.
- **Interfaces:** Tool registration API for plugins and MCP; execution pipeline (validate → approve → execute → store).
- **Dependencies:** Policy engine, artifact store, MCP client.

#### Component: MCP Client (Rust)
- **Responsibility:** Manage MCP server lifecycle, discovery, schema normalization, timeouts, cancellation, and result limits. Treats remote metadata as untrusted.
- **Technology:** Rust; official MCP protocol libraries.
- **Interfaces:** Feeds normalized tools to the tool registry.
- **Dependencies:** Tool registry, policy engine, secrets/keychain.

#### Component: Skill Loader (Rust)
- **Responsibility:** Discover skills (SKILL.md convention), expose name/description summaries, load full instructions into context on demand; skill scripts execute only through authorized tools.
- **Interfaces:** Discovery and load APIs used by the context builder.
- **Dependencies:** Context engine, tool registry (for scripts).

#### Component: Plugin Host (Rust)
- **Responsibility:** Run out-of-process plugins against a versioned lifecycle-hook interface; less-trusted code gets OS-level sandbox restrictions.
- **Interfaces:** V1 hook set (minimal): `tool.before_execute`, `tool.after_execute`. The full hook surface (session.opened, context.before/after_build, generation.before_start, compaction.before_commit, automation.triggered, session.closed) is designed but deferred until real plugin demand exists (post-v1). Hook payloads, ordering, and mutation rules for the V1 set are specified in the Phase 2 execution plan.
- **Dependencies:** OS process supervision, IPC (to plugins only).

#### Component: Context & Session Engine (Rust)
- **Responsibility:** Maintain the separation of durable history, rendered context, and KV cache; build versioned context snapshots; run exact token counting; enforce context admission; execute transactional compaction.
- **Technology:** Rust; SQLite for events; artifact files for large outputs; HF tokenizers crate for counting.
- **Interfaces:** Engine-core render_and_count; compaction transaction API; checkpoint format.
- **Dependencies:** Engine core.

#### Component: Automation Engine (Rust)
- **Responsibility:** Schedule/event triggers create durable jobs that run through the same harness and policy path as interactive requests; overlap policy, deadlines, retries, approval pauses, crash recovery.
- **Interfaces:** Job definitions (YAML), persistent job state, run creation.
- **Dependencies:** Agent supervisor, session store, policy engine.

#### Component: Engine Core (Rust, in-process)
- **Responsibility:** Model lifecycle (load/unload/evict), multi-session scheduling over shared weights, KV-cache management (retention, prefix reuse, idle policy — no disk persistence in V1; KV state is in-process only, §3.8), memory admission control, prefill chunking, and backend selection (native mlx-c engine vs oMLX adapter). Owns the unified-memory optimization program: zero-copy weight residency via MLX shared CPU/GPU arrays, Metal memory-limit/cache-limit configuration, allocator-cache reclamation under pressure, and compiled/lazy graphs to bound temporaries.
- **Technology:** Rust; direct ownership of mlx-c arrays/streams in the engine's memory domain.
- **Interfaces:** Async engine API: render_and_count, estimate_memory, generate (streaming), cancel, release_session_state, capabilities (contract in Appendix B.1).
- **Dependencies:** mlx-c FFI layer, model-family crates, oMLX adapter.

#### Component: mlx-c FFI Layer (Rust)
- **Responsibility:** Safe RAII bindings over mlx-c (arrays, streams, devices, closures, compile); vendored MLX/mlx-c sources built via build.rs + CMake and statically linked; pinned commits recorded (PINS.md pattern).
- **Technology:** Rust + bindgen; CMake.
- **Interfaces:** Idiomatic array/ops API for the infra crate; mlx_compile access for graph compilation of decode steps.
- **Dependencies:** mlx-c (vendored).
- **Decision point (resolved, v1.6):** generate an **own thin bindgen layer** over the vendored mlx-c pin — maximum control of the full stack, following the ironbark DR-0002 precedent; the layer is policy-free and low-level (arrays, devices, streams, closures, `mlx_compile`), regenerated on pin bumps with a coverage inventory that fails CI on drift. Recorded fallback: adopt or fork mlx-c-rust (complete MLX-C bridge, production-hardened) if own-layer maintenance proves too heavy; the infra-crate API isolates family crates from the binding layer either way.

#### Component: Shared Inference Infrastructure Crate (Rust)
- **Responsibility:** The reusable model-building blocks: KV cache (full/rotating, quantized where supported), RoPE, attention variants, quantized matmuls, sampling — shared by every family crate (OminiX-MLX pattern).
- **Interfaces:** Builder APIs consumed by family crates; capability flags surfaced to the engine core.
- **Dependencies:** mlx-c FFI layer.

#### Component: Model-Family Crates (Rust, one per architecture)
- **Responsibility:** Implement each architecture (Qwen3, Llama 3.x, Gemma 3, Qwen-VL, Whisper, TTS) over the infra crate; register against `architectures` values from config.json; declare per-family capabilities (context limits, cache modes, multimodal inputs).
- **Interfaces:** Family registry; weight loading from safetensors; forward/decode loops.
- **Dependencies:** Infra crate, mlx-c FFI layer.

#### Component: Model Registry & Validation (Rust + optional Python build tool)
- **Responsibility:** inspect/resolve/download model sources; verify artifact manifests and hashes; run numerical spot checks and the agent-conformance suite per model profile; manage profiles. Conversion recipes live in the optional Python build-time CLI (FR-6); the runtime only consumes finished artifacts.
- **Technology:** Rust; HF Hub API; safetensors; Python CLI wraps mlx-lm conversion (never at runtime).
- **Interfaces:** Artifact manifest format (JSON), validation reports.
- **Dependencies:** Engine core, artifact store.

#### Component: oMLX Backend Adapter (Rust)
- **Responsibility:** Present oMLX's local OpenAI/Anthropic-compatible API as a selectable backend per model profile; surface tiered-cache and batching behavior through the same capabilities contract.
- **Interfaces:** Same engine API as the native engine; local HTTP transport (the only inference-plane network client in the default trust boundary, opt-in; HF Hub downloads and MCP-over-HTTP are also outbound clients, enabled by user configuration).
- **Dependencies:** oMLX (external, user-installed).

#### Component: Session Store (Rust)
- **Responsibility:** SQLite-backed append-only event log, artifact storage, checkpoint storage, job state.
- **Interfaces:** Event append/read APIs; artifact reference resolution.
- **Dependencies:** None (leaf).

### 3.3 Data Model

Key entities:

```text
ModelArtifact (HF-style dir) ──verified_by──▶ ValidationReport
     │
     ├── has ──▶ Manifest (hashes: weights, tokenizer, template, recipe?)
     └── referenced_by ──▶ ModelProfile
                              │ (backend: native_mlx | omlx; family crate)

Session ──appends──▶ Events (user.message, assistant.message, tool.*,
               │              artifact.created, context.compacted, …)
               ├── has ──▶ ContextSnapshot (versioned render)
               ├── has ──▶ Checkpoints (compaction outputs)
               └── has ──▶ SessionState (native: in-process engine handle; oMLX backend: external reference; both carry the KV identity)

Tool ──governed_by──▶ ToolPolicy (effect, approval, timeout, max_output)
Skill (SKILL.md)      Plugin (hooks)      Automation ──▶ Job ──▶ Run
Artifact (large tool outputs, checkpoints, model dirs, reports)
```

- **Creation:** events append-only; artifacts created by tools, validation, compaction, download.
- **Mutation:** sessions are logically append-only; context snapshots and checkpoints are new versions, never in-place edits.
- **Archival/deletion:** per-session retention policy; artifacts garbage-collected when unreferenced.
- **Consistency:** strong for session state (single process, SQLite); cache state is advisory (correctness never depends on it — provable prefix match or rebuild).

### 3.4 API & Interface Design

**Engine core API** (in-process async Rust; the contract former v1.0 put on IPC now holds at the trait boundary — schemas in Appendix B):

```text
render_and_count(context_snapshot) → { token_count, render_hash, tokenizer_id }
estimate_memory(model_profile, token_count) → { reserved_bytes, fits, degradation_options[] }
generate(model_profile, context, opts) → stream of generation events
cancel(request_id)
release_session_state(session_state_id)
capabilities(artifact_id) → { task, family, cache_modes, context_limit_tokens, … }
```

**Tool result contract:**

```text
ToolResult = { call_id, content | artifact_ref, is_error: bool,
               error_kind: "protocol" | "execution" | "timeout" | "policy" }
```

**CLI surface (illustrative):**

```text
alloy chat --profile local-coder
alloy run "task..." --profile researcher
alloy models list | pull | verify | bench
alloy convert --recipe q4 <source>     # TS CLI delegates to the optional Python
                                       # converter (uv: uvx alloy-convert); clear error if absent
alloy sessions resume <id>
alloy skills list | show
alloy mcp add | list
alloy automation add | list | pause
alloy stats
```

**Compaction summarizer (design contract):**

- **Summarizer model:** the session's own model (no separate summarizer process or model in V1).
- **Prompt contract:** the summarization prompt instructs the model to emit a checkpoint conforming to the B.3 schema; the schema is validated structurally, never trusted from raw output.
- **Trigger threshold:** compaction is forced when rendered input + output reserve + safety margin exceeds the admitted context (FR-26), or when an upcoming tool result is projected to exceed the remaining budget.
- **Compact-vs-reject rule:** compact if the validated checkpoint restores ≥ 20% context headroom; otherwise reject the request with an explanation (never loop compaction).
- **Failure handling:** one retry with a stricter instruction; on second failure, fall back to the previous checkpoint (or no checkpoint) and reject the request with an explanation. A bad checkpoint is never silently committed.

**V1 builtin tool set (enumerated; default policies in B.2 format):**

| Tool ID | Effect | Default approval | Notes |
|---------|--------|------------------|-------|
| `fs.read` | read | never | path allowlist via policy |
| `fs.list` | read | never | |
| `fs.write` | write | policy | |
| `shell.run` | destructive | always | full policy gating, output-size limited |
| `http.fetch` | read | policy | network-capable, policy-limited |

**SKILL.md convention (structural definition):**

- Frontmatter: `name` (required, unique), `description` (required, one-line summary used at discovery), `version` (optional).
- Body: markdown instructions loaded into context on demand.
- Optional subdirectories: `references/` (files the skill may cite), `scripts/` (executable only via authorized tools, never auto-executed).
- Discovery paths: `~/.alloy/skills/`, `./.alloy/skills/` (project-local, takes precedence), plus plugin-provided skills registered through the tool registry.

### 3.5 Data Flow — Agent Turn (happy path)

```text
1. User input (CLI/TUI/automation) → Agent Supervisor
2. Context engine builds snapshot: policy + tools schemas + skills +
   compacted history + recent events + artifacts + request
3. Engine: render_and_count → admission check (context + memory)
   ├─ over budget → compaction transaction or reject with explanation
   └─ fits → proceed
4. Cache manager: token-prefix match? reuse / else prefill (chunked)
5. generate → stream events; parse tool calls from structured blocks
6. Tool registry: validate args → policy check → (approval?) → execute
   └─ oversized output → artifact store → reference into context
7. Tool result appended as event; loop to 2 until completion
8. Session state updated; KV cache kept hot per idle policy
   (steps 3-5 and 8 execute in the same process as 1-2 and 6-7)
```

### 3.6 Integration Points

- **Hugging Face Hub:** model download, source resolution (token via keychain); mlx-community pre-converted artifacts as the zero-conversion path.
- **MCP servers:** stdio and HTTP transports; user-configured.
- **oMLX:** OpenAI/Anthropic-compatible API, local HTTP, opt-in backend.
- **OS keychain:** credential storage.
- **mlx-c / MLX:** the inference foundation (vendored, pinned).
- **Optional Python conversion CLI:** build-time only, wraps mlx-lm tooling.

### 3.7 Security Architecture

- Rust daemon is local-only: it listens on a Unix domain socket (filesystem permissions enforce the single user) — no network listeners by default. Outbound connections: the oMLX adapter is an opt-in local HTTP client, and the telemetry client (FR-35) is the only always-on outbound connection (anonymous counts, TLS — see Appendix C); plugins and MCP servers run as separate sandboxed processes. TypeScript clients contain no policy or harness logic — a compromised client can only issue requests the policy engine still gates. Same-user vector (documented residual risk): any local process running as the user can connect to the UDS and drive policy-gated requests; the policy engine gates tool execution but cannot distinguish which local process issued a request.
- Approval semantics: `never` = no prompt; `always` = prompt for every call; `policy` = auto-approve read-effect tools, prompt for write/destructive. A pending approval blocks its session; automations under deadline eventually abort the run (approval-timeout policy per job). Headless approvals: pending approvals are durable and queued — surfaced on the next client connect, or via the (post-v1) notification hook; an automation run never self-approves.
- Session content at rest: plain SQLite and artifact files; at-rest protection relies on macOS FileVault (full-disk encryption) — no application-layer encryption in V1 (documented assumption).
- Tool policy engine is the single authorization point: effect classes (read/write/destructive), approval modes (never/always/policy), per-tool timeouts and output limits.
- MCP annotations untrusted; plugin code runs out-of-process with OS sandbox profiles for less-trusted extensions.
- Audit log for every tool intent/decision/result; automation runs under explicit permission scopes with approval pauses for side-effecting actions.
- Crash/kill of the process never silently re-executes external side effects — idempotency keys required for destructive tools where supported.
- Supply chain: vendored MLX/mlx-c pinned by commit with hashes recorded; model artifacts verified by manifest hash before load.

### 3.8 Resilience Design

- **Process model:** one process — a crash pauses inference and sessions resume from durable history (the KV cache is lost; correctness never depends on it). This is the accepted trade-off of the single-process design vs. v1.0's worker isolation.
- **Retries/backpressure:** generation and tool calls cancellable; MCP calls have deadlines; scheduler applies admission backpressure instead of OOM.
- **Cache failure modes:** KV cache is an optimization — any prefix mismatch invalidates reuse and falls back to re-prefill; correctness never depends on cache state.
- **Validation failures:** artifact never marked usable without a passing validation report; failed downloads/conversions leave existing artifacts untouched.
- **Degradation ladder under memory pressure:** evict idle cache → unload idle models → reduce concurrency → shrink prefill chunks → reject with explanation.
- **Graph-compilation safety:** `mlx_compile`'d decode paths must produce bit-identical outputs to eager paths in the conformance suite before being enabled per family.

### 3.9 Observability

- Structured logging (tracing crate) with per-session correlation IDs.
- Metrics: tokens/s, TTFT, cache hit rate, admission rejections, tool latencies, memory reservations vs. actuals (reconciled after load). The telemetry client (FR-35) reports these same dimensions in aggregate as anonymous counts, per the published event schema.
- `alloy doctor`: environment report (chip, RAM, wired-limit, mlx-c/MLX versions from PINS, disk) for issue triage.
- Benchmark harness publishes scenario tables (vs. llama.cpp/Ollama/oMLX) with each release.

### 3.10 Infrastructure & Deployment

- **Distribution:** Homebrew tap (primary) and GitHub Releases shipping the Rust daemon binary (MLX Metal kernels dominate size, ~150 MB budget); the TypeScript CLI/TUI ships as an npm package (with the TS SDK) that detects an existing daemon or bootstraps one by downloading the release binary from GitHub Releases with checksum verification (CLI/daemon compatibility per the B.6 `api_version` policy); `cargo install` from source for contributors. The optional conversion CLI is a separate download via uv (`uvx alloy-convert`) for those who need raw-checkpoint conversion. oMLX optional via its own tap.
- **Environments:** dev (local Mac), CI (Linux: unit + protocol tests; self-hosted Mac runner: integration + model conformance + benchmarks).
- **CI/CD:** GitHub Actions; Linux fast lane on every PR (no MLX needed — engine mocked); self-hosted Mac lane on merge-to-main and release tags; artifacts (benchmark tables, validation reports) attached to releases.
- **Rollback:** versioned artifacts and pinned dependency sets make any release reproducible; sessions are forward-compatible via event-log versioning.

---

## 4. Implementation Plan

### 4.1 Build Phases

**Target repository layout** (files are created per phase; each phase begins with a one-page execution plan listing its exact file set and named test cases, reviewed against the exit criteria):

```text
repo/
  crates/
    alloyd/            # daemon binary: API server, wiring
    harness-core/      # agent loop, context engine, session store, policy
    tool-registry/     # registry + builtin tools
    mcp-client/
    engine/            # engine core: scheduler, cache manager, admission
    mlx-ffi/           # vendored mlx-c bindings (build.rs + CMake, PINS.md)
    inference-infra/   # KV cache, RoPE, attention, sampling
    families/          # one crate per architecture
      qwen3/ llama3/ gemma3/ qwen-vl/ whisper/ tts/
    om-adapter/        # oMLX backend adapter (SHOULD — conditional)
  ts/
    cli/               # TypeScript CLI (Node LTS per Phase 0 pin)
    sdk/               # published TS SDK (JSON-RPC client library)
  tools/convert/       # optional Python conversion CLI (build-time only)
  docs/specs/
```

#### Phase 0: Decision spike (week 1 of M1, before any engine code)
- **Goal:** Validate and execute the Phase 0 decisions (resolved in the spec at v1.6/v1.7, 2026-11-03) — zero open decisions blocking Phase 1.
- **Scope:** (a) Q2 build spike: generate the own thin bindgen layer over the vendored mlx-c v0.8.x pin (build.rs + CMake, PINS.md) and verify the coverage inventory (all public headers → Rust declarations, CI fails on inventory drift across a pin bump); if the own layer proves heavier than ~1.5 weeks of build work, exercise the recorded fallback (adopt/fork mlx-c-rust) rather than re-opening the decision; (b) Q1 name reservation: create the GitHub repo, reserve crates.io `alloyd`, npm `alloyml`, PyPI `alloy-convert` (the uvx conversion CLI, Q3), and the Homebrew tap (formula `alloy`); config dir `~/.alloy/`; (c) confirm FR-9 (minijinja) and TS runtime pins (current Node.js LTS at Phase 0, minimum 22 — Node 22 reaches EOL 2027-04-30, before v1.0); (d) Q6 publication: commit the frozen benchmark corpus (adopted ex-ante — tiers: 36 GB hardware + 8 GB simulated budget; models: Qwen3-4B/14B 4-bit, Llama-3.1-8B 4-bit and 8-bit, Gemma-3-12B 4-bit, all mlx-community artifacts, same model+quant on llama.cpp/Ollama/oMLX; scenario classes: code Q&A on a fixed repo snapshot (50 questions), agentic tool loops (20 multi-turn tasks), long-context summarization (8k and 32k documents), prefill-heavy large prompts; metrics: decode tokens/s, TTFT p50/p95, peak memory, agent-loop overhead) to `bench/corpus/` before any performance-sensitive work.
- **Exit criteria:** bindgen layer builds against the vendored pin with a passing coverage inventory (or the fallback exercised and recorded); names reserved; `bench/corpus/` published in-repo; Phase 1 execution plan written.

#### Phase 1: Vertical slice (mlx-c FFI + first family + harness skeleton)
- **Goal:** One model family, one session, streamed chat end-to-end from TS client through daemon to engine.
- **Scope:** mlx-c FFI layer with vendored build (build.rs + CMake, PINS.md); shared infra crate (KV cache, RoPE, attention, sampling); Qwen3 family crate; HF tokenizers + minijinja chat-template rendering; agent supervisor skeleton; SQLite session store; daemon local JSON-RPC API (Appendix B.6); thin TypeScript CLI (`alloy chat`); prefix-cache stub (the real basic single-session prefix cache lands in Phase 3, where its tests live — Phase 1 needs only the interface); telemetry client (FR-35): in-memory anonymous-count event collection with HTTPS/TLS flush to the project endpoint (offline drop, failure-isolated) and the initial event schema.
- **Exit criteria:** `alloy chat` (TS client ⇄ daemon API ⇄ engine) streams tokens from a quantized Qwen3 model loaded from an mlx-community artifact, in-process; session survives process restart; engine API traits documented and versioned; vendored-build reproducibility verified twice; telemetry emits anonymous counts per the initial event schema.

#### Phase 2: Tools and policy
- **Goal:** Canonical tool loop with validation and approval.
- **Scope:** Tool registry, structured tool-call parsing per model family, argument validation, policy engine (effect classes, approvals), cancellation, artifact store for oversized outputs; plugin host with the minimal V1 hook set (FR-24: `tool.before_execute` / `tool.after_execute`).
- **Exit criteria:** A model can call a builtin tool (file read, shell with policy), with approval gating on destructive effects; malformed tool calls rejected, never executed; a plugin receives before/after-execute events for a tool call and can veto or annotate it.

#### Phase 3: Context, compaction, exact counting
- **Goal:** Correct context accounting and transactional compaction.
- **Scope:** render_and_count, context snapshots, token budgets with output reserve, compaction checkpoints, basic single-session prefix cache with cache-prefix invalidation on compaction (moved here from Phase 1 so the cache ships with its tests); second family crate (Llama 3.x) to prove the infra crate generalizes.
- **Exit criteria:** A long session that would overflow context compacts to a checkpoint and continues without event-history loss; cache reuse never serves a stale prompt (property-based test); Llama runs through the same pipeline as Qwen.

#### Phase 4: MCP and skills
- **Goal:** External tool ecosystems and task guidance.
- **Scope:** MCP client (stdio + HTTP), tool normalization, skill discovery/loading, keychain secrets.
- **Exit criteria:** An MCP server's tools appear in the registry behind policy; a skill's instructions load into context on demand and its scripts execute only via authorized tools.

#### Phase 5: Scheduling, caching, admission
- **Goal:** Multi-session shared-model operation within memory budgets.
- **Scope:** Engine scheduler (prefill/decode, session concurrency), KV-cache manager (generalizing Phase 3's basic prefix cache to retention modes, multi-session prefix sharing, idle policy), memory admission controller, prefill chunking control; `mlx_compile`d decode path behind conformance gates — enabled per family only if eager-mode benchmarks show a > 10% decode-speed gap (conditional feature, not a default assumption); speculative decoding with draft models (FR-12, committed v1.7 — default pairing Qwen3-0.6B drafting Qwen3-4B, profile-declared) behind per-family conformance gates; `alloy doctor` (FR-19): environment report with per-process Metal memory limits and wired-limit guidance.
- **Exit criteria:** Concurrent sessions share one loaded model; an over-budget request is rejected with an explanation rather than OOM-crashing; measured memory within 20% of estimates (reconciled and calibrated); compiled decode is bit-identical to eager on enabled families; at least one family ships draft-enabled speculative decoding with its conformance suite passing (SHOULD-level: cuttable per the descope ladder without blocking v1.0).

#### Phase 6: Automations and oMLX backend
- **Goal:** Durable unattended runs; second backend.
- **Scope:** Automation engine (schedule/event triggers, jobs, overlap policy, approval pauses, crash recovery), oMLX adapter behind the capabilities contract. The oMLX adapter is FR-5 SHOULD — if capacity is short, it is the first Phase 6 item to drop without blocking v1.0. The Phase 6 execution plan delivers the job-definition YAML schema (trigger, overlap policy, deadline, retry policy, approval timeout, permission scope — FR-29) and pins the V1 trigger vocabulary: `schedule` (cron, with catch-up-on-restart) and `event` (V1: `session.completed` only; extension post-v1).
- **Exit criteria:** A scheduled job survives a daemon restart without duplicate side effects (MUST); if the adapter is in scope, a model profile selects oMLX and uses its tiered cache through the same session semantics (SHOULD-level demo).

#### Phase 7: Families, conversion, multimodal
- **Goal:** Full model coverage and the conversion pipeline.
- **Scope:** Remaining text families (Gemma 3) and the multimodal set (Qwen-VL with OCR, Whisper STT, TTS: Kokoro-82M per Q7) per FR-4b; optional Python conversion CLI with quantization recipes and manifest output; agent-conformance suite per profile; embeddings/rerankers if time permits (FR-10 SHOULD — in scope; first-cut candidate under the R1 contingency).
- **Exit criteria:** All FR-4a text families pass their conformance suites (MUST); multimodal families per FR-4b SHOULD — image description, OCR, and voice-in/voice-out work through model profiles where implemented, evaluated against the FR-4b per-modality conformance criteria (M7 gate), with any deferrals recorded per the FR-4b TTS rule; a raw checkpoint converts via the build-time CLI to a verified MLX artifact.

#### Phase 8: Benchmarks, docs, v1 release
- **Goal:** Performance claims proven; project shippable.
- **Scope:** Benchmark matrix vs. llama.cpp/Ollama/oMLX on the self-hosted Mac tier set; capacity-model tuning (Appendix D); docs, contributor guide (including "how to add a model family" — the key contribution path), Homebrew tap, release automation; telemetry privacy policy finalized + event schema published and versioned (FR-35 — the emission client has shipped since Phase 1; the policy and initial schema precede the first tagged 0.x pre-release that emits, and Phase 8 reviews and finalizes them).
- **Exit criteria:** Published benchmark tables meet the NFR targets (≥ 70% meet-or-exceed scenarios, oMLX included); docs cover install, extension (tools/skills/plugins/MCP/families), and architecture; v1.0 tagged; telemetry privacy policy finalized and event schema published (FR-35); daemon binary ≤ 150 MB (TS CLI ships separately via npm).

### 4.2 Testing Strategy

- **Unit (Linux CI):** Rust crates, policy engine, context builder, template rendering, engine API against mocks — pure logic. Engine/infra-crate unit tests that need real MLX run against CPU-backend MLX on Linux (mlx-c is portable); Metal-specific tests belong to the Mac lane.
- **Integration (self-hosted Mac):** full process + real models: tool-loop conformance, compaction, cache correctness, automation recovery, admission/chaos tests on simulated 8 GB and 36 GB budgets (both are first-class CI targets per Appendix D).
- **Model conformance suite:** per model profile — tool-call format adherence, argument-schema compliance, stop conditions, long-context behavior, compaction quality; compiled-vs-eager decode bit-identity; a model is only "supported" when its suite passes.
- **Cache correctness:** property-based tests proving prefix reuse ⇒ identical outputs vs. no-cache runs.
- **Performance:** reproducible benchmark scenarios (fixed prompts, seeds, hardware reported), run per release; comparisons include oMLX. Ex-ante criteria: the scenario corpus was adopted and frozen at spec time (v1.6 — Q6: 36 GB hardware + 8 GB simulated tiers, the model/scenario/metric set recorded in §8) and is published in-repo at `bench/corpus/` in Phase 0 before performance-sensitive work; Phase 5 performance work and the Phase 8 benchmark matrix both run against this frozen corpus, preventing post-hoc scenario selection.
- **Security:** policy bypass attempts (malformed tool calls, MCP annotation spoofing, plugin sandbox escape) as regression tests.
- **Telemetry:** the FR-35 client is tested against the published event schema — field-level anonymity invariants (no prompts, file paths, or model identifiers), offline drop semantics (no buffering), flush cadence, and failure isolation (telemetry errors never surface in sessions); a schema change without a version bump fails CI.
- **Build reproducibility:** vendored mlx-c pin produces identical binaries from clean CI runs (hash check).

### 4.3 Rollout Strategy

- Open-source from day one (Apache-2.0), `main` always buildable; tagged pre-releases (0.x) as phases complete; v1.0 at end of month 7 (2027-05-31).
- No feature flags needed (local app); breaking engine API changes gated by trait versioning.
- Rollback = previous release tag; sessions forward-compatible via versioned event log.

### 4.4 Operational Readiness

- `alloy doctor` environment report; runbook for common failures (wired-memory limits, model download issues, MCP server failures, vendored-build problems).
- Benchmark and validation reports published with releases.
- Issue templates capturing hardware tier + `alloy doctor` output.

---

## 5. Milestones

| Milestone | Goal | Exit Criteria | Target Date | Owner |
|-----------|------|---------------|-------------|-------|
| M1 | Vertical slice | Phase 1 exit criteria | 2026-12-03 | Solo |
| M2 | Tool loop + policy | Phase 2 exit criteria | 2027-01-07 | Solo |
| M3 | Context + compaction + 2nd family | Phase 3 exit criteria | 2027-02-04 | Solo |
| M4 | MCP + skills | Phase 4 exit criteria | 2027-02-25 | Solo |
| M5 | Scheduling + admission | Phase 5 exit criteria | 2027-03-18 | Solo |
| M6 | Automations + oMLX | Phase 6 exit criteria | 2027-04-08 | Solo |
| M7 | Families + conversion + multimodal | Phase 7 exit criteria | 2027-04-29 | Solo |
| M8 | Benchmarks + v1.0 | Phase 8 exit criteria | 2027-05-31 | Solo |

Slack and descope policy: each milestone carries ~1 week of implied slack via phase overlap (the dependency graph's M6/M7 parallelism); if a milestone slips more than one week, apply descope in this order — (1) drop Phase 6's oMLX adapter (FR-5 SHOULD), (2) drop speculative decoding (FR-12 SHOULD — a performance optimization; cut before user-facing capability), (3) defer FR-4b multimodal items to v1.1 per the R1 contingency (TTS first), (4) defer FR-10 embeddings first-cut, (5) defer the TUI (FR-30b SHOULD) — before touching any MUST item. v1.0 date moves only if MUST items are at risk.

### Dependency Graph

```text
M1 ──▶ M2 ──▶ M3 ──▶ M4 ──▶ M5 ──▶ M6 ──▶ M8
                │               │       ▲
                └──────▶ M7 ◀──┘       │
                         └────────────┘
(M7 families/conversion/multimodal depends on M3 context engine and M5 admission;
 it can proceed in parallel with M6 from M5 onward)
```

---

## 6. Success Criteria

### 6.1 Launch Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Daily driver | Author uses it as primary local agent harness for ≥ 80% of daily coding/research agent tasks within 4 weeks of v1 | Self-reported usage log, with an objective proxy: session counts and task mix recorded in the daemon's own SQLite event log over the final 4 weeks (queryable via `alloy stats`), so the claim is checkable without telemetry |
| Community growth | ≥ 200 GitHub stars and ≥ 5 external contributors (merged PRs) within 3 months of v1 | GitHub metrics |
| Model coverage | FR-4a text families passing the conformance suite (Qwen3 text+Coder, Llama 3.x, Gemma 3) — MUST; FR-4b multimodal (Qwen-VL, Whisper, TTS) — SHOULD with recorded deferrals | CI conformance matrix |
| Performance | Benchmark NFR targets met (≥ 70% meet-or-exceed vs. llama.cpp/Ollama; oMLX in all tables) | Release benchmark reports |
| Stability | No P0 correctness bugs (wrong-context generation, event loss) in the first month | Issue tracker |

Note: if the R2 contingency (§7) is exercised, the performance success bar is the reframed NFR it records (within 10% on 90% of scenarios) — §2.2, §6.1, and the R2 contingency co-move; they are never two live bars at once.

### 6.2 Ongoing Monitoring

- Release cadence: monthly minor releases with refreshed benchmark tables.
- Review cadence: monthly self-review of milestone burndown and issue triage; quarterly roadmap re-forecast.
- Community signals: issues/PRs triaged within 7 days.

### 6.3 Remediation Triggers

- Any cache-correctness or event-loss bug: hotfix release within 72 hours, conformance test added.
- Benchmark regression > 20% on any tier vs. prior release: investigate before next release.
- A model family failing its conformance suite after an mlx-c/MLX bump: mark profile unsupported (honest failure) until fixed.

---

## 7. Risks

| ID | Risk | Impact | Likelihood | Mitigation | Contingency |
|----|------|--------|-----------|------------|-------------|
| R1 | Scope too large for one developer in ~7 months — now includes owning model implementations (per-family crates) plus conversion and multimodal | High | High | Proven precedent (mlex, OminiX-MLX) provides the pattern and reference code; infra crate amortizes family cost; mlx-community artifacts skip conversion; oMLX backend covers broad architecture needs | Slip FR-4b multimodal items to v1.1 (TTS first, per the FR-4b deferral rule — a SHOULD-level deferral, not a MUST violation); if capacity is still short, cut FR-4a text families (Gemma 3 first) to v1.1 — a conscious MUST-level scope cut recorded as such, never silent; adopt family crates from community |
| R2 | "Beat llama.cpp/Ollama" performance bar not met on enough scenarios | Medium | Medium | Scoped honestly (≥ 70% scenarios, published tables); direct in-process control avoids IPC overhead; optimize where MLX is structurally advantaged (prefill, unified memory); oMLX backend gives users the fastest MLX path where the native engine lags | Reframe NFR as "within 10% on 90% of scenarios" and lead with harness features; document losing scenarios |
| R3 | mlx-c API churn or maturity gaps vs. the Python API (younger surface; some features arrive late or differ) | High | Medium | Pin vendored commits (PINS.md); conformance suite catches regressions per family; the own bindgen layer is regenerated on pin bump, so new MLX-C surface is callable immediately (coverage inventory fails CI on drift); mlx-c is Apple-official and Swift-aligned (same path as MLX Swift) | Hold pin at last-good version; report upstream; gate affected features off by capability registry |
| R4 | Owning the binding layer becomes a maintenance liability (~bindgen upkeep across MLX-C releases; the cost ironbark accepted at ~26k lines for a full wrapper) | Medium | Medium | The own layer is thin and policy-free (raw ABI + RAII handles only — the engine crate owns semantics); regeneration is mechanical (bindgen + coverage inventory); the infra-crate API isolates family crates from the binding layer | Adopt or fork mlx-c-rust (complete, production-hardened bridge) — a drop-in replacement at the same boundary; the decision is recorded and reversible |
| R5 | Small-model tool-call quality too weak for reliable agent loops on 8 GB Macs | Medium | Medium | Agent-conformance suite makes quality measurable per model; recommend per-tier model profiles; structured output/repair passes | Document minimum viable models per tier; on 8 GB Macs position harness for lighter tasks |
| R6 | Apple wired-memory limits cause load failures on large models despite admission control | Medium | Medium | Capacity model (Appendix D) with conservative headroom; admission controller + degradation ladder; `alloy doctor` detects limits | Auto-adjust default wired-limit guidance; document per-tier max model classes |
| R7 | Security incident via plugins/MCP/automation (local tool running destructive commands) | High | Low | Policy engine single authorization point; sandboxed plugins; approval gates on destructive effects; audit log | Disable third-party plugins by default in a hotfix; audit-log forensics; responsible disclosure process |
| R8 | Single-process design: an engine panic or MLX crash takes down sessions mid-run | Medium | Medium | Panic boundaries (`catch_unwind` at the engine API edge) contain Rust panics only — native crashes inside MLX's C++/Metal core (segfault, abort) cannot be caught in-process and remain fatal; durable event log makes resume cheap; cache loss is acceptable by design; conformance suite drives panic-free families | Split the engine into a child process post-v1 if native crashes prove frequent (the engine API trait boundary makes this a transport change, not a redesign). Do not rely on the panic boundary for native crashes |
| R9 | Community adoption stalls (another project ships the same composition first) | Low | Medium | Monthly releases, benchmark tables, and docs as differentiation; dogfooding keeps the project honest; "add a family crate" is an attractive contribution path | Merge/compose rather than compete: the oMLX backend path means the harness survives even if the inference layer is commoditized |
| R10 | oMLX upstream churn or abandonment (external dependency) | Medium | Medium | FR-5 is SHOULD with a promotion trigger, not a MUST; adapter targets oMLX's stable OpenAI/Anthropic-compatible API surface, tested against a pinned oMLX release per cycle; the capabilities contract isolates the adapter | Freeze the supported oMLX version (documented pin); if the project dies, the adapter is deprecated gracefully and the native engine path is unaffected — the system never depends on oMLX for core operation |
| R11 | Always-on telemetry rejected by privacy-sensitive users (community backlash, fork-to-remove); operating the collection endpoint is a solo-dev obligation (cost, retention, breach exposure of collected counts) | Medium | Medium | Anonymous counts only with a published event schema and privacy policy; HTTPS/TLS; offline drop semantics (no buffering, nothing retained on device); minimal endpoint footprint (aggregate counts, short retention) | If backlash is material post-v1, revisit the no-opt-out decision as a recorded v1.x decision (e.g., a documented opt-out) — never silent; the anonymity invariants keep the collected data low-value to attackers |

---

## 8. Decisions (Resolved Questions)

| # | Question | Owner | Due Date |
|---|----------|-------|----------|
| 1 | **RESOLVED (v1.6, 2026-11-03):** project named **alloy** — binaries `alloy` (TS CLI) / `alloyd` (Rust daemon); crates.io `alloyd` and npm `alloyml` verified available (bare `alloy` is taken on both crates.io and npm); Homebrew tap formula `alloy`; config dir `~/.alloy/`; Phase 0 reserves the names | User | Resolved |
| 2 | **RESOLVED (v1.6, 2026-11-03):** generate an **own thin bindgen layer** over the vendored mlx-c pin (v0.8.x) — ironbark DR-0002 precedent, maximum control; coverage inventory fails CI on pin-bump drift; recorded fallback: adopt/fork mlx-c-rust if own-layer upkeep proves too heavy; Phase 0 validates with a build spike | User | Resolved |
| 3 | **RESOLVED (v1.7, 2026-11-03):** the conversion CLI ships as an optional **uv** tool (`uvx alloy-convert`); mlx-community artifacts remain the primary zero-conversion path; the CLI is documented but never required | User | Resolved |
| 4 | **RESOLVED (v1.7, 2026-11-03):** telemetry is **required and always-on** (not opt-in; no opt-out in V1 — revisit post-v1 per R11) — FR-35: anonymous usage events (version, hardware tier, session/feature counts, plus aggregate §3.9 performance dimensions) to the project endpoint; the emission client ships in Phase 1 and the privacy policy + event schema precede the first tagged 0.x pre-release that emits (Phase 8 finalizes) | User | Resolved |
| 5 | **RESOLVED (v1.7, 2026-11-03):** speculative decoding **committed to shipping** — FR-12 promoted COULD → SHOULD, scheduled into Phase 5 scope and exit criteria; a family is draft-enabled only when its conformance suite passes with drafting enabled | User | Resolved |
| 6 | **RESOLVED (v1.6, 2026-11-03):** benchmark corpus adopted ex-ante — tiers: 36 GB (hardware) + 8 GB (simulated budget); models: Qwen3-4B/14B 4-bit, Llama-3.1-8B 4-bit and 8-bit, Gemma-3-12B 4-bit (all mlx-community artifacts; same model+quant on llama.cpp/Ollama/oMLX); scenario classes: code Q&A on a fixed repo snapshot (50 questions), agentic tool loops (20 multi-turn tasks), long-context summarization (8k/32k documents), prefill-heavy large prompts; metrics: decode tokens/s, TTFT p50/p95, peak memory, agent-loop overhead; published in-repo at `bench/corpus/` in Phase 0 and frozen | User | Resolved |
| 7 | **RESOLVED (v1.7, 2026-11-03):** **Kokoro-82M** is the V1 TTS family — mlx-community artifacts (Kokoro-82M-bf16/8bit/4bit), 82M params fits every tier, Apache-2.0 weights, 24 kHz output; a native non-Python MLX implementation is proven (kokoro-swift), so the Rust family crate is feasible; misaki G2P phonemization is part of the family crate's pipeline; the M7 conformance gate and the FR-4b deferral rule are unchanged | User | Resolved |

---

## 9. Appendices

### Appendix A: Glossary

| Term | Definition |
|------|-----------|
| Unified memory | Apple Silicon architecture where CPU and GPU share one physical memory pool |
| MLX | Apple's open-source array framework optimized for Apple Silicon GPUs (C++ core) |
| mlx-c | Apple's official C API for MLX — the bridge other languages bind to (used by MLX Swift and this project) |
| MLX-LM | Apple's Python package for running LLMs on MLX — used here only as the optional build-time conversion tool |
| HF-style model directory | Hugging Face model layout: config.json + safetensors shards + tokenizer files; the engine's canonical artifact |
| mlx-community | Hugging Face org publishing pre-converted MLX model artifacts — the zero-conversion path |
| Family crate | Rust crate implementing one model architecture over the shared infra crate |
| Infra crate | Shared inference-infrastructure crate: KV cache, RoPE, attention, quantized matmuls, sampling |
| oMLX | Apache-2.0 MLX-native inference server: continuous batching, paged tiered KV cache (RAM hot / SSD cold), multi-model serving |
| Alloy | Project name (v1.6): the harness + engine as a whole; binaries `alloy` (TS CLI) / `alloyd` (Rust daemon) |
| alloyd | The Rust daemon: harness + engine in one process; serves the local JSON-RPC API used by TS clients |
| mlex / OminiX-MLX | Precedent projects proving pure-Rust LLM runtimes on mlx-c (vendored builds, per-family crates) |
| KV cache | Stored key/value tensors for previously processed tokens, enabling incremental decoding |
| Prefix reuse | Reusing KV state when a new request's exact token prefix matches a cached one |
| Paged KV cache / CoW | Block-based cache management with copy-on-write, enabling sharing across sessions |
| Rotating cache | KV retention mode that keeps only the most recent N tokens (bounds memory, loses older context) |
| Prefill | Processing the input prompt (before first output token); chunked prefill trades speed for peak memory |
| mlx_compile | MLX graph compilation API exposed through mlx-c; used for compiled decode paths behind conformance gates |
| Compaction | Transactional summarization of older session history into a structured checkpoint to fit context |
| Context snapshot | The versioned, exact rendered prompt sent to the model (template + tools + skills + history) |
| Model profile | Named configuration binding an artifact to a backend, context/cache/prefill/agent settings |
| Artifact manifest | Hash-recorded metadata (weights, tokenizer, template, recipe) for a model artifact |
| Conformance suite | Per-model-profile tests for tool-call format, context behavior, compiled-decode identity, and compaction quality |
| MoSCoW | Prioritization: Must / Should / Could / Won't |
| MCP | Model Context Protocol — an open protocol connecting models to tools and data sources |
| Admission control | Pre-flight reservation of memory (weights + caches + temporaries + headroom) before load/session |
| Wired memory | macOS kernel memory that cannot be paged out; MLX allocations count against a per-process limit |
| TTFT | Time to first token |
| Draft model / speculative decoding | Small model proposes tokens; large model verifies — accelerates decode (FR-12, committed for v1.0) |
| Kokoro-82M | V1 TTS family (decision, v1.7 — Q7): 82M-param TTS, MLX artifacts at mlx-community/Kokoro-82M (bf16/8bit/4bit), Apache-2.0, 24 kHz output; misaki G2P phonemization in the family crate |

### Appendix B: API Contracts

#### B.1 Engine core API (in-process async Rust traits; the versioned contract between harness and engine)

The engine is a trait implementation behind an async handle; the same trait is implemented by the oMLX HTTP adapter. Wire compatibility is replaced by trait-version discipline (breaking changes bump the engine API version).

```text
// Harness → Engine (async, in-process)
render_and_count { context_snapshot } → { token_count, render_hash, tokenizer_id }
estimate_memory   { model_profile, token_count } → { reserved_bytes, fits, degradation_options[] }
generate         { model_profile, context_snapshot, opts } → stream:
                   text_delta | tool_call_delta | generation_completed { usage } |
                   runtime_warning { code, message } | request_failed { error_kind, message }
cancel           { request_id }
release_session_state { session_state_id }
capabilities      { artifact_id } → { task, family, cache_modes[], context_limit_tokens,
                                     supports_streaming, supports_prefix_reuse, … }

// Session state identity (cache-reuse key)
SessionStateId = hash(artifact_id + family_impl_version + lora_id + token_prefix +
                      cache_representation + engine_api_version)

// Hash specification: BLAKE3 over a canonical serialization.
// Canonical serialization: each field is length-prefixed (u64 LE) and UTF-8
// encoded, fields concatenated in the order listed above; token_prefix is the
// token-id sequence rendered as its exact tokenizer byte stream. The hash
// algorithm (BLAKE3) and serialization rules are pinned — they are
// cache-identity-critical: a session resuming on a daemon that hashes
// differently must NOT reuse another daemon's KV state.
// cache_representation: a stable string naming the KV layout semantics that
// affect reuse (cache mode + block size + quantization of the cache), defined
// per family crate and bumped with family_impl_version on layout changes.
```

#### B.2 Tool definition and result

```json
{
  "id": "mcp.github.search_issues",
  "version": "1.0.0",
  "description": "Search issues in a GitHub repository",
  "inputSchema": { "type": "object", "properties": { "repo": { "type": "string" } } },
  "origin": { "kind": "mcp", "source_id": "github" },
  "policy": {
    "effect": "read",
    "approval": "policy",
    "timeout_ms": 15000,
    "max_output_bytes": 1048576
  }
}
```

```json
{
  "call_id": "tc_01",
  "content": { "type": "text", "text": "3 issues found" },
  "is_error": false
}
```

Error kinds: `protocol | execution | timeout | policy`.

#### B.3 Compaction checkpoint format

```yaml
checkpoint_version: 1
goal: "Refactor the auth module"
constraints: ["No API changes", "Rust 2024 edition"]
decisions: ["Use trait objects for backends"]
completed_work: ["Auth trait extracted"]
open_tasks: ["Migrate middleware"]
artifact_references: ["art:12", "art:14"]
critical_identifiers: ["PR #42", "spec-1"]
failures_and_lessons: ["Bearer parsing must handle whitespace"]
next_action: "Continue middleware migration"
```

#### B.4 Event log (SQLite, append-only)

```text
-- Schema v1 (see versioning below)
CREATE TABLE events (
  id            INTEGER PRIMARY KEY,   -- rowid, monotonic
  session_id    TEXT NOT NULL,
  seq           INTEGER NOT NULL,     -- per-session sequence, gapless
  ts            INTEGER NOT NULL,    -- unix epoch ms
  kind          TEXT NOT NULL,       -- closed enum below
  payload_json  TEXT NOT NULL,       -- schema per kind
  artifact_refs TEXT NOT NULL DEFAULT '[]'  -- JSON array
);
CREATE INDEX idx_events_session_seq ON events(session_id, seq);

-- Versioning: a single-row meta table (schema_version, spec_version) is
-- bumped on any DDL or payload-schema change; migrations are forward-only
-- (old events are never rewritten); readers reject logs with a newer
-- schema_version than they understand.
kinds: session.created, user.message, assistant.message, tool.requested,
       tool.approved, tool.started, tool.completed, skill.loaded,
       artifact.created, context.compacted, model.changed,
       run.started, run.paused, run.completed
-- Session/run lifecycle status (paused, active) lives in a separate
-- sessions table (schema pinned in the Phase 1 execution plan); the
-- enum above is the append-only event stream only.
```

#### B.5 Family crate registration

```rust
/// Every model family implements this and registers by config architectures.
/// `impl_version` is the B.1 `family_impl_version` used in the cache-identity key —
/// bump it on any change to the family's numerical behavior or tensor layout.
pub trait ModelFamily {
    fn architectures(&self) -> &[&str];          // e.g. "Qwen3ForCausalLM"
    fn impl_version(&self) -> Semver;
    fn load(&self, dir: &ModelDir) -> Result<LoadedModel>;
    fn capabilities(&self) -> FamilyCapabilities; // context limit, cache modes,
                                                  // modalities, quantization support
}
```

#### B.6 Daemon local API (TS clients ⇄ alloyd)

Unix domain socket, JSON-RPC 2.0, versioned via `api_version` on every request. The same surface is the public TS SDK. Approval flows round-trip through the daemon — clients render prompts, never decide.

```text
// Client → daemon (JSON-RPC methods)
session.create { profile } → { session_id }
session.send { session_id, content } → stream: text_delta | tool_call_delta |
  tool_call_started | tool_result { call_id, is_error, summary } |
  approval_requested { call_id, tool_id, summary } | generation_completed |
  runtime_warning | request_failed
session.approve { session_id, call_id, allow } → tool proceeds / aborts
session.pause { session_id }    // cancels any in-flight generation first; durable events stay
session.resume { session_id }    // continues a paused live session
sessions.resume { session_id }   // reloads a historical session's events after a daemon restart
                              // (distinct from pause/resume: reconstructs state, re-renders context)
models.list | models.pull { source } | models.verify { artifact_id }
models.capabilities { artifact_id } → { … B.1 capabilities … }
skills.list → [{ name, description, source }] | skills.show { name } → SKILL.md content
mcp.add { name, transport, command | url } | mcp.list → [{ name, status, tool_count }]
automation.add { name, trigger, scope, profile } | automation.list | automation.pause { job_id }
daemon.status → { models_resident, memory_reserved, sessions_active }
daemon.doctor → { chip, ram_bytes, wired_limit, mlx_version, pins_hash }
daemon.stats { window } → { session_count, task_mix }   // backs `alloy stats` (§6.1)

// Streaming is server-push over the socket (notify-style JSON-RPC messages).
// Full JSON schemas per method, the error-code registry, and the api_version
// policy (compatibility window) are specified in the Phase 1 execution plan
// when the daemon API is first implemented — this appendix pins the method
// surface and streaming event names.
```

### Appendix C: Security Threat Model

#### Assets

local files, shell execution, credentials (keychain, HF tokens), session content, automation side effects (emails, commits, purchases).

#### Trust Boundaries

```text
Untrusted:  model output, MCP server metadata, skill/plugin code,
            downloaded model files
Trusted:    harness + engine core (one process, Rust), policy engine,
            user approvals
Boundary:   tool registry (single authorization point),
           plugin sandbox (OS-level, separate process),
           MCP normalization layer, vendored-build pin verification
```

#### Key Threats and Mitigations

| Threat | Vector | Mitigation |
|--------|--------|-----------|
| Prompt-injected tool invocation | Model output contains fake tool calls | Tools execute only from parsed structured blocks via the registry; never from raw text |
| MCP annotation spoofing | Server marks a destructive tool as "read-only" | Remote metadata describes; local policy authorizes — annotations never drive decisions |
| Malicious skill/plugin code | SKILL.md scripts, plugin hooks | Scripts run only through authorized tools; plugins out-of-process with OS sandbox profiles |
| Runaway automation | Unattended job takes destructive actions | Explicit permission scopes per automation; approval pauses for side effects; idempotency keys; no self-granted permissions |
| Credential exfiltration | Tool output containing secrets | Output scanning on artifact store; policy limits on network-capable tools; audit log |
| DoS via memory | Oversized contexts / many sessions | Admission controller + degradation ladder (§3.8) |
| Supply chain | Malicious model files, vendored sources, MCP servers, plugins | Pin + hash model artifacts and vendored mlx-c commits (PINS.md); manifest verification on load; documented trust model for third-party sources |
| Engine panic | MLX or family-crate panic kills sessions | Panic boundaries at the engine API edge contain Rust panics only — native crashes inside MLX's C++/Metal core remain fatal (R8); durable event log; resume-from-history is cheap by design (§3.8) |
| Telemetry egress | Always-on usage reporting (FR-35) is an outbound data path | Events are anonymous counts only (version, hardware tier, session/feature counts, plus aggregate §3.9 performance dimensions) — no prompts, no file paths, no model identifiers; the published event schema defines the exact fields; TLS to the project endpoint; the egress is a client connection, never a listener |

#### Residual Risks

A user who approves a malicious action bypasses the policy engine by definition (single-user system — the approval IS the user). Same-user vector: any local process running as the user can connect to the daemon's Unix domain socket and issue requests — the policy engine gates tool execution but cannot distinguish which local process issued a request (§3.7). At rest: session content sits in plain SQLite/artifact files; protection relies on macOS FileVault, not application-layer encryption (V1 assumption, §3.7).

### Appendix D: Capacity Model

Apple's RAM sizes are not uniformly power-of-two (8, 16, 18, 24, 32, 36, 48, 64, 96, 128 GB and up). Budgets are computed from **detected physical RAM** at runtime — the tiers below are defaults for model profiles and benchmark scenarios, not classes the admission controller assumes. The author's 36 GB MacBook is a first-class benchmark and CI target.

Initial per-tier budgets (calibrated by measurement in Phase 5; headroom = 25% of unified memory reserved for OS + app). Budget derivation rule: budgets for tiers without a dedicated row are linearly interpolated between the nearest listed tiers and clamped to the 25%-headroom rule (weights + KV ≤ 75% of RAM). Non-8/36-GB rows are extrapolated defaults pending Phase 5 calibration — only the 8 GB and 36 GB tiers are measured first-class targets.

| Tier | Weights budget | KV/context budget | Target model class (V1) | Prefill chunk | Notes |
|------|----------------|-------------------|-------------------------|---------------|-------|
| 8 GB | ≤ 4.5 GB | ≤ 1 GB | 3-4B @ 4-bit, short context (8k), rotating cache default | 512 | Aggressive idle-cache eviction; compaction early |
| 16 GB | ≤ 9 GB | ≤ 2.5 GB | 7-8B @ 4-bit, 16k context | 1024 | Rotating cache optional |
| 18 GB | ≤ 10 GB | ≤ 2.5 GB | 8B @ 4-bit or 7B @ 6/8-bit, 16k context | 1024 | Sits between 16/24 defaults; same policy as 16 GB with slightly larger weights budget |
| 24 GB | ≤ 13.5 GB | ≤ 3.5 GB | 12-14B @ 4-bit, 32k context | 1024 | Full cache becomes practical |
| 32 GB | ≤ 18 GB | ≤ 6 GB | 14B @ 4-bit or 8B @ 8-bit, 32k context | 1024 | Full cache; one model + STT |
| **36 GB** | **≤ 20 GB** | **≤ 7 GB** | **16-20B @ 4-bit or 14B @ 6-bit, 32-64k context** | **1024** | **Author's MacBook — first-class benchmark tier; full cache + concurrent sessions + STT resident** |
| 48 GB | ≤ 26 GB | ≤ 10 GB | 30B-class @ 4-bit, 64k context | 2048 | Full cache + concurrent sessions (weights+KV+headroom = 48 GB exactly) |
| 64 GB | ≤ 36 GB | ≤ 12 GB | 30B-class @ 4-bit, 64k context | 2048 | Multiple models resident |
| 96 GB | ≤ 54 GB | ≤ 18 GB | 70B-class @ 4-bit or MoE, 128k context | 2048 | Multiple models + SSD cold tier via oMLX backend |
| 128 GB+ | ≤ 72 GB | ≤ 24 GB | 70B-class @ 4-bit or MoE, 128k context | 2048 | Same, with headroom for draft models |

#### Unified-Memory Feature Usage (per FR-17/FR-19)

| MLX capability | Engine usage |
|---------------|--------------|
| Shared CPU/GPU arrays | Weights loaded once; CPU and GPU views alias the same unified memory (zero-copy) — no host-side duplicate of any resident tensor |
| Metal memory limit | Per-process limit configured from admission budget; generation that would exceed it fails cleanly, never OOMs the OS |
| MLX allocator cache limit + reclamation | Idle KV/activation caches reclaimed first under pressure; model eviction is the next step, OS OOM never |
| Wired-limit guidance | `alloy doctor` reports the macOS wired limit and recommends a per-process value below it for large-model profiles |
| Lazy evaluation / `mlx_compile` | Temporaries bounded via graph compilation of decode steps; peak-memory-aware prefill chunking bounds prompt-processing peaks |
| Quantized KV (where family supports) | Cache budget halved per FR-11 capability flags — gated per family by conformance |

Planning formula for full-attention transformers (approximation, calibrated per family crate):

```text
M_KV ≈ 2 · L · B · T · H_KV · D_head · s
```

(2 = keys+values; L layers, B batch, T tokens, H_KV KV heads, D_head head dim, s bytes per element — halved for 8-bit KV where the family supports it.)

Admission flow: inspect → estimate (formula + calibration table, scaled by detected RAM) → reserve → load → measure → reconcile (update calibration table). The 8 GB and 36 GB tier profiles are first-class CI targets (simulated budgets) — the harshest correctness environment and the author's daily driver, respectively. In-process admission benefits from direct allocator visibility: reserved-vs-actual reconciliation reads real memory pressure rather than a foreign process's self-report.

---

*End of SPEC-1. Status: Draft v1.9 — consistency-reviewed (v1.3, v1.8), Goldfish-reviewed (v1.4/v1.5/v1.9, all 4 lenses), and all decisions resolved (v1.6: name = alloy, own-bindgen decision, benchmark corpus; v1.7: conversion via uv, telemetry required, speculative decoding committed, Kokoro-82M TTS; v1.9: multimodal artifacts, draft pairing, job schema/trigger vocabulary pinned). Zero open decisions — the spec is ready for Phase 0 execution.*
