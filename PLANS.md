# PLANS.md — Outstanding Work (feature/mamba-3)

Coverage gaps, documentation deficits, and standards alignment for the
`feature/mamba-3` branch at HEAD `b72f6a0`, ordered by ROI. Each item lists
scope, acceptance criteria, files touched, and parallelization notes.

**Session-to-date shipped capability** (for context): Mamba-3 end-to-end (real
ib-ssm 370M, streaming decode, MIMO kernel/block); safetensors loader dispatch
(Llama/Mistral/Phi/Qwen/Mixtral/Qwen-MoE/DeepSeek-V2/V3/Granite-MoE); HF
tokenizer.json adapter (SPM+Metaspace+ByteFallback); multi-shard; MLA
attention kernel + TransformerModel integration; MoE GroupedGEMM refactor
(bit-identical); DeepSeek multi-shared-expert; Phi-3 fused-tensor loader;
Granite-MoE fused-per-expert loader. Real-weight validation for: Phi-3.5-mini
(7.6 GB), Granite-3.0-MoE (6.3 GB), Qwen2.5-0.5B (999 MB), TinyLlama-1.1B
(2.1 GB), ib-ssm/mamba3-370M (1.55 GB).

---

## P0 — Blocks existing claims (highest ROI)

### P0.1 HF ByteLevel BPE tokenizer adapter
Our `tokenizer.json` adapter handles Metaspace pretokenizer + ByteFallback
decoder (Llama-2/Mamba-3). It does **not** handle ByteLevel pretokenizer +
ByteLevel decoder (GPT-2/Qwen/Llama-3/Granite/Phi-3). Without this we cannot
encode real text for any of the dense transformer architectures we "support".

- **Scope**: Extend `HfTokenizerJsonParser` + `HfBpeTokenizerFactory` to detect
  and route ByteLevel pretokenizer (regex-split + byte-to-unicode mapping) and
  ByteLevel decoder. Add the GPT-2 255-byte alphabet mapping. Handle
  `Sequence` of pretokenizers (Qwen2 uses `[Split, ByteLevel]`).
- **Acceptance**: round-trip "Hello world" on Qwen2.5-0.5B, Phi-3.5,
  Granite-3, TinyLlama tokenizers produces identical token IDs to HF Python's
  `AutoTokenizer.encode`. Unit tests for Split+ByteLevel composition.
- **Files**: `src/DotLLM.Tokenizers/Hf/{HfTokenizerJsonParser,HfBpeTokenizerFactory}.cs`,
  new `ByteLevelPreTokenizer.cs`, new `ByteLevelDecoder.cs`, tests under
  `tests/DotLLM.Tests.Unit/Tokenizers/Hf/`.
- **Deps**: none. **Parallel**: yes.

### P0.2 DeepSeek-V2-Lite real-weight end-to-end
Download in flight (~31 GB, `b3rt1lmmu`). Once present, the existing
`RealHfSafetensorsEndToEndTests` pattern gives real-weight validation for MLA
+ multi-shared-expert MoE.

- **Scope**: add `DeepSeekV2Lite_LoadsAndForwardsEndToEnd` gated by
  `DOTLLM_DEEPSEEK_V2_LITE_PATH` + auto-detect `C:/temp/dotllm-deepseek-v2-lite/`.
  Forward `[0, 1, 2]`, assert `[3, 102400]` finite, stddev nonzero, MLA config
  populated (`kv_lora_rank=512`, `qk_nope_head_dim=128`, etc.), MoE populated
  (`n_routed_experts=64`, `n_shared_experts=2`).
- **Acceptance**: test passes. Documents drift max_abs vs PyTorch reference if
  feasible, else just finite check.
- **Files**: `tests/DotLLM.Tests.Integration/Models/Loaders/RealHfSafetensorsEndToEndTests.cs`.
- **Deps**: download completion.
- **Parallel**: independent of everything else.

### P0.3 Mamba-3 MIMO weight loader + model wiring
`Mamba3TransformerModel` currently throws `NotSupportedException` when the
config has `IsMimo=true`. The MIMO kernel and Block paths are complete and
test-covered (step 60f); the remaining gap is the weight loader needing
`[H, R, N]`-shaped B_bias/C_bias + `mimo_x`/`mimo_z`/`mimo_o` tensors.

- **Scope**: extend `Mamba3WeightLoader` to load MIMO tensors; remove the
  `NotSupportedException`; route MIMO models through `ForwardMimo` with
  streaming state.
- **Acceptance**: synthetic MIMO model loads + forwards + generates. No public
  real-MIMO checkpoint exists — synthetic fixture suffices. Existing 83
  Mamba-3 unit tests stay green.
- **Files**: `src/DotLLM.Models/Architectures/Mamba3WeightLoader.cs`,
  `Mamba3TransformerModel.cs`, new unit test + synthesis script.
- **Deps**: none. **Parallel**: yes.

### P0.4 Unified "Supported Models" matrix
A single, authoritative page listing every architecture dotLLM can load, with
required tokenizer type, RoPE variant, quant support, and the exact HF
`config.json` field set. Solves discoverability and prevents claim drift.

- **Scope**: new `docs/SUPPORTED_MODELS.md` + link from README. Table:
  arch, enum, config fields, tokenizer, RoPE, KV-cache, MoE, notes.
  Populate from current code; reference real-weight proofs.
- **Acceptance**: every `Architecture` enum variant is listed with accurate
  field mapping and at least one verified checkpoint (or "verified with
  tiny-random" / "synthetic only" tag).
- **Files**: new `docs/SUPPORTED_MODELS.md`, `README.md` link.
- **Deps**: none. **Parallel**: yes (docs-only).

---

## P1 — Standards alignment

### P1.1 Shared attention tensor loader abstraction
`TransformerWeightsSafetensorsLoader` now has `LoadLlamaAttentionLayer`,
`LoadPhi3AttentionLayer` (fused QKV), `LoadDeepSeekMlaLayer`,
`LoadGraniteMoeLayer` — each with ~30-60 lines of tensor-name resolution.
Repeat pattern ripe for DRY.

- **Scope**: extract a shared `AttentionTensorLoader` service that takes an
  attention variant enum (Llama/GQA, Phi3/FusedQKV, MLA) + per-variant tensor
  names and returns a populated `TransformerLayerWeights` attention slot.
  Bit-identical output.
- **Acceptance**: all 1285 unit + full integration tests pass unchanged. Each
  per-arch layer loader is <20 lines.
- **Files**: `src/DotLLM.Models/Architectures/TransformerWeightsSafetensors.cs`,
  possibly new `AttentionTensorLoader.cs`.
- **Deps**: **conflicts with P2.2 (YaRN)** on MLA loader lines. Sequence P1.1
  before P2.2. **Parallel**: yes with everything else.

### P1.2 Remove `goto FfnBranch` in TransformerModel.Forward
The MLA branch introduced a `goto` label to jump past GQA Q/K/V/RoPE/O code
to the shared FFN dispatch. Unusual for C#; refactor into explicit branch /
helper method.

- **Scope**: extract `RunAttentionLayer(config, lw, ...)` helper returning
  post-attention hidden; both GQA and MLA paths call it; then FFN runs. No
  numerical change.
- **Acceptance**: bit-identical logits on all real-weight tests
  (Phi-3.5, Qwen2.5, TinyLlama, Granite-MoE, DeepSeek-V2-tiny, Mamba-3 ib-ssm).
- **Files**: `src/DotLLM.Models/Architectures/TransformerModel.cs`.
- **Deps**: **conflicts with P1.1**. **Parallel**: no — must sequence against
  P1.1 on same file.

### P1.3 Benchmark regression sweep
No benchmarks run since the session began. The GroupedGEMM refactor, pooled
scratch, and MLA integration could silently regress throughput. Run the
existing BDN suite and commit deltas.

- **Scope**: `dotnet run -c Release --project benchmarks/DotLLM.Benchmarks`
  against SmolLM-135M, Llama-3.2-1B (or equivalent); compare vs `main`.
- **Acceptance**: prefill + decode tok/s within ±5% of main. Any regression
  documented with root cause.
- **Files**: `benchmarks/*.cs` (only if new benchmarks needed), report in
  PLANS.md or separate doc.
- **Deps**: none. **Parallel**: yes (separate artifact).

### P1.4 Analyzer / style pass
Spot-check for IDE warnings, CA-analyzer hits, stale XML doc references,
unused using statements accumulated over the session's many commits.

- **Scope**: `dotnet build -warnAsError=false`; triage warnings; fix trivial.
  Run any `.editorconfig`-enforced formatter.
- **Acceptance**: warning count on `feature/mamba-3` ≤ warning count on
  `main`. No new suppressions.
- **Files**: various, all new code primarily.
- **Deps**: best done last to sweep up everything. **Parallel**: yes (touches
  many files but non-semantic).

---

## P2 — Hardening & correctness expansion

### P2.1 Transformer generation loop integration test
We've only exercised generation end-to-end on Mamba-3. Each real transformer
(Qwen2.5, TinyLlama, Phi-3.5, Granite-MoE) should have an equivalent
`GeneratesText_FromTokenizedPrompt`-style test.

- **Scope**: gated integration test per architecture; encodes a short prompt
  via the HF tokenizer (requires **P0.1**), iteratively forwards growing
  context, argmax, decode; asserts finite, valid token IDs, no infinite loops.
- **Acceptance**: 5-token generation from "The capital of France is" or
  similar completes for each arch, producing at least one non-trivial token.
- **Files**: `tests/DotLLM.Tests.Integration/Models/Loaders/`
  (new `*GenerationTests.cs` per arch or one shared).
- **Deps**: **P0.1 (ByteLevel tokenizer)** for Qwen/Phi/Granite/Llama-3;
  TinyLlama can go without (SPM).
- **Parallel**: after P0.1.

### P2.2 YaRN RoPE mscale
`MlaConfig` stores YaRN scaling fields (`RopeScalingFactor`,
`RopeScalingMscale`, `RopeScalingMscaleAllDim`,
`RopeScalingOriginalMaxPositionEmbeddings`) but `MlaAttention.Execute`
ignores them. DeepSeek-V3 uses YaRN for 128K+ context; without it, long
prompts produce wrong logits.

- **Scope**: apply mscale correction in the attention scale (`scale *=
  mscale^2`) and in RoPE frequency scaling per YaRN spec. Preserve default
  (non-YaRN) behavior when fields are null.
- **Acceptance**: DeepSeek-V2/V3 long-context test (>4K tokens) produces
  finite logits; drift vs known-good reference within tolerance.
- **Files**: `src/DotLLM.Cpu/Kernels/MlaAttention.cs`,
  `src/DotLLM.Models/Architectures/TransformerModel.cs` (if RoPE table
  construction moves).
- **Deps**: sequences after P1.1 on MLA loader lines.
- **Parallel**: conflicts with P1.1 and P2.3 — serialize.

### P2.3 MLA latent KV-cache + absorption
MLA's production win is KV-memory reduction (store `kv_lora_rank=512` latent
instead of `n_heads * (qk_nope + v) = 3072+` per head). Absorption fuses
`W_q_nope @ W_k_nope^T` at load time, eliminating runtime GEMM. Currently
PoC reruns the full attention forward per call.

- **Scope**: `LatentMlaKvCache` storing compressed latent; `MlaAttention`
  variant consuming it; absorption precomputed at load time.
- **Acceptance**: DeepSeek-V2-Lite generates text; memory footprint drops
  measurably; logits match pre-optimization within 1e-3.
- **Files**: `src/DotLLM.Engine/KvCache/LatentMlaKvCache.cs` (new),
  `src/DotLLM.Cpu/Kernels/MlaAttention.cs`,
  `src/DotLLM.Models/Architectures/TransformerModel.cs`.
- **Deps**: after P2.2 (shared file). **Parallel**: no.

### P2.4 Sliding-window attention test
Qwen2.5-0.5B config has `sliding_window=32768`; Mistral uses sliding window
smaller. Currently untested — we feed 3 tokens so window never triggers.
Kernel correctness for >window contexts is unverified.

- **Scope**: test that feeds >`sliding_window` tokens (or a smaller
  synthetic config with window=8, seqlen=16) and asserts attention pattern
  respects the window. Compare against a reference Python implementation or
  a masked-brute-force scalar.
- **Acceptance**: sliding window masks correctly zero out pre-window
  attention weights; output matches masked reference.
- **Files**: `src/DotLLM.Cpu/Kernels/Attention.cs` (check existing),
  `tests/DotLLM.Tests.Unit/Cpu/Kernels/AttentionSlidingWindowTests.cs` (new).
- **Deps**: none. **Parallel**: yes.

### P2.5 OLMoE-1B-7B real-weight Mixtral-convention validation
Mixtral-convention MoE (`block_sparse_moe.experts.{j}.w{1,2,3}` with per-
expert separate tensors) only tested with 522 KB tiny-random. OLMoE-1B-7B
(14 GB) is current (2024), open, Mixtral-convention.

- **Scope**: download `allenai/OLMoE-1B-7B-0924`; add
  `OLMoE1B7B_LoadsAndForwardsEndToEnd` gated test.
- **Acceptance**: load + forward + finite logits.
- **Files**: download script, test addition to
  `RealHfSafetensorsEndToEndTests.cs`.
- **Deps**: 14 GB download + disk availability.
- **Parallel**: yes (independent download + test).

### P2.6 bf16/F16 correctness test
Every real checkpoint so far is BF16 on disk, upcasted to F32 at load. The
upcast is assumed correct but never numerically compared against a reference
running on native BF16 (e.g., PyTorch). Drift in edge cases (denormals,
NaN propagation through softmax) could pass finite checks but still be
wrong.

- **Scope**: pick one small BF16 model, load via dotLLM → F32 and via
  PyTorch → BF16; compare forward outputs at N tokens.
- **Acceptance**: max_abs <1e-2 (very loose; BF16 has ~7-bit mantissa so
  drift vs F32 is expected ~1e-3 per layer, accumulating).
- **Files**: new Python reference script + C# test.
- **Deps**: none. **Parallel**: yes.

---

## P3 — Documentation refinement

All of these are independent text-only edits; can run fully in parallel. One
agent, one file each, trivial coordination.

### P3.1 `docs/ATTENTION.md` — MLA section
Current doc covers MHA/MQA/GQA. Add MLA section: down-projection via
`q_a_proj`/`kv_a_proj_with_mqa`, RMSNorm, up-projection, decoupled RoPE,
per-head SDPA with MQA-shared K_rope, o_proj. Reference `MlaAttention.cs`.

### P3.2 `docs/TOKENIZERS.md` — HF adapter section
Add section on `HfTokenizerJsonParser` + `HfBpeTokenizerFactory`. Document
which pretokenizer/decoder combinations are supported (Metaspace +
ByteFallback today; ByteLevel per P0.1 pending). Include how `model_type` →
tokenizer style mapping works, and the `ModelLoader.LoadTokenizerFromHfDirectory`
helper.

### P3.3 Per-architecture config reference
New `docs/ARCHITECTURES.md` (or section in SUPPORTED_MODELS.md) — for each
arch, list the `config.json` fields consumed by `HfConfigExtractor` and any
tensor naming quirks. Link to relevant kernel/loader sources.

### P3.4 Stale `docs/*.md` sweep
Spot-check every `docs/*.md` for references to out-of-date file paths,
missing features, or claims that no longer hold (e.g., "supports Llama,
Mistral, Phi, Qwen, DeepSeek" in the README may need updates for
Mamba-3/MoE/Granite).

### P3.5 Qwen2.5 + TinyLlama News entry
Single README News bullet covering both (real-weight validations of Qwen-dense
and Llama-family). Reference commit `b72f6a0`.

### P3.6 PR #136 description refresh
Per-commit comments have accumulated. PR description is the top-level view;
ensure it reflects the full scope shipped (a bullet list of arches supported,
tests passed, known gaps from this PLANS.md).

---

## P4 — Deferred / blocked

### P4.1 Chat templates from `tokenizer_config.json`
Not loaded today. Nice-to-have for generation UX. Defer.

### P4.2 Quantized safetensors (int8 / GPTQ / AWQ)
Significantly expands supported-checkpoint universe but is a full new feature
(new kernels, new loader paths). Separate PR / issue.

### P4.3 MIMO Mamba-3 real-weight verification
Blocked — no public MIMO Mamba-3 checkpoint exists. P0.3 covers the code
path; real-weight waits indefinitely.

---

## Parallelization analysis

**Fully independent (any agent, any file — safe for parallel worktrees):**
- P0.1 (ByteLevel tokenizer) — `src/DotLLM.Tokenizers/Hf/`
- P0.2 (DeepSeek-V2-Lite test) — test file only
- P0.3 (Mamba-3 MIMO loader) — `Mamba3WeightLoader.cs`, `Mamba3TransformerModel.cs`
- P0.4 (Supported Models matrix) — `docs/SUPPORTED_MODELS.md` (new)
- P1.3 (benchmarks) — `benchmarks/*`, separate artifact
- P2.4 (sliding-window test) — new test file + `Attention.cs` (usually isolated)
- P2.5 (OLMoE) — download + test file
- P2.6 (bf16 correctness) — new test + Python script
- All of P3 (docs) — one file each, text only

**Serialized chains (same-file conflicts):**
- P1.1 → P1.2 → P2.2 → P2.3 → P1.4 on `TransformerModel.cs` / `MlaAttention.cs`
- P0.1 → P2.1 (generation tests depend on ByteLevel tokenizer)
- P0.3 wiring → updating `.continue-here.md` / docs entries

**Download-bound (no compute but network):**
- P0.2 (DeepSeek-V2-Lite — already downloading)
- P2.5 (OLMoE — needs ~14 GB download)

**First-wave dispatch plan (maximum parallelism, no conflicts):**

| Wave | Items | Expected time |
|---|---|---|
| W1 (parallel, 4 agents + 1 orchestrator task) | P0.1, P0.3, P0.4, P1.1, P1.3 | ~30-60 min each |
| W2 (serial after W1 on MLA file) | P1.2, P2.2, P2.3 | ~30 min each |
| W3 (parallel after W1) | P2.1 (after P0.1), P2.4, P2.5 (after OLMoE download), P2.6 | ~30 min each |
| W4 (parallel, all W1-W3 prereqs satisfied) | P3.1, P3.2, P3.3, P3.4, P3.5, P3.6 | 10-20 min each |
| W5 (final sweep) | P1.4 (analyzer pass) | 15 min |

**Automated / passive items:**
- DeepSeek-V2-Lite download (~hours, no action)
- P0.2 test runs when download completes (automatic via the gated test)

**Critical-path estimate** (serial MLA chain): W1 + W2 ≈ 2-3 hours if MLA
chain is taken on one file sequentially. Everything else parallelizes around
that.
