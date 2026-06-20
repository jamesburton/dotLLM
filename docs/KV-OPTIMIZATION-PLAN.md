# KV-Cache Optimization Plan — TurboQuant + OSCAR + EpiCache

> **Status: PLAN ONLY.** No code changes accompany this document. This is a
> research + integration-design note for layering three complementary KV-cache
> optimisation techniques onto dotLLM's existing KV-cache stack.

> **Honesty note.** All three techniques are real, named papers (see citations).
> The composition analysis below is partly inferred from the papers + a public
> comparison article (MarkTechPost, 2026-06-18) — where a claim is inferred
> rather than stated by a primary source, it is marked **(inferred)**. Numbers
> are quoted as the papers state them; "blog microbenchmark" peak numbers are
> flagged as such.

---

## 1. Technique Summaries

### 1.1 TurboQuant — data-oblivious near-optimal vector quantization
- **Paper:** *TurboQuant: Online Vector Quantization with Near-optimal
  Distortion Rate*, Google Research / NYU, ICLR 2026.
  arXiv:[2504.19874](https://arxiv.org/abs/2504.19874).
- **Mechanism:** Each K (or V) vector is **randomly rotated** (a data-oblivious
  transform, e.g. a random/structured orthogonal map) so its coordinates become
  *nearly independent and approximately Gaussian* (a concentrated Beta
  distribution on coordinates in high dimension). Then an **optimal scalar
  (Lloyd–Max) quantizer is applied per coordinate**. Because MSE-optimal
  quantizers bias inner-product (attention-score) estimation, TurboQuant adds a
  **two-stage correction**: MSE quantization followed by a **1-bit Quantized
  Johnson–Lindenstrauss (QJL) transform on the residual**, yielding an
  *unbiased* inner-product estimator.
- **Optimises:** VRAM + attention bandwidth/speed via low-bit storage; quality
  preserved because distortion is provably near information-theoretic optimal
  (within ~2.7× of the lower bound) at *all* bit-widths/dimensions.
- **Data dependency:** **None** — data-oblivious, calibration-free. Works on any
  model, online. This is its headline differentiator.
- **Claimed numbers:** "Absolute quality neutrality at 3.5 bits/channel";
  "marginal degradation at 2.5 bits/channel"; near-full-precision
  Needle-in-a-Haystack recall at ~4× compression. The "3-bit / 6× / 8× faster
  attention" figures come from Google's blog microbenchmark, not the paper.
- **Granularity:** per-coordinate scalar quantization of each K/V vector.
  Treats keys and values with the same scheme (no explicit K-vs-V split).

### 1.2 OSCAR — offline attention-aware rotation for INT2
- **Paper:** *OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV
  Cache Quantization*, Together AI.
  arXiv:[2605.17757](https://arxiv.org/abs/2605.17757). Open-sourced; integrated
  into SGLang as an INT2 KV mode compatible with paged attention.
- **Mechanism:** Estimates **attention-aware covariance offline** (per layer)
  and derives **fixed rotation matrices + clipping thresholds**. Keys are rotated
  into the **query-covariance eigenbasis**; values into the **score-weighted
  value-covariance** basis (so the rotation aligns with what attention actually
  consumes, unlike a generic Hadamard rotation). It then applies a **Hadamard
  transform + bit-reversal permutation** to spread channel importance evenly
  before **INT2** quantization. Uses **mixed precision**: recent tokens in BF16,
  history in INT2 (a paged mixed-precision cache).
- **Optimises:** VRAM (8× at INT2) + decode throughput at long context; quality
  via attention-aligned rotation that survives 2-bit where naive rotation
  collapses.
- **Data dependency:** **Requires offline per-model calibration** to compute
  covariance/rotation/clip thresholds. This is the cost of its better INT2
  quality.
- **Claimed numbers:** ~8× KV memory at ~2.28 effective bits; within ~1.42
  points of BF16 on Qwen3-8B (naive-rotation INT2 "collapses to nearly zero");
  up to 7.83× job-level throughput and ~3× faster single-token decode at 100K
  context; robust to 128K.
- **Granularity:** per-layer rotation; per-channel clip; mixed-precision split
  by token recency (recent BF16 / history INT2).

### 1.3 EpiCache — episodic eviction on the temporal axis
- **Paper:** *EpiCache: Episodic KV Cache Management for Long Conversational
  Question Answering*, Apple. arXiv:[2509.17396](https://arxiv.org/abs/2509.17396).
  Training-free.
- **Mechanism:** Operates **orthogonally to quantization — on the temporal /
  token axis, not the numerical one**. (1) **Block-wise prefill** bounds peak
  memory (avoids the unbounded peak of "evict after full-context prefill").
  (2) **Episodic clustering** groups conversation history into coherent
  "episodes". (3) **Episode-specific KV eviction** keeps a per-episode cache
  instead of a single query-narrowed cache (fixing the multi-turn failure mode
  of query-dependent eviction). (4) **Query-to-episode routing** at inference
  selects the relevant episode(s). (5) **Adaptive layer-wise budget allocation**
  measures each layer's eviction sensitivity and distributes the memory budget
  accordingly.
- **Optimises:** Peak VRAM + latency for long multi-turn conversations, while
  keeping topic-relevant context (accuracy).
- **Data dependency:** Training-free; no separate clustering model required
  (clusters from the conversation's own representations). Routing is per-query.
- **Claimed numbers:** up to 40% accuracy over eviction baselines on LongConvQA;
  near-full-cache accuracy at 4–6× compression; up to 3.5–3.7× lower peak memory
  and ~2.4× lower latency.
- **Granularity:** episode (segment of tokens) for eviction; per-layer for
  budget; per-query for routing.

---

## 2. Composition Analysis

### 2.1 Why TurboQuant and OSCAR are complementary
Both quantize K/V, but they occupy **different points on the
quality/portability/bit-width curve** rather than different cache regions:

- **Same target, different regime.** TurboQuant is **data-oblivious /
  calibration-free** and shines at **3–4 bit** with provable distortion bounds
  across any model. OSCAR is **calibrated / per-model** and pushes to **INT2**
  by aligning rotation with attention statistics. They are "complementary" in
  the **deployment-choice** sense: pick TurboQuant when you can't calibrate
  (new model, online, model-agnostic); pick OSCAR when you can calibrate and
  need 2-bit.
- **Shared structural seam — the mixed-precision split.** OSCAR's design is
  *recent BF16 + history INT2*. dotLLM **already has this exact split**
  (`QuantizedKvCache` = full-precision ring window + append-only quantized
  region). The quantizer used for the history region is pluggable: it can be
  TurboQuant-style rotate+scalar or OSCAR-style calibrated-rotate+INT2.
  **(inferred)** A real "merge" is therefore *one mixed-precision cache where
  the history-region codec is selectable* (TurboQuant for portability, OSCAR
  for max compression), not two stacked codecs on the same bytes.
- **Genuinely stackable sub-component.** TurboQuant's **QJL residual
  correction** (unbiased inner-product) is conceptually orthogonal to OSCAR's
  **attention-aware rotation** and could be applied on top of an
  OSCAR-rotated/quantized history to debias scores. This is the strongest
  literal "compose both" path, but it is **(inferred / research-grade)** — no
  source demonstrates the combination.

**Recommendation:** treat TurboQuant and OSCAR as **two selectable codecs behind
one mixed-precision KV interface**, not as two layers applied to the same bytes.
Start with the data-oblivious one (TurboQuant) because it needs no calibration
pipeline.

### 2.2 How EpiCache layers on top
EpiCache is on the **temporal axis** and is therefore the **outermost layer**:
it decides *which tokens/episodes stay resident*; the quantizer decides *how
those resident tokens are stored*. The natural order:

```
EpiCache (segment history into episodes, route query, evict non-relevant)
   └─> resident episodes' KV  ──> TurboQuant or OSCAR codec (low-bit storage)
          └─> recent tokens kept in the full-precision window
```

So EpiCache compresses *count* (fewer tokens cached), the quantizer compresses
*width* (fewer bits/token). Wins are **multiplicative on VRAM** (e.g. EpiCache
4–6× × quantizer 4–8×) but EpiCache's biggest gain — **bounded peak memory via
block-wise prefill** — is a scheduling/eviction property dotLLM lacks today and
must be added regardless of the codec.

### 2.3 Expected combined wins & quality risk
- **VRAM:** plausibly an order of magnitude on long multi-turn workloads
  (eviction × low-bit). Conservative target: 4–8× from quantization alone
  (already dotLLM's Q4_0/Q8_0 ballpark, improved to 2–3 bit), plus 2–4× from
  episodic eviction on conversational traffic.
- **Speed:** decode is bandwidth-bound (see MEMORY.md vulkan note); fewer bits ×
  fewer tokens both directly reduce bytes moved per decode step → both help.
- **Quality risk (highest → lowest):**
  1. **INT2 (OSCAR)** without correct per-layer calibration → collapse. Needs a
     real calibration harness + per-layer covariance; highest risk.
  2. **EpiCache routing** evicting an episode the query actually needed → wrong
     answer (not just degraded). Needs eval on multi-turn benchmarks.
  3. **TurboQuant 2.5-bit** — "marginal" per paper, lowest risk; 3.5-bit is
     "neutral".

---

## 3. dotLLM Integration Map (current seams)

Existing stack (verified by reading the source):
- **`IKvCache`** (`src/DotLLM.Core/Attention/IKvCache.cs`) — `Update` /
  `GetKeys[Ref]` / `GetValues[Ref]` / `Rollback`. The universal seam.
- **`IQuantizedKvCache : IKvCache`** (`.../IQuantizedKvCache.cs`) — exposes
  `QuantizedLength` / `WindowLength` / `WindowCapacity`, per-layer pointers
  (`GetQuantizedKeysPtr`, `GetWindowKeysPtr`, …), `KeyDType`/`ValueDType`,
  row-byte sizes. Attention kernels branch on `if (kvCache is IQuantizedKvCache)`.
- **`KvCacheConfig`** (`src/DotLLM.Core/Configuration/KvCacheConfig.cs`):
  `(KeyDType, ValueDType, MixedPrecisionWindowSize)`.
- **`KvCacheDType`** enum: `F32 / Q8_0 / Q4_0` — **this enum is the choke point**
  for adding new codecs (TurboQuant/OSCAR are new entries here).
- **`QuantizedKvCache`** (`src/DotLLM.Engine/KvCache/QuantizedKvCache.cs`) —
  dual-region (quantized append-only + FP32 ring window), `BlockSize=32`,
  Q8_0=34B / Q4_0=18B per block, **per-layer eviction tracking already present**
  (`_layerQuantizedLength`). CUDA mirror: `CudaQuantizedKvCache`.
- **Paged path:** `PagedKvCache` / `KvBlockPool` / `KvBlockTable`;
  `PrefixTrieManager` for cross-request prefix sharing. **Docs note paged and
  quantized are currently mutually exclusive** (paged falls back to quantized
  with a warning) — see KV_CACHE.md "Limitations (v1)".
- **Attention integration:** CPU `Attention.ExecuteTiledQuantizedHead`
  (per-tile dequant); GPU scratch-buffer dequant. Any new codec needs a matching
  dequant path in CPU + Vulkan + CUDA (see CLAUDE.md cross-backend rule).

### 3.1 Per-layer / per-head layout constraint (Gemma 4) — IMPORTANT
`QuantizedKvCache` assumes a **uniform** `_headDim` and `_kvStride =
numKvHeads*headDim` across all layers (constructor takes a single `headDim`;
asserts `kvStride % 32 == 0`). But **Gemma 4 has per-layer-strided KV**:
- `ModelConfig.GetLayerHeadDim(layerIdx)` returns **global_head_dim (512)** for
  full-attention layers vs **head_dim (256)** for sliding-window layers
  (`GlobalHeadDim`, `PerLayerSlidingWindow` in `ModelConfig.cs`).
- Sliding layers: 8 KV heads × 256; global layers: 2 KV heads × 512 (per task
  note) → **different kvStride per layer**.

**Implications for each technique:**
- **TurboQuant/OSCAR rotation matrices are dimension-specific** → must be
  generated per *distinct* (kvStride / head_dim), i.e. at least two variants
  (sliding 256 vs global 512). The cache must size quantized rows **per layer**,
  not from one global `headDim`. This requires generalising `QuantizedKvCache`
  to per-layer `_kvStride[]` (today it is scalar).
- **OSCAR per-layer offline covariance** is naturally per-layer anyway — fits,
  but calibration must cover both layer types.
- **EpiCache adaptive layer-wise budget** must respect that sliding layers
  already evict (windowed) — budget allocation should account for the existing
  sliding window so it does not double-evict.

---

## 4. Phased Integration Plan

Order rationale: build the **mixed-precision codec seam** first with the
**calibration-free** quantizer (TurboQuant) → add the **calibrated INT2** codec
(OSCAR) reusing the same seam → add **episodic eviction** (EpiCache) on top as a
scheduling/eviction layer. Quantization before eviction because the cache
storage seam and per-layer-stride generalisation are prerequisites for both, and
TurboQuant needs no calibration infrastructure.

### Phase 0 — Generalise the KV cache to per-layer stride (prerequisite)
> **Status: F32 / CPU correctness payoff DONE; quantized + CUDA outstanding.**
> The stale "touch `QuantizedKvCache`/`CudaQuantizedKvCache`" framing was aimed at
> the wrong file — the real blocker was the contiguous F32 `SimpleKvCache` plus the
> `GuardKvCacheHeadDim` throw on the CPU gemma-4 path. See `docs/KV-PHASE0-PLAN.md`
> for the verified file-level plan. Vulkan was already per-layer (`VulkanKvCache`).

**Landed (`dev`):**
- ✅ Core `src/DotLLM.Core/Attention/KvGeometry.cs` descriptor (+ `ModelConfig.GetLayerKvHeads`,
  `KvGeometry.FromConfig`) — the single per-layer-stride source of truth.
- ✅ Core `src/DotLLM.Core/Attention/IPerLayerKvCache.cs` — geometry-query capability
  (keeps `IKvCache` geometry-free) used to validate a supplied cache.
- ✅ `SimpleKvCache` generalised to `KvGeometry` (scalar ctor forwards via `Uniform`,
  byte-identical for uniform models; per-layer buffers/offsets for gemma-4).
- ✅ CPU gemma-4 cached decode: `RunGemma4Layer` now consumes `IKvCache` on the
  autoregressive (`_pkvPhase == None`) path; `GuardKvCacheHeadDim` replaced with a
  geometry-match assertion (mismatched cache → `ArgumentException`; matched → runs).
- ✅ Tests: `SimpleKvCachePerLayerStrideTests` (§5.1) + `Gemma4CpuKvCacheTests`
  (§5.2, the acceptance gate — distinct sliding 32 / global 64 strides, prefill+decode
  == cacheless). Full Unit/KvCache + gemma-4 forward suites green (the only failures,
  `WeightRepackingTests.ComputeRowsQ8_0Interleaved`, are pre-existing and unrelated).

- ✅ `QuantizedKvCache` (CPU) generalised to `KvGeometry` + per-layer quant/window buffers +
  `IQuantizedKvCache.KeyQuantizedRowBytesOf/ValueQuantizedRowBytesOf(int)` (default interface
  methods, so CUDA compiles unchanged) + CPU quantized attention consumer reads `*Of(layer)`
  (KV-PHASE0-PLAN §2.3/§3.2/§3.4). Server quantized factories use `KvGeometry.FromConfig`.
  Test: `QuantizedKvCachePerLayerStrideTests` (§5.3). Full Unit suite 2425/0.

- ✅ Vulkan refactor (§3.5): `VulkanKvCache` gains a `KvGeometry` ctor + implements
  `IPerLayerKvCache`; `VulkanTransformerModel.CreateKvCache` calls `KvGeometry.FromConfig`;
  `GemmaLayerKvHeads` delegates to `Config.GetLayerKvHeads`. No functional change — verified
  byte-identical via `Gemma4VulkanKvCacheTests` (GPU, passed under the GPU lock).
- ✅ CUDA Option-1 (§3.6): `CudaKvCache` + `CudaQuantizedKvCache` take `KvGeometry` (scalar
  ctors forward via `Uniform`), per-layer buffers/row-bytes, shared dequant scratch sized to
  `max(stride)`, implement `IPerLayerKvCache` + `*Of`; `CudaTransformerModel.CreateKvCache`
  (both overloads) uses `FromConfig`. No gemma-4 CUDA decode path. Build-verified here
  (runtime needs T5500); byte-identical for the uniform models CUDA actually serves.

**KV Phase 0 is COMPLETE** (CPU correctness + per-layer F32/quantized caches; Vulkan + CUDA
type surface unified on `KvGeometry`). Next: Phase 1 (TurboQuant codec).
- ⬜ CUDA Option-1 constructor/scratch generalisation (§3.6) — gemma-4 CUDA is cacheless
  so off the critical path; constructor surface only.
- ⬜ Batched (continuous-batching) gemma-4 path stays guarded — it has no gemma-4 layer
  implementation at all, so its `HasDistinctPerLayerHeadDim` throw is left in place.
- **Acceptance (quantized, when done):** Gemma 4 (sliding 256 / global 512) runs with
  Q8_0/Q4_0 KV and matches FP32 logits within tolerance; no regression for Llama.
- **Risk:** block-size constraint (`kvStride % 32`) — 256 and 512 fine; validate each
  layer's stride for any future odd head_dim. Low.

### Phase 1 — TurboQuant codec (data-oblivious)
- **Add:** `KvCacheDType.TurboQuant` (or a sub-config carrying bit-width 2.5/3.5
  and a rotation seed). New codec module implementing rotate→scalar-quantize and
  the QJL residual path; CPU kernel first, then Vulkan + CUDA dequant
  (cross-backend rule — fix all three before "done").
- **Touch:** `KvCacheDType.cs`, `KvCacheConfig.cs` (bit-width field),
  `QuantizedKvCache.cs` (codec dispatch on write/read), CPU
  `Attention.ExecuteTiledQuantizedHead`, Vulkan + CUDA dequant paths, CLI
  parsing in `KvCacheConfig.ParseDType`.
- **Acceptance:** 3.5-bit ≈ FP32 logits (paper "neutral"); 2.5-bit within a
  small, measured perplexity delta; rotation is deterministic (seeded) so
  prefix-cache/rollback stay valid; bandwidth/decode bench shows the expected
  byte reduction. A regression test that **discriminates** rotate-correct vs
  rotate-wrong (not a degenerate shape) per CLAUDE.md.
- **Risk:** rotation cost on the hot path (must be cheap structured transform,
  not dense matmul); QJL residual storage layout; interaction with the
  full-precision window boundary.

**Phase 1 progress (slices):**
- ✅ **Slice 1 — MSE codec** (`TurboQuantCodec`): RHT rotation + per-coord standard-normal
  Lloyd–Max scalar quant + per-vector norm. Centroids match the paper's b=1/b=2 values;
  reconstruction relMse tracks ε_b; wrong-seed decode collapses (discriminating).
- ✅ **Slice 2 — cache** (`TurboQuantKvCache`): per-(pos,head) code+norm store, dequant-to-scratch
  through the plain-fp32 attention path; `KvCacheDType.TurboQuant`, `tq2..tq8` CLI, server factory.
- ✅ **Slice 3 — QJL residual** (Algorithm 2, opt-in `useQjl`): MSE at `bits-1` + a seeded
  Gaussian sketch `S`; stores `sign(S·r)` + `γ=‖r‖` packed in the code blob; decode folds in
  `x̃_qjl=(√(π/2)/d)·γ·Sᵀ·q` so `E_S[⟨y,x̃⟩]=⟨y,x⟩`. Measured: self/cross-score contraction bias
  −0.032/−0.034 (MSE) → +0.0006/+0.0002 (QJL); ℓ2 relMse rises (0.009→0.055) — the QJL trade.
  CLI `tqq`/`tq2q..tq8q`; cache `useQjl` ctor + `KvCacheConfig.TurboQuantUseQjl`. Discriminating
  tests in `TurboQuantCodecTests`/`TurboQuantKvCacheTests`.
- **Slice 4 — cross-backend dequant** (Vulkan primary, CUDA secondary): the GPU payoff. **Targets the
  MSE path** (the model-level quality-neutral winner — see benchmark below); cheaper O(d log d) shader.
  GPU QJL (`Sᵀ·q`, O(d²), not a 4-bit win) deferred. Sub-slices:
  - ✅ **4a — Vulkan dequant kernel** (`turboquant_dequant_f32.comp` + `.spv` +
    `TurboQuantDequantF32Kernel`): centroid lookup → cooperative in-place unnormalized Walsh–Hadamard
    → ×invSqrtD×sign×norm; one workgroup per head-vector, headDim power-of-two ≤ 256; codec constants
    (centroids/signs/invSqrtD) passed as buffers/push-constants so the kernel is backend-pure.
    `TurboQuantCodec` exposes `MseBits`/`RotationSigns`/`InvSqrtD`. **GPU-verified bit-exact** vs the CPU
    codec on gfx1151 (`VulkanTurboQuantDequantF32KernelTests`, maxAbsDiff 0.0).
  - ⬜ **4b — `VulkanTurboQuantKvCache` + routing + end-to-end.** Store codes+norms in device buffers;
    on update encode the fresh K/V (readback+CPU-encode first cut, or a GPU encode shader) → upload
    codes; on attention prep dispatch the 4a dequant kernel into the F32 scratch the attention shader
    already reads. Route in `VulkanTransformerModel.CreateKvCache` on `KvCacheConfig.IsTurboQuant`.
    Needs integration into the fence-pipelined forward graph + an end-to-end GPU parity run vs CPU.
  - ⬜ **CUDA mirror** (`CudaTurboQuantKvCache` + a `turboquant_dequant.cu` PTX kernel, mirroring the
    existing `CudaQuantizedKvCache` dequant-to-FP16-scratch). Build-verify only on this box (no NVIDIA);
    runtime-verify on T5500.
- ✅ **Model-level parity benchmark** (`TurboQuantKvParityTests`, env-gated `DOTLLM_TURBOQUANT_PARITY_GGUF`):
  teacher-forced prefill+decode of Llama-3.1-8B-Q4_K_M (8 prefill + 48 decode) with a full-precision
  `SimpleKvCache` reference vs `tq4` and `tq4q`. **Results** (vs F32 PPL 3.038):
  `tq4` PPL **3.091 (+0.05)**, top-1 argmax agreement **97.9%**, mean|Δlogit| 0.196, meanKL 0.0089 —
  4-bit MSE TurboQuant is quality-neutral. `tq4q` PPL 3.309 (+0.27), top-1 93.8%, mean|Δlogit| 0.418.
  **Finding:** at an iso-*total*-bit budget, plain MSE beats QJL — QJL (3-bit MSE + 1-bit residual)
  removes the inner-product *bias* but the extra JL noise costs more ℓ2 accuracy than the debiasing
  buys at 4-bit. QJL is expected to help only at very low bits or iso-*MSE*-bits, not iso-budget.

### Phase 2 — OSCAR codec (calibrated INT2)
- **Add:** offline **calibration harness** (compute per-layer attention-aware
  covariance → rotation + clip thresholds), serialised alongside / next to the
  GGUF (new artifact). `KvCacheDType.OscarInt2`. Reuse the Phase-1 mixed-precision
  seam (recent window already BF16/FP32).
- **Touch:** new `src/DotLLM.Models/` (or tools/) calibration utility; codec in
  Engine; CPU/Vulkan/CUDA dequant; loader to attach calibration artifact;
  per-layer rotation table keyed by layer head_dim (Gemma-4 aware).
- **Acceptance:** INT2 within a few points of FP32 on a long-context eval (paper:
  ~1.42 pts Qwen3-8B); demonstrably better than naive-rotation INT2; calibration
  reproducible; works for both Gemma-4 layer types.
- **Risk:** **highest quality risk** (INT2 collapse without good calibration);
  calibration data selection; artifact distribution/versioning; per-model cost.

### Phase 3 — EpiCache episodic eviction (on top)
- **Add:** block-wise prefill (bounded peak), episodic clustering + per-episode
  KV grouping, query→episode routing, adaptive per-layer budget. This is an
  **eviction/scheduling layer** above the codec.
- **Touch:** likely a new `IKvCache` decorator or scheduler-level component
  (`src/DotLLM.Engine/`, alongside `PrefixTrieManager` / scheduler);
  integrate with paged path (episodes ≈ block groups) — note paged+quantized are
  currently exclusive, so this phase may need to land the paged↔quantized
  coexistence first. Layer-budget logic reads `ModelConfig` sliding-window info
  to avoid double-evicting sliding layers.
- **Acceptance:** bounded peak memory during long prefill; multi-turn accuracy
  at 4–6× token compression near full-cache on a LongConvQA-style eval; routing
  never silently drops a needed episode below a measured threshold.
- **Risk:** routing correctness (wrong-answer failure mode); clustering cost;
  composition with prefix-trie sharing and sliding-window layers.

---

## 5. Open Questions (need user input)

1. **Merge vs select.** Is the goal a *single* mixed-precision cache where the
   history codec is *selectable* (TurboQuant ⟂ OSCAR), or a genuine
   *stack* (OSCAR rotation + TurboQuant QJL debias on the same bytes)? The
   latter is research-grade and unproven. Recommendation: selectable first.
2. **Target bit-width / hardware.** dotLLM's primary GPU path is Vulkan
   (STRIX_HALO). OSCAR's kernels are CUDA/SGLang; INT2 dequant on Vulkan
   gfx1151 is unproven. Is INT2 (OSCAR) a CUDA-only feature for now, with
   TurboQuant 2.5–3.5-bit the cross-backend default?
3. **Calibration pipeline.** Are we willing to own an offline per-model
   calibration step + artifact distribution (OSCAR), or stay calibration-free
   (TurboQuant only) for v1?
4. **Paged + quantized coexistence.** EpiCache wants episodes ≈ block groups,
   but paged and quantized caches are currently mutually exclusive. Do we invest
   in unifying them (needed for EpiCache + serving), or scope EpiCache to the
   non-paged path first?
5. **Scope of EpiCache.** Is conversational/multi-turn (its design target) a
   priority workload for dotLLM, or is single-shot long-context more important
   (where plain quantization + sliding window may suffice)?
6. **Per-layer rotation tables for Gemma 4.** Confirm the two stride classes
   (sliding 8×256 vs global 2×512) are the only variants we must support, or
   whether other per-layer head_dim models are in scope.

---

## 6. Sources
- TurboQuant — arXiv:[2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026,
  Google/NYU).
- OSCAR — arXiv:[2605.17757](https://arxiv.org/abs/2605.17757) (Together AI);
  [MarkTechPost coverage](https://www.marktechpost.com/2026/05/25/together-ai-open-sources-oscar-an-attention-aware-2-bit-kv-cache-quantization-system-for-long-context-llm-serving/).
- EpiCache — arXiv:[2509.17396](https://arxiv.org/abs/2509.17396) (Apple);
  [Apple ML Research page](https://machinelearning.apple.com/research/epicache).
- Comparison / composition — MarkTechPost,
  [*The KV Cache Compression Race: TurboQuant vs OSCAR vs EpiCache*](https://www.marktechpost.com/2026/06/18/the-kv-cache-compression-race-turboquant-vs-oscar-vs-epicache/)
  (2026-06-18).
- dotLLM existing stack — `docs/KV_CACHE.md`,
  `src/DotLLM.Core/Attention/IKvCache.cs`, `IQuantizedKvCache.cs`,
  `src/DotLLM.Engine/KvCache/QuantizedKvCache.cs`,
  `src/DotLLM.Core/Configuration/KvCacheConfig.cs` / `KvCacheDType.cs`,
  `src/DotLLM.Core/Models/ModelConfig.cs` (`GetLayerHeadDim`, `GlobalHeadDim`,
  `PerLayerSlidingWindow`).
