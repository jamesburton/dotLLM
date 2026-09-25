# KV Phase 0 — Generalise the KV cache to per-layer stride

> **Status: PLAN ONLY.** No code changes accompany this document. It is the
> file-level implementation plan for "Phase 0" of `docs/KV-OPTIMIZATION-PLAN.md`
> (generalise the quantized/contiguous KV cache so different layers can have
> different KV geometry). All file:line references were verified against the
> `dev` integration branch on 2026-06-19. Where the prior planning notes /
> handoff were stale, this doc corrects them.

---

## 0. TL;DR — what's actually true on `dev`

| Backend | KV cache type | Per-layer stride today? | Gemma-4 distinct-head-dim with cache? |
|---|---|---|---|
| **Vulkan** | `VulkanKvCache` | **YES — already shipped** (`int[] _kvStride`, gemma4-aware factory) | **Works** (only backend that does) |
| **CPU** | `SimpleKvCache`, `QuantizedKvCache` | **No** — scalar `_headDim`/`_kvStride` | **Throws** `NotSupportedException` (`GuardKvCacheHeadDim`) |
| **CUDA** | `CudaKvCache`, `CudaQuantizedKvCache` | **No** — scalar `_kvStride` | **N/A** — gemma4 attention is cacheless (recomputes K/V each call) |

The single most important correction to the stale notes: **Phase 0 is already
done on Vulkan.** The handoff line "per-layer KV cache for gemma4 landed" refers
to `VulkanKvCache` only. The roadmap's Phase 0 entry (which says "Touch
`QuantizedKvCache.cs` + `CudaQuantizedKvCache`") is **aimed at the wrong file
for the wrong reason** — see §1.3. Phase 0's real remaining job is:

1. Lift the per-layer-stride concept that Vulkan already proved into a
   **Core-level descriptor type** so it is not Vulkan-private.
2. Generalise **`SimpleKvCache`** (CPU) to per-layer stride — this is the cache
   gemma-4 CPU decode actually needs, and it is the prerequisite the
   `GuardKvCacheHeadDim` throw is blocking.
3. Generalise **`QuantizedKvCache`** (CPU) + **`CudaQuantizedKvCache`** to
   per-layer stride — required later by TurboQuant/OSCAR, lower priority for
   gemma-4 correctness (gemma-4 runs F32 KV on Vulkan today).
4. Decide CUDA's stance: gemma-4 attention is cacheless on CUDA, so a per-layer
   CUDA KV cache is **not on the gemma-4 critical path** — it is only needed if
   CUDA gains a gemma-4 *decode* path or for cross-backend symmetry.

---

## 1. What exists vs what is missing (verified)

### 1.1 Vulkan — per-layer stride ALREADY implemented (reference design)

`src/DotLLM.Vulkan/VulkanKvCache.cs`
- Field `private readonly int[] _kvStride;` (line 46) — one stride per layer.
- Two constructors:
  - `VulkanKvCache(device, numLayers, numKvHeads, headDim, maxSeqLen)` (line 61)
    — uniform; builds an `int[]` filled with `numKvHeads*headDim` via
    `BuildUniformStrides` (line 98). **This is the byte-identical default path.**
  - `VulkanKvCache(device, int[] kvStridePerLayer, maxSeqLen)` (line 72) — the
    per-layer path. Allocates `_keys[i]/_values[i]` sized `maxSeqLen * _kvStride[i]`.
- All offset math already indexes by layer: `rowBytes = _kvStride[layerIndex] * sizeof(float)`
  in `UpdateDevice` (line 129), `RecordUpdate` (line 181), `IngestFromHost` (line 327).
- Public accessors: `KvStride` (layer-0 shortcut, line 289) and
  `KvStrideOf(int layerIndex)` (line 292).

`src/DotLLM.Vulkan/VulkanTransformerModel.cs`
- `CreateKvCache(int maxSeqLen)` (line 329): when `Config.Gemma4DualFfn`, builds
  `strides[l] = GemmaLayerKvHeads(l) * Config.GetLayerHeadDim(l)` and calls the
  per-layer constructor; otherwise the uniform constructor.
- `GemmaLayerKvHeads(l)` (line 1056) and `Config.GetLayerHeadDim(l)` resolve the
  sliding-vs-global geometry.

**Regression coverage already exists**:
`tests/DotLLM.Tests.Integration/Engine/Gemma4VulkanKvCacheTests.cs` —
`Vulkan_Gemma4_KvCacheDecode_MatchesCachelessForward`. It deliberately uses
`GlobalKvHeads = 2` so sliding stride (2×16=32) ≠ global stride (2×32=64),
asserts `probe.KvStrideOf(0) != probe.KvStrideOf(last)` *before* the parity
check, then proves prefill-then-decode == cacheless argmax. This is the
discriminating template for the CPU test (§5).

### 1.2 Core — no per-layer-stride concept exists yet

`src/DotLLM.Core/Attention/IKvCache.cs` — the universal seam (`Update` /
`GetKeys[Ref]` / `GetValues[Ref]` / `Rollback`). Carries **no stride / geometry
metadata at all** (geometry is implied by the `TensorRef` row count the
implementation returns). Nothing to change for correctness, but it means the
"per-layer stride" idea currently lives only inside `VulkanKvCache` as a private
`int[]`. There is **no Core type** describing per-layer KV geometry.

`src/DotLLM.Core/Models/ModelConfig.cs`
- `GetLayerHeadDim(int layerIdx)` (line 260) — returns `GlobalHeadDim` for
  full-attention layers, else `HeadDim`. Collapses to uniform when
  `GlobalHeadDim is null`.
- `IsFullAttentionLayer(int layerIdx)` (line 277).
- `GlobalHeadDim` (line 249), `PerLayerSlidingWindow` (line 81),
  `NumGlobalKvHeads` (referenced by callers).
- **Missing**: there is no `GetLayerKvHeads(int)` on `ModelConfig` itself. Each
  backend re-derives it privately (`VulkanTransformerModel.GemmaLayerKvHeads`
  line 1056; `TransformerModel.GetLayerKvHeads` line 667;
  CUDA inline at `CudaTransformerModel.cs` line 1873). Promoting this to
  `ModelConfig` is the cleanest way to feed every cache factory (see §2.1).

`src/DotLLM.Core/Configuration/KvCacheConfig.cs` / `KvCacheDType.cs` —
`(KeyDType, ValueDType, MixedPrecisionWindowSize)`; enum `F32/Q8_0/Q4_0`. No
change required for Phase 0 (codecs are Phase 1+).

### 1.3 CPU — scalar; gemma-4 + cache is explicitly blocked

`src/DotLLM.Engine/KvCache/SimpleKvCache.cs`
- Scalar `_headDim` (line 18), `_kvStride = numKvHeads * headDim` (line 46).
- One buffer per layer, all sized `maxSeqLen * _kvStride` (line 51-56).
- Every offset uses the scalar stride: `Update` (line 73, 84-92),
  `GetKeysRef`/`GetValuesRef` (line 111/116), `KeysSpan`/`ValuesSpan` (line 142/154).
- Public `KvStride` (line 134) used by the hybrid CPU-prefill→Vulkan-decode handoff.

`src/DotLLM.Engine/KvCache/QuantizedKvCache.cs`
- Scalar `_headDim` (line 30), `_kvStride` (line 32, `= numKvHeads*headDim` line 88).
- **Constructor guard `if (_kvStride % BlockSize != 0) throw` (line 93)** keyed
  to the single stride. (For gemma-4: 256 and 512 are both %32==0, so the guard
  itself is not the blocker — the single-value storage is.)
- `_keyQuantRowBytes`/`_valueQuantRowBytes` (line 100-101) are single scalars.
- All quant buffers sized from the scalar `_keyQuantRowBytes` (line 112-116).
- `Update` offset math (line 146, 163-168, 181-189, 204-209) all use scalar
  `_kvStride`/`_keyQuantRowBytes`.
- `_layerQuantizedLength[]` (line 36) — **already per-layer**, but that tracks
  eviction *count*, not *stride*. Roadmap §3.2's "per-layer eviction tracking
  already present" is true but unrelated to per-layer stride.

`src/DotLLM.Models/Architectures/TransformerModel.cs` (CPU forward)
- `HasDistinctPerLayerHeadDim` (line 369): `GlobalHeadDim is int ghd && ghd != HeadDim`.
- **`GuardKvCacheHeadDim(IKvCache?)` (line 383)** — **throws `NotSupportedException`
  whenever a distinct-per-layer-head-dim model is given any KV cache.** This is
  THE gate Phase 0 removes on CPU. Message: "supported on the cacheless forward
  path only … Per-layer KV-cache strides are tracked as future work."
- `GetLayerKvHeads(layer)` (line 667); per-layer head dim used in attention at
  lines 836-841 and 2339; a second guard/throw at line 2253-2258 for the
  batched gemma-4 path.
- CPU has **no `CreateKvCache` method** — callers (`TextGenerator.cs` line 1032/1142,
  `HybridPrefillDecodeStrategy.cs` line 186, tests, benchmarks) construct
  `SimpleKvCache`/`QuantizedKvCache` directly with `(NumLayers, NumKvHeads,
  HeadDim, …)`. There is no single factory to update — see §2.2 risk.

### 1.4 CUDA — scalar; gemma-4 is cacheless (cache not on critical path)

`src/DotLLM.Cuda/CudaKvCache.cs` — scalar `_kvStride` (line 16, `= numKvHeads*headDim`
line 36); FP16 storage; offsets at lines 62, 79-95, 134, 188.

`src/DotLLM.Cuda/CudaQuantizedKvCache.cs` — scalar `_kvStride` (line 28, line 78);
same `%BlockSize` guard (line 84); scalar `_keyQuantRowBytes`/`_valueQuantRowBytes`
(line 31-32); single dequant scratch `_kScratch`/`_vScratch` sized from scalar
stride (line 38-39).

`src/DotLLM.Cuda/CudaTransformerModel.cs`
- `CreateKvCache(maxSeqLen)` (line 2171) → `new CudaKvCache(NumLayers, NumKvHeads,
  HeadDim, …)`; `CreateKvCache(maxSeqLen, KvCacheConfig)` (line 2182) → `CudaKvCache`
  or `CudaQuantizedKvCache`. **Both use scalar `Config.HeadDim`.**
- `RunGemma4AttentionF32` (line 1867) — resolves `headDim = Config.GetLayerHeadDim(layer)`
  and `numKvHeads` per layer (line 1872-1874), projects K/V fresh into
  `_state.KF32`/`_state.VF32`, calls `LaunchAttentionF32` over the full `seqLen`
  (line 1928) and **never touches a KV cache.** Confirmed by reading lines
  1920-1934: there is no `CudaKvCache` read/write in the gemma-4 path.
- Therefore CUDA gemma-4 = cacheless forward only (matches handoff: CUDA was
  validated via synthetic forward parity, not decode).

---

## 2. Core interface / type changes

### 2.1 New Core descriptor type: `KvGeometry` (the load-bearing change)

Add `src/DotLLM.Core/Attention/KvGeometry.cs`:

```csharp
namespace DotLLM.Core.Attention;

/// <summary>
/// Per-layer KV-cache geometry. For every dense/GQA/MoE model this is uniform
/// (one stride repeated across layers); for Gemma-4 the sliding and global
/// layers carry different KV-head counts and head dims, so each layer's cached
/// K/V row is a different width.
/// </summary>
public readonly struct KvGeometry
{
    private readonly int[] _kvStridePerLayer;   // numKvHeads(l) * headDim(l)

    public int LayerCount => _kvStridePerLayer.Length;
    public int KvStrideOf(int layer) => _kvStridePerLayer[layer];
    public bool IsUniform { get; }              // fast path flag
    public int UniformStride { get; }           // valid only when IsUniform

    public static KvGeometry Uniform(int numLayers, int numKvHeads, int headDim);
    public static KvGeometry PerLayer(int[] kvStridePerLayer);

    // The single helper every backend factory calls instead of re-deriving:
    public static KvGeometry FromConfig(ModelConfig config);
}
```

`FromConfig` centralises the logic that is currently triplicated
(`VulkanTransformerModel.GemmaLayerKvHeads`, `TransformerModel.GetLayerKvHeads`,
CUDA inline). It returns `Uniform` when `GlobalHeadDim is null` (so non-gemma
models get the identical scalar path) and `PerLayer` otherwise. To support
`FromConfig`, **also add `ModelConfig.GetLayerKvHeads(int)`** to
`ModelConfig.cs` (mirrors the existing `GetLayerHeadDim`), and have the three
backend-private copies delegate to it (de-dup, single source of truth — required
by the cross-backend rule).

`IsUniform` exists so hot paths can keep a single scalar local and avoid an
array index per row when the model is uniform (≈ all current models).

### 2.2 Cache constructors gain a per-layer overload (additive, default unchanged)

For each scalar cache, **keep the existing `(…, numKvHeads, headDim, …)`
constructor byte-identical** (it stays the public default), and add a
`KvGeometry`-taking constructor that the existing one forwards to via
`KvGeometry.Uniform(...)` — exactly the pattern `VulkanKvCache` already uses
(`BuildUniformStrides`). New constructors:

- `SimpleKvCache(KvGeometry geometry, int maxSeqLen)`
- `QuantizedKvCache(KvGeometry geometry, int maxSeqLen, KvCacheDType keyDType, KvCacheDType valueDType, int windowSize)`
- `CudaKvCache(KvGeometry geometry, int maxSeqLen)`
- `CudaQuantizedKvCache(KvGeometry geometry, int maxSeqLen, KvCacheConfig config)`

(`VulkanKvCache` already has the equivalent — optionally refactor its `int[]`
ctor to take `KvGeometry` for consistency; not required.)

### 2.3 `IQuantizedKvCache` per-layer row bytes

`src/DotLLM.Core/Attention/IQuantizedKvCache.cs` currently exposes scalar
`KeyQuantizedRowBytes` / `ValueQuantizedRowBytes` (line 54/59). For a per-layer
quantized cache these become layer-dependent. Add per-layer accessors:

```csharp
int KeyQuantizedRowBytesOf(int layerIndex);
int ValueQuantizedRowBytesOf(int layerIndex);
```

Keep the existing scalar properties (default them to layer-0 for uniform caches,
or mark them as "uniform-only" and have the CPU attention kernel switch to the
`*Of(layer)` form). The CPU consumer
`src/DotLLM.Cpu/Kernels/Attention.cs::Execute(float* q, IQuantizedKvCache, …)`
(line 886) reads `kvCache.QuantizedLength`, `GetWindowKeysPtr`, and computes
`kvStride = numKvHeads * headDim` from its **call arguments** (line 904) — so the
kernel is *already parameterised per call*; the only change is that the caller
must pass the per-layer `numKvHeads`/`headDim`, and the quantized row-byte sizes
must come from `*Of(layer)` rather than the scalar property.

---

## 3. Per-backend change list (files · methods · offset math)

### 3.1 Core (`DotLLM.Core`)
- **NEW** `Attention/KvGeometry.cs` — the descriptor (§2.1).
- `Models/ModelConfig.cs` — add `GetLayerKvHeads(int)` (mirror `GetLayerHeadDim`);
  no change to existing members.
- `Attention/IQuantizedKvCache.cs` — add `KeyQuantizedRowBytesOf/ValueQuantizedRowBytesOf(int)`.
- `Attention/IKvCache.cs` — **no change** (geometry stays out of the universal seam).

### 3.2 Engine — CPU caches (`DotLLM.Engine`)
- `KvCache/SimpleKvCache.cs`:
  - Replace scalar `_headDim`/`_kvStride` with `KvGeometry _geom` (+ cached
    `_uniformStride` when `_geom.IsUniform`).
  - Per-layer buffer sizing: `bufferBytes(i) = maxSeqLen * _geom.KvStrideOf(i)`
    (was uniform line 51).
  - `Update` (line 63): `int stride = _geom.KvStrideOf(layerIndex); int rowBytes
    = stride * sizeof(float);` use `stride` in both `MemoryCopy` offsets (lines
    84-92, currently `_kvStride`).
  - `GetKeysRef`/`GetValuesRef` (line 110/115), `KeysSpan`/`ValuesSpan` (line
    142/154), `AllocatedBytes` (line 31): all switch to per-layer stride.
  - `KvStride` (line 134): keep for uniform; add `KvStrideOf(int)` (the hybrid
    handoff in `VulkanKvCache.IngestFromHost` consumes contiguous host rows —
    must agree on per-layer stride end-to-end).
- `KvCache/QuantizedKvCache.cs`:
  - Replace scalar `_headDim`/`_kvStride`/`_keyQuantRowBytes`/`_valueQuantRowBytes`
    with `KvGeometry _geom` + `int[] _keyQuantRowBytes`/`_valueQuantRowBytes`.
  - Move the `% BlockSize` guard (line 93) into a per-layer loop (validate each
    `_geom.KvStrideOf(l)`).
  - Per-layer quant buffer sizing (line 112-116) and per-layer window sizing
    (line 124) from per-layer stride.
  - `Update` (line 141): hoist `int stride = _geom.KvStrideOf(layerIndex)` and
    use it for `fpRowBytes`, the eviction `QuantizeRow` calls (line 163-168,
    note `ringIdx * stride` and `evictPos * _keyQuantRowBytes[layer]`), the
    window ring writes (line 182-189), and the pure-quantized path (line 204-209).
  - Implement `KeyQuantizedRowBytesOf/ValueQuantizedRowBytesOf`.
- `TextGenerator.cs` (line 1032, 1142) + `HybridPrefillDecodeStrategy.cs` (line
  186): switch the `new SimpleKvCache(NumLayers, NumKvHeads, HeadDim, …)` call
  sites to `new SimpleKvCache(KvGeometry.FromConfig(cfg), size)`.

### 3.3 Models — CPU forward (`DotLLM.Models`)
- `Architectures/TransformerModel.cs`:
  - **Remove / relax `GuardKvCacheHeadDim` (line 383)** — once `SimpleKvCache`
    supports per-layer stride, a distinct-head-dim model + cache is valid. Replace
    the unconditional throw with a check that the supplied cache's geometry
    matches `KvGeometry.FromConfig(Config)` (fail fast on a *mismatched* cache
    rather than on the *concept*).
  - Same for the batched-path guard at line 2253-2258.
  - The attention sites already compute per-layer `numKvHeads`/`headDim` (lines
    836-841, 2339) — verify they pass the layer's geometry into the cache
    `Update`/read calls. The KV write must land at `KvStrideOf(layer)`-wide rows.
  - `GetLayerKvHeads` (line 667) → delegate to `Config.GetLayerKvHeads`.
- **A CPU `CreateKvCache(ModelConfig, …)` factory does not exist.** Either add one
  on `TransformerModel` (preferred — single place to call
  `KvGeometry.FromConfig`) or update each external call site. Adding the factory
  is lower-risk and mirrors Vulkan/CUDA.

### 3.4 CPU attention kernel (`DotLLM.Cpu`)
- `Kernels/Attention.cs::Execute(float* q, IQuantizedKvCache kvCache, …)` (line
  886): replace the scalar `KeyQuantizedRowBytes`/`ValueQuantizedRowBytes` reads
  with `kvCache.KeyQuantizedRowBytesOf(layerIndex)` etc.; ensure the
  `numKvHeads`/`headDim` passed in are the layer's values (caller already has
  them). `kvStride` is computed locally (line 904) so no struct field change.
  The non-quantized `Execute` overloads already take `numKvHeads`/`headDim`
  per-call (lines 121, 605) — no change beyond the caller passing per-layer
  values.

### 3.5 Vulkan (`DotLLM.Vulkan`)
- **No functional change required** — `VulkanKvCache` is the reference. Optional:
  refactor its private `int[] _kvStride` + two constructors to consume the new
  `KvGeometry` for symmetry, and have `VulkanTransformerModel.CreateKvCache`
  (line 329) call `KvGeometry.FromConfig(Config)` instead of building `strides[]`
  inline (de-dups against the new Core helper). Verify the existing
  `Gemma4VulkanKvCacheTests` still passes unchanged.

### 3.6 CUDA (`DotLLM.Cuda`)
- **Decision point (see §4 / Risks).** gemma-4 CUDA attention is cacheless, so a
  per-layer CUDA KV cache is **not required for gemma-4 correctness**. Two options:
  1. **Minimal (recommended for Phase 0):** generalise `CudaKvCache` /
     `CudaQuantizedKvCache` constructors to accept `KvGeometry` (forwarding the
     scalar ctor through `KvGeometry.Uniform`) **without** wiring a gemma-4 CUDA
     decode path — keeps the cross-backend type surface consistent and unblocks
     a future CUDA gemma-4 decode, but ships no behavioural change on CUDA. Update
     `CudaKvCache.cs` (`_kvStride` → `_geom`, offsets at lines 62/79-95/134/188),
     `CudaQuantizedKvCache.cs` (scalar fields → per-layer; per-layer dequant
     scratch `_kScratch`/`_vScratch` sized to `max(stride)`), and
     `CudaTransformerModel.CreateKvCache` (line 2171/2182) to pass
     `KvGeometry.FromConfig`.
  2. **Defer CUDA entirely:** leave CUDA scalar; document that gemma-4 on CUDA is
     cacheless-only. The cross-backend rule (CLAUDE.md) says fix all backends
     when a *correctness bug* is duplicated — this is a *capability extension*,
     not a bug, and CUDA has no broken cached gemma-4 path to fix. Option 2 is
     defensible **provided** the `KvGeometry` type and `IQuantizedKvCache`
     additions are still made (so CUDA compiles against the new interface).
- Recommendation: **Option 1, scratch-buffer sizing only** — cheap, keeps the
  type surface uniform, no new CUDA kernel work. Do NOT attempt a CUDA gemma-4
  decode path in Phase 0.

---

## 4. Migration: keep single-geometry models byte-identical

This is the critical constraint. The design guarantees it structurally:

1. **`KvGeometry.Uniform(n, kvHeads, headDim)` produces `[kvHeads*headDim] × n`**
   — identical to the scalar `_kvStride`. The `IsUniform`/`UniformStride` fast
   path means a uniform cache hoists one scalar local and runs the *same offset
   arithmetic* as today (no per-row array index). Vulkan already proves this:
   the uniform `VulkanKvCache` ctor fills an `int[]` and the gemma-4 decode test
   plus all existing dense/Qwen/Llama Vulkan tests are green on `dev`.
2. **Existing scalar constructors stay public and unchanged**, forwarding to the
   `KvGeometry` ctor via `Uniform(...)`. Every current call site (TextGenerator,
   tests, benchmarks, ServerStartup line 160/166, CrossBackendTimingHarness)
   keeps compiling and behaving identically.
3. **`KvGeometry.FromConfig` returns `Uniform` when `GlobalHeadDim is null`** —
   i.e. for every non-gemma-4 model. Only gemma-4 takes the per-layer branch.
4. **The `% BlockSize` guard semantics are preserved** per layer (uniform models
   hit the exact same check on the exact same value).
5. **Numerical path unchanged**: per-layer stride only changes *addressing*, not
   the quantize/dequantize math or the attention reduction. A uniform model
   addresses identically, so logits are bit-identical.

Acceptance for "byte-identical": run the full existing CPU + Vulkan suites and
confirm zero diffs in dense/Llama/Qwen logits and KV-cache unit tests
(`SimpleKvCacheTests`, `QuantizedKvCacheTests`, `CudaQuantizedKvCacheGraphTest`).

---

## 5. Test plan — discriminating regression tests

A degenerate equal-stride fixture **cannot** catch a per-layer-stride bug (the
Vulkan test's own remark and CLAUDE.md both warn this). Each new test must make
sliding-stride ≠ global-stride and assert it before relying on parity.

### 5.1 CPU `SimpleKvCache` per-layer-stride decode parity (PRIMARY)
New: `tests/DotLLM.Tests.Unit/Engine/KvCache/SimpleKvCachePerLayerStrideTests.cs`
- Construct `SimpleKvCache(KvGeometry.PerLayer([32, 64]), maxSeqLen)` (two layers,
  distinct strides). Assert `KvStrideOf(0) != KvStrideOf(1)`.
- Write known K/V rows to each layer at several positions, read back via
  `GetKeysRef/GetValuesRef`/`KeysSpan`, assert exact recovery **per layer**. A
  scalar cache would mis-address layer 1 (reading/writing at the layer-0 stride),
  corrupting the round-trip — this discriminates.
- Negative control: the same test with `PerLayer([32,32])` (degenerate) must also
  pass, proving the per-layer path collapses to correct uniform behaviour.

### 5.2 CPU gemma-4 cached decode == cacheless forward (END-TO-END)
New: `tests/.../Engine/Gemma4CpuKvCacheTests.cs` — mirror the **existing Vulkan**
`Gemma4VulkanKvCacheTests` exactly (it is the proven oracle):
- Synthetic gemma-4 with `GlobalKvHeads = 2` so sliding (2×16=32) ≠ global
  (2×32=64); assert the strides differ.
- Oracle = single cacheless `Forward` over the whole sequence (the gemma-4 CPU
  cacheless forward is already trusted — `TransformerModelGemma4MoeForwardTests`).
- Under test = prefill `[0,last)` into a per-layer `SimpleKvCache`, then decode
  the last token. Assert argmax equality (structural) + per-logit envelope
  (`absTol≈6e-2, relTol≈5e-3`, the same drift bound the Vulkan test uses).
- **This test fails today** (the `GuardKvCacheHeadDim` throw) and passes only
  after Phase 0 — it is the acceptance gate.

### 5.3 CPU `QuantizedKvCache` per-layer-stride round-trip
Extend `QuantizedKvCacheTests.cs` with a `PerLayer([32,64])` Q8_0 case:
quantize→dequant per layer, assert per-layer error bound. Discriminates the
per-layer `_keyQuantRowBytes[]` indexing.

### 5.4 `KvGeometry.FromConfig` unit test
- gemma-4 config → non-uniform geometry with the expected per-layer strides
  (sliding `NumKvHeads*HeadDim`, global `NumGlobalKvHeads*GlobalHeadDim`).
- A Llama config → `IsUniform == true`, `UniformStride == NumKvHeads*HeadDim`.

### 5.5 (If CUDA Option 1) CUDA per-layer constructor smoke
Extend `CudaQuantizedKvCacheGraphTest` with a `KvGeometry.PerLayer` construction
+ per-layer scratch-size assertion (SkippableFact, GPU-gated).

### 5.6 Regression-guard: existing suites unchanged
`SimpleKvCacheTests`, `PagedKvCacheTests`, `QuantizedKvCacheTests`,
`CudaQuantizedKvCacheGraphTest`, dense/Llama/Qwen forward parity — must stay
green with zero numeric change (proves the byte-identical migration).

---

## 6. Risks

1. **Hybrid prefill→decode handoff stride agreement.** `SimpleKvCache.KeysSpan`
   feeds `VulkanKvCache.IngestFromHost`, which sizes by `_kvStride[layer]`. Both
   sides must use the *same* per-layer stride or the upload silently misaligns
   for gemma-4. Mitigation: drive both from `KvGeometry.FromConfig` and add an
   ingest-stride assertion. (Today this path is dense-only, so the risk is latent
   until gemma-4 uses hybrid decode.)
2. **No CPU `CreateKvCache` factory.** Many scattered `new SimpleKvCache(...)`
   call sites. Risk of missing one and constructing a uniform cache for a
   gemma-4 model (which would then mis-address). Mitigation: add the factory +
   the geometry-mismatch assertion in `Forward` (replaces the blanket guard).
3. **`IQuantizedKvCache` interface change** ripples to every implementer
   (`QuantizedKvCache`, `CudaQuantizedKvCache`) and consumer (CPU attention,
   any GPU dequant). Keep the scalar properties to limit churn; add the `*Of`
   methods alongside.
4. **CUDA scope creep.** Tempting to add a CUDA gemma-4 *decode* path while
   touching the caches. Out of scope for Phase 0 — gemma-4 CUDA is cacheless and
   VRAM-gated on the T5500 anyway. Keep CUDA to constructor/scratch generalisation.
5. **Block-size constraint for future odd head_dims.** 256/512 are %32==0, but a
   per-layer cache must validate *each* layer's stride. Low risk now, real once
   arbitrary per-layer head dims appear.
6. **Paged + quantized exclusivity** (`PagedKvCache`) is unrelated to Phase 0 but
   note `PagedKvCacheFactory` also assumes uniform geometry — out of scope, flag
   for Phase 3/EpiCache.

---

## 7. Ordered task breakdown

**Do first (unblocks everything, no behaviour change):**
1. Add `KvGeometry` (Core) + `ModelConfig.GetLayerKvHeads` + `KvGeometry.FromConfig`.
   Unit-test `FromConfig` (§5.4). Pure addition, nothing consumes it yet.
2. Add `IQuantizedKvCache.*Of(int)` methods; implement on existing caches as
   layer-0 passthrough (still uniform). Suites stay green.

**Then (the gemma-4 CPU correctness payoff — sequential, shares files):**
3. Generalise `SimpleKvCache` to `KvGeometry` (scalar ctor forwards). Add §5.1.
4. Add CPU `CreateKvCache` factory; replace `GuardKvCacheHeadDim` throw with a
   geometry-match assertion; thread per-layer geometry through CPU gemma-4
   attention. Add the end-to-end §5.2 (the acceptance gate).

**Parallelisable after step 1 (independent files, separate agents OK):**
5. Generalise CPU `QuantizedKvCache` + CPU attention quantized consumer (§3.2/§3.4).
   Add §5.3. — *Agent A*
6. (Optional, recommended) Vulkan refactor to consume `KvGeometry` (§3.5);
   re-run existing `Gemma4VulkanKvCacheTests`. — *Agent B*
7. CUDA Option 1 constructor/scratch generalisation (§3.6) + §5.5. — *Agent C*

**Last:**
8. Full CPU + Vulkan suite run; confirm byte-identical dense logits (§4/§5.6);
   update `docs/KV-OPTIMIZATION-PLAN.md` Phase 0 to "done" and correct its file
   list (it currently points only at `QuantizedKvCache`/`CudaQuantizedKvCache`
   and omits the real blocker, `SimpleKvCache` + `GuardKvCacheHeadDim`).

**Recommended first implementation step:** task 1 — land `KvGeometry` +
`ModelConfig.GetLayerKvHeads` + `KvGeometry.FromConfig` with its unit test. It is
a pure, side-effect-free addition that every later step depends on, de-duplicates
the three existing private per-layer-kv-head derivations, and lets the CPU/CUDA
work proceed in parallel against a stable Core type.
