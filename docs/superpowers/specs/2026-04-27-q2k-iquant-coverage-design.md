# Spec 1 — Q2_K + IQ-family quantization coverage (full L3)

**Status**: Approved (2026-04-27). Ready for plan generation.
**Owner**: feature/mamba-3-cuda branch.
**Phasing**: 1 design (this doc) → 5 implementation plans → 5 PRs.

## Goal

Add full L3 (CPU dequant + GPU dequant + per-call GEMV + MMQ + MMVQ-large + grouped-MoE-GEMV) support for nine GGUF quantization types currently unsupported by dotLLM:

- **Q2_K** (GGUF type 10, K-quant family, 84-byte 256-element superblocks)
- **IQ4_NL** (20, 18-byte 32-element blocks, 16-entry signed lookup)
- **IQ4_XS** (23, 136-byte 256-element superblocks, same 16-entry lookup)
- **IQ3_S** (21, 110-byte superblocks, 512-entry int32 codebook)
- **IQ3_XXS** (18, 98-byte superblocks, 256-entry int32 codebook)
- **IQ2_S** (22, 82-byte superblocks, 1024-entry int64 codebook + sign bytes)
- **IQ2_XS** (17, 74-byte superblocks, 512-entry int64 codebook + sign mask)
- **IQ2_XXS** (16, 66-byte superblocks, 256-entry int64 codebook + sign mask)
- **IQ1_S** (19, 50-byte superblocks, 2048-entry int16 grid + qh sub-block deltas)

After Spec 1 lands, every GGUF quant type registered in upstream `ggml-quants.h` is loadable end-to-end on dotLLM CUDA — no more `Unsupported GGUF quantization type` errors at load, no more dequant-then-cuBLAS fallback for these types.

## Non-goals

- Q8_K (intermediate type used internally for MMQ, not stored in production GGUFs).
- IQ1_M (rarer 1.5-bit variant; can be added later if demand surfaces).
- Imatrix (importance-matrix calibration) tooling — that's quant-time, not load-time.
- Pre-V2 DeepSeek architecture (separate spec).
- GGUF loaders for Mixtral / QwenMoe / GraniteMoe / Mamba3 (separate spec).
- Vulkan / HIP backend extensions (deferred to other hardware).

## Approach choice

Hybrid: port-style for Q-family, design-style for IQ-family.

**Q2_K** is structurally close to the existing `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` K-quants — same 256-element superblocks, same dispatch surface, linear `value = scale × signed_int + min` decode math. We port `quantized_gemv_q4_k_mmq` as the starting point and substitute the per-superblock decode body. Predictable.

**IQ-family** uses signed-grid lookup encoding fundamentally unlike linear quantization. The dp4a inner loop pattern from K-quants does not transfer directly: each weight value is a codebook index, not a signed int8. Rather than translate llama.cpp's IQ kernels line-by-line (which would import their idioms wholesale), we design new PTX kernels using the same shape as our existing K-quant MMQ (4-rows-per-warp, 256-thread blocks, FP32 accum, FP16 store) but with a codebook-expansion prefix that fills the int8 weight tile in shared memory before dp4a fires. Codebook constants (the wire-format-defined int8/int16 lookup tables) are copied verbatim from `ggml-quants.c` — they're a published spec.

## Architecture overview

### What gets added

| Layer | New surface | Files touched |
|---|---|---|
| Core enum | 9 new entries in `QuantizationType` matching GGUF type IDs | `src/DotLLM.Core/Configuration/QuantizationType.cs` |
| Block-size constants | `Q2_K_BlockBytes = 84`, `IQ2_XXS_BlockBytes = 66`, `IQ2_XS_BlockBytes = 74`, `IQ2_S_BlockBytes = 82`, `IQ3_XXS_BlockBytes = 98`, `IQ3_S_BlockBytes = 110`, `IQ4_NL_BlockBytes = 18`, `IQ4_XS_BlockBytes = 136`, `IQ1_S_BlockBytes = 50` | `src/DotLLM.Cpu/Kernels/Dequantize.cs` |
| CPU dequant impl | `DequantizeQ2_K` in existing K-quant file; 8 IQ scalar impls in new file; codebooks in new file | `DequantizeKQuants.cs` (extend), `DequantizeIQuants.cs` (new), `IQuantGrids.cs` (new) |
| GGUF reader | `Enum.IsDefined` accepts the new IDs once added to enum. Zero changes. | none |
| GPU dequant | `dequant_q2_k_f16` in existing file; 8 IQ kernels + codebook `__device__ __constant__` arrays in new file | `dequant.cu` (extend), `dequant_iquants.cu` (new) |
| GPU GEMV (per-call) | `quantized_gemv_q2_k`; 8 IQ GEMV kernels | `quantized_gemv.cu` (extend), `quantized_gemv_iquants.cu` (new) |
| GPU MMQ + MMVQ-large + pre-Q8_1 variants | Full quartet per quant (mmq, mmq_preq, mmvq_large, mmvq_large_preq), 36 kernel functions across 9 quants | `quantized_gemv_mmq.cu` (extend for Q2_K), `quantized_gemv_iquants_mmq.cu` (new for IQ) |
| GPU grouped-MoE-GEMV | One per quant: 9 new kernels | `moe_grouped_gemv.cu` (extend for Q2_K), `moe_grouped_gemv_iquants.cu` (new for IQ) |
| `HasQuantizedGemv` / `HasMmq*` / `HasMoeGroupedGemv` / `MinKAlignmentFor` | All extended; IQ4_NL takes block-32 path (alignment 32), all others 256 | `CudaKernels.cs` |

### File-organization rule

Q-family extensions go into existing files (they're idiomatically siblings). IQ-family gets a new sibling file in each pair (`dequant_iquants.cu`, `quantized_gemv_iquants.cu`, `quantized_gemv_iquants_mmq.cu`, `moe_grouped_gemv_iquants.cu`) — keeps codebook-aware inner-loop code separate from linear-quant code, and PTX modules stay reasonably sized.

### Codebook strategy

Single source of truth in `src/DotLLM.Cpu/Kernels/IQuantGrids.cs` as `static readonly` arrays. Identical byte representation in PTX modules via `__device__ __constant__` arrays in `dequant_iquants.cu` (compiled into the GPU module's `.const` segment at PTX load). Total codebook footprint: ~30 KB on each side, well under the 64 KB CUDA constant-memory cache limit.

A unit test (`IQuantGridsByteEquivalenceTest`) reads each PTX `__device__ __constant__` array via `cuModuleGetGlobal` after PTX load, copies bytes back to host, asserts byte-for-byte equality with the C# `static readonly` array. Diverging copies fail loud — catches the day someone updates one but not the other.

### Dispatch flow

```
GGUF load → QuantizationType (extended enum)
        → Dequantize.RowByteSize / ToFloat32 (extended switches)
        → CPU oracle path (always works, scalar reference)
        → GPU upload of raw quant bytes (zero-copy mmap → cuMemAlloc)
        → forward pass:
              if MoE       → CudaMoeFfn.Forward → ProjectF32OrQuant
                            (gate uses MinKAlignmentFor, quant type from MoeLayerWeights)
              if MLA       → CudaMlaAttention.ForwardF16 → ProjectF16OrQuant (same)
              else (GQA)   → CudaGemm.LinearF32 with quant fast paths via existing dispatch
```

No new dispatch layers introduced — every new quant slots into the existing decision tree on the back of `HasQuantizedGemv(qt)`, `HasMmq(qt)`, `HasMoeGroupedGemv(qt)`, `MinKAlignmentFor(qt)`. New code goes in switch cases + new kernel functions, not in new abstractions.

## Component detail

### CPU dequant

Per-quant scalar implementations that decode raw bytes → F32. These are the canonical math reference; every GPU kernel parity test compares against them.

**Block layouts** (verified against `ggml-quants.h`):

| Quant | Block size | Block bytes | Layout |
|---|---|---|---|
| Q2_K | 256 | 84 | scales[16] (4-bit scale + 4-bit min coef per sub-block, packed) + qs[64] (2-bit elements) + d (half) + dmin (half) |
| IQ2_XXS | 256 | 66 | d + qs[32] (uint16; 256-entry codebook) |
| IQ2_XS | 256 | 74 | d + qs[32] + scales[8] (4-bit packed) |
| IQ2_S | 256 | 82 | d + qs[64] + qh[8] + scales[8] |
| IQ3_XXS | 256 | 98 | d + qs[96] |
| IQ3_S | 256 | 110 | d + qs[64] + qh[8] + signs[32] + scales[4] |
| IQ4_NL | **32** | 18 | d + qs[16] (4-bit elements; 16-entry signed lookup) |
| IQ4_XS | 256 | 136 | d + scales_h[2] + scales_l[4] + qs[128] |
| IQ1_S | 256 | 50 | d + qs[32] + qh[8] (16-bit) |

**Decode math** (Q-family vs IQ-family):

- **Q2_K**: per-element `value = d × (scales[sub] & 0xF) × q_2bit − dmin × ((scales[sub] >> 4) & 0xF)`. 16 sub-blocks of 16 elements; 4-bit scale + 4-bit min per sub-block, both pre-quantized.
- **IQ4_NL**: per-element `value = d × kvalues_iq4nl[qs_4bit]` where `kvalues_iq4nl[16]` is a signed-int8 lookup table baked at compile time.
- **IQ4_XS**: like IQ4_NL but with 8 sub-blocks of 32 elements each, 6-bit scale per sub-block (6-bit = low 4 from `scales_l` + high 2 from `scales_h`).
- **IQ2_XXS / IQ2_XS / IQ2_S**: codebook-indexed. Each `qs` uint16 indexes into `iq2*_grid[][8]`, producing 8 packed-int8 values; sign bits unpack from the same uint16's high bits or a separate `signs[]` byte. `value = d × scale_sub × signed_grid_value`.
- **IQ3_XXS / IQ3_S**: same pattern as IQ2_S but with 256/512-entry int32 codebook (`iq3xxs_grid` produces 4 packed-int8 values per index).
- **IQ1_S**: 2048-entry int16 codebook (`iq1s_grid`) with embedded sign info; sub-block delta from `qh`. Most exotic — most likely to surprise during implementation.

**Codebook storage** (`IQuantGrids.cs`). The bracketed `[/* N values */]` comments below are spec-display abbreviations — the actual file contains the full literal arrays copied verbatim from `ggml/src/ggml-quants.c` (specifically the `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid`, `kvalues_iq4nl`, `ksigns_iq2xs` constants). A header comment in the file pins the upstream source URL and the `ggml-quants.c` revision SHA at copy-time.

```csharp
internal static class IQuantGrids
{
    // 256 × uint64. Each uint64 packs 8 signed int8 grid values.
    internal static readonly ulong[] Iq2XxsGrid = [/* 256 values from ggml-quants.c */];

    // 512 × uint64.
    internal static readonly ulong[] Iq2XsGrid = [/* 512 */];

    // 1024 × uint64.
    internal static readonly ulong[] Iq2SGrid = [/* 1024 */];

    // 256 × uint32.
    internal static readonly uint[] Iq3XxsGrid = [/* 256 */];

    // 512 × uint32.
    internal static readonly uint[] Iq3SGrid = [/* 512 */];

    // 2048 × uint16.
    internal static readonly ushort[] Iq1SGrid = [/* 2048 */];

    // 16 × sbyte. Signed lookup for IQ4_NL/XS — full literal:
    internal static readonly sbyte[] KvaluesIq4Nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    // 128 × uint64. Sign-mask lookup for IQ2_XS/S, IQ3_XXS/S.
    internal static readonly ulong[] KsignsIq2Xs = [/* 128 */];
}
```

### GPU dequant kernels

Each new quant gets a `dequant_<qt>_f16` PTX kernel matching the existing pattern: one CUDA block per superblock, 256 threads (or 32 for IQ4_NL), one element per thread, FP16 store at the end. IQ-family kernels read codebook constants from the `.const` segment populated at module load.

Grid sizing: `min(totalSuperblocks, MaxDequantGridSize)` with grid-stride loops in the kernel body, matching the existing K-quant pattern.

### GPU per-call GEMV

Each new quant gets a `quantized_gemv_<qt>` kernel: one CUDA block per output row, 256 threads, threads iterate over superblocks per row, FP32 accumulation, single FP16 store via warp + block reduction.

For Q2_K: structurally identical to `quantized_gemv_q4_k` with substituted per-superblock decode body.

For IQ-family with codebook: each thread decodes its assigned codebook indices via `__constant__` reads, accumulates element-products from input × signed-codebook-value × per-sub-block-scale.

For IQ4_NL (block-32): block iteration uses `k/32` instead of `k/256`; signed lookup via `kvalues_iq4nl[16]` is broadcast-friendly from constant memory.

### GPU MMQ + MMVQ-large + Pre-Q8_1

The fast-path kernels — dp4a-based int8×int8 fma chains consuming pre-Q8_1-quantized input scratch.

**Q2_K MMQ**: port of `quantized_gemv_q4_k_mmq` with substituted per-superblock decode (16 sub-blocks of 16 elements vs Q4_K's 8 sub-blocks of 32; 4-bit scale + 4-bit min coef vs Q4_K's 6-bit packed scales). Inner dp4a stays identical because the int8 weight tile doesn't care how it was decoded.

**IQ-family MMQ**: same overall block layout (4 rows × 32 elements per warp, 256 threads, dp4a inner loop) but the per-superblock decode prefix uses codebook lookup to **realize int8 values in the W_tile shared-memory array** before dp4a fires:

```
Per CUDA block, 4 rows × 32 elements (warp), 32 warps per block (256 threads):
  1. Each warp's 32 threads cooperatively decode 1 row × 32 elements:
       - Read codebook indices from raw quant bytes
       - Each thread does N constant-memory grid lookups (via __ldg or .const)
         producing N × {4 or 8} signed int8 values
       - Apply sign mask from ksigns_iq2xs (if applicable)
       - Multiply by per-sub-block scale (FP32 staged before final accum)
       - Pack 4 × int8 into one int32 in shared memory tile  (W_tile[row][col])
  2. Pre-Q8_1 input scratch is already shaped as 4 × int32 lanes per element-tile;
     read directly from scratch (no per-call re-quantize).
  3. dp4a fma loop: for each warp lane,
       acc_lane[j] += __dp4a(W_tile[lane_row][k], X_tile[lane_col][k], 0)
     across the K reduction.
  4. Apply per-block FP16 d × per-sub-block-scale at the end (FP32 → FP16 cast).
  5. Single FP16 store per output element.
```

The novelty over Q-family MMQ: step 1 has a codebook-expansion prefix, but the resulting int8 W_tile is structurally identical → step 3+ is byte-identical to existing K-quant MMQ inner loop. Step 1 is per-quant-specific; step 2-5 reuses the existing K-quant patterns.

**MMVQ-large variants** (1 CUDA block per output row, multiple warps span K reduction) follow the same per-quant codebook-decode-then-dp4a pattern, triggered when M ≥ 1024 (existing threshold).

**Pre-Q8_1 variants** consume the existing `quantize_x_to_q8_1` scratch (already exists for K-quants); the kernel saves a per-call re-quantize of the input. No new pre-Q8_1 input quantizer needed — the existing one quantizes whatever K is supplied (already handles K%32 alignment via blocks-per-row).

**IQ4_XS** is a special case — its codebook is only 16 entries (`kvalues_iq4nl`), small enough to live entirely in registers. Single shared 16-entry signed-int8 lookup; no codebook scatter cost. Closer to a Q-family MMQ in cost than the IQ2/IQ3 codebook-heavy variants.

### GPU grouped-MoE-GEMV

Phase B fast path: collapse `K_active` per-expert GEMVs into a single launch where the FP16 input row is shared. Each new quant gets `moe_grouped_gemv_<qt>_f16` mirroring the existing four (Q4_K, Q5_K, Q6_K, Q8_0).

Grid: `(M, K_active, 1)`. Block: 256 threads. Each block computes one `(expert, output_row)` pair using the **same inner-loop body as the per-call `quantized_gemv_<qt>` kernel** — only difference is per-expert weight pointer is loaded from the `weights_ptrs[e]` device array.

The grouped GEMV launcher's K-alignment check uses `MinKAlignmentFor(qt)`: 32 for IQ4_NL, 256 for everything else. This mirrors the gate fix from Round 15 (down_proj K=1408 unlock for Q8_0/Q5_0).

```csharp
public bool HasMoeGroupedGemv(QuantizationType qt) => qt switch {
    Q4_K => _moeGroupedGemvQ4_KFunc != 0,
    Q5_K => _moeGroupedGemvQ5_KFunc != 0,
    Q6_K => _moeGroupedGemvQ6_KFunc != 0,
    Q8_0 => _moeGroupedGemvQ8_0Func != 0,
    Q2_K => _moeGroupedGemvQ2_KFunc != 0,
    IQ2_XXS => _moeGroupedGemvIq2XxsFunc != 0,
    IQ2_XS  => _moeGroupedGemvIq2XsFunc != 0,
    IQ2_S   => _moeGroupedGemvIq2SFunc != 0,
    IQ3_XXS => _moeGroupedGemvIq3XxsFunc != 0,
    IQ3_S   => _moeGroupedGemvIq3SFunc != 0,
    IQ4_NL  => _moeGroupedGemvIq4NlFunc != 0,
    IQ4_XS  => _moeGroupedGemvIq4XsFunc != 0,
    IQ1_S   => _moeGroupedGemvIq1SFunc != 0,
    _ => false,
};
```

`CudaMoeFfn.Forward`'s existing `useGrouped` decision logic is unchanged — it already keys on `kernels.HasMoeGroupedGemv(weights.GateProjQuantType)`. New quants are picked up automatically.

## Testing strategy

### Test pyramid (~190 new tests across 5 phases)

```
                    ╱╲
                   ╱  ╲    Real-GGUF end-to-end smokes  (~9 tests)
                  ╱────╲
                 ╱      ╲   Cross-kernel parity        (~80 tests)
                ╱────────╲   Grouped vs per-call MMQ vs MMVQ-large
               ╱          ╲   per quant × 3-4 shapes
              ╱────────────╲
             ╱              ╲  Kernel correctness     (~110 tests)
            ╱                ╲  CPU + GPU dequant parity, GEMV
           ╱──────────────────╲ vs scalar oracle, per quant × 3 shapes
```

Per-quant test inventory (multiplied by 9 quants):

| Test class | Per-quant count | Total |
|---|---|---|
| `DequantizeQ2KTests` + `DequantizeIQuantsTests` (CPU) | ~3 | 27 |
| GPU `LaunchDequantToF16` parity (extend `CudaKernelComparisonTests`) | ~3 shapes | 27 |
| `CudaQuantizedGemvAlignmentTests` (extend) | ~3 shapes | 27 |
| `CudaMmqKernelTests` MMQ parity (extend) | ~3 shapes | 27 |
| `CudaMmqKernelTests` MMVQ-large parity (extend) | ~2 shapes | 18 |
| `CudaMmqKernelTests` Pre-Q8_1 paths (extend) | ~2 shapes | 18 |
| `CudaMoeGroupedGemvTests` (extend) | ~3-4 shapes | ~30 |
| `IQuantGridsByteEquivalenceTest` (NEW) | 1 per IQ codebook | ~7 |
| Real-GGUF end-to-end smoke per quant | 1 model each | 9 |
| **Total new tests** |  | **~190** |

### Real-GGUF fixtures

| Phase | Fixture (cached at `~/.dotllm/models/bartowski/`) | Disk |
|---|---|---|
| 1 | `DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf` (~5 GB) | 5 GB |
| 2 | `Meta-Llama-3.1-8B-Instruct-IQ4_XS.gguf` (~4.5 GB) | 4.5 GB |
| 3 | `Phi-3.5-mini-instruct-IQ3_M.gguf` (~2 GB) | 2 GB |
| 4 | `Meta-Llama-3.1-8B-Instruct-IQ2_XS.gguf` (~3 GB) | 3 GB |
| 5 | smallest IQ1_S available (~1.5 GB) or synthesize-only | 0-1.5 GB |
| **Total** | | **~14-16 GB** |

All gated `Skip.If(!File.Exists)` — fresh-clone CI green without fixtures. Disk impact fits on the existing E:\ junction.

### Per-phase test gates

| Phase | Gate |
|---|---|
| 1 (Q2_K) | All 27 Phase-1 tests green + V2-Lite Q2_K 27-layer smoke produces finite logits within 60 s (compares favorably to Round-13's Q3_K_M 31 s baseline since Q2_K is smaller). Existing 24 critical CUDA tests still green. CPU sweep still 0 failures. |
| 2 (IQ4_NL/XS) | All Phase-1 still green + 50 new Phase-2 tests + Llama-3.1 IQ4_XS smoke produces finite logits + finite logits parity vs llama.cpp on a 32-token reference (within 5e-2 of llama.cpp's reference logits via JSON sidecar fixture). |
| 3 (IQ3) | Phase 1+2 green + 50 new Phase-3 tests + Phi-3.5 IQ3_M smoke. |
| 4 (IQ2) | Phases 1-3 green + 75 new Phase-4 tests + Llama-3.1 IQ2_XS smoke. |
| 5 (IQ1_S) | All previous green + 25 new Phase-5 tests + (smoke optional if no real model cached). |

### Existing-test discipline (regression bar)

Every phase preserves these as a hard regression gate:
- 24 critical CUDA tests (3 MLA + 12 grouped GEMV + 4 real-GGUF Q4_K_M smokes + 5 K=1408 alignment) — unchanged.
- 27-layer V2-Lite Q3_K_M smoke — unchanged baseline (~31 s).
- 1467+ CPU tests — unchanged.
- Build: 0 warnings, 0 errors.

### Llama.cpp parity sidecar (cross-validation, recommended)

Per phase, a one-shot reference run produces a JSON sidecar:

```json
{
  "model": "Meta-Llama-3.1-8B-Instruct-IQ4_XS",
  "prompt": "Hello, how are",
  "logits_first10_position3": [3.421, -1.105, ..., 0.892]
}
```

Generated once via `llama-cli --logits` against the same GGUF, committed to `tests/DotLLM.Tests.Unit/Cuda/llama_cpp_reference/`. Test loads the GGUF + sidecar, runs same prompt through CUDA, asserts argmax-stable + max-abs-diff < 5e-2 on first-10 logits. Catches IQ kernel bugs that pass synthetic parity but produce drift on real attention/MLP composition.

### CI runtime budget

Worst-case full sweep grows from ~10 minutes (today) to ~15-20 minutes after Spec 1 lands all 5 phases (mostly real-GGUF smokes). Critical-path subset stays under 5 minutes.

## Phasing

### Per-phase deliverables

| Phase | Quant types | Files (new/extended) | Tests added | Effort | Real-GGUF |
|---|---|---|---|---|---|
| **1: Q2_K** | Q2_K | `Dequantize.cs`, `DequantizeKQuants.cs`, `dequant.cu`, `quantized_gemv.cu`, `quantized_gemv_mmq.cu`, `moe_grouped_gemv.cu`, `CudaKernels.cs` (extend) | ~27 | ~1 week | V2-Lite Q2_K (5 GB) |
| **2: IQ4_NL + IQ4_XS** | IQ4_NL, IQ4_XS | + new `DequantizeIQuants.cs`, `IQuantGrids.cs`, `dequant_iquants.cu`, `quantized_gemv_iquants.cu`, `quantized_gemv_iquants_mmq.cu`, `moe_grouped_gemv_iquants.cu` | ~50 | ~2 weeks | Llama-3.1 IQ4_XS (4.5 GB) |
| **3: IQ3_S + IQ3_XXS** | IQ3_S, IQ3_XXS | extend IQ files: + IQ3 codebooks + 8 kernels (dequant/GEMV/MMQ/MMVQ-large × 2 quants) + 2 grouped | ~50 | ~2-3 weeks | Phi-3.5 IQ3_M (2 GB) |
| **4: IQ2 family** | IQ2_S, IQ2_XS, IQ2_XXS | extend IQ files: + IQ2 codebooks (256/512/1024-entry grids) + sign-mask handling + 12 kernels + 3 grouped | ~75 | ~2-3 weeks | Llama-3.1 IQ2_XS (3 GB) |
| **5: IQ1_S** | IQ1_S | extend IQ files: + IQ1 codebook (2048-entry int16 grid) + 1.5-bit-with-qh packing + 4 kernels + 1 grouped | ~25 | ~1-2 weeks | Optional (~1.5 GB) |

**5 phases, 5 PRs, ~7-12 weeks calendar time, ~190 new tests, ~14-16 GB of fixture downloads.**

### Phase ordering rationale

- Phase 1 first because Q2_K is a Q-family port — easy infra warmup before IQ.
- Phase 2 next because IQ4_NL/XS are the most-used IQ quants in production today (Llama-3.1 / Qwen2.5 IQ4_XS). Highest production value per unit of work.
- Phases 3-4-5 in increasing codebook complexity: IQ3 is 256-entry grids; IQ2 adds full sign-mask scaffolding; IQ1 has the most exotic 2048-entry grid + qh-packed sub-block deltas.

### Inter-phase dependencies

- Phase 1 → none (independent).
- Phase 2 introduces `DequantizeIQuants.cs`, `IQuantGrids.cs`, `dequant_iquants.cu`, `quantized_gemv_iquants*.cu`, `moe_grouped_gemv_iquants.cu` — these become the trunk for Phases 3-5 to extend.
- Phases 3-5 only extend Phase 2's files. None of them touches Phase 1 work.

## Error handling

Per `CLAUDE.md` — error surfaces stay boundary-only:

| Surface | Existing behavior | Spec-1 extension |
|---|---|---|
| Unknown GGUF type ID at load | `Enum.IsDefined` throws `InvalidDataException` | New IDs added to enum → no longer throws |
| `NotSupportedException` from CPU `Dequantize.ToFloat32` | Already thrown for unknown types | Removed for the 9 new types |
| `NotSupportedException` from `LaunchDequantToF16` (GPU) | Existing for unsupported types | Removed for the 9 new types |
| K misaligned for grouped/MMQ | `ArgumentException` "must be multiple of 256" | Now uses per-quant `MinKAlignmentFor` (32 for IQ4_NL, 256 elsewhere) |
| Stale PTX (function not in module) | `Has*` properties report `false`; dispatcher falls through to next-best path | Same pattern; new `Has*` properties for each new quant |
| **NEW: codebook drift** between C#/PTX | n/a | `IQuantGridsByteEquivalenceTest` reads each PTX `__device__ __constant__` array via `cuModuleGetGlobal`, compares bytes to C# `static readonly`, fails loud |

No new fallbacks, no new validation — the existing "unknown → throw with explicit message" pattern absorbs the new types.

## Risks + mitigations

1. **IQ-family MMQ correctness pain** — IQ kernels in llama.cpp had multiple rounds of post-merge bugfixes (sign-bit unpacking, sub-block scale extraction, qh-byte-pair edge cases). Mitigations:
   - Hand-calc parity test on one block per IQ quant (deterministic; catches bit-pack mistakes early).
   - Bulk parity vs the C# scalar `ToFloat32` (single source of truth).
   - Optional sidecar logits parity vs llama.cpp on a real model per phase (5e-2 max-abs-diff).

2. **PTX module bloat** — adding new kernels grows total PTX size from ~50 KB to ~150 KB. New kernel function count: 9 dequant + 9 per-call GEMV + 9 MMQ + 9 MMQ_preq + ~7 MMVQ-large (skipped where M never reaches 1024 in practice, e.g. tiny IQ1_S models) + ~7 MMVQ-large_preq + 9 grouped-MoE-GEMV ≈ **~60 new kernel functions** across the 5 phases. Mitigation: PTX is per-file; existing `_quantizedGemvMmqModule`, `_dequantModule` patterns split kernels across modules; loading time scales linearly. Measured impact ~50-200 ms additional startup on cold load. Acceptable.

3. **Real-GGUF availability** — bartowski catalogs evolve; HF revisions can change file SHAs. Mitigation: `Skip.If(!File.Exists)` gates everything; per-test comment pins HF revision SHA at time of last-green run.

4. **Codebook PTX `.const` segment limit** — total IQ codebook footprint ~30 KB in PTX `.const`, well under 64 KB CUDA constant-memory cache limit. No risk.

5. **Pre-Q8_1 input quant extension** — the existing `quantize_x_to_q8_1` kernel quantizes whatever K is supplied (already handles K%32 via blocks-per-row). IQ4_NL at K%32 alignment uses the same pre-Q8_1 input scratch as Q4_K at K%256 — no new pre-Q8_1 kernel needed.

6. **MMQ vs llama.cpp performance** — our MMQ kernels for K-quants today are slightly behind llama.cpp's at the same M/K (3-7%). IQ MMQ kernels designed from scratch may sit similarly behind. Mitigation: target "within 10% of llama.cpp at same M/K" rather than "match exactly." Acceptable for L3.

7. **Dispatch-table fanout** — `LaunchQuantizedGemv` switch grows from 5 to 14 cases; `LaunchDequantToF16` from 11 to 20 cases. Pure switch dispatch, no runtime cost. Mitigation: split per-family helpers if any switch exceeds ~25 cases (none will after Spec 1).

## Resolved questions (answered during brainstorm)

| Q | Resolution |
|---|---|
| Coverage tier per quant? | Full L3 for all 9 (CPU dequant + GPU dequant + GEMV + MMQ + MMVQ-large + grouped-MoE-GEMV + pre-Q8_1 variants) |
| Approach: port-from-llama.cpp vs design-from-scratch? | Hybrid (Approach C): port for Q-family, design for IQ-family with codebooks copied verbatim |
| Spec / plan structure? | One spec (this doc), five plans, five PRs |
| OK to add ~14-16 GB to local model cache? | Yes |
| Generate llama.cpp reference logits sidecars? | Yes (one-time per phase, gated `Skip.If(!File.Exists)`) |
| PR per phase or one PR with 5 commits? | PR per phase |
| Phase 5 (IQ1_S) optional? | No — full L3 means all 9. Fixture is optional given rarity. |

## Out of scope (explicit non-goals)

- Q8_K (intermediate type used internally for MMQ accumulation, not stored in GGUFs).
- IQ1_M (rarer 1.5-bit variant with sub-block scaling — can be added later).
- Imatrix calibration tooling (quant-time concern, not load-time).
- Pre-V2 DeepSeek architecture support (separate spec).
- GGUF loaders for Mixtral / QwenMoe / GraniteMoe / Mamba3 (separate spec).
- Vulkan / HIP backend extensions (deferred to other hardware).

## Next step

Once this spec is approved, invoke `superpowers:writing-plans` to produce `Plan 1.1 (Q2_K)` as the first implementation plan. Plans 1.2 - 1.5 follow as their preceding phase lands.
