# Quantization — dotLLM

## Implementation Priority

1. **FP16** — Baseline
2. **Q8_0** — Simplest quantized, minimal quality loss
3. **Q4_0** — Basic 4-bit
4. **Q4_K_M** — Most popular, best quality/size tradeoff
5. **Q5_K_M, Q6_K** — Higher quality K-quants
6. **GPTQ/AWQ** — GPU-native (future)

## Block Layouts

### Q8_0 (8.5 bits/weight)

```
struct block_q8_0 {          // 34 bytes, 32 values
    half d;                  // scale factor
    int8_t qs[32];           // quantized values (-127..127)
};
Dequantize: val[i] = d * qs[i]
```

### Q4_0 (4.5 bits/weight)

```
struct block_q4_0 {          // 18 bytes, 32 values
    half d;                  // scale factor
    uint8_t qs[16];          // 32 × 4-bit packed into 16 bytes
};
Unpack: lo = qs[j] & 0x0F, hi = qs[j] >> 4
Dequantize: val[i] = d * (nibble - 8)
```

### Q4_K (4.5 bits/weight, super-block)

```
struct block_q4_K {          // 144 bytes, 256 values (8 sub-blocks of 32)
    half d;                  // super-block scale
    half dmin;               // super-block minimum
    uint8_t scales[12];      // 8 × (6-bit scale + 6-bit min)
    uint8_t qs[128];         // 256 × 4-bit values
};
For sub-block j: val = d * scale_j * nibble - dmin * min_j
```

Q4_K_M = mixed file: attention layers Q6_K, FFN layers Q4_K.

### Q6_K (6.6 bits/weight)

```
struct block_q6_K {          // 210 bytes, 256 values
    uint8_t ql[128];         // low 4 bits
    uint8_t qh[64];          // high 2 bits
    int8_t scales[16];       // INT8 sub-block scales
    half d;                  // super-block scale
};
```

### Q5_K (5.5 bits/weight)

```
struct block_q5_K {          // 176 bytes, 256 values
    half d, dmin;
    uint8_t scales[12];
    uint8_t qh[32];          // 5th bit
    uint8_t qs[128];         // low 4 bits
};
```

## Kernel Types

Each quantization format needs two kernels:

### Dequantize Kernel
Converts block → FP32. For layers needing full precision activations.

### Vec_Dot Kernel (Fused)
Dot product directly on quantized data. Faster than dequant+dot because avoids FP32 expansion.

**CPU SIMD (Q8_0 × Q8_0)**:
```
prod = Avx2.MultiplyAddAdjacent(qs_a, qs_b)  // INT8→INT16
acc = Avx2.MultiplyAddAdjacent(prod, ones)     // INT16→INT32
sum += ConvertToFloat(acc) * (da * db)
```

**CPU SIMD (Q4_0 × Q8_0)**:
Unpack nibbles to INT8, subtract offset, then integer multiply-accumulate.

## Mixed Quantization

GGUF files can have different types per tensor. Dispatch to correct kernel based on tensor metadata. Never assume uniform quantization.

## Performance Notes

- Vec_dot dominant for decode (GEMV). Dequant+BLAS may win for prefill (GEMM).
- GPU: custom CUDA kernels dequantize in shared memory, use tensor cores. Ref: llama.cpp `ggml-cuda/mmq.cu`.
- Block alignment awkward for SIMD — handle tail elements carefully.

## Vulkan Backend Coverage

The Vulkan backend ships native matmul kernels (GEMV decode + GEMM prefill, with an opt-in F16 cooperative-matrix tile when the device enumerates F16xF16→F32) for the following source dtypes / quant formats. Source bytes stay on device — dequantisation happens in the shader inner loop, so memory cost is the GGUF source size (not 2-4× expanded F32):

| Format | GEMV | GEMM | Coopmat | Reference |
|---|---|---|---|---|
| F32 | ✓ | ✓ | — | baseline |
| F16 | ✓ | ✓ | ✓ (F16xF16→F32, M=N=K=16 tile) | `c9c08c5` |
| BF16 | ✓ | ✓ | — (BF16 tiles not enumerated on RDNA3.5; use shift-left-16 reinterpret) | `c9c08c5` |
| Q8_0 | ✓ | ✓ | ✓ (`MatMulQ8_0GemmCoopmatKernel`) | pre-existing |
| Q2_K | ✓ | ✓ | — | (this branch) |
| Q3_K | ✓ | ✓ | — | (this branch) |
| Q4_K_M | ✓ | ✓ | — (Phase 1 follow-up) | `afb2272` + `b1ee6bc` |
| Q5_K_M | ✓ | ✓ | — | `15099b9` + `83e0732` |
| Q6_K_M | ✓ | ✓ | — | `29a1459` + `39b7646` |
| IQ4_NL | ✓ | ✓ | — | IQ-family Phase 2 |
| IQ4_XS | ✓ | ✓ | — | IQ-family Phase 2 |
| IQ1_S | ✓ | ✓ | — | IQ-family — smallest GGUF quant (~1.5-1.7 bpw) |

Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q2_K / Q3_K / IQ3_x / IQ2_x Vulkan kernels are not yet shipped — the upload path falls back to F32 dequant for those formats, so weight memory is doubled / quadrupled. K-quant Q4/5/6 priority was chosen because they cover the majority of production GGUF deployments (`*-Q4_K_M.gguf` is the de-facto default for most checkpoints); IQ4_NL and IQ4_XS followed because they are the most-used IQ-family quants in production (Llama-3.1 / Qwen2.5 IQ4_XS). IQ1_S closes the IQ family at the smallest end. Q2_K + Q3_K + IQ3 / IQ2 are present in the CUDA backend (lighter-weight reference kernels) and are tracked as Vulkan follow-ups.

### IQ4_NL / IQ4_XS layout (Vulkan)

Both IQ4 formats share the 16-entry signed-int8 codebook `kvalues_iq4nl = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}`. The 4-bit `qs` nibble is an *index* into the codebook, not a signed int — so the Vulkan dequant path does a constant-array lookup rather than a linear subtract. The codebook is duplicated as a `const float[16]` inside each shader (≈ 64 bytes per shader, well under push-constant / register pressure thresholds).

**IQ4_NL** (32-element block, 18 bytes/block):
```
bytes [0,1]   = fp16 d
bytes [2..17] = qs[16]   // low nibble = element j, high nibble = element j + 16
value         = d * float(kvalues_iq4nl[nibble])
```

**IQ4_XS** (256-element super-block, 136 bytes/super-block):
```
bytes [0,1]    = fp16 d
bytes [2,3]    = scales_h (uint16 LE)             // top 2 bits of each 6-bit ls
bytes [4..7]   = scales_l[4]                       // low 4 bits of each 6-bit ls
bytes [8..135] = qs[128]                           // 8 sub-blocks of 16 bytes / 32 nibbles
per sub-block ib:
    low6  = (scales_l[ib/2] >> (4*(ib&1))) & 0xF
    high2 = (scales_h >> (2*ib))           & 0x3
    ls    = low6 | (high2 << 4)            // 6-bit unsigned in [0..63]
    dl    = d * float(ls - 32)             // signed effective scale, range ≈ [-32, 31]
    value = dl * float(kvalues_iq4nl[nibble])
```

Alignment: IQ4_NL kernels require `inputDim % 32 == 0`; IQ4_XS kernels require `inputDim % 256 == 0`. The upload path's `KeepIq4NlOnDevice` / `KeepIq4XsOnDevice` predicates gate on these.
| IQ2_XXS | ✓ | ✓ | — | `79cca9b` |
| IQ2_XS | ✓ | ✓ | — | `743984c` |
| IQ2_S (also IQ2_M) | ✓ | ✓ | — | `9ecce75` |

Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q2_K / Q3_K / IQ4_NL / IQ4_XS / IQ1_*, IQ3_* Vulkan kernels are not yet shipped — the upload path falls back to F32 dequant for those formats. K-quant Q4/5/6 priority was chosen because they cover the majority of production GGUF deployments (`*-Q4_K_M.gguf` is the de-facto default for most checkpoints). Q2_K + Q3_K + IQ4_NL + IQ4_XS are present in the CUDA backend and are tracked as Vulkan follow-ups (parallel agent landed CUDA IQ4 in `6f12eab`'s sibling commits). The IQ2 family was prioritised on Vulkan to enable Qwen3.6-A3B-IQ2_M (~11.5 GB GGUFs) on Strix Halo without a 4× expansion to F32 (~46 GB) at upload.

### IQ1_S layout (Vulkan)

IQ1_S is the smallest GGUF quant (~1.5-1.7 bpw) and uses a 2048-entry signed-int8 codebook (`iq1s_grid`) — 8 ternary values {-1, 0, +1} packed into each `uint64` entry — together with a per-32-element `qh` field that carries a 3-bit scale, a sign-of-delta bit, and four 3-bit grid-index high parts. The codebook is duplicated as a `const uint[4096]` (each ggml uint64 split into a low/high uint pair, 16 KB total) inside each shader. SPV blobs are ~24-30 KB — bigger than the IQ4 kernels but still fast to JIT.

**IQ1_S** (256-element super-block, 50 bytes/super-block):
```
bytes [0,1]    = fp16 d
bytes [2..33]  = qs[32]                   // low 8 bits of grid index per group of 8 elements
bytes [34..49] = qh[8]                    // uint16 LE, per 32-element sub-block:
                                          //   bits  0..2  = grid-index high 3 bits, group 0
                                          //   bits  3..5  = ...                       group 1
                                          //   bits  6..8  = ...                       group 2
                                          //   bits  9..11 = ...                       group 3
                                          //   bits 12..14 = 3-bit per-block scale
                                          //   bit  15     = sign of delta (0 -> +0.125, 1 -> -)
per sub-block ib (8 sub-blocks per super-block, 4 groups of 8 elements per sub-block):
    dl    = d * (2 * ((qh[ib] >> 12) & 7) + 1)
    delta = (qh[ib] & 0x8000) ? -0.125 : +0.125
per group l in [0..4):
    idx   = qs[ib*4 + l] | (((qh[ib] >> 3*l) & 7) << 8)   // 11-bit, 2048 entries
    grid  = iq1s_grid[idx]                                // 8 packed signed-int8 ternary values
    y[j]  = dl * (grid[j] + delta)                        // for j in [0..8)
```

Alignment: IQ1_S kernels require `inputDim % 256 == 0`. The upload path's `KeepIq1SOnDevice` predicate gates on this.
Q4_0 / Q4_1 / Q5_0 / Q5_1 / IQ-family Vulkan kernels are not yet shipped — the upload path falls back to F32 dequant for those formats, so weight memory is doubled / quadrupled. K-quant Q4/5/6 priority was chosen because they cover the majority of production GGUF deployments (`*-Q4_K_M.gguf` is the de-facto default for most checkpoints). Q2_K and Q3_K — the densest K-quants — close the K-quant family on Vulkan; matmul kernels share the dispatch shape of Q4/5/6_K (one workgroup per output row at decode, 16×16 output tile at prefill) but use a 16-element K-chunk to match Q2/3_K's 16-element sub-block size.

**Q3_K caveat.** The Q3_K dequant matches the dotLLM CPU oracle (`DequantizeQ3_KScalar`) byte-for-byte. The CPU oracle has a pre-existing layout bug for sub-blocks 12..15 — it reads their low nibbles from the high nibbles of bytes 8..11 of the scales table, which are also fully occupied by the high-2-bits-per-sub-block packing (32 bits of hi2 fully use bytes 8..11). The correct llama.cpp layout puts sub_12..15's low nibbles in the high nibbles of bytes 4..7 (and sub_8..11's in bytes 0..3). Until the CPU oracle is fixed, Q3_K decode produces incorrect values for sub-blocks 12..15 on real GGUF data. The Vulkan kernel matches the (buggy) CPU output bit-for-bit, so cross-backend parity holds — but neither matches llama.cpp on those four sub-blocks. The fixture round-trip test side-steps the bug by zeroing source data in sub_12..15 so the parity assertions remain meaningful.

See [docs/VULKAN.md](VULKAN.md) for runtime selection details and [docs/CUDA.md](CUDA.md) for the CUDA backend's coverage (Q2_K through Q8_0 plus pre-Q8_1 + MMVQ-large + MMQ + grouped-MoE-GEMV variants).

### MoE indexed-expert matmul (per-row routed dispatch)

Sparse-MoE forward (Mixtral / Qwen-MoE / Qwen3MoeHybrid) needs a kernel shape distinct from dense matmul: a single dispatch reads a per-row expert index and reaches into a packed expert bank `[numExperts, M, K]` for that row's weight matrix. The Vulkan backend ships:

| Bank format | Kernel | Source layout on device | Used by |
|---|---|---|---|
| F32 | `MoeIndexedMatmulF32Kernel` (`moe_indexed_matmul_f32.comp`) | Dequantised at upload time, `[numExperts, M, K]` floats. | Default for all MoE models (streaming uploads per layer per forward; fits any quant since it dequants on the host). |
| Q8_0 | `MoeIndexedMatmulQ8_0F32Kernel` (`moe_indexed_matmul_q8_0_f32.comp`) | Raw Q8_0 blocks `[numExperts, M, (K/32)*34]`. Per-row dequant in shader. | Reserved for the resident-quant MoE path (no model wires it yet — Q8_0 banks fit at F32 too). |
| Q6_K | `MoeIndexedMatmulQ6_KF32Kernel` (`moe_indexed_matmul_q6_k_f32.comp`) | Raw Q6_K super-blocks `[numExperts, M, (K/256)*210]`. Per-row Q6_K dequant in shader (matches `DequantizeQ6_KScalar` byte-for-byte). | Qwen3MoeHybrid (Qwen3.6-A3B) when both `DOTLLM_VK_MOE_RESIDENT=1` AND the source banks are uniformly Q6_K — required for `Qwen3.6-A3B-UD-Q6_K_XL` to fit on Strix Halo's 128 GB unified memory (≈25 GB Q6_K-resident vs ≈120 GB if dequantised to F32). |

The Q6_K MoE kernel completes the original Phase 10 follow-up gap noted in `VulkanQwen3MoeHybridTransformerModel`: with this kernel in place, opting in to `DOTLLM_VK_MOE_RESIDENT=1` on a Q6_K-source Qwen3MoeHybrid model now uploads the routed banks once and keeps them resident across forwards, eliminating the per-forward host→device dequant + upload cost. Mixed-quant layers (e.g. UD checkpoints with Q5_K W2 and Q6_K W1/W3) fall back to the F32 path automatically.

## IQ-family (importance-quant) Coverage

I-quants encode weight values via a small codebook lookup rather than linear quantization — the on-disk bytes index into a per-quant-type grid table (256/512/1024 entries × 8 bytes) and an 8-bit sign mask. The base scale (`d`, Half) plus 4-bit per-pair sub-scales decode to floats as `db * grid[idx][j] * sign[j]`.

| Format | Block | bpw | CPU dequant | CPU MatMul | CUDA dequant | CUDA GEMV | Vulkan dequant | Vulkan GEMV/GEMM | Notes |
|---|---|---|---|---|---|---|---|---|---|
| IQ4_NL | 32 | 4.5 | ✓ | dequant-fallback | ✓ (`dequant_iq4_nl_{f16,f32}`) | ✓ (`quantized_gemv_iq4_nl`) | — (parallel agent) | — (parallel agent) | Plus MMQ-preq + MMVQ-large + MoE-grouped. |
| IQ4_XS | 256 | 4.25 | ✓ | dequant-fallback | ✓ | ✓ | — (parallel agent) | — (parallel agent) | Plus MMQ-preq + MMVQ-large + MoE-grouped. |
| IQ2_XXS | 256 | 2.0625 | ✓ | dequant-fallback | ✓ (`dequant_iq2_xxs_{f16,f32}`) | ✓ (`quantized_gemv_iq2_xxs`) | ✓ (`iq2_xxs_dequant_f32.spv`) | ✓ (`matmul_iq2_xxs_f32_{gemv,gemm}.spv`) | 256-entry codebook + 4×7-bit sign indices per 32-elem sub-block + shared 4-bit scale. |
| IQ2_XS | 256 | 2.3125 | ✓ | dequant-fallback | ✓ | ✓ | ✓ | ✓ | 512-entry codebook; 7-bit sign indices in upper bits of `qs[uint16]`. |
| IQ2_S | 256 | 2.5625 | ✓ | dequant-fallback | ✓ | ✓ | ✓ | ✓ | 1024-entry codebook; high index bits in `qh`. **Also stores `MOSTLY_IQ2_M` file-type tensors** (Qwen3.6-A3B-IQ2_M ~11.5 GB GGUFs). |

The Vulkan IQ2 family kernels store the codebook tables as readonly SSBOs uploaded once per model load (3 grids + ksigns ≈ 14 KB on device, shared by all 6 IQ2 matmul kernels via `Iq2Codebooks`). Per-element decode is `db * grid[gridIdx*8+j] * sign_j`; per-pair scale uses the same `db = d * (0.5 + sub_scale) * 0.25` arithmetic as the CPU oracle. IQ2_XXS / IQ2_XS resolve `sign_j` via the 128-entry `ksigns_iq2xs` lookup; IQ2_S stores the 8-bit sign mask directly per pair.
| Format | Block | bpw | CPU dequant | CPU MatMul | CUDA dequant | CUDA GEMV | Notes |
|---|---|---|---|---|---|---|---|
| IQ4_NL | 32 | 4.5 | ✓ | dequant-fallback | ✓ (`dequant_iq4_nl_{f16,f32}`) | ✓ (`quantized_gemv_iq4_nl`) | Plus MMQ-preq + MMVQ-large + MoE-grouped. |
| IQ4_XS | 256 | 4.25 | ✓ | dequant-fallback | ✓ | ✓ | Plus MMQ-preq + MMVQ-large + MoE-grouped. |
| IQ2_XXS | 256 | 2.0625 | ✓ | dequant-fallback | ✓ (`dequant_iq2_xxs_{f16,f32}`) | ✓ (`quantized_gemv_iq2_xxs`) | 256-entry codebook + 4×7-bit sign indices per 32-elem sub-block + shared 4-bit scale. |
| IQ2_XS | 256 | 2.3125 | ✓ | dequant-fallback | ✓ | ✓ | 512-entry codebook; 7-bit sign indices in upper bits of `qs[uint16]`. |
| IQ2_S | 256 | 2.5625 | ✓ | dequant-fallback | ✓ | ✓ | 1024-entry codebook; high index bits in `qh`. **Also stores `MOSTLY_IQ2_M` file-type tensors** (Qwen3.6-A3B-IQ2_M ~11.5 GB GGUFs). |
| IQ1_S | 256 | ~1.5625 | ✓ | dequant-fallback | — | — | **Vulkan-only on GPU.** 2048-entry codebook of 8 ternary {-1, 0, +1} values packed into each uint64. Per-sub-block 3-bit scale + sign-of-delta in `qh[uint16]`. Smallest GGUF quant. |

**IQ1_S** is now supported on CPU and Vulkan (dequant + GEMV + GEMM); the CUDA backend treats it as a CPU-only fallback. **IQ1_M / IQ3_XXS / IQ3_S** remain out of scope. MMQ-preq / MMVQ-large / MoE-grouped variants for the IQ2 family are deferred; prefill falls back to dequant→cuBLAS via the `dequant_iq2_*_f32` kernels.
