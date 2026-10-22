# Numeric Precision per Operation and Device

> **Status: DRAFT (#1109).** This is the survey workstream. Sections marked **TODO(cuda)** and **TODO(cpu)** are
> filled from the experiment workstreams (`.docs/precision/cuda-results.md` and `.docs/precision/cpu-results.md`).
> **TODO(vulkan)** sections wait on #1100–#1102. Until then, every "recommended" cell is a **hypothesis backed by
> literature and existing repo measurements**, not yet a measured decision for dotLLM.

## 1. Rules (summary)

The hard rules live in [`CLAUDE.md` → Precision & Expansion Rules](../CLAUDE.md#precision--expansion-rules-user-decisions-2026-10-22-1107)
(#1107). In short:

1. **No FP32 tensor storage on hot paths.** FP16/BF16 is the widest default for anything stored or streamed. Always
   evaluate formats below FP16 too (q8_0, q4_0, FP8/MXFP4 where the hardware has them).
2. **Accumulate in F32 only where the structure needs it.** That means:
   - norms (sum of squares);
   - softmax and attention sums;
   - long-KV attention;
   - recurrent-state updates;
   - split-reduction combines.

   Compute-bound GEMMs may accumulate in F16, with promotion to F32 every 64–256 terms, if they pass KL/PPL parity.
   Integer dots accumulate in int32.
3. **Quantized weights and KV run quantized.** Dequantize in registers or shared memory inside the tile load. Never
   expand into a wider stored copy.

This document is the evidence base those rules point to. Every place where we **keep FP32** has an explicit
"**Keep FP32 because …**" entry with sources. Anything without one is a candidate for removal.

### How to read the tables

- **Storage** is the dtype written to memory: KV, activations, scratch, state, tables.
- **Compute** is the operand dtype fed to the ALU or tensor core.
- **Acc** is the accumulator.
- Notation `W4A8` means 4-bit weights × 8-bit activations. `Q8_1` means int8 with a per-32 scale *and* a per-block
  sum, as in llama.cpp.
- `[R]` marks a figure recalled from a paper and not re-checked. `[D]` marks something we derived. `UNVERIFIED` marks
  a claim with no primary source found.

### Accuracy bar (from #1109)

- KL vs an F32-accumulate reference on fixed prompts and windows, plus top-1 agreement and PPL delta.
- Gate on **median or trimmed KL plus outlier rate**, not the mean: massive-activation rows dominate the mean.
- No degenerate test shapes. Use mutants where useful.

---

## 2. Device capability matrix

Sources are vendor whitepapers and ISA guides, uops.info, and local `vulkaninfo`/CPUID probes run on 2026-10-22 on
the dev box (155H + RTX 3060 + Arc), `strix` and `t5500`. Raw notes and links are in `.docs/precision/SURVEY.md` §2.

### 2.1 Summary

Ratios are per-clock throughput relative to the same device's FP32 FMA.

| format | RTX 3060 (GA106, CC 8.6) | Radeon 8060S (gfx1151) | Arc iGPU (Xe-LPG, MTL) | 155H CPU (AVX2 + AVX-VNNI) | Ryzen AI Max+ 395 (Zen 5) | Xeon X5670 (Westmere) |
|---|---|---|---|---|---|---|
| FP32 | 1× (12.7 TF) | 1× (14.8 TF; 2× with VOPD dual issue) | 1× (4.6 TF) | 1× (16 FMA/clk per P-core) | 1× (32 FMA/clk, full 512-bit) | 1× (4 mul + 4 add/clk, no FMA) |
| FP16 | mma **4×** (F16 acc) / **2×** (F32 acc); HFMA2 1–2× (sources conflict) | WMMA ≈4×; `v_pk_fma_f16` 2× | packed 2× (9.2 TF) | storage only (F16C convert) | storage only (no AVX512-FP16) | **no F16C**: software convert |
| BF16 | mma 2× (F32 acc only) | WMMA ≈4× in ISA, but **no bf16 coopmat** in either Vulkan driver | none (UNVERIFIED) | shift-convert only | **`VDPBF16PS` 2×**, F32 acc | shift-convert (cheap) |
| FP8 E4M3/E5M2 | **none** (Ada CC 8.9+) | **none** (gfx1170 / RDNA4+) | none | none | none | none |
| INT8 | mma **8×** (101.9 TOPS); dp4a (rate UNVERIFIED) | WMMA iu8 = f16 rate (**no speed gain**, bytes only) | **DP4a 4×** (18.4 TOPS) | **VPDPBUSD ymm 4×** (u8×s8) | **VPDPBUSD zmm 4×** | SSSE3 `pmaddubsw` ≈1× |
| INT4 | mma 16× (203.8 TOPS) | WMMA iu4 2× of iu8 | none (unpack to int8) | none | none | none |
| Bandwidth | 360 GB/s, 12 GB | ≈256 GB/s LPDDR5X-8000, shared | ≈90 GB/s DDR5-5600, shared with CPU | ≈90 GB/s | ≈256 GB/s shared (CCD reach UNVERIFIED) | ≈32 GB/s per socket, NUMA |

**What follows.** Decode is bandwidth-bound everywhere, so **bytes per weight and per KV element decide decode speed**.
Prefill is where the compute rates matter:

- **RTX 3060:** INT8 mma, or FP16 mma with F16 accumulate. F32-accumulate FP16 mma runs at **half rate on GeForce**,
  which is a GeForce-specific trap.
- **Radeon 8060S:** f16 WMMA. int8 WMMA only saves bytes.
- **Arc iGPU, 155H, Zen 5:** integer dot (DP4a or VNNI) is **4× FP32**.
- **Westmere:** int8 via SSSE3, with BF16 rather than F16 for any 16-bit storage.

### 2.2 RTX 3060

The rates are derived from the per-tensor-core figures in the
[GA102 whitepaper](https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf),
Table 6 and Appendix A: 28 SMs × 4 TCs × 1.777 GHz.

| format | instruction | dense peak | accumulator |
|---|---|---|---|
| FP16 | `mma.sync.m16n8k16.f16.f16.f16.f16` | 50.9 TFLOPS | F16 |
| FP16 | `mma.sync.m16n8k16.f32.f16.f16.f32` | **25.5 TFLOPS (half rate, GeForce)** | F32 |
| BF16 | `mma.sync.m16n8k16.f32.bf16.bf16.f32` | 25.5 TFLOPS | F32 only |
| TF32 | `mma.sync.m16n8k8.tf32` | 12.7 TFLOPS | F32 |
| INT8 | `mma.sync.m16n8k32.s32.s8.s8.s32` | 101.9 TOPS | int32 |
| INT4 | `mma.sync.m16n8k64.s32.s4.s4.s32` | 203.8 TOPS | int32 |
| FP32 | FFMA | 12.7 TFLOPS | — |
| INT8 dot | `__dp4a` (IDP4A) | UNVERIFIED; plausibly ≈25 TOPS | int32 |

- **FP8:** none.
- **Vulkan (driver 621.01):** exposes `VK_KHR_cooperative_matrix`, `VK_NV_cooperative_matrix2` and bf16 coopmat.
- **HFMA2 rate:** the [CUDA throughput table](https://docs.nvidia.com/cuda/archive/12.4.0/cuda-c-programming-guide/index.html#arithmetic-instructions)
  says 256 FP16 results/clk/SM (2× FP32); the whitepaper implies 1:1. **Microbenchmark before relying on either.**

### 2.3 Radeon 8060S (RDNA 3.5, gfx1151, 40 CUs)

| format | instruction | per CU per clock | peak at 2.9 GHz | accumulator |
|---|---|---|---|---|
| FP16 | `V_WMMA_F32_16X16X16_F16` / `V_WMMA_F16_16X16X16_F16` | 512 FLOP | 59.4 TF | F32 / F16 |
| BF16 | `V_WMMA_F32_16X16X16_BF16` / `V_WMMA_BF16_16X16X16_BF16` | 512 | 59.4 TF | F32 / BF16 |
| INT8 | `V_WMMA_I32_16X16X16_IU8` | 512 | 59.4 TOPS | int32 |
| INT4 | `V_WMMA_I32_16X16X16_IU4` | 1024 | 118.8 TOPS | int32 |
| packed FP16 | `V_PK_FMA_F16` | 256 | 29.7 TF | F16 |
| FP32 | `V_FMA_F32` (VOPD dual issue) | 128 (256) | 14.8 (29.7) TF | — |

- Source: [GPUOpen WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/).
- **F16-accumulate WMMA vs F32-accumulate WMMA:** GPUOpen gives one rate for both. Whether F16 accumulate is faster
  is **UNVERIFIED → TODO(vulkan)**. It still halves accumulator registers, which can raise occupancy.
- **Vulkan coopmat:**
  - The repo probe (`docs/ATTENTION.md`) lists 11 shapes, all 16×16×16, all f16 or int8. There is no F32 operand type.
  - The AMD Windows driver reports `shaderBFloat16CooperativeMatrix = false`.
  - [RADV](https://gitlab.freedesktop.org/mesa/mesa/-/raw/main/src/amd/vulkan/radv_physical_device.c) disables bf16
    coopmat on GFX11/11.5 ("precision issues").
  - FP8 coopmat exists only on gfx1170 and GFX12.
- **Integer dot:** `integerDotProduct4x8BitPacked*Accelerated = true`. `shaderFloat16 = true`.
- **Wave size:** WMMA executes wave32 although the default subgroup is 64.

### 2.4 Intel Arc iGPU (Meteor Lake Xe-LPG, 8 Xe-cores / 128 XVE)

- **Rates:** FP32 4.6 TF, packed FP16 9.2 TF, DP4a 18.4 TOPS. Sources:
  [Chips and Cheese](https://chipsandcheese.com/p/intels-ambitious-meteor-lake-igpu),
  [HotHardware](https://hothardware.com/reviews/intel-meteor-lake-architecture?page=3).
  Chips and Cheese also notes that the compiler often fails to emit FP16 FMA.
- **Missing:** no XMX/DPAS (Arrow Lake-H adds DPAS) and **no `VK_KHR_cooperative_matrix`** (driver 101.9033).
- **Features:** `shaderFloat16`, `shaderInt8` and every 8-bit integer-dot-product "accelerated" flag are true.
- **Memory:** the dev box has DDR5-5600 SO-DIMMs, ≈89.6 GB/s, shared with the CPU. Earlier notes assumed LPDDR5x
  ≈120 GB/s; that was wrong.
- **Existing repo evidence (#229):**
  - I2_S GEMM with an F16 weight tile: 1.22–1.50×, bit-exact (now production).
  - An int8 weight tile: 0.91–1.09×, so no gain over F16.

### 2.5 Core Ultra 7 155H CPU (Redwood Cove P + Crestmont E)

- **Present:** AVX2, FMA, F16C, AVX-VNNI (VEX).
- **Absent:** AVX-512, AVX-VNNI-INT8 (signed×signed `VPDPBSSD`, which arrives in Arrow/Lunar Lake), AVX-NE-CONVERT,
  BF16, AMX, AVX-IFMA.

Throughput from [uops.info](https://uops.info/html-instr/VPDPBUSD_VEX_YMM_YMM_YMM.html):

| op | P-core lat / recip-tput | per clock | E-core |
|---|---|---|---|
| `VFMADD231PS ymm` | 4 / 0.5 | 16 FP32 FMA | 8 |
| `VPDPBUSD ymm` | 5 / 0.5 | **64 u8×s8 MAC (4×)** | 32 |
| `VPMADDUBSW` + `VPMADDWD` | 0.5 each (shared p01) | ≈32 MAC; s16 saturation hazard | ≈16 |
| `VCVTPH2PS ymm` | 8 / 1.0 | 8 conversions | 4 |

- `VPDPBUSD` is **u8×s8**. A signed×signed product needs the +128 bias trick and a block-sum correction, which is
  what `Q8_1`'s `s` term and dotLLM's `ComputeRowsVnni` do.

### 2.6 Ryzen AI Max+ 395 CPU (Zen 5, 16 cores)

- **Full 512-bit FPU**, the same as desktop Zen 5
  ([Chips and Cheese](https://chipsandcheese.com/p/amds-chiplet-apu-an-overview-of-strix)). Strix Point is the
  double-pumped part.
- **Present:** AVX-512 F/DQ/CD/BW/VL/IFMA/VBMI/VBMI2/VNNI/BITALG/VPOPCNTDQ/VP2INTERSECT/**BF16**, plus AVX-VNNI (VEX).
- **Absent:** AVX512-FP16, AVX-VNNI-INT8, AMX.

| op | lat / recip-tput | per clock |
|---|---|---|
| `VFMADD231PS zmm` | 3 / 0.5 | 32 FMA |
| `VPDPBUSD zmm` | 3 / 0.5 | **128 MAC (4×)** |
| `VDPBF16PS zmm` | 5 / 0.5 | 64 bf16 MAC (2×), F32 acc |
| `VCVTPH2PS zmm` | 6 / 0.8 | ≈20 conversions |

Sources: [numberworld Zen 5 teardown](https://numberworld.org/blogs/2024_8_7_zen5_avx512_teardown/) and uops.info.

- **Repo lesson** (memory note, commit `9c9ae31`): `vpmovsxbd zmm` (byte→dword widen) is ≈8× slower than the ymm
  form. Every 512-bit kernel that widens bytes lost to AVX2:
  - `Q8_0ToF32Avx512` ran at 0.12×.
  - The Q8_0 R4 VNNI-512 kernel was 1.7–1.9× slower (`MatMul.cs:420`).

  Only dot-dense loops with no widen (Q4_0 nibble unpack 1.17×, fused RMSNorm→Q8 1.20×, BF16 dot) win. **Gate AVX-512
  per kernel by measurement, not by `IsSupported`.**
- **.NET access:** .NET 10 exposes 512-bit VNNI only via `Avx10v1.V512`, which reports *unsupported* on Zen 5
  (`MatMulVnni.cs:33`). The `AvxVnni.V512` and `Avx512Bf16` APIs are on the `dev-dotnet11` preview track.

### 2.7 Xeon X5670 (T5500, 2 sockets) — the floor

- **ISA:** SSE4.2 only. No AVX, FMA, F16C or VNNI.
- **int8:** `PMADDUBSW`/`PMADDWD` (SSSE3) both issue on p1, ≈8 MAC/clk.
- **F16:** decode is **software** (a LUT or ≈10–15 integer ops per 4 values; cost UNVERIFIED).
- **BF16:** decode is one shift or unpack, so **BF16 is the cheaper 16-bit format here.**
- **Bandwidth:** ≈32 GB/s per socket.

---

## 2A. Engine practice at a glance

All sources are default branches as of 2026-10-22. Links are in §3 and in `.docs/precision/SURVEY.md` §5.

| op class | llama.cpp | vLLM | TensorRT-LLM | SGLang | MLX |
|---|---|---|---|---|---|
| **weight GEMM** | CPU: W?A8 via `vec_dot_type` (Q8_0/Q8_1/Q8_K). CUDA: MMQ int8 mma on Turing+ at *any* batch; the cuBLAS fallback dequantizes to F16 and uses `COMPUTE_16F`. | Marlin W4A16, F32 acc (F16 only on sm75). FP8 W8A8 on SM ≥ 8.9; Ampere runs FP8 weight-only. | cuBLASLt FP16/BF16/FP8/FP4 at `COMPUTE_32F`; SmoothQuant INT8 W8A8; W4A8 on Ada+ | FP8 128×128 block-scale GEMM, F32 acc | affine group-64 4-bit; float activations; F32 acc |
| **activations** | **F32** graph tensors | model dtype (BF16/F16) | model dtype | model dtype | model dtype |
| **KV** | K and V types set independently (f16, bf16, q8_0, q4_0, …); a quantized V needs FA | fp8 e4m3/e5m2, int8/int4 per-token-head, nvfp4; scale 1.0 unless calibrated | INT8/FP8 per-tensor scale (FP8 KV allowed on Ampere as storage) | fp8 e4m3/e5m2, bf16, fp4 | 8-bit group-64 (`kv_bits`) |
| **attention** | CUDA FA: KQ F32, **PV half2**. Vulkan and CPU FA: F32. The graph *requests* F32 everywhere. | F32 S and acc | FP16 FMHA: **BMM1 F16 acc by default**; F32 for BF16/FP8 or opt-in | F32 | F32 (`precise` softmax) |
| **norms** | Σx² in **double** (CPU) | F32 | — | — | F32 |
| **RoPE** | computed per op in F32, no table | F32 compute, **table cast to model dtype** | — | — | in-kernel |
| **SSM/GDN state** | **F32, hard-coded** | NemotronH F32 ("only float32 is known to have no accuracy issues"); Qwen3.5 F32 via config; generic `auto` = model dtype | `mamba_ssm_cache_dtype` (auto) | **F32 default** | **F32** |
| **draft/MTP** | — | inherits the quant config; `draft_lm_head_quantization` fp8/nvfp4 ("only the acceptance length can change") | — | — | — |
| **logits** | F32 buffer | head in model dtype; F32 head only for RL consistency; sampler F32 | — | — | — |
| **split partials** | F32 (FA `VKQ_parts`, MMQ stream-K fixup) | Marlin F32 reduce | — | — | — |

**Where engines disagree** (this is what decides our choices):

1. **F16 accumulation is accepted** by llama.cpp (cuBLAS F16 path, Vulkan f16acc default), ExLlamaV2, TRT-LLM FP16
   FMHA BMM1 and Marlin-on-Turing. It is **rejected** by TRT-LLM GEMMs, vLLM/Marlin on Ampere+, SGLang and MLX.
   llama.cpp's answer to failures was a **per-model F32 opt-in**, not a global switch:
   - FFN-down and `wo` for GLM4, GLM4_MOE and JAIS2 ([#13101](https://github.com/ggml-org/llama.cpp/pull/13101));
   - the SQRT_SOFTPLUS MoE router;
   - non-FA KQ, always.

   This is exactly our "F16 where measured, F32 where structural" rule, plus a **per-model override hook**.
2. **llama.cpp keeps activations in F32**; every other engine uses 16-bit. dotLLM follows the majority.
3. **llama.cpp expands quantized data in two places**: a Vulkan FA F16 K/V scratch, and the CUDA cuBLAS path's weight
   dequant to F16. dotLLM's rule 2 deliberately diverges here.

---

## 3. Per operation class

Each subsection gives:

- the **recommendation per device**;
- the **rationale and evidence**;
- the **current repo state** (see §4 for byte costs);
- a **"Keep FP32 because"** entry where one applies.

### 3.1 Weight GEMV (decode) and GEMM (prefill)

**Literature and engines**

- **Weight-only 4-bit (W4A16)** is the safe low-bit default: [AWQ](https://arxiv.org/abs/2306.00978),
  [GPTQ](https://arxiv.org/abs/2210.17323), Marlin.
- **W8A8 needs outlier handling.** Emergent outlier features above ≈6.7B break per-tensor INT8
  ([LLM.int8()](https://arxiv.org/abs/2208.07339)). Fixes:
  - [SmoothQuant](https://arxiv.org/abs/2211.10438) migrates outliers into the weights;
  - fine-grained activation scales: llama.cpp's per-32 `Q8_0`/`Q8_1`, and
    [DeepSeek-V3](https://arxiv.org/abs/2412.19437)'s 1×128 tiles.
- **llama.cpp runs W4A8/W8A8 with block-scaled int8 activations on every backend.** Quality is accepted
  community-wide.
  - **CPU:** `vec_dot_type` is `Q8_0`/`Q8_1`/`Q8_K`
    ([ggml-cpu.c](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ggml-cpu.c)). `Q8_K` carries an
    F32 `d` plus int16 `bsums`.
  - **CUDA MMQ:** `quantize_q8_1`, then dp4a or int8 `mma`. On **Turing and later, which includes the RTX 3060, int8-mma
    MMQ is used at every batch size** for supported types
    (`ggml_cuda_should_use_mmq`, [mmq.cu](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cu)).
    It accumulates int32 → F32 `sum[]`.
  - **Vulkan:** integer-dot MMQ, `matmul_*_q8_1`, always with an F32 accumulator
    ([ggml-vulkan.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp)). Its FP
    coopmat matmuls **default to f16acc** on fp16 devices.
- **F32 GEMM accumulation elsewhere:**
  - **TensorRT-LLM**'s cuBLASLt FP16/BF16/FP8/FP4 configs all use `CUBLAS_COMPUTE_32F`
    ([cublasMMWrapper.cpp](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/common/cublasMMWrapper.cpp)).
  - **vLLM Marlin** accumulates in F32 except on sm75
    ([marlin_template.h](https://github.com/vllm-project/vllm/blob/main/csrc/quantization/marlin/marlin_template.h)).
  - **vLLM on Ampere** runs FP8 checkpoints as weight-only W8A16 (FP8 Marlin), with no FP8 compute.

  Block scales of 32 or 256 sidestep the per-tensor outlier problem
  ([massive activations](https://arxiv.org/abs/2402.17762): values up to 2622 on Llama-2-7B and 7100 on Mixtral, on
  fixed dims and the BOS token).
- **[QServe](https://arxiv.org/abs/2405.04532): W4A8 beats W4A4 on Ampere-class GPUs.** Per-group W4A4 needs
  int→float partial-sum dequant on CUDA cores inside the main loop, ≈50 tensor-core MACs each on A100. Atom-W4A4 was
  20–25% *slower* than W4A16/W8A8. QServe dequantizes 4→8-bit (clamped to [-119, 119]) so every GEMM runs on int8
  tensor cores.
- **W4A4** is accurate only with rotations, and still loses about 1–3 zero-shot points:
  - [QuaRot](https://arxiv.org/abs/2404.00456): Hadamard; ≤0.47 PPL on Llama-2-70B; 8-bit RTN becomes lossless.
  - [SpinQuant](https://arxiv.org/abs/2405.16406): learned rotations.

  Not a near-term target.
- **Accumulator width matters at large K:**
  - DeepSeek-V3: H800 FP8 tensor-core accumulation keeps ≈14 bits, so ≈2% max relative error at K=4096. Their fix
    is to promote to F32 every 128 terms.
  - [SageAttention2](https://arxiv.org/abs/2411.10958): the "f32" FP8 mma accumulator on Ada and Hopper is really
    **FP22**.
  - [Higham & Mary](https://eprints.maths.manchester.ac.uk/2673/1/paper.pdf): for **zero-mean** products (weights ×
    activations) the probabilistic error is ≈√K·u, or O(u) independent of K. That is why F16 accumulation *can* work
    for GEMMs but not for same-sign sums.

**Recommendation**

| device | decode GEMV | prefill GEMM | accumulator |
|---|---|---|---|
| RTX 3060 | quantized weights, **Q8_1 activations, dp4a** (done for K-quants/IQ4/PQ2_0/I2_S) | **INT8 mma MMQ (W4A8/W8A8 with Q8_1 B)**, to replace dequant→F16→cuBLAS (#495). Interim: cuBLAS `COMPUTE_16F` (G1). | int32 (int8); F16 with F32 promotion every ≤128 for HMMA, if KL passes **TODO(cuda)** |
| Radeon 8060S | quantized weights, **Q8_1 activations, dp4a** (done) | f16 coopmat with in-tile dequant (done, F32 acc). Test **F16-acc coopmat** and **iu8 WMMA** (bytes only). | F32 today; F16-acc **TODO(vulkan)** |
| Arc iGPU | quantized weights, **Q8_1 + DP4a** (done; F32-input fallback when dp4a is missing) | **DP4a MMQ** (int8 is 4× FP32; no coopmat). The F16 weight tile is the measured sweet spot for I2_S. | int32 |
| 155H CPU | quantized weights, **Q8_0/Q8_1/Q8_K activations, VNNI** (`VPDPBUSD ymm`) | VNNI W8A8/W4A8 tiles. Today Q8_0 prefill tiles use maddubs, not VNNI, and Q4_0/Q4_1/Q5_1/IQ4_NL prefill is dequant→F32 dot. | int32 → F32 per block |
| Zen 5 CPU | same; 256-bit VNNI. 512-bit only per-kernel where measured faster (no widen). | VNNI-512 tiles where no byte widen is needed; **`VDPBF16PS` for BF16/F16 weights** (2× FP32, F32 acc) | int32; F32 for BF16 dot |
| Westmere | Q8 activations + SSSE3 `pmaddubsw` (exists: SSSE3 tiers) | same | int32 (watch s16 saturation) |

**Repo state**

- **Decode is already W?A8 almost everywhere.** Exceptions:
  - CUDA Q8_0/Q5_0/IQ2_* decode use F16/F32 activations with FP FMA.
  - The hybrid Q8_0 decode is F32-in.
- **Prefill expands on CUDA:** dequant→F16 scratch→cuBLAS. Byte costs are in §4.2.
- **Vulkan IQ1/2/3:** prefill dequantizes into an F16 scratch before coopmat.
- **CPU:**
  - Q4_0 prefill dequantizes rows to F32.
  - The safetensors BF16 loaders widen whole tensors to F32.
  - The R4 repack holds a second same-width copy.

**Experiments**

- **TODO(cuda):** F16 vs F32 accumulation on Bonsai-2 prefill GEMMs (K up to 17,408): KL median, KL p99 and outlier
  rate. Also INT8-mma W8A8/W4A8 feasibility and speed.
- **TODO(cpu):** VNNI W8A8/W4A8 vs F32/BF16 on the 155H and Zen 5, for speed and KL.

> **Keep FP32 because …** — *none for weights.* F32 weight storage is never justified for inference. The only F32 here
> is the **in-register accumulator** for (a) per-block scale application after an int32 dot, and (b) long-K FP
> GEMMs until F16-acc-with-promotion passes KL. Neither is storage.

### 3.2 Activations between kernels

**Evidence**

- **Massive activations reach 2622 (Llama-2-7B) and 7100 (Mixtral)**
  ([Sun et al.](https://arxiv.org/abs/2402.17762)). That is under FP16's 65,504, with only 9–25× headroom.
- **Gemma-3 overflows in FP16** (NaN): [transformers#36822](https://github.com/huggingface/transformers/issues/36822).
  The clamp fix was rejected as "significantly incorrect".
- **BitNet's residual exceeds 65,504 in dotLLM**, which is why `fused_add_rmsnorm_f32res.cu` carries an F32 residual.
- FP16 has 3 more mantissa bits than BF16, so FP16 is the better *precision* default and BF16 the *range* fallback.

**Recommendation**

| device | storage | notes |
|---|---|---|
| RTX 3060 | **F16** (the main CUDA path already is) | the hybrid path is all-F32 today; move to F16 |
| Radeon 8060S / Arc | **F16** | Vulkan is all-F32 today. The measured ffn-down F16 activation gives 13.5→23.5 TFLOPS at k=9216 (Infinity Cache fit), with NLL Δ 1e-5. |
| 155H / Zen 5 | **F16 or BF16** for stored prefill activations; at decode, fuse norm→Q8 so the activation never lands in F32 | CPU has no F16 arithmetic, so F16 costs a convert (F16C ≈1/clk). Decode activations are L1/L2-resident, so the gain is mostly at prefill. **TODO(cpu)** |
| Westmere | **BF16** | no F16C |

- Fall back to **BF16 per model** (a config or `ModelConfig` flag) where FP16 overflows: Gemma-3 and TPU/BF16-trained
  families.
- The **residual stream** is the first place FP16 overflows. Track its max-abs per model in the precision harness.

> **Keep FP32 because …** the **residual stream of models whose measured max |x| exceeds the FP16 range** (BitNet
> today). Use BF16 first. Use F32 only if BF16's 8-bit mantissa measurably moves KL, which the residual
> accumulation (`x += f(x)` over L layers) could do. **Needs measurement and user sign-off per model.**

### 3.3 KV cache (K and V separately)

**Evidence**

- **K has fixed outlier channels; V does not.** Quantize K per-channel (or smooth it) and V per-token:
  - [KIVI](https://arxiv.org/abs/2402.02750): 2-bit.
  - [KVQuant](https://arxiv.org/abs/2401.18079): K per-channel and **pre-RoPE**; under 0.1 PPL loss at 3-bit.
  - [QServe SmoothAttention](https://arxiv.org/abs/2405.04532): K outlier channels are ≈10× typical, and the scale
    is folded into Q.
- **K usually needs more bits than V.** [KVTuner](https://arxiv.org/abs/2502.04420) finds K 8→4 bits raises the
  attention error 13.9×, so K4V2 ≫ K2V4.
  - **Model-dependent:** [TurboAngle](https://arxiv.org/abs/2603.27467) finds TinyLlama (hd 64) V-limited.
- **llama.cpp** offers `--cache-type-k/-v` ∈ {f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1}. A quantized V needs
  flash attention. In [PR #7412](https://github.com/ggml-org/llama.cpp/pull/7412), q8_0 KV ≈ f16, and K q8_0 + V q4_0
  beat q6_K weights.
- **vLLM, TensorRT-LLM and SGLang** ship FP8 E4M3 KV with per-tensor k_scale/v_scale. In vLLM the default scale is
  1.0 unless calibrated
  ([quantized_kvcache.md](https://github.com/vllm-project/vllm/blob/main/docs/features/quantization/quantized_kvcache.md)),
  and `--kv-cache-dtype-skip-layers sliding_window` exists because some layer types are more sensitive.
  - **None of our devices has native FP8**, so FP8 KV would be a storage-only format with a software decode. It has no
    advantage over q8_0 there, and q8_0 keeps a scale per 32 elements.
- **llama.cpp's Vulkan FA dequantizes quantized K and V into an F16 scratch** for N ≥ 64 on non-coopmat2 devices. Intel
  Xe1 is excluded because it regressed. This is the pattern our rule 2 forbids; we read q8_0 in-tile instead.
- **Repo, Bonsai-2 on the RTX 3060** (`docs/KV_CACHE.md`):

  | format | PPL | Δ |
  |---|---|---|
  | F16 | 10.2350 | — |
  | Q8_0 | 10.2346 | −0.004% |
  | K8V4 | 10.2529 | +0.18% |
  | Q4_0 | 10.2661 | +0.32% |

  Needle retrieval was 3/3 at 72k. Decode at 72k was 18.1 / 17.1 / 18.8 tok/s.
- **Sink tokens** ([StreamingLLM](https://arxiv.org/abs/2309.17453)) and **recent tokens** (KIVI's residual window)
  are the high-leverage entries. A small higher-precision window is an established pattern, and dotLLM's
  `QuantizedKvCache` already has one.

**Recommendation**

| device | K | V | notes |
|---|---|---|---|
| all GPUs | **q8_0** (default candidate) | **q8_0**; q4_0 when capacity-bound | Read directly in the attention tile, with no scratch (the CUDA hybrid `kvq` path, #1094). F16 is the floor, never F32. |
| CPUs | **q8_0** | q8_0 / q4_0 | Score QKᵀ with **int8 dots** (Q quantized to Q8, VNNI or SSSE3) instead of dequantizing tiles to F32. |
| Westmere | q8_0 | q8_0 | F16 KV costs a software convert on every read, so q8_0 is cheaper *and* smaller. |

- Make **K ≥ V in bits** the default asymmetry (K8V4). Validate per model.
- The F32 window in `QuantizedKvCache` (CPU) and F32 quantized-KV scales (Vulkan) should become F16.

**Experiments**

- **TODO(cuda):** KV F16 / q8_0 / q4_0 KL (median, p99, outlier rate) on Bonsai-2 and one Llama-class model.
- **TODO(cpu):** CPU KV F16 / INT8 (#978): speed and KL on the 155H and Zen 5.
- **TODO(vulkan):** hybrid KV F16 (#1102) and quantized KV on the hybrid path.

> **Keep FP32 because …** — *none.* No source or measurement supports F32 KV. It is 2× F16 and ≈3.8× q8_0
> (Llama-3-8B: 256 KiB/token F32 vs 68 KiB q8_0).

### 3.4 Attention: scores, softmax, P·V

**Evidence**

- **FlashAttention keeps the running max *m*, sum *ℓ* and the O accumulator in F32**
  ([FA2](https://arxiv.org/abs/2307.08691) [R]).
- **[SageAttention](https://arxiv.org/abs/2410.02367)** shows:
  - INT8 QKᵀ after K mean-smoothing is more accurate than FP8 and 2× faster;
  - **P·V with an F16 accumulator** is 2× faster on GeForce, with "no accuracy loss";
  - INT8 P·V fails.

  [SageAttention2](https://arxiv.org/abs/2411.10958) adds INT4 QK and FP8 PV with two-level accumulation (F32 every
  64).
- **Engines disagree on the PV and QK accumulators:**
  - **llama.cpp CUDA FA ignores the graph's F32 request.** The mma kernel's KQ accumulator is F32 but its **VKQ is
    `half2`** ([fattn-mma-f16.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-mma-f16.cuh)).
  - **llama.cpp Vulkan and CPU FA** accumulate in F32.
  - **TensorRT-LLM FP16 context FMHA** accumulates BMM1 (QKᵀ) in **F16 by default**, with F32 forced for BF16 and FP8
    ([fmhaRunner.cpp](https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaRunner.cpp)).
  - **FlashInfer** `use_fp16_qk_reduction` is off by default: "faster at the cost of slight precision loss".
  - **vLLM and MLX** are F32 throughout.
- **llama.cpp CUDA FA uses F16 VKQ accumulators** and has hit overflow. Granite-4.0-h-1B
  ([#17746](https://github.com/ggml-org/llama.cpp/pull/17746)) was fixed by offsetting the KQ max by ln 2. Metal
  chose F32 accumulators instead. GLM4 forces F32 attention precision (#9130).
- **[FA3](https://arxiv.org/abs/2407.08608) FP8** needs block quantization plus Hadamard incoherent processing:
  2.6× lower error than naive FP8.
- [Higham–Mary](https://nhigham.com/2021/09/14/can-we-solve-linear-algebra-problems-at-extreme-scale-and-low-precisions/)
  [D]: ℓ is a **same-sign** sum of up to T terms, so it drifts like T·u. In FP16 it stagnates once ℓ is about 2048×
  the typical increment.
- **Repo:**
  - All CUDA `mma.sync` attention is `f32.f16.f16.f32`: F16 operands, F32 accumulator, F32 *m*/*ℓ*.
  - Vulkan coopmat flash prefill stages Q/K/V/P as F16 in LDS with F32 accumulation. The QKᵀ F16 input is the whole
    residual error (maxAbs 1.45e-4).
  - Vulkan decode attention is scalar F32, because coopmat decode was 1.27–1.56× slower on gfx1151.

**Recommendation**

| item | format | device notes |
|---|---|---|
| Q, K, V operands | F16 (or int8 for QKᵀ, Sage-style, with K smoothing) | 3060: int8 mma QKᵀ is a candidate (8× FP32). CPU: int8 VNNI QKᵀ against q8_0 K. |
| S = QKᵀ accumulator | F32 (≤ head_dim terms; F16 acc is possible but buys little) | — |
| softmax *m*, *ℓ* | **F32** | all |
| P (stored or staged) | F16 | all |
| P·V accumulator | F32 by default. **F16 acc is a candidate on the 3060** (2× mma rate), with the ln 2 max-offset headroom trick and a per-model F32 escape hatch. | **TODO(cuda)** |

> **Keep FP32 because …**
>
> - **softmax running max and sum (*m*, *ℓ*):** same-sign sums over T keys swamp in FP16, and an *m* error shifts
>   every exponent (FA2, Higham–Mary). They are registers, not storage.
> - **P·V accumulator by default:** FP16 accumulators have overflowed in production (llama.cpp #17746, #9130). Move
>   to F16 only per device after KL passes with the max-offset headroom.

### 3.5 Norms (RMSNorm / LayerNorm)

**Evidence**

- [D] In FP16, x² overflows once |x| > 255.9. Massive activations (2622) **guarantee** overflow, and even |x| ≈ 4 over
  d = 4096 reaches 65,536.
- [DeepSeek-V3](https://arxiv.org/abs/2412.19437) keeps all norm operators at BF16/FP32.
- **Engines:**
  - llama.cpp CPU RMSNorm accumulates in **double**
    ([ops.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cpu/ops.cpp)).
  - vLLM uses `float variance`
    ([layernorm_kernels.cu](https://github.com/vllm-project/vllm/blob/main/csrc/layernorm_kernels.cu)).
  - MLX uses `float acc`.
  - The I/O stays in the storage dtype.
- **Repo:** F32 sum of squares in every backend.

**Recommendation**

- Read inputs at the activation dtype (F16) and write F16 or fused Q8 output.
- Accumulate Σx² in F32.
- Norm weight vectors are tiny (H floats). F16 is fine, but the gain is negligible. Low priority.

> **Keep FP32 because …** **the sum-of-squares accumulator** overflows FP16 structurally (above), and a pre-scaled F16
> alternative costs an extra pass. It is a register, not storage.

### 3.6 RoPE / YaRN tables

**Evidence**

- **HF transformers forces F32** in `LlamaRotaryEmbedding` (autocast disabled, "bfloat16 loses precision on long
  contexts", PR #29285).
- **[AnchorAttention](https://arxiv.org/abs/2411.13476):** BF16 RoPE breaks relative-position invariance. The first
  token dominates the error, and it grows with length.
- [D] FP16 represents integers exactly only to 2048 and BF16 only to 256, so the **angle pos·θ must be computed in F32**
  (an F32 ulp at pos = 131,072 is ≈0.0078 rad). Use F64 for the YaRN/NTK correction math, as `RoPE.cs` does.
- [D] Once the angle is computed in F32, **cos/sin values stored in F16** carry an absolute error of ≤2.4e-4,
  comparable to the F16 rounding of Q/K. BF16 tables (≈2e-3) are what the AnchorAttention errors look like.
- **Engines split on table storage:**
  - **vLLM** computes `cos_sin_cache` in F32 and then **casts it to the model dtype** (BF16/F16)
    ([rotary_embedding/base.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding/base.py)).
    So 16-bit tables built from F32 angles are production practice.
  - **llama.cpp** computes per op in F32 with no table (`ggml_rope_cache_init`).
- **Repo:**
  - CUDA and Vulkan compute sin/cos in-kernel from F32 angles, with no table (an `inv_freq` F32 vector only).
  - CPU stores managed `float[]` cos/sin tables of `maxSeq × ropeDim × 4` B (64 MiB at 128k for Llama-8B),
    duplicated in `RoPEPositionEncoding`.

**Recommendation**

- GPUs: keep in-kernel F32 angles (no table).
- CPU, either:
  - (a) compute per position with SIMD sincos from an F32/F64 angle; or
  - (b) build **F16 tables from F32/F64 angles**.

  Measure the long-context (≥32k) KL for (b). **TODO(cpu)**

> **Keep FP32 because …** **the angle computation** (pos·θ, and the YaRN ramp in F64): 16-bit positions or angles are
> wrong almost immediately (HF #29285, AnchorAttention). This is compute, not a stored table. **F32 cos/sin *tables*
> are not justified** unless (b) fails the KL gate.

### 3.7 Recurrent state (GDN / Mamba / SSM)

**Evidence**

- **[Mamba README](https://github.com/state-spaces/mamba):** "SSMs are sensitive to their recurrent dynamics"; keep
  parameters in fp32 under AMP. This is training guidance.
- **Engines are close to unanimous on F32 state:**
  - **llama.cpp** hard-codes `recurrent_type_s = GGML_TYPE_F32`
    ([llama-memory-recurrent.cpp](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory-recurrent.cpp)),
    with no override.
  - **SGLang** defaults the SSM state to F32 and the conv state to BF16
    ([configs/mamba_utils.py](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/configs/mamba_utils.py)).
  - **MLX-LM** initialises the GDN state as `float32`, and its fast path requires it
    ([gated_delta.py](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/gated_delta.py)).
  - **vLLM**'s generic `auto` follows the model dtype, but its model overrides force F32 for NemotronH ("*Only `float32`
    is known to have no accuracy issues by default*"). Qwen3.5 follows the HF config
    `mamba_ssm_dtype: "float32"`
    ([models/config.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/config.py),
    [Qwen3.5-35B-A3B config](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/main/config.json)).
- **fla Gated DeltaNet kernels**
  ([`chunk_delta_h.py`](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/common/chunk_delta_h.py)):
  - the state accumulator is `tl.float32`;
  - **per-chunk state snapshots are stored in k's dtype (BF16/FP16)**;
  - the final state is F32.
- **Quantization papers:**
  - [Quamba](https://arxiv.org/abs/2410.13229) and [Quamba2](https://arxiv.org/abs/2503.22879) quantize the scan
    *inputs* (x, B, C, Δ) to 8-bit with calibration. Naive W8A8 loses 7–18 points.
  - [MambaQuant](https://arxiv.org/abs/2501.13484): the scan *amplifies* outliers.
  - None of them stores the carried state below 16 bits.
- [D] Unlike KV, the state is **rewritten every token** (h ← a·h + b·x). Rounding error compounds over T steps rather
  than staying per-entry, so it is the one tensor where storage precision feeds back into itself.
- **Repo:** all GDN, Mamba-2/3, Nemotron-H SSM and LFM2 conv state is F32 on every backend. The scans are bit-exact F32
  ports of the CPU loop.
- **Size:** the state is **constant per sequence, not per token**. For example, 32 V-heads × 128² × 4 B = 2 MiB per
  GDN layer (illustrative). That is negligible against KV at long context.

**Recommendation**

- Keep the **update arithmetic in F32** on every device.
- **State storage:** F32 stays for now (sign-off below). Evaluate a **BF16/F16 carried state with F32 compute** over
  long decodes (≥8k tokens), with KL at the end of the window. Narrow only if it passes.
- **Snapshots** (MTP rollback, prefix-cache clones) **may be F16/BF16**, as fla does. Measure the restore KL.
- **Conv state** (the last d_conv−1 inputs) is just activations: F16.

> **Keep FP32 because …** **the carried SSM/GDN state** is a feedback loop where rounding compounds over T steps (Mamba
> README; fla keeps state compute and final state in F32; HF `mamba_ssm_dtype: float32`). Its storage is O(1) in
> sequence length, so the byte saving from narrowing is small. **This needs the user's sign-off as a standing exception
> pending a long-decode KL measurement.** Model-declared `float32` is evidence, not an automatic exemption (CLAUDE.md).

### 3.8 MTP / draft heads

**Evidence**

- Speculative decoding preserves the target distribution exactly
  ([Leviathan](https://arxiv.org/abs/2211.17192), [Chen](https://arxiv.org/abs/2302.01318)). Draft precision only
  moves the **acceptance rate**.
- [QSpec](https://arxiv.org/abs/2410.11305) drafts with W4A4-class precision and verifies with a high-precision
  verifier, for 1.64×.
- A cheaper but worse draft can lose net speed: 3 of 5 Apple-silicon configs slowed down
  ([2602.23881](https://arxiv.org/abs/2602.23881)).
- **vLLM** lets the drafter use its own `kv_cache_dtype`, and quantizes the drafter's shared lm_head copy with
  `draft_lm_head_quantization` ∈ {fp8, nvfp4}, because "*the target verifies with its own head, so only the acceptance
  length can change*" ([speculative.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/speculative.py)).

**Recommendation**

- MTP head KV at **F16 or q8_0**. Today it is F32 on CUDA, CPU and Vulkan (#1095), at 8 KiB/pos on Bonsai-2, doubled
  during growth.
- Pending hidden and absorb scratch at F16.
- Tune the format by **acceptance × cost**, not KL.

> **Keep FP32 because …** — *none.* The verify step guarantees correctness.

### 3.9 LM head / logits

**Evidence**

- DeepSeek-V3 keeps the output head at BF16/FP32 in *training*.
- llama.cpp ships `output.weight` at Q6_K/Q8_0 in most quant mixes [R], and keeps logits as F32 in `llama_get_logits`.
- **vLLM** runs the head in the model dtype. The F32 head option is documented as "*required for RL
  training-inference consistency*", not for quality. Even that path uses `torch.mm(out_dtype=float32)` to avoid an F32
  weight copy. The sampler converts logits to F32
  ([logits_processor.py](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/logits_processor.py)).
- [D] Softmax and log-sum-exp over V ≈ 150–250k are same-sign sums, so they need F32.
- [D] F16 logit *storage* has ≈5e-4 relative error, ≈0.02 absolute at |logit| = 40. That can flip near-tied greedy
  argmax and perturbs PPL.
- [Gemma-2](https://arxiv.org/abs/2408.00118) soft-caps logits at 30, which bounds the range.

**Recommendation**

- **Weights:** quantized, read in place. **Never expand** (#1099: 2.54 GB F16 on Bonsai-2 for >8 rows).
- **Logits:**
  - Materialize F32 **only for the rows that are sampled or scored** (1 row normally, k+1 for verify).
  - Never materialize `capacity × V × 4` scratch. The CPU `Logits` buffer holds all positions, plus a second
    `seqLen × V × 4` copy; the CUDA hybrid buffer is `rows × V × 4`.
  - For verify, do **argmax/top-k on device** and return indices.

> **Keep FP32 because …** **the sampled-row logits buffer and the softmax/LSE reduction**: V-wide same-sign sums need F32,
> and F16 storage can flip near-tied greedy argmax. The cost is 4·V bytes per *sampled* row (≈1 MB at 248k), which is
> negligible. Not for all-positions buffers.

### 3.10 Embeddings

**Evidence**

- A lookup has no accumulation, so quantized storage with per-row dequant is numerically benign.
- DeepSeek keeps embeddings at BF16/FP32 for *training* only.
- Tied embeddings inherit the LM-head sensitivity; llama.cpp keeps `token_embd` at higher-bit quants [R/A].
- Gemma multiplies by √d. That inflates the residual, so it is the one place to watch FP16 range.

**Recommendation:** keep the table in its **GGUF quant type**. Dequantize **one row per token** into the F16
activation. **No widening.**

**Violations today**

- Vulkan widens every non-Q8_0 table to F32: 5.09 GB for Qwen3.8-27B vs ≈339 MB packed.
- CPU safetensors BF16 → F32: 2.1 GB per Llama-8B table.
- CUDA expands to F16 when no lookup kernel exists for the quant type.

> **Keep FP32 because …** — *none.*

### 3.11 Accumulators (cross-cutting)

| context | accumulator | why |
|---|---|---|
| int8 dot (dp4a, VNNI, int8 mma, iu8 WMMA) | **int32** → F32 per block for scale | exact by construction |
| FP GEMM, K ≤ 256 per chunk | **F16** allowed in compute-bound kernels if KL passes | zero-mean products: error ≈ O(u) to √K·u (Higham–Mary) |
| FP GEMM, long K | F16 within chunks of 64–256, **promoted to F32** between chunks | DeepSeek-V3 (128), SageAttention2 (64) |
| norms Σx², softmax *ℓ*, LSE, recurrent update, split combine | **F32** | same-sign sums: n·u drift and swamping ([Sakr et al.](https://arxiv.org/abs/1901.06588)); FP16 overflow |
| BF16 dot (`VDPBF16PS`, BF16 mma) | F32 (fixed by the ISA) | — |

### 3.12 Split-K / split-KV partials

**Evidence**

- Flash-decoding combines per-split (*m*, *ℓ*, O) with log-sum-exp rescaling. Engines store the partials in F32:
  - llama.cpp CUDA FA `VKQ_parts` is float and `VKQ_meta` is `float2`
    ([fattn-common.cuh](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/fattn-common.cuh)), and
    the MMQ stream-K fixup is float.
  - vLLM Marlin uses `use_fp32_reduce = True`.
  - The **outlier is ExLlamaV2**, whose split partials are F16 `atomicAdd`.
- [D] An LSE error δ becomes a relative weight error of ≈δ. An F16 LSE at magnitude 20 has an ulp of ≈0.016, giving
  **≈1.6% weight error**.
- [D] The O partial is bounded and normalized, so F16 is tolerable.
- [D] Split-K GEMM partials are zero-mean dot sums, so F16 is tolerable for small split counts, with the combine in
  F32.
- **Repo:**
  - CUDA split-KV partials are F32 (`numHeads·maxSplit·(hd + 2)·4`, ≈0.8 MB).
  - Vulkan split-KV partials are F32, and the Vulkan split-K GEMM slabs are 5.24 MB F32 per instance.
  - The CPU has no split reductions.

**Recommendation**

- *m* and *ℓ* (or LSE) partials stay F32. They are a few bytes per head and split.
- O partials and split-K GEMM slabs go to F16, with the combine accumulating in F32. **TODO(cuda), TODO(vulkan)**

> **Keep FP32 because …** **the per-split *m*/*ℓ* (LSE) values**: an FP16 LSE gives ≈1.6% weight error. The size is
> `numHeads·splits·2·4` B, a few KB. These are stored F32 partials, so they need sign-off under rule 1. **Requesting it
> as a standing exception.**

---

## 4. Repo inventory: FP32 stores and quantized expansions (as of `dev` @ `040cbbb4`)

Example models:

- **L1B** = Llama-3.2-1B (H 2048, I 8192, 16 layers, 8 KV heads × 64)
- **L8B** = Llama-3.1-8B (H 4096, I 14336, 32 layers, 8 × 128, V 128,256)
- **B2** = Bonsai-2 / qwen35 hybrid (H 5120, V 248,320, 16 attention slots, 4 × 256)

Full file:line references are in the agent sweeps summarised in `.docs/precision/SURVEY.md` §3.

### 4.1 CPU

**Rule 1 — FP32 storage**

| # | item | cost | ref |
|---|---|---|---|
| C1 | `SimpleKvCache` (the default KV; also the draft-model KV) | `L·nkv·hd·2·4` per token: **L8B 256 KiB/tok (32 GiB @128k)**; L1B 64 KiB/tok | `SimpleKvCache.cs:45-86` (#978) |
| C2 | All activations and inter-kernel tensors F32 | `4·(3H+Q+A+2KV+3I)` per token of capacity: L8B 256 KiB/tok (512 MiB @ cap 2048) | `TransformerForwardState.cs:224-241` |
| C3 | Logits for all positions, plus a second output copy | `cap·V·4` + `seqLen·V·4`: L8B 501 KiB/tok each (≈2 GiB @ 2048) | `TransformerForwardState.cs:236`, `TransformerModel.cs:3070-3100` |
| C4 | `PagedKvCache` / `KvBlockPool` F32 blocks, plus a full-history F32 staging gather per layer per step | pool `2·L·blocks·bs·stride·4`; staging `2·maxSeq·stride·4` (L8B @8k: 64 MiB) | `PagedKvCache.cs:86,161-173` |
| C5 | `QuantizedKvCache` F32 recent window | `L·W·stride·8` (L8B, W=128: 32 MiB) | `QuantizedKvCache.cs:158-171` |
| C6 | RoPE cos/sin managed `float[]`, duplicated | `maxSeq·ropeDim·4` (L8B @128k: 64 MiB) | `TransformerForwardState.cs:150-170`, `RoPEPositionEncoding.cs:19-35` |
| C7 | MTP KV and hidden | `2·cap·stride·4` (4096 pos: 32 MiB) | `CpuMtpState.cs:83-87` |
| C8 | MLA latent/expanded KV, TurboQuant norms, MoE scatter buffers | see sweeps | `MlaLatentKvState.cs`, `TurboQuantKvCache.cs`, `MoeSwiGluMlp.cs:506` |
| C9 | Recurrent state and snapshots (pending sign-off, §3.7) | `Lrec·(conv + state)·4` | `GdnStateCache.cs`, `SsmStateCache.cs`, `Mamba3State.cs` |

**Rule 2 — expansions**

| # | item | cost | ref |
|---|---|---|---|
| X1 | Safetensors BF16 linear, embed and lm_head widened to F32 | ×2 of BF16: **L8B ≈ +16 GB**; L1B +2.5 GB | `SafetensorsTensorResolver.cs:80-86` |
| X2 | GGUF DeepSeek/MLA-MoE routed experts dequantized to F32 | `E·3·I·H·4` (≈57 GB on V2-Lite, per the code comment) | `TransformerWeights.cs:2639-2648` |
| X3 | GGUF MLA projections, MoE shared experts and Granite-MoE experts to F32 | `Σ out·in·4` | `TransformerWeights.cs:2348-2440, 2684` |
| X4 | **R4 repack: a second full same-width copy** (not a widening, but a stored duplicate) | +1× quantized weights: L8B Q4_K_M ≈ +4.9 GB | `WeightRepacking.cs:121-141` |
| X5 | TurboQuant full-history F32 dequant scratch | `2·maxSeq·stride·4` (L8B @128k: 1 GiB) | `TurboQuantKvCache.cs:121-124` |
| X6 | Dequant-row prefill (Q4_0/Q4_1/Q5_1/IQ4_NL n>1, BF16, F16, MXFP4, PQ2_0 prefill) | transient `k·4` per worker (bounded); **a missing W?A8 prefill kernel** | `MatMulDequantRows.cs`, `MatMulLegacyQuants.cs:591` |
| X7 | Quantized-KV attention tile dequant to F32 | bounded, `tile·hd·8` per head per worker, **allocated per call** | `Attention.cs:1026-1042` |

### 4.2 CUDA

**Rule 1 — FP32 storage**

| # | item | cost | ref |
|---|---|---|---|
| G1 | **F32 activation set allocated for every model**. Only the F32-forward architectures (Gemma, Granite, OLMo2, SmolLM3, MLA) and pipeline download use it. | `4·cap·(3H + 2·nh·hd + 2·nkv·hd + 3I)`: **L1B ≈ 587 MB @ cap 4096**; Qwen3-8B ≈ 487 MB @ 2048 | `CudaForwardState.cs:180-189` |
| G2 | `DequantScratchF32`, always allocated | `4·maxProj`: L1B 67 MB, Qwen3-8B 201 MB | `CudaForwardState.cs:128` |
| G3 | Whole hybrid forward state F32 (activations, GDN buffers, logits `rows·V·4`) | logits alone 970 MiB @ 1024 rows for V = 248k | `CudaQwen3HybridDenseForwardState.cs:199-245` |
| G4 | Hybrid F32 K/V staging (tiers without flash/kvq) | `2·seqKv·kvElems·4`: **B2 8 KiB/tok (512 MiB @64k)** | `CudaQwen3HybridDenseTransformerModel.cs:4140-4162` |
| G5 | MTP head KV F32, doubled during growth | B2 8 KiB/pos (16 KiB/tok in `ContextBytesPerToken`) | `CudaMtpState.cs` (#1095) |
| G6 | G3 attention F32 score matrix | `4·nh·s²` (+2 for F16 P): 32 heads @ 4096 ≈ 3.2 GB | `CudaG3Attention.cs:140-180` |
| G7 | Nemotron-H KV, MLA latent KV, default MLA KV; TurboQuant norms | see sweep | `CudaNemotronHKvCache.cs:43,58`, `CudaMlaLatentKvCache.cs` |
| G8 | Gemma F32 KV round-trip scratch and activation scratch | `2·kvLen·kvStride·4` | `CudaTransformerModel.cs:3576-3728` |
| G9 | Split-KV partials (O, *m*, *ℓ*) | ≈0.8 MB | `attention_mma_decode_gqa_split.cu` |
| G10 | Recurrent state (pending sign-off, §3.7) | — | `CudaGdnStateCache.cs`, `CudaMamba3StateCache.cs` |

**Rule 2 — expansions**

| # | item | cost | ref |
|---|---|---|---|
| GX1 | **Prefill dequant → F16 scratch → cuBLAS per projection** (Q4_K batched MMQ off, PQ2_0 MMQ opt-in) | `2·m·k` scratch plus a 2 B write and read per weight per forward: L1B 33.5 MB, Qwen3-8B 101 MB, 70B 470 MB | `CudaTransformerModel.cs:3887-3906`, hybrid `:4474` (#495) |
| GX2 | LM head with more than 8 rows → F16 dequant | `2·V·H` = **2.54 GB on B2** | hybrid `:2104-2124` (#1099) |
| GX3 | `CudaQuantizedKvCache` / `CudaTurboQuantKvCache` full-history F16 scratch, **always allocated** | `4·maxSeq·stride` (stride 1024 @32k: 128 MB) | `CudaQuantizedKvCache.cs:164-168` — **not fixed by #1094** |
| GX4 | Hybrid KV F16 dequant scratch (remaining tiers) | B2 4 KiB/tok (≈0.5 GiB @128k) | hybrid `:4086-4117` |
| GX5 | Load-time F16 copies for quant types without a GEMV kernel, and the embedding table | `2·n·k`; embedding Qwen3-8B 1.16 GiB | `CudaWeights.cs:315-345` (gated by `EnsureQuantExpansionAllowed`) |
| GX6 | Gemma `ProjectF32`: quant → F16 → F32 per projection | `4·maxProj` | `CudaTransformerModel.cs:3818-3850` |

### 4.3 Vulkan (HIP is a scaffold with nothing to inventory)

**Rule 1 — FP32 storage**

| # | item | cost | ref |
|---|---|---|---|
| V1 | Dense KV, F32 default | `2·L·nkv·hd·4` per token: L1B 64 KiB/tok, Llama-3.2-3B 224 KiB/tok | `VulkanKvCache.cs:113-115` |
| V2 | Hybrid / Nemotron-H KV, F32 default (F16 only for Qwen4-Exp, opt-in) | `2·Lattn·stride·4`: **B2 128 KiB/tok** | `VulkanNemotronHKvCache.cs:88-116` (#1102) |
| V3 | All inter-kernel activations F32 (except the ffn-down F16 tile) | `4·seqLen·(3H + 2q + 2kv + 3I)` | `VulkanForwardState.cs:387-416` |
| V4 | MTP KV and hidden F32 | `2·maxSteps·stride·4` | `VulkanMtpState.cs:109,236` |
| V5 | Split-KV partials and split-K GEMM slabs | `numHeads·splits·514·4`; 5.24 MB per split-K instance | `VulkanSplitKvAttentionKernel.cs:407`, `CoopmatSplitK.cs:41` |
| V6 | Logits `rows·V·4`, MoE scratch, recurrent state, MLA KV, Q8_1 `(d, s)` as F32, quantized-KV scales F32 | see sweep | `VulkanForwardState.cs`, `VulkanQuantizedKvCache.cs:119` |

**Rule 2 — expansions**

| # | item | cost | ref |
|---|---|---|---|
| VX1 | **Token embedding widened to F32 for every type but Q8_0** (and PQ2_0 on hybrid-dense) | `V·H·4`: Qwen3.8-27B **5.09 GB** vs ≈339 MB packed | `VulkanWeights.cs:1108,1727` |
| VX2 | Any matrix outside `DeviceQuantTypeFor` widened to F32 (Q4_1, Q5_1, MXFP4, odd K, I2_S lm_head, …) | `m·k·4` | `VulkanWeights.cs:1465-1489,1727` |
| VX3 | Quantized-KV and TurboQuant dequant into a shared F32 full-history scratch | `2·maxSeq·stride·4`, plus a per-layer rewrite of `curLen·stride·4` | `VulkanQuantizedKvCache.cs:121-199` |
| VX4 | IQ1/2/3 (and fallback NVFP4) prefill dequant → F16 scratch → coopmat | `maxLayerElements·2`: Llama-3.2-3B 50.3 MB, plus 4 B/weight traffic per call | `IqF16PrefillMatmul.cs:133` |
| VX5 | Shared-expert and expert-bank F32 fallbacks; legacy MLA F32 | up to ≈265 GB for Bolt banks if they fall through | `VulkanQwen3MoeMoeUpload.cs:68-96` |

**Also flagged (#374):** the Vulkan device **never enables `shaderFloat16`, 16-bit storage or `shaderInt8`**, although
≈64 shaders use them (`VulkanDevice.cs:1316,1429-1475`). AMD tolerates this. Intel is reportedly strict. **Fix this
before any F16-storage migration on Vulkan.**

---

## 5. Experiment results (placeholders)

### 5.1 CUDA (RTX 3060) — from `.docs/precision/cuda-results.md`

**TODO(cuda).** Fill in from the CUDA workstream.

| experiment | model | ref | variant | KL median | KL p99 | outlier rate | top-1 agree | PPL Δ | speed Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| prefill GEMM accumulate | Bonsai-2 (K ≤ 17,408) | F32 acc | F16 acc (`COMPUTE_16F`) | | | | | | | |
| prefill GEMM accumulate | Bonsai-2 | F32 acc | F16 acc + F32 promote / 128 | | | | | | | |
| flash-attn QKᵀ accumulate | Bonsai-2 | F32 | F16 | | | | | | | |
| flash-attn P·V accumulate | Bonsai-2 | F32 | F16 (+ ln 2 max offset) | | | | | | | |
| INT8 mma W8A8 | | F16 cuBLAS | int8 mma, Q8_1 B | | | | | | | |
| INT8 mma W4A8 | | F16 cuBLAS | int8 mma, 4→8 dequant | | | | | | | |
| KV F16 / q8_0 / q4_0 / K8V4 | Bonsai-2 + one Llama | F16 | each | | | | | | | |

### 5.2 CPU (155H, Zen 5) — from `.docs/precision/cpu-results.md`

**TODO(cpu).** Fill in from the CPU workstream.

| experiment | CPU | ref | variant | KL median | KL p99 | outlier rate | top-1 agree | PPL Δ | tok/s Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Q8_1 / VNNI activations | 155H | F32 act | W4A8 / W8A8 VNNI | | | | | | | |
| Q8_1 / VNNI activations | Zen 5 | F32 act | VNNI-256 / VNNI-512 | | | | | | | |
| BF16 dot (`VDPBF16PS`) | Zen 5 | F32 | BF16 | | | | | | | |
| KV F16 / q8_0 (#978) | 155H, Zen 5 | F32 `SimpleKvCache` | F16, q8_0, q4_0 | | | | | | | |
| prefill activations F16/BF16 | both | F32 | F16 / BF16 | | | | | | | |
| RoPE F16 tables (≥32k) | both | F32 tables | F16 tables | | | | | | | |
| where integer beats float | both | — | per kernel | | | | | | | |

### 5.3 Vulkan (Strix, Arc)

**TODO(vulkan)**, after #1100–#1102. Experiments:

- WMMA f16 vs iu8;
- F16 vs F32 coopmat accumulation;
- packed-half shader paths;
- KV formats on the hybrid path;
- Arc DP4a MMQ prefill.

| experiment | device | ref | variant | KL median | KL p99 | outlier rate | PPL Δ | speed Δ | verdict |
|---|---|---|---|---|---|---|---|---|---|
| | | | | | | | | | |

### 5.4 Recurrent state (all backends)

**TODO.** Long-decode (≥8k) KL with carried state at F32 vs BF16 vs F16, using F32 compute in all cases.

---

## 6. Candidate optimisations, ranked by expected benefit per device

Ranking key: (bytes saved or speed gained on the hot path) × (likelihood of passing the accuracy bar) ÷ effort.
"Evidence" points to §3.

### RTX 3060
1. **Remove prefill dequant → F16 scratch (GX1, #495) with an int8-mma MMQ (W4A8/W8A8, Q8_1 activations).**
   - Speed: int8 mma is 101.9 TOPS, 4× the F32-acc HMMA rate.
   - Memory: the 2 B/weight write and read and the scratch disappear.
   - Evidence: llama.cpp MMQ, QServe. Risk: low (decode already runs dp4a W?A8).
2. **KV q8_0 as the default** (K8V4 when capacity-bound) with in-tile dequant. It **deletes GX3's always-on F16
   scratch.** Evidence: Bonsai PPL −0.004%.
3. **Drop the unconditional F32 activation set (G1) and `DequantScratchF32` (G2)** for FP16-path models. 0.3–0.8 GB
   freed at zero accuracy risk.
4. **LM head >8 rows in place (GX2, #1099)**, and verify-time argmax on device. 2.54 GB on Bonsai-2.
5. **Hybrid activations F32 → F16 (G3), MTP KV F32 → F16/q8_0 (G5), F32 staging (G4) removed.**
6. **F16-accumulate HMMA (G1 default-on for hybrids) and F16 P·V accumulator in flash-mma.** Up to 2× on those
   instructions; gated on **TODO(cuda)** KL.
7. Sage-style int8 QKᵀ for long prefill. Research grade.

### Radeon 8060S (Vulkan)
1. **Hybrid and dense KV F32 → F16 (#1102), then q8_0 read in-shader (no F32 scratch, VX3).** 2–3.8× KV bytes; B2:
   128 KiB → 64 KiB → ≈34 KiB/tok.
2. **Embedding table kept packed (VX1).** −4.7 GB on Qwen3.8-27B.
3. **Activations F32 → F16 (V3).** Measured 1.74× on ffn-down at k=9216 (Infinity Cache fit). Fix #374 feature
   enables first.
4. Native coopmat for the remaining IQ types (VX4) and in-place kernels for VX2's fall-through types.
5. F16-accumulate coopmat: register and occupancy gain; rate gain UNVERIFIED. **TODO(vulkan)**
6. iu8 WMMA MMQ prefill: bytes only on RDNA3.5, same rate as f16. Low priority.

### Arc iGPU (Vulkan)
1. **KV and activations to F16 / q8_0.** At ≈90 GB/s shared bandwidth every byte counts.
2. **DP4a MMQ prefill everywhere** (4× FP32; there is no coopmat to fall back on).
3. Fix #374 (Intel strictness) before any F16 storage.

### Core Ultra 155H CPU
1. **KV F32 → q8_0 (or F16) for `SimpleKvCache` (C1, #978), with int8 VNNI QKᵀ.** 3.8× fewer KV bytes. The ≈30% Llama-1B
   decode deficit vs llama.cpp is hypothesised to be F32 KV (`docs/PERFORMANCE.md`).
2. **VNNI prefill tiles for Q8_0 and W4A8 for Q4_0/Q4_1/Q5_1/IQ4_NL/PQ2_0** (X6). VPDPBUSD is 4× FP32 FMA. Today they
   use maddubs or dequant→F32.
3. **Logits: sampled rows only (C3).** ≈2 GiB at 2048 for L8B.
4. Fuse norm → Q8 everywhere, so decode activations never round-trip F32.
5. Activations and RoPE tables to F16 (C2, C6), mostly memory.

### Zen 5 CPU
1. Same KV item as the 155H (C1).
2. **`VDPBF16PS` for BF16/F16 weights** (safetensors BF16 is widened to F32 today, X1: +16 GB on L8B). Runs at 2×
   FP32 and **removes the expansion**. Needs `Avx512Bf16`, which is on the `dev-dotnet11` track (keep a mainstream
   fallback).
3. VNNI W8A8/W4A8 prefill (256-bit mainstream; 512-bit only per-kernel where no byte widen is needed). Gate by
   measurement (the Zen 5 widen trap).
4. Drop or bound the R4 duplicate (X4) where repacking does not measurably win.

### Westmere (T5500)
1. q8_0 KV and Q8 activations with SSSE3 dots: smaller *and* cheaper than F16.
2. Prefer **BF16 over F16** for any 16-bit storage (shift decode vs software F16).

### Cross-device
- **Recurrent state:** keep F32 pending the §5.4 measurement. Narrow snapshots and conv state to F16 now.
- **Split partials:** O partials and split-K slabs to F16; *m*/*ℓ* stay F32.

---

## 7. Open questions and UNVERIFIED items

- RTX 3060 HFMA2 rate (1× vs 2× FP32) and `__dp4a` rate. Microbenchmark them.
- RDNA 3.5 F16-accumulate WMMA rate vs F32-accumulate. Also whether the Windows driver exposes F16-result coopmat
  shapes (re-run `Probe533CoopmatDeviceEvidenceTests` and commit the shape table).
- How much of the ≈256 GB/s the Strix CPU CCDs can reach.
- Whether long-context (≥64k) RoPE with F16 cos/sin tables is KL-neutral. Our derivation says yes; unmeasured.
- Carried recurrent state at 16 bits over long decodes. No published inference study found.
- The K vs V sensitivity ordering per model family (TurboAngle conflict).
