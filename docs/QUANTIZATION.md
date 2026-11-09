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

### MXFP4 (4.25 bits/weight)

```
struct block_mxfp4 {         // 17 bytes, 32 values (GGUF type 39)
    uint8_t e;               // E8M0 shared scale (power of two)
    uint8_t qs[16];          // 32 × 4-bit e2m1 indices packed into 16 bytes
};
Value table (doubled e2m1): {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}
Scale: e8m0_to_fp32_half(e) = 2^(e-127) / 2  (halved to compensate doubling)
Unpack: lo nibbles → elements 0..15, hi nibbles → elements 16..31
Dequantize: val = kvalues[nibble] * scale
```

OCP Microscaling FP4. Used by OpenAI gpt-oss checkpoints for MoE expert
weights. CPU vec_dot pairs MXFP4 weights with Q8_0-quantized activations
(nibble→sbyte pshufb LUT + integer MAC), mirroring llama.cpp's
`ggml_vec_dot_mxfp4_q8_0`.

### NVFP4 (4.5 bits/weight, GGUF type 40, #1038)

```
struct block_nvfp4 {         // 36 bytes, 64 values (QK_NVFP4 = 64, QK_NVFP4_SUB = 16)
    uint8_t d[4];            // one UE4M3 scale per 16-element sub-block
    uint8_t qs[32];          // 64 x 4-bit e2m1 indices
};
Value table: the doubled e2m1 table shared with MXFP4
UE4M3: unsigned, 4 exponent bits (bias 7), 3 mantissa bits, returned halved;
       0x00 and 0x7F decode to 0, bit 7 is ignored (ggml_ue4m3_to_fp32)
Unpack: sub-block s owns qs[8s..8s+7]; byte j -> element 16s+j (low nibble), 16s+8+j (high nibble)
Dequantize: val = kvalues[nibble] * ue4m3_half(d[s])
```

NVIDIA FP4. A tensor `X.weight` stored as NVFP4 usually has a sibling `X.scale`
(F32, one element, NVIDIA's "scale2") that multiplies the matmul **output**
(llama.cpp `build_lora_mm`: `mul_mat(w, x) * w_s`). In NVIDIA's Nemotron 3.5 Bolt
GGUF these are about 1e-4, so a loader that drops one is off by ~4 orders of
magnitude. `Nvfp4Support` makes every GGUF loader name the scales it applies and
refuse the rest. Scale layouts: a per-tensor scalar `[1]` (attention q/k/v/output, dense
FFN, shared expert, `output.scale` for an NVFP4 LM head) and a **per-expert vector `[E]`**
for the routed banks (`ffn_{gate,up,down}_exps.scale`, weights `[in, out, E]`); expert `e`'s
matmul result is multiplied by `scale[e]` (llama.cpp `build_lora_mm_id`), gate and up
before the SwiGLU, down before the router weight.

Support matrix (#1579 tracks the rest): **CPU** applies the scalar scales in the generic
`TransformerModel` (any architecture whose NVFP4 weights are the projections above or the
LM head) and the per-expert/shared scales in the grouped MoE kernel (Flint, MiMo-V2;
prefill GEMM and n=1 decode row kernel, bit-identical); an NVFP4 token embedding with a
scale is still refused. **Vulkan** (#1579) reads NVFP4 in place on the generic
transformer (Flint verified; dense projections via the packed GEMV / coopmat GEMM with the
per-tensor scale in the epilogue, routed banks via an indexed GEMV for decode and a grouped
coopmat GEMM for prefill with the per-expert scale in the epilogue; no F16/F32 copy of any NVFP4
weight; the shared-expert scalars ride the dense path). MLA, Gemma-4, per-layer-embedding and
the hybrid-SSM architectures still refuse NVFP4 on Vulkan. **CUDA** applies scales on the
Nemotron-H SSM projections only and refuses NVFP4 elsewhere (stage 3). Qwen3-hybrid, qwen4exp, LFM2-MoE,
BERT and gemma-embedding loaders still refuse scaled NVFP4. Measured on
`RecursiveSI/flint-nvfp4-gguf` (`flint-nvfp4.gguf`): wikitext-2 LF 14 KB, 512 ctx, 5 windows,
CPU 24.36 +/- 2.94 vs llama.cpp (master dde879bc, `-dev none`) 24.26 +/- 2.91.

The CPU vec_dot pairs NVFP4 with Q8_0 activations (two Q8_0 blocks per NVFP4 block),
a transcription of llama.cpp's `ggml_vec_dot_nvfp4_q8_0` (scalar + AVX2). Since #1587 every CPU path
(scalar, single-row AVX2, the eight-row multi-column prefill/decode kernel `ComputeRowsNvfp4Avx2_8Rows`) is
**bit-identical**: exact int32 sub-block sums, then the scalar float recurrence `sumf += (dy*d)*sumi` in
sub-block order. The eight-row kernel (rows one per lane, 64-token chunks, kvalues shifted to `u = kv + 12`
for `vpdpbusd` with `-12*Sum(q)` folded into the accumulator start) unpacks each weight block once for all
tokens: 12.2 -> 3.5-4.0 ms at m=4096 k=2048 n=256 on Zen 5 (Q4_1: 2.7-3.1 ms), a token's result no longer
depends on the batch. `DOTLLM_CPU_NVFP4_ROWS8=0` restores the row-at-a-time path (same numerics).
**`DOTLLM_CPU_NVFP4_LEGACY_ORDER=1`** (read once) restores the pre-#1587 float summation order (8 lane partials with FMA, summed horizontally; row-at-a-time, single- and multi-row, decode included) so perplexity figures measured before #1587 reproduce (Flint 4 windows: 25.39 legacy vs 26.23 default). The shift is summation order only (five float orders over identical integer sums span 25.39-26.23, docs/PERPLEXITY.md); the new order was accepted as the default for the speed-up. Note this changed the
float summation order versus the pre-#1587 AVX2 (8-lane FMA partials, which mirrored llama.cpp's x86 order),
which moved Flint wikitext perplexity on a 4-window (2048-token) CPU run from 25.39 to 26.23 and on a
2-window run from 23.30 to 24.84 (same weights, same Q8_0 activations; only the float accumulation order
differs, and the new kernel is bit-equal to the scalar transcription that the llama.cpp oracle tests pin).
Treat that as the rounding-order sensitivity floor of this 256-expert top-6 model when comparing engines (#1584). Vulkan keeps
the blocks packed and dequantizes in the shader against FP32 activations
(`matmul_nvfp4_f32_gemv/gemm`, shared `nvfp4_decode.glsl`), applying `X.scale` to the output.
CUDA (`native/kernels/nvfp4.cu`) decodes with a dp4a GEMV (`nvfp4_gemv_q8`, #1061): the token
is quantized to int8 once by the house F32 activation quantizer (`pq2_0_dp4a_quantize_x`,
byte-exact with the CPU's Q8_0 activation quantizer), the E2M1 codes are looked up as int8 with
`__byte_perm`, and `X.scale` is folded in. That is the CPU's NVFP4 × Q8_0 numerics (only the FP32
summation order differs) at ~315 GB/s of weights on an RTX 3060 at Bolt's `ssm_in` shape
(136 µs, against 431 µs / 99 GB/s for #1038's FP32-activation GEMV). The FP32-activation GEMV
remains for weights under 4 MiB, where the extra quantize launch costs more than it saves
(`DOTLLM_NVFP4_DECODE_GEMV=auto|dp4a|f32`). Prefill goes through dequant-to-F16 + cuBLAS
followed by an in-place scale kernel.

### I2_S (BitNet b1.58 ternary, ~2 bits/weight, GGUF type 36)

```
Row-major W[m,k]: 4 ternary codes {0,1,2} → {-1,0,+1} packed per byte (2 bits each).
128-element block = 32 bytes. Byte at group_pos (0..31) holds codes for elements
{group_pos, +32, +64, +96} at bit offsets {6,4,2,0}.
ONE per-tensor float32 scale at the tensor tail, byte offset m·k/4.
Dequantize: val = (code - 1) * scale
```

**Ragged K (`k % 128 != 0`, issue #206).** Most I2_S GGUFs have every row length (`k`) an exact
multiple of 128 (e.g. microsoft/bitnet-b1.58-2B-4T: 2560/6912). At least one real checkpoint family
(1bitLLM-style `bitnet_b1_58-large`/`-xl`: hidden=2048, intermediate=5460, `5460 % 128 == 84`) has a
genuinely non-128-aligned `ffn_down` row length. The critical subtlety, verified against the real
GGUF's tensor byte offsets and against the upstream bitnet.cpp writer (`ggml-bitnet-mad.cpp`'s
`quantize_i2_s`): **the 128-element block interleave is computed over the flattened `m·k` element
stream, not reset at each row boundary.** So a ragged row generally does not start on a block
boundary at all (only every `128/gcd(k,128)`-th row does) — a "tail cleanup after the fast path"
approach would be wrong for most of a ragged tensor's rows, not just the last few elements. When `k`
is a multiple of 128 this is moot: every row boundary is also a block boundary, so per-row
block-reset addressing and the flattened-stream addressing are bit-identical (why the 128-aligned
fast paths never noticed the distinction). dotLLM's ragged-K support (CPU: `MatMul.I2S.cs`'s
`I2SRaggedCode`/`UnpackRowRagged`; CUDA: `i2_s_gemv_{f16,f32}in_ragged` / `dequant_i2_s_f16_ragged`
in `native/kernels/`) is a scalar, correctness-first fallback reached only when `k % 128 != 0` — the
aligned SIMD/uint4 fast paths are untouched.

### TERN_B64_F16 (ternary, 2.25 bits/weight, dotLLM-private type 243, issue #1706)

Per-64-column FP16-scaled ternary weights, used by tensor overlays (below). Not a ggml type: the id 243 exists only in dotLLM overlay GGUFs.
It is **not** `PQ2_0` (128-weight groups, scale first, codes 0->-1) and **not** upstream `Q2_0` (same 18 B/64 geometry, but codes 0..3 = -1,0,+1,+2).

```
tensor  : row-major [out, in] (GGUF dims [in, out]); in % 64 == 0
block   : 64 consecutive input columns of ONE row, 18 bytes
          [0..1]  fp16 scale, little-endian IEEE
          [2..17] 16 bytes of 2-bit trits; byte k holds columns 4k..4k+3, column 4k+i in bits 2i..2i+1 (LSB first)
trit    : 00 = 0, 01 = +1, 10 = -1, 11 = INVALID (rejected at load)
weight  : w[row, 64*b + j] = trit * scale[row, b]        (a zero block has scale 0)
order   : blocks ascending within a row, rows ascending
```

CPU kernels (`DotLLM.Cpu.Kernels.TernB64F16`): the weights are read packed, never expanded into a stored copy (one row is unpacked to int8 in a per-thread
scratch). `GemmQ8`/`GemvQ8` is the production kernel (activations through the house Q8_0 quantizer, `int32` dot per 32 activations, folded with the 64-column
scale); `GemmF32Act` is the scalar F32-activation reference (the KL oracle; `DOTLLM_TERN_F32ACT=1` runs a whole model on it). Against the OracleWrittenTernary
real-tensor vectors the F32 kernel agrees to ~1e-6 of rms(y) and the int8 kernel to 0.5% rms / ~3% max of rms(y). `Dequantize` exists as a test oracle
only on the CPU; the Vulkan kernel (S3 of #1706) reads the same layout, see "Vulkan" under Tensor overlays.

## Tensor overlays (`--tensor-overlay`, issue #1706)

An overlay is a small GGUF that replaces **only the named FFN projections** of a base model, so any subset of `(layer, expert, projection)` can run in another
format (typically `TERN_B64_F16`) while everything else keeps its original precision. Available on `run`, `perplexity` and `serve`
(`--tensor-overlay overlay.gguf`, plus `--overlay-skip-base-hash`); CPU backend, plain transformer / stacked-expert MoE GGUF models and HF safetensors
bases (OLMoE, Mixtral-style, dense Llama family) in this release. On a safetensors MoE base each layer's per-expert F32 matrices join the same side table
(no copy), so the layer runs `MoeQuantSwiGluMlp`.

| rule | detail |
|---|---|
| names | `blk.{L}.ffn_{gate,up,down}.{E}.weight` (routed expert E), `blk.{L}.ffn_{gate,up,down}.weight` (dense FFN). Anything else is an error. |
| shapes | GGUF dims `[in, out]`, equal to the base projection (gate/up `[hidden, ffn]`, down `[ffn, hidden]`). Mismatch is an error. |
| dtype | per-tensor GGUF type. `TERN_B64_F16` is validated at open (in % 64 == 0, no `0b11` code); other types (F16, F32, Q8_0...) are accepted wherever the CPU GEMV supports them. |
| I8 container | python `gguf` can only write I8, so a tensor may be stored as raw **unsigned** bytes in an `I8` tensor of GGUF dims `[in/64*18, out]` with `dotllm.overlay.dtype.{name} = "TERN_B64_F16"` and `dotllm.overlay.shape.{name} = [out, in]` metadata. Accepted only when `general.architecture = dotllm-overlay` (a model GGUF with an I8 tensor is still rejected). Byte count must equal `out*(in/64)*18`. A native type-243 tensor of shape `[in, out]` is equivalent. |
| base binding | metadata `dotllm.overlay.base_sha256` = lowercase hex SHA-256 of the base file (first shard of a split set). Missing or mismatched is an error; the literal value `UNSPECIFIED` skips the check with a logged warning. A safetensors-directory base has no single file to hash: the check is skipped with a note. The digest is cached per (path, length, mtime) in `~/.dotllm/hash-cache` (`DOTLLM_HASH_CACHE_DIR`); `--overlay-skip-base-hash` disables the check (a 4 GB base hashes in seconds, a 50 GB one in minutes on first use). |
| never ignored | unknown name, wrong layer/expert, dense name on a MoE layer (or vice versa), a loader that did not consume the overlay (GPU backends, other architectures) are all hard errors (`TensorOverlay.VerifyAllApplied`). |

How it is applied (the contract Vulkan reuses): `TensorOverlay.Open/Attach` validates and attaches to the base `GgufFile`; the CPU `TransformerModel` loader then
swaps a dense projection's pointer/format, or fills a per-layer `ExpertOverlayTable` (per-expert, per-projection address + format) beside the untouched stacked
`ffn_*_exps` banks. `MoeQuantSwiGluMlp` consults the table per selected expert and per projection (token routing, top-k and `norm_topk_prob` are unchanged);
with no overlay the table is null and the code path is the pre-existing one. Overlay tensors are read in place from the overlay's mmap, never copied.

### Vulkan overlay support (#1706 S3)

`--tensor-overlay` also runs on `--device vulkan` for GGUF bases of plain-SwiGLU dense and routed-MoE models (Llama, Mistral, Qwen, Mixtral, OLMoE / Qwen-MoE; no
shared-expert or sigmoid-routing variants, no pipeline windows) and **`TERN_B64_F16` tensors only**; any other overlay dtype, architecture or backend (CUDA) is
still the loud "not applied / not supported" error. A safetensors directory base is CPU-only (`perplexity` refuses it on `--device vulkan`).

- **Storage:** packed, 18 B per 64 weights (2.25 bpw), never widened. Dense FFN overrides ride the normal upload (`DeviceQuantType = TERN_B64_F16`, zero-copy
  import when the platform allows). Routed experts: per layer and projection one bank of the overlaid experts' tensors back to back plus an `int32[numExperts]`
  slot table (`VulkanTernOverlay`); the original bank still holds the replaced experts' original weights (unread for those cells).
- **Kernel:** `matmul_tern_b64_mmvq.comp`, dense and indexed in one shader (wave-size independent: shared-memory reduction). Activations go through the house
  Q8_1 row quantizer; one int32 dot per 16 columns, folded with the fp16 block scale and the per-32 activation scale (`acc += scale * d_x * dot`), the same
  contract as `TernB64F16.GemmQ8` (the CPU quantizer rounds the activation scale to fp16, so the two are KL-level, not bit-identical). Grid = one 32-lane
  workgroup per (output row, activation row): decode and prefill use the same kernel; weight rows are re-read per activation row (cache-resident), so prefill is
  not coalesced into a tiled GEMM yet.
- **Mixed formats in one layer:** a layer with ternary experts takes the existing generic indexed path for every expert (original formats, unchanged), then a
  ternary pass rewrites the (token, slot) cells whose expert is ternary for that projection (`slotOf[expert] >= 0`). Routing, top-k and `norm_topk_prob` are
  untouched. Such a layer does not use the grouped-coopmat / fused-MMVQ arms (their intermediates cannot be overwritten), so overlay layers decode slower than
  untouched ones.
- **Reference:** the F32-activation oracle is the CPU `TernB64F16.GemmF32Act` (`DOTLLM_TERN_F32ACT=1`); there is no F32-activation Vulkan variant.

### NanoQuant (community GGUF, sub-1-bit factorised; issue #866)

Not a block quant: a linear is a **composite** of tensors, `W ~ diag(post) U diag(mid) V diag(pre)` with U, V in {+-1}.
Community layout (`quantization.nanoquant.version = 2`, `bit_order = lsb_first`, `positive_bit = 0`,
`outlier_format = column_side_path`), per linear `<base>`:

| tensor | ggml dtype | GGUF dims (fastest first) |
|---|---|---|
| `.nq_u` | **I32** (type 26) | `[ceil(r/32), d_out]`  sign words, bit set = -1, LSB-first, tail clear |
| `.nq_v` | **I32** | `[ceil(d_in/32), r]` |
| `.nq_scale_pre / _mid / _post` | BF16 (30) / F16 / F32 | `[d_in] / [r] / [d_out]` |
| `.nq_salient_idx` | I32 | `[k]` strictly increasing input columns (scale_pre is exactly 0 there) |
| `.nq_salient_weight` | F16/BF16/F32 | `[k, d_out]`; `y += W_sal . x[idx]` using the RAW x |

`y = post * (U (mid * (V (pre * x)))) + salient`. The sign layout is byte-identical to the LittleBit spike (#832), so
`NanoQuantLayer` (`DotLLM.Cpu.Kernels.Experimental`) is one `LittleBitPath` plus the salient side path, with a float64
scalar reference (`NanoQuantReference`). `NanoQuantLoader` (`DotLLM.Models.Quantization`) reads the composites. Stacked
`attn_qkv` is just `d_out = dq+dk+dv`; ranks differ per tensor. Scales are held fp16 by the spike, so loading refuses
bf16 values that are not exact in fp16 (the whole Qwen3-0.6B file is exact). `I32` is a raw-word type: no matmul kernel
takes it. Quality of the available community files is low (Qwen3-0.6B WikiText PPL 55 -> 292); they are format/kernel
vehicles. Test: `DOTLLM_NANOQUANT_GGUF` (or the HF-cache copy of `arelath/Qwen3-0.6B-nanoquant-GGUF`).

**End-to-end CPU load (issue #869).** `NanoQuantLoader` composites are registered as `QuantizationType.LittleBit` weights
(`FactorizedWeights`, see LittleBit above): `TransformerModel.LoadFromGguf` loads the community Qwen3-0.6B file as an ordinary
dense Qwen3. Stacked `attn_qkv` is split by row into q (n_heads*head_dim), k, v (n_kv*head_dim) layers with the shared right
factor replicated (3x stage-1 work for qkv; ~0.3 MiB each, negligible memory); the salient columns are applied by
`GemmLittleBit` after the base GEMM on the raw input. Scales are F32 (bf16 widened exactly), so the former fp16-exactness guard is
gone. Vulkan/CUDA/HIP/hybrid loaders and any non-Qwen3 architecture throw `NotSupportedException` ("CPU backend only" / "dense
Qwen3"); the CPU-only switch is `TransformerWeights.LoadFromGguf(..., allowNanoQuant: true)`.

- **Oracle** (modified llama.cpp fork `arelath/llama.cpp@nanoquants`, mingw CPU build, no repack, 3 prompts / 22 positions): top-1 agreement
  22/22, per-position KL(oracle||dotLLM) 1.2e-7..7.9e-5 nats, max |dlogit| 0.002..0.065 over the 151,936-entry vocab; greedy 16-token
  continuation identical for 2 of 3 prompts and 13/16 for the third (first divergence at generated token 14, cause not investigated; per-position logits agree to KL<1e-4). Test: `NanoQuantEndToEndTests` (committed top-1/top-5/greedy in `NanoQuantData/oracle_p*.json`; full logits from
  `DOTLLM_NANOQUANT_ORACLE_DIR` / `~/.dotllm/test-cache/nanoquant-oracle`, skipped when absent).
- **F32-decoded control** (`DOTLLM_LITTLEBIT_DENSE_CONTROL=1`: every projection incl. salient columns decoded to dense F32): layer-level
  kernel-vs-dense gap 3.4e-6 of RMS (blk.0.ffn_gate); the model logits are (near-)bit-identical because the Q8_0 head quantises its
  input to int8, which absorbs that gap.
- **Perplexity** (`dotllm perplexity`, WikiText-2 test LF fixture, token ids dumped from the GGUF run so teacher and student score the
  identical stream): 64 windows x 128 tokens (closest to the card protocol; card: BF16 55.17, NanoQuant 292.26, ratio 5.30x) -> dense bf16
  Qwen3-0.6B (rev c1899de) **53.88**, NanoQuant **302.49**, ratio **5.61x**. 32 windows x 512 (second half scored): teacher 19.30, NanoQuant
  99.96 (5.18x). Absolute numbers differ from the card because the card's own window sampling / batching is not reproducible from the card;
  the ratio is the comparable figure and agrees to ~6%.
- **Memory**: the 196 factorised projections hold 63.4 MiB resident (packed signs + F32 scales + F32 salient columns) vs 840 MiB as dense
  bf16 (1680 MiB as the F32 the dense CPU path upcasts to): 13x/26x smaller. File 375 MiB total, dominated by the Q8_0 embedding and head
  (2 x 157.6 MiB, mmap'd).
- **Decode** (`dotllm run`, 64 tokens, greedy, CPU, 3 interleaved same-session pairs on a loaded box): NanoQuant 43.9 / 29.0 / 23.9 tok/s vs dense
  bf16 (upcast to F32) 10.1 / 10.8 / 15.2: per-pair ratios 4.3x / 2.7x / 1.6x, median 2.7x. The NanoQuant file also benefits from its Q8_0 head
  (the dense baseline streams an F32-upcast tied head), so read it as an upper bound of the projection-only gain.


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

### Decode and prefill are two valid orderings of the same sum (issue #538)

A quantized projection is computed by a **different kernel** depending on how many
tokens are in flight, and the two do not agree bitwise. On Vulkan Q8_0, `n == 1`
takes `matmul_q8_0_mmvq` (the decode GEMV) and `n > 1` takes `matmul_q8_0_mmq`
(the prefill GEMM). This is expected, not a defect — but two things about it are
easy to get wrong, so they are recorded here.

**The activations are identical; only the accumulation order differs.** The two
Q8_1 activation quantizers (`QuantizeQ8_1Kernel` and `QuantizeQ8_1RowsKernel`)
emit **bitwise identical** `xq`/`xds` at `n == 1`, measured, with a control that
reports a difference on two different rows. So the arms are summing the same
numbers in a different order, and both land ~1E-07 RMS from an f64 accumulation
of the same quantities. Neither is wrong.

**But the prefill kernel is the less accurate of the two**, by 1.65× at K=2048
and 2.37× at K=8192 — and the gap widens with K because the mechanism is
**reduction depth**, not a missing scale (both scale per 32-block):

| | how the block products are summed | error growth |
|---|---|---|
| `matmul_q8_0_mmvq` (decode) | blocks split across a subgroup, finished with `subgroupAdd` — a tree | O(log n) |
| `matmul_q8_0_mmq` (prefill) | all K/32 blocks accumulated sequentially into one float | O(n) |

Two consequences worth carrying:

1. **Do not "unify" a dispatch split by routing `n == 1` through the `n > 1`
   kernel.** In both halves of #538 the multi-token path is the *less* accurate
   one (the same is true of attention, where prefill uses f16 coopmat tiles and
   the 1-token path the f32 dense kernel). Converging on it degrades output.
2. **Perplexity is scored through the prefill path**, so prefill accuracy is a
   quality lever in its own right, independent of any chunking-consistency
   question.

`Probe538Q8SplitTests` holds this as a standing gate: the quantizers must stay
bit-identical, MMVQ must stay deterministic, neither arm may drift from the f64
oracle, and MMQ may not fall further than 5× behind MMVQ.

## Vulkan Backend Coverage

The Vulkan backend ships native matmul kernels (GEMV decode + GEMM prefill, with an opt-in F16 cooperative-matrix tile when the device enumerates F16xF16→F32) for the following source dtypes / quant formats. Source bytes stay on device — dequantisation happens in the shader inner loop, so memory cost is the GGUF source size (not 2-4× expanded F32):

| Format | GEMV | GEMM | Coopmat | Reference |
|---|---|---|---|---|
| F32 | ✓ | ✓ | — | baseline |
| F16 | ✓ | ✓ | ✓ (F16xF16→F32, M=N=K=16 tile) | `c9c08c5` |
| BF16 | ✓ | ✓ | — (BF16 tiles not enumerated on RDNA3.5; use shift-left-16 reinterpret) | `c9c08c5` |
| Q8_0 | ✓ | ✓ | ✓ (`MatMulQ8_0GemmCoopmatKernel`) | pre-existing |
| Q2_K | ✓ | ✓ | — | layout fixed in #498 |
| Q3_K | ✓ | ✓ | — | layout fixed in #311 |
| Q4_K | ✓ | ✓ | — (Phase 1 follow-up) | `afb2272` + `b1ee6bc` |
| Q5_K | ✓ | ✓ | — | `15099b9` + `83e0732` |
| Q6_K | ✓ | ✓ | — | `29a1459` + `39b7646` |
| Q5_0 | ✓ | ✓ | — | legacy-quant pair of Q8_0 |
| IQ4_NL | ✓ | ✓ | — | IQ-family Phase 2 |
| IQ4_XS | ✓ | ✓ | — | IQ-family Phase 2 |
| IQ3_XXS | ✓ | ✓ | — | IQ-family |
| IQ3_S | ✓ | ✓ | — | IQ-family |
| IQ2_XXS | ✓ | ✓ | — | `79cca9b` |
| IQ2_XS | ✓ | ✓ | — | `743984c` |
| IQ2_S (also IQ2_M) | ✓ | ✓ | — | `9ecce75` |
| IQ1_S | ✓ | ✓ | — | IQ-family — smallest GGUF quant (~1.5-1.7 bpw) |
| I2_S | ✓ | ✓ | ✓ (`MatMulI2SGemmF32Kernel`) | BitNet b1.58 ternary |
| PQ2_0 | ✓ | ✓ | ✓ (`MatMulPQ2_0GemmF32Kernel`) | PrismML ternary + per-group scale |
| PTQ1_0 | ✓ | ✓ (untiled) | — | PrismML dense base-3 trits, 28 B / 128 (1.75 bpw), scale last; CPU + Vulkan only, CUDA refuses at load (#1280) |
| Q4_0 | ✓ (+ dp4a MMVQ) | ✓ | ✓ (`MatMulQ4_0GemmCoopmatKernel`, wave64) | #1018 |
| Q4_1 | ✓ (+ dp4a MMVQ) | ✓ | ✓ (same classes, `quant: Q4_1`, wave64) | #1171 (Recursive Flint's official file) |
| Q5_1 | ✓ (+ dp4a MMVQ `matmul_q5_1_mmvq_v2`) | ✓ (int8-coopmat `matmul_q5_1_mmq_cm1` on AMD, dp4a `matmul_q5_1_mmq` elsewhere, F32-in tiled GEMM fallback) | ✓ (`matmul_q5_1_f32_gemm_coopmat_128x128x4`, same classes as Q4_1, wave64) | #1600 F32-in pair, #1613 fast arms (GLM-4.5-Air's shared-expert `down` is Q5_1 on 24 of 46 layers); `DOTLLM_VK_Q5_1_FAST=0` keeps the F32-in pair |

Routed MoE banks of Q4_0 / Q4_1 (and the IQ family) are read in place by `MoeIndexedMatmulIqMmvqKernel` (decode, any row count when
no grouped kernel exists) and `MoeGroupedMatmulIqCoopmatKernel` (prefill, wave64 coopmat).

**The unshipped dense set is Q4_0 under `DOTLLM_VK_Q4_0_NATIVE=0` only** (dense Q5_1 shipped in #1600). For those the upload path falls back to
F32 dequant, so weight memory is quadrupled; every other `QuantizationType` stays on device
in its packed source form. The authority is `VulkanWeights.DeviceQuantTypeFor` — each
`Keep*OnDevice` predicate there additionally requires the contraction axis to be a multiple of the
format's group size, and a misaligned tensor of *any* format falls back to F32 regardless of kernel
availability.

**This table is the kernel inventory and the _dense_ (`VulkanWeights`) loader's coverage. The other
weight loaders keep strictly less**, because each carries its own `DeviceQuantTypeFor` /
`KeepQuantOnDevice` rather than sharing the dense one — a format having a kernel does not mean every
architecture's loader will route to it:

| loader | additionally falls back to F32 |
|---|---|
| `VulkanWeights` (dense) | — (the table above) |
| `VulkanNemotronHWeights` | I2_S, PQ2_0 (Q5_0 stays packed since #568). nemotron_h_moe routed expert banks: Q8_0 and Q5_1 as stored, Q5_0 repacked to Q5_1; any other expert quant is refused with a `NotSupportedException` (#402, #407) |
| `VulkanMamba3Weights` | Q5_0, I2_S, PQ2_0 |
| `VulkanQwen3MoeHybridWeights` | I2_S, NVFP4 only. Q2_K / Q3_K / IQ4_NL / IQ4_XS / IQ1_S packed since #627; Q4_0 / Q4_1 / Q5_0 / Q5_1 since #921 (`DOTLLM_VK_HYBRID_LEGACY_PACKED=0` restores the F32 widening for A/B; Q4_0 also honours `DOTLLM_VK_Q4_0_NATIVE=0`) |
| `VulkanQwen3HybridDenseWeights` | same as `VulkanQwen3MoeHybridWeights` (it uploads through that class's `UploadProjectionMatrix`; its own code handles only the packed token embed). Both hybrid hosts' `RecordMatmul` carry an arm per kept format — `VulkanHybridKeepPackedDispatchTests` fails the build of a keep arm without its dispatch arms |

Widening a hybrid loader is mechanical — add the missing `Keep*OnDevice` arms AND the `RecordMatmul`
arm in both hybrid hosts (the loader is shared) — but it is a code change, not a documentation one,
so check the loader for the architecture you are running before assuming a format stays packed.

Q5_1 is also supported as a routed-MoE `down` bank (`moe_indexed_matmul_q5_1_{f32,mmvq}.comp`), and Q5_0 `down` banks are repacked
bit-exactly to Q5_1 at upload. The dense Q5_1 kernels are the #1600 F32-in pair (decode GEMV, 16x16-tiled prefill GEMM; the always-present
correctness arm) plus the #1613 fast arms: dp4a MMVQ at decode on the shared Q8_1 row, int8-coopmat (AMD) or dp4a MMQ at prefill on Q8_1 rows,
and the 128x128x4 F16 coopmat GEMM where cooperative matrices exist but no MMQ does. The Q8_1-activation arms are not bit-identical to the
F32-in pair (activation rounding, the Q4_1 / Q4_0 contract of #1235 / #1447).

Ordering rationale, for why the family filled in as it did: K-quant Q4/5/6 came first because they
cover the majority of production GGUF deployments (`*-Q4_K_M.gguf` is the de-facto default for most
checkpoints). IQ4_NL and IQ4_XS followed as the most-used IQ-family quants in production
(Llama-3.1 / Qwen2.5 IQ4_XS). The IQ2 family was prioritised to enable Qwen3.6-A3B-IQ2_M
(~11.5 GB GGUFs) on Strix Halo without a 4× expansion to F32 (~46 GB) at upload. IQ1_S closed the IQ
family at the smallest end, and Q2_K + Q3_K — the densest K-quants — closed the K-quant family;
their matmul kernels share the dispatch shape of Q4/5/6_K (one workgroup per output row at decode,
16×16 output tile at prefill) but use a 16-element K-chunk to match Q2/3_K's 16-element sub-block
size.

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

**Q3_K layout (fixed in #311).** Q3_K's decode is the easiest K-quant to get wrong, and dotLLM shipped it wrong in *every* backend until #311. Two independent transpositions, both now corrected against llama.cpp's `dequantize_row_q3_K`:

1. **6-bit scale high bits.** The high 2 bits of sub-block `sub`'s scale live in `scales12[8 + (sub % 4)]` at shift `(sub / 4) * 2` — **not** `scales12[8 + sub/4]` at shift `(sub % 4) * 2`. The two agree only for `sub ∈ {0, 5, 10, 15}`, so the wrong form corrupts 12 of 16 sub-block scales.
2. **Element ordering.** The 2-bit quants are **not** stored four consecutive elements per byte. Each 128-element half of the super-block uses 32 `qs` bytes, and every byte supplies **four elements 32 apart**: element `t` reads bit-pair `(t/32) % 4` of `qs[(t % 32) + 32*(t/128)]`. The `hmask` is transposed the same way: bit `t/32` of `hmask[t % 32]` (llama.cpp's `shift`/`m` loop). The old `qs[t/4] >> (t%4)*2` / `hmask[t/8] >> t%8` form scatters every element into the wrong sub-block scale.

Combined, the old decode produced **noise**: dequantized Bielik-1.5B Q3_K tensors correlated **0.006** with the same tensors from the Q8_0 build of the model (0.988 after the fix). This was invisible to CPU↔Vulkan parity because all backends shared the bug, and invisible to the kernel unit tests because `Q3KFixture` *encoded* with the same wrong layout it decoded with — a closed loop that never touched real GGUF bytes. `DequantizeKQuantTests.Q3_K_DenseRandomBlocks_MatchLlamaCppReference` now pins the CPU oracle to a literal transcription of llama.cpp over dense pseudorandom super-blocks, which is the layer that had been missing.

**Q2_K layout (fixed in #498).** Q2_K carries the *same* element transposition Q3_K did, and it
survived the #311 sweep because nobody re-checked Q2_K at the time — the Q3_K fix comment in
`DequantizeKQuants.cs` already named the wrong form as "the old dotLLM layout". Authority is
llama.cpp's `dequantize_row_q2_K` (`ggml-quants.c`):

- **Element ordering (was wrong).** The 2-bit quants are **not** four consecutive elements per
  byte. Each 128-element half of the super-block consumes 32 `qs` bytes and every byte supplies
  **four elements 32 apart**: element `t` is at byte `32*(t>>7) + (t&31)`, shift `2*((t>>5)&3)`.
  The old `qs[t/4] >> (t%4)*2` form scatters every element.
- **Scale/dmin sub-block (was, and remains, correct).** The 4-bit scale and 4-bit dmin for element
  `t` come from sub-block `t>>4`. That index is **not** transposed — only the `qs` addressing was.

Evidence: `blk.0.attn_q.weight` of `Llama-3.2-1B-pure-Q2_K.gguf` correlated against the same
tensor from the Q8_0 build of the same base gives **0.954** under the llama.cpp layout and
**0.069** under the old one (Q3_K, checked identically as a control, is 0.988 — correct and
untouched). As with Q3_K, the unit tests could not see it: `Q2_K_SingleBlock_HandCalculated`
writes one non-zero `qs` byte and reads elements 0/1/16, the three indices where the two layouts
coincide, and `Q2KFixture` *encoded* with the same wrong order it decoded with. The oracle that
was missing is `DequantizeKQuantTests.Q2_K_DenseRandomBlocks_MatchLlamaCppReference`, a literal
transcription of llama.cpp over dense pseudorandom super-blocks. Vulkan and CUDA carried the same
transposition and were fixed in the same issue. **Any Q2_K quality number from before #498 is
meaningless** — the decoded weights were noise.

See [docs/VULKAN.md](VULKAN.md) for runtime selection details and [docs/CUDA.md](CUDA.md) for the CUDA backend's coverage (Q2_K through Q8_0 plus pre-Q8_1 + MMVQ-large + MMQ + grouped-MoE-GEMV variants).

### MoE indexed-expert matmul (per-row routed dispatch)

Sparse-MoE forward (Mixtral / Qwen-MoE / Qwen3MoeHybrid) needs a kernel shape distinct from dense matmul: a single dispatch reads a per-row expert index and reaches into a packed expert bank `[numExperts, M, K]` for that row's weight matrix. The Vulkan backend ships:

| Bank format | Kernel | Source layout on device | Used by |
|---|---|---|---|
| F32 | `MoeIndexedMatmulF32Kernel` (`moe_indexed_matmul_f32.comp`) | Dequantised at upload time, `[numExperts, M, K]` floats. | Default for all MoE models (streaming uploads per layer per forward; fits any quant since it dequants on the host). |
| Q8_0 | `MoeIndexedMatmulQ8_0F32Kernel` (`moe_indexed_matmul_q8_0_f32.comp`) | Raw Q8_0 blocks `[numExperts, M, (K/32)*34]`. Per-row dequant in shader. | Reserved for the resident-quant MoE path (no model wires it yet — Q8_0 banks fit at F32 too). |
| Q6_K | `MoeIndexedMatmulQ6_KF32Kernel` (`moe_indexed_matmul_q6_k_f32.comp`) | Raw Q6_K super-blocks `[numExperts, M, (K/256)*210]`. Per-row Q6_K dequant in shader (matches `DequantizeQ6_KScalar` byte-for-byte). | Qwen3MoeHybrid (Qwen3.6-A3B) when both `DOTLLM_VK_MOE_RESIDENT=1` AND the source banks are uniformly Q6_K — required for `Qwen3.6-A3B-UD-Q6_K_XL` to fit on Strix Halo's 128 GB unified memory (≈25 GB Q6_K-resident vs ≈120 GB if dequantised to F32). |

The Q6_K MoE kernel completes the original Phase 10 follow-up gap noted in `VulkanQwen3MoeHybridTransformerModel`: with this kernel in place, opting in to `DOTLLM_VK_MOE_RESIDENT=1` on a Q6_K-source Qwen3MoeHybrid model now uploads the routed banks once and keeps them resident across forwards, eliminating the per-forward host→device dequant + upload cost. Mixed-quant layers (e.g. UD checkpoints with Q5_K W2 and Q6_K W1/W3) fall back to the F32 path automatically.

**Resident routed-expert banks (Vulkan), current set (#849):** Q4_K, Q5_K, Q6_K, Q5_1 and Q8_0, resolved per bank (`VulkanQwen3MoeMoeUpload.BankStaysPacked`, shared with the qwen4exp pre-load residency estimate). Any other bank quant still widens to F32. Q5_1 / Q8_0 banks (UD-Q4_K_XL down projections) have the scalar F32-in indexed kernel, an indexed MMVQ (dp4a) decode arm and a grouped coopmat prefill arm (16-row tiles; K % 64). The K-quant fast paths need K % 256, the legacy arms only K % 32 (MMVQ) / K % 64 (grouped), so a 640-wide expert (qwen4exp) takes the fast arms with a Q5_1/Q8_0 down bank. Constraint: the legacy grouped kernel is 16-row only, so a layer with a legacy-quant down bank runs its Q4_K gate/up grouped GEMM on the 16-row kernel instead of the 32-row row-pair form. The fused single-token Qwen3.6 decode layer still requires a Q5_K/Q6_K down bank.

**IQ-family resident banks (#823):** IQ3_S, IQ4_XS, IQ4_NL, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and upstream-ggml Q2_0 also stay packed (K % block: 256 for the super-block types, 32 for IQ4_NL, 64 for Q2_0; the generic-host upload gate is K % 256 for the super-block types and K % 64 for the 32-element legacy blocks IQ4_NL / Q4_0 / Q4_1 - `VulkanWeights.MoeIqKGranule`, #1600: glm4moe's down bank has K = 1408; integer-dot-product required, `DOTLLM_VK_IQ_RESIDENT=0` widens them to F32 again). Decode uses `MoeIndexedMatmulIqMmvqKernel` (`moe_indexed_matmul_iq*_mmvq.comp`, GENERATED from the dense `matmul_iq*_mmvq.comp` by `native/vulkan/gen_moe_indexed_iq.py`; Q2_0 hand-written) for any row count the grouped path does not take; prefill >= 16 tokens uses `MoeGroupedMatmulIqCoopmatKernel` (`moe_grouped_matmul_<fmt>_coopmat_m64.comp`, GENERATED by `native/vulkan/gen_moe_grouped_iq.py`; 16-row tiles, native wave64 + cooperative matrix only, `DOTLLM_VK_MOE_IQ_GROUPED=0` leaves IQ banks on MMVQ). There is NO scalar F32-in indexed kernel for these formats, which is why residency requires dp4a. IQ1_M (ggml id 29) banks are resident too since #993 (`moe_indexed_matmul_iq1_m_mmvq[_v2]` + grouped m64). **V2 indexed MMVQ lane layout** (`*_mmvq_v2.comp`, default; `DOTLLM_VK_MOE_IQ_V2=0` selects V1): a lane owns a whole 32-element sub-block of one of four blocks per step (IQ4_NL: a whole block, 32 consecutive blocks per wave) instead of one 8-element group, so a wave issues ~4-5x fewer load instructions per weight byte; same integer dp4a math, different float summation order. #991 shipped IQ1_M / IQ2_XXS / IQ2_XS (1.6-1.9x per kernel on MiMo); #1678 added IQ3_S / IQ4_XS / IQ4_NL / IQ2_S / IQ3_XXS (hand-written, not generated). Only Q2_0 remains V1. qwen4exp's BF16 gated-residual projections (ISTA files) upload as F16 when exactly representable so they get the cooperative-matrix prefill GEMM instead of the tiled F32 one (`VulkanQwen4ExpGrWeights`, `DOTLLM_VK_Q4E_BF16_AS_F16=0` keeps BF16). Upstream ggml `Q2_0` shares GGUF id 42 with PQ2_0: `GgufReader.ReclassifyUpstreamQ2_0` decides from the on-disk tensor extent and re-labels such tensors with the internal `QuantizationType.Q2_0` (242).

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
| IQ4_NL | 32 | 4.5 | ✓ | dequant-fallback | ✓ (`dequant_iq4_nl_{f16,f32}`) | ✓ (`quantized_gemv_iq4_nl`) | Plus MMQ-preq + MMVQ-large + MoE-grouped; rows4 decode + 2..8-column rows4 (`_mmvq_rows4_preq[_c2..8]`, #1316/#1375, k % 256 == 0). |
| IQ4_XS | 256 | 4.25 | ✓ | dequant-fallback | ✓ | ✓ | Plus MMQ-preq + MMVQ-large + MoE-grouped; rows4 decode + 2..8-column rows4 (#1316/#1375). |
| IQ2_XXS | 256 | 2.0625 | ✓ | dequant-fallback | ✓ (`dequant_iq2_xxs_{f16,f32}`) | ✓ (`quantized_gemv_iq2_xxs`) | 256-entry codebook + 4×7-bit sign indices per 32-elem sub-block + shared 4-bit scale. |
| IQ2_XS | 256 | 2.3125 | ✓ | dequant-fallback | ✓ | ✓ | 512-entry codebook; 7-bit sign indices in upper bits of `qs[uint16]`. |
| IQ2_S | 256 | 2.5625 | ✓ | dequant-fallback | ✓ | ✓ | 1024-entry codebook; high index bits in `qh`. **Also stores `MOSTLY_IQ2_M` file-type tensors** (Qwen3.6-A3B-IQ2_M ~11.5 GB GGUFs). |
| IQ1_S | 256 | ~1.5625 | ✓ | dequant-fallback | — | — | **Vulkan-only on GPU.** 2048-entry codebook of 8 ternary {-1, 0, +1} values packed into each uint64. Per-sub-block 3-bit scale + sign-of-delta in `qh[uint16]`. Smallest GGUF quant. |
| IQ3_XXS | 256 | 3.0625 | ✓ | dequant-fallback | — | — | **Vulkan-only on GPU.** 256-entry codebook (4 unsigned int8 grid points per row); 4-bit per-32-element sub-block scale + 4×7-bit sign indices packed into a uint32 (`scales_and_signs[8]`); signs resolved via the shared 128-entry `ksigns_iq2xs` table. 98 B / super-block. |
| IQ3_S | 256 | 3.4375 | ✓ | dequant-fallback | — | — | **Vulkan-only on GPU.** 512-entry codebook (4 unsigned int8 grid points per row); 9-bit grid indices split low 8 bits in `qs` + high 1 bit in `qh`; 32-byte `signs[]` stores the 8-bit sign mask per pair directly (no ksigns indirection). Paired sub-blocks share a 4-bit scale byte. 110 B / super-block. |

**Dense IQ prefill on Vulkan (#1677):** IQ1_S, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S have native blocked coopmat GEMMs
(`matmul_iq*_gemm_coopmat_128x128x4.comp`, GENERATED by `native/vulkan/gen_iq_gemm_coopmat.py` from the IQ2_XXS wrapper of #1069;
`MatMulIqGemmCoopmatKernel`, one class for the five formats). The 128x128, BK=32, four-wave64-subgroup tile decodes each 32-element
sub-block (codebook grid + signs + sub-scale) straight into the F16 A operand, so the dense host (`VulkanTransformerModel`) and the
Nemotron-H host no longer dequantise these weights into the F16 scratch of `IqF16PrefillMatmul` (#621, PRECISION.md VX4) and no longer
run the scalar F32 GEMM for 5..31 tokens: every `seqLen > 1` not taken by the 2..16-row multi-column GEMVs goes to the native kernel
(K % 256 == 0, native wave64 + cooperative matrix). The staged operand is the same expression, in the same order, as
`iq*_dequant_f16.comp`, so the result equals the F16 scratch path up to F32 accumulation order (unit test bound 1e-4 of max|y|).
`DOTLLM_VULKAN_DISABLE_IQ_COOPMAT=1` restores the F16 scratch path (A/B arm). IQ2_XXS keeps its own kernel (#1069), IQ4_XS its own
(#601); the hybrid hosts (`VulkanQwen3HybridDense*`, `VulkanQwen3MoeHybrid*`) still take the F16 scratch (#1149 routes them through the
dense kernels).

**IQ1_S, IQ3_XXS, and IQ3_S** are supported on CPU and Vulkan (dequant + GEMV + GEMM); the CUDA backend treats them as CPU-only fallbacks. **IQ1_M** remains out of scope. MMQ-preq / MMVQ-large / MoE-grouped variants for the IQ2 / IQ3 families are deferred; CUDA prefill falls back to dequant→cuBLAS via the `dequant_iq2_*_f32` kernels.

The Vulkan IQ3 family kernels store the two codebook tables as readonly SSBOs uploaded once per model load (1 KB `Iq3XxsGrid` + 2 KB `Iq3SGrid` + the 128-byte `ksigns_iq2xs` table shared with the IQ2 family ≈ 3 KB on device, shared by all 6 IQ3 matmul/dequant kernels via `Iq3Codebooks`). Per-element decode is `db * grid[gridIdx*4 + j] * sign_j` with two grid rows per 8-element pair (g1 for elements 0..3, g2 for 4..7). The IQ3_XXS shaders bind the shared `ksigns_iq2xs` SSBO at binding 4 (matching the IQ2_XXS layout); IQ3_S omits this binding since the 8-bit sign mask is stored inline per pair.

## Mach-1 Additive Codec (issue #266, Phases A & B)

`SyzygyResearch/Mach-1-Additive-35B` (a Qwen3.6-35B-A3B checkpoint, `Architecture.Qwen3MoeHybrid`) ships a bespoke ~1.7 bpw "additive" trellis codec — a tail-biting-trellis + randomized-Hadamard-transform (QTIP/QuIP#-family) format, not a GGUF quant type. It has no GGUF representation (no container for trellis + RHT + per-tile gamma) and is decoded via a ported, standalone C# codec rather than the GGUF dequant pipeline described above.

**Phase A (this section) is decoder-only**: `src/DotLLM.Models/Quantization/Mach1/` reproduces the vendor's own `decode.py` (Apache-2.0, ported with attribution) primitive-for-primitive:

| Primitive | Type | Notes |
|---|---|---|
| Tail-biting trellis bit-unpack | `Mach1TrellisCodec.UnpackTileStates`/`UnpackAllTiles` | `L`-bit shift register, `K·V` fresh bits/step, MSB-first big-endian-word packing, final `L-K·V` bits wrap from the stream start. |
| `quantlut_sym` LUT expansion | `Mach1QuantLutSym` / `Mach1LutCache` | Hashes a persisted `[2^tlutBits, V]` table up to the full `[2^L, V]` table via `p=s*(s+1)`; cached by `(L, tlutBits, content hash)`. |
| Orthonormal Walsh-Hadamard transform | `Mach1WalshHadamard` | Sylvester order, single fp32 **division** by `sqrt(dim)` (not a reciprocal multiply) after the butterfly passes — bit-exactness depends on this. |
| Wavefront tile gamma | `Mach1WaveGamma` | 16×16-tile granularity; the wavefront schedule deliberately double-writes the top-right corner tile, later wave wins. |
| int5-g64 (LM head) | `Mach1Int5G64Codec` | 8 signed 5-bit codes packed into 5 little-endian bytes; per-64 fp16 scale; optional exact protected-row overrides. |
| Affine int-bits (embeddings) | `Mach1AffineEmbedCodec` | Asymmetric per-64-group int4 (or int3), MSB-first bit-packed; optional exact bf16-bit-pattern exceptions. |
| Trellis weight reconstruction | `Mach1TrellisWeightDecoder` | Full op order: `unpack → recons → fp16 round-trip → [wave_gamma] → [×wscale] → H_n → ×su → transpose → H_m → ×sv → transpose → crop`. No FMA fusion anywhere in this path — the reference multiplies/adds/divides as separate fp32 ops, and the port matches that exactly. |

Two expert-tier containers exist and are dispatched on the layer file's own metadata, never assumed (`Mach1ExpertContainerKind`/`Mach1ExpertContainer.Detect`): the chunked `trained_susv_wave_gamma_chunked_v1` container (`Mach1ExpertLayerDecoderV3T`, the one shipped in the current repo) and the older per-expert manifest-driven container (`Mach1ExpertLayerDecoderV2`, structurally ported but unverified — no sample of this container currently exists to check against). The NE ("spine": attention / linear-attn / shared-expert) tier uses the same trellis workhorse with int8-sign SU/SV + a scalar Wscale instead of continuous su/sv (`Mach1NeSpineDecoder`).

**Bit-exactness is validated**, not just approximated: `tests/DotLLM.Tests.Integration/Models/Quantization/Mach1/Mach1GoldenBitExactTests.cs` decodes layer 0 / expert 0's gate/up/down projections and compares every element's raw bit pattern against the vendor's own `goldens/L0_e0_fp32.safetensors` — zero mismatches. The fixture resolves via `DOTLLM_MACH1_35B_DIR`, falling back to `TestFixtureResolver` (`~/.dotllm/models/SyzygyResearch/Mach-1-Additive-35B/`, legacy `test-cache`, HF cache — #1279), and the test skips (does not fail) when absent. Per-primitive unit tests in `tests/DotLLM.Tests.Unit/Models/Quantization/Mach1/` use discriminating shapes (non-power-of-two dims that force real padding, `Mb != Nb` grids) and, for the trellis/quantlut_sym/expert-pipeline primitives, cross-check against vectors captured by running the vendor's own `decode.py` — not a re-derivation of this port.

**Phase B (load path)** wires the Phase A decoders into the standard model-loading surface, entirely via existing infrastructure — the `packed/` layout turned out to be readable by the existing generic `SafetensorsFile` reader as-is (composite `|`-delimited keys and per-file `__metadata__` blobs are just ordinary safetensors headers), so no new binary parser was needed:

- **`Mach1PackedCheckpoint`** (`src/DotLLM.Models/Quantization/Mach1/Mach1PackedCheckpoint.cs`) orchestrates the four tiers: opens `packed/experts/codec.json` and dispatches the expert-tier container per-file via `Mach1ExpertContainer.Detect` (never assumes a fixed layout — issue #266 explicitly calls out `container` field churn), lazily opens one experts file + one NE file at a time (closing the previous on layer change), and decodes experts/NE/head/embeddings on demand. An optional `__zsc__` zstd sidecar (mentioned in `decode.py`'s reader as present on some v3 files) is detected and rejected with a clear `NotSupportedException` rather than silently mis-read — no file in the real fixture (as of 2026-08) carries one, and zstd decompression is not wired in.
- **`Mach1ExtrasReader`** reads `extras.safetensors` (~52 MB, bf16) — the small sidecar of tensors the codec does not quantize at all: per-layer RMSNorm gains, GDN scalars (`A_log`, `dt_bias`, `conv1d.weight`, alpha/beta input projections), the MoE router gate, the shared-expert sigmoid gate, and full-attention-layer QK-norm.
- **`Qwen35MoeConfigExtractor`** (`src/DotLLM.Models/SafeTensors/Qwen35MoeConfigExtractor.cs`) parses `qwen3_5_moe` / `qwen3_5_moe_text` `config.json`, hoisting the nested `text_config` and skipping `vision_config` (mirrors `Gemma3nConfigExtractor`'s pattern), and builds `GatedDeltaNetConfig`/`MoeConfig`/`HybridLayerLayout` from the HF key names (`linear_num_key_heads`, `linear_key_head_dim`, `full_attention_interval`, `layer_types`, ...). Wired into `ModelLoader.OpenSafetensorsAndConfig`'s dispatch, alongside the `gemma3n` hoist.
- **`Qwen3MoeHybridTransformerModel.LoadFromMach1Packed`** decodes every tensor to dense fp32 `NativeMemory`-owned buffers and populates the *exact same* `Qwen3MoeLayerWeights` / `GdnTokenMixingWeights` / `Qwen3FullAttnWeights` / `MoeLayerWeights` shapes `LoadFromGguf` builds for this architecture — **zero forward-pass changes**. HF tensor names (verified against the real fixture, not guessed): GDN layers carry `linear_attn.{in_proj_qkv,in_proj_z,out_proj}.weight`; full-attention layers carry `self_attn.{q,k,v,o}_proj.weight` with `q_proj` fused Q+Gate at `2*numAttentionHeads*headDim` output width (same convention the GGUF path's `attn_q.weight` already assumes); both carry `mlp.shared_expert.{gate,up,down}_proj.weight` and route through `mlp.gate.weight` (router). The GDN `conv1d.weight`'s HF shape `[channels, 1, dConv]` (PyTorch depthwise `Conv1d`) flattens row-major to the exact same `c*dConv+k` byte layout `Conv1dCausal.Execute` already expects for GGUF's `[dConv, channels]` — no transpose needed, just a dtype upcast.

**Resident-weights design matches the GGUF path exactly — this is not a loader inefficiency.** `LoadFromMach1Packed` opens and decodes `packed/experts/L{LL}.safetensors` / `packed/ne/L{LL}.safetensors` layer files sequentially (one pair open at a time, closed on layer change) purely to bound *decode-time* scratch, but the resulting per-layer weight bundles are all kept resident in the `layers` array for the model's lifetime — identical to `LoadFromGguf`, which also builds one `layers` array once and keeps every layer resident for as long as the model is alive. The only difference is *where* the bytes live: GGUF's bundle is zero-copy pointers into an mmap'd, OS-page-cached file (so "resident" is cheap — pages fault in on first touch and are shareable across processes), while Mach-1's bundle is genuinely-owned `NativeMemory`-allocated dense fp32 (decoding is mandatory here — the trellis + RHT codec has no "raw bytes, dequant this row on demand" shortcut the way GGUF's K-quants do, so there is no cheaper representation to memory-map). Both loaders therefore hold **100% of layers resident simultaneously** by design; Mach-1 is more expensive only because F32-dense is a larger on-disk-equivalent footprint than any GGUF quant, not because the loading strategy differs.

**Memory is consequently the load-bearing constraint on full-model validation, not correctness.** Decoding all 256 experts × 40 layers × 3 projections to F32 is ≈128 GB resident, and even bf16 would be ≈64 GB — both a known, hardware-driven limitation (not a bug, not something this loader should paper over): the issue itself frames dense-decode as "a validation vehicle, not a deliverable" needing on this order of RAM. Full 40-layer end-to-end generation and top-1 token agreement vs. the base GGUF therefore require a machine with substantially more RAM than a typical workstation (the *loading path* itself has no such requirement below full-model scale — see the validated-at-real-scale coverage below); running that comparison is not attempted in this repo state and needs a bigger-RAM machine to execute. Full-scale resident-memory avoidance without needing that RAM is exactly what Phase C's fused-kernel path solves (computing the GEMV directly in the rotated/trellis domain, so routed-expert weights never leave their ~6.2 GB packed-byte form) — it is not something Phase B's loader is expected to solve, and no attempt was made to redesign it into a streaming/lazy-per-layer architecture to work around the RAM ceiling (that redesign is Phase C's job).

**Phase B validation coverage, all against the real fixture (`DOTLLM_MACH1_35B_DIR`, skip-if-absent), no synthetic substitutes:**

| Tier | Test | What it proves |
|---|---|---|
| Config extraction | `Mach1PackedCheckpointLoaderTests.ConfigExtractor_RealConfigJson_MatchesExpectedShape` | Nested `text_config` hoist + `vision_config` skip against the real `config.json`; GDN/MoE/RoPE/HybridLayout fields match the known real values. |
| NE spine (GDN) | `DecodeNeTensor_GdnLayer0_ProducesFiniteDeterministicValues` | `linear_attn.{in_proj_qkv,in_proj_z,out_proj}` + shared-expert decode to finite, deterministic (decode-twice, bit-identical) tensors. |
| NE spine (full-attn) | `DecodeNeTensor_FullAttnLayer3_ProducesFiniteDeterministicValues` | `self_attn.{q,k,v,o}_proj` (Q+Gate fused width confirmed against the real file's own `dims` metadata, not assumed) decode the same way. |
| Extras (norms/scalars) | `ExtrasReader_RealExtrasSafetensors_ResolvesLayerAndModelLevelNorms` | GDN-vs-full-attention key presence (`linear_attn.*` XOR `self_attn.q_norm`/`k_norm`) matches `layer_types`; bf16 upcast is finite. |
| Routed experts | `DecodeExpertProjection_Layer0Expert0_MatchesVendorGolden_ThroughOrchestrationLayer` | `Mach1PackedCheckpoint`'s orchestration (not just the bare Phase A primitive) reproduces the vendor's `goldens/L0_e0_fp32.safetensors` bit-exactly. |
| LM head (`int5g64_packed`) | `DecodeHead_RealFixture_FiniteDeterministic_NoChunkBoundaryGaps` | All 8 `head_c{0..7}of8.safetensors` chunks assemble into a finite, deterministic `[248320, 2048]` table with no all-zero/stale row at any of the 8 chunk-row boundaries (31,040 rows/chunk, matching each chunk's own declared dims). |
| Embeddings (affine int4 + exceptions) | `DecodeEmbedding_RealFixture_FiniteDeterministic_AndAppliesExceptionOverrides` | Full `[248320, 2048]` table decodes finite/deterministic, AND a sample of the 246,714 exact-overwrite exceptions is cross-checked against the raw `exc_idx`/`exc_bits` tensors via the same bf16-bit-pattern formula the decoder uses — proves the override path actually fires, not just the group-quant fallback. |
| Full per-layer weight mapping (GDN) | `Mach1SingleLayerLoadTests.LoadLayerFromMach1_GdnLayer0_BuildsCorrectlyShapedWeights` | The complete `LoadLayerFromMach1`/`LoadGdnLayerFromMach1`/`LoadMoeLayerFromMach1` wiring (not the lower-level tier decoders in isolation) builds a correctly-shaped `Qwen3MoeLayerWeights` bundle for a GDN layer, with `W1[0]` (expert 0 gate) cross-checked bit-exact against the vendor golden. ~9.5 min (256-expert single-threaded CPU decode). |
| Full per-layer weight mapping (full-attn) | `LoadLayerFromMach1_FullAttnLayer3_BuildsCorrectlyShapedWeights` | Same, for a full-attention layer — confirms `LoadFullAttnLayerFromMach1`'s shape wiring. |
| Chat template | `Mach1ChatTemplateTests` (3 tests) | The real `chat_template.jinja`'s `{% macro render_content %}` (macro support landed separately, issue #273) renders correctly: simple turns, system+history ordering, tool-call round-trip. |

**Not yet run: the full public `LoadFromMach1Packed` entry point end-to-end** (all 40 layers + embed + head in one call) and consequently **full-model generation / top-1 token agreement vs. `DOTLLM_QWEN36_A3B_Q6_K_XL_GGUF` / perplexity delta**. Every tier and both layer kinds are independently validated above (including bit-exactness against the vendor golden where a golden exists); what remains untested is purely the "all 40 layers simultaneously resident" scale, which is gated on the ~70-128 GB RAM this section describes, not on any known defect.

### Mach-1 fused additive expert GEMV (issue #266, Phase C — CPU, partial)

**Status: CPU-only, correctness-validated on real weights, not yet wired into the forward pass; CUDA/Vulkan not started.** This is the actual point of the codec: computing `y = W·x` directly from the packed trellis stream so the routed-expert weights (6.21 of 7.53 GB) never leave their packed-byte form, instead of decoding to dense fp32 first (which is what Phase A/B's `DecodeExpertProjection` does, and what makes full-model dense-decode need ~70-128 GB — see above).

**Derivation.** `Mach1TrellisWeightDecoder.Decode` builds the dense weight as `W = diag(sv) · H_m · [wave_gamma ⊙ Wunit] · H_n · diag(su)` (crop omitted). For a GEMV `y = W·x`, both orthonormal (hence symmetric) Hadamard transforms and `diag(su)` move onto the activation side by associativity instead of being baked into a materialized weight matrix:

```
y = diag(sv) · H_m · [wave_gamma ⊙ Wunit] · x'      where x' = H_n(su ⊙ x)
```

`Mach1FusedExpertGemv.Compute` (`src/DotLLM.Models/Quantization/Mach1/Mach1FusedExpertGemv.cs`) implements exactly this: (1) pad + scale the activation by `su` and Hadamard-transform it — `O(n log n)`, once per call; (2) unpack each 16×16 tile's trellis states and immediately multiply-accumulate its lattice values against `x'` — the trellis is never written to a dense `[m,n]` buffer, matching `codec.json`'s own documented `kernel_contract` (per-weight ops are add/subtract-dominated on the exact integer lattice, `wave_gamma` applied at tile granularity); (3) Hadamard-transform the accumulated `[m]`-length result and scale by `sv`, crop to `m0`. `Mach1ExpertLayerDecoderV3T.GemvExpertProjection` / `Mach1PackedCheckpoint.GemvExpertProjection` expose it with the identical key-lookup/su/sv/gamma extraction as `DecodeExpertProjection`, so the two paths are directly A/B-comparable.

**Correctness**, both synthetic (fixture-free, `Mach1FusedExpertGemvTests`, non-power-of-two/non-square/`Mb != Nb` shapes with production `cb_params`) and against real fixture weights (`Mach1FusedExpertGemvRealFixtureTests.GemvExpertProjection_Expert0AllProjections_MatchesDenseDecodeThenMatVec`, layer 0 / expert 0, all three projections), comparing the fused path against dense-decode-then-matvec (`DecodeExpertProjection` + a plain row-dot-product loop) as reference:

| Case | max abs error | max rel error |
|---|---|---|
| Synthetic, m0=16 n0=16 (single tile) | 3.6e-6 | 5.1e-5 |
| Synthetic, m0=48 n0=20 | 3.8e-6 | 6.1e-6 |
| Synthetic, m0=20 n0=48, with gamma | 8.1e-6 | 2.2e-6 |
| Synthetic, m0=20 n0=48, no gamma | 5.7e-6 | 1.5e-5 |
| Real fixture, expert 0 gate [512×2048] | 1.1e-6 | 2.3e-3 |
| Real fixture, expert 0 up [512×2048] | 7.0e-7 | 4.0e-4 |
| Real fixture, expert 0 down [2048×512] | 2.8e-7 | 1.5e-4 |

All errors are consistent with ordinary fp32 reassociation noise (the two paths sum the same terms in different orders — direct row-dot-product vs. Hadamard-butterfly reformulation), not a structural bug; a real transpose/index/gamma error produces errors many orders of magnitude larger, which is what these tests are actually built to catch (see the test docstrings for why the shapes are chosen).

**Performance** (`Mach1FusedExpertGemvRealFixtureTests.Perf_InterleavedAB_DenseDecodeVsFusedGemv_Expert0Gate`, interleaved A/B, 10 trials, real layer-0/expert-0 gate weights, this project's `MtpBenchProfile`-style methodology): dense-decode-then-matvec median **107.3 ms/call**, fused GEMV median **35.2 ms/call** — **~3.0x faster**, single-threaded, scalar (no SIMD in either path). This is not primarily a memory-vs-compute tradeoff: the fused path is asymptotically cheaper because it Hadamard-transforms one `n`-length vector and one `m`-length vector (`O(n log n + m log m)`) instead of Hadamard-transforming *every row* of the dense `[m,n]` matrix twice, once per axis (`O(m·n·log n + n·m·log m)`), plus it skips the dense path's two full-matrix transpose passes.

**Resident memory** (`Mach1FusedExpertGemvRealFixtureTests.Memory_DenseDecodeRetained_vs_FusedDiscarded_NExperts`, real layer-0 weights, 32 experts' gate projection = one full chunk, managed-heap growth via `GC.GetTotalMemory` bracketing each phase): decoding and retaining 32 dense `[512,2048]` fp32 arrays grew the heap by **138.9 MB** (theoretical 128.0 MB, difference is array-object/GC overhead); computing the fused GEMV for the same 32 experts and discarding each result grew it by **1.0 MB**. This is a micro-scale, single-projection/single-layer measurement (RAM-budget-appropriate for this session, not a full-model run); the model-wide claim — 256 experts × 3 projections × 40 layers dense fp32 (~64 GB) vs. 6.21 GB packed-resident — is issue #266's own figure, not independently re-measured here, but this test's per-expert delta (≈4.3 MB retained per dense `[512,2048]` projection vs. ≈0 retained for fused) is directly consistent with it by simple multiplication.

**What Phase C still needs to close issue #266:**

- **CUDA and Vulkan implementations**, each validated against this same dense-decode reference (issue #266 explicitly requires all three backends; this session is CPU-only per its own scope, gated on the CPU measurement above per the issue's phasing).
- **SIMD vectorization of the CPU inner loop.** `Mach1FusedExpertGemv.Compute` is scalar; the accumulate loop (`temp[row] += w * xPrime[col]`) and the trellis bit-unpack are both unvectorized. The ~3x win above is from the algorithmic reformulation alone.
- **The `kernel_contract`'s documented perf reordering**: accumulate each tile's raw lattice-value·`x'` partial sum first (add/subtract-dominated, since `Wunit` is an exact integer lattice), then apply the tile's single `wave_gamma` multiply to the partial sum ("~1 multiply per 16 weights") instead of per-element — algebraically identical, not yet implemented.
- **Wiring into the actual MoE forward pass** (`Qwen3MoeHybridTransformerModel`'s FFN dispatch) — this pass validates the kernel in isolation against real weights, not end-to-end token generation through the fused path. The forward pass currently uses `LoadFromMach1Packed`'s dense-decode-then-cache route only.
- **The other three tiers' stepping-stone completion** (NE spine 0.70 GB, LM head 0.33 GB, embeddings 0.29 GB) — issue #266's permitted stepping-stone decodes these to dense/Q8_0 at load while only the routed-expert GEMV fuses; that decode-at-load path already exists from Phase B, it just hasn't been threaded into a real "packed-resident" load mode (currently everything, including these three tiers, decodes to dense fp32 the same way, so the ~7.5 GB headline number is not yet achievable end-to-end).
- **Model-scale (not per-expert) resident-memory and perplexity/top-1 measurement**, gated on the ~70-128 GB RAM issue #266 already documents as needed for a full dense-decode baseline run, or on the forward-pass wiring above making a dense baseline unnecessary.

## LittleBit factorized linears (issue #864, CPU only)

`QuantizationType.LittleBit` (id 100, not a GGUF type) is a *factorized-linear* weight kind: a projection stored as sign
factors plus FP scales, `W ~= sum_paths diag(u1) . Us . diag(l) . Vs . diag(v2)` with `l = bf16(v1*u2)`, `Us in {+-1}^(out x r)`,
`Vs in {+-1}^(r x in)`, primary path plus an optional residual path (`*_R`). Implemented clean-room from the paper equations and our own
trainer (`tools/littlebit-qat`); no third-party code.

- **Checkpoint**: a safetensors directory with `config.json` (Qwen3) and `littlebit_config.json`. Per projection the keys are
  `{prefix}U_packed, V_packed, U_shape, V_shape, u1, u2, v1, v2` and `U_R_packed ... v2_R`. Packed words are int32, LSB-first,
  bit 1 = -1, padded with +1; `U` is `[out, ceil(r/32)]` and `V` is `[r, ceil(in/32)]`, which is byte-identical to the kernel layout, so
  loading is a row copy (`LittleBitLoader`). Embeddings, norms and the (tied) head stay bf16 and are upcast to F32 by the normal loader.
- **Where it plugs in**: `SafetensorsTensorResolver.ResolveLinear` returns `(token, QuantizationType.LittleBit, out, in)` when `{prefix}weight`
  is absent but `{prefix}U_packed` exists; `TransformerModel.Gemm` dispatches on the type. Everything else (QK-norm, RoPE, GQA attention,
  SwiGLU, KV cache, `generate`, perplexity) is the stock dense-Qwen3 path. Fused decode kernels are skipped (`SupportsFusedDecode` is false).
- **Kernels** (`DotLLM.Cpu.Kernels.Experimental.LittleBit`): two-stage signed-sum GEMV (stage 1 `t = l .* (Vs (v2 .* x))`, stage 2
  `y += u1 .* (Us t)`), AVX2 float path. Decode is row-parallel over the pool (two dispatches per projection); prefill parallelises over
  tokens, each thread running the single-thread GEMV with its own scratch (bit-identical per token). Scalar double reference in
  `LittleBitReference`. The VNNI int8 variant exists but is not used by the model (it adds activation-quantisation error).
- **Rejections** (explicit `NotSupportedException`, never a fall-through): the shared weight loader used by Vulkan, CUDA and HIP refuses a
  factorized checkpoint ("supported on the CPU backend only"); the CPU path refuses any architecture other than dense Qwen (`Architecture.Qwen`,
  no MoE) and layers with a bias; AVX2 is required. `dotllm perplexity` accepts a safetensors directory on `--device cpu` only
  (with `--tokens-file`).
- **Memory** (Qwen3-0.6B, eff_bit 0.55, 196 projections, residual on): sign bits 26.5 MiB + F32 scales 5.1 MiB = **31.6 MiB** against 840 MiB for the same
  projections in bf16 (26.6x smaller; 0.60 stored bits per weight, 0.55 with bf16 scales). Scales are held as F32 (exact widening of the bf16 source,
  and not clamped to fp16 range): that costs 2.5 MiB more than bf16 scales would. Padding to x32 is included. The paper's bit formula counts 2 bits
  per sign; this layout stores 1.
- **Precision**: scales are bf16 in the checkpoint, widened exactly; `l` is rounded once to bf16 like the reference forward. The kernel
  accumulates in F32. Layer level (step 1): max relative error 1.7e-7 vs a float64 reference, 5.8e-3 vs the trainer's all-bf16 forward.
- **Whole-model validation** (Qwen3-0.6B-055, `LittleBitModelTests`, fixtures from `tools/littlebit-qat/gen_dotllm_reference.py`): against torch fp32,
  120/120 teacher-forced top-1 and byte-identical 40-token greedy continuations on 3 prompts, max |logit diff| 0.044 over torch's top-8. Against the
  torch **bf16** forward only 115/120 top-1 (the bf16 forward itself agrees with torch fp32 on 115/120: it flips near-tied argmaxes on this
  PPL~400 model, so bf16 is the wrong oracle for an fp32 engine). F32-decoded control (`DOTLLM_LITTLEBIT_DENSE_CONTROL=1` decodes every
  projection to dense F32 and runs the stock GEMM): identical metrics, last-position logits differ from the factorized kernel by 1e-5.
- **Perplexity**: WikiText-2 test, first 40 non-overlapping 2048-token windows, scored as the trainer's `eval_ppl.py` does (targets 1..2047 of each window):
  trainer 401.02, `dotllm perplexity <dir> --tokens-file wikitext2_first40.tokens --context 2048 --stride 2048 --unscored-prefix 1` gives **400.89**
  (ratio 0.9997). The teacher (bf16 Qwen3-0.6B) is 19.6; this checkpoint is a 1500-step spike run, so the model is poor even though the engine matches.
- Fixtures (`~/.dotllm/test-cache/littlebit-qwen3-0.6b-055`, or `DOTLLM_LITTLEBIT_CKPT`); tests skip cleanly when absent.
- **Decode speed** (`dotllm run`, 64 tokens, CPU, same session, runs interleaved dense/LittleBit): LittleBit 19-30 tok/s vs 10-14 tok/s for the dense
  Qwen3-0.6B baseline of the same model, per-pair ratios 1.4x-2.9x (median 2.3x). The "dense bf16" baseline is the safetensors loader's bf16->F32
  upcast (there is no resident-bf16 GEMV), so it streams 4 B/weight; the box was at 100% CPU from other processes, so judge the ratio, not the absolutes.
  Both runs include the F32-upcast tied lm_head (594 MiB/token), which is now a large share of the LittleBit decode traffic. Prefill (perplexity run): ~213 tok/s.
