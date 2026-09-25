# MoE prefill MMQ feasibility notes (research, not a test)

Date: 2026-07-18. Scope: is a dp4a/MMQ-class indexed GEMM feasible for MoE
**prefill** (seqLen > 1), closing the gap noted in `.docs/KERNEL_MAP.md` §7
item 2 and §11 (`mul_mat_id` comparison)? No source files were edited for
this investigation.

## 1. Inventory of MoE indexed-matmul kernels

`src/DotLLM.Vulkan/Kernels/MoeIndexedMatmul*.cs` (10 files) + their
`.comp` shaders (`native/vulkan/shaders/moe_indexed_matmul_*.comp`):

| Kernel | Shader | Math | Tiling |
|---|---|---|---|
| `MoeIndexedMatmulF32Kernel` | `moe_indexed_matmul_f32.comp` | F32, per-output-cell scalar dot | none (1 thread/output) |
| `MoeIndexedMatmulTiledF32Kernel` | `moe_indexed_matmul_tiled_f32.comp` | F32, dequant on the fly | 1 WG per **token row** `n`, TILE_M=16 output cols share an x-row in LDS |
| `MoeIndexedMatmulQ4_KF32Kernel` / `Q5_KF32` / `Q6_KF32` / `Q5_1F32` / `Q8_0F32Kernel` | `moe_indexed_matmul_{q4_k,q5_k,q6_k,q5_1,q8_0}_f32.comp` | **dequant weight block → F32, then F32 multiply-accumulate** (e.g. Q8_0 shader: `d = readHalf(...); acc += d * blockSum` at `moe_indexed_matmul_q8_0_f32.comp:61-67`) | 16×16 threads, 1 output cell/thread, no dp4a |
| `MoeIndexedMatmulQ4KMmvqKernel` / `Q5_1MmvqKernel` / `Q8_0MmvqKernel` | `moe_indexed_matmul_{q4_k,q5_1,q8_0}_mmvq.comp` | **int8 dp4a** (`dotPacked4x8AccSatEXT`), Q8_1-quantized activation | **decode-only**: one wave32 subgroup per `(n, m)` output cell (`moe_indexed_matmul_q8_0_mmvq.comp:56-104`), no cross-token weight-tile reuse |

Confirmed: every "F32" kernel dequantizes the weight block to float before
the multiply — the naming is accurate, these are not disguised int8 paths.
The MMVQ kernels are the only dp4a-based indexed kernels, and per their own
header comments (`moe_indexed_matmul_q8_0_mmvq.comp:5,26-27`) they are a
GEMV pattern: one subgroup reduces one `(token, weight-row)` cell — no
64-token/64-row shared-memory tile like the dense `_mmq` kernels use.

## 2. `RecordMoeLayer` / `RecordGemma4Ffn` dispatch — confirmed F32 for prefill

- `RecordMoeLayer` (`src/DotLLM.Vulkan/VulkanTransformerModel.cs:4550-4655`) always
  calls `RecordMoeIndexedMatmul` (`:4890-4936`) for W1/W3/W2. That helper
  dispatches by **weight quant type only** — Q8_0 → `_moeIndexedMatmulQ8`
  (the **F32-dequant** kernel, not the MMVQ one), Q4_K → `_moeIndexedMatmulQ4K`
  (also F32-dequant), else F32 bank → tiled/scalar F32. **`seqLen` never
  enters this decision** — every prefill call, and every decode call that
  goes through the generic `RecordMoeLayer` path, uses the dequant kernel.
- The dp4a MMVQ kernels are only reachable through `RecordGemma4Ffn`
  (`:4202-4341`), gated explicitly:
  ```
  bool useMoeMmvq = seqLen == 1 && ... // :4259
  ```
  with the comment at `:4257-4258`: *"Multi-token forwards (prefill /
  diffusion) keep the scalar kernels, so every existing seqLen>1 parity
  path stays byte-identical."* This is a direct, first-party confirmation
  that prefill is intentionally excluded from the quantized dp4a path today
  — not an oversight, a deferred scope boundary.
- `.docs/KERNEL_MAP.md` §7 independently documents the same gap ("MoE
  prefill expert matmuls are scalar/tiled F32-in indexed kernels (dp4a MMVQ
  is decode-only; no indexed MMQ)") and flags it as colocated with an open
  DEVICE_LOST fault, `#373`, at `expandedRows > 16` — worth investigating
  together since both live in the same prefill dispatch path.

**(a) Answer:** confirmed. `src/DotLLM.Vulkan/VulkanTransformerModel.cs:4897-4915`
(dispatch by quant type, no seqLen branch) and `:4259` (Gemma-4's explicit
`seqLen == 1` MMVQ gate) both show prefill unconditionally uses the F32
dequant path, even when the bank is stored Q8_0/Q4_K and a same-quant dense
MMQ kernel (`matmul_q8_0_mmq.comp`) already exists.

## 3. Dense MMQ vs indexed MMVQ/F32 side-by-side — the crux question

Read `matmul_q8_0_mmq.comp` (dense prefill GEMM, issue #366 tiling) against
`moe_indexed_matmul_tiled_f32.comp` and `moe_indexed_matmul_q8_0_mmvq.comp`.

**Dense MMQ's whole performance story is shared-tile reuse.** From
`matmul_q8_0_mmq.comp:56-99`: one workgroup owns a fixed `TILE_M=64` weight
rows × `TILE_N=64` token rows. Per K-block it stages **one shared copy** of
64 weight rows (`sharedWq`) and 64 token rows (`sharedXq`) into LDS
(`:101-161`), then every one of the 256 threads in the WG reads from that
*same* shared tile to compute a 4×4 register sub-tile of dp4a products
(`:166-189`). The entire 8×-per-load amortization the header comment brags
about (`:33-39`, "8 dp4a per LDS load") depends on **all 64 token columns in
the WG consuming the same 64 weight rows** — i.e. weight row `m` is a
function of `m` alone, shared across every token in the tile. That is true
for a dense matmul (one weight matrix for all tokens) and **false for MoE**:
each token row `n` in an indexed matmul selects its own `expertIdx =
indices[n]`, so 64 consecutive token rows in a naive tile can, and in the
worst case (topK routing over `numExperts` >> 64) generally will, address
64 *different* expert weight matrices. `moe_indexed_matmul_tiled_f32.comp`
already documents exactly this obstruction in its header, unprompted, at
`:13-21`:

> *"Why 'one workgroup per output row'? The expert index varies per output
> row (idx = indices[n]) — that means the 'weight A tile' cannot be shared
> across rows of x. The standard GEMM tile pattern (TILE_M × TILE_N output
> cells per WG, shared A and B tiles) breaks because each row of B would
> need a different slab of A."*

So the tiled F32 kernel already made the same design call MMQ would have to
make: pin the WG to a single `n` (or a single expert), sacrificing the
N-axis tile-sharing that gives MMQ its LDS-reuse multiplier, and only reuse
the K-axis tile.

**(b) Answer to the core feasibility question:** yes, per-token expert
indexing genuinely conflicts with MMQ's N-tile-sharing design **as written
today** — but the conflict is resolvable, not fatal, via **grouping tokens
by expert before the matmul**, exactly the pattern llama.cpp's
`mul_mat_id` `_mmqid` warptiles + `count_experts.comp` pre-pass use
(`.docs/KERNEL_MAP.md` §11, `ggml-vulkan.cpp:8834-8846,8685-8709`). dotLLM
already has this exact machinery built — just for a different quant:
`MoeExpertOffsetsKernel` (count/prefix over `indices[]`) +
`MoeExpandGroupByExpertF32Kernel` (permute expanded rows into
per-expert-contiguous blocks) + `MoeGroupedMatmulF16CoopmatKernel`
(`moe_grouped_matmul_f16_coopmat.comp`), wired in
`RecordMoeGroupedF16Layer` (`VulkanTransformerModel.cs:4670-4729`) and
gated by `CanUseGroupedF16Moe` (`:4657-4668`) — but that gate is hard-coded
to `QuantType.F16` banks only. **No int8/dp4a grouped variant exists.**
Once rows are grouped-by-expert (which the existing kernels already do),
each expert's contiguous row-block *is* a dense sub-GEMM over a single
weight matrix — the dense `matmul_q8_0_mmq.comp` tiling applies almost
unchanged inside that sub-block. The remaining wrinkle is variable-length,
data-dependent sub-block sizes (an expert's row count depends on the
routing at runtime, not compile time), which needs either (i) per-expert
indirect/`vkCmdDispatchIndirect` dispatch sized from the device-computed
offsets, or (ii) a fixed over-provisioned tile grid with an early-exit
bounds check per WG (simpler, some wasted WGs on ragged tails) — llama.cpp
uses the indirect-dispatch-free "worst case padding + early exit" version
in its own `_mmqid` variant per the KERNEL_MAP note.

## 4. What a "moe MMQ" kernel needs, relative to dense MMQ

1. **A grouping/sort pre-pass** — already exists (`MoeExpertOffsetsKernel`
   + `MoeExpandGroupByExpertF32Kernel`), needs no new shader, only wiring.
2. **Per-expert weight-row base lookup** — trivial, same
   `expertIdx * matrixByteStride` arithmetic already used in every
   `moe_indexed_matmul_*.comp` shader (e.g. `moe_indexed_matmul_q8_0_f32.comp:50-55`).
   Once grouped, this is looked up once per WG (all rows in the WG share
   one expert) instead of once per token.
3. **Q8_1 row-quantization of the (already-grouped) activation** — the
   `quantize_q8_1_rows.comp` step Gemma-4's MMVQ path already runs
   (`VulkanTransformerModel.cs:4276`) is reusable verbatim; only its input
   ordering changes (grouped-permuted rows instead of raw expanded rows).
4. **Ragged-tail handling** — dense MMQ's `TILE_N=64`/`TILE_M=64` assumes a
   full tile is always populated by real data (out-of-range rows/cols get
   masked to 0 in dense GEMM because `N`/`M` are the true global bounds).
   In grouped MoE, per-expert row counts are irregular, so tiles at the end
   of each expert's block will be partially populated by the *next*
   expert's rows unless the pre-pass pads/aligns expert boundaries to
   `TILE_N`. This is the one genuinely new correctness hazard (not present
   in the F16 grouped kernel's simpler coopmat path, but same-shape
   concern) and needs explicit design/tests before landing.
5. Everything else — the dp4a inner loop, LDS staging shape, `dw`/`dx`
   scale application — is a mechanical copy from `matmul_q8_0_mmq.comp`.

## 5. Real-model shapes (paths only, not loaded)

- `~/.cache/huggingface/hub/models--unsloth--gemma-4-26B-A4B-it-GGUF` — the
  Gemma-4-26B-A4B MoE model already exercised by `RecordGemma4Ffn`'s
  decode-side MMVQ (#137); per `.docs/KERNEL_MAP.md` §7, decode reached
  30.5 tok/s with indexed MMVQ vs the F32 baseline.
- `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-GGUF` (and
  `-MTP-GGUF`, `mudler--Qwen3.6-35B-A3B-APEX-GGUF`) — larger MoE model,
  relevant to the `Qwen3MoeHybrid` per-bank-resident work (#371/#372).
- KERNEL_MAP does not list per-model hidden/intermediate/expert-count
  dimensions in §7 beyond the tok/s figures above; pulling them would
  require parsing GGUF metadata, out of scope per the task instructions
  (research-note only, no model loading).

## 6. Priority recommendation

**Worth pursuing, but scope it as a grouped-GEMM port, not a bespoke
indexed-MMQ shader.** The naive reading of the task ("write
moe_indexed_matmul_q8_0_mmq.comp with per-token indexing inside the MMQ
tile") is a dead end — §3 shows why token-level indexing inside a 64-wide
N-tile defeats MMQ's entire reuse argument. The viable path is to extend
the **already-built** grouped-matmul infrastructure
(`MoeExpertOffsetsKernel` / `MoeExpandGroupByExpertF32Kernel` /
`RecordMoeGroupedF16Layer`) from its current F16-coopmat-only gate to also
cover Q8_0/Q4_K banks via a new `moe_grouped_matmul_q8_0_mmq.comp` (and
Q4_K sibling) that is a near-verbatim adaptation of `matmul_q8_0_mmq.comp`
reading from the grouped/permuted buffers instead of the raw activation —
this reuses ~80% of both the grouping kernels and the MMQ tiling code, so
scope is "large but mostly mechanical port + one new correctness concern
(ragged tail padding)," not a from-scratch design. Given #137's decode-side
precedent (dequant → dp4a indexed MMVQ was a 20-30x-class win on MoE
decode, `.docs/KERNEL_MAP.md` §7), and that prefill on MoE models today
falls all the way back to scalar/lightly-tiled F32 (worse than dense
prefill's already-known ~0.09x gap per KERNEL_MAP line 29), this is
plausibly the single largest remaining MoE performance lever in the
codebase. Two flags before committing engineering time:
(i) coordinate with `#373` (DEVICE_LOST at `expandedRows > 16` in the same
prefill dispatch path) since a new kernel there should not be built on top
of an unexplained fault; (ii) start with Gemma-4/Qwen3-MoE Q4_K banks (the
grouped F16 infra's sibling quant already has an MMVQ decode kernel
(`MoeIndexedMatmulQ4KMmvqKernel`) to validate against for correctness
before generalizing to Q8_0/Q5_1/Q5_K/Q6_K.
