# Numeric Precision per Operation and Device

> **Status (#1109, 2026-10-23).** CPU, CUDA and Vulkan arms are measured; their evidence is condensed in §5 (tables
> E1–E12). Raw results: `.docs/precision/{cpu,cuda,vulkan}/RESULTS.md`, `.docs/precision/SURVEY.md` and
> `.docs/precision/raw/` (git-ignored, local). Follow-up issues are listed in §7.

## 1. Rules and how to read this document

The hard rules live in [`CLAUDE.md` → Precision & Expansion Rules](../CLAUDE.md#precision--expansion-rules-user-decisions-2026-10-22-1107)
(#1107). §6 maps each rule to the evidence here.

1. **No FP32 tensor storage on hot paths.** FP16/BF16 is the widest stored default; evaluate formats below FP16 too.
2. **Quantized weights and KV run quantized.** Dequantize inside the tile load, never into a wider stored copy.
3. **Disk ↔ device transfers go direct.** (Outside this study's measurements.)

Terms:

- **Storage** is the dtype written to memory; **compute** is the ALU/tensor-core operand dtype; **acc** is the
  accumulator.
- `W4A8` = 4-bit weights × 8-bit activations. `Q8_0`/`Q8_1` = int8 with one scale per 32 (Q8_1 adds the block sum);
  `Q8_K` = one scale per 256.
- **Accuracy bar:** KL(ref ‖ test) over the full vocabulary, gated on the **median** (or trimmed mean) plus the outlier
  rate (rows with KL > 1e-3 / 1e-2) and top-1 agreement. Mean KL is reported but not gated: massive-activation rows
  dominate it.
- Evidence is cited as **E1–E14** (§5). `[lit]` marks a literature-only claim; `not measured (#n)` names the issue that
  will measure it.

---

## 2. Device capability matrix

Peak rates are per clock relative to the device's own FP32 FMA, from vendor whitepapers, ISA guides, uops.info and
local `vulkaninfo`/CPUID probes (2026-10-22). **Measured** rows come from E1–E12 and override the peak where they
disagree.

| | RTX 3060 (GA106, CC 8.6) | Radeon 8060S (gfx1151) | Arc iGPU (Xe-LPG, MTL) | Core Ultra 7 155H | Ryzen AI Max+ 395 (Zen 5) | Xeon X5670 (Westmere) |
|---|---|---|---|---|---|---|
| FP32 | 1× (12.7 TF) | 1× (14.8 TF; 2× with VOPD) | 1× (4.6 TF) | 1× (16 FMA/clk/P-core) | 1× (32 FMA/clk, full 512-bit) | 1× (no FMA) |
| FP16 | mma 4× (F16 acc) / **2× (F32 acc, GeForce half rate)**; HFMA2 1–2× (not measured) | WMMA ≈4×; `v_pk_fma_f16` 2× | packed 2× | storage only (F16C) | storage only (no AVX512-FP16) | **no F16C**: software convert |
| BF16 | mma 2× (F32 acc only) | WMMA ≈4× in ISA; **no bf16 coopmat** in either driver | none | shift-convert | `VDPBF16PS` 2× (preview runtime only) | shift-convert |
| FP8 | **none** | **none** | none | none | none | none |
| INT8 peak | mma **8×**; dp4a | WMMA iu8 = f16 rate (bytes only) | DP4a 4× | `VPDPBUSD ymm` 4× | `VPDPBUSD zmm` 4× | SSSE3 `pmaddubsw` ≈1× |
| INT4 | mma 16× | WMMA iu4 2× iu8 | none | none | none | none |
| Bandwidth (spec) | 360 GB/s, 12 GB | ≈256 GB/s LPDDR5X, shared | ≈90 GB/s DDR5-5600, shared | ≈90 GB/s | ≈256 GB/s shared | ≈32 GB/s per socket, NUMA |
| **Measured (E1, E2, E9–E12)** | plain F16-acc GEMM only **1.11–1.20×** end to end on Bonsai prefill; dp4a W2A8 MMQ 0.82–0.93× of the dequant + COMPUTE_32F path | **F16-accumulate coopmat no faster than F32-accumulate** (1.00–1.01×); Q8_0 prefill: F32-acc coopmat 1.10–1.25× faster than dp4a MMQ; int8 MMVQ decode 1.6× (Q4_0) / parity (Q8_0) vs F32-in | **every F16 arithmetic variant slower** (0.60–0.68×; compiler emits no packed FMA); DP4a MMQ prefill **34×** the scalar F32 GEMM; int8 MMVQ decode 1.3–1.7× | **VNNI = AVX2 maddubs** (1.00×); F32 GEMV reaches 23–26 GB/s on 6 P-cores (issue-bound, not DRAM-bound) | **VNNI 0.91×, AVX-512BW 0.82× of AVX2** per core; 8 cores reach ≈59 GB/s (F32 GEMV); multithreaded AVX-512BW +5–10% | F32 GEMV 9.9 GB/s (1 socket) / 21.4 GB/s (2); `GemvF16` 0.58×, BF16 shift-widen 1.41× of F32 (1 thread) |

Corrections against the draft:

- **VNNI's 4× peak does not show up** on the 155H or Zen 5: integer paths win by reading fewer bytes and doing less
  widening, not by VNNI (E1). .NET 10 cannot reach 512-bit VNNI on Zen 5 at all (`Avx10v1` reports unsupported).
- **Zen 5 CPU bandwidth:** 8 cores reach ≈59 GB/s, so CPU decode on Strix is DRAM-bound at 8 threads (E1).
- **3060 tensor-core peaks:** `.docs/precision/cuda/RESULTS.md` §4 quotes half of the whitepaper-derived figures used
  here (e.g. 25.5 vs 50.9 TF for F16-acc). The ratios agree; only the absolute scale differs. Use the ratios.
- **F16-acc HMMA's 2× rate does not become 2× prefill:** prefill is also bound by the F16 dequant expansion and, at 8k,
  attention (E2).
- **Vulkan features:** the device now enables `shaderFloat16`, 16-bit storage and `shaderInt8` (#374, fixed on `dev`
  in 972d5c53), so F16 storage on Vulkan is unblocked.
- **Westmere:** BF16 is the cheaper 16-bit format (shift vs software F16), now measured (E1).
- **Strix (RDNA 3.5):** F16- and F32-accumulate WMMA run at one rate, so F16 accumulation buys nothing and costs 13× the
  3060's F16-acc error (median KL 1.6e-3 on Bonsai). The draft's "F16-acc rate gain UNVERIFIED" is now **no** (E11).
- **Arc:** packed F16 (2× peak) does not materialise; the scalar F32 kernels are the right ones (E9, E10).

### 2.1 Engine practice (survey; links in `.docs/precision/SURVEY.md` §5)

| op class | llama.cpp | vLLM | TensorRT-LLM | SGLang | MLX |
|---|---|---|---|---|---|
| weight GEMM | CPU W?A8 (`vec_dot_type` Q8_0/Q8_1/Q8_K); CUDA int8-mma MMQ on Turing+ at any batch; cuBLAS fallback F16 + `COMPUTE_16F` | Marlin W4A16, F32 acc | cuBLASLt `COMPUTE_32F`; INT8 SmoothQuant | FP8 block-scale, F32 acc | group-64 4-bit, F32 acc |
| activations | **F32** | model dtype | model dtype | model dtype | model dtype |
| KV | K/V types independent (f16 … q4_0) | fp8 / int8 / nvfp4 | INT8/FP8 | fp8 / bf16 / fp4 | 8-bit group-64 |
| attention acc | CUDA FA: KQ F32, **PV half2**; CPU/Vulkan F32 | F32 | FP16 FMHA BMM1 F16 by default | F32 | F32 |
| norms | Σx² in **double** (CPU) | F32 | — | — | F32 |
| SSM/GDN state | **F32, hard-coded** | F32 for NemotronH / Qwen3.5 | `auto` | **F32** | **F32** |
| logits | F32 buffer | model dtype head; sampler F32 | — | — | — |
| split partials | F32 | F32 (Marlin reduce) | — | — | — |

llama.cpp accepts F16 accumulation but opts specific models back into F32 (GLM4 FFN-down and `wo`, #13101; non-FA KQ);
dotLLM's equivalent is "F16 where measured, F32 promotion where K is long" (E2).

---

## 3. Keep FP32 because …

Only what the data proved. Everything here is **in registers or within a tile**, not storage.

| # | what | where | evidence |
|---|---|---|---|
| K1 | **P·V accumulator across KV tiles** (flash prefill O) | CUDA and Vulkan, every device, every depth | CUDA: whole-range F16 O drifts with depth, median KL 1.7e-5 at 4k → 4.4e-4 at 32k → **8.8e-4 at 100k**, 47% of rows > 1e-3 at 100k (E7). Strix: 7.8e-6 at 2k → **2.8e-3 at 128k, 73.6% of rows > 1e-3, 10% > 1e-2, top-1 94.7%**; Tev1-4B PPL **+18%** at 128k (E9). F16 *within* a tile, F32 across (pv16t), is at the noise floor on both, but buys nothing (CUDA ≤ 0.3%, Strix 0.94–0.96×, Arc 0.61×). |
| K2 | **F32 promotion every ≤ 256 K-terms** in FP GEMM | CUDA prefill (any FP-accumulate GEMM with long K) | Plain F16 accumulation error grows with K: 2.0e-3 at K = 5,120 → **7.6e-3 at K = 17,408** (ffn_down, 3.7×). End to end: median KL 1.16e-4, 1.1% of rows > 1e-3, 0.6% top-1 flips. Promotion every 256 → median KL **6.3e-6** (18× lower), no row > 1e-3, 99.8% top-1. Promotion every 1,024 → 2.4e-5. (E2) |
| K2v | **F32 accumulation throughout** the coopmat GEMMs | Vulkan Strix (PQ2_0 ladder, Q8_0, Q4_0) | F16 accumulation: Bonsai median KL **1.6e-3**, 65.8% of rows > 1e-3, top-1 96.7%; Q4_0 KL max 0.106; **speed 1.00–1.01×** (one WMMA rate on RDNA 3.5). The promote-every-256 variant could not be measured (driver compiler crash) and has nothing to gain here. (E11, E12) |
| K6 | **Decode P·V accumulators** | Vulkan (Strix, Arc) | F16: median KL 7e-9 at 2k → **7.5e-5 at 128k** (Bonsai), 1.75e-4 with 2.1% of rows > 1e-2 and PPL +1.0% (Tev1-4B); no speed (the pass is bandwidth-bound). (E10) |
| K3 | **Softmax running max / sum and the online rescale** | CPU (registers only); every GPU kernel | No CPU here has FP16 arithmetic, so F32 accumulation is free. Vulkan decode with F16 statistics is accuracy-free at today's split sizes (KL ≤ 1.2e-8) but gives **no speed** (±1%), and would stagnate with fewer, longer splits; keep F32 (E10). GPU prefill variants never narrowed *m*/*ℓ* (E7, E9). |
| K4 | **RMSNorm sum of squares** | CPU (registers only), all backends by policy | Same as K3: F32 accumulate is free on AVX2/SSE; FP16 x² overflows above |x| ≈ 256 `[lit]`. |
| K5 | **Per-block scale promotion** `d_w·d_x·Σint32` | CPU int8 dots (per 32 for Q8_0/Q8_1, per 32-sub-block within 256 for Q8_K); CUDA MMQ | int32 cannot overflow inside a 32-element block (32·127·127 < 2³¹); this *is* the chunked promotion of K2 for integer paths (E4). |

**F32 kept by policy, not proven** — each needs a measurement and the user's sign-off under rule 1:

| what | status |
|---|---|
| Recurrent (GDN/SSM) carried state and its snapshots | not measured → **#1144** (engines unanimous on F32 `[lit]`; HF `mamba_ssm_dtype: float32` is evidence, not exemption) |
| Split-KV *m*/*ℓ* (LSE) partials, split-K combine | not measured → **#1144** (an F16 LSE at magnitude 20 gives ≈1.6% weight error `[lit, derived]`) |
| Sampled-row logits and the V-wide LSE | policy (4·V B per sampled row) → bounded to sampled rows by **#1137** |
| RoPE angle `pos·θ` (F64 for the YaRN ramp) | compute only; F16 cos/sin tables not measured → **#1142** |
| Residual stream of models that overflow F16 (BitNet) | per-model exception; measurement → **#1138**. Flint (3.5e4) instead carries an F16 stream scaled by 1/8 on CUDA (E13) |

**Not justified anywhere:** F32 weights, F32 KV, F32 MTP/draft KV, F32 inter-kernel activations (pending #1138), F32
embedding tables.

**Not required either:** the Vulkan GQA prefill's **Q hi/lo split** (an extra F16 term carrying Q's F32 residual into
QKᵀ). Dropping it gives KL at the noise floor (median ≤ 1.5e-7, max ≤ 4.7e-6 at 2k–128k on Bonsai; no Tev1 row
> 1e-3) and 1.67× attention, 1.46× a 1k prefill tile at 128k (E9). The unit test that needs it uses scores 2–3× and a
dropped term 4–8× larger than 50 M real Bonsai rows produce. Turning it off is the user's call: **#1148**.

---

## 4. Per-operation recommendations by device

"Xeon+3060" is the T5500 pairing: the 3060 holds what fits; the Xeon runs overflow layers, embedding and possibly the
LM head. Its GPU side follows the RTX 3060 column.

### 4.1 Recommendation matrix

| operation | 155H | Zen 5 | Xeon SSE4.2 | Xeon+3060 (CPU side) | RTX 3060 | Strix 8060S | Arc | evidence |
|---|---|---|---|---|---|---|---|---|
| **Weight GEMV (decode)** | int8 activations (Q8_0/Q8_1/Q8_K), AVX2 maddubs; VNNI optional | int8 activations, AVX2 maddubs; AVX-512BW only multithreaded (+5–10%) | int8 activations, SSSE3 (**only** viable tier) | keep the GGUF's quant format, W4A8/W8A8; prefer Q4_K/Q4_0 over Q8_0 | quantized weights × Q8_1 activations, dp4a (done) | **int8 MMVQ** (Q4_0 1.6×, Q8_0 parity vs F32-in; KL 1–3e-4) | **int8 MMVQ** (1.3–1.7× vs F32-in) | E1, E4, E12 |
| **Weight GEMM (prefill)** | Q4_K int8 tile; **Q8_0/Q4_0 need the Q4_K-style int8 microkernel (#1119)** | same; Q8_0 int8 today 0.6× of F32 | Q8_0 int8 R4 (1.4–1.6× F32); Q4_0 no gain | prefill on the 3060 where possible; else Q8_0 int8 | **int8-mma MMQ, Q8_1 activations, no weight expansion** (#1132 hybrid, #1135 dense) | f16 coopmat with in-tile dequant, **F32 acc**; Q8_0: coopmat 1.10–1.25× over dp4a MMQ (#1150); hybrid IQ4_NL/K-quants still scalar F32 (#1149) | **DP4a MMQ (34× the scalar F32 GEMM)**; hybrid host gap #1149 | E2, E3, E11, E12 |
| **FP GEMM accumulator** | F32 (no F16 ALU) | F32 / int32 | F32 / int32 | F32 / int32 | F16 within ≤ 256 K, **F32 across** (fused, K2); plain `COMPUTE_16F` rejected for Bonsai | **F32** (F16-acc no faster, KL 1.6e-3; K2v) | int32 | E2, E11 |
| **Attention QKᵀ** | int8 QKᵀ over q8_0 K (+10–25%) | int8 QKᵀ (+20–35%) | **int8 QKᵀ (~1.9×)** | int8 QKᵀ over q8_0 (after #970, #1120) | F32 S default; F16 S with exact pre-scale is safe (1.2e-4 at 100k) for ≤ 5% | F32 S; **Q as plain F16, no hi/lo split** (1.67×, #1148, needs sign-off); F16 S rejected (Tev1 tail KL 1.47) | F32 (F16 variants 0.66×) | E7, E8, E9 |
| **Attention P·V** | F32 acc | F32 acc | F32 acc | F32 acc | **F32 across tiles (K1)**; F16 within a tile allowed, no gain | **F32 across tiles (K1)**; within-tile F16 0.94× → F32; decode P·V F32 (K6) | F32 (F16 0.61×) | E7, E9, E10 |
| **Softmax / norm reductions** | F32 registers | F32 registers | F32 registers | F32 registers | F32 registers | F32 registers (F16 stats accurate but no gain) | F32 registers | K3, K4, E10 |
| **KV storage** | **q8_0** (0.53× F16 bytes); F16/BF16 safe; never q4_0 K | q8_0 (1.43× decode vs F32 at 16k) | q8_0 + int8 QKᵀ; **never F16** (software widen) | q8_0 + int8 QKᵀ | **q8_0 default; K8V4 when memory forces it; q4_0 last** (#1133) | hybrid F16 default (#1102), q8_0 in-tile (#1112); **dense host F32 → F16** (Llama-1B KL ~5e-4, 3B ~1e-6–2e-4; #1151) | same as Strix | E5, E6, E10b |
| **Recurrent state** | F32 (pending #1144) | same | same | same | same | same | same | — |
| **MTP / draft KV** | trunk KV type (#1141) | same | same | same | **trunk KV type, done** (#1095: 16,392 → 1,156 B/token at q4_0, acceptance 2.267 → 2.262) | trunk type (#1141); F16 measured invisible (8 → 4 KiB/token, KL 3.4e-8) | trunk type (#1141) | PR #1126, E10b |
| **LM head weights** | quantized, int8 GEMV | same | same | quantized on whichever side holds it | **packed ≤ 8-row chunks, no F16 copy** (done, #1099: 2,425 → 170 MiB, PPL +0.03%) | packed | packed | PR #1125 |
| **Logits** | F32 for sampled/scored rows only (#1137) | same | same | same | same (+486 MiB per 512-row window today) | same | same | — |
| **Embeddings** | GGUF quant type, one row dequantized per token | same | same | same, CPU-side lookup | same; no load-time F16 table (#1135) | packed, not F32 (#919) | packed | — |
| **Activations / scratch** | F16 or BF16 (not measured, #1138); fuse norm → Q8 at decode | same | **BF16** (no F16C) | hand-off to the 3060 in F16/BF16 (6 KB/token on 3B) | F16 (dense path already); hybrid F32 → F16 (#1138) | F16 (#1138; ffn-down F16 tile measured 13.5 → 23.5 TF) | F16 | E1 |
| **Weights that must be 16-bit** | BF16 shift-widen kernel (#1118) | BF16 (`VDPBF16PS` on the preview track) | **BF16, not F16** | BF16 | F16 | F16 | F16 | E1 |

### 4.2 Notes per operation class

**Weight GEMV / GEMM (E1–E4).**

- Integer activations win on **every CPU**: Xeon 5–8× over the float tier, 155H 1.2–1.6×, Zen 5 at parity
  (DRAM-bound, so formats win by bytes: Q4_K 7.1×, Q4_0 4.7×, Q8_0 3.0× over F32).
- Per-32 activation scales cost KL ≈ 1–2e-4 (ΔNLL ≈ 0). Q8_K's single scale per 256 costs 5–10× that (1.07e-3, 2.25%
  of rows > 1e-2), still ΔNLL ≈ 0. A Q8_K variant with per-32 sub-scales would recover it if K-quant accuracy is chased.
- F16 and BF16 weight paths have **kernel defects, not format defects**: `GemmF16` 0.18–0.24× of `GemmF32`, `GemvF16`
  0.69× on the 155H (#1117); BF16 has no kernel (dequant-rows 0.43×) while a shift-widen kernel runs 1.4–2.5× (#1118).
- CUDA: W2A8 (Q8_1 activations, int32 accumulate) is **more accurate than plain F16 accumulation** (median KL 5.1e-5 vs
  1.16e-4) and its per-projection error is flat in K (2.3–4.6e-3 relative), while F16-acc grows with K.

**Attention (E7, E8).**

- CPU decode attention is latency-bound per key at hd = 128 (≤ 6 GB/s on the 155H, ≤ 32 GB/s on Zen 5), so narrower KV
  buys memory, not speed, until the kernel is restructured. Production quantized-KV attention re-dequantizes each tile
  once per query head in a GQA group (#970).
- CUDA: F16 QKᵀ holds only because Bonsai's scale is a power of two and q/k are RMS-normed; recheck for unnormalized
  models. Verdict for the 3060: keep F32 accumulators; qk16 only if ≥ 100k prefill attention becomes the bottleneck.
- Vulkan agrees on every accumulator (E9, E10). The one real Vulkan win is dropping the Q hi/lo split (1.67× attention,
  KL at the noise floor; #1148). Real Bonsai scores stay at |s| ≤ 32.6 from 2k to 128k (q/k are RMS-normed); re-check
  with `DOTLLM_VK_EXP_FA_ACC=stats` for any model whose q/k are not normed. F16 S (qk16) has a heavy tail on Tev1-4B
  (one row at KL 1.47) and is rejected.

**KV storage (E5, E6).**

- **K needs ≥ 8 bits; V tolerates 4.** q4_0 *K* fails on both backends; K8V4 is the middle ground.
- At 4k, decode tok/s is format-independent on CPU (weights dominate); at 16k, q8_0 is 1.43× F32 on Zen 5. On the Xeon,
  today's production q8_0 path is 0.67× of F32 (0.86× with the int8-QKᵀ prototype) because of the per-head re-dequant.
- On the 3060, q8_0 costs nothing measurable (KL ≤ 9e-5 median, 100% top-1) at 53% of F16's bytes.
- Vulkan's remaining F32 KV stores narrow to F16 cleanly: the MTP head is invisible (KL 3.4e-8), and the dense host costs
  Llama-1B ~5e-4 median (the CPU F16 figure on the same model) and Llama-3B Q4_0 ≤ 2e-4 (E10b; #1151, #1141). Vulkan
  q8_0/K8V4/q4_0 KV on the hybrid host is not measured yet (#1112).

**Recurrent / MTP state.** The carried state is rewritten every token, so storage rounding compounds; it is O(1) in
sequence length, so narrowing saves little. Keep F32 pending #1144. MTP/draft KV follows the trunk: the verify step
keeps the output exact, so tune by acceptance × cost, not KL (#1095 measured no acceptance change at q4_0).

**RoPE.** Compute the angle in F32 (F64 for YaRN); GPUs compute sin/cos in-kernel (no table). CPU F32 tables are
duplicated (64 MiB × 2 at 128k on Llama-8B); F16 tables from F32 angles are not measured (#1142).

**Split partials.** *m*/*ℓ* stay F32 (a few KB); O partials and split-K slabs to F16 are not measured (#1144).

---

## 5. Measured evidence

Dated 2026-10-22/23. CPU models: Llama-3.2-3B Q8_0 (Q4_0 / Q4_K_M requantized from it, valid for activation-format
comparisons only) and Llama-3.2-1B Q8_0. CUDA model: Bonsai-2 27B PQ2_0-MTP (hidden 5,120, FFN 17,408, 64 layers, 16
full-attention, 24/4 heads × 256, vocab 248,320). 155H absolutes were taken under 97–100% background load (a quiet
re-run reproduced every ratio); use same-run ratios only.

### E1 — CPU weight GEMV, production entry points (ffn gate/up 8192×3072, COLD, speed × `GemvF32`)

| format (path) | 155H t6 | 155H t6 VNNI off | Strix t8 | Strix t16 | Xeon t1 | Xeon t6 | Xeon t12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| F32 `GemvF32` (GB/s) | 1.00 (24.7) | 1.00 (26.1) | 1.00 (≈59) | 1.00 | 1.00 (8.6) | 1.00 (9.9) | 1.00 (21.4) |
| F16 `GemvF16` | **0.69** | 0.80 | 2.02 | 1.58 | 0.58 | 1.28 | 1.40 |
| BF16 dequant-rows (no kernel) | **0.43** | 0.59 | 1.82 | 1.07 | 0.38 | 0.96 | 1.03 |
| BF16 shift-widen (study kernel) | 1.44 | 1.64 | 2.06 | 1.71 | 1.41 | 1.72 | 1.90 |
| Q8_0 float tier | 1.07 | 1.38 | **3.72** | 2.58 | 0.25 | 0.72 | 0.70 |
| Q8_0 int8 `GemvQ8_0` | **2.42** | 2.83 | 3.01 | 2.61 | **2.04** | **3.80** | 3.29 |
| Q4_0 float tier | 0.30 | 0.35 | 1.20 | 0.73 | 0.26 | 0.82 | 0.76 |
| Q4_0 int8 | **2.88** | 3.12 | 4.69 | 3.07 | **2.23** | **5.90** | **5.20** |
| Q4_K int8 `GemvQ4_K` | **4.14** | 4.04 | **7.13** | 4.12 | 2.57 (hot) | 6.41 (hot) | 6.47 (hot) |

Kernel tiers (Strix t8, COLD): Q8_0 × Q8_0 AVX2 3.02×, VNNI 3.01×, AVX-512BW **3.44×**; Q4_0 × Q8_1 VNNI prototype
bit-exact, no gain. Single-thread HOT on Zen 5: AVX2 1.10×, **VNNI 0.91×, AVX-512BW 0.82×**.

### E2 — CUDA prefill GEMM accumulation (Bonsai-2, RTX 3060)

Per projection (one 512-token tile, 496 GEMM calls; reference COMPUTE_32F; shared F16-output floor ≈ 3e-4):

| K | projection | F16-acc rel err | F16-acc row max | F16-acc + F32 every 256 | A8 (Q8_1 acts) | y absmax |
|---:|---|---:|---:|---:|---:|---:|
| 5,120 | attn q / k / v, GDN, ffn gate/up | 1.2–2.1e-3 | ≤ 5.5e-3 | 5–7e-4 | 2.3–4.4e-3 | 18–99 |
| 6,144 | ssm_out / attn_output | 2.72e-3 | 1.29e-2 | 8.0e-4 | 4.6e-3 | 156 |
| **17,408** | **ffn_down** | **7.58e-3** | **2.52e-2** | **5.8e-4** | 3.1e-3 | **1,036** |

End to end (1,136 scored rows: 3 chat prompts + 1,024 wikitext rows; full-vocab KL vs COMPUTE_32F; reference repeat ≤ 3e-7):

| variant | KL median | KL p99 | KL max | rows > 1e-3 | top-1 | PPL Δ (wikitext) | prefill × ref @512 / 2k / 8k |
|---|---:|---:|---:|---:|---:|---:|---|
| F16-acc (`COMPUTE_16F`) | **1.16e-4** | 1.04e-3 | 9.49e-3 | **1.1%** | **99.4%** | +0.012% | **1.20 / 1.11 / 1.14** |
| F16-acc, F32 every 1,024 | 2.41e-5 | 1.47e-4 | 4.03e-4 | 0% | 99.7% | −0.032% | 1.08 / 0.94 / 0.98 |
| **F16-acc, F32 every 256** | **6.30e-6** | 3.51e-5 | 2.45e-4 | 0% | **99.8%** | −0.016% | 0.82 / 0.74 / 0.85 |
| W2A8 dp4a MMQ | 5.12e-5 | 6.61e-4 | 1.26e-3 | 0.4% | 99.5% | −0.053% | 0.86 / 0.82 / 0.93 |

The promoted variants are slow only because cuBLAS cannot express promotion: the experiment writes `K/chunk` F16
partials and reduces them in a second pass. A fused kernel promotes in registers at F16-acc speed (#1132). Bonsai's
default prefill is COMPUTE_32F today: its host never calls `CudaGemm.ConfigureDefault`, so G1 (F16-acc) was validated on
Llama-3.2-1B (K ≤ 8,192) only.

### E3 — CPU prefill GEMM (n = 64, 8192×3072, GFLOP/s at min)

| kernel | 155H t6 (loaded) | Strix t8 | Xeon t6 |
|---|---:|---:|---:|
| F32 `GemmF32` | 54 | **504** | 45 |
| F16 `GemmF16` | **9.5** | **123** | 36 |
| Q8_0 float tier | 40 | 467 | 36 |
| Q8_0 int8 `GemmQ8_0` | 48 | 297 | 64 |
| Q8_0 int8 R4 tiled (production) | 49 | 313 | 70 |
| Q4_0 int8 | 47 | 411 | 35 |
| **Q4_K int8 `GemmQ4_K`** | **121** | **1006** | 52 |

Whole-model 3B prefill on the same loaded 155H: Q4_K_M **18.4 tok/s** vs Q8_0 5.3 vs Q4_0 5.0.

### E4 — CPU activation-format accuracy (same weights, F32-activation reference; 1,024 positions)

| weights | activation format | KL median | p99 | KL > 0.01 | top-1 | ΔNLL/token |
|---|---|---:|---:|---:|---:|---:|
| Q8_0 (155H) | Q8_0 (per 32) | 2.33e-4 | 3.80e-3 | 0.10% | 98.73% | −0.0004 |
| Q8_0 (Strix, AVX-512/VNNI) | Q8_0 | 2.38e-4 | 5.16e-3 | 0.10% | 98.73% | −0.0015 |
| Q4_0 | Q8_1 (per 32) | **1.13e-4** | 5.09e-4 | 0.00% | 99.32% | +0.0004 |
| Q4_K_M | **Q8_K (per 256)** | **1.07e-3** | 1.53e-2 | **2.25%** | 97.17% | −0.0004 |

### E5 — CPU KV formats (vs F32 KV; accuracy is ISA-independent, reproduced on 155H and Xeon)

| model / ctx | KV | bytes/token | KL median | p99 | KL > 0.01 | top-1 | ΔNLL |
|---|---|---:|---:|---:|---:|---:|---:|
| 3B / 4k | f16 (sim) | 114,688 | 2.42e-4 | 6.4e-4 | 0% | 97.9% | −0.002 |
| | bf16 (sim) | 114,688 | 2.50e-4 | 7.9e-4 | 0% | 97.9% | −0.001 |
| | **q8_0** | 60,928 | 3.41e-4 | 1.3e-3 | 0% | 97.9% | −0.005 |
| | q8_0 + int8 QKᵀ | 60,928 | 4.34e-4 | 1.8e-3 | 0% | 97.9% | −0.002 |
| | k q8_0 / v q4_0 | 46,592 | 2.11e-3 | 1.1e-2 | 2.1% | 97.9% | +0.003 |
| | **q4_0** | 32,256 | **2.42e-2** | 1.3e-1 | **72.9%** | 95.8% | **+0.036** |
| 1B / 4k | f16 / bf16 (sim) | 32,768 | 4.7e-4 / 4.6e-4 | 4.6e-3 / 1.8e-3 | 0% | 95.3% / 96.9% | −0.004 / −0.003 |
| | q8_0 | 17,408 | 7.54e-4 | 4.5e-3 | 0% | 95.3% | −0.003 |
| | k8 / v4 | 13,312 | 4.92e-3 | 2.8e-2 | 12.5% | 92.2% | −0.016 |
| | **q4_0** | 9,216 | **1.26e-1** | 4.8e-1 | **90.6%** | **71.9%** | **+0.145** |
| 1B / 16k | f16 / bf16 (sim) | 32,768 | 7.2e-4 / 6.6e-4 | 5.3e-3 / 4.1e-3 | 0% | 97.9% / 100% | −0.007 / +0.006 |
| | q8_0 | 17,408 | 1.01e-3 | 8.8e-3 | 0% | 100% | +0.002 |
| | k8 / v4 | 13,312 | 6.77e-3 | 9.2e-2 | 31.3% | 91.7% | +0.010 |
| | **q4_0** | 9,216 | **1.36e-1** | 9.1e-1 | **93.8%** | 83.3% | **+0.311** |

Whole-model decode tok/s: Strix 1B/16k F32 9.88 vs **q8_0 14.1**; Xeon t6 1B/2k F32 **6.68** vs q8_0 4.45 vs q8_0 + int8
QKᵀ 5.72. F16/BF16 KV were simulated by rounding on write (exact for accuracy; speed comes from E8 only).

### E6 — CUDA KV formats (Bonsai-2, vs F16 KV; 64 prefill-chunk + 48 decode rows per cell)

| ctx | KV | KiB/token (16 attn layers) | KL median (prefill / decode) | KL p99 (prefill / decode) | rows > 1e-2 | top-1 |
|---:|---|---:|---|---|---|---|
| 8k | **q8_0** | 34 | 6.6e-6 / 8.3e-5 | 1.9e-4 / 9.8e-4 | 0% / 0% | 100% / 100% |
| 8k | k8v4 | 26 | 5.7e-4 / 1.5e-3 | 4.9e-3 / 2.7e-2 | 0% / 2.1% | 95.3% / 93.8% |
| 8k | **q4_0** | 18 | 1.4e-3 / 2.9e-3 | **9.9e-2 / 1.3e-1** | **7.8% / 10.4%** | 95.3% / 95.8% |
| 32k | **q8_0** | 34 | 2.7e-5 / 9.1e-5 | 1.2e-4 / 4.8e-4 | 0% / 0% | 100% / 100% |
| 32k | k8v4 | 26 | 1.1e-3 / 9.6e-4 | 6.8e-3 / 5.0e-3 | 0% / 0% | 98.4% / 100% |
| 32k | **q4_0** | 18 | 3.1e-3 / 1.6e-3 | **1.2e-1 / 4.0e-2** | **4.7% / 10.4%** | 100% / 97.9% |

Prefill speed is format-independent (±1%); decode differences are within single-run noise. At 128k, K8V4 costs 3.25 GiB
vs 2.25 GiB for q4_0. Earlier repo PPL (`docs/KV_CACHE.md`): F16 10.2350, q8_0 −0.004%, K8V4 +0.18%, q4_0 +0.32%.

### E7 — CUDA flash prefill accumulators (Bonsai-2; shared prefix, variant only on the scored forwards)

| ctx (KV) | variant | KL median | KL p99 / max | rows > 1e-3 | top-1 | tile tok/s × F32 |
|---:|---|---:|---:|---:|---:|---:|
| 4k (F16) | qk16 / pv16 / pv16t | 6.9e-6 / 1.7e-5 / 4.7e-6 | ≤ 8.0e-5 | 0% | 100% | 1.004–1.016 |
| 32k (Q4_0) | qk16 | 1.2e-4 | 8.2e-4 | 0% | 100% | 1.018 |
| 32k | **pv16** (whole-range F16 O) | **4.4e-4** | 1.1e-2 | **15.6%** | **96.9%** | 1.006 |
| 32k | pv16t (F16 within a 16-key tile) | 1.7e-4 | 1.4e-3 | 3.1% | 100% | 1.003 |
| 100k (Q4_0) | qk16 | 1.2e-4 | 1.9e-3 | 1.6% | 98.4% | **1.047** |
| 100k | **pv16** | **8.8e-4** | 2.6e-2 | **46.9%** | 98.4% | 1.016 |
| 100k | pv16t | 1.5e-4 | 7.9e-4 | 0% | 98.4% | 1.002 |
| 100k | qk16pv16 | 9.5e-4 | 2.8e-2 | 50.0% | 98.4% | 1.073 |

Kernel level (synthetic 4,224 keys, F16 KV, relRms): qk16 1.04e-3, pv16 3.34e-3, **pv16t 2.5e-4**. The F32 repeat
gives KL ≤ 1e-7. 32k with F16 KV did not fit next to the speed tile, so 32k/100k use Q4_0 KV.

### E8 — CPU decode attention (1 layer, 24q/8kv heads, hd 128; speed × harness F32 tile, min)

| arm | 155H 4k | 155H 16k | Strix 4k | Strix 16k | Xeon 4k |
|---|---:|---:|---:|---:|---:|
| harness F16 KV (row convert) | 0.59 | 0.45 | 1.03 | 1.10 | 0.91 |
| harness q8_0 (dequant row) | 1.35 | 0.82 | 1.23 | 1.28 | 1.18 |
| harness **q8_0 int8 QKᵀ** | **1.62** | 0.92 | **1.35** | 1.28 | **1.95** |
| production q8_0 (dequant tile) | 1.51 | 0.77 | 0.77 | 1.10 | 1.10 |
| production q8_0 + int8 QKᵀ (prototype) | **1.65** | 0.88 | 1.22 | 1.22 | **1.89** |

### Vulkan arm (Strix 8060S, Arc) — E9 to E12

Strix: gfx1151, coopmat 16×16×16 f16→f32 and f16→f16, s8→s32. Arc: no coopmat, DP4a. Models: Bonsai-2 (F16 KV, the
#1102 default), Tev1-4B (Qwen3.5 hybrid, 8 attention layers, 16/4 × 256; IQ4_NL on Strix, Q4_K_M on the Arc),
Llama-3.2-1B Q8_0, Llama-3.2-3B Q4_0 (requantized). The reference is the shipped kernel in the same process; an F32
repeat gives KL ≤ 1e-7 everywhere. Depth runs grow one wikitext prefix 2k → 128k; each cell scores 512 rows through the
flash prefill kernel or 48 teacher-forced decode rows. Arc timings were taken on a loaded box: use only interleaved
kernel ratios.

#### E9 — Vulkan GQA flash prefill accumulators (#441 kernel)

Bonsai-2, Strix, shared prefix (variant only on the scored forwards):

| ctx | variant | KL median | KL max | rows > 1e-3 | rows > 1e-2 | top-1 | PPL Δ | 1k tile at depth × F32 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2k | **Q lo off** (no hi/lo split) | 1.3e-7 | 2.7e-6 | 0% | 0% | 100% | +0.003% | 1.022 |
| 2k | pv16 (whole-range F16 O) | 7.8e-6 | 1.5e-4 | 0% | 0% | 99.8% | −0.001% | 1.000 |
| 32k | **Q lo off** | 1.5e-7 | 1.2e-6 | 0% | 0% | 100% | +0.002% | **1.208** |
| 32k | qk16 (F16 S) | 7.1e-6 | 2.5e-4 | 0% | 0% | 99.8% | −0.031% | 1.231 |
| 32k | pv16t (F16 within a 32-key tile) | 1.5e-7 | 2.4e-6 | 0% | 0% | 100% | −0.001% | 0.971 |
| 32k | pv16 | 5.7e-4 | 9.9e-3 | 27.0% | 0% | 99.2% | −0.228% | 1.001 |
| 64k | **Q lo off** | 7.9e-8 | 4.7e-6 | 0% | 0% | 100% | −0.005% | **1.333** |
| 64k | qk16 | 6.6e-6 | **5.8e-2** | 1.6% | 0.2% | 100% | +0.042% | 1.371 |
| 64k | pv16 | 7.5e-4 | 5.1e-2 | 42.4% | 2.7% | 97.7% | +0.386% | 1.002 |
| 128k | **Q lo off** | 1.2e-7 | 3.0e-6 | 0% | 0% | 100% | +0.002% | **1.464** |
| 128k | qk16 | 4.7e-6 | 1.0e-4 | 0% | 0% | 99.8% | +0.012% | 1.593 |
| 128k | pv16t | 1.2e-7 | 3.2e-6 | 0% | 0% | 100% | −0.001% | 1.013 |
| 128k | **pv16** | **2.8e-3** | 8.0e-2 | **73.6%** | **10.0%** | **94.7%** | +0.679% | 1.001 |

Tev1-4B IQ4_NL (Strix): Q lo off median 3–7e-8, worst row 8.4e-4, none > 1e-3; qk16 one row at **KL 1.47** (8k);
pv16 at 128k median 1.0e-2, 51% of rows > 1e-2, **PPL +18.1%**.

Kernel level (synthetic 24/4/256, F16 KV, TFLOP/s and × default):

| depth | default (Q hi/lo split) | Q lo off | qk16 | pv16t | pv16 |
|---:|---:|---:|---:|---:|---:|
| 8k | 4.01 | 6.67 (1.67×) | 7.03 (1.76×) | 3.77 (0.94×) | 4.02 (1.00×) |
| 128k | 3.93 | 6.58 (1.67×) | 7.08 (1.80×) | 3.71 (0.94×) | 3.95 (1.01×) |

Score statistics over 50 M real Bonsai rows (2k–128k): max |s| 29.3–32.6, per-row p99.9 < 24, dropped Q_lo·K term ≤ 6.8e-3
(bulk 0.5–2e-3). The unit test that fails without the split has scores ~37–56 and a dropped term up to 1.3e-2.

Arc (scalar GQA kernel, Tev1-4B Q4_K_M at 2k): qk16 KL 1.6e-7, **0.66×**; pv16t 2.3e-8, **0.61×**; pv16 1.0e-4 with 4.3% of
rows > 1e-3, 0.60–0.68×.

#### E10 — Vulkan decode attention (#1102 grouped split-KV pass, F16 KV), Bonsai-2, Strix

| ctx | variant | KL median | KL max | top-1 | PPL Δ | Δ ms per attention layer |
|---:|---|---:|---:|---:|---:|---:|
| 2k | st16 (F16 max/sum) | 6.7e-11 | 6.7e-8 | 100% | +0.001% | +0.004 |
| 2k | pv16 (F16 P·V acc) | 7.4e-9 | 1.6e-7 | 100% | +0.003% | +0.004 |
| 32k | st16 | 6.4e-9 | 9.3e-8 | 100% | +0.003% | −0.001 |
| 32k | pv16 | 3.9e-6 | 2.7e-5 | 100% | −0.008% | +0.020 |
| 128k | st16 | 1.2e-8 | 1.0e-7 | 100% | −0.003% | +0.013 |
| 128k | **pv16** | **7.5e-5** | 4.3e-4 | 100% | +0.192% | +0.037 |

Baseline decode 43.5 / 67.7 / 140.1 ms/token at 2k / 32k / 128k. Tev1-4B: pv16 median 1.75e-4, 2.1% of rows > 1e-2, PPL
+1.0% at 128k. Kernel level (ms per layer, Strix / Arc) at 32k: default 1.494 / 12.09, st16 1.497 / 12.17, pv16 1.503 /
12.48. No variant is faster on either device.

#### E10b — Vulkan F32 KV stores narrowed to F16 (stored rows rounded to F16 values; reference = same host, F32 rows)

| store | model / device | rows | KL median | KL p99 / max | rows > 1e-3 | top-1 | PPL Δ | bytes/token F32 → F16 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Bonsai-2 MTP head KV (`VulkanMtpState`) | Bonsai-2, Strix | 128 drafts | **3.4e-8** | 4.0e-7 / 5.2e-7 | 0% | 100% (54/128 draft hits both arms) | +0.005% | 8 → 4 KiB |
| dense `VulkanKvCache` | Llama-3.2-1B Q8_0, Strix, prefill / decode | 1,024 / 64 | 4.7e-4 / 5.4e-4 | 3.3e-3 / 3.0e-3 | 15.1% / 21.9% | 98.6% / 98.4% | −0.16% / +0.82% | 64 → 32 KiB (K+V) |
| same | Llama-3.2-1B Q8_0, Arc, prefill / decode | 1,024 / 64 | 4.7e-4 / 4.2e-4 | 3.3e-3 / 3.3e-3 | 16.3% / 10.9% | 99.0% / 100% | −0.08% / −0.36% | |
| same | Llama-3.2-3B Q4_0, Strix, prefill / decode | 1,024 / 64 | 7.6e-7 / 1.7e-4 | 6.1e-6 / 2.0e-3 | 0% / 3.1% | 100% / 100% | −0.006% / +0.24% | 224 → 112 KiB (K+V) |
| same | Llama-3.2-3B Q4_0, Arc, prefill / decode | 512 / 48 | 3.7e-6 / 1.8e-4 | 1.7e-5 / 1.2e-3 | 0% / 4.2% | 99.8% / 97.9% | 0.00% / −0.10% | |

The Llama-1B dense-KV cost (~5e-4) matches the CPU arm's F16 KV result on the same model (E5), so it is the format on a
small GQA-8 model, not a kernel defect. (`.docs/precision/vulkan/RESULTS.md` §3 quotes the dense bytes per K or V side;
the figures here are K+V.)

#### E11 — Vulkan coopmat GEMM accumulation, Bonsai-2 PQ2_0 prefill (Strix, 128×128 ladder)

| variant | rows | KL median | KL p99 | KL max | rows > 1e-3 | rows > 1e-2 | top-1 | PPL Δ (all / wikitext) | prefill × F32 @512 / 2k / 8k |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| **F16-acc** | 1,136 | **1.6e-3** | 1.4e-2 | 2.6e-2 | **65.8%** | 2.2% | **96.7%** | +0.54% / +0.27% | **1.00 / 1.01 / 1.01** |
| F16-acc, F32 every 256 | not measured: the AMD Windows driver crashes in `vkCreateComputePipelines` for every format | | | | | | | | |

F32-acc reference: 230.3 / 222.4 / 207.4 tok/s. Same 1,136-row protocol as E2: Strix F16-acc error is 13× the 3060's.

#### E12 — Vulkan dense Q8_0 / Q4_0 paths (reference = shipped path: Strix Q8_0 prefill dp4a MMQ, Q4_0 prefill F16 coopmat F32-acc, decode dp4a MMVQ)

| model / device | arm | prefill KL median / max | decode KL median / max | prefill × default (512 / 2k / 8k) | decode × default |
|---|---|---|---|---|---|
| 1B Q8_0, Strix | coopmat F32-acc (no MMQ) | 3.2e-4 / 4.0e-2 | 5.6e-4 / 6.2e-3 | **1.25 / 1.20 / 1.10** | 1.00 |
| 1B Q8_0, Strix | coopmat F16-acc | 1.8e-3 / 1.9e-2 | 6.2e-4 / 1.9e-2 | 1.21 / 1.20 / 1.11 | 1.00 |
| 1B Q8_0, Strix | F32-in GEMV decode (no MMVQ) | 0 | 3.0e-4 / 2.8e-3 | 1.00 | 1.01 |
| 3B Q4_0, Strix | coopmat F16-acc | 3.6e-4 / **0.106** | 2.9e-4 / 2.3e-3 | 1.01 / 0.98 / — | 1.00 |
| 3B Q4_0, Strix | F32-in GEMV decode | 0 | 1.1e-4 / 1.4e-3 | 1.00 / 0.98 / — | **0.62** |
| 1B Q8_0, Arc | scalar F32 GEMM (no MMQ) | 3.2e-4 / 5.3e-3 | 5.4e-4 / 1.3e-2 | **0.029** (10.9 vs 370 tok/s) | 0.92 |
| 1B Q8_0, Arc | F32-in GEMV decode | 0 | 2.6e-4 / 1.8e-3 | 1.01 | **0.76** |
| 3B Q4_0, Arc | F32-in GEMV decode | 0 | 1.5e-4 / 1.3e-3 | 1.01 | 0.60 (1 rep) |

No F32-weight reference exists on Vulkan, so the MMQ vs coopmat comparison is path against path (KL 3.2e-4 apart);
which is closer to F32 truth is #1150.

### E13 — Activation range at GPU F16 / Q8_1 storage sites (#1191)

Measured on the CPU forward with `DOTLLM_AMAX_PROBE=<tsv>` (`ActivationRangeProbe`: per layer and site, max |x| and the
largest 32-element block sum, which is what a Q8_1 block's F16 `s` holds). Prefill in 512-token chunks through the KV
cache. Corpora: an 8K-token Flint-format coding-agent trace (system + tools + tool calls over repo sources and git logs)
and 8K tokens of wikitext-2 test.

**Recursive Flint Q4_1** (16,512 tokens). Max over all layers; the large values all sit in layers 46-48.

| site | max \|x\| | max \|block sum\| | layer | GPU storage | headroom vs 65504 |
|---|---|---|---|---|---|
| routed down input (SwiGLU out) | **239,269** | 245,325 | 48 | Vulkan grouped coopmat F16, CPU Q8_1 `s` | **overflows**: guarded (#1158 CPU, #1171 Vulkan) |
| residual stream | **34,605** | 39,922 | 48 | CUDA F16 (`Residual`) | **1.9x**: scaled by 1/8 on CUDA (#1191) |
| MoE output | 32,610 | 34,953 | 46 | CUDA F16 (MoE staging out) | 2.0x: scaled in F32 before the store (#1191) |
| shared-expert down input | 6,308 | 8,561 | 48 | Vulkan coopmat F16 (prefill), CPU Q8_1 `s` | 10x / 7.7x: unguarded |
| FFN input (RMSNorm out) | 3,359 | 3,296 | 46 | coopmat F16 gate/up | 20x |
| attention in | 218 | 204 | 48 | | 300x |
| o_proj input | 126 | 226 | 48 | coopmat F16, CUDA F16 | 290x |
| q / k / v | 38 / 45 / 127 | 65 / 68 / 226 | | F16 KV | > 500x |
| dense down input (layers 0-1) | 9.3 | 9.0 | 0 | coopmat F16, CUDA `swiglu_f16` | 7000x |

The residual grew with the corpus (17.4K after 3K agent tokens, 24.2K after 6.6K, 34.6K once wikitext followed), so it
is the site to watch at longer contexts.

On Strix Vulkan (8K agent trace and 8K wikitext, 512-token chunks through the F16 KV cache, 64 teacher-forced decode steps
each) every logit row is finite. With the grouped-down guard off (`DOTLLM_VK_MOE_DOWN_RESCALE=0`) the rows also stay finite,
because the F16 KV write clamps to ±65504, but they are wrong: KL(guard on ‖ off) 52-54 nats, top-1 agreement 0%. A
finiteness check alone cannot see this overflow. Short prompts vs the CPU are unchanged (KL mean 1.44e-3, top-1 99.0%, the
same as #1171). 16K / 32K prefill loses the device on Strix before reaching the end (#1200), so those lengths were not
checked on Vulkan.

**Qwen3.6-35B-A3B Q8_0** (5.6K tokens, same corpora): every site below 200 (routed down input 98, shared down input
164, residual 39), so its unguarded Vulkan grouped down (`VulkanQwen3MoeHybridTransformerModel`) has > 300x headroom.

Storage map (where an activation becomes F16, or Q8_1 with F16 `d`/`s`):
- **Vulkan**: every Q8_1 producer writes `d` and `s` as F32, so the only F16 activation stores are the coopmat
  `float16_t` operand casts (dense GEMM, grouped MoE B operand) and F16 KV (clamped to ±65504). Residual and MoE
  outputs are F32.
- **CUDA**: `quantize_x_to_q8_1` stores `d` as half and `s` as an integer sum (exact). The generic FP16 loop holds the
  residual, norm outputs, SwiGLU output and MoE staging in half; Flint's Q4_1 experts take F32 input. LFM2's
  `lfm2_quant_q8_1_f32` stores `s = d·Σq` as half (routed down guarded; shared / dense down not, LFM2 values small).
- **CPU**: Q8_1 `s` is F16 (routed down guarded; shared down at 7.7x).

### E14 — Flint per-row KL at depth: Q8_1 activation rounding, not a backend defect (#1218)

Real Flint Q4_1. Corpus: the first 8,192 tokens of the wikitext-2 test text E13 used, tokenised by llama.cpp (ids taken
from its `--kl-divergence-base` file, so every engine sees the same stream). dotLLM rows come from `NemotronProbe steps`
with `STEPS_ALLROWS=1` (512-token prefill chunks through the KV cache). llama.cpp rows come from `llama-perplexity`, which
scores the second half of each window, so its shallow bins are separate runs at `-c 64 … 8192` over the same prefix.

**Per-row KL by depth (mean / median, top-1 %).** NLL of the true next token per bin is within 0.03 for every dotLLM pair.

| positions | dotLLM CPU vs Strix VK | dotLLM CPU vs Arc VK | dotLLM CPU vs CPU + 1e-6 noise | llama CPU vs llama Strix VK | llama CPU vs dotLLM CPU |
|---|---|---|---|---|---|
| 0-256 | 0.030 / 0.005, 96.1 | 0.031 / 0.005, 95.7 | 0.026 / 0.005, 93.0 | | 0.11-0.15 / 0.09-0.14 (32-255) |
| 256-512 | 0.024 / 0.003, 94.9 | 0.041 / 0.003, 93.8 | 0.028 / 0.003, 93.4 | | 0.15-0.18 / 0.14, 93 |
| 512-1024 | 0.108 / 0.003, 94.1 | 0.11-0.14 / 0.003-0.004 | 0.07-0.14 / 0.003-0.004 | | 0.18-0.21 / 0.12-0.13 |
| 1K-2K | 0.078 / 0.003, 92.7 | 0.060 / 0.004, 92.8 | | | 0.23-0.26 / 0.15 |
| 2K-4K | 0.077 / 0.004, 93.1 | 0.089 / 0.005, 92.4 | | | 0.20-0.25 / 0.15-0.17 |
| 4K-6K | 0.084 / 0.003, 93.4 | 0.076 / 0.004, 93.0 | | 0.05-0.11 / 0.005, 92-95 | 0.19-0.29 / 0.14-0.17 |
| 6K-8K | 0.072 / 0.004, 93.1 | 0.080 / 0.004, 93.5 | | 0.07-0.10 / 0.004-0.007, 91-94 | 0.23-0.27 / 0.16-0.19 |

- Nothing blows up with depth: not at the 513 window, not at 1K, 2K or 8K. Every pair has a small median and a heavy tail
  (isolated rows at KL 10-20 nats where one engine is confident and the other is not).
- **The tail is the same between two runs of one engine that differ by a 1e-6 relative embedding perturbation**, and the same
  between llama.cpp CPU and llama.cpp Vulkan. Strix Vulkan F16 vs F32 KV: KL 0.008 / 0.0006 at 0-511 and 0.059 / 0.0015 at
  2048-2515, i.e. the same kind of disagreement.
- llama.cpp vs dotLLM has a larger median (~0.15) at every depth from 32 tokens on: llama.cpp quantises the activations of
  every Q4_1 matmul to Q8_1, dotLLM's dense prefill does not (see the Flint section of `docs/MODEL_CONFIG.md`).
- llama.cpp CPU `-ctk f32 -ctv f32` is bit-identical to its F16 KV (same PPL 24.4359, KL 0 at 512 and 8K).

**Mechanism** (dotLLM CPU, diagnostic patch not committed: embedding noise, per-layer residual dump, router top-8 log).
With the default packed Q4_1 × Q8_1 expert path, a 1e-6 relative embedding perturbation stays at 1-2e-6 through layers 0-1
(dense), then jumps to >1e-4 at 156 of 512 positions in layer 2 (the first MoE layer) and to 2-3e-3 at every position after
layer 3 (attention spreads it). Q8_1 rounding is discontinuous: a 1e-6 change flips the rounding of some elements by a full
quantum, and Flint's per-block outliers make the quantum large. From there router top-6 near-ties flip (16.8 % of
(layer, token) decisions differ by the end, 6th-7th logit gap median 0.09) and the residual difference reaches O(1) at the
rows that blow up. With exact F32 expert activations (`packedLegacyExperts` off) the same perturbation stays at 1e-6 through
layer 20 and gives KL 1e-4 (512 wiki rows) / 0.02 (last 128 rows of the 8K agent trace), top-1 97-100 %.

| 8K agent trace, last 128 prefill rows | KL mean / median | top-1 | NLL |
|---|---|---|---|
| dotLLM CPU vs CPU + 1e-6 noise (Q8_1 experts) | 1.03 / 0.26 | 64.8 % | 5.27 |
| dotLLM CPU (local) vs dotLLM CPU (Strix, #1161) | 1.18 / 0.24 | 64.1 % | |
| same perturbation, exact F32 experts | 0.020 / 0.0001 | 96.9 % | 5.17 / 5.21 |
| Q8_1 experts vs exact F32 experts | 0.89 / 0.22 | 68.8 % | 5.27 vs 5.17 |

**Consequence:** per-row KL between any two Flint implementations that both quantise expert activations measures the
spread of their rounding, not correctness. Gate long context on NLL / perplexity over ≥ 4K scored tokens (here every dotLLM
pair agrees within 0.03 nats per bin) plus median KL; keep the mean-KL gate for short prompts. Whether Q8_1 expert
activations cost Flint quality (exact was 0.1 nats better over 128 agent rows, level on wikitext) is open.

---

## 6. Decision table: CLAUDE.md rule → evidence

| CLAUDE.md rule (Precision & Expansion) | evidence | verdict |
|---|---|---|
| 1. FP32 gives no meaningful accuracy gain over FP16/BF16 | F16/BF16 KV vs F32: median KL 2.4–7.2e-4, ΔNLL ≈ 0, 0% rows > 1e-2 (E5); Vulkan F32 → F16 KV: MTP head 3.4e-8, dense Llama-1B ~5e-4, Llama-3B ≤ 2e-4 (E10b) | **confirmed** for KV on all three backends; activations not measured (#1138) |
| 1. FP16/BF16 is the widest stored default | KV (E5, E6); MTP head q4_0 vs F32: acceptance 2.262 vs 2.267, greedy output identical (#1095) | confirmed |
| 1. Always evaluate formats below FP16 | q8_0 KV ≈ F16 on both backends (E5, E6); int8 activations ≤ 2.4e-4 KL (E4); **q4_0 K fails** (E5, E6); FP8: no device has it | confirmed; q4_0 K excluded |
| 1. Accumulate F32 by default for norms, softmax/attention sums, long-KV attention | **PV across tiles proven on CUDA and Vulkan** (K1: 8.8e-4 at 100k CUDA, 2.8e-3 and 73.6% of rows > 1e-3 at 128k Strix); **decode P·V proven** (K6, E10); softmax stats in F16 accurate at today's split sizes but no faster (E10); norms: no F16 ALU on CPU (K3, K4) | **confirmed** for long-range sums; softmax/norm kept F32 because narrowing buys nothing |
| 1. … recurrent-state updates, split-reduction combines | not measured | open → #1144 |
| 1. Compute-bound GEMM: F16 accumulation where faster and KL-parity passes; promote every 64–256 for long K | RTX 3060: plain F16-acc 1.11–1.20× faster but fails on Bonsai (median 1.16e-4, 0.6% top-1 flips, grows 3.7× with K); **promote every 256: 6.3e-6** (E2). **Strix: F16-acc is not faster (1.00–1.01×)** and fails (1.6e-3, 65.8% of rows > 1e-3; Q4_0 max 0.106) (E11, E12). Arc: no F16 path is faster (0.60–0.68×) | **3060: yes, with promotion** (fused, #1132). **Strix, Arc: no** — F32 accumulation |
| 1. Integer dot products accumulate int32 | CPU W8A8/W4A8 KL 1–2e-4 at per-32 scales (E4); CUDA W2A8 5.1e-5, better than F16-acc (E2); Vulkan MMVQ 1–3e-4 vs F32-in, 1.3–1.7× (Arc), 1.6× (Strix Q4_0); Arc MMQ prefill 34× (E12) | confirmed; per-32 activation scales preferred over per-256; Strix Q8_0 prefill prefers the F32-acc coopmat (#1150) |
| 1. Stored F32 partials (split-K, split-KV) need sign-off | not measured | open → #1144 |
| (not a rule; a kernel extra) Q hi/lo split in Vulkan GQA prefill | dropping it: KL ≤ 1.5e-7 median, 1.67× attention, 1.46× a 128k tile (E9) | turn off by default → #1148 (user sign-off) |
| 1. Model-declared FP32 state is evidence, not exemption | engines unanimous `[lit]`; no dotLLM measurement | open → #1144 |
| 2. Kernels read quantized blocks directly | int8 tiers beat dequant tiers on every CPU (E1); W2A8 MMQ more accurate than dequant + F16 GEMM (E2); LM head packed: 2,425 → 170 MiB, PPL +0.03% (#1099); kvq prefill without the F16 scratch (#1094) | confirmed; remaining expansions in §7 |
| 2. Expansion fallbacks bounded, admitted, logged, with byte cost | LM-head tiles bounded and admitted (#1099); MTP short-history tier bounded at 256 rows (#1095) | in place for those paths; others open (§7) |
| 3. Disk ↔ device transfers go direct | outside this study | not measured |

---

## 7. Violation inventory (rule 1 FP32 stores, rule 2 expansions)

Models: **L1B** Llama-3.2-1B, **L8B** Llama-3.1-8B, **B2** Bonsai-2 27B. "Fixed on dev" means merged into `dev`; the PR
to `main` may still be open. Sources: the draft sweep (`.docs/precision/SURVEY.md` §3), the #1104 comment on #1107, and
PRs #1103, #1122, #1125, #1126.

| id | backend | location | cost | status | issue |
|---|---|---|---|---|---|
| C1 | CPU | `SimpleKvCache` F32 (default KV, draft KV) | L8B 256 KiB/token (32 GiB @128k) | open; target q8_0 + int8 QKᵀ | #978, #1120, #1133 |
| C2 | CPU | all activations F32 | L8B 256 KiB/token of capacity | open | #1138 |
| C3 | CPU | logits for all positions + output copy | L8B 501 KiB/token each (≈2 GiB @2048) | **state buffer fixed for `TransformerModel`** (#1137 PR): sized to the rows the head computes; Llama-3.2-1B Q8_0, 2071-token prompt: peak private commit 2,911 -> 904 MiB, RSS unchanged (last-row prefill never touched the pages), text identical. Still open: Qwen3HybridDense / Qwen3MoeHybrid / NemotronH / Lfm2 CPU states, the per-call `rows·V·4` result copy, CUDA G3L, Vulkan V6L | #1137 |
| C4 | CPU | `PagedKvCache` F32 blocks + full-history staging | was: blocks L8B 256 KiB/token, staging @8k 64 MiB/seq; now F16 blocks 128 KiB/token (Q8_0 68 KiB), no staging on the attention path | fixed (#1268) | #1142 |
| C5 | CPU | `QuantizedKvCache` F32 recent window | L8B W=128: 32 MiB | open | #1142 |
| C6 | CPU | RoPE cos/sin `float[]`, twice | L8B @128k 64 MiB ×2 | open | #1142 |
| C7 | CPU | MTP KV and hidden F32 | 32 MiB @4096 positions | open | #1141 |
| C8 | CPU | MLA KV, TurboQuant norms, MoE scatter F32 | per token | open | #1140 |
| C9 | CPU | recurrent state + snapshots F32 | O(1) per sequence (speculative verify: read-only scan + in-place replay of row inputs, no state copy at all, #1194) | sign-off pending | #1144 |
| X1 | CPU | safetensors BF16 → F32 | L8B +16 GB; L1B +2.5 GB | open | #1143 (needs #1118) |
| X2 | CPU | MLA-MoE routed experts → F32 | ≈57 GB on V2-Lite | open | #1140 |
| X3 | CPU | MLA projections, shared / Granite-MoE experts → F32 | `Σ out·in·4` | open | #1140 |
| X4 | CPU | R4 repack: second same-width copy | L8B Q4_K_M +4.9 GB | **fixed**: R4 is a zero-copy view; same kernels on the mmap rows, bit-identical, same speed | #1156 (from #1143) |
| X5 | CPU | TurboQuant full-history F32 scratch | L8B @128k 1 GiB | open | #1139 |
| X6 | CPU | dequant-rows prefill (Q4_0/Q4_1/Q5_1/IQ4_NL, BF16, F16, MXFP4, PQ2_0) | bounded `k·4` per worker | open (missing kernels) | #1119, #1118, #489 |
| X7 | CPU | quantized-KV tile dequant per head, allocated per call | `tile·hd·8` per head call | open | #970, #1120 |
| G1 | CUDA | F32 activation set for every model | L1B ≈587 MB @4096 | open | #1134 |
| G2 | CUDA | `DequantScratchF32` always allocated | L1B 67 MB; Qwen3-8B 201 MB | open | #1134 |
| G3 | CUDA | hybrid forward state F32 (activations, GDN, logits `rows·V·4`) | logits 970 MiB @1024 rows; +486 MiB per 512-row window | open | #1138 (activations), #1137 (logits) |
| G4 | CUDA | hybrid F32 K/V staging (tiers without flash/kvq) | B2 8 KiB/token | open | #1136 (siblings: #968) |
| G5 | CUDA | MTP head KV F32, doubled on growth | B2 16,392 B/token | **fixed on dev** (#1095, PR #1126): 1,156 B/token at q4_0 | #979 (duplicate) |
| G6 | CUDA | G3 attention F32 score matrix | 32 heads @4096 ≈3.2 GB | open | #1134 |
| G7 | CUDA | Nemotron-H KV F32; MLA latent / default MLA KV F32 | per token | open | #968 (Nemotron-H), #1140 (MLA) |
| G8 | CUDA | Gemma F32 KV round-trip + activation scratch | `2·kvLen·kvStride·4` | open | #1134 |
| G9 | CUDA | split-KV partials F32 | ≈0.8 MB | sign-off pending | #1144 |
| G10 | CUDA | recurrent state F32 | O(1) per sequence | sign-off pending | #1144 |
| GX1 | CUDA | prefill dequant → F16 scratch → cuBLAS | L1B 33.5 MB, 70B 470 MB + 2 B/weight traffic | hybrid PQ2_0 **fixed** (#1222: int8-mma W2A8 MMQ default on sm_80+); dense open | #1132 (hybrid PQ2_0), #1135 (dense) |
| GX2 | CUDA | LM head > 8 rows → whole-head F16 | B2 2.54 GB | **fixed on dev** (#1099, PR #1125): 2,425 → 170 MiB | — |
| GX2b | CUDA | multi-row LM head for non-PQ2_0 heads: tiled F16 dequant | ≤ max(layer scratch, 64 MiB) per tile | open (bounded, admitted) | #1136 |
| GX3 | CUDA | quantized-KV full-history F16 scratch | stride 1024 @32k: 128 MB | hybrid **fixed on dev** (#1094, PR #1103); dense `CudaQuantizedKvCache`/TurboQuant open | #1139 |
| GX4 | CUDA | hybrid KV F16 dequant scratch, remaining tiers | B2 4 KiB/token | open | #1136 |
| GX5 | CUDA | load-time F16 copies (no GEMV kernel), embedding, dense LM head | Qwen3-8B embedding 1.16 GiB | open (gated) | #1135 |
| GX6 | CUDA | Gemma `ProjectF32` quant → F16 → F32 | `4·maxProj` | open | #1135 |
| M1–M3 | CUDA | MTP short-history (< 256 keys) dequant + F32 staging; non-mma F16 → F32 staging; F32 K/V projection scratch | bounded (256 rows; 4 KiB/row) | open (bounded) | #1136 |
| V1 | Vulkan | dense `VulkanKvCache` F32 default | L1B 64 → 32 KiB/token (K+V) at F16, KL ~5e-4; L3B 224 → 112 KiB, KL ≤ 2e-4 (E10b) | open; F16 measured | #1151, #918 |
| V2 | Vulkan | hybrid KV F32 default | B2 128 KiB/token | **fixed on dev for B2** (#1102, PR #1122: F16 default, 64 KiB/token); q8_0 open | #1112 |
| V3 | Vulkan | inter-kernel activations F32 | `4·seqLen·(3H+2q+2kv+3I)` | open | #1138, #677 |
| V4 | Vulkan | MTP KV and hidden F32 | B2 8 → 4 KiB/token at F16, KL 3.4e-8 (E10b) | open; F16 measured invisible | #1141 |
| V5 | Vulkan | split-KV partials, split-K slabs F32 | 5.24 MB per split-K | sign-off pending | #1144 |
| V6 | Vulkan | logits `rows·V·4`; recurrent state; MLA KV; Q8_1 `(d,s)` and KV scales F32 | various | open | #1137, #1144, #1140, #1146 |
| VX1 | Vulkan | token embedding → F32 for every type but Q8_0 | Qwen3.8-27B 4.85 GiB vs 682 MiB (Q4_K); Bolt 2 GiB vs 544 MiB (Q8_0) | **fixed for Nemotron-H and qwen35 dense** (#1174: Q8_0 / Q4_K / Q6_K / PQ2_0 gathered packed); dense `VulkanWeights` K-quants and `VulkanQwen3MoeHybrid` open | #919 |
| VX2 | Vulkan | matrices outside `DeviceQuantTypeFor` → F32 | `m·k·4` | open | #921, #627 |
| VX3 | Vulkan | quantized-KV / TurboQuant F32 full-history scratch | `2·maxSeq·stride·4` | open | #1139 (hybrid: #1112) |
| VX4 | Vulkan | IQ1/2/3 (+ fallback NVFP4) prefill F16 scratch | L3B 50.3 MB + 4 B/weight traffic | open | #1146 (#1064, #1069) |
| VX5 | Vulkan | shared-expert / expert-bank F32 fallbacks; legacy MLA F32 | up to ≈265 GB if they fall through | open | #1140 |
| F374 | Vulkan | 16-bit storage / `shaderFloat16` / `shaderInt8` never enabled | — | **fixed on dev** (972d5c53, PR #1122) | #374 |
| Q1 | Vulkan qwen4exp | shared experts widened to F16 | +0.34 GiB | open | #1145 |
| Q2 | Vulkan qwen4exp | router BF16 → F32 | +0.117 GiB | open | #1145 |
| Q3 | Vulkan qwen4exp | PLE BF16 → F32 device + host copy | +0.061 GiB device, +0.122 GiB host | open | #1145 |
| Q4 | Vulkan qwen4exp | `hc_*_inject` BF16 → F32 | +7.5 MiB | open | #1145 |
| Q5 | Vulkan qwen4exp | QSA KV F32 default; indexer keys F32 | 48 vs 24 KiB/token (12 → 6 GiB @262k) | open | #1145 |
| Q6 | Vulkan qwen4exp | activation scratch F32 @1024 planned rows | 3,753 MiB | open | #1138 |
| Q7 | Vulkan qwen4exp | MTP head routed bank re-quantised Q8_0 → Q4_K/Q5_1 | smaller than file (lossy, not an expansion) | noted only | — |
| H1 | Vulkan hybrid | IQ4_NL / K-quant prefill on scalar F32 GEMMs (not an expansion; the missing MMQ/coopmat path) | Tev1 63 tok/s Strix, 3.7 tok/s Arc | open | #1149 |
| L2 | Vulkan LFM2 MoE | F32 KV (`VulkanNemotronHKvCache` F32 path) | per token | open; not measured | #1151 |

New issues filed by this study: #1134–#1146 (inventory), #1148–#1151 (Vulkan arm). Earlier study follow-ups: #1117–#1120
(CPU), #1132, #1133 (CUDA).

| issue | Vulkan-arm follow-up |
|---|---|
| #1148 | Drop the Q hi/lo split in the GQA coopmat prefill by default (KL at the noise floor, 1.46× a 128k tile; needs user sign-off) |
| #1149 | Hybrid host: IQ4_NL / K-quant prefill on scalar F32 GEMMs; use MMQ / coopmat |
| #1150 | Strix Q8_0 dense prefill: dp4a MMQ default is 1.10–1.25× slower than the F32-acc coopmat; needs an F32 reference arm |
| #1151 | F16 KV for the dense Vulkan host (F32 today) |

---

## 8. Not measured

| item | why it matters | issue |
|---|---|---|
| F16/BF16 stored activations (all backends); residual-stream max |x| per model | rule 1 for the largest per-token store after KV | #1138 |
| Recurrent state at 16 bits over ≥ 8k-token decodes; 16-bit snapshots | the one store that feeds back into itself (dense `qwen35` verify rows no longer store states at all, #1194: a 16-bit snapshot would break greedy identity, row inputs + replay do not) | #1144 |
| Split-KV O partials / split-K slabs in F16 | small bytes, needs sign-off either way | #1144 |
| F16 RoPE tables at ≥ 32k | CPU table bytes | #1142 |
| Vulkan F16 accumulation promoted every 256 K | the AMD Windows driver crashes compiling it; moot on Strix (F16-acc no faster) | — |
| Vulkan hybrid KV in q8_0 / K8V4 / q4_0 | Strix/Arc long-context KV choice | #1112 |
| Vulkan iu8 WMMA prefill vs f16 WMMA (bytes only on RDNA 3.5) | Strix Q8_0 prefill choice | #1150 |
| Vulkan path accuracy against an F32-weight reference | MMQ vs coopmat differ by 3.2e-4; which is closer is unknown | #1150 |
| LFM2 MoE Vulkan F32 KV narrowed to F16 | not on Strix | #1151 |
| ~~Fused int8-mma / promoted-F16 PQ2_0 GEMM speed (and the dp4a vs mma rate)~~ measured (#1222): Bonsai-2 pp512 157 → ~600 tok/s on the 3060 (fork 590); PPL 10.6182 vs 10.6179 (dequant + cuBLAS) | — | #1132 |
| CUDA KV formats on a Llama-class model (CPU covered Llama) | KV policy beyond Bonsai | #1133 |
| q4_0 K failure cross-checked against llama.cpp `-ctk q4_0` | rules out a dotLLM nibble-order bug | #1120 |
| Q8_K activations on massive-activation models (Qwen, Gemma) | per-256 scale outliers (E4) | #1138 |
| Real (non-simulated) CPU F16 KV store speed | E5 simulates F16 by rounding | #978 |
| `VDPBF16PS` / 512-bit VNNI on Zen 5 | preview runtime only (`dev-dotnet11`) | #1118 (mainstream kernel) |
| RTX 3060 HFMA2 rate | no kernel depends on it | — |
| Weight-format quality (CPU Q4_0/Q4_K_M were requantized from Q8_0) | activation comparisons are unaffected | — |
| FP8 / MXFP4 compute | no FP8 hardware on any target device; FP8 storage offers nothing over q8_0 | — |
| Disk ↔ device transfer paths (rule 3) | outside this study | — |
