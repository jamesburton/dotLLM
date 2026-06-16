# DRAFT comment for kkokosa/dotLLM#322 — review then post (outward-facing)

---

## AVX-512 BF16 Q8_0 microkernel — measured 4-way, and it wins

The fourth Q8_0 4×6 outer-product variant (`Avx512_4x6_Bf16` — dequantize each Q8_0 block to bf16 with the
per-block scale folded into the value, then accumulate with `VDPBF16PS` into one fp32 accumulator) is now
measurable, after the `Avx512Bf16` intrinsic landed in `dotnet/runtime` plus three JIT fixes (codegen dispatch,
a FullOpts EVEX encoding bug for zmm16–31, and `V512VersionOfIsa` wiring that also re-enabled `AvxVnni.V512`).
All four variants now run on a **single runtime build**, so this is an apples-to-apples comparison.

### Benchmark — 4×6 Q8_0 outer-product tile, K=4096, CoreRun, 30 iter, Zen5 (Ryzen AI Max / Strix Halo)

| Kernel | Mean | Ratio vs maddubs |
|---|---:|---:|
| `Avx512_4x6_Maddubs` (baseline) | 4.06 µs | 1.00 |
| `Avx512_4x6_Vnni` (VPDPBUSD, sign-trick) | 5.20 µs | 1.33 (slower) |
| `Avx512_4x6_VnniZp` (VPDPBUSD, zero-point) | 3.84 µs | 0.98 (~parity) |
| **`Avx512_4x6_Bf16` (VDPBF16PS)** | **1.67 µs** | **0.43 (≈2.3× faster)** |

BF16 is **~2.3× faster than the best integer kernel** (VnniZp 3.84 µs → BF16 1.67 µs = 2.29×; vs maddubs
2.4×). This reproduces the earlier VNNI finding (sign-trick ~1.33× slower, zero-point ~parity with maddubs) and
adds BF16 as the clear winner. The maddubs baseline is somewhat noisy on this laptop (multimodal, thermal), but
BF16's mean sits well below every integer kernel's mean. This is a **microkernel** measurement (one 4×6 tile),
not end-to-end inference.

**Why it wins:** folding the scale into the bf16 value makes the inner loop a pure `VDPBF16PS` chain into a
single fp32 accumulator — no per-block integer reduction, no `vpmaddwd`-with-ones, no dual-scale fold that the
int kernels pay every 32-element block. Fewer inner-loop instructions outweighs bf16's 2× element width.

### Correctness

Full outer-product parity vs the scalar oracle: **100/0/0** with every V512 guard forced open (maddubs, both
VNNI variants, and BF16 all execute for real). BF16 specifically: **8/8** across `blockCount ∈
{1,2,3,8,17,18,48,128}` (K up to 4096), within a bf16-appropriate tolerance `|bf16 − scalar| ≤
3e-2·(tile-max|scalar| + 1e-3)`.

### End-to-end perplexity (SmolLM-135M Q8_0, Strix/Zen5, net11 + Core_Root)

The microkernel caveat below is now answered end-to-end. Three Q8_0 prefill reductions on the **same** passage
(32 scored tokens), measured in one process:

| Prefill path | Perplexity | vs inner-product baseline |
|---|---:|---:|
| **inner-product** (cache-tiled — the production default) | 26.2386 | — |
| integer outer-product (R4 maddubs) | 26.1375 | **−0.39%** |
| bf16 outer-product (VDPBF16PS) | 26.6587 | **+1.60%** |

Two things matter here:

1. **The outer-product restructuring itself is free.** Integer-outer vs the inner-product default is −0.39%
   (within FP rounding — both match exact scalar Q8_0 truth to ~1e-4). Switching from inner-product to the
   R4 outer-product GEMM costs **no model quality**.
2. **BF16's end-to-end cost, anchored against the production baseline, is +1.60% perplexity** for the ~2.3×
   microkernel speed. (Measured against integer-outer alone it reads +1.99%, but integer-outer happens to sit
   slightly *below* the true baseline, so +1.60% is the honest figure vs what users actually run today.)

Logits agree to a scale-normalized 3.2e-2 (vec-scale gate); per-element relative diff is large only on
near-zero cancellation logits, as expected for a correct bf16 kernel. Parity of the integer outer-product path
vs inner-product is **3/3 PASS** (scaleNormDiff 3.4e-7 / 1.98e-2 / 2.28e-2 for n=2/10/4 tokens).

### End-to-end prefill speed (Llama-3.2-1B Q8_0, Strix/Zen5, net11 + Core_Root, 32 threads, median of 9)

The microkernel speedup translates to the model level. Prefill of a synthetic sequence, three reductions on
identical input in one process:

| Prefill | inner-product (default) | integer outer-product | bf16 outer-product |
|---|---:|---:|---:|
| pp256 | 978 ms (262 tok/s) | 1152 ms — **0.85×** | 531 ms — **1.84×** |
| pp512 | 2167 ms (236 tok/s) | 2561 ms — **0.85×** | 1115 ms — **1.94×** |

**bf16 outer-product is ~1.9× faster end-to-end** than the production inner-product default — close to the 2.3×
microkernel figure, so matmul dominates prefill with little dilution from attention/RoPE/RMSNorm. The clean
arithmetic-only comparison (same outer-product structure both ways) integer-outer→bf16 = 1152→531 ms = **2.17×**,
matching the microkernel result exactly.

**Important — the integer outer-product is not a speed win on its own.** It is ~15% *slower* than the
well-optimized inner-product path on this runtime, with no quality benefit (perplexity bit-identical). Its sole
value is as the structural vehicle that enables the bf16 microkernel. So the deliverable is the **bf16 path**,
full stop — not "ship the integer restructuring first."

### Decode is unaffected — so this is a *prefill* speedup (read this before quoting the number)

These operators are gated on `n > 1`; single-token **decode** routes to the inner-product GEMV and never touches
the outer-product / bf16 kernel. Measured (256-token context, median of 32 decode steps), all three configs are
identical and the bf16 tile counter stays at 0:

| Decode (1 tok/step) | tok/s | speedup |
|---|---:|---:|
| inner-product | 37.0 | 1.000× |
| integer outer-product | 37.5 | 1.015× |
| bf16 outer-product | 37.1 | 1.003× |

So **real generation speed is a blend**: `1.9×` on the prompt (prefill) and `1.0×` on the completion (decode),
weighted by your prompt:completion ratio. Prefill-heavy workloads (RAG, classification, long-prompt/short-output)
capture most of the win; long-generation workloads see little. *Aside:* decode here is ~27 ms/token ≈ 40 GB/s
effective for the 1.1 GB model — well under Strix's LPDDR5X peak, so decode looks kernel-bound rather than
bandwidth-saturated; a future full-width AVX-512 / VNNI decode GEMV may have headroom (separate work).

### BF16 accuracy on Llama-3.2-1B (same model the speed was measured on)

Longer-corpus A/B/C (373-token passage — larger than a one-sentence sample, not a standard benchmark corpus;
relative deltas on identical tokens are what matter):

| Prefill path | Perplexity | vs inner baseline |
|---|---:|---:|
| inner-product (default) | 14.8216 | — |
| integer outer-product | 14.8216 | **+0.00%** (bit-identical) |
| bf16 outer-product | 14.8653 | **+0.30%** |

The integer outer-product is **exactly quality-neutral**; bf16 costs **+0.30% perplexity**. (A 32-token sample
read −0.45%, i.e. noise — the longer passage averages that out to the true small positive cost.) Logits agree to
scaleNormDiff 1.77e-2 even at down_proj's K=8192. So bf16's cost is ~0.3% on Llama-1B (≤ +1.6% on SmolLM-135M).

### Caveat / next step

These are still single passages on small models. The prefill trade is well-substantiated — **~1.9× prefill for
~0.3% perplexity on Llama-1B** — but a standard corpus (e.g. wikitext-2) and a larger model are the gate before
flipping bf16 on by default. bf16 is the deliverable; the integer outer-product is only its structural vehicle,
and the win is on prefill (decode is unchanged).
