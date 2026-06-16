# AVX-512 "fully unlock" — audit, decode refinement, and plan (hand-back)

**Date:** 2026-06-13 · **Context:** follow-up to the bf16 Q8_0 prefill work (#322). Audits the CPU backend
for AVX-512 / VNNI / bf16 coverage gaps after the runtime saga unlocked `Avx512Bf16` and `AvxVnni.V512`,
and resolves the "decode has headroom" refinement. **No Strix runs yet — awaiting confirmation Strix is free.**

---

## TL;DR (the honest headline)

1. **Most of the inference hot path is *already* AVX-512** — softmax, RMSNorm, SiLU/SwiGLU/GeGLU, residual
   add/mul, attention score·V and the online-softmax exp kernel all run through `TensorPrimitives` or an
   explicit `Avx512F` path, so they auto-use 512-bit lanes on Zen5 today. **"Fully unlocking" is largely
   already done for everything except the integer quant kernels.**
2. **The decode "headroom" refinement is CONFIRMED WRONG (probe ran 2026-06-14).** Decode scales 2→8T
   (1.91×, 1.36×) then plateaus hard 8→16T (1.03×); 4× the threads past 8T buys only 1.19× — **bandwidth/
   latency-bound, not compute-bound.** A wider/bf16 decode GEMV cannot help a non-arithmetic bottleneck.
   **Decode-AVX-512 is closed — nothing to implement.** (My earlier "40 GB/s ≪ 256 GB/s peak" was an invalid
   comparison — 256 GB/s is unified iGPU+CPU theoretical, and that was a 2–8-thread result vs 32-core peak.)
   The probe did surface a *config* finding: the decode-thread cap of 8 leaves ~19% on the table at 32T
   (45.6 vs 38.4 tok/s), and the old "SpinWait collapses at 32T" note did NOT reproduce — a threading
   investigation, not an ISA one (see §2).
3. **VNNI is demoted.** The prefill 4-way microbench already measured VNNI (VPDPBUSD) at *parity-or-worse*
   with maddubs on Zen5; bf16 won by collapsing instruction count, not by width. So "add VNNI to the integer
   GEMVs" — the audit's top mechanical suggestion — is **not worth it on this hardware**.
4. **The only genuinely-new arithmetic kernel worth even an experiment is a bf16 decode GEMV — and only if
   the probe shows decode is compute-bound** (unlikely). It would also compound bf16 rounding over a full
   generation, so it needs its own perplexity re-measure.

Net: there is **little real AVX-512 work left to do**, and the highest-leverage decode question is a
*threading/quantization* question, not an ISA-width one. Details and the decision points are below.

---

## 1. What is already AVX-512 (no work needed)

All via `System.Numerics.Tensors.TensorPrimitives` (width-agnostic, uses AVX-512 when present) or an explicit
`Avx512F` branch:

| Op | Where | Mechanism |
|---|---|---|
| Softmax (full + fast) | `Softmax.cs`, `FastMath.ExpSumAndStore` | TensorPrimitives + explicit `Avx512F` exp kernel |
| RMSNorm | `RmsNorm.cs` | TensorPrimitives |
| SiLU / SwiGLU / GeGLU-tanh / ReLU² | `SiLu.cs`, `FusedOps.cs`, `ReluSquared.cs` | TensorPrimitives (tiled) |
| Residual add / elementwise mul | `Add.cs`, `Multiply.cs` | TensorPrimitives |
| Attention Q·Kᵀ, softmax·V, soft-cap | `Attention.cs` (incl. tiled decode path) | TensorPrimitives `Dot`/`MultiplyAdd` |
| MoE final accumulate, MLA matvec | `MoeSwiGluMlp.cs`, `MlaAttention.cs` | TensorPrimitives |
| f32 GEMV, KV→Q8_0 quantize | `MatMul.GemvF32`, `QuantizeF32ToQ8_0Avx512` | TensorPrimitives / explicit `Avx512F` |

**Implication:** the only hot path NOT auto-scaling to AVX-512 is the hand-written **integer quant matmul /
GEMV** kernels and a few **quantize helpers** — that's the entire remaining surface.

---

## 2. The decode refinement — resolved by reading the code; probe built to confirm

### Why "decode has headroom" is almost certainly wrong
- The decode Q8_0 GEMV is 256-bit maddubs (even inside the methods named `*Avx512`), no VNNI, no bf16. True.
- But decode throughput is gated *before* arithmetic by **two measured limits already encoded in the code**:
  1. **Bandwidth:** `ThreadingConfig.DecodeThreadCount` doc — *"Decode is memory-bandwidth-bound, so more
     threads than memory channels don't help"*; `ComputeThreadPool` caps decode at the memory-channel
     estimate (Strix single-node ⇒ 2) or `min(8, threadCount)`.
  2. **Dispatch coordination:** `ComputeThreadPool` (lines ~92–99) — SpinWait dispatch *collapses* at 32
     threads: the 30-dispatch decode burst measured **10.6 ms at 32T vs 32 µs at 8T**
     (`.perf-runs/.../dispatch-microbench.md`).
- My "40 GB/s" was a 2–8-thread decode result compared against 32-core aggregate peak — invalid. A bf16/VNNI
  GEMV streams the same 1.1 GB of Q8_0 weights/token; if the bottleneck is the load, faster MACs do nothing.

### Probe RESULT (ran on Strix net10, 2026-06-14 — `Llama32DecodeRooflineProbe`)

| decode threads | ms/tok | tok/s | scaling vs 2T |
|---:|---:|---:|---:|
| 2  | 67.5 | 14.8 | 1.00× |
| 4  | 35.4 | 28.3 | 1.91× |
| 8  | 26.0 | 38.4 | 2.60× |
| 16 | 25.2 | 39.7 | 2.68× |
| 32 | **21.9** | **45.6** | 3.08× |

**Verdict: bandwidth/latency-bound past ~8T — NOT compute-bound — so a wider/bf16 decode GEMV is confirmed
not worth it.** Step scaling: 2→4 = 1.91× (near-linear), 4→8 = 1.36×, 8→16 = **1.03× (hard plateau)**,
16→32 = 1.15×. Past 8 threads, 4× the threads buys only 1.19× throughput — decode is clearly traffic/latency
limited there, not arithmetic-limited. Faster MACs (bf16/VNNI/512) cannot help a bottleneck that isn't
arithmetic. **The decode-AVX-512 question is closed: nothing to implement.**

**Two surprises worth noting (config, not kernels):**
1. **The current decode-thread cap (8, or memory-channel=2) leaves a real ~19% on the table on this 32-core
   host:** 32T = 45.6 tok/s vs 8T = 38.4 tok/s. Raising the cap is a *config* win, not a kernel. Modest and
   free — but see #2 before changing the default.
2. **The probe CONTRADICTS the `dispatch-microbench.md` "SpinWait collapses at 32T" finding** — here 32T is
   the *fastest*, not collapsed. Likely because the real decode forward dispatches large matmul slices (~112
   dispatches/token of real work) that amortize coordination, vs the near-empty dispatches the old microbench
   timed. **Reconcile this before raising the default cap** — the collapse may still bite on smaller models /
   shorter rows where per-dispatch work is tiny. This is a threading investigation, separate from AVX-512.

(Original pre-probe reasoning kept below for the record.)

#### Pre-probe expectation (the knee is the answer):
- **Plateau by ~4–8T ⇒ bandwidth/dispatch-bound** → ISA-width kernels are wasted effort on decode; the lever
  is *bytes-moved* (lower-bit weight quant, KV-cache quant) — a different project. **Expected outcome.**
- **Monotonic climb toward 32T ⇒ compute-bound** → the decode-thread cap is too conservative on this host
  (**a one-line config change, not a kernel**), and *then* a bf16 decode GEMV becomes worth a microbench.

This converts the decode question from "write speculative AVX-512 kernels" to "run one cheap, decisive probe."

---

## 3a. STATUS — minor-gap batch IMPLEMENTED + validated (2026-06-14)

The minor gaps below were batched and validated byte-exact vs scalar on real AVX-512 hardware (Strix),
73/73 unit tests, 0 failures. Each kernel uses cross-platform `Vector512<T>` where possible, raw `Avx512F`
only where byte-exactness requires it (MXCSR-rounding `VCVTPS2DQ`; order-preserving `VPMOVSDB` narrow, which
sidesteps the AVX2 `PackSignedSaturate`+`PermuteVar8x32` lane dance). Each has a discriminating
`*_Avx512_MatchesScalar` test.

- ✅ wave 1 (`519a4bb`): `KvQuantize.Q8_0ToF32`, `KvQuantize.F32ToQ4_0`, `FusedOps.RmsNormQuantizeQ8_0`, `RoPE.ApplyRotationNeoX`
- ✅ wave 2 (`96a082f`): `KvQuantize.Q4_0ToF32`, `FusedOps.RmsNormQuantizeQ8_1`, `FusedOps.RmsNormQuantizeQ8_K`
- ⏸️ **descoped:** interleaved `RoPE.ApplyRotation` (512-bit deinterleave) — highest-risk, lowest-value
  (latency-bound ~1.3×). Left on the AVX2 path; revisit only if RoPE shows up in a profile.

These are correctness-complete; their end-to-end perf impact remains small by design (they're slivers next to
weight streaming) — the point was uniform AVX-512 coverage, not a throughput win.

## 3. Remaining real AVX-512 gaps, ranked (all low end-to-end value)

These are genuine 256-bit-only hot-path kernels, but each is a *sliver* of per-token cost next to weight
streaming, so even a clean 2× on any one is small end-to-end. Ranked by plausibility of mattering:

| # | Kernel(s) | File | Status | Honest expected value |
|---|---|---|---|---|
| 1 | `FusedOps.RmsNormQuantizeQ8_0 / Q8_1 / Q8_K` | `FusedOps.cs` | AVX2 only, no 512 branch | Runs every layer/decode to prep activations; small vs the GEMV it feeds. ~1.5–2× on a tiny slice ⇒ low single-digit % at most. |
| 2 | `KvQuantize.Q8_0ToF32 / F32ToQ4_0 / Q4_0ToF32` | `KvQuantize.cs` | AVX2 only | **Only matters at long context** (cost ∝ KV length). Negligible at short ctx; worth revisiting for long-context decode. |
| 3 | `RoPE.ApplyRotation / ApplyRotationNeoX` | `RoPE.cs` | AVX2 only | Latency-bound on small headDim; ~1.3–1.5× on the rotate loop ⇒ marginal end-to-end. NeoX port is mechanical. |
| 4 | Integer GEMV width upgrade (Q8_0/Q5_0/Q4_K/Q5_K/Q6_K) — full-512 or VNNI | `MatMul.cs`, `MatMulQ5_0.cs`, `MatMulKQuants.cs` | 256-bit maddubs; K-quants have no 512 path at all | **Demoted.** VNNI=parity on Zen5 (measured); decode bandwidth-bound (probe pending). Not worth it unless the probe surprises. Also: no 512-bit `VPSIGNB`, so an exact 512 path needs VNNI-zeropoint or sxbw+pmaddwd — extra complexity for a doubtful win. |
| 5 | `MlaAttention` scalar helpers (softmax/RMSNorm/RoPE) | `MlaAttention.cs` | Scalar | Fix = *call the existing vectorized kernels*, not new intrinsics. Only matters for DeepSeek-V2/V3 MLA models. Cheap correctness/perf cleanup, not an "AVX-512" task. |

Note bf16/load-time dequant (all `Dequantize*.cs`, `WeightRepacking`) is **cold** (runs once at model load) —
ISA width is irrelevant there; excluded.

---

## 4. What I built this round (compiles locally; AVX-512 paths need Strix to run)

Local box is AVX2 + AVX-VNNI(256), **no AVX-512** — so scalar/AVX2 correctness is locally testable, but the
512-bit/bf16 perf must run on Strix.

- **`Llama32DecodeRooflineProbe.cs`** — the decode thread-scaling probe (section 2). The decisive next step.
- (Prior, already landed) `Llama32PrefillOperatorBenchmark.cs` now also measures decode tok/s across the 3
  flag configs (the blend denominator); `Llama32Bf16PerplexityCorpusTests.cs` (longer-corpus accuracy).

No speculative AVX-512 kernels were written — per the evidence above they'd likely be wasted effort, and
they can't be validated locally. They're gated behind the probe result.

---

## 5. Decision points for you (the hand-back)

1. ✅ **Decode roofline probe — DONE (2026-06-14).** Bandwidth/latency-bound past ~8T (8→16T = 1.03×).
   **Decode-AVX-512 is closed: no arithmetic kernel will help.** The decode lever is *bytes-moved* — lower-bit
   weight quant and KV-cache quant — a separate track, not AVX-512.
2. **Config follow-up (optional, ~19% decode win, NOT a kernel):** decode at 32T (45.6 tok/s) beats the
   current 8-cap (38.4 tok/s). Raising `DefaultDecodeThreadCountCap` / the memory-channel clamp would capture
   it — BUT first reconcile why the probe's 32T didn't hit the `dispatch-microbench.md` "collapse at 32T"
   (likely real matmul slices amortize dispatch vs the old near-empty microbench; the collapse may still bite
   tiny-row models). This is a **threading** investigation; hand it to whoever owns `ComputeThreadPool`.
4. **The minor gaps (§3 #1–#3)** are individually low-value. Worth doing only as a batch if you want the CPU
   backend uniformly AVX-512-clean, or specifically #2 (KvQuantize) if long-context decode becomes a target.
5. **VNNI across the integer GEMVs (§3 #4)** — recommend NOT pursuing on Zen5 given the parity measurement,
   unless a different target CPU (e.g. an Intel SPR/GNR box) is in scope, where VPDPBUSD economics differ.

**Bottom line to hand back:** the "fully unlock" surface is much smaller than it first looked — the hot path
is already AVX-512 — and the decode refinement most likely resolves to "bandwidth-bound, nothing to unlock
with ISA width." Confirm with the one probe before investing in any kernel.
