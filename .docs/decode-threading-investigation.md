# Decode Thread-Scaling Anomaly — Investigation & Recommendation

**Date:** 2026-06-14
**Author:** investigation agent (read-only; no source modified)
**Worktree investigated:** `C:\Development\dotllm-bf16e2e` (read-only)
**Subject:** Why a Llama-3.2-1B decode probe shows 32 threads fastest (45.6 tok/s) while the codebase deliberately caps decode threads at 8.

---

> ## ⚠️ ADDENDUM 2026-06-14 — e2e measurement OVERTURNS the recommendation below
> The §5 validation plan was executed (multi-model sweep + an edge discriminator, net11+Core_Root, 32-core Zen5).
> **The reconciliation hypothesis in §2 was wrong, and Options A/C below are dead.** Findings:
> - **The knee is NOT model-size-dependent.** SmolLM-135M, Bielik-1.5B and Llama-3.2-1B ALL scale to ~24–30
>   decode threads (1.5–1.9× over cap-8). SmolLM does **not** plateau by 8T e2e — its "plateau/collapse" in the
>   2026-04-24 synthetic dispatch microbench was an artifact of that benchmark having zero useful work between
>   dispatches. So **Option A (work-size-adaptive gate) solves a non-problem — NOT built.**
> - **The 32T "collapse" is intermittent OS oversubscription, not a hard cliff and not cache contention.** The
>   edge probe (SmolLM ctx128 {24,28,30,31,32}×4) shows 32T usually ~460 tok/s with an occasional catastrophic
>   stall (15.3); medians are flat 28→32T. The catastrophic tail is isolated to the full-core count (spin-wait
>   workers on every core starve OS/GC). Because it is oversubscription (not contention), **Option C (cache-line
>   padding) has no target — NOT built.**
> - **Actual fix (implemented, commit `5f79bb7`):** auto decode cap = `threadCount - 2` (leave OS headroom),
>   floored at 2, topology-independent. This also retires the cap-2 pinning footgun by construction (the separate
>   minimal footgun fix `21b06a4` is subsumed). Validated on one 32-core single-node Zen5; ≥64-core/multi-socket
>   untested. The §3 "flat raise regresses small models" risk did not materialise — small models gain too.
> The analysis below is preserved as the pre-measurement reasoning; treat §2/§4 recommendations as superseded.

---

## TL;DR

The cap-at-8 default and the "collapse at 32T" comment are **both correct — for the model they were tuned on (SmolLM-135M)**. The 2026-04-24 microbench (`.perf-runs/cross-lib-20260424/dispatch-microbench.md`) and the end-to-end profile in the same folder measured **SmolLM-135M**, whose decode matmul rows are tiny (~576–1536 elements). At that work size, SpinWait dispatch coordination cost dwarfs the per-thread compute, and at 32 threads cache-line contention on `_dispatchGeneration` / `_completion` makes it collapse (10.6 ms per 30-dispatch burst vs 32 µs at 8T). The `DefaultDecodeThreadCountCap = 8` was added *as the fix* for that finding.

The new probe (`Llama32DecodeRooflineProbe.cs`) measures **Llama-3.2-1B**, whose per-dispatch matmul work is ~15–20× larger per thread. At that work size the coordination cost is fully amortized, so 32T wins. **The contradiction is resolved by model/work size, not by "microbench vs real model"** — the collapse is real in a real model (the SmolLM end-to-end run burned 76% of CPU in the spin loop).

Two practically important facts:
1. The default production path passes **`topology: null`** to `ComputeThreadPool` (NUMA/P-core pinning are off by default), so the real-world decode cap is **8**, not the memory-channel estimate of 2. The probe's 32T figure (45.6) vs the default-8 figure (38.4) means the realistic loss today is **~19%**, not 3×.
2. The 3× loss (cap = 2 → 14.8 tok/s) only bites if the user passes `--numa-pin` or `--pcore-only`, which is the *only* path that builds a `NumaTopology` and therefore the only path that drives the cap down to `MemoryChannelEstimate = 2` on Strix.

Recommendation: **do not flat-raise the cap.** Make it work-size-adaptive (gate on per-dispatch `M·K/threads`), keep cap-8 behavior as the small-model floor, and treat the cache-line root-cause fixes as the durable win. Re-measure SmolLM-135M at 30T vs 32T under the *production config path* before changing the default.

---

## 1. Findings (code trace)

### 1.1 Dispatch-mode selection on the decode path

`src/DotLLM.Models/Architectures/TransformerModel.cs:420`:

```csharp
// Adaptive dispatch mode: spin-wait for decode (short, frequent dispatches),
// event-based for prefill (long dispatches where kernel transition cost is negligible).
_threadPool?.SetDispatchMode(seqLen == 1 ? DispatchMode.SpinWait : DispatchMode.EventBased);
```

So **single-token decode uses `DispatchMode.SpinWait`** — exactly the mode the microbench found collapses at 32T. (The batched path at `TransformerModel.cs:1372` forces `EventBased` because a batch is "prefill-shaped".)

### 1.2 How SpinWait reduces active workers to the decode cap

`src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:144-151`:

```csharp
public void SetDispatchMode(DispatchMode mode)
{
    _currentMode = mode;
    _activeWorkerCount = mode == DispatchMode.SpinWait
        ? Math.Clamp(_decodeThreadCount - 1, 1, _threadCount - 1)
        : _threadCount - 1;
}
```

In decode, `_activeWorkerCount` is clamped to `_decodeThreadCount - 1` (the −1 is because the caller acts as thread 0). Every `Dispatch` then drives exactly `_decodeThreadCount` threads (`ComputeThreadPool.cs:169-192`). The decode thread count is therefore the lever the probe sweeps.

### 1.3 How `_decodeThreadCount` is computed

`src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:99-104`:

```csharp
const int DefaultDecodeThreadCountCap = 8;
_decodeThreadCount = config.DecodeThreadCount > 0
    ? Math.Clamp(config.DecodeThreadCount, 2, threadCount)
    : topology is not null
        ? Math.Clamp(topology.MemoryChannelEstimate, 2, threadCount)
        : Math.Clamp(DefaultDecodeThreadCountCap, 2, threadCount);
```

Three branches:
- **Explicit** `DecodeThreadCount > 0` → clamp to `[2, threadCount]`. **This is the probe's path** (it sweeps 2/4/8/16/32 by setting `DecodeThreadCount` explicitly).
- **Topology present** → `MemoryChannelEstimate`. On Strix (single NUMA node) `NumaTopology.MemoryChannelEstimate` = 2 (`NumaTopology.cs:112`: `NumaNodeCount > 1 ? 4*N : 2`). → cap **2**.
- **No topology** → `min(8, threadCount)` = **8**.

### 1.4 The critical production-path fact: topology is null by default

`src/DotLLM.Models/Architectures/TransformerModel.cs:148-167`:

```csharp
ComputeThreadPool? pool = null;
if (threading.IsParallel)
{
    int effectiveThreads = threading.EffectiveThreadCount;
    if (threading.EnableNumaPinning || threading.EnablePCorePinning)
    {
        var topology = NumaTopology.Detect();
        ...
        pool = new ComputeThreadPool(effectiveThreads, topology, threading);
    }
    else
    {
        pool = new ComputeThreadPool(effectiveThreads, topology: null, threading);   // ← default path
    }
}
```

`NumaTopology` is **only built when NUMA or P-core pinning is explicitly enabled**. CLI defaults (`src/DotLLM.Cli/Commands/RunCommand.cs:85-93`) are `--numa-pin false`, `--pcore-only false`, `--decode-threads 0`. So the **default decode cap on every CPU host is 8** (branch 3), and the memory-channel=2 clamp (branch 2) only activates when a user opts into pinning.

Mapping to the probe table on Strix:
| decode threads | tok/s | which production config reaches it |
|---|---|---|
| 2  | 14.8 | `--numa-pin`/`--pcore-only` (topology built → cap 2) |
| 8  | 38.4 | **default** (no pinning → cap 8) |
| 32 | 45.6 | only the probe's explicit `--decode-threads 32` |

So today's default leaves **~19% on the table** vs 32T; a user who turns on pinning to "go faster" actually gets **3× slower** decode — a sharp, surprising footgun.

### 1.5 Matmul dispatch + the existing single-thread gate

GEMV/GEMM call sites dispatch through the pool but already short-circuit tiny problems. `src/DotLLM.Cpu/Kernels/MatMul.cs:2292`: `private const int ParallelMinRows = 32;` and e.g. `MatMul.cs:2570`:

```csharp
if (pool is null || m < ParallelMinRows) { GemvQ8_0(...); return; }   // single-thread, no dispatch
```

This gate is on **row count (M) only**, not on total work `M·K` or per-thread work `M·K/threads`. It is enough to keep a 32-row projection off the pool but does nothing to stop a 1536-row × narrow-K decode matmul from being split into sub-microsecond per-thread chunks across 32 threads — exactly the regime the microbench flagged as a net loss (its "Option B", `dispatch-microbench.md:100`).

The GEMM worker (`MatMul.cs:2454-2508`) does a 2D (row × token) partition. For single-token decode (N=1) this degenerates to a pure row partition, so per-thread work is `M/threads` rows. SmolLM Gate/Up M≈1536; at 32T that's ~48 rows/thread of ~576-element dot products — sub-microsecond, below the dispatch cost. Llama-1B Gate/Up M≈8192, K≈2048; at 32T that's ~256 rows/thread of 2048-element dots — comfortably above dispatch cost.

### 1.6 Why SpinWait collapses at 32T (from the microbench)

`.perf-runs/cross-lib-20260424/dispatch-microbench.md` (SmolLM-135M decode matmul sizes, Strix 32T):

- Per-dispatch no-op cost: 1,042 ns @ 8T → **333,405 ns @ 32T** (SpinWait).
- 30-dispatch burst: 32,705 ns @ 8T → **10,627,799 ns @ 32T** (SpinWait).
- Root causes (microbench §"Why SpinWait degrades so badly at 32T"): cache-line contention on `_dispatchGeneration` (31 workers `Volatile.Read` one int; caller's increment invalidates 31 cores), contended `_completion.Signal()` (single interlocked counter), OS oversubscription (32 logical threads on 32 cores + harness/OS), and false sharing of `_activeWorkerCount`/`_dispatchThreadCount` on the same line as `_dispatchGeneration`.

The matching end-to-end profile (`.perf-runs/cross-lib-20260424/README.md`) shows the **real SmolLM-135M server run at Auto→32T spending ~76% of CPU in `WorkerLoop` spin** — confirming the collapse is not a microbench artifact; it is the actual production pathology *for that model*.

**The cap-8 default was added as the fix for exactly this finding** — this is sourced, not inferred. The microbench's Implications #1 (`dispatch-microbench.md`) documents the *pre-fix* behavior: with no `NumaTopology` provided (the common path), `_decodeThreadCount` "falls back to `threadCount` (32), so SpinWait runs on all 31 workers," and recommends it "should instead default to `min(8, threadCount)`." The current code (`ComputeThreadPool.cs:99-104`) implements exactly `min(8, threadCount)`. So that README's 8.66 tok/s SmolLM run, which shows the 32T collapse, *predates the cap* — the cap is what stops a default SmolLM run from reproducing it today.

---

## 2. Reconciliation hypothesis

**The discriminator is per-dispatch work size, which is set by the model — not microbench-vs-real-model.**

The pool's coordination cost per decode dispatch is roughly fixed (~1 µs at 8T, exploding to ~300 µs at 32T once cache-line contention dominates). Whether parallelism is a win depends on whether the *useful per-thread compute* exceeds that overhead:

- **SmolLM-135M:** decode matmul rows ≈ 576–1536. Split across 32 threads → tens of rows of short dots per thread → **sub-µs useful work per thread**. Coordination cost > compute at every count ≥4 (microbench table). At 32T the spin loop additionally collapses under cache contention, so 32T is catastrophic. Cap-8 keeps it in the "merely bad, not catastrophic" regime, and is the right defensive default *for this size*.

- **Llama-3.2-1B:** decode matmul M is ~5–8× larger and K is ~3.5× larger (hidden 2048 vs 576; intermediate 8192 vs 1536). Per-thread useful work at 32T is ~256 rows × 2048-element dots ≈ tens of µs — **an order of magnitude above the per-dispatch coordination cost.** The fixed coordination overhead is now amortized, so adding threads keeps paying off and 32T is fastest. The cache-contention collapse seen on SmolLM does not dominate because each thread spends far longer in the kernel between generation-counter reads, reducing the contention frequency on `_dispatchGeneration`.

Note the amortization argument is strongest for the **FFN/QKV matmuls**, which dominate decode FLOPs and grow directly with hidden/intermediate size. The **attention dispatch** remains small-work even on Llama-1B at short context (≈1 head/thread × ~256 dots at 256 ctx) and runs on the same SpinWait primitives — so "Llama-1B scales to 32T" should not be read as "every Llama-1B decode dispatch is immune." This is exactly why context-length is in the validation plan (§5): longer context grows the attention dispatch relative to the matmul dispatch and can move the knee.

This cleanly explains all the data:
- Why the cap exists (correct for SmolLM, the model it was benchmarked on).
- Why the probe contradicts it (Llama-1B is 15–20× more work per dispatch).
- Why the probe's own `DecodeThreadCount=32` path "didn't reproduce the collapse": it differs from the production default (which would cap at 8) **and** runs a model large enough that the collapse never triggers in the first place.

**Caveat on "compute-bound."** The probe's interpretation guide says "monotonic climb toward 32T ⇒ compute-bound." The actual curve is **not** a clean climb: 8→16 is a plateau (38.4→39.7, +3%), then 16→32 jumps (+15%). That shape is equally consistent with **bandwidth-bound but needing more cores to saturate memory-level parallelism** (e.g. 16T fits one Zen5 CCD; 32T spans both CCDs + SMT and unlocks more outstanding loads). The probe alone **cannot distinguish** compute-bound from bandwidth-bound-needs-more-cores. The config conclusion ("32T helps Llama-1B decode on this host") holds regardless; but this curve does **not** justify building a wider/bf16/VNNI decode GEMV — that only follows from a true compute-bound finding, which is not established here.

---

## 3. Risk analysis — what breaks if we just raise `DefaultDecodeThreadCountCap`

Raising the flat default (8 → higher, or → threadCount) would help Llama-1B-and-larger on big-core hosts but **regress exactly the cases the cap was built to protect**:

1. **Small models (the original target).** SmolLM-135M, Qwen2.5-0.5B, TinyLlama at decode would go back to the 32T SpinWait collapse — the 10.6 ms/burst, 76%-CPU-in-spin regime. A flat raise silently undoes the 2026-04-24 fix. This is the dominant risk.

2. **Short rows / narrow projections generally.** Any model+op where decode per-thread work is sub-µs (small hidden size, MQA/GQA with tiny KV projections, LoRA delta GEMVs) sits in the loss regime independent of total model size.

3. **Other CPUs.** Hosts with *fewer* cores than Strix never hit 32T contention but also gain little; hosts with *more* cores (Threadripper/EPYC, 64–128T) would hit the collapse *harder* and *sooner* — the contention is superlinear in worker count. A flat "raise to threadCount" is worst on the biggest machines.

4. **Real NUMA / multi-socket.** On a 2-socket box, `MemoryChannelEstimate` = `4·N` = 8, and crossing sockets with 32+ spinning workers adds cross-node cache-line traffic on `_dispatchGeneration`. The current memory-channel clamp is *protective* there; a flat raise removes that protection.

5. **The pinning-path footgun (orthogonal but should be fixed alongside).** Today `--numa-pin`/`--pcore-only` on Strix silently drops decode to cap 2 (14.8 tok/s, 3× slower) because single-node `MemoryChannelEstimate` = 2. Whatever policy is chosen, the topology branch should not produce a *lower* cap than the no-topology branch for the same host. This is arguably a bug worth its own fix regardless of the cap decision.

**Conclusion: a flat raise trades one model's win for another model's regression. An adaptive policy is warranted.** The right axis is per-dispatch work size (which the call sites already know — they have M and K), not a single global constant.

---

## 4. Recommendation

Prefer the **smallest change that captures the Llama-1B win without regressing SmolLM**, layered:

### Option A (recommended primary) — work-size-adaptive dispatch gate
Generalize the existing `ParallelMinRows` (`MatMul.cs:2292`) gate from "M < 32" to a **per-thread-work** gate: only dispatch (and only use the full requested thread count) when estimated per-thread work `M·K / activeThreads` exceeds a tuned threshold; otherwise fall back to single-thread or to a smaller thread count. This is precisely the microbench's own Option B (`dispatch-microbench.md:100-103`). It automatically:
- runs SmolLM's tiny decode matmuls single-threaded (skips the collapse entirely),
- lets Llama-1B's large decode matmuls use all threads.

Tradeoff: requires threading a work-size estimate (or thread-count override) into the `Dispatch` decision at each GEMV/GEMM call site. The call sites already have `M·K`, so it is local and cheap. Threshold needs tuning (start ~ the dispatch break-even, e.g. the point where 8T `Dispatch` cost ≈ single-thread work, ~1–2 K·rows on this host).

### Option B (cheap interim, lower ceiling) — make the cap model-size-aware
Keep a cap but derive it from `Config.HiddenSize` / `Config.IntermediateSize` rather than a flat 8: e.g. `decodeCap = clamp( round(intermediateSize / smallModelRowsPerThreadTarget), 2, threadCount )`. Llama-1B (8192) → high cap; SmolLM (1536) → ~8. One small change in `ComputeThreadPool` construction (it would need the config, or the caller computes the cap). Less precise than A (per-op work varies within a model) but no call-site changes.

### Option C (durable root-cause; do regardless) — fix the coordination primitives
The microbench §"Implications" #3 and `README.md` recommendation #1 both call for this:
- Cache-line-pad `_dispatchGeneration`, `_activeWorkerCount`, `_dispatchThreadCount` (`ComputeThreadPool.cs:36,37,47`) onto separate 64-byte lines to kill false sharing.
- Replace the single `CountdownEvent _completion` (`ComputeThreadPool.cs:28`) with a tree/two-level reduction so 31 workers don't all interlocked-decrement one counter.
- Bound the spin by elapsed time, not a fixed 10,000-iteration budget (`SpinIterations`, `ComputeThreadPool.cs:24`), so workers fall through to event-wait faster between bursts (the README notes workers spin their full budget *between* bursts, inflating CPU).

These raise the collapse threshold for **all** models and are the only changes that make a higher flat cap *safe*. They do not require call-site changes.

### Also fix (small, independent)
The pinning-path cap-2 footgun: ensure the topology branch (`ComputeThreadPool.cs:102-103`) never yields a smaller decode cap than the no-topology branch (e.g. `Math.Max(MemoryChannelEstimate, DefaultDecodeThreadCountCap)` or drop the channel clamp on single-node systems). Otherwise enabling pinning makes decode 3× slower on Strix.

**Suggested sequencing:** C (root-cause, safe, no call-site churn) → A (adaptive gate, captures the win) → revisit whether any flat cap is still needed. Option B is a viable shortcut if A's call-site plumbing is deemed too invasive for now.

---

## 5. Validation plan (re-measure on Strix / AVX-512 before changing the default)

All runs on Strix Halo (Zen5, 32T, AVX-512). Use the existing probe pattern (`Llama32DecodeRooflineProbe.cs`) generalized to multiple models and to the **production config path**, not only the explicit-`DecodeThreadCount` path.

1. **Reproduce the contradiction precisely.** Re-run the Llama-1B probe (already exists) and confirm 45.6@32T / 38.4@8T. Add a SmolLM-135M sweep with the *same* probe to confirm the collapse still bites at 32T under current code. **This is the gating measurement** — if SmolLM no longer collapses, the whole risk picture changes.

2. **Find the knee vs model size.** Sweep decode threads {2,4,8,16,24,30,32} on:
   - SmolLM-135M (hidden 576) — expect knee ≤8, collapse at 32.
   - a mid model, e.g. Qwen2.5-1.5B or Phi-3.5-mini (hidden ~2048–3072) — expect knee where the per-thread-work gate should sit.
   - Llama-3.2-1B (hidden 2048) — expect monotonic-ish to 32.
   The crossover model size where 32T flips from loss to win **is the threshold for Options A/B.**

3. **Isolate oversubscription from cache contention.** Add a **30T vs 32T** point on SmolLM. If 30T is healthy and only 32T collapses, the trigger is OS oversubscription (no spare core for OS/harness) → leaving 1–2 cores free is a partial mitigation. If 30T also collapses, it's pure cache contention → Option C is mandatory.

4. **Context-length sensitivity.** Repeat the SmolLM and mid-model sweeps at short (128) and long (2048–4096) context. Longer context grows the attention dispatch (scores over more KV) relative to the matmul dispatch; confirm the knee is stable or document how it moves.

5. **Production-path runs (not just explicit DecodeThreadCount).** Run end-to-end (Sample.Server, same harness as `cross-lib-20260424/README.md`) with:
   - default (no pinning → cap 8) — baseline today,
   - `--numa-pin` (topology → cap 2) — confirm/repro the 3× footgun,
   - the candidate adaptive policy.
   Compare against llama.cpp's 34.7 tok/s on SmolLM to see whether the change closes the coordination gap.

6. **Before/after Option C in isolation.** Re-run the `ThreadPoolDispatchBenchmarks` microbench (the artifact that produced `dispatch-microbench.md`) after cache-padding + tree-reduction completion, to quantify how far the 32T collapse threshold moves. If padding alone makes 32T sane on SmolLM, a higher flat cap may become acceptable and Option A's threshold can be relaxed.

**Decision rule:** adopt a higher default decode thread count (or the adaptive gate) only if (a) SmolLM-135M does not regress below its cap-8 throughput at the new setting, and (b) the mid-model knee is captured by the chosen threshold. Otherwise keep cap-8 as the small-model floor and rely on Option A to lift large models.

---

## Appendix — key file:line references

| What | Location |
|---|---|
| Decode → SpinWait selection | `src/DotLLM.Models/Architectures/TransformerModel.cs:420` |
| Batched path forces EventBased | `src/DotLLM.Models/Architectures/TransformerModel.cs:1372` |
| `SetDispatchMode` clamps active workers to decode cap | `src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:144-151` |
| `_decodeThreadCount` 3-branch computation + `DefaultDecodeThreadCountCap=8` | `src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:99-104` |
| Cited collapse comment (10.6 ms @32T) | `src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:92-98` |
| SpinIterations budget (10,000) | `src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:24` |
| Hot fields likely false-sharing | `src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:35-47` |
| `_completion` single CountdownEvent | `src/DotLLM.Cpu/Threading/ComputeThreadPool.cs:28` |
| `MemoryChannelEstimate` = 2 on single node | `src/DotLLM.Cpu/Threading/NumaTopology.cs:112` |
| Default path passes `topology: null` (pinning off) | `src/DotLLM.Models/Architectures/TransformerModel.cs:148-167`, `284-307` |
| CLI defaults: decode-threads 0, numa-pin/pcore-only false | `src/DotLLM.Cli/Commands/RunCommand.cs:80-93` |
| `EffectiveDecodeThreadCount` (Core, parallel helper) | `src/DotLLM.Core/Configuration/ThreadingConfig.cs:51-54` |
| `ParallelMinRows = 32` row-only gate | `src/DotLLM.Cpu/Kernels/MatMul.cs:2292` |
| GEMV single-thread short-circuit | `src/DotLLM.Cpu/Kernels/MatMul.cs:2570`, `2601`, `2618`, `351`, `2161` |
| Decode GEMM 2D partition (degenerates to row-only at N=1) | `src/DotLLM.Cpu/Kernels/MatMul.cs:2454-2508` |
| The probe (explicit DecodeThreadCount sweep) | `tests/DotLLM.Tests.Integration/Models/Architectures/Llama32DecodeRooflineProbe.cs` |
| Microbench (SmolLM sizes, 32T collapse) | `.perf-runs/cross-lib-20260424/dispatch-microbench.md` |
| End-to-end profile (76% CPU in spin, SmolLM) | `.perf-runs/cross-lib-20260424/README.md` |
