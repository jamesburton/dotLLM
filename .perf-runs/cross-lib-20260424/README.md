---
date: 2026-04-24
model: SmolLM-135M Q8_0 (~145 MB)
host: Strix Halo (Ryzen AI Max+ 395 Zen5 32T), AMD Radeon 8060S iGPU
harness: OllamaBenchmarks/scripts/benchmark_throughput_openai.py
prompt: "Write a concise explanation of dependency injection with one short Python example."
max_tokens: 192
---

## Cross-library throughput (end-to-end, same harness)

| Backend          | toks/s | total (s) | RAM peak | GPU util | Notes |
|------------------|-------:|----------:|---------:|---------:|-------|
| llama.cpp CPU    |  34.73 |      5.53 |  0.25 GB |     14.6 | `llama-server -ngl 0 -c 2048` |
| dotLLM CPU       | **8.66** | **22.17** | 1.27 GB |     18.7 | Sample.Server, `ThreadingConfig.Auto` → 32 threads |

**Gap: dotLLM is ~4.0× slower end-to-end** on this workload.

Numbers captured with the fix below. Without it, dotLLM appeared to run at ~2.6 toks/s because generation was silently truncated to 12 tokens.

## Correctness bug fixed during the run

Found and patched in `src/DotLLM.Engine/TextGenerator.cs:850`:

```csharp
// before
if (entry.KvCache.MaxLength >= requiredSize || entry.KvCache.MaxLength >= promptLen)
    return (entry.KvCache, matchedTokens, false);
// after
if (entry.KvCache.MaxLength >= requiredSize)
    return (entry.KvCache, matchedTokens, false);
```

Symptom: every `/v1/chat/completions` request after the first returned `finish_reason=length` with ~12 generated tokens, regardless of `max_tokens`. Cause: the warmup request primed the prefix cache with a small cache (`promptLen + warmup_max_tokens`); subsequent chat requests matched the template prefix and reused that cache. The `|| MaxLength >= promptLen` branch accepted a cache with zero room for generation, and the decode loop silently broke at `pos >= cacheSize` after exhausting the undersized cache.

The fix reuses a prefix-cache entry only when it can fit the full `promptLen + maxTokens`, otherwise falls through to allocate fresh. This matches the intent of the comment one line above.

Introduced in 17bc4383 (Step 36: Paged KV-cache); `/v1/completions` was unaffected in the benchmark because the harness did not issue a pre-warmup with matching tokens. Production chat users and the harness were affected.

## Profile summary (dotnet-trace, `dotnet-sampled-thread-time`)

30s trace while generating 192+100 tokens. Hot path on the request thread (inclusive time under `TextGenerator.Generate`, summed over the trace):

| Frame | of request time |
|-------|---:|
| `TransformerModel.Forward`                 | 100% |
| ├─ `GemmInterleaved` (prefill/lm_head/O/Down) | 41% |
| ├─ `FusedGateUpDecode` → `FusedDecodeGemv2`   | 19% |
| ├─ `FusedQkvDecode` → `FusedDecodeGemv3`      | 19% |
| ├─ `Attention.Execute` (two overloads)        | 35% |
| └─ `ComputeThreadPool.Dispatch`               | 19% |

100% of leaf self-time is `UNMANAGED_CODE_TIME` because the SIMD inner loops (`VecDotQ8_0Avx512_4Rows`, `VecDotQ8_0Avx2_4RowsR4`) are JIT-emitted intrinsics that managed-stack sampling cannot walk. Kernel ETW CPU sampling (PerfView) would be needed for leaf attribution.

Artefacts:
- `dotllm-profile.nettrace` (3.7 MB) — raw trace, can be loaded in PerfView
- `dotllm-profile.speedscope.json` (32 MB) — viewable at https://speedscope.app
- `profile-summary.txt` — flat top-N text summary

## PerfView kernel CPU profile

`PerfView /ThreadTime collect` with kernel stack sampling, 25 s window, same
150-token chat request as above. Parsed with a small TraceEvent-based
analyzer in `.perf-runs/tools/EtlStackDump/`; full output in
`kernel-profile.txt`. CPU samples attributed to `dotnet`: 100,059
(no-stack: 1,665).

### Where every CPU sample went (self-time, `dotnet` process only)

| Bucket | Samples | % of dotnet CPU |
|---|---:|---:|
| `ComputeThreadPool.WorkerLoop` spin (JIT'd) — sum of all `coreclr!0x7FFF17B42B2x-6x` plus `WorkerLoop` self plus `SpinWait.SpinOnce` | ~76,000 | **~76%** |
| kernel32/ntoskrnl/ntdll (thread wake / sched) | ~7,000 | ~7% |
| `VecDotQ8_0Avx512_4Rows` (inner matmul kernel) | 827 | 0.84% |
| `ComputeRowsQ8_0R4Worker` (R4-layout matmul worker) | 122 | 0.12% |
| `AttentionWorker` | 7 | 0.01% |
| `QuantizeF32ToQ8_0Avx512` | 1 | <0.01% |
| everything else (JIT, GC, HTTP, ancillary) | remainder | ~15% |

Attribution trick: for every `coreclr!0x7FFF17B42B??` unresolved leaf, the
analyzer walks up the stack to the nearest managed frame. In every case
>99% of those samples resolve to `ComputeThreadPool.WorkerLoop` or
`SpinWait.SpinOnce`. So the "66% in unresolved coreclr" is the JIT-emitted
body of the pool's spin loop — it's not JIT compilation, not GC, not
unrelated runtime work.

### Signal

- **The 4× gap vs llama.cpp is a worker-pool coordination gap, not a
  kernel-quality gap.** Real matmul work is ~1% of dotnet CPU time across
  32 threads. llama.cpp on the same box does the same compute in ~1/4
  the wall time because its OMP-style barriers don't burn three-quarters
  of thread-time on spin.
- **Kernel choice is fine.** The non-R4 AVX-512 path (`VecDotQ8_0Avx512_4Rows`)
  dominates the kernel work at 827 samples vs R4's 122 — MatMul is picking
  the best available variant for this workload.
- **AVX-512 R4 kernel would help only modestly.** Even if we halved the
  122 R4 samples to 0, we'd save ~0.1% of dotnet CPU. It's not where the
  gap lives on this CPU for this model.

## Recommended next work (in ROI order)

1. **`ComputeThreadPool` rework — the actual target.** Some combination of:
   (a) shrink the spin horizon when no work is queued (bound spin by
   elapsed ns, not by iteration count); (b) for decode-path matmuls under
   some threshold (say, lm_head / QKV / Gate-Up-Down at 576×576 for
   SmolLM-135M) run single-threaded and skip dispatch entirely; (c) batch
   Q/K/V dispatches across layers when possible. Pre-work: write a
   microbench in `benchmarks/DotLLM.Benchmarks/ParallelBenchmarks.cs`
   measuring `Dispatch(work=0)` round-trip vs `Dispatch(work=small)` vs
   single-thread at realistic decode matmul sizes. ~1-2 days for the
   microbench + first cut.
2. **Tiled GEMM with cache blocking** (prefill-side). Still relevant for
   the 7× prefill gap in `llama-bench` numbers, which the PerfView trace
   didn't exercise (we only decoded 150 tokens from a short prompt).
3. **Vulkan LLM kernels.** Largest absolute headroom on this iGPU
   (llama.cpp Vulkan = 392 tok/s decode), but only worth pursuing after
   the CPU path stops blocking on its own coordination.
4. **Push 53cee5d + d7599c4 to origin**, then open an issue that tracks
   the ComputeThreadPool rework with this profile linked.

Artefacts:
- `dotllm-kernel.etl.zip` (127 MB) — PerfView ETL archive, openable in
  PerfView (CPU Stacks → dotnet)
- `etl-extract/dotllm-kernel.etl` (176 MB) — unzipped for TraceEvent
- `kernel-profile.txt` — text summary from the analyzer
- `kernel-profile.txt` includes per-`coreclr!0x…` caller attribution
- `tools/EtlStackDump/` — the TraceEvent-based analyzer used to produce
  the above; build with `dotnet build -c Release`

## Repro

```bash
# llama.cpp
C:/Development/llama.cpp/llama-server.exe \
    -m C:/Users/james/.dotllm/test-cache/QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf \
    --port 8000 --host 127.0.0.1 -ngl 0 -c 2048 --alias SmolLM-135M --no-webui

# dotLLM
dotnet samples/DotLLM.Sample.Server/bin/Release/net10.0/DotLLM.Sample.Server.dll \
    C:/Users/james/.dotllm/test-cache/QuantFactory/SmolLM-135M-GGUF/SmolLM-135M.Q8_0.gguf --port 8000

# harness (in either case)
cd C:/Development/OllamaBenchmarks/scripts
python benchmark_throughput_openai.py --model SmolLM-135M \
    --base-url http://127.0.0.1:8000 --num-predict 192 \
    --process-name {llama-server|dotnet} \
    --output .../throughput.json
```
