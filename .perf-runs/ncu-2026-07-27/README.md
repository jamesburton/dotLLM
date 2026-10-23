# Elevated `ncu --set full` profiling — Bonsai-27B decode, 2026-07-27

Real `Nsight Compute` (not the coarse `DOTLLM_HYBRID_PROFILE` category profiler) run against real
Bonsai-27B decode-step kernel launches on the RTX 3060 (elevated PowerShell / UAC, per
`ncu-elevation-workflow` — this box requires elevation for GPU perf counters, `ERR_NVGPUCTRPERM`
otherwise). Two targets, each `--launch-skip 20 --launch-count 8` (skip warmup/prefill launches,
capture 8 real decode-step launches), `dotllm bench --device cuda -p 8 -n 12 -r 1`.

Raw `.ncu-rep` binary reports (~17 MB total) are NOT committed — this README is the durable
record. Findings are also folded into `docs/CUDA.md`'s Future Work entry (Flash Attention) and
`native/kernels/pq2_0_gemv.cu`'s file-header investigation log, which are the authoritative,
in-tree locations for anyone picking this up later.

## `attention_f32` (attn-6c-core) — confirms and quantifies "grid too small"

Decode launches: grid=(24,1,1), block=(256,1,1), CC 8.6.

| Metric | Value |
|---|---|
| Compute (SM) Throughput | 4.2-4.3% |
| Memory Throughput | 8.6-8.7% |
| Achieved Occupancy | 16.5-16.8% (Theoretical: 100%) |
| Waves Per SM | 0.14 |
| Warp Cycles Per Issued Instruction | 42.3-42.8 |
| Duration | ~20-21 us/launch |
| ncu's own "Est. Speedup" (occupancy) | ~83% |

**Reading**: grid=24 (=`numHeads` for Bonsai) badly underfills the 3060's 28 SMs (0.14 waves/SM).
But both compute AND memory throughput sit near-idle simultaneously — this is not simply
occupancy-bound, it's **latency-bound** (42 cycles stalled per issued instruction). That
distinction matters for what fixes it: more parallel blocks (issue #183's split-KV/Flash-Decoding
kernel, already shipped opt-in) helps an occupancy-bound kernel but doesn't reduce per-memory-access
latency on a latency-bound one — which is consistent with, and now explains, #183's real-world A/B
coming back within noise (+0.5% to +2% best-of at depth 256-1024, not a clean win). The actual lever
for a latency-bound kernel is fewer/larger/better-pipelined memory transactions (tiled shared-memory
staging + online softmax + deeper software pipelining) — i.e. the flash-attention rewrite already
tracked in `docs/CUDA.md`'s Future Work, now with quantitative justification rather than just
category-level profiling.

## `pq2_0_gemv*` family — confirms the existing "well-substantiated stopping point"

| Kernel | Grid | Achieved Occupancy | Compute/Memory Throughput | Waves/SM | Duration |
|---|---|---|---|---|---|
| `pq2_0_gemv2_f32io_small` | 128 | 51.76% | 54.49% | 0.76 | 35.07 us |
| `pq2_0_gemv_f32io` | 320 | 71.15-71.84% | 74.91-77.46% | 2.29 | 77-205 us |
| `pq2_0_gemv2_f32io_small` | 2176 | 96.69% | 79.61% | 12.95 | 407.90 us |
| `pq2_0_gemv_f32io_small` | 640 | 90.71% | 81.59% | 3.81 | 118.24 us |
| `pq2_0_gemv_f32io_small` | 384 | 85.29% | 77.12% | 2.29 | 75.62 us |
| `pq2_0_gemv2_f32io_small` | **6** | (theoretical 100%) | **4.15%** | **0.04** | 21.79 us |

The dominant, large-grid launches (dense FFN gate/up/down, attention K/V/O — everything except the
last row) are already well-occupied (52-97%) with healthy throughput (54-82%) — this directly
confirms the file header's prior "well-substantiated stopping point" conclusion (9 documented
negative results, closed pending fresh `ncu` evidence). **This is that fresh evidence, and it does
not warrant reopening the investigation.**

One new, narrow finding: a `pq2_0_gemv2_f32io_small` launch at grid=6 shows the same "grid too
small" pathology as `attention_f32` (0.04 waves/SM, 4.15% throughput) — almost certainly one of the
small per-layer GDN gating projections (alpha/beta, `NVHead`-wide output), not a dominant bank. Its
absolute duration (21.79 us) is tiny next to the dominant launches' 77-408 us, so any fix here has a
small absolute ceiling. Logged as a low-effort, low-risk follow-up candidate (e.g. batch multiple
small per-layer projections into fewer launches) in the kernel file header — not pursued further
this session, not worth reopening the broader investigation over.

## Context for next session

Current end-to-end numbers this same session (`benchmarks/perf-matrix/results.csv`,
`devsyncwork-4d118ff1`): decode 17.8 tok/s median, prefill 99.7 tok/s median (`-p 64 -n 32 -r 5`).
Cumulative from the original 2026-07-20 naive baseline: decode 10.4x, prefill 10.1x. The
flash-attention rewrite above is the clearest remaining large lever; the GEMV kernel family is
correctly left alone per this session's fresh evidence.
