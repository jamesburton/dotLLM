# Elevated `ncu --set full` profiling — I2_S/BitNet decode kernels, 2026-07-28

Real Nsight Compute data (elevated PowerShell / UAC, per `ncu-elevation-workflow` memory) targeting
the open questions from tonight's #206/#212/#213/#218 work. Raw `.ncu-rep` binaries and per-target
`*-details.csv` exports are NOT committed (large, regenerable) — this README is the durable record.

Recipe: `ncu --set full --kernel-name "regex:<exact kernel name>" --launch-count 8` wrapping either
`dotnet test --filter FullyQualifiedName~CudaAttentionDynVsScalarPerfTest` (for the attention pair)
or `dotnet run --project src/DotLLM.Cli -- bench <model> --device cuda -p 8 -n 16 -r 1` (for the GEMV
kernels). **Lesson learned**: `--kernel-name regex:` with a `|` alternation broke through `ncu.bat`'s
cmd.exe re-invocation on Windows (the pipe leaked out as a real shell pipe) — target one exact kernel
name per invocation instead of trying to bundle multiple with alternation.

## `attention_f16` vs `attention_f16_dyn` — issue #213/#218's open question, now answered

Same shape (seq_kv=64, grid=(20,1,1), block=(256,1,1), CC 8.6), same registers (40), same occupancy
(~16.6%), same waves/SM (0.12) — every launch-configuration variable is identical. The only
difference is `attention_f16_dyn` reading `seq_kv`/`position_offset` via device pointers (so a CUDA
graph replay can grow them without re-instantiation) instead of `attention_f16`'s plain scalar kernel
args.

| Metric | `attention_f16` (scalar) | `attention_f16_dyn` | Δ |
|---|---|---|---|
| Duration | 22.4 us | 28.1 us | **+25%** |
| Compute (SM) Throughput | 4.9% | 4.0% | lower (doing less useful work per unit time) |
| Warp Cycles Per Issued Instruction | 28.2 | 35.3 | **+25%**, tracks duration almost exactly |
| CTA-barrier wait (of total stall cycles) | 16.5 cyc (58.6% of 28.2) | 19.9 cyc (56.6% of 35.3) | **+3.4 cycles**, ~half the total gap |

**Root cause, now precise rather than inferred**: the dyn kernel spends measurably more cycles
stalled at a `__syncthreads()`-class CTA barrier than the scalar kernel does — the device-pointer
dereference for `seq_kv`/`position_offset` adds latency that isn't fully hidden before the barrier,
so warps wait longer for their siblings to catch up. This accounts for roughly half the total
cycles-per-instruction gap (barrier-wait alone: +3.4 of +7.1 total cycle increase); the remainder is
spread across other stall categories not yet individually isolated in this pass. **Actionable next
step**: try issuing/prefetching the device-pointer read earlier in the kernel (before other
independent setup work), so its latency overlaps rather than landing right before the sync point —
a targeted, low-risk kernel edit, not the more invasive `cuGraphExecKernelNodeSetParams` graph
restructure #217 flagged as the alternative. Someone picking this up should try that change and
re-run this same `--kernel-name "regex:attention_f16_dyn"` capture to confirm the barrier-wait
number drops.

## `i2_s_gemv_f16in_ragged` (bitnet_b1_58-xl `ffn_down`, k=5460) — issue #206 follow-up

| Metric | Value |
|---|---|
| Duration | ~104.7 us |
| Compute (SM) Throughput | 73.9% |
| Achieved Occupancy | 55.9% |
| Waves Per SM | 2.29 |
| Warp Cycles Per Issued Instruction | 11.6 |
| Registers | 28 |

**Revises the plan's P1 priority downward.** This is the scalar, correctness-first fallback #206
shipped for non-128-aligned K — the assumption going in was that it likely left real throughput on
the table versus a vectorized path. The data says otherwise: 74% compute throughput and 56%
occupancy is a reasonably healthy profile, nowhere near the "grid too small" pathology the attention
kernels show (16.6% occupancy, 0.12 waves/SM). There's a real but modest gap versus the aligned fast
path (`i2_s_gemv2_f16in` below: 78% compute, 61% occupancy) — worth revisiting if `bitnet_b1_58-xl`
becomes a priority target specifically, but not the low-hanging fruit it looked like before this data
existed.

## `i2_s_gemv2_f16in` (BitNet-2B-4T fused gate+up GEMV) — fresh post-tonight baseline

| Metric | Value |
|---|---|
| Duration | ~127.6 us |
| Compute (SM) Throughput | 77.7% |
| Achieved Occupancy | 61.3% |
| Waves Per SM | 7.71 |
| Warp Cycles Per Issued Instruction | 13.0 |
| Registers | 54 |

Healthy occupancy and throughput — consistent with this being the well-tuned, 128-aligned fast path.
No action indicated here; recorded as the current-state reference point for future comparison.

## Context

Session narrative: `[[bitnet-support]]` project memory. Issues referenced: #206 (ragged I2_S),
#212/#213/#217 (BitNet CUDA-graph capture + depth regression), #218 (generalizing the depth
regression fix beyond BitNet — this profiling run was gathered specifically to inform that issue).
