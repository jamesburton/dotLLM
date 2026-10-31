# MoE verify-row fusions (uzu #1020 / #1009) - profile and refutation (#1362)

Host: Strix Halo (gfx1151, UMA), Vulkan, `Qwen3.6-35B-A3B-UD-Q4_K_M` (`qwen35moe`, 40 MoE layers, top-8 of 256, Q4_K gate/up,
Q5_K/Q6_K down, Q8_0 shared expert). Rows 2..8 take the indexed-MMVQ routed path (below `GroupedMinTokens`).
Tool: `Qwen4ExpRowProbe moestage <gguf> <rows,..> <reps> [arms]` (`benchmarks/Qwen4ExpRowProbe/MoeVerifyStage.cs`).

## Method

Perturbation by skipping a dispatch (`VulkanQwen3MoeHybridTransformerModel.PerturbSkip`, diagnostic, output is garbage), arms
interleaved inside each rep (`shared,base,scatter,base,shared`), 30 reps, warm-up first, both locks held, no other GPU process
(idle ollama only). Wall ms of one N-row forward, medians (min in brackets):

| rows | base (2 arms) | skip `weighted_scatter` | skip shared expert | shared skip vs base |
|---|---|---|---|---|
| 2 | 36.6 / 36.7 (33.5 / 34.0) | 37.5 (35.5) | 34.6 / 35.2 (32.1 / 30.8) | -1.5..-2.0 ms, 4-5% |
| 5 | 54.0 / 54.2 (48.6 / 47.1) | 58.5 (52.2) | 51.3 / 52.9 (45.6 / 45.7) | -1.1..-2.8 ms, 2-5% |
| 8 | 72.6 / 71.4 (64.2 / 63.2) | 81.2 (69.4) | 66.4 / 68.4 (58.2 / 58.9) | -3.0..-6.2 ms, 4-8% |

(1 row, first run, 12 reps: base 23.6, scatter-skip 23.8, shared-skip 24.0 ms: nothing, as expected - 1 row is on the fused #647 chain.)
Run-to-run spread is large (max up to 2-3x median from other-process bursts); quote ranges, not points.

Split-submit stage profile (sync floor ~75 us x 40 layers = ~3 ms per stage label, so tiny stages read ~3 ms): at 8 rows
`down` 21 ms, `gate_up` 19 ms are the only stages above the floor in the MoE half; `weighted_scatter` 3.4 ms is the floor itself.

## Idea 1: fused down + weighted combine (M >= 2): REFUTED

* Skipping the combine dispatch entirely gives no gain (it measured 0..+10% *slower*, an artefact of stale `NormOutput` feeding
  later layers; the point is there is no headroom to find).
* Analytic: the intermediate `MoeDownRows` is rows x topK x hidden x 4 B = 0.5 MB at 8 rows per layer; x 40 layers = 20 MB
  = ~0.1 ms at ~200 GB/s (<0.2% of the verify). uzu's saving is Apple M5 Pro activation traffic at 2.8-5.6 MB/layer.
* A fused kernel would also change the reduction order of the top-k sum (the down MMVQ output is combined per column today),
  so it could not meet the #1244 bit-identical rule without a fixed-order epilogue. Not worth building. Below the issue's 3% bar.

## Idea 2: shared expert as an extra routed slot (+ gate fold): NOT FEASIBLE as specified; residual lead is launch overlap

* Upper bound: skipping the whole shared-expert chain saves 4-8% of the 2..8-row forward (a hard ceiling, since it also removes
  the shared expert's own bytes: 3 x 2048 x 512 Q8_0 = 3.3 MB/layer = ~0.7 ms over 40 layers). So ~3-6% is launch/barrier cost.
* As a routed slot it needs the shared weights in the routed bank format. Here they are Q8_0 and the banks are Q4_K/Q5_K;
  transcoding to the bank type is lossy and expanding to a wider type violates CLAUDE.md rule 2. Gate fold into the router
  GEMV would only remove one 1 x hidden dot (already fused into `MoeSharedGateAdd` for seqLen > 1, #693).
* The sigmoid-gated add and slot weights would change the summation order, so greedy identity is not preserved.

## Residual lead (filed separately): overlap the shared-expert chain with the routed chain

The shared chain (norm, gate, up, swiglu, down: ~10 dispatches, 5 barriers per layer) has no data dependency on the routed
experts until the final gated add. Interleaving it into the routed barrier epochs removes the launch cost without touching
numerics (bit-identical by construction), as #647 does for 1 row. Needs its own Q8_1 activation scratch (the shared Q8_0
matmuls today share `MoeExpandedInputXq/Xds` with the routed gate/up). Ceiling 4-8%, realistic 2-4%.

## Not measured

* Qwen4Exp (`qwen4exp`): the 111 GB UD-Q4_K_XL file is not on this machine; it shares `RecordMoeLayer`'s non-grouped path, so
  the conclusions for idea 1 (traffic argument) carry over, the shared-expert ceiling is unmeasured.
* Nemotron-H Bolt (`moe_dense` 14 ms of 84 ms): different model class and includes the latent projections; not covered here.

## Shared-expert overlap (#1452): LANDED, 1.5-3.5% at 2..8 rows

`RecordMoeLayer` (non-grouped, packed Q8_0 shared expert, 2..8 rows, fused gated add available) now records the six steps of the shared
chain (norm, Q8_1 quantize, gate+up, swiglu, Q8_1 quantize, down) into the barrier epochs the routed chain already has (router, topk,
broadcast, gate_up, swiglu, down); only the final gated add stays after the weighted scatter. Serial chain: ~11 barrier epochs per layer
(each Q8_0 matmul re-quantised the same input behind its own barrier); overlapped: 1. It uses its own Q8_1 scratch
(`SharedXq/Xds`, `SharedSiluXq/Xds`: rows <= 8, a few KB; no F32 storage) because the routed gate/up own `MoeExpandedInputXq/Xds` (#952).
Steps a routed variant does not reach are flushed before the scatter, so the order is always correct. `DOTLLM_VK_SHARED_OVERLAP=0` restores the serial chain.

* Same kernels, same inputs, quantising once instead of twice (deterministic): bit-identical. `Qwen4ExpRowProbe ovident <gguf> 2,5,8`
  (warm-up pass, then off vs on in one process, `ResetSequenceState` per sequence): logits BIT-IDENTICAL and greedy identical at chunks 2/5/8
  (405 teacher-forced forwards each; overlap counter 0 off, 8040/3320/2080 layer-records on). Control (off vs off) is identical too, so the probe can discriminate.
  Recurrent state leaks across sequences unless `ResetSequenceState` is called: without it even off-vs-off differs.
* Paired in-process A/B (`moestage <gguf> 2,5,8 20 ovoff,ovon,ovon,ovoff`, 3 runs, per-arm medians, wall ms per N-row forward):

| rows | serial (off) | overlap (on) | gain |
|---|---|---|---|
| 2 | 26.6-27.5 | 25.7-26.9 | ~0.8-1.0 ms, 3-4% |
| 5 | 38.5-39.6 | 37.9-39.3 | ~0.2-1.0 ms, 1-3% |
| 8 | 51.0-52.7 | 50.3-52.0 | ~0.4-0.9 ms, 1-2% |

  The skip-perturbation ceiling was 4-8%; about a third to half of it is realised.
* Not measured: qwen4exp (file not local; same path with the pre-staged shared input, step 0 omitted) and Bolt.
