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
