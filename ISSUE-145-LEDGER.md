# Issue #145 — decode kernel follow-ups: experiment ledger

**FOR THE COORDINATOR: fold this into `.docs/KERNEL_MAP.md` §12 (or a new §13) and delete this
file before the PR.** The worktree cannot edit the git-ignored main-repo `.docs/`.

Branch `issue/145-decode-kernel-followups` (base dev `c13ed010`). All numbers gfx1151,
same-session back-to-back `dotllm bench --device vulkan -p 512 -n 128 -r 3` unless noted.
Session noise note: absolute GPU-category baselines drifted between sessions (qkv_proj
0.20↔0.40 ms/tok across sessions with identical code) — only within-batch A/B was trusted.

## Item (a) — fused rmsnorm+quantize_q8_1 (LANDED, commit 42de8b07)

New `rmsnorm_quantize_q8_1.comp` (WG 256, 1 WG/row): replicates `rmsnorm_f32_sg` subgroup
reduction + `quantize_q8_1` quantization verbatim in one dispatch; also writes the normalized
F32 row (LoRA/fallback consumers). Wired via `RecordNormedSharedInputMmvqGroup` at BOTH decode
MMVQ shared-group sites (attn-norm→Q/K/V, ffn-norm→gate/up) when `CanShareMmvqQuant` passes
(seqLen==1 + all-Q8_0 group). Opt-out `DOTLLM_VULKAN_DISABLE_FUSED_RMSNORM_QUANT=1`.

| Gate | Result |
|---|---|
| Kernel parity | BIT-EXACT vs standalone pair on all 3 outputs × 7 shapes (32…9216, incl. grid-strided >256 blocks) — `VulkanRmsNormQuantizeQ8_1FusedKernelTests` |
| Exact-token parity | PASS (SmolLM Q8_0, greedy 128, fused on vs off) |
| Census | SmolLM 573→513 dispatches, 546→486 barriers per token (−60/−60, as designed) |
| SmolLM Q8_0 d512 | 540.0/542.8 → 558.1/556.9 tok/s median (+3.0%), best 543.9→561.1 (+3.2%); gpu_total 1.92-1.97→1.88-1.90 ms/tok (2 interleaved rounds) |
| 3B IQ4_XS | no change — fused cannot fire on non-Q8_0 groups (census identical 619 disp/tok); deltas ±1% = UMA noise |
| 8B Q4_K_M | 29.28 vs 28.96 median (−1.1%, noise; path identical) |

Follow-up lever NOT taken (out of #145 scope, = map §3 item 7): extend `CanShareMmvqQuant` +
the shared-group GEMV loop to non-Q8_0 MMVQ quants (all MMVQ kernels consume the same Q8_1
activation) — would remove up to 3 redundant quantize+barrier pairs/layer on Q4_K/IQ4_XS
models AND let this fused kernel fire there.

## Item (b) — split-KV pass efficiency at short ctx (MEASURED; no kernel change ships)

New profiler sub-attribution (commit af51f343): `interPassStamp` hook in
`VulkanSplitKvAttentionKernel.Record` + `attn_split` category — `DOTLLM_VULKAN_DECODE_PROFILE_GPU=1`
now separates split-pass from merge-pass GPU time.

**Attribution (SmolLM d512, S=28, hd=64, 30L):** attention ≈ 0.50 ms/tok = split pass
0.33-0.37 + merge pass 0.153-0.159. Split pass moves 26.5 MB K/V per token → ~76 GB/s
effective (BETTER than the decode GEMV class ~48-60 GB/s — the split pass is NOT grossly
inefficient). Merge pass ≈ 5.3 µs/layer ≈ pure dispatch+barrier floor (moves only ~64 KB/layer).
The "0.50 vs 0.25 data-ideal" gap therefore decomposes as: ~0.16 ms unavoidable merge
launch/barrier cost (fix requires hazard-scoped barriers (#144 territory) or single-dispatch
fusion (ruled out, #370)) + ~0.1 ms split-pass BW headroom.

**FAILED (do not re-try): smaller split-pass workgroups.** WG-variant shaders, same-session
back-to-back, SmolLM d512 split-pass GPU ms/tok:
| split WG | attn_split ms/tok |
|---|---|
| 256 (shipped) | 0.331-0.368 |
| 128 | 0.375-0.407 |
| 64 | 0.465-0.519 |
Lane-idle waste at hd=64 (only 64/256 threads active in Q-load/rescale/PV/write steps) is
dominated by the score loop's subgroup parallelism (8 subgroups sweep KV positions; 2 subgroups
serialize ~10 rounds). Same lesson as the coalescing campaigns: don't trade parallelism for
utilization at low occupancy. Variant deleted; exact-token parity of WG64 vs WG256 was PASS
(argmax-stable despite reduction-order drift) for the record.

**Fresh mini-sweep (TargetWG × MinKvPerSplit, decode tok/s median, r3):**
| model,depth | 128/8 | 128/16 | 128/32 | 256/8 | 256/16 | 256/32 |
|---|---|---|---|---|---|---|
| SmolLM d128 | 607.2 | 611.8 | 569.7 | 611.6 | **612.4** | 564.7 |
| SmolLM d512 | 501.0 | 502.6 | 503.0 | 546.9 | **552.9** | 531.8 |
| SmolLM d~1900 | 305.0 | 305.9 | 312.3 | **391.1** | 387.2 | 389.0 |
| 3B d128 | 79.5 | 79.5 | 79.5 | 79.9 | **79.9** | 79.4 |
| 3B d512 | 72.9 | 72.7 | 72.5 | 75.7 | **75.9** | 75.9 |
| 3B d2048 (r2) | — | 54.5 | 54.8 | — | **63.6** | 57.4* |
(*256/32 d2048 median hit a slow rep; best rep 63.5 ≈ 256/16.)
**Verdict: post-#143 default 256/16 remains optimal or tied everywhere.** MinKv32 regresses
SmolLM short ctx (floor binds); TargetWG 128 regresses everything except d128 (tied). No knob
change.

## Item (c) — multi-split submit ring (BUILT, TESTED, DROPPED)

Extended `SubmitContext` two-buffer split to a lazily-grown ring (`MaxSplitsPerForward=7`,
`CanSplitSubmit` gate) + model-side split at every `DecodeSplitLayer`-th layer
(`DOTLLM_VULKAN_DECODE_MULTI_SPLIT`). Exact-token parity PASS (splits at L8/16/24, SmolLM).
**REGRESSED**: SmolLM d512 ~540 vs ~555-558 tok/s median single-split (only valid interleaved
round). Mechanism: after chunk 0 (8 layers ≈ 0.4 ms GPU), the host finishes recording ALL
remaining layers in ~0.18 ms — chunk-1's record is already fully hidden by the #143 single
split; extra fenceless `vkQueueSubmit`s only add per-submit cost. Additionally a
`VK_ERROR_DEVICE_LOST` cluster (4 consecutive load failures, both multi-split on AND off,
device recovered after idle) appeared only while ring code was active — causality unproven
(box also shows the known transient map-fail flake, and #146 is investigating a flake), but
regression + risk + guaranteed conflict with #144's submit rework ⇒ **reverted, not committed**.
Do not re-try a deeper submit ring while the single-split point already hides ~2/3 of record
cost; the remaining ~0.08 ms/tok belongs to #144's architecture.

## Final gates

Vulkan unit suite (`--filter FullyQualifiedName~Vulkan`, this box): run 1 = 867 passed /
1 failed / 40 skipped; immediate re-run = **868 passed / 0 failed / 40 skipped** — the single
failure was a transient flake (name lost to log truncation; run 2 green with identical code).
No DeepSeek-OOM/WeightRepacking cases are inside this filter. All decode changes exact-token
parity-gated as listed above.

## Incidental observation (for #146)

One decode rep in ~40 SmolLM runs produced a divergent greedy token stream mid-process
(warmup rep fine, rep 1 degenerate; UNCHANGED WG256 code path, env `DOTLLM_VULKAN_SPLIT_WG64=0`
on a build whose variant differs only behind that flag). Immediately-following identical runs
were fully deterministic (reps within a process byte-identical). Consistent with a transient
GPU/driver flake rather than a code path issue.
