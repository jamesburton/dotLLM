# Issue #149 — KHR coopmat flash-attention prefill kernel (ledger)

**FOR THE COORDINATOR:** fold this into `.docs/KERNEL_MAP.md` §6 (attention) and the §0b
env-var index, then delete this file — it duplicates the same ledger content the
worktree's gitignored `.docs/KERNEL_MAP.md` can't carry (per repo convention, the
worktree cannot edit main's gitignored `.docs/`, so this tracked doc is the handoff
vehicle). Branch `issue/149-coopmat-attention`, base `dev` @ `76de82c3` (re-based onto
`916c5190` — a `main`-merge commit — after starting; see git log).

Machine: Strix Halo (gfx1151, UMA), Windows, AMD proprietary driver. GPU under
`scripts/gpu-lock.sh` as `agent-149-coopmat`.

## Mission

Post-#139, dense prefill sat at ~0.63-0.64× llama.cpp pp512 (3B 1581 vs 2485, 8B 605 vs
962 tok/s) with the prefill flash-attention shader identified as scalar F32
(~0.9 TFLOP/s effective) — the #139 ledger's "remaining lever": llama.cpp uses
`KHR_coopmat` f16 QK^T/PV on this driver (its `FA_COOPMAT1` path, `flash_attn_cm1.comp`).
Build the equivalent for dotLLM's prefill (`seqQ > 1`) path only; decode (split-KV) stays
untouched.

## Why the EXISTING opt-in (`DOTLLM_VULKAN_USE_COOPMAT_ATTENTION`) wasn't promoted

That kernel (`attention_f32_coopmat.comp`, landed #220, commit `533faad8`) is the
**per-token legacy-attention** analogue (one workgroup per (query-token, head), the
decode-shaped dispatch geometry), NOT a flash-attention prefill kernel — it was measured
1.27-1.56× SLOWER than the scalar per-token kernel because (a) it was benchmarked
predominantly at decode shapes (sq=1), where llama.cpp's own tuning (`get_fa_tuning_params`)
*also* forces `FA_SCALAR` — coopmat is a prefill-only technique in llama.cpp too, and (b) at
WG=64 (one wavefront) it held ~40 KB LDS resident, capping occupancy to ~1-2 workgroups per
WGP on RDNA3.5. It was never wrong to leave it opt-in; it was solving the wrong shape. This
issue targets `attention_flash_f32.comp`'s prefill dispatch (BR=16, BC=64, WG=256, one
workgroup per (query-tile, head), KV read once per Q-tile) — the tiling llama.cpp's
`FA_COOPMAT1` also builds on (`Br=16, Bc=4×16` subgroup slices).

## Kernel design

New `native/vulkan/shaders/attention_flash_f32_coopmat.comp` +
`src/DotLLM.Vulkan/Kernels/VulkanFlashAttentionCoopmatKernel.cs` — a drop-in swap for
`VulkanFlashAttentionF32Kernel` (identical bindings, push-constant layout, dispatch
geometry: `groups.x = ceil(seqQ/16) * numHeads`, `WG_SIZE=256`).

- Tiling: BR=16 Q-rows, BC=64 KV-columns per workgroup iteration, same as the scalar FA
  shader. The 256-thread workgroup is split into 4 subgroup-scope 16×16×16 coopmat tiles
  (`NUM_SLICES=4`) — each of the 4 KV-column slices (16 columns) owned by
  `gl_SubgroupID % 4`. This is llama.cpp's `FA_COOPMAT1` shape (`coopmat_block_rows=16,
  coopmat_block_cols=16, num_subgroups=4` in `get_fa_tuning_params_coopmat1`).
- QK^T: Q and K staged to LDS as f16 (zero-padded past `headDim`/tile bounds), multiplied
  via `coopMatMulAdd` in 16-wide inner-dim chunks with an **F32 accumulator**, stored to an
  F32 score tile.
- Scale / causal / sliding-window / soft-cap mask: applied to the F32 scores exactly as in
  the scalar FA shader (same order: `dot*scale - alibi`, then softcap, then mask →
  `NEG_INF`).
- Softmax: row-owner scan (one thread per Q-row, matching the scalar shader's #139
  optimization) computes running max/sum in F32; the `exp(s-max)` weights are rounded to
  f16 for the P tile (fed to the P·V coopmat multiply) and the running **sum accumulates
  `float(f16(w))`** so numerator and denominator round identically (no floor/ceiling drift
  between the softmax weight used for P·V and the weight counted in the denominator).
- P·V: **F32 accumulator** (stricter than llama.cpp's `cm1` default f16 accumulator) —
  chosen to keep the output-side rounding class closer to the scalar reference and give
  more parity headroom. Two rounds cover headDim up to 128 (`round*64 + slice*16`).
  headDim 512 (Gemma-4 global layers) does NOT fit `MAX_HEAD_DIM=128` — same cap as the
  scalar FA kernel; not attempted here (would need a second output-tiling pass or an LDS
  redesign — flagged as follow-up, not attempted this issue per the "headDim 512 if the
  tiling allows without contortion" scope note).
- O accumulation stays in **registers**, not LDS (`oReg[8]` per thread, 2 rows × 128 dims /
  256 threads) — avoids an extra LDS buffer and matches the register-residency style the
  scalar shader already uses for its own output accumulator, adapted for the coopmat
  P·V's per-slice staging (`oStage` is only used as a narrow coopMatStore relay, not the
  running accumulator).
- **AMD LLPC alignment requirement discovered empirically**: the scalar shader's `BC+1`
  score-tile row stride (used to bank-stagger the row-owner softmax scan) is NOT safe for
  `coopMatStore`/`coopMatLoad` — the driver requires row strides on 16-byte boundaries for
  cooperative-matrix ops. Using `BC+1` (float) / `BC+2` (f16) strides silently produced
  garbage (100% of elements outside tolerance, magnitudes in the tens of thousands) with
  no validation error. Fixed by widening to `BC+4` floats / `BC+8` f16 (both land on
  16-byte multiples). This is new information for the "AMD LLPC has miscompiled complex
  shaders before" caution in the issue brief — this wasn't a miscompile, it was an
  undocumented (from the KHR_cooperative_matrix spec text) alignment precondition the
  driver doesn't diagnose.

## Numerics contract (deliberate, not a defect)

Q/K/V are rounded to f16 for both matmuls — the same rounding class llama.cpp ships (its
KV cache is f16-resident). Softmax state (running max/sum/correction) and BOTH matmul
accumulators (QK^T and P·V) stay F32 — stricter than llama.cpp's own `cm1` default (which
uses an f16 P·V accumulator). Parity vs the all-F32 CPU oracle is therefore epsilon-level,
not bit-exact.

## Parity gates

**Kernel-level CPU-oracle parity** (`VulkanFlashAttentionCoopmatKernelTests`, 18 shapes —
discriminating per the repo rule: GQA groupSize 3 (asymmetric `hq/group` vs `hq%group`),
headDim 64/72/80/128 (72 and 80 are non-multiples of 128 but ARE multiples of the 16-wide
coopmat chunk at 80, and a genuinely padded tail chunk at 72), partial Q/KV tiles, non-zero
`positionOffset` (chunked-prefill continuation), causal/bidirectional/hybrid masks,
sliding-window incl. cross-tile-boundary, soft-cap, ALiBi, Gemma-3 QPAS scale override,
2048-token long context):

- Tolerance: abs 1e-3 / rel 1e-2 (looser than the existing per-token coopmat kernel's
  5e-4/5e-3 — this kernel's P tile ALSO rounds through f16, one more f16 hop than the
  per-token coopmat kernel's Q/K/V-only rounding).
- **Result: 18/18 PASS.** Measured maxAbs across all 18 shapes ≤ 4.0e-4 (well inside the
  1e-3 gate) once the LDS stride bug above was fixed. Before the fix: 18/18 FAIL,
  maxAbs in the tens of thousands (unambiguous garbage, not a tolerance question).

**End-to-end greedy-token stability** (`DOTLLM_BENCH_DUMP_TOKENS`, coopmat vs
`DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION=1`, 128-token greedy continuations,
same-process A/B):

- **Llama-3.2-3B IQ4_XS, p256/n128: IDENTICAL token-for-token.**
- **SmolLM-135M Q8_0, p256/n128: diverges at decode-step index 7** (coopmat picks token
  504, scalar picks token 33) in the FIRST quick A/B. Investigated per the issue's
  tolerance-aware gate ("if a token diverges, verify the logit gap is sub-epsilon or
  fail") — **root-caused as a PRE-EXISTING SmolLM/Vulkan run-to-run non-determinism, not
  a coopmat-kernel defect**:
  1. A controlled same-process replay of the EXACT `BenchRunner.RunOneRep` protocol
     (prefill 256 tokens → 8 sequential single-token decode `Forward` calls appending to
     one KV cache — decode always uses the unchanged split-KV/legacy kernel, since the new
     coopmat FA only fires at `seqQ>1`) shows coopmat and scalar producing the
     **IDENTICAL 8-token dumped sequence**, with the step-7 top-1/top-2 logit gap a
     comfortable **5.64 (coopmat) / 5.58 (scalar)** — nowhere near tie-breaking territory.
  2. Repeating the **coopmat-only** bench command (identical code, identical env, 3
     separate process launches) reproduced the SAME divergence pattern AMONG ITSELF: runs
     1-2 picked token 504 at step 7, run 3 picked token 33 — with coopmat held constant
     the whole time. This is process-to-process nondeterminism unrelated to the kernel
     choice.
  3. This matches the exact symptom already on record in the #145 ledger's "Incidental
     observation" (`.docs/KERNEL_MAP.md`): *"One decode rep in ~40 SmolLM runs produced a
     divergent greedy token stream mid-process... Consistent with a transient GPU/driver
     flake rather than a code path issue."* SmolLM-135M's degenerate repeating-token
     output (`27003 690 260 23790 2767 30 216 [216|504] ...`) sits near several argmax
     ties across its tiny vocabulary distribution, and gfx1151/driver-level scheduling
     nondeterminism (workgroup dispatch order affecting reduction order in some
     unidentified op) occasionally flips one. The coopmat kernel does not change this
     baseline rate in the samples gathered here (3/3 scalar-only repeats stable at 504;
     1/3 coopmat-only repeats hit the flake — consistent with the ledger's ~1/40 rough
     rate at this sample size, not evidence of a new or amplified failure mode).
  - **Verdict: no action required from this kernel; flagged as a pre-existing SmolLM
    Vulkan decode nondeterminism for whoever picks up the #146-adjacent flake
    investigation.** Larger models (3B, tested) do not show it at this sample size.

## Full Vulkan unit-test suite (regression gate)

Two full runs of `--filter FullyQualifiedName~Vulkan` on this branch:
**922 passed / 1 failed / 41 skipped**, both times, same single failure:
`VulkanPipelineParityTests.PipelinedForwardBatch_MatchesPerSequenceForward`. Isolated
re-run of just that test: **2/2 clean passes** — confirms it's flaky, not deterministic.
This is the exact pre-existing flake documented in the #147 ledger
(`.docs/KERNEL_MAP.md`): dual-device pipeline-parallel batched-decode cross-talk,
unrelated to attention (decode, not prefill; a different `IModel` composition path this
issue never touches). Not a regression from this branch.

## Prefill before/after (`dotllm bench --device vulkan`, same-session back-to-back,
coopmat = default vs `DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION=1`)

| Model | p512 before (scalar) | p512 after (coopmat) | Δ | p2048 before | p2048 after | Δ |
|---|---|---|---|---|---|---|
| SmolLM-135M Q8_0 | 11,921 tok/s | 13,530 tok/s | **+13.5%** | 6,902 tok/s | 9,191 tok/s | **+33.2%** |
| Llama-3.2-3B IQ4_XS | 1,906-1,940 tok/s | 2,169-2,177 tok/s | **+12-13%** | 963-969 tok/s | 1,578-1,608 tok/s | **+64%** |
| Llama-3.1-8B Q4_K_M | 645-684 tok/s | 681-725 tok/s | **+6%** | 440 tok/s | 581 tok/s | **+32%** |

Decode is unaffected by construction (coopmat FA only fires at `seqQ>1`) — confirmed via
identical `decode_tok_s_median` across the A/B pairs above (e.g. 3B 80.2-81.3 tok/s both
sides; 8B 29.1-29.2 tok/s both sides) and via the 128-token exact-token-parity dumps.

The win **grows with context length** (small at p512, large at p2048) because the
scalar-vs-coopmat delta is purely in the attention bucket, and attention's SHARE of total
prefill time grows with `seqKv` (the projections are O(seqLen), attention is
O(seqLen×seqKv) — quadratic in context, same reason llama.cpp's own FA investment pays
off more at long context). Per-bucket profiler attribution (3B, p512,
`DOTLLM_VULKAN_PREFILL_PROFILE=1`): attention bucket 73-77 ms (24-25% of total) on scalar
→ 39-44 ms (15%) on coopmat — roughly halved at p512; the halving compounds at p2048 where
attention is a much larger fraction of total prefill time.

## Remaining gap to llama.cpp pp512

Not re-measured end-to-end this issue (out of scope — the #139 ledger's cross-runtime
standing was 0.63-0.64× at p512 pre-#149). Given decode is untouched and the attention
bucket improvement is concentrated in a 15-25%-of-total slice, the p512 headline
tok/s gain (+6-13%) is consistent with roughly halving just that slice while
`ffn_gate_up`/`ffn_down`/`qkv_proj`/`o_proj` (the dominant ~75-85% of prefill, per the
#139 MMQ-tile-shape ledger) are unchanged — i.e. this issue narrows the gap but the
MMQ-tile-shape lever (#139's own "remaining levers": tile-shape search, double-buffering)
is still the larger remaining piece for reaching parity with llama.cpp's ~2485 tok/s (3B)
/ ~962 tok/s (8B) pp512 figures. At long context (p2048+), where attention's share is
larger, this issue's win is the dominant lever measured so far.

## Follow-ups not attempted this issue

1. **headDim 512** (Gemma-4 global layers) — falls to the legacy per-token kernel at
   prefill exactly as before (unchanged, same cap as the scalar FA kernel). Extending
   `MAX_HEAD_DIM` past 128 needs either a wider O-register footprint per thread or an
   LDS-staged O accumulator (trading the register-residency win) — flagged, not built.
2. **f16 P·V accumulator** (llama.cpp's `cm1` default) was deliberately NOT used here
   (kept F32) for parity headroom; worth a follow-up A/B once the current numerics are
   proven in production — could close additional gap on RDNA3.5's f16 accumulator path if
   the driver has a faster instruction for it, unverified.
3. **SmolLM Vulkan decode nondeterminism** (documented above) — pre-existing, belongs to
   whichever issue picks up the #145/#146 flake trail, not this one.

## Env vars added

| Var | Meaning (default) |
|---|---|
| `DOTLLM_VULKAN_DISABLE_COOPMAT_ATTENTION` | Kill-switch (#149): force prefill attention onto the scalar `VulkanFlashAttentionF32Kernel` instead of the new coopmat kernel. Default: coopmat used whenever the device advertises a subgroup-scope 16×16×16 F16×F16→F32 `VK_KHR_cooperative_matrix` tile and `headDim ≤ 128`. `VulkanTransformerModel.cs` |
