# qwen4exp (Qwen3.8-Flash-Next) Vulkan prefill profile on Strix Halo (#1334)

Measured 2026-10-27 on dev `ca8c5355` (after #1325), Radeon 8060S (gfx1151), UD-IQ4_XS, MTP **off**, F16 K/V + QSA
(`DOTLLM_VK_QWEN4EXP_KV=f16`, `DOTLLM_VK_QWEN4EXP_CONTEXT=12288`), engine chunking = 1024 rows per forward, last-row logits only.
GPU lock held for every run, no other GPU process present before or after. One process, model loaded once (`benchmarks/Qwen4ExpRowProbe`).

## 1. Matched-shape throughput (plain prefill tok/s, 3 timed reps after 1 discarded warm-up rep)

Real text (wikitext-2 test, different window per rep), fresh state per rep.

| shape | reps (tok/s) | median | range |
|---|---|---|---|
| 1K fresh | 473.2 320.3 338.1 | 338.1 | 320-473 |
| 2K fresh | 452.0 420.0 427.6 | 427.6 | 420-452 |
| 4K fresh | 417.9 395.1 398.2 | 398.2 | 395-418 |
| 8K fresh | 387.3 372.5 366.0 | 372.5 | 366-387 |
| 2K append after 8K history | 349.8 344.1 334.8 | 344.1 | 335-350 |

Cross-check with the product path: `dotllm bench -d vulkan -p 4096 -n 16 -r 3` (tiled seed prompt, default F32 K/V, 8192 capacity):
422 tok/s (417-423). Tiled text routes to fewer distinct experts than wikitext, which probably explains 422 vs 398.

**The issue's "~132 tok/s on one long prompt" is NOT reproduced by plain prefill** (2.5-3.2x below the medians measured here). Not explained;
untested hypotheses: MTP head absorb during prefill, server path overheads, an older build. Do not treat 132 as the plain-prefill figure.

**ds4 comparison is still not matched**: ds4 reports 507/565/595/614 tok/s at 1K/2K/4K/8K with a Q2 quant on ROCm 10/Linux; ours is IQ4_XS
(roughly 2x the expert bytes). Ratio at 8K is 372/614 = 0.61, at 1K 338/507 = 0.67, with the quant difference unaccounted for.
Our curve falls with length from 2K (428) to 8K (372), 1K reps vary widely (320-473); ds4's rises.

## 2. Per-stage GPU-timestamp profile (`pf ... prof=2`, % of wall; 2 reps; timestamps do not distort: 8K 364 tok/s profiled vs 372 unprofiled)

| stage group | 1K | 4K | 8K fresh | 2K after 8K |
|---|---|---|---|---|
| MoE grouped gate/up (`grouped_gate_up`) | 26.7 | 22.7 | 21.6 | 20.7 |
| MoE grouped down (`grouped_down`) | 15.2 | 12.9 | 12.2 | 11.6 |
| attention core: rope + KV write + QSA indexer + attention (`attn_rope_core`) | 7.2 | 20.5 | 24.3 | 28.0 |
| gated-residual (`gr.grouprms` + `gr.down` + `gr.up` + `gr.act` + `gr.mixmean` + `gr_write_x`) | 21.3 | 18.3 | 17.3 | 16.3 |
| GDN (`gdn_proj` + `gdn_outproj` + `gdn_scan` + conv/l2/decay/gate) | 15.7 | 13.2 | 12.5 | 11.9 |
| attention q/kv projections + qk-norm + sigmul | 2.6 | 2.2 | 2.1 | 2.0 |
| shared expert (`shx_*`) | 2.7 | 2.2 | 2.1 | 2.1 |
| PLE (device projections + host apply/transfer + `~ple`) | 4.2 | 3.7 | 3.4 | 3.1 |
| routing glue (router, topk, offsets, expand, swiglu, scatter, embed, lm_head) | 4.6 | 3.9 | 3.6 | 3.5 |

(Values rounded from the per-label table; groups sum to ~99-100%.) Biggest single labels at 8K: `attn_rope_core` 24.3%, `grouped_gate_up` 21.6%,
`grouped_down` 12.2%, `gdn_proj` 6.0%, `gr.down` 5.7%, `gr.grouprms` 5.5%, `gdn_outproj` 2.7%, `gr_write_x` 2.5%, `gdn_scan` 2.3%.
Inter-submit host gaps (96 submits per 1024-row chunk) are charged to whichever stage follows them, so small stages carry a little submit overhead.

## 3. Perturbation (skip the dispatches between two stage markers; 4K fresh, 3 reps, base wall 10.8-11.2 s, `perturb` probe command)

| skipped | wall change | timestamp share | verdict |
|---|---|---|---|
| `grouped_gate_up` | 11155 -> 8283 ms (-25.7%) | 22.7% | confirmed (routing tiles +2%, so not understated by a routing collapse) |
| attention core (`attn_rope_core`) | 11044 -> 8792 ms (-20.4%) | 20.5% | confirmed; at 2K-after-8K: 6037 -> 4329 ms (-28.3%) vs 28.0% |
| `grouped_down` | 10977 -> 8988 ms (-18.1%) | 12.9% | CONFOUNDED: routing tile count fell 12.6%, so later layers did less MoE work |
| `gr.down` (+inject) | 10988 -> 9092 ms (-17.3%) | 6.0% | CONFOUNDED: routing tile count fell 21%; true share is between 6% and ~10% |
| `gr.up` | 11162 -> 11071 ms | 1.8% | below noise |
| shared expert chain | 11075 -> 10899 ms (-1.6%) | ~2% | below noise |
| `gdn_scan` | 11075 -> 11467 ms (+3.5%) | 2.4% | below noise (the following baseline run was a 14.6 s outlier; drift) |
| `gdn_outproj` | 11234 -> 11076 ms (-1.4%) | 2.8% | below noise |

Method note: a skipped stage leaves stale buffers, so downstream routing can change. The deterministic `tiles(count)` counter (163089 per 4K prefill in
every baseline run) is the check: any perturbation that moves it is unreliable. Noise is roughly +-3% of wall between identical runs, so shares under ~3%
cannot be resolved by perturbation and rest on the timestamps alone.

## 4. Ranked levers (estimates are of wall time at 4K-8K and are NOT measured wins)

1. **Routed-MoE grouped GEMMs (34-36% of wall, 40%+ with glue).** Confirmed by perturbation. The ds4 "prefetched gate/up kernel for busy experts"
   reports +3-5% on its own total, i.e. roughly 1-2% of ours; a real win needs a faster grouped IQ4_XS / Q4_K coopmat tile (this is the Q4_K-class
   `grouped_gate_up`, whose gate+up pair re-reads the activation per bank). No roofline has been computed here; computing bytes vs FLOPs per expert tile
   for the actual quant mix is the first step before building anything. Maps to a Vulkan change: yes (kernel work, medium).
2. **Attention core (20-28%, grows with context, only 7% at 1K).** Confirmed by perturbation. It lumps rope, KV write, the QSA indexer
   (pool/score/top-512) and the gather attention; it is NOT yet split. Next step is sub-stage labels inside the hook (cheap) to see whether the indexer
   or the gather dominates. ds4's WMMA attention with half-intermediate correction maps to our f16-coopmat flash/QSA kernels only partially (we already
   run f16 coopmat; the residual-correction trick is about precision of the P*V accumulate, not speed). Maps: partial. A long-context fix would pay most
   (the curve falling with context is this stage).
3. **Gated-residual machinery (17-22%).** `gr.grouprms` + `gr.down` (Q8_0 big-K GEMM) are the bulk. Timestamps say ~6% each for the two; the perturbation of
   `gr.down` is confounded. Fusing grouprms with the Q8_1 activation quantise (the decode-side fusion already exists) could give a few percent of wall.
   Maps: yes (small, 2-4% guess).
4. **GDN projections (6%) and out-proj (2.7%)** are dense GEMMs through the shared coopmat path; the scan (2.3%) is already small (chunked scan was refuted
   in the Qwen3.6 work, see memory). Nothing here is worth a dedicated kernel.
5. **PLE (3.6-4.3%)** still does the host apply step (`host.ple_apply` 1%) plus a download/upload round trip per chunk; fully on GPU would save at most the
   host part, ~1-2%.
6. **Not mapped / not worth it:** DPP-style wave reductions (the prefill stages are GEMM/coopmat bound, not reduction bound); weight prefetch/reuse is
   covered by item 1; lm_head is one row (0%).

Nothing in this profile justified a "trivially safe win" change, so none was made. The only source changes are instruments: `VulkanApi.SkipDispatches` +
`DOTLLM_VK_PERTURB=target=after1|after2` (timing attribution only, results are garbage) and the probe commands `ids`, `pf`, `perturb`.

## 5. Reproduce

```
bash scripts/gpu-lock.sh acquire <name> "reason" 600 21600
DOTLLM_VK_QWEN4EXP_CONTEXT=12288 DOTLLM_VK_QWEN4EXP_KV=f16 dotnet benchmarks/Qwen4ExpRowProbe/bin/Release/net10.0/Qwen4ExpRowProbe.dll <gguf> <workdir> <lockname>
# job file in <workdir>/jobs/<id>.job, one command per line:
ids <tokens.i32>            # made with: Qwen4ExpRowProbe mkids <gguf> <text> <out.i32> 60000
pf 8192 0 3                 # throughput: total hist reps
pf 8192 0 2 2               # GPU-timestamp stage profile
perturb grouped_gate_up tile_list_build   # then pf ... 2 ; "perturb off" to clear
```

## 6. Lever 1 follow-up: the grouped MoE GEMMs (#1348) - 32-token tile + LDS diet wins 4-9%

Measured 2026-10-27/28, UD-IQ4_XS, F16 K/V, one process, GPU lock held, no human GPU process before/after. All A/B below flip `DOTLLM_VK_MOE_IQ_VARIANT` in-process.

**The quant mix is not what section 4 assumed.** gate/up banks are IQ3_S in 47 of 48 layers (IQ4_XS in 1); down banks are IQ4_NL in 43 layers, Q8_0 in 5. The hot kernels are `moe_grouped_matmul_iq3_s_coopmat_m64` and `..._iq4_nl_...` (Q8_0-down layers keep the 16-token kernels). Banks per layer: IQ3_S 360 MB, IQ4_NL 472 MB, Q8_0 891 MB.

**Real routing (profile-only counters, wikitext, 1024-row chunk):** only **352 of 512 experts** get any row per layer-chunk. Experts by rows: 1-4: 111, 5-8: 47, 9-16: 54, 17-32: 59, 33-64: 48, 65-128: 21, 129+: 13. Tiles per layer-chunk by token-tile width: **16: 850, 32: 556, 48: 465, 64: 424, 128: 375**.

**Roofline per 1024-row chunk (205 GB/s, 59 TFLOPS f16 coopmat peak):** gate/up 0.57 s measured vs 0.117 s weight-read floor (active experts once) / 0.28 s with 850 tiles and no cache reuse / 3.2 TFLOP = 10% of peak; down 0.32 s vs 0.083 s / 0.20 s / 1.6 TFLOP = 8%. Neither DRAM traffic nor MMA is the limiter alone.

**rga (gfx1151, live driver; no spills anywhere):**

| shader | VGPR | LDS |
|---|---|---|
| iq3_s m64 (production, 16 tokens) | 96 | 15,872 B |
| iq3_s n32 (32 tokens, per-subgroup epilogue staging) | 75 | 22,016 B |
| iq3_s m64diet (16 tokens, shared 1 KB epilogue staging) | 65 | 12,800 B |
| iq3_s n32diet (shipped) | 73 | 14,848 B |
| iq4_nl m64 / n32 / diet / n32diet | 93 / 58 / 62 / 56 | 15,872 / 22,016 / 12,800 / 14,848 B |

Occupancy probe: `bal6` = the production kernel plus 6 KB of referenced-but-dead LDS (22,016 B, the n32 footprint). It ran 3-13% slower than its neighbours in a palindromic session (mean -6.5%), so LDS occupancy matters: plain n32 spends it and gains little, the diet gives it back.

**Phase perturbations (timing only, wrong output; GPU-timestamp `grouped_gate_up` / `grouped_down`, 4K, palindromic, normalised for the changed tile count; base 2540 / 1394 ms per 4K prefill):**

| removed | gate/up | down | meaning |
|---|---|---|---|
| end-of-round barrier (half the barriers) | +1% | +1% | **lever 2 (K=128, fewer barriers) refuted before building**; it would also add ~7 KB of LDS (-6.5% by `bal6`) |
| weight dequantisation (A tile written as zeros, loads kept) | -15.5% | -16% | upper bound for lever 3's dequant/LDS-write part: <= ~5% of wall; a register-built A operand would also need the implementation-defined coopmat element mapping, not built |
| global weight loads (dequant of zeros) | -33% | -26% | fetch incl. tile re-reads is ~1/3 of gate/up, already near the DRAM floor for 850 tiles |

**Shipped: `n32diet`.** The generator emits 32-token variants (`..._m64n32diet`) of the IQ3_S/IQ4_XS/IQ4_NL grouped shaders: two accumulators per subgroup share one dequantised A tile, the second token half is skipped (workgroup-uniformly) when the tile has <= 16 rows, and the epilogue stages through one 1 KB tile reused by the four subgroups in turn. A layer uses it when gate/up and down both have the variant (Q8_0-down layers stay at 16); `DOTLLM_VK_MOE_IQ_VARIANT=off` restores the 16-token kernels. Tile list tileRows = 32.

*Parity:* **bit-identical** to the 16-token kernel. Unit test (legacy grid + indirect, expert row counts 0..128 around the 16/32 edges, K up to 2560, vs the 16-token kernel `==` and vs the CPU dequant oracle) passes; end-to-end last-row logits over 2/4/3 vectors (512- and 1024-row chunks, 1K-3K tokens) gave meanKL 0, max|dlogit| 0, top-1 identical.

*Throughput (`pf`, median of 3 reps per call, palindromic base-new-new-base blocks so linear drift cancels; new/base):*

| shape | block ratios | mean |
|---|---|---|
| 4K | 1.037, 1.079, 1.083, 1.155 | **1.088** (+3.7..+15.5%) |
| 8K | 1.028, 1.046 | **1.037** (+2.8..+4.6%) |

An earlier 10-call palindromic session gave n32diet +2.5% (range wide), plain `n32` +1.2%, `diet` alone -1%, so the gain needs both parts. Absolute tok/s drifted 10-15% over every session on this box (thermal/power); only the paired ratios are meaningful. Session-1 numbers for plain n32 (off-first +3..8%, on-first -5..+9%) are the same effect plus drift.

**Not tried / left:** the remaining ~50% of gate/up time is MMA + LDS + B staging + epilogue; the IQ3_S stage load issues ~11 single-byte 32-bit loads per thread per round (the load-issue pattern from the decode GEMV work), which could be coalesced into aligned word loads; a TN=48/64 tile (424 tiles) would need a smaller LDS budget; Q8_0-down layers (5 of 48) still use the 16-token down kernel and force those layers back to 16.

Instruments: profile counters `tilesT16..T128`, `activeExperts`, `expertsWithRows[...]` (counts, not ms) in the `pf ... prof=2` stage table. Perturbation and ballast variants live on branch `issue/1348-moe-grouped-token-tile-prototype`, not in the tree.

## 7. Lever 3 follow-ups on the grouped MoE GEMMs (#1352, #1354, #1353)

Same protocol as section 6 (UD-IQ4_XS, F16 K/V, context 12288, one process, GPU lock held, no human GPU process before/after, in-process switches). Absolute tok/s drifted
~15% per session (matched calls ranged 340-430 tok/s), so only paired ratios below count: blocks of `base,new,new,base` and `new,base,base,new` (3 reps per call, median per
call, ratio = mean of the new calls / mean of the base calls inside a block).

### 7.1 IQ3_S stage load: word loads instead of single-byte loads (#1352) - +3.3% (range 0.3..6.7%), shipped

Every offset an IQ3_S thread reads is even (rows are `nsb*110` bytes, every field sits at an even offset), so the fp16 `d`, the two sign bytes and the 4 `qs` bytes are
reachable with one aligned word load each (the `qs` quad straddles at most two words and is funnel-shifted). Weight loads per thread per round: 9-11 single-byte/straddling
`readByte`/`readHalfBits` -> 6 (`d`, `qs` x2, signs, `qh`, scale byte). rga (gfx1151): ISA 7,184 -> 6,744 B, **VGPR 73 and LDS 14,848 B unchanged** (so occupancy cannot be the
reason for any difference). The change is in the shared IQ3_S body, so the 16-token `m64` kernel gets it too.

*Parity:* **bit-identical** to the byte-load kernel. During the A/B the word-load 32-token kernel was compared bitwise against the (then still byte-load) 16-token kernel by
`GroupedVariant_MatchesNarrowBitwiseAndCpuOracle` and to the CPU dequant oracle; end-to-end last-row logits over the 4 probe prompts: meanKL 0, max|dlogit| 0, top-1 identical. The test now also covers `K = 768` (`rowBytes % 4 == 2`), where odd rows start mid-word and the funnel shift has a non-zero phase.

*Throughput* (`pf 4096`, new/base per block, blocks in both orders): 1.003, 1.035, 1.067 (base first); 1.017, 1.030, 1.045 (new first). Mean **1.033**. A first 2-call
smoke gave 1.025. Consistent with section 6: removing the weight loads outright was -33% of gate/up, so trimming their issue count buys a few percent, not the whole third.

### 7.2 Q8_0 down: 32-token tile for the 5 Q8_0-down layers (#1354) - +5% (range 0.3..9.3%), shipped

`n32diet` was on only where gate/up and down both had the variant, so the 5 qwen4exp layers with a Q8_0 down bank (891 MB each) stayed on 16-token tiles. The generator now derives
`moe_grouped_matmul_q8_0_coopmat_m64n32diet.comp` from the hand-written Q8_0 shader with the same wide/diet transform (the per-expert scale epilogue is kept); same 14,848 B LDS
as the IQ variants. `MoeGroupedMatmulLegacyQuantCoopmatKernel` loads it as a variant (`RowTile` 32) and `RecordGroupedExperts` pairs it with an `n32diet` IQ gate/up on Q8_0-down
layers. `DOTLLM_VK_MOE_IQ_VARIANT=off` still restores every 16-token kernel (the Q8_0 variant only exists inside the variant branch); `DOTLLM_VK_MOE_Q8_DOWN_WIDE=0` turns off just
the Q8_0 piece for A/B.

*Path proof:* the profile's `tiles(count)` for a 2048-token prefill is 54,269 with the Q8_0 variant on (all 48 layers at 32 tokens) and higher with `DOTLLM_VK_MOE_Q8_DOWN_WIDE=0`
(the 5 Q8_0 layers back at 16-token tiles).
*Parity:* `Q8_0_WideVariant_IsBitIdenticalToNarrow` - bitwise equal to the 16-token kernel on the legacy grid and the indirect tile list, scale epilogue on and off, expert row counts
0..128 around the 16/32 edges, K up to 704; end-to-end last-row logits meanKL 0 / max|dlogit| 0.
*Throughput* (`pf 4096`, `Q8_DOWN_WIDE` 1/0, ABBA blocks): 1.003, 1.056, 1.093; mean **1.05**. The ceiling is small (5 of 48 layers) but those layers carry the largest bank, so the
share is bigger than 5/48; the first block's drift in the base arm is why the range is wide.

### 7.3 64-token tile (#1353) - +3.6% at 4K, +6.8% at 8K, shipped as the default (`n64diet`)

Token-tile width is the number of 16-token MMA halves per dequantised A fragment (`NT`). `render_nt` in the generator emits any `NT` (NT=2 reproduces `n32diet` byte for byte, checked
for IQ3_S/IQ4_NL/IQ4_XS); the A tile (9 KB, scales with weight rows) does not grow, only the staged activations do (+2.25 KB per extra half), and each thread keeps NT activation
vec4s in flight. rga (gfx1151, no spills):

| shader | VGPR | LDS | workgroups per WGP (64 KB) |
|---|---|---|---|
| iq3_s n32diet | 73 | 14,848 B | 4 |
| iq3_s n48diet | 82 | 17,408 B | 3 |
| iq3_s n64diet | 91 | 19,456 B | 3 |

So this trades one workgroup of occupancy (4 -> 3; `bal6` in section 6 measured 4 -> 2 at -6.5%) for 24% fewer tiles per layer-chunk (556 -> 424 at 1024 rows: weight read, dequantisation
and LDS staging are paid per tile). It wins anyway. `n48diet` is in the same 3-workgroup LDS class as `n64diet` with less reuse, so it was not pursued (it was generated and checked for bit-identity but
never timed). Q8_0-down layers use `moe_grouped_matmul_q8_0_coopmat_m64n64diet` (same transform); otherwise they would fall back to 16-token tiles and the shared tile list could not be used.
`DOTLLM_VK_MOE_IQ_VARIANT=n32diet` selects the 32-token kernels, `off` the 16-token ones.

*Path proof:* `tiles(count)` in the 2048-token stage profile: 54,269 at `n32diet` -> 41,861 at `n64diet` (the 64-token count measured in section 6, so every layer including the Q8_0 ones is on the
new tile list); GPU-timestamp `grouped_gate_up` 1099 -> 1025 ms and `grouped_down` 579 -> 538 ms in that single profiled pair (-6.8% / -7.1%).
*Parity:* bit-identical to the 16-token kernels - `GroupedVariant_MatchesNarrowBitwiseAndCpuOracle` and `Q8_0_WideVariant_IsBitIdenticalToNarrow` now include `n64diet`, with expert row counts
47/48/49/63/64/65 around the 48/64 edges (legacy grid and indirect list, vs the 16-token kernel `==` and vs the CPU dequant oracle); end-to-end last-row logits (4 probe prompts, 13-70 tokens):
meanKL 0, max|dlogit| 0, top-1 identical for `n32diet`, `n48diet` and `n64diet`.
*Throughput* (`pf`, `n64diet` / `n32diet`, ABBA blocks in both orders, both arms include the Q8_0 variant). 4K block ratios: 1.019, 1.022, 1.061, 1.054, 0.995, 1.042 (base first),
1.078, 1.020 (new first): mean **1.036**, 7 of 8 blocks above 1. 8K: 1.075, 1.061, mean **1.068**.

**Not tried:** a `SB=32` staging round (one block per round: A 5 KB + B 5 KB at 64 tokens = 11.3 KB, 5 workgroups per WGP) would keep occupancy at 4+ with 64 tokens, but halves the weight
bytes each thread owns per round (8 elements), which cuts against the load-issue gain of 7.1 and needs every format's `Stage` struct rewritten; the follow-up if prefill needs another step here.

## 8. Lever 2: the attention core split (#1356) - cooperative-matrix QSA gather, +10..14%

Same protocol as sections 6-7 (UD-IQ4_XS, F16 K/V, context 12288, one process, GPU lock + `cpu-bench` held for timing, no human GPU process). `attn_rope_core` is now split by
GPU-timestamp labels `ar_rope`, `ar_kvwrite`, `ar_ix_proj`, `ar_ix_pool`, `ar_ix_qrope`, `ar_ix_score`, `ar_ix_select`, `ar_gather_partial`, `ar_gather_merge`
(the old `attn_rope_core` label is now an empty residual).

**Split (scalar gather kernel, `pf ... prof=2`, % of wall):**

| sub-stage | 4K | 8K | 2K after 8K |
|---|---|---|---|
| `ar_gather_partial` (QSA gather attention, partial pass) | 16.4-19.5 | 20.2 | 20.6 |
| `ar_ix_proj` (indexer K/Q projections + raw-key store) | 2.4-2.9 | 3.3 | 3.5 |
| `ar_ix_score` | 0.2 | 0.5 | 1.1 |
| rope, kv write, pool, qrope, select, merge | each <= 0.1 | | |

The gather kernel IS the attention core: everything else together is 3-5%. Perturbation (4K, skip `ar_gather_partial`): 9310 -> 7931 ms (-14.8%, timestamp share 19.5%; routing tile count moved
-6%, so a slight understatement-of-confound). Skipping `ar_ix_score` gave -38% wall but moved the routing counter and the timestamp share is 0.2%: confounded, ignored. The other labels are below the noise floor.

**Why it is slow:** `qsa_attention_f16kv` is a *scalar* decode-style kernel (one workgroup per query head and KV split, subgroup-per-key Q.K, per-element V accumulate), not a coopmat kernel, and each of the 12 heads of
a GQA group re-reads the same gathered K/V. ~1.4 TFLOP/s effective.

**Shipped: `qsa_attention_cm_f16kv.comp`.** One workgroup per (KV head, KV split, query); the group's 12 heads are one 16-row coopmat tile (padding rows inert), the query's virtual key list is gathered
through the selection in 32-key tiles, S = Qh.K + Ql.K (Q hi/lo f16 split as in the hd256 flash kernel, K is the exact F16 cache), F32 online softmax, P rounded to f16, O += P.V in F32. Same bindings, push
constants and partial layout as the scalar kernel, so the merge kernel is reused. Used for F16 KV, head_dim 256, group <= 16 and sub-chunks of >= 16 queries; decode / short verify rows keep the scalar kernel.
`DOTLLM_VK_QSA_CM=0` (or `Qwen4ExpQsaKernels.CoopmatGather`) restores it. LDS ~28 KB.

*Parity:* not bit-identical (P f16-rounded). Unit test `VulkanQwen4ExpQsaCoopmatGatherTests` (vs the scalar F16 gather, per-(query, head) relL2, 6 shapes incl. 8/12/16-row groups, dense+sparse mix,
ragged tails, several splits, x16-amplitude scores): worst 1.8e-4..2.6e-4. End to end, real wikitext, last-row logits per chunk, cm vs scalar: 1024-row chunks / 6000 tokens meanKL 7.5e-3, maxKL 2.8e-2, top-1 6/6;
512-row / 3000 tokens meanKL 5.7e-3, maxKL 2.9e-2, top-1 6/6. Noise-floor control (scalar kernel, KV splits 1 vs 2 on 8-row chunks, 300 vectors): meanKL 2.4e-2, top-1 290/300 - the new kernel sits below a pure
summation-order change. (On the repeated probe prompts the same comparison reads meanKL 4.5e-2; that text is degenerate and amplifies any difference.)

*Throughput* (`pf`, median of 3 reps per call, palindromic ABBA blocks both orders, scalar -> cm, tok/s): 4K blocks 447.8 -> 505.6 (x1.129) and 453.5 -> 510.0 (x1.125); 8K 413.9 -> 457.2 (x1.104, wide rep spread 365-482);
2K after 8K history 383 -> 435 (x1.136). Stage profile with cm on: `ar_gather_partial` 1747 -> 1142 ms (4K, 16.4% -> 13.3%), 3991 -> 2674 ms (8K), 1280 -> 790 ms (2K after 8K). The kernel is still the single biggest attention label:
it is latency-bound per tile (3 barrier phases per 32 keys, only 2 of 4 subgroups run QK^T); headroom left in BC=64 tiles and spreading QK^T over all four subgroups.

## 9. Lever 3: gated-residual read (#1357) - premise corrected

On prefill (> 1 token) `gr.down` is the Q8_0 coopmat GEMM reading F32 activations directly: there is no Q8_1 activation quantise to fuse with `gr.grouprms` (that exists only on the 1-token decode path). The
`gr.grouprms` label is also the first stage after each submit, so it carries the inter-submit host gap (its share ranged 8-19% across identical runs: 760-2170 ms per 4K prefill), which makes it unusable as a kernel time.
The remaining candidate is the group-RMS kernel itself (scalar F32 loads, 8-barrier tree reduction); `benchmarks/Qwen4ExpRowProbe grbench` is the kernel-only microbench.
