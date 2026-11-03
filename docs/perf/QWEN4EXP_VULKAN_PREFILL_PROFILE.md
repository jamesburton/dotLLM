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

**The issue's "~132 tok/s on one long prompt" is a small-ubatch figure (#1355).** Plain whole-prompt prefill does not produce it (2.5-3.2x below the medians above), but the prefill chunk size does.
Reproduced 2026-10-31 on dev `f7083dc5` with the product path, `dotllm run <UD-IQ4_XS> -d vulkan --prompt-file <3438-token wikitext> -n 8 --temp 0` (default F32 K/V, MTP on by default, GPU lock + `cpu-bench` held,
one fresh process per row, prefill line of the run summary):

| `--prefill-chunk-size` (ubatch rows) | prefill tok/s |
|---|---|
| 0 (default: whole prompt, engine splits at the model's 1024-row forward) | 367.8 |
| 256 | 279.3 |
| 64 | 158.7 |
| 32 | 99.2 |

Throughput falls with rows per forward because every forward re-reads the full set of expert weights (MoE) and the whole GDN/attention weight set: at 32-64 rows the run is weight-bandwidth bound, not compute
bound, and ~132 tok/s sits between the 64 and 32 rows (~48 rows). Defaults do not do this (`--prefill-chunk-size` and the server's `MaxPrefillTokensPerStep` both default to 0 = whole prompt), so the 132
figure came from a run with a small ubatch (an explicit `--prefill-chunk-size`/`--ubatch-size`, or a server configured with a small prefill step cap), or from prefill that was fed in small pieces. The original run's
command line was not recorded, so this is the configuration that reproduces the number, not proof of what that run used. Ruled out: MTP (on by default here, 368 tok/s with it), F32 vs F16 K/V (368 vs 398-422
in the probe, a few %), cold first process (the run above is a fresh process, no warm-up). Not testable here: a cold OS file cache (weights paged in from NVMe during the first prefill), which can only lower a first run further.
If a 132 shows up again, check the effective ubatch first (`--prefill-chunk-size`, server `--prefill-chunk-size`, scheduler step cap).

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
**Refuted by `grbench` (kernel-only, 1024 rows x 4 x 2560 F32, 40 back-to-back dispatches, 6 rounds):** the out-of-place group-RMS runs at **0.44 ms per dispatch = 190 GB/s read+write**, i.e. at the DRAM floor
(~205 GB/s). At 96 calls per 1024-row chunk that is ~42 ms of a ~2300 ms chunk, **~1.8% of wall, already at the roofline**; a vec4 / subgroup-reduction rewrite or a fusion with the norm cannot beat the bytes it must
move (the F32 read of the residual stream and the F32 write of `_xn` stay unless the GEMM B-load absorbs the norm, which needs the per-row sum of squares first). The 8-19% `gr.grouprms` timestamp share is therefore
inter-submit host gap attributed to the first stage after each submit, not kernel time. No code change; #1357 closed as refuted.

## 10. Lever 2 follow-up: the cm gather kernel itself (#1489) - 2.4x faster kernel, +5..17% prefill

The profile (section 8) named BC=64 tiles, all-four-subgroup QK^T and fewer barriers. Measured on a kernel-only microbench (`Qwen4ExpRowProbe qsabench`: released geometry 24 heads / 2 KV heads, hd 256, 64-key
blocks, budget 8, 1024 queries after 3072 tokens, 1 split, 20 back-to-back dispatches per sample, 6 samples, alternating order, first round discarded as clock ramp; GPU lock + `cpu-bench` held), those were NOT the levers.
Every variant below is the same source with macros; all are within worst relL2 2.6e-4 of the scalar gather.

| variant (ms per dispatch, median of 6, range) | ms |
|---|---|
| shipped #1442 kernel (BC=32, unpadded, one-thread-per-row softmax) | 8.14-8.89 |
| BC=32 + vec4 staging only (pad 0, serial softmax) | 6.90 |
| BC=32 + vec4 + 16-lane softmax (pad 0) | 6.24 |
| BC=32 + vec4 + pad 8 halves (serial softmax) | 3.97 |
| **BC=32 + vec4 + pad 8 + 16-lane softmax (shipped)** | **3.30-3.37** (range 3.27-3.43) |
| BC=64 + pad 8 + 16-lane softmax (QK^T on all 4 subgroups) | 5.06 |
| BC=64, pad 0 | 8.11 |
| BC=16, pad 8 | 3.87 |
| BC=32, pad 16 | 3.50 |
| BC=32, pad 4 | broken (NaN: 17-vec4 stride breaks the 16-byte coopmat load alignment) |

2x2 on {pad 8, 16-lane softmax} at BC=32: neither 6.90, softmax only 6.24, pad only 3.97, both 3.35. **The dominant cost was LDS bank conflicts**: the K/V tile stride was 128 B and the Q tile stride 512 B,
so every coopmat load hit one bank group; an 8-half (16 B) row pad nearly halves the kernel. The serial softmax (12 threads, 32 serial exps over 256 threads) is the second lever (-15%). BC=64 is *worse* (5.06 vs 3.35):
its LDS (+8 KB for the tile, +2 KB S/P) cuts workgroup residency while the per-key barrier count it saves was not the limit. Spreading QK^T over four subgroups therefore does not pay at this LDS budget.
Scalar-gather reference on the same bench: 13.0 ms (so the cm kernel is now 3.9x the scalar one, was 1.6x). 8K history: 8.96 -> 4.03 ms; 3 KV splits: 9.53 -> 4.55 ms.

Shipped: `qsa_attention_cm_f16kv.comp` (same bindings/push constants/partial layout; vec4 f16 LDS tiles with a 16 B row pad, K/V gathered as `uvec2`, softmax as 16 lanes per head row with subgroup xor-shuffles,
softmax row state in registers; needs subgroup size >= 16). Not tried (next): lazy-rescale persistent O accumulators in coopmat registers (removes the oStage round trip and 8 of ~20 barriers per tile) and register-prefetch
double buffering of the K/V tile.

*End to end* (`pf`, median of 3 reps per call, same process, old kernel vs new alternating; absolute tok/s drifts ~10% with other activity on the box, so read the pairs): 4K old 402.4 / 412.2 vs new 441.5 / 444.8 (x1.08-1.10);
8K old 404.5 / 410.9 vs new 447.7 / 412.9 (x1.00-1.11, one new call hit the same slow phase as the old); 2K after 8K history old 343.5 vs new 401.6 (x1.17, one pair).
*Parity* (wikitext, 6000 tokens, 1024-row chunks, last-row logits, new vs old cm kernel): meanKL 2.2e-3, maxKL 8.7e-3, max|dlogit| 0.90, top-1 6/6 - below the 2.4e-2 summation-order floor of section 8.
(A `qsasplits 2` control is vacuous at 1024-row chunks: the host already uses 1 split there, KL reads exactly 0.) Unit test `VulkanQwen4ExpQsaCoopmatGatherTests` (6 shapes vs the scalar gather): pass; `VulkanQwen4ExpQsa*`: 23 passed.

## 11. Baseline against llama.cpp and the pp512 fixed-cost lever (#1537)

Measured 2026-11-01/02 on dev `59e77ae2`, UD-IQ4_XS, F16 K/V, context 12288, Radeon 8060S. GPU lock held for every GPU run, `cpu-bench` held for the llama.cpp runs. **The box was not quiet for the whole
session** (other agents' GPU/CPU jobs; the same pp512 call read 278 and 545 tok/s, one forward drifted 1.3 s -> 2.4 s), so dotLLM figures carry that spread and ratios are indicative.

**Reference engine:** llama.cpp upstream master `dde879bc` (3195), Vulkan, built from `C:\Development\llama.cpp-q4exp` (`-DGGML_VULKAN=ON`, MSVC Release). It is the only locally buildable engine that runs `qwen4exp`, and it has an MTP
graph for it (`--spec-type draft-mtp -md mtp-Qwen3.8-Flash-Next-Q8_0.gguf`). The older local llama.cpp builds (b9016, `llamacpp-vulkan`, the May checkout) do not know the architecture. ds4-ROCm numbers
(507-674 tok/s, Q2 quant, Linux ROCm) remain an unmatched published target; the local ds4 build is DeepSeek-V4 only.

| metric | dotLLM (reps / range) | llama.cpp dde879bc | ratio | code path |
|---|---|---|---|---|
| pp512 | median 278-288 clean (240-394); 545 once warm; 356 (332-427) first session | 446 +- 2 (`llama-bench -fa 1 -ub 1024 -r 3`) | ~0.62-0.80 (1.2 on the warm outlier) | `Qwen4ExpRowProbe pf` (chunk 1024, last-row logits) |
| pp2048 | median 440-450 (407-533) | 498 +- 1 | ~0.88 | same |
| pp8192 | median 300 (288-340) | 306 +- 51 | ~0.98 | same |
| plain decode | 24-25 tok/s early (39.7-42 ms), 19.4 later in the contended phase | 22.6 (`llama-bench tg64`), 23.1 (`llama-cli`) | 1.05-1.10 early | probe `specp` plain arm / `fwd 1` |
| MTP K=3 / K=4 (vs own plain) | K=3 1.20 1.21 1.54 1.39; K=4 1.27 1.12 1.68 1.21 (4 prompts, 128 tok) | K=3 26.0 vs 23.1 plain (1.13x); K=4 21.0 (0.91x); ONE prompt, one run | ~1.13x (K=3), ~1.34x (K=4) | `specp all 128 K` / `llama-cli -md ... --spec-draft-n-max K` |

Reading: prefill is at parity with llama.cpp at 8K, ~0.9x at 2K and behind at 512 (a short prompt amortises fixed per-forward cost over few rows); decode and MTP are ahead. The llama.cpp MTP comparison is a single
prompt per K (its own run-to-run MTP variance is large; see the Qwen3.8-27B notes), so treat it as directional. The probe's greedy MTP text diverged from plain at near-ties on the short prompts and was identical at 1K context, as in `docs/SPECULATIVE.md`.

### 11.1 pp512 fixed per-forward cost: REFUTED (no shippable lever)

Hypothesis: at 512 rows the per-forward constants (96 half-layer submits with fence waits, the host PLE step, n-gram page-in) cost enough to explain the 0.62-0.80 ratio. In-process A/B, 512-row forward (`abfwd 512 8 0 <Prop> a b`, arms interleaved
after a warm-up, both orders), minimum ms of 8 reps:

| switch | A | B | verdict |
|---|---|---|---|
| `SplitAbove` 16 vs 100000 (one command buffer) | 1276 / 1272 | 1281 / 1402 (the 16 arm in pair 2) | pair 1 neutral, pair 2 single-submit ~9% faster in min and not in median: noise-bound |
| `PleOnGpu` 1 vs 0 | 1684 / 1543 | 2826 / 2206 | already the default; the GPU PLE projection is a 30-40% win at 512 rows |
| `NgramPrefetch` 1 vs 0 | 2290 | 2298 | neutral |

A single command buffer for a 512-row forward is ~1.3 s of GPU work, uncomfortably near the driver watchdog (that is why each half layer is submitted separately above 16 rows), so even the best case is not a safe default.
The submit gaps are ~0.1 ms x 96 = ~10 ms of a 1.0-1.8 s forward (<1%); the per-stage profile (`pf 512 prof=2`) puts the PLE host apply at ~2% and `~ple` at ~2.2%. The 2x spread of identical calls was environmental. No code change.
What is left at 512 rows is the weight-read floor: every one of the 352+ active experts per layer is read once for few rows, which is the ceiling `--prefill-chunk-size` experiments in section 1 already showed.

Tooling note for next time: a `quit` line inside a probe job file makes the probe exit before later job files are read; put `quit` in its own final job.

### 11.2 MTP upward depth probe (#1335): no measured end-to-end win, not shipped

Implemented on branch `issue/1335-mtp-upward-probe` (PR #1536, closed unmerged): after 6 consecutive fully accepted rounds at the top K the controller raises its ceiling one step (qwen4exp declares a limit of 7) and the cost model keeps the deeper K
only if its measured ms/token is lower; unit-tested with five mutants killed. Motivation, fixed-K sweep (UD-IQ4_XS, real Q8_0 head, greedy 192 tok, plain and MTP in one process, contended box so ratios only): the one high-acceptance
prompt scored 1.98x (K=4) -> 2.17x (K=6) -> 2.33x (K=7), the other three prompts lost from K=5 (0.82-1.38x).

End to end (`dotllm run -d vulkan --mtp-head ... -n 384 --temp 0`, UD-IQ4_XS, `DOTLLM_MTP_UPWARD_PROBE` 1 vs 0, one fresh process per run, order code1 code0 prose1 prose0 prose0 prose1 code0 code1, GPU lock + `cpu-bench` held, box otherwise idle,
decode tok/s of the run summary):

| prompt | probe on | probe off | on / off (means) |
|---|---|---|---|
| code (BST + unittest) | 35.64, 31.89 | 35.07, 33.39 | 0.99 (ranges overlap) |
| prose (router training) | 29.38, 26.30 | 26.73, 26.89 | 1.04 (ranges overlap) |

Verdict: within the run-to-run spread (the on-arm pairs themselves differ by 4-3.5 tok/s); no win to ship. **Path proof is missing**: `DOTLLM_MTP_TRACE` was off, so these runs do not show that the ceiling ever rose; a single request
rarely sustains 6 fully accepted K=4 rounds (histograms in section "Making MTP a default win" show 5-of-5 accepted in about a third of rounds on the best prompt), and the benefit only exists above the fixed K=4 that the controller already starts at.
A re-test needs a trace of the chosen K per round (or a counter) on a high-acceptance long generation; until then the fixed `MtpPreferredDraftTokens = 4` stands.

### 11.3 UD-Q4_K_XL verify rows against llama.cpp (#1555): parity; the #1511/#1520 structure does not apply

UD-Q4_K_XL (79 GiB; dense projections, shared experts and the hyper-connection projections are Q8_0, routed gate/up Q4_K, routed down Q5_1), Vulkan, F16 K/V, context 12288, no MTP head attached, reference llama.cpp `dde879bc`
(`llama-bench -fa 1 -p 1,2,3,5,512 -n 32 -r 5`; its `pp N` is an N-token batch forward, the same shape as an N-row verify). Locks: Strix GPU + memory (+ `gpu-lock.sh`), one process per engine, never resident together.

| rows | dotLLM `fwd` ms (median, min-max, 20 reps, ctx 64) | llama.cpp ms (from tok/s) | dotLLM / llama.cpp |
|---|---|---|---|
| 1 | 41.6 (40.4-44.4) | 45.7 (21.87 +- 0.09 tok/s) | 0.91 |
| 2 | 51.6 (47.7-77.5) | 51.4 (38.92 +- 0.39) | 1.00 |
| 3 | 58.3 (55.5-61.4) | 60.7 (49.41 +- 1.12) | 0.96 |
| 5 | 78.3 (72.3-106.9) | 77.6 (64.44 +- 2.68) | 1.01 |
| decode | ~24 tok/s (1 row) | tg32 19.41 +- 1.93 | ~1.2 |
| prefill | pp2048 482 (465-535), `pf` chunk 1024 | pp512 281 +- 43 (different length; llama.cpp's own spread is 15%) | not matched |

Code path: `Qwen4ExpRowProbe fwd <rows> 20 64` (one command buffer for <= 16 rows; fused single-token MoE chain at 1 row, indexed multi-row MoE MMVQ at 2..8). 1K context: 5 rows 124.7 ms (115.6-341.3), 1 row 82.3 ms.
A 5-row verify therefore costs 1.88x of a 1-row forward at ctx 64 (1.5x at 1K), matching llama.cpp's shape; there is no row-count cliff to fix on this file.

**Why the #1511/#1520 structure does not apply here.** Its win is sharing one weight read across up to 16 activation columns of the SAME weight matrix. On this file (a) every dense projection is Q8_0, whose multi-column kernel was measured at the DRAM ceiling in
#1515/#1516 (200+ GB/s through 12 columns, all structures refuted), and (b) the routed experts, the dominant verify cost, are different weights per row (10 of 512 experts per row).

Split-submit stage profile at 5 rows (distorted by per-stage syncs: forward 181 ms against 78 ms unprofiled; read ratios only): `gate_up` 32.0 ms vs 10.5 ms at 1 row (3.0x), `down` 16.3 vs 6.4 (2.6x), `gdn_proj` 10.7 vs 9.9 (1.1x),
`gdn_outproj` 5.7 vs 5.8, gated-residual stages flat. So the 1 -> 5 row growth is the MoE experts: five rows touch up to 5x as many experts; the shared dense stages are free. The remaining lever is the expert read itself (overlapping the
5 rows' expert tiles with the dense work, or the expert-grouped path that already exists from 16 rows) - open, not attempted; the dense side has nothing left. Not measured here: MTP ratios on this file with the current build (last measured 1.33-1.38x, `docs/SPECULATIVE.md`).
