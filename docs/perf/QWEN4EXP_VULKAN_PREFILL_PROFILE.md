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
