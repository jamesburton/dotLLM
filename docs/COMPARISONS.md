# dotLLM vs llama.cpp — consolidated comparisons

One page for every published dotLLM-vs-llama.cpp figure, per model and backend. **Except for the LFM3 sections (2.10, 4.3),
which were measured for the PRs they cite, nothing here was re-measured for this page**: every other number is copied from the document or PR named in its "Source" column, together with that source's hardware,
build hash and method. Where a source quotes a range, the range is kept (the boxes are shared and llama.cpp's Vulkan
prefill is noisy; see [Caveats](#caveats)). Ratios are `dotLLM / llama.cpp`, so **> 1 means dotLLM is faster**.

Last consolidated against `dev` at `59e77ae2` (2026-11-01). Figures older than that are dated in their row. Items that
could not be sourced, or that predate large changes, are listed under [Needs re-measure](#needs-re-measure).

Refreshed 2026-11-02 (Flint smoke, pruned Qwen3.8-Flash-Next Coder, Flash-Next UD-IQ4_XS, tiled dense prefill; sections 2.2 and 2.6-2.9): those figures were measured on Strix Halo Vulkan that week against llama.cpp upstream master `dde879bc` (Vulkan) unless stated, on a noisy box, so only ranges are quoted.

How to read the build columns: `0ae0e1976` = llama.cpp master (Vulkan `llama-server` / `llama-bench`), `b11739` =
llama.cpp CPU (Bolt), `b9016` = llama.cpp CPU `llama-server --embeddings`, `b9041` = llama.cpp CPU/CUDA of the May-2026
baseline, `b87ca7225` = the PrismML llama.cpp fork (CUDA; the only llama.cpp that loads PQ2_0).

## 1. Summary

| Model / quant | Backend, hardware | Workload | dotLLM vs llama.cpp | Section |
|---|---|---|---|---|
| Qwen3.8-27B Q4_K_M | Vulkan, Strix Halo 8060S | plain decode | 0.97 (0.93-1.02) | [2.1](#21-qwen38-27b-q4_k_m-vulkan-strix-halo-8060s) |
| Qwen3.8-27B Q4_K_M + MTP head | Vulkan, Strix Halo | MTP K=2 / 3 / 4 | 1.00 / **1.14** / **1.20** | [2.1](#21-qwen38-27b-q4_k_m-vulkan-strix-halo-8060s) |
| Qwen3.8-27B Q4_K_M + DFlash2 | Vulkan, Strix Halo | DFlash K=3 | 1.03 (1.01-1.06) | [2.1](#21-qwen38-27b-q4_k_m-vulkan-strix-halo-8060s) |
| Recursive Flint Q4_1 (39.8B MoE) | Vulkan, Strix Halo | pp512 / decode @0.5K..32K | ~1.0-1.3 / 1.10-1.26 .. 1.06-1.15 | [2.2](#22-recursive-flint-q4_1-vulkan-strix-halo) |
| Recursive Flint Q4_1 | Vulkan, Strix Halo, smoke (2026-11) | pp512 / tg64 / 128K tg64 | no llama.cpp comparison (dotLLM only) | [2.2](#22-recursive-flint-q4_1-vulkan-strix-halo) |
| Qwen3.8-Flash-Next pruned Coder (IQ1_M dir, 58.4 GB) | Vulkan, Strix Halo | pp512 / pp2048 / tg64 / MTP K=4 | 1.14 / 1.14 / ~0.93-1.02 / 1.38-1.53x own plain (no llama.cpp MTP) | [2.7](#27-qwen38-flash-next-pruned-coder-vulkan-strix-halo) |
| Qwen3.8-Flash-Next UD-IQ4_XS | Vulkan, Strix Halo | pp512 / pp2048 / pp8192 / decode | 0.62-0.80 / ~0.88 / ~0.98 / 0.84-1.10 | [2.8](#28-qwen38-flash-next-ud-iq4_xs-vulkan-strix-halo) |
| Dense Qwen3-4B Q4_K_M (tiled GQA prefill, #1553) | Vulkan, Strix Halo | dotLLM pp512 / pp2048 vs llama.cpp -fa 1 -ub 1024 | ~1.2-1.4 / ~1.2-1.4 (derived from ranges, see 2.9) | [2.9](#29-dense-vulkan-prefill-after-pr-1553-tiled-gqa-flash) |
| Recursive Flint Q4_1 | CUDA, RTX 3060 12 GB, `--n-cpu-moe` | pp512 / tg64 | 1.00-1.05 / 0.9-1.1 (parity) | [3.1](#31-recursive-flint-q4_1-on-an-rtx-3060-12-gb-cuda-moe-offload) |
| Ternary Bonsai-2 27B PQ2_0 | CUDA, RTX 3060 | pp512 / tg128 / MTP K=3 | 1.01-1.02 / 1.05-1.06 / 1.11-1.15 (vs PrismML fork) | [3.2](#32-ternary-bonsai-2-27b-pq2_0-cuda-rtx-3060) |
| Nemotron 3.5 Bolt 74B-A9B (AtomicChat quants) | CPU, Core Ultra 7 155H | pp512 / tg128 | 0.22-0.41 / 0.37-1.02 | [4.1](#41-nemotron-35-bolt-74b-a9b-cpu) |
| Nemotron 3.5 Bolt 24GB + LiLiCorr | Vulkan, Strix Halo | plain / speculative | 1.02-1.04 / 0.74-0.78 | [2.3](#23-nemotron-35-bolt-24gb-vulkan-strix-halo) |
| Qwen3.6-35B-A3B UD-Q4_K_M | Vulkan, Strix Halo | plain / MTP K=2..4 (opt-in) | 0.88 / 0.62-0.85 (superseded, see 2.4) | [2.4](#24-qwen36-35b-a3b-vulkan-strix-halo) |
| LFM3-1.7B Q8_0 | Vulkan, Strix Halo 8060S | pp512 / pp2048 / tg128 | 0.86 / 0.72 / 0.94-0.96 | [2.10](#210-liquid-ai-lfm3-17b-q8_0-vulkan-strix-halo) |
| LFM3-1.7B Q8_0 | Vulkan, Strix Halo | pp16384 / pp32768 / tg @32K | before #1572: **0.31 / 0.21 / 0.47**; after the sliding-window fix and #1666 (contended session, 2.10): ~0.5-0.63 / ~0.46-0.48 / ~1.1 | [2.10](#210-liquid-ai-lfm3-17b-q8_0-vulkan-strix-halo) |
| LFM3-1.7B Q8_0 / Q4_K_M | CPU, Ryzen AI Max+ 395 (32 thr) | pp512 / tg128 | **0.41 / 0.87** (Q8_0), **0.28 / ~1.0** (Q4_K_M) | [4.3](#43-liquid-ai-lfm3-17b-cpu) |
| Llama-3.2-1B Q4_K_M / pure Q3_K | CPU | perplexity gap | +0.032 % / +0.093 % | [5](#5-quality-parity) |
| Embedding encoders (bge, MiniLM, nomic, ...) | CPU (32 thr) | docs/s | ~0.4-0.6 (indicative, derived) | [6](#6-embeddings) |

Reading the summary: on Strix Halo Vulkan the 27B dense model and Flint are at parity or ahead on everything except plain
1-row decode of the 27B (-3 %, the Q4_K / Q6_K / Q8_0 1-row GEMVs). dotLLM is behind on CPU prefill of the very large MoE
(Bolt) and on speculative decoding of the Nemotron-H MoE, and the embedding encoders have not been compared against
llama.cpp since their large speed-ups.

## 2. Strix Halo, Vulkan (AMD Ryzen AI Max+ 395, Radeon 8060S, gfx1151, UMA)

llama.cpp is always master `0ae0e1976` (Vulkan), run interleaved with dotLLM on the same box, same file, with the GPU lock
and the `cpu-bench` lock held (UMA decode swings by up to ~40 % with CPU bandwidth contention, so only same-session pairs
are comparable).

### 2.1 Qwen3.8-27B Q4_K_M, Vulkan, Strix Halo 8060S

Source: [SPECULATIVE.md](SPECULATIVE.md), "Re-measure vs llama.cpp after #1517 and #1520" (issue #1220, dotLLM `dev` at
`fc8fa0b0`). Files: `ggml-org` Qwen3.8-27B Q4_K_M + `mtp-Qwen3.8-27B-Q8_0.gguf` head, and DFlash2 Q4_K_M. 4 prompts x 128
greedy tokens, `DOTLLM_MTP_ADAPTIVE=0`, one process per run, llama.cpp `llama-server` (`cache_prompt=false`, `top_k 1`).
3 reps, pair order reversed in rep 2. The machine rebooted mid-sweep, so the rep spread is wider than in the earlier table.
Cells are the mean of the 4 prompts' tok/s; ranges are min..max over the 3 reps of that mean.

| arm | dotLLM tok/s | llama.cpp tok/s | dotLLM / llama.cpp (per-rep range) | peak private / GPU shared MiB: dotLLM | llama.cpp |
|---|---|---|---|---|---|
| plain | 8.97 (8.52-9.42) | 9.26 (8.88-9.65) | 0.97 (0.93-1.02) | 19,210 / 18,551 | 19,499 / 18,636 |
| MTP K=2 | 20.04 (19.69-20.39) | 20.18 (19.33-21.44) | 1.00 (0.92-1.04) | 19,693 / 19,004 | 23,707 / 22,150 |
| MTP K=3 | 24.55 (23.99-24.83) | 21.50 (20.17-22.88) | **1.14** (1.08-1.23) | 19,694 / 19,013 | 24,129 / 22,301 |
| MTP K=4 | 26.30 (24.00-29.25) | 21.89 (20.75-23.37) | **1.20** (1.16-1.25) | 19,700 / 19,018 | 24,280 / 22,462 |
| DFlash2 K=3 | 26.55 (24.55-28.91) | 25.83 (24.34-28.33) | 1.03 (1.01-1.06) | 20,374 / 19,684 | 22,336 / 21,036 |

- Absolute tok/s in this session are about 14 % lower for **both** engines than in the preceding (#1510) session (plain
  8.97 / 9.26 vs 10.44 / 10.73): compare the ratios, not the absolutes, across the two tables.
- How the ratios moved: before #1510 (MTP K=4 0.93, K=3 1.00, K=2 0.96, DFlash K=3 0.94, plain 0.97) -> after #1510
  (K=4 **1.14**, K=3 1.02, K=2 0.94, DFlash 0.95; K=4 27.65 vs 24.32 tok/s) -> after #1517 / #1520 (the row above).
  The lever was Q4_K multi-column GEMVs: 150 -> 190-204 GB/s at 2..16 columns ([SPECULATIVE.md](SPECULATIVE.md), #1511),
  then the Q5_K structure (#1515) and the Q4_K tables (#1516).
- Memory: dotLLM is 0.3-3.9 GiB lower in every arm (private bytes) (the llama.cpp MTP head also loads its own `token_embd` / `output`).
- Per prompt, MTP K=4 is 1.04 / 1.44 / 1.11 / 1.39. Prompts 0 and 2 degenerate into repetition (acceptance ~0.95+ for both
  engines); prompt 1 is the non-degenerate one. llama.cpp's own MTP varies run to run (K=4 20.8-23.4).
- Greedy identity: every dotLLM MTP / DFlash run equals dotLLM plain on the first 300 characters (60 / 60 including plain).
  llama.cpp's own text differs from dotLLM plain on some prompts.
- Remaining gap: plain 1-row decode only (0.96-0.99 per prompt). Open lever: the 1-row Q4_K / Q6_K / Q8_0 GEMVs (~205 GB/s)
  and the 248k-row Q6_K LM head (4.8 ms of ~105 ms). The follow-up profile is PR #1526 / issue #1525.

### 2.2 Recursive Flint Q4_1, Vulkan, Strix Halo

Source: [GPU.md](GPU.md), "Recursive Flint on the Strix 8060S (Vulkan, #1161)". Official Q4_1 file (39.8B MoE, 2.5B
active), chunk / ubatch 512, F16 KV with the #1176 sliding-window ring, `llama-bench -ngl 99 -fa 1`, interleaved, two
rounds, warm (the first round of each engine runs slower).

| | dotLLM before (#1200) | dotLLM (#1161) | llama.cpp `0ae0e1976` | ratio range (dotLLM min/llama max .. dotLLM max/llama min) |
|---|---|---|---|---|
| pp512 | 1,004-1,239 | **1,588-1,994** | 1,560-1,623 | 0.98-1.28 |
| pp8K | 981 | **1,455-1,768** | 1,304-1,385 | 1.05-1.36 |
| pp32K | 620 | **1,138-1,206** | 936-938 | 1.21-1.29 |
| decode @ 0.5K | 70-72 | **81-90** | 71.6-73.7 | 1.10-1.26 |
| decode @ 8K | 54 | **72-77** | 63.5-65.4 | 1.10-1.21 |
| decode @ 32K | 28.2 | **55.8-60.0** | 52.0-52.8 | 1.06-1.15 |
| device weights + scratch (MiB) | 25,001 | **23,828** | 23,611 model + 343 compute (+ 1,173 host) | |
| private working set (GiB) | 25.5-26.7 | 24.4-25.1 | 24.1 (`-lm mmap`) / 25.2 (`-lm none`) | |

- The pp512 lower bound overlaps llama.cpp's range: read prefill at 512 as "parity to ahead", and the long-context rows as
  clearly ahead.
- A separate llama.cpp run on the same file (GPU.md, "host import"): pp512 1,633-1,670, tg64 73.2-73.4.
- Accuracy at depth: per-row KL against the CPU model at 8K is 0.4-1 for every Vulkan variant, and llama.cpp CPU vs
  llama.cpp Vulkan shows the same profile (Q8_1 activation rounding amplifies a 1e-6 perturbation; [PRECISION.md](PRECISION.md) E14, #1218).
- **Later dotLLM-only smoke (2026-11, Vulkan, Strix Halo)**: pp512 1,967-1,988 tok/s, tg64 90.0-90.4 tok/s,
  128K-context tg64 31.8 tok/s, ~29 GiB peak process memory; the server and `--device auto` both work. Greedy parity
  against CPU diverges at a near-tie after 13 tokens (the #1218 behaviour). **No llama.cpp comparison exists for
  these figures** (likely unsupported or not tried; needs re-measure): the ratios above remain the #1161 `0ae0e1976` ones.
  See [Needs re-measure](#needs-re-measure).

### 2.3 Nemotron 3.5 Bolt 24GB, Vulkan, Strix Halo

Source: [SPECULATIVE.md](SPECULATIVE.md), "LiLiCorr drafter for Nemotron-3.5 Bolt" (issues #1237, #1359, #1365, #1368).
`NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr-GGUF`, K = 7, 128 greedy tokens, medians of 3 interleaved reps, p0 = code
prompt (high acceptance), p1 = prose (3.3-3.75 tokens/round). llama.cpp `draft-dflash` + LiLiCorr, same file.

| Strix, Bolt 24GB | p0 tok/s | p1 tok/s | ratio p0 / p1 | peak private / GPU shared MiB |
|---|---|---|---|---|
| dotLLM plain | 30.66 | 30.36 | 1.04 / 1.02 | 21,853 / 21,180 |
| llama.cpp plain | 29.59 | 29.66 | | 21,865 / 21,359 |
| dotLLM + LiLiCorr (after #1365) | 44.27 | 23.00 | 0.74 / 0.78 | 22,894 / 22,178 |
| llama.cpp `draft-dflash` + LiLiCorr | 60.22 | 29.38 | | 24,970 / 24,068 |

dotLLM speculation beats its own plain decode on code (1.44x) but not on prose; llama.cpp wins both. The gap is the 8-row
verify (81 ms, 2.5x a decode step; the indexed routed experts barely share reads across 8 tokens) plus ~45 ms of host-side
LiLiCorr head per round. Memory is 2.0 GiB lower (private) / 1.8 GiB lower (GPU shared) than llama.cpp. The #1420 / #1459
small-row GEMV work came after this table (Nemotron-3-Nano-4B only) and was not re-compared; see
[Needs re-measure](#needs-re-measure).

### 2.4 Qwen3.6-35B-A3B, Vulkan, Strix Halo

Source: [SPECULATIVE.md](SPECULATIVE.md), "`qwen35moe` MTP head (issue #1188)". UD-Q4_K_M, 4 prompts x 128 tokens, MTP is
**opt-in** (`DOTLLM_VK_QWEN35MOE_MTP=1`) and was neither exact nor a clear win when measured. Superseded in part by later
MoE verify work (#1362, #1374); not re-compared.

| arm | dotLLM tok/s | llama.cpp tok/s | ratio |
|---|---|---|---|
| plain | 50.7 | 57.5 | 0.88 |
| MTP K=2 / 3 / 4 | 49.0 / 52.1 / 55.1 | 72.2 / 84.2 / 64.6 | 0.68 / 0.62 / 0.85 |

### 2.5 Models with no llama.cpp comparator on Strix

Bonsai-2 27B PQ2_0 (llama.cpp master cannot load PQ2_0), the qwen4exp family (the pinned llama.cpp `3cf0325` has no
qwen4exp MTP graph), and the Vulkan embedding encoders (no llama.cpp Vulkan embedding run exists). Their absolute numbers
are in [SPECULATIVE.md](SPECULATIVE.md), [GPU.md](GPU.md) and [EMBEDDINGS.md](EMBEDDINGS.md).

### 2.6 Notes on the 2026-11 refresh sections

Sections 2.7-2.9 are copied from the week's measurements (llama.cpp `dde879bc` Vulkan, `-fa 1 -ngl 99` unless stated,
Windows, shared noisy box; the same pp512 call drifted 278-545 tok/s within one session in 2.8). Treat every ratio as a
range, not a point.

### 2.7 Qwen3.8-Flash-Next pruned Coder, Vulkan, Strix Halo

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`, IQ1_M directory: 256 of 512 experts kept, mixed tensor types
(IQ3_XXS / IQ4_NL / Q2_0 / IQ4_XS / Q4_K / Q6_K / BF16), 58.4 GB in two shards.

| workload | dotLLM | llama.cpp `dde879bc` | ratio (derived) |
|---|---|---|---|
| pp512 (tok/s) | 469 | 412 | 1.14 |
| pp2048 | 496 | 436 | 1.14 |
| ~8K prefill | 249-272 (p7900) | 340 +/- 101 (noisy) | not meaningful (ranges overlap) |
| tg64 (tok/s) | 21.7-23.2 | 22.7-23.3 | ~0.93-1.02 (parity) |
| MTP K=4 (tok/s) | 27.3 and 30.1 (1.38x and 1.53x over dotLLM plain; acceptance 78% / 83%) | no MTP for this arch | n/a |

- MTP needs the #1530 fix for the expert-count mismatch. Two prompts only.
- CPU: dotLLM oracle 1.8 tok/s plain, 3.9 with MTP, vs llama.cpp CPU 9.6-11.2 tok/s.
- CPU greedy vs llama.cpp over 24 tokens: 1 of 3 prompts identical, the others diverge at tokens ~3 and ~12. The
  F32-decoded control was **not run**, so the divergence is not attributed.
- Memory ~63 GB total (58.4 GB weights including a 28.8 GB mapped n-gram shard, plus 4.7 GB scratch). GPU heap was not measured.

### 2.8 Qwen3.8-Flash-Next UD-IQ4_XS, Vulkan, Strix Halo

Source: `docs/perf/QWEN4EXP_VULKAN_PREFILL_PROFILE.md`, section 11. Baseline against llama.cpp `dde879bc`.

| workload | dotLLM / llama.cpp |
|---|---|
| prefill pp512 | 0.62-0.80 (one warm outlier 1.2) |
| pp2048 | ~0.88 |
| pp8192 | ~0.98 |
| plain decode | 1.05-1.10 early in the session, 0.84 in a late drifted run |
| MTP K=3 | 1.13 (llama.cpp, one prompt) |
| MTP K=4 | ~1.34 (directional) |

The box drifted 278-545 tok/s for the same pp512 call, so the early/late split is partly contention; re-measure
interleaved before quoting.

### 2.9 Dense Vulkan prefill after PR #1553 (tiled GQA flash)

The tiled GQA flash kernel is now the ordinary prefill attention; `DOTLLM_VULKAN_ATTN_PREFILL=ref` restores the old path.
Qwen3-4B Q4_K_M, full prefill GPU time with 1024-row chunks:

| context | before | after |
|---|---|---|
| 1.9K tokens | 1.43 s | 0.90 s |
| 3.7K | 4.0 s | 2.0 s |
| 7.4K | 14.7 s | 4.7 s (3.1x) |
| 14.8K | ~71 s | 13.2 s |
| 33K | ~305 s | 67 s |

llama.cpp Vulkan `-fa 1 -ub 1024`, same model: pp512 1,650-1,815, pp2048 1,467-1,602, pp8192 1,086-1,202 tok/s; dotLLM at
7.4K is ~1,580 tok/s of GPU time (about 1.3-1.5x the llama.cpp pp8192 range; derived by this page, indicative). The
Summary ratio is likewise derived from the ranges (dotLLM pp512 2,240-2,357 and pp2048 1,939-2,090 below).

pp512 / pp2048 tok/s, new vs old:

| model | new pp512 / pp2048 | old pp512 / pp2048 |
|---|---|---|
| Qwen3-4B Q4_K_M | 2,240-2,357 / 1,939-2,090 | 1,893-2,019 / 1,170-1,239 |
| Llama-3.2-3B Q8_0 | 2,132-2,205 / 1,999-2,150 | 1,933-2,003 / 1,483-1,510 |
| Llama-3.2-1B Q8_0 | 5,701-6,063 / 5,583-6,091 | 5,235-5,368 / 4,383-4,513 |
| Qwen2.5-0.5B Q8_0 | 14,966-15,666 / 12,814-14,020 | 12,511-13,273 / 8,764-9,727 |

Not run: the hybrids (Qwen3.8-27B, Qwen3.6) take a separate hd256 path, so the section 2.1 figures are unaffected and
unrefreshed. llama.cpp was measured only for Qwen3-4B.

### 2.10 Liquid AI LFM3-1.7B Q8_0, Vulkan, Strix Halo 8060S

Source: PR #1573 / issue #1572 (measured 2026-11-03 for this section; dotLLM at `issue/1562-lfm3-vulkan` `6b836e51`). `LiquidAI/LFM3-1.7B-GGUF`
`LFM3-1.7B-Q8_0.gguf`; llama.cpp master build `dde879bc` `llama-bench` (Vulkan, defaults, `-r 3`; `-r 2` for the 16K/32K rows); dotLLM
`dotllm bench -d vulkan -n 64|128 -r 3` (`-r 2` deep). `Strix:GPU` and `cpu-bench` held, ollama idle, llama.cpp and dotLLM run back to back
in one session (not interleaved: the lock was held for the whole block). Ranges are stdev of the reps for llama.cpp (as `llama-bench` prints
them) and min-max for dotLLM. LFM3 is a conv + sliding-window (256) + global-attention hybrid, so context should be cheap here.

| test | dotLLM | llama.cpp | dotLLM / llama.cpp |
|---|---|---|---|
| pp512 | 4739-4757 | 5504 (+/- 114) | 0.86 |
| pp2048 | 3997-4047 | 5562 (+/- 51) | 0.72 |
| pp4096 | 3307-3323 | not measured | |
| pp16384 | 1421-1423 | 4533 (+/- 26) | 0.31 |
| pp32768 | 763-775 | 3719 (+/- 18) | 0.21 |
| tg128 @ 0 | 93.3-95.6 | 98.3-99.7 | 0.94-0.96 |
| tg64 @ 4K | 77.5-78.7 | 92.5 (+/- 1.3) | 0.84 |
| tg64 @ 16K | 55.5-57.2 | 89.8 (+/- 0.6) | 0.62 |
| tg64 @ 32K | 38.5-39.7 | 82.4 (+/- 0.3) | 0.47 |

- The 2026-11-03 table above is the unpatched baseline: sliding layers walked the whole KV there (fixed, see the update below). llama.cpp
  loses 16 % of its decode rate from 0 to 32K, dotLLM 59 %.
- dotLLM's `tg @ N` rows decode after an N-token prefill (`-p N`); llama.cpp's use `-d N` (prefill the depth, then decode). Same context.
- **Update 2026-11-07 (issue #1572, patched: windowed layers clip the single-row split-KV range and prefill on the GQA flash kernel, plus #1666 row-sliced long-prefill attention).** dotLLM at branch `issue/1572-lfm3-sliding-kvstart` (dev `3abab54f` + the #1572 patch), llama.cpp `dde879bc` (`llama-bench`, Vulkan, `-r 2`), same session, `Strix:GPU` held, one run after the other, **box contended** (about 78 % CPU from other users' python jobs for the whole block, so llama.cpp's own tg varies 81.8 +/- 4.6 at depth 0 against 98-100 on 2026-11-03 and its pp512 is 4622 against 5504). Same-session ratios only; dotLLM ranges are min-max of the reps (3 reps to 8K, 2 beyond, 1 at 48K/64K).

| test (Q8_0) | dotLLM patched | llama.cpp (same session) | ratio |
|---|---|---|---|
| pp512 | 4887 (1 run, `-r 3` best) | 4622 (+/- 29) | ~1.06 |
| pp2048 | 4380 | 4284 (+/- 88) | ~1.02 |
| pp16384 | 1570-1948 (two sessions of runs) | 3074 (+/- 89) | 0.51-0.63 |
| pp32768 | 1080-1123 | 2342 (+/- 27) | 0.46-0.48 |
| tg64 @ 4K | 85.0-85.9 | 72.8 (+/- 8.6) | ~1.2 (noisy) |
| tg64 @ 16K | 61-71 (tg varies run to run with the contention) | 60.7 (+/- 10.4) | ~1.0-1.2 (noisy) |
| tg64 @ 32K | 57.2-57.6 | 50.3 (+/- 4.5) | ~1.1 (noisy) |

  Against the 2026-11-03 numbers (same dotLLM prefill path, unpatched, 763-775 at 32K and tg 38.5-39.7 at 32K) the patch is about 1.45x at pp32768 and 1.45-1.5x at tg @32K. Paired same-session A/B of the patch switch (`DOTLLM_VK_LFM3_SLIDING_OPT=0` restores the old path), pp16384, order 0,1,0,1: old 1082-1206 tok/s, patched 1570-1797. The prefill gap to llama.cpp that remains at 16K+ is in the 6 global-attention layers (O(n^2), per-head flash at head_dim 128) and the rest of the layer stack; not addressed here.
- **Long-prefill ladder, patched (dotLLM only, `-n 64`)**: pp 28672 1261-1268 (the length that lost the device before #1666), 32768 1080-1123, 49152 814 (1 rep, tg 32.8), 65536 522 (1 rep, tg 28.5). No device loss, no TDR, up to 64K. A 96K+ run was not done. Q4_K_M: pp512 4006, pp2048 3576, pp16384 1220-1365, pp28672 997-1029 (llama.cpp Q4_K_M same session: 3419 / 3127 / 2554 / 2196, so Q4_K_M is 1.17x / 1.14x at short context and 0.48x / 0.47x at 16K / 28K). BF16 runs but the BF16 prefill path is slow (pp512 128 tok/s); only pp512 measured.

- Quality: Q8_0 vs llama.cpp CPU on identical ids, mean KL 1.96e-3, top-1 98.24 %, PPL 25.9396 vs 25.9400 (section 5 method). After the #1572 patch (wikitext-2 LF, llama.cpp CPU `-dev none` oracle via `--kl-divergence-base`, `--tokens-file` WITHOUT `--bos` since LFM3 does not add BOS): Q8_0 ctx 2048 x 6 windows mean KL 1.68e-3 / top-1 98.26 % (patch off: 1.68e-3 / 98.27 %), ctx 8192 x 2 windows 1.29e-3 / 98.25 % (patch off 1.29e-3 / 98.28 %); Q4_K_M ctx 2048 x 6 windows 5.7e-3 / top-1 96.6 %.
- The unpatched path lost the Vulkan device at -p 28672 (one-layer attention submit ~2.5 s); fixed by #1666.
- **Update 2026-11-07, global-attention layers (PR #1687, issue #1572 remainder; dotLLM-only, INDICATIVE: box under heavy build load from ~10 sibling agents, no llama.cpp arm, single runs).** Global layers prefill on the GQA-grouped coopmat flash kernel (`DOTLLM_VK_LFM3_GLOBAL_GQA=0` restores the per-head kernel), and long prefills are sliced progressively (each slice sized at its own KV depth, 2x growth cap, attention cost scaled by depth beyond the fitted depth). GPU-timestamp stage profile at pp16384 (`DOTLLM_VULKAN_MOE_STAGE_PROFILE=1 DOTLLM_VK_STAGE_TS=1`): global attention 4966 ms of 8317 ms (60 %) with the per-head kernel vs 1044 ms of 4323 ms (24 %) with the GQA kernel; FFN (2221 ms, 51 %) is now the largest stage. Interleaved A/B pp16384 `-n 16` (old, new, old, new; second pair quoted): old 1657, new 2435 pp tok/s. Ladder (new, `-n 8|1`): pp28672 2554-3157, pp65536 1542-1804 (the #1672 ladder had 1261-1268 and 522); longest one-layer submit at 64K ~0.8 s (per-slice GPU totals), no device loss. Re-measure against llama.cpp in a quiet window (replay commands in PR #1687); tg rows not re-measured.


## 3. RTX 3060 12 GB, CUDA

### 3.1 Recursive Flint Q4_1 on an RTX 3060 12 GB (CUDA, MoE offload)

Source: [MOE_OFFLOAD.md](MOE_OFFLOAD.md). eGPU (2.25 GB/s link), Core Ultra 7 155H, 96 GB RAM. `dotllm bench -p 512 -n 64
-r 2 --warmup 1` vs `llama-bench -p 512 -n 64 -r 2` (llama.cpp master CUDA, default 16 threads), three interleaved rounds,
median (min-max).

| engine, placement | pp512 tok/s | tg64 tok/s | peak VRAM | host |
|---|---|---|---|---|
| llama.cpp `-ncmoe 34` | 79.5 (78.9-79.8) | 21.3 (19.2-22.0) | 8.57 GiB | 0.8 GiB |
| dotLLM `--n-cpu-moe 34` | **83.7** (83.5-84.2) | 19.1 (18.0-22.2) | 9.23 GiB | 0.9-1.2 GiB |
| dotLLM `--n-cpu-moe 36` | **80.0** (79.9-80.1) | 19.7 (19.7-20.9) | 8.23 GiB | 0.9-1.2 GiB |
| llama.cpp `-ncmoe 31` | 87.5 (87.3-87.7) | 19.5 (18.8-22.5) | 9.97 GiB | 0.8 GiB |
| dotLLM auto (= `--n-cpu-moe 31`) | **88.3** (88.0-88.4) | **21.0** (19.7-22.0) | 10.73 GiB | 1.0-1.2 GiB |

Prefill is ahead at every placement (1.01-1.05 at equal `--n-cpu-moe`); decode is at parity within this machine's noise
(single runs vary 15-23 tok/s for both engines). dotLLM keeps ~0.65 GiB more VRAM than llama.cpp at the same placement
(runtime scratch, #1211). The same row before #1187 / #1191 was 24.8 pp512.

### 3.2 Ternary Bonsai-2 27B PQ2_0, CUDA, RTX 3060

Source: [GPU.md](GPU.md), "Ternary Bonsai-2 27B PQ2_0 on the RTX 3060 (CUDA, #1222)". Abliterated-v2 PQ2_0-MTP file vs the
**PrismML llama.cpp fork** `b87ca7225` (`llama-bench -ngl 99 -fa 1`, pp512 / tg128, 3 reps, two rounds each).

| | dotLLM (#1222) | PrismML fork | ratio | dotLLM before (`c8d4569a`) |
|---|---:|---:|---:|---:|
| prefill pp512 (tok/s) | 597-601 | 587-591 | 1.01-1.02 | 157 |
| plain decode tg128 (tok/s) | 37.1-37.2 | 35.0-35.3 | 1.05-1.06 | 20.5 |
| MTP K=3, gate forced (tok/s) | 54-56 | 48.8 (`llama-server --spec-type draft-mtp`) | 1.11-1.15 | 35.0 |
| VRAM, plain bench (MiB) | 7,370 | 7,388 | 1.00 | 8,475 |

Perplexity (wikitext-2, 16 x 512, no BOS): 10.6182 (W2A8 MMA) vs 10.6166 (fork).

## 4. CPU

### 4.1 Nemotron 3.5 Bolt 74B-A9B, CPU

Source: [SUPPORTED_MODELS.md](SUPPORTED_MODELS.md), "Throughput" (2026-10-20). Intel Core Ultra 7 155H (AVX2 + AVX-VNNI, no
AVX-512), 14 threads, 95 GB RAM with ~68-72 GB available. `llama-bench -dev none -t 14 -p 512 -n 128 -r 1` (llama.cpp
`b11739`) vs `dotllm bench -d cpu --threads 14 -p 512 -n 128 -r 1 --warmup 0`. **One run each**; files larger than free RAM
page from NVMe, so the Q6_K and Q8_0 rows are partly I/O-bound for both engines.

| quant | llama.cpp pp512 / tg128 | dotLLM pp512 / tg128 | ratio pp / tg | dotLLM + #1042 scan threading | ratio pp / tg |
|---|---|---|---|---|---|
| AD-Q4_K_M | 11.27 / 2.63 | 2.51 / 1.87 | 0.22 / 0.71 | **4.61 / 1.84** | 0.41 / 0.70 |
| AD-Q6_K | 7.83 / 1.29 | 1.94 / 1.31 | 0.25 / 1.02 | not measured | |
| AD-Q8_0 | 6.58 / 2.02 | 1.58 / 0.75 | 0.24 / 0.37 | not measured | |

Prefill is still 2.4x behind after #1042: the selective scan is scalar (no SIMD over `d_state`), and the grouped expert
GEMMs run with only ~13 tokens per expert at pp512. The NVIDIA `24GB` IQ2_XXS + NVFP4 file was measured under 76-93 %
unrelated CPU load and is not comparable (see source).

### 4.2 Small dense models, CPU (historical)

These predate later CPU tuning (weight repacking, GEMM and decode work, the .NET 11 runtime result) and are kept only
as a time series. They must not be quoted as current.

| Date, hardware, llama.cpp | Model | prefill ratio | decode ratio | Source |
|---|---|---|---|---|
| 2026-03-06, Ryzen 7 5800HS (8C/16T), b5291 | SmolLM-135M Q8_0 | 0.32 | 0.74 | [BENCHMARKS.md](BENCHMARKS.md) |
| same | Llama-3.2-1B Q8_0 | 0.49 | 0.95 | [BENCHMARKS.md](BENCHMARKS.md) |
| same | Llama-3.2-3B Q8_0 | 0.57 | 1.01 | [BENCHMARKS.md](BENCHMARKS.md) |
| 2026-05, Core Ultra 7 155H (22 thr vs llama.cpp 16 thr), b9041 | Llama-3.2-1B Q8_0 | 0.62 | 0.69 (llama.cpp CV 75 %) | [PERFORMANCE.md](PERFORMANCE.md) section 3 |
| same | Qwen3.6-35B-A3B Q6_K_XL | 0.28 | 0.25 (memory-contended) | [PERFORMANCE.md](PERFORMANCE.md) section 3.1 |
| same, RTX 3060 CUDA | SmolLM-135M / Llama-3.2-1B Q8_0 decode | 1.02 / 0.85 (prefill) | 0.43 / 0.60 | [PERFORMANCE.md](PERFORMANCE.md) section 3 |

### 4.3 Liquid AI LFM3-1.7B, CPU

Source: PR #1569 / issue #1570 (measured 2026-11-03; dotLLM `issue/1559-lfm3-cpu` `28b7f0f8`). Ryzen AI Max+ 395 (Zen 5, 16C/32T),
`LFM3-1.7B-GGUF` Q8_0 and Q4_K_M. llama.cpp master `dde879bc` `llama-bench -dev none -t 32`; dotLLM `dotllm bench -d cpu --threads 32`.
`cpu-bench` held; a GPU job of another session ran alongside (about 10 % total CPU load before the runs), so single-run rows are
indicative and the disturbed reps are shown, not dropped.

| quant | test | dotLLM | llama.cpp | dotLLM / llama.cpp |
|---|---|---|---|---|
| Q8_0 | pp512 | 165.1 (164.4-168.8, 3 reps) | 404.8 (+/- 3.3) | 0.41 |
| Q8_0 | tg128 | 39.6 (39.5-39.8) | 45.5 (+/- 0.14) | 0.87 |
| Q8_0, 16 thr | pp512 / tg128 | 127.2 / 35.1 | 285.5 / 48.5 | 0.45 / 0.72 |
| Q8_0 | pp16384 / pp32768 (1 rep) | 103.9 / 66.7 | 298.6 / 260.4 | 0.35 / 0.26 |
| Q8_0 | tg64 @ 4K / 16K / 32K | 35.7 / 28.1 / 16.7 | 39.9 / 32.2 / 22.7 | 0.89 / 0.87 / 0.74 |
| Q4_K_M | pp512 | 293-302 | 1068 (+/- 14) | 0.28 |
| Q4_K_M | tg128 | 73-76 in 4 of 6 reps (42 and 53 in two disturbed reps) | 72.8 (+/- 1.0) | ~1.0 |

- At T=512 the forward spends 97 % of its time in the GEMMs (instrumented run), so the short-context prefill gap is the shared Q8_0 / Q4_K
  CPU GEMM kernels (#1119), not LFM3 code. At 32K the forward is 491 s against ~200 s of GEMM time: the rest is attention on the 6 global
  layers (#1570).
- The architecture's context advantage shows on CPU too, but less than on Vulkan: both engines lose about half their decode rate by 32K,
  because the 6 global layers still read 32K keys per token.
- Decode at depth uses `-p N` for dotLLM and `-d N` for llama.cpp (see 2.10).
- **Update 2026-11-07 (PR #1687, issue #1570; INDICATIVE, box at ~84 % CPU from sibling builds).** Causal prefill attention over a long KV is blocked as KV head x 32-token row block x key tile (`Attention.ExecuteBlockedCausal`, bit-identical to the per-row kernel; `DOTLLM_CPU_ATTN_BLOCKED=0` restores it). In-process kernel timing, LFM3 shape, a 1024-token chunk, 32 threads: KV end 16384 per-row 2148-5185 ms vs blocked 509-774 ms; KV end 32768 per-row 5421-7953 ms vs blocked 1147-1462 ms (4-7x). Whole-forward pp8192 under that load: 129 vs 124 tok/s (noise; attention is a small share at 8K). The pp16384 / pp32768 and tg rows above are NOT re-measured; replay commands in PR #1687.

## 5. Quality parity

Throughput is meaningless if the output differs, so the perplexity comparisons are summarised here; the protocol
(shared token ids, `--bos`, the CRLF and BOS traps) is in [PERPLEXITY.md](PERPLEXITY.md) and must be followed to reproduce them.

| Model | corpus | dotLLM | llama.cpp CPU | gap | Source |
|---|---|---|---|---|---|
| Llama-3.2-1B Q4_K_M | `wiki.test.lf.raw`, ctx 512, 64 windows | 15.7606 | 15.7555 (`b11739`) | +0.032 % (was +0.077 % before the #1507 Q8_K activation quantizer) | PERPLEXITY.md, #1340 / #1519 |
| Llama-3.2-1B pure Q3_K | same | 23.3185 | 23.2969 | +0.093 % (was +0.131 %) | same |
| Llama-3.2-1B Q8_0 | wikitext-2 LF, ctx 4096, 3 chunks | 11.1899 | 11.1840 | +0.05 % | SUPPORTED_MODELS.md |
| Bonsai-2 27B PQ2_0 (W2A8 MMA) | wikitext-2, 16 x 512, no BOS | 10.6182 | 10.6166 (PrismML fork) | +0.015 % | GPU.md, #1222 |

The #1507 change made the CPU Q8_K activation quantizer follow llama.cpp's `quantize_row_q8_K_ref` (`iscale = 127/amax`).
Both builds move slightly toward llama.cpp but the deltas are inside noise (paired dNLL -0.0004, se 0.0006-0.0008), and
greedy text is **not** preserved across the change (6 of 8 prompts diverged): compare perplexity, not text, across it.
Embedding-model cosine parity against llama.cpp `b9016` and sentence-transformers is in
[EMBEDDINGS.md](EMBEDDINGS.md) section 2 (F16 min cosine >= 0.9999, Q8_0 ~0.999).

## 6. Embeddings

Sources: [EMBEDDINGS.md](EMBEDDINGS.md) sections 3, 4 and 5. Strix Halo, 48 mixed-length `docs/*.md` chunks (mean 148
tokens), tokenise + embed + pool + normalise. llama.cpp `b9016` `llama-server --embeddings -dev none -t 32` (CPU; no llama.cpp
GPU embedding run exists). Throughput in docs/s.

| Model (F16 unless noted) | llama.cpp CPU | dotLLM CPU at the #1369 baseline | dotLLM CPU after #1402 (attention) + #1403 (GELU) | ratio vs llama.cpp (derived) |
|---|---|---|---|---|
| bge-small | 84 | 3.5-3.8 | 39.7-40.7 | 0.47-0.48 |
| bge-small Q8_0 | 77 | 4.9-6.7 | 30.9-33.1 | 0.40-0.43 |
| bge-base | 44 | 2.3-2.6 | 20.6-21.0 | 0.47-0.48 |
| bge-base Q8_0 | 34 | 4.3-4.6 | 14.2-14.7 | 0.42-0.43 |
| MiniLM-L6 Q8_0 | 140 | 13.7-14.4 | 64.0-65.3 | 0.46 |

The ratio column is **derived by this page** from two documents measured in different sessions on the same text set (the
llama.cpp figure from the #1369 survey, the dotLLM figure from the later A/B tables); it is indicative only. The same
EMBEDDINGS.md section 5 records dotLLM CPU on the #741 tree at 49-51 docs/s for bge-small F16 B=48 (ratio ~0.6 against 84).
The #1369 baseline "5-25x slower than llama.cpp" is **historical** and no longer holds.

dotLLM Vulkan (iGPU 8060S) on the same text set, F16 activations (#1474), packed varlen batches (#1437), against
dotLLM's own CPU (there is no llama.cpp Vulkan embedding comparator):

| Model | B | Vulkan docs/s | dotLLM CPU docs/s (same tree) | Vulkan / CPU |
|---|---|---|---|---|
| bge-small | 1 | 250-260 | 41-43 | ~6x |
| bge-small | 48 | 809-825 | 49-51 | ~16x |
| bge-base | 48 | 360-369 | 21.8-23.1 | ~16x |
| EmbeddingGemma-300M (Q8_0, packed) | batch | 211-213 | 7.1-7.3 (16 thr, unbatched) | ~29x |
| nomic-embed-text-v1.5 F16 | 1 / 48 | 95-97 / 185-192 | 13-14 / 4 | ~7x / ~47x |

## Caveats

- **llama.cpp Vulkan prefill is noisy** (about 2x between launches on this box, and the first round of each engine runs
  slower). Quote ranges and interleave; never compare a dotLLM run against a llama.cpp figure from another session.
- **UMA decode contention.** On Strix Halo the CPU and GPU share DRAM bandwidth, and a background process can move decode
  by ~40 %. Absolute tok/s drift between sessions (the #1510 and #1517 tables differ by 14 % for both engines); ratios from
  interleaved pairs are the only portable quantity.
- **Clock ramp.** A cold launch vs a warm launch shows 2-3x phantom deltas. All tables above are warm and paired.
- **Greedy text identity is not a quality metric.** Activation-quantizing kernels digitise ULP-level reduction differences
  into different text. Use perplexity or KL ([PERPLEXITY.md](PERPLEXITY.md)).
- **llama.cpp's Vulkan backend is not an oracle.** It is within +0.4 % of its own CPU perplexity; numerics are referenced to
  llama.cpp on `-dev none`.
- **Degenerate prompts inflate speculative ratios.** Two of the four 27B prompts degenerate into repetition (acceptance
  ~0.95+ for both engines); the per-prompt column in 2.1 shows where the lead comes from.
- **Different comparators.** Bonsai-2 PQ2_0 is compared with the PrismML fork (mainline llama.cpp cannot load it). Bolt CPU is
  one run per cell, partly I/O-bound.
- **dotLLM builds.** Each source states its dotLLM commit; this page does not mix tables from different commits in one row.

## Needs re-measure

Gaps and stale rows, each of which a reader could mistake for a current figure:

1. **Flint on current dev vs llama.cpp** `dde879bc`: the 2026-11 smoke (pp512 1,967-1,988, tg64 90.0-90.4, 128K tg64 31.8)
   has no llama.cpp pair (likely unsupported or not tried); the ratios in 2.2 are from `0ae0e1976`.
2. **llama.cpp Vulkan on Strix for small dense models** (SmolLM-135M, Llama-3.2-1B Q8_0 at 81.8 tok/s, Bielik, Qwen2.5-0.5B):
   [PERFORMANCE.md](PERFORMANCE.md) section 6.2 says "not measured". Open since the May baseline.
3. **Embedding encoders vs llama.cpp on the current tree**: CPU (the ratio in section 6 is derived across sessions) and
   Vulkan (no llama.cpp `-ngl` embedding run has ever been made).
4. **Bolt (CPU and Vulkan LiLiCorr) vs llama.cpp** after #1042 / #1365 / #1420 / #1459. The CPU rows are single runs from
   2026-10-20; the Vulkan speculative row predates the Nemotron-H small-row GEMV tables.
5. **Qwen3.6-35B-A3B Vulkan MTP vs llama.cpp** after the MoE verify work (#1362, #1374).
6. **CPU prefill of Llama-family Q8_0 on Strix / Zen 5** (current engine vs llama.cpp). Rows in 4.2 are 2026-03 and
   2026-05 on other hardware and pre-date later CPU tuning and the .NET 11 result.
7. **CUDA Llama decode on the RTX 3060** (0.43x / 0.60x in the May baseline) after #124 / #125. Needs the user's go-ahead
   to use the T5500 and the GPU lock.
8. **Qwen3.8-27B in a quiet-box session**: the #1517 / #1520 table was taken in a slower state (-14 % for both engines);
   ratios are valid, absolutes are not records.
9. **Pruned Flash-Next Coder** (2.7): GPU heap, an F32-decoded control for the CPU divergence, MTP on more than two prompts,
   and ~8K prefill (llama.cpp 340 +/- 101 is too noisy to rank).
10. **Flash-Next UD-IQ4_XS** (2.8): interleaved pp512 and decode in a quiet-box session (the box drifted 278-545 tok/s).
11. **Tiled prefill hybrids** (2.9): Qwen3.8-27B and Qwen3.6 (hd256 path) after #1553, and llama.cpp for the models other
    than Qwen3-4B in the new/old table.
