# Embeddings for vector-RAG: survey, baseline and lever ranking (#1369)

Phase 1 of #1369: **measure only, no new kernels.** The harness is `benchmarks/EmbeddingBench`; the one source
change under `src/` is a null-checked per-stage timer in `BertEncoderModel` (`EncoderStageProfile`, off by default).

Host: Strix Halo (Zen 5, 16C/32T, AVX-512 on 256-bit datapaths), Windows, .NET 10, `cpu-bench` lock held, box idle
(9-13 % total CPU before each block). CPU only. All GGUFs come from the HF hub cache (no copies in the tree).
References: sentence-transformers (PyTorch CPU, **fp32**, venv under the session scratchpad), and llama.cpp
`llama-server --embeddings -dev none -t 32` (b9016, zen4 CPU backend; Vulkan is never used for numerics).
Quote ranges, not points: the box is shared, so treat differences under ~15 % as noise.

## 1. What loads today

| Model (GGUF) | Arch | Params | Dim | Pooling | Max len | Loads | Quants tried |
|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | bert | 22M | 384 | mean | 512 (ST truncates at 256) | yes | F16, Q8_0 |
| bge-small-en-v1.5 | bert | 33M | 384 | cls | 512 | yes | F16, Q8_0 |
| e5-small-v2 / gte-small | bert | 33M | 384 | mean | 512 | yes | F16 |
| snowflake-arctic-embed-s | bert | 33M | 384 | cls | 512 | yes (smoke only) | F16, Q8_0 |
| bge-base-en-v1.5 / e5-base-v2 / gte-base | bert | 110M | 768 | cls / mean | 512 | yes | F16 (+Q8_0 bge) |
| nomic-embed-text-v1.5 | nomic-bert (RoPE, SwiGLU) | 137M | 768 | mean | 2048 | yes | F16, Q8_0, Q4_K_M |
| mxbai-embed-large-v1 | bert | 335M | 1024 | cls | 512 | yes | F16, Q8_0 |
| Qwen3-Embedding-0.6B | qwen3 (decoder) | 600M | 1024 | last | 32768 | yes (EOS appended automatically since #1398) | F16 |
| multilingual-e5-small (XLM-R, SentencePiece Unigram) | bert | 118M | 384 | mean | 512 (some GGUFs say 511) | yes (#1401) | F16, Q8_0 |
| EmbeddingGemma-300M | gemma-embedding (bidirectional Gemma3 + dense heads) | 300M | 768 | mean | 2048 | yes (#1399), CPU only | Q8_0 (ggml-org), F32 / F16 / BF16 (heads added) |
| nomic-embed-text-v2-moe | nomic-bert-moe (XLM-R Unigram vocab) | 475M (8x277M, top-2) | 768 | mean | 512 | yes (#1400), CPU only | Q8_0 (official GGUF) |

Not attempted (same code path as a row above, or outside the 20-600M download rule): bge-large / e5-large / gte-large
(bert), Qwen3-Embedding-4B/8B (qwen3 decoder), gte-Qwen2, bge-m3 and jina-v3/v4 (XLM-R / custom code; bge-m3 uses the same Unigram tokenizer that multilingual-e5 now loads through, #1401). Encoder = bidirectional BERT-style (`BertEncoderModel`, CPU only);
decoder-based embedders (Qwen3-Embedding, last-token pooling) run on `TransformerModel`, also CPU only.

GPU: Vulkan `VulkanTransformerModel` implements `IEmbeddingModel` since #1406 (section 5); CUDA models do not (the server returns 501),
and encoders are refused on a GPU device (#741 is the work item).

## 2. Quality gate

Method (`benchmarks/EmbeddingBench dump` + scratchpad `cmp.py`): 48 fixed texts (24 short, 16 medium, 4 over-512-token
chunks of `docs/*.md`, plus a query, a unicode string and a one-character string), dotLLM vs reference, same tokens
(the over-length texts keep the trailing `[SEP]` when truncated to 512), L2-normalised, per-text cosine, min and mean,
plus top-1 nearest-neighbour agreement over the 48x48 similarity matrix. Gate: **cosine >= 0.9999 for F32/F16 weights**;
for quantised weights record the figure (the gate does not apply, but recall must not move).
Prefixes are applied to both sides (`query: ` for e5, `search_document: ` for nomic).

| Model / quant | vs sentence-transformers fp32 (min / mean) | vs llama.cpp CPU (min / mean) | NN-1 agree |
|---|---|---|---|
| MiniLM-L6 F16 | 0.999997 / 0.999999 | 1.000000 / 1.000000 | 1.00 |
| bge-small F16 | 0.999997 / 0.999999 | 1.000000 / 1.000000 | 1.00 |
| e5-small F16, gte-small F16 | 0.999999 / 0.999999 | - | 1.00 |
| bge-base F16 | 0.999990 / 0.999994 | 0.999999 / 1.000000 | 1.00 |
| e5-base F16 / gte-base F16 | 0.999994 / 0.999996 ; 0.999983 / 0.999996 | 1.000000 ; 0.999999 | 1.00 |
| mxbai-large F16 | 0.999983 / 0.999991 | 0.999999 / 1.000000 | 1.00 |
| nomic-v1.5 F16 | n/a (ST `trust_remote_code` is broken on current transformers) | 0.999999 / 1.000000 | - |
| Qwen3-Embedding-0.6B F16, **EOS appended** | 1.000000 / 1.000000 | 0.999999 / 1.000000 | 1.00 |
| Qwen3-Embedding-0.6B F16, before #1398 (no EOS) | 0.466 / 0.804 | 0.466 / 0.804 | 0.44 |
| Qwen3-Embedding-0.6B F16, `embed` / `/v1/embeddings` after #1398 | 1.000000 (4-text fixture, ST fp32, >= 0.9999 gated) | - | - |
| MiniLM Q8_0 / bge-small Q8_0 / bge-base Q8_0 / mxbai Q8_0 | 0.99928 / 0.99962 ; 0.99961 / 0.99984 ; 0.99966 / 0.99980 ; 0.99864 / 0.99977 | bge-base 0.99990, mxbai 0.99955 | 0.95-1.00 |
| nomic Q8_0 / Q4_K_M | - | 0.99959 / 0.99988 ; 0.99317 / 0.99736 | 0.94 / 0.83 (vs F16 oracle) |

Reading: every F16 BERT-class model passes the 0.9999 gate against **both** references, including nomic-bert (RoPE+SwiGLU).
Q8_0 costs 1e-4 to 1.4e-3 of cosine (worst case is the one-character text); Q4_K_M nomic is visibly degraded (0.93 mean vs
its own F16) and is not recommended for RAG. Qwen3-Embedding was wrong by default until #1398: the declared-last-pooling decoder embedder now gets the tokenizer EOS appended (llama.cpp b9016 does the same regardless of the `--pooling` override: prompt_tokens 3 for a 2-token text with last or mean; it does not for SmolLM2, which declares no pooling). Override: `--add-eos true|false` on `dotllm embed`, `add_eos` in `/v1/embeddings`. Guard: `Qwen3EmbeddingEosParityTests` against `Fixtures/Embeddings/st-qwen3-embedding-0.6b.json`.
**multilingual-e5-small (#1401, XLM-R tokenizer `tokenizer.ggml.model = "t5"`)**, 107 texts (48 English `docs/*.md` chunks + 59 multilingual, all `query: ` prefixed, up to the 512-token limit):
sentence-transformers fp32 on a GGUF whose position table is repaired to HF's (see below): F16 min 0.999997 / mean 0.999999, NN-1 1.00; llama.cpp b9016 CPU on the same repaired file 1.000000 / 1.000000.
On the unrepaired community F16 file (`hdrrayan/multilingual-e5-small-gguf`) dotLLM equals llama.cpp (1.000000 / 1.000000, n=104) and both sit at 0.9881 min / 0.9989 mean from ST. Its Q8_0 (`TwinSunsLLC`) vs llama.cpp on the same file: 0.999924 / 0.999964.
**Position-table trap**: `intfloat/multilingual-e5-small` is an HF `BertModel` (absolute positions 0..511, 512 rows), but llama.cpp's `convert_hf_to_gguf.py` treats it as XLM-R and drops `pad_token_id + 1 = 1` rows (the config's `pad_token_id` is 0), so the community
GGUFs (`hdrrayan`, `TwinSunsLLC`, `keisuke-miyako`) carry 511 rows shifted by one: every token sees position k+1 instead of k. llama.cpp is wrong in exactly the same way, so only a sentence-transformers comparison exposes it. Fix = re-convert, or rewrite
the tensor from `model.safetensors` (rows 0..511, `bert.context_length` 512); the 0.9999 gate was measured on such a rewrite. The `keisuke-miyako` Q8_0 is a worse conversion (`tokenizer.ggml.model = "bert"`, no scores, every piece `▁`-prefixed) and still fails with the original `[CLS]/[SEP]/[UNK]` error.
A retrieval-recall set (BEIR slice) is still missing; the cosine+NN-1 gate is what guards speed work for now.

**nomic-embed-text-v2-moe (#1400, `nomic-bert-moe`)**, 103 texts (48 English chunks truncated to 512 tokens keeping the last token, plus 55 multilingual, `search_query: ` / `search_document: ` alternating), CPU. Tokens come from the Unigram tokenizer of #1401. Layout: the nomic-bert encoder (post-LN, fused QKV, RoPE NeoX, token-type row 0) with a biased GELU FFN on even layers and, on odd layers (`moe_every_n_layers = 2`), a top-2-of-8 MoE: softmax over all 8 experts, the two selected probabilities used **as is** (no renormalisation), `up_e` then GELU then `down_e`, no gate projection. Tokens are grouped per expert so each expert runs one GEMM; the 3-D Q8_0 banks are read in place.
The reference is a PyTorch re-implementation written from the published `modeling_hf_nomic_bert.py` semantics (the remote-code model does not run under current transformers), fp32, on the original safetensors.

| GGUF / arm | vs reference with `experts.bias` (min / mean) | vs llama.cpp b9016 CPU, same file (min / mean) |
|---|---|---|
| F32 built from safetensors, **exact-erf** GELU, with bias | 1.0000000 / 1.0000000, NN-1 1.00 | - |
| same, default tanh GELU (llama.cpp's `ggml_gelu`), with bias | 0.999806 / 0.999987, NN-1 1.00 | - |
| F32 without the bias tensor (what the official GGUF contains) | 0.998753 / about 0.9996 | 0.999993 / 1.000000 |
| official Q8_0 | 0.997079 / 0.998978 (llama.cpp itself 0.997333 / 0.998998) | 0.998370 / 0.999802 |

Findings: (1) **The HF model adds a learned `mlp.experts.bias` [768] after the weighted expert sum on every MoE layer (max abs 0.10-0.43); llama.cpp's converter drops it, so the official GGUF cannot reproduce the reference** (cosine 0.9988 min / 0.9996 mean from it before any quantisation, and llama.cpp has the same defect). dotLLM follows llama.cpp when the tensor is absent and adds it when a GGUF carries `blk.N.ffn_down_exps.bias` [768] (the name llama.cpp already uses for expert bias tensors); that is what the 0.9999 row used. Upstream: llama.cpp should keep `experts.bias`. (2) With top-2 routing a 1e-4 GELU approximation difference can flip an expert choice on a borderline token, so the tanh arm shows min 0.9998 on one text where the dense models stay above 0.99999; exact erf removes it. dotLLM keeps the tanh GELU on purpose (same as every other BERT-class model and as llama.cpp). (3) CPU only (`--device cpu`, like the other encoders).

**EmbeddingGemma-300M (#1399, `gemma-embedding`)**, 103 texts (48 English `docs/*.md` chunks, up to 752 tokens, plus 55 multilingual; alternating model-card prompts `task: search result | query: ` / `title: none | text: `), CPU, reference = sentence-transformers fp32 (Transformer, mean Pooling, Dense 768-3072, Dense 3072-768, Normalize; `unsloth/embeddinggemma-300m` mirrors the gated `google/embeddinggemma-300m`). Token ids identical to the reference on 103/103 texts (`<bos> ... <eos>`; the SPM `add_space_prefix=false` GGUF keys are honoured).

| GGUF | vs ST fp32 (min / mean) | vs llama.cpp b9016 CPU, same file (min / mean) |
|---|---|---|
| F32 with dense heads (`unsloth` F32 + heads from the ST modules) | 1.0000000 / 1.00000000, NN-1 1.00 | 0.999999 / 1.000000 |
| F32 without heads (`unsloth`) vs ST modules 0-1 only | 1.0000000 / 1.00000000, NN-1 1.00 | - |
| Q8_0 with heads (`ggml-org`) | 0.998849 / 0.999713 (llama.cpp itself: 0.998812 / 0.999609) | 0.999091 / 0.999712 |

How it works: the Gemma 3 tower (QK-norm, four RMSNorms, GeGLU, dual RoPE, 5:1 local/global) with **bidirectional** attention. Sliding-window layers (window 512) use a **symmetric** window, keys `q-256 .. q+256` (llama.cpp `LLAMA_SWA_TYPE_SYMMETRIC`; HF folds this into `sliding_window // 2 + 1`), new `AttentionMaskMode.SymmetricWindow`. The window only matters past 257 tokens: with it ignored the nine >257-token texts drop to 0.9886 (mutant run), with it they are 1.0000000. The heads are applied after pooling in llama.cpp and sentence-transformers; both are linear and mean/CLS/last pooling are linear row combinations, so dotLLM applies them per token row inside `ForwardHidden` (`dense_3(dense_2(h))`, Q8 weights read in place, 128-row chunks) and every existing consumer (CLI, server, batcher, bench) keeps pooling + normalising unchanged. Cost: two small GEMMs, 2 x 768 x 3072 MACs per token, a few percent of the 24-layer tower.
Gotchas: (1) many community GGUFs (`unsloth`, `second-state`) were converted **without** `--sentence-transformers-dense-modules` and carry no `dense_2/dense_3`; llama.cpp and dotLLM then return the raw pooled vector, which is not the model's output space (cosine about 0 against the published embeddings). Use `ggml-org/embeddinggemma-300M-GGUF`, or add the heads. (2) the tokenizer frames `<bos> text <eos>` (the factory sets `AddEosToken` for this architecture only). (3) CPU only: `--device vulkan|gpu` fails loudly, `auto` falls back to the CPU; a causal `Forward` on the model throws instead of silently returning wrong hidden states. (4) prompts are the caller's job (`task: search result | query: ` for queries, `title: none | text: ` for documents; others in the model card's `config_sentence_transformers.json`).

## 3. Throughput and latency on CPU (32 threads)

The engine runs **one sequence per `ForwardHidden`** (no cross-sequence batching; `/v1/embeddings` loops sequences
under a global model gate). So "batch B" here = B documents back to back; batch 1/8/32/128 are **flat within +-10 %**
(the B=1/8 cells at seq 64 carry cold-start noise). Docs are real `docs/*.md` text cut to exactly the sequence length,
CLS/SEP framed. Steady-state at B=32 (min-max over 3 reps in `benchmarks/EmbeddingBench/results/`; docs/s | tokens/s):

| Model | seq 64 | seq 256 | seq 512 |
|---|---|---|---|
| MiniLM-L6 Q8_0 | 59-62 \| 3.9k | 8.6-8.7 \| 2.2k | 2.8 \| 1.5k |
| bge-small F16 | 10.9-11.2 \| 0.72k | 1.7-2.1 \| 0.53k | 0.86 \| 0.44k |
| bge-base Q8_0 | 15.8-16.3 \| 1.0k | 2.9 \| 0.76k | 0.96 \| 0.49k |
| nomic-v1.5 Q8_0 | 21-22 \| 1.4k | 3.4 \| 0.88k | 1.0 \| 0.52k |
| mxbai-large Q8_0 | 4.2-4.8 \| 0.29k | 0.95 \| 0.24k | 0.31 \| 0.16k |
| Qwen3-Embedding-0.6B F16 | 2.8 \| 0.18k | 0.66 \| 0.17k | 0.32 \| 0.16k |

Single-document latency, p50 / p99 ms (60 docs each; one-off outliers excluded from p50 only):

| Model | 16 tok | 64 tok | 256 tok |
|---|---|---|---|
| MiniLM-L6 Q8_0 | 7.3 / 21.6 | 16.7 / 18.8 | 120 / 135 |
| bge-small F16 | 28.9 / **876** | 95.7 / 108 | 428 / 511 |
| bge-base Q8_0 | 18.8 / 36.5 | 65.3 / 68.6 | 354 / 734 |
| nomic-v1.5 Q8_0 | 15.1 / 34.3 | 48.0 / 83.4 | 293 / 953 |
| mxbai-large Q8_0 | 51.8 / 307 | 192 / 237 | 1122 / 1403 |

Same-text end-to-end docs/s (48 mixed-length chunks of `docs/*.md`, mean 148 tokens; tokenise + embed + pool + normalise;
dotLLM 3 passes, min-max; llama-server batches the 46 texts <= 512 tokens it accepts in one request and 400s the longer ones;
sentence-transformers is fp32 PyTorch):

| Model | dotLLM | llama.cpp CPU | sentence-transformers bs=1 / bs=8 / bs=32 |
|---|---|---|---|
| MiniLM-L6 F16 / Q8_0 | 6.6-7.6 / 13.7-14.4 | 143 / 140 | - / - / 67 |
| bge-small F16 / Q8_0 | 3.5-3.8 / 4.9-6.7 | 84 / 77 | 52-53 / 52-60 / 14-31 |
| e5-small F16, gte-small F16 | 3.7-3.8, 3.5-3.8 | 87, 83 | - / - / 33-34 |
| bge-base F16 / Q8_0 | 2.3-2.6 / 4.3-4.6 | 44 / 34 | 18.2-18.6 / 17.5-18.1 / 8.9-10.0 |
| e5-base F16, gte-base F16 | 2.4-2.5, 2.5-2.6 | 43, 44 | - / - / 10.7-10.8 |
| nomic-v1.5 F16 / Q8_0 / Q4_K_M | 2.3-2.7 / 4.9-5.1 / 5.5 | 36 / 24 / 32 | n/a |
| mxbai-large F16 / Q8_0 | 0.78-0.93 / 1.54-1.58 | 8.7 / 9.1 | 1.7-5.2 / 5.1-5.4 / 2.9-3.0 |
| Qwen3-Embedding-0.6B F16 | (0.3-2.8, see above) | 6.2 | 1.35 (bs=32) |

**dotLLM is 5-25x slower than llama.cpp (Q8_0 5-12x, F16 10-25x) and 3-15x slower than fp32 PyTorch (bs=1) on the same CPU.** Because
sentence-transformers at bs=1 is already 3-15x faster, the gap is **kernel efficiency, not missing batching** (PyTorch at
bs=32 is *slower* than bs=1 here: padding waste).

Other numbers: tokenisation is not a lever (WordPiece 17-18k docs/s = 2.2M tokens/s; Qwen BPE 5.8k docs/s = 0.67M tokens/s,
versus 60-250 ms to embed a document). Server-side `/v1/embeddings` before #1405 serialised all
requests on `ExecuteOnModelAsync` and looped sequences, so concurrent clients added latency, never throughput (see the #1405 section in 4 for the fix).

### Padding waste (what a padded batched path would face)

2 301 real RAG-style chunks (mean 102 tokens, p50 74, max 506). Today's engine pays **none** of this (no padding), the
figures size the design choice for a future batched path:

| Batch | Padding waste, arrival order | Length-sorted |
|---|---|---|
| 8 | 61 % | 0.7 % |
| 32 | 73 % | 2.9 % |
| 128 | 78 % | 9.5 % |

Length-sorting (or packing/varlen attention) is mandatory for any padded path; the PyTorch bs=32 regression above is this effect.

## 4. Where the time goes (stage profile, `EmbeddingBench profile`)

Per-document ms, min over two runs, 1 thread vs 32 threads (full table in `benchmarks/EmbeddingBench/results/profile.jsonl`; helper scripts in `benchmarks/EmbeddingBench/scripts/`, which hold scratchpad paths that need adapting; `FfnUp` includes the
gate GEMM for nomic):

| Model, seq | threads | Qkv | Attention | OutProj | FfnUp | **Act** | FfnDown | LN x2 | total |
|---|---|---|---|---|---|---|---|---|---|
| bge-base Q8, 64 | 1 | 51.7 | 9.7 | 17.2 | 68.5 | 22.0 | 65.6 | 1.8 | 237-321 |
| bge-base Q8, 64 | 32 | 9.1 | **8.8** | 3.3 | 9.1 | **21.3** | 12.0 | 2.2 | 68-116 |
| bge-base Q8, 256 | 32 | 28.7 | **137.9** | 13.7 | 29.3 | **86.2** | 29.6 | 6.8 | 339-344 |
| bge-base Q8, 512 | 1 | 401 | 545 | 136 | 536 | 171 | 516 | 15 | 2324-2414 |
| bge-base Q8, 512 | 32 | 53.1 | **572** | 21.4 | 56.0 | **167** | 53.7 | 14.2 | 945-974 |
| bge-small Q8, 512 | 32 | 19.2 | **514** | 9.5 | 20.2 | **104** | 23.3 | 7.1 | 701-744 |
| nomic Q8, 512 | 32 | 44.1 | **590** | 21.2 | 102.5 | 11.4 | 52.7 | 14.1 | 843-854 |
| mxbai Q8, 512 | 32 | 161 | **1851** | 61.5 | 198 | **460** | 189 | 39 | 2963-3046 |
| bge-small F16, 64 | 32 | **29.2** | 29.2 | 9.9 | 12.2 | 13.7 | 14.1 | 1.2 | 110-115 |

* **Attention does not scale with threads** (545 -> 572 ms going from 1 to 32 threads): it is single-threaded and runs at
  only ~9-18 GFLOP/s (versus ~54 GFLOP/s for one Q8 GEMM thread). It is 41 % of a 256-token document and 60-73 % of a 512-token one at 32 threads.
* **GELU is a serial scalar `MathF.Tanh` loop** (~11 ns/element, identical at 1 and 32 threads): 18-31 % of the document on
  every GELU model (bge/e5/gte/mxbai). nomic's SiLU is vectorised and is 1 %. LayerNorm/bias/QKV-split are serial but ~2-3 %.
* The GEMMs do parallelise but only 5.5-9.6x on 32 threads; 1-thread Q8 GEMM runs ~54 GFLOP/s.
* **F16 GEMM is 2.5-4.4x slower than Q8_0 and barely scales** (bge-small QKV: 48 ms at 1 thread, 29 ms at 32; Q8: 13 -> 11).
  `MatMul.GemmF16` re-converts each F16 weight row once **per token** on the non-SSE tier (loop order token-outer,
  row-inner, convert + dot per pair) - convert-once-per-tile would remove an n-fold redundant conversion.
  Most public embedding GGUFs are F16, so this is what users hit by default. (It is a bounded per-row scratch, not a stored widened copy, so it
  is a perf defect rather than a precision-rule violation.)
* **Fixed (#1404).** Two causes: (a) per-token row conversion, (b) `ComputeTileM` gave 256-row tiles so bge-small (m=384) had
  2 tiles for 32 threads. Now `MatMul.GemmF16RowsFma` (`MatMul.F16Tile.cs`) converts 4 weight rows once into a 4-row per-thread
  scratch and runs a 4-row x 2-token FMA register tile; the pooled path shrinks tiles (multiple of 4 rows, floor 8) so every worker
  gets one. Weights stay F16 in place. Per-output arithmetic is tiling-independent (ST and pooled are bit-identical, tested).
  `DOTLLM_F16_NO_FMA=1` selects the legacy loop for same-session A/B. bge-base F16, 32 threads, `profile`, two paired runs each
  (ms/doc, GEMM stages Qkv+Out+FfnUp+FfnDown): seq 64 **228 / 123 legacy -> 25-29 new** (total 144-268 -> 69-74, now below the Q8_0
  figures above); seq 256 **390 / 508 -> 66-67** (total 636-846 -> 297-341). Throughput B=8 seq 64: 6.1-7.1 -> 17.6 docs/s
  (one noisy rep 11). Attention (single-threaded) and GELU now dominate (#1402/#1403). Parity vs legacy on 50 texts: min cosine
  0.9999999; unit test vs double-precision reference on m%4, k%8, odd-n shapes.
* **Fixed (#1402): encoder attention is parallel.** `Attention.ExecuteBidirectional` (`Attention.cs`) splits `head x query-block`
  work items (about 8 per thread) over `ActivePool` (never `_pool`: batch workers run inside a pool dispatch). Every query row runs
  the unchanged per-row arithmetic of the old single-threaded path (`ExecuteHeadRows`), so the output is **bit-identical**
  (tested for nh/hd = 12/32, 12/64, 16/64, 3/32, 4/16 and seq 1..1500 incl. the tiled long-row path; and `EmbeddingBench dump`
  on 48 texts is byte-identical for MiniLM F16, bge-small F16/Q8, bge-base F16/Q8 and nomic F16, i.e. cosine 1.0 vs the old path,
  so the cosine >= 0.9999 gate vs sentence-transformers/llama.cpp above is untouched). `EmbeddingBench ... --legacy-attn 1` selects
  the old call for same-session A/B. Not done: a vectorised score/softmax micro-kernel (the stage is now ~45-52 ms at 512 tokens,
  about the size of one Q8 GEMM stage, so it is no longer the lever). Paired A/B, 32 threads, 3 reps alternating order, single
  document `profile` (ms/doc, min-max), legacy -> new:

| Model | seq | Attention stage | Document total |
|---|---|---|---|
| bge-base Q8_0 | 256 | 152-160 -> 14.7-21.2 | 372-373 -> 215-268 |
| bge-base Q8_0 | 512 | 644-655 -> 51.6-52.3 | 1029-1057 -> 443-450 |
| bge-base F16 | 256 / 512 | 152-154 -> 13.6-15.3 / 627-644 -> 45-53 | 320-331 -> 171-175 / 931-960 -> 359-366 |
| bge-small Q8_0 | 256 / 512 | 141-145 -> 13.8-16.2 / 576-581 -> 40.6-51.2 | 245-262 -> 107-111 / 764-771 -> 229-241 |
| bge-small F16 | 256 / 512 | 144-149 -> 14.9-17.5 / 576-594 -> 45-47 | 236-239 -> 98-100 / 740-762 -> 208-211 |
| MiniLM Q8_0 | 256 / 512 | 71-72 -> 12-16.7 / 288-297 -> 20-22 | 122-123 -> 56-61 / 374-382 -> 105-108 |
| nomic Q8_0 | 256 / 512 | 159-165 -> 13.6-16.1 / 645-654 -> 45-51 | 302-310 -> 134-149 / 893-923 -> 283-297 |

  Attention stage 10-14x at 256-512 tokens (target >= 10x met); whole document 1.6-3.6x at 512. The 64-token cells are
  dominated by cold-start noise (8 iterations) and show no reliable change. End-to-end 48-text set (mean 148 tokens, docs/s
  median over 3 reps): bge-base Q8 4.4-4.6 -> 7.6-7.8, bge-base F16 5.3-5.6 -> 9.6-9.9, bge-small Q8 6.4-6.9 -> 14.9-15.8,
  bge-small F16 7.0-7.3 -> 16.9-17.3, MiniLM Q8 13.7-14.0 -> 33.9-34.7, nomic Q8 4.7-4.8 -> 10.9-11.0 (raw rows:
  `benchmarks/EmbeddingBench/results/ab-attention-1402.jsonl`; this bench runs documents back to back, the server batcher of #1405
  already parallelises across documents). Remaining serial stage: GELU (#1403).
* **Fixed (#1403): GELU is vectorised and parallel.** `Gelu.ExecuteTanh` (`Gelu.cs`, ggml's tanh form, `UseErfGelu` seam
  unchanged) runs `TensorPrimitives` Multiply/Add/Tanh over 512-float L1 tiles, tile ranges split across `ActivePool`
  (shipping .NET 10 intrinsics only). Max abs error vs the exact double-precision tanh GELU: 6.3e-7 over 100k inputs in
  [-40, 40] plus 0, +-1e-8, +-9, +-100 (test gate `2e-6 * max(1, |x|)`); serial and parallel results are bit-identical. Model
  parity vs the old scalar loop (`EmbeddingBench dump --legacy-gelu 1` vs `0`, 48 texts): F16 models min cosine 1.000000000
  (max |dx| 1.2e-7: MiniLM, bge-small, bge-base; nomic has no GELU and is exactly 0); Q8_0 models min cosine 0.99973-0.99991
  vs the old path, because Q8 activation quantisation digitises the ~1e-7 difference. Against the sentence-transformers fp32
  reference the Q8_0 figures are unchanged within noise (bge-small 0.999612 / 0.999836 old vs 0.999612 / 0.999837 new,
  bge-base 0.999664 / 0.999804 vs 0.999591 / 0.999804, mxbai 0.998643 / 0.999769 vs 0.998745 / 0.999767, min / mean), i.e. the
  gate (F16 >= 0.9999; Q8 figure recorded, recall unmoved) holds. `--legacy-gelu 1` is the same-session A/B arm. Paired A/B,
  32 threads, 3 reps alternating order, `profile` Act stage and document total (ms/doc, min-max), legacy -> new, measured on
  the tree that already has the parallel attention of #1402 (so the totals are no longer diluted by serial attention):

| Model | seq | Act stage | Document total |
|---|---|---|---|
| bge-base Q8_0 | 256 / 512 | 95.6-97.2 -> 1.6-1.8 / 194-197 -> 2.1-2.2 | 221-224 -> 109-111 / 448-456 -> 238-255 |
| bge-base F16 | 256 / 512 | 93-96 -> 1.6-4.0 / 192-194 -> 1.8-2.0 | 176-180 -> 78-92 / 367-377 -> 171-181 |
| bge-small Q8_0 | 256 / 512 | 56.5-57 -> 1.4-4.7 / 113-114 -> 1.5-1.6 | 110-114 -> 51-59 / 241-243 -> 112 |
| bge-small F16 | 256 / 512 | 56 -> 2.5-3.8 / 113-114 -> 1.5-1.7 | 99-100 -> 54-62 / 213-218 -> 90-100 |
| MiniLM Q8_0 | 256 / 512 | 23.6 -> 2.6-3.2 / 47 -> 0.8-1.9 | 53-68 -> 43-81 (noisy) / 111-114 -> 57-60 |

  The GELU stage drops 25-100x and the whole document is 1.9-2.2x faster at 256-512 tokens (bge-small Q8 at 512: 241-243 ->
  112 ms; bge-base Q8: 448-456 -> 238-255 ms). The 64-token cells are cold-start noisy (8 iterations) and the MiniLM 256 total
  has one contended rep. 48-text e2e docs/s (median of 3 reps): bge-base Q8 7.8 -> 14.2-14.7, bge-base F16 9.4-9.6 -> 20.6-21.0,
  bge-small Q8 14.8-14.9 -> 30.9-33.1, bge-small F16 16.7-16.8 -> 39.7-40.7, MiniLM Q8 32.5-33.1 -> 64.0-65.3. Raw rows:
  `benchmarks/EmbeddingBench/results/ab-gelu-1403.jsonl` (rerun on the merged tree; the earlier pre-attention run is kept in
  the same file before it: it showed the same Act-stage drop with totals diluted by serial attention).
* Amdahl view at 32 threads, bge-base Q8 at 64 tokens: ~44 % of the document is the two serial stages (Attention + GELU).

### Inter-sequence parallelism and server batching (#1405)

`IEmbeddingModel.ForwardHiddenBatch(sequences, consume, device)` (default = sequential loop). `BertEncoderModel` overrides it:
sequences are sorted longest-first, then run **one sequence per core, single-threaded** when the batch is at least as large as the pool,
or an even thread split in between (`G = min(B, T / (T/B))` concurrent sequences, each with its own `T/B`-thread sub-pool; B=1 uses the
whole pool as before). Results are **bit-identical** to one-at-a-time (`BertEmbeddingBatchParityTests`: batch sizes 1,2,3,5,13,40,
Q8_0 / F16 / nomic; every GEMM output is independent of partitioning). The server (`EmbeddingBatcher`, shared by `/v1/embeddings` and
`/api/embed`) queues requests, waits `DOTLLM_EMBED_BATCH_WINDOW_MS` (default 2 ms, only while fewer than `ProcessorCount` sequences are
pending), takes the request gate once per batch and fans results back per request in input order. A failing batch is replayed per request so one
bad request cannot fail its neighbours. `DOTLLM_EMBED_PAR_MIN` (default 2) is the smallest batch that uses the parallel path.

Measured, bge-base Q8_0, 32 threads (16C/32T), `cpu-bench` held, docs/s (min-max):

| Batch (in-process `EmbeddingBench batch`) | 1 | 2 | 4 | 8 | 16 | 32 | 128 |
|---|---|---|---|---|---|---|---|
| seq 64, sequential (before, flat) | 14-15 | | | 11-15 | | 15 | 15 |
| seq 64, batched | 14-15 | 24-27 | 33-35 | 45-47 | 49-54 | 35-51 | 52-53 |
| seq 256, sequential (before) | 2.8 | | | 2.8 | | | 2.8 |
| seq 256, batched | 2.7-2.9 | 2.6-2.9 | 2.9-3.3 | 3.6 (one 6.1) | | 11.3-12.0 | 11.5-11.7 |

(The seq-256 batched rows used the first build, with only the one-thread-per-sequence path for B >= 4 and no sub-pool split; seq-64 rows
used the final build.) Saturation is **3.5x at seq 64 and 4.1x at seq 256**, not the 4-8x estimated: one thread embeds a 64-token document in
~266 ms (3.75 docs/s); 8 threads give 26 docs/s, 16 threads 43.6, 24 threads 48.9, 32 threads 52.6, so the ceiling is the 16 physical cores
(~3.3 docs/s each, SMT adds ~20 %). Further gains must come from per-thread efficiency (#1402/#1403/#1404 shrink each sequence's time and
compose with this).

HTTP, `EmbeddingBench http` (token-id inputs, one document per request, N concurrent clients, seq 64, two paired server runs each):

| Clients | before docs/s (p50 ms) | after docs/s (p50 ms) |
|---|---|---|
| 1 | 9.0-10.3 (102-124) | 9.0-10.6 (96-111) |
| 8 | 9.7-11.4 (767-824) | 31.6-36.1 (220-251) |
| 32 | 10.8-11.5 (2.8-3.0 s) | 39.8-40.9 (774-836) |

A single client is unchanged within noise (the 2 ms window is invisible against a 100 ms document); concurrency now scales 3.3x at 8 clients
and 3.7x at 32. Decoder embedders (`TransformerModel`, Qwen3-Embedding) use the default sequential implementation, so they gain request
coalescing but no parallelism.

## 5. Vulkan / CUDA

**Decoder embedders run on Vulkan (#1406).** `VulkanTransformerModel` implements `IEmbeddingModel`: the forward is the unchanged
graph up to llama.cpp's `result_norm`, then the final RMSNorm of the pooled row(s) only, written straight into a host-readback buffer.
No LM head runs, and for last/cls pooling only **one row** (`hiddenSize` F32 = 4 KiB for Qwen3-Embedding) is copied, normalised and read back;
mean pooling reads all rows. The new `IEmbeddingModel.ForwardHiddenForPooling(sequences, pooling, consume, device)` delivers exactly the rows
pooling reads (default = slice of `ForwardHiddenBatch`, so the CPU models are unchanged); `EmbeddingBatcher`, `dotllm embed` and
`EmbeddingBench` all use it. Cacheless forward, positions 0..n-1, one sequence per submit (no cross-sequence batching on the GPU yet).
Weights stay F16 and resident (no expansion); the only added storage is the readback buffer (`rows x hidden x 4 B`, one row for last/cls).

Use: `dotllm embed <qwen3-embedding.gguf> -d vulkan "text"` (default device stays `cpu`), `dotllm serve ... --device vulkan` then
`POST /v1/embeddings` (batched requests go through the #1405 batcher; Vulkan runs the sequences back to back). Encoders (BERT / nomic-bert)
on a GPU device stay a **loud refusal**: `embed` fails with a `DeviceUnavailableException` message naming #741; the server's encoder loader throws
(no silent CPU run, `auto` falls to CPU with the usual warning). Validated architecture: Qwen3 (Qwen3-Embedding-0.6B F16) only; other
architectures on `VulkanTransformerModel` share the graph but are unvalidated.

**Parity** (`Qwen3EmbeddingVulkanParityTests`, 9 texts: the committed 4-text sentence-transformers fixture + a 1-char text, unicode, 3 long
texts up to ~450 tokens): Vulkan vs CPU last-token cosine >= 0.9999 and Vulkan vs sentence-transformers fp32 >= 0.9999 (all pass), mean pooling
Vulkan vs CPU >= 0.9999, one-row readback identical to the last row of the all-row readback, and control arms (unrelated texts < 0.95; no-EOS Vulkan
vector < 0.99 vs the reference). `dotllm embed "Hello"` on cpu vs vulkan differs in the 4th-5th significant digit (F16 GEMM order).

**Throughput**, `EmbeddingBench throughput --device vulkan|cpu`, Qwen3-Embedding-0.6B F16, 8 documents back to back, `gpu-lock` + `cpu-bench` held,
paired in order V C C V V C (raw: `benchmarks/EmbeddingBench/results/vulkan-1406.jsonl`), docs/s | tokens/s, min-max over runs:

| seq | Vulkan (iGPU 8060S) | CPU (32 threads, tree with #1404) | ratio |
|---|---|---|---|
| 64 | 28.4-29.7 \| 1.8-1.9k | 12.7-13.1 \| 0.81-0.84k (one contended run 6.5) | 2.2x |
| 256 | 18.2-18.7 \| 4.7-4.8k | 3.16-3.34 \| 0.81-0.85k | 5.6x |
| 512 | 11.1-11.3 \| 5.7-5.8k | 1.41-1.50 \| 0.67-0.77k | 7.5-8x |

The "9-30x" quoted in the survey compared against the pre-#1404 CPU numbers (163-182 tok/s); F16 GEMM on the CPU has since improved ~4.5x, so
the honest Vulkan advantage today is 2x (short) to ~8x (512 tokens). Vulkan is per-document-latency bound at 64 tokens (35 ms: launch +
submit per document); batching several short documents into one forward is #1437, below.

**Packed varlen forward (#1437).** `VulkanTransformerModel.ForwardHiddenForPooling` now packs consecutive documents (>= 2 tokens each) into one
forward up to a token budget (`DOTLLM_EMBED_PACK_TOKENS`, default 1024; `DOTLLM_EMBED_PACK=0` turns it off). No padding: the token rows are
concatenated, so every GEMM / norm / RoPE / FFN op runs once over all rows with weights read once; positions restart per document (RoPE reads the
per-row positions buffer); attention is one causal coopmat-flash dispatch **per document** over its slice of Q/K/V/AttnOutput (sub-range
descriptors, byte offsets multiple of 256: no copies, no new F32 storage, no shader change), so a query never sees another document's keys; last/cls
pooling copies each document's pooled row into one `[docs, hidden]` norm + readback (4 KiB per document). Scope: `Architecture.Qwen` (validated on
Qwen3-Embedding-0.6B) with no window / soft-cap / sinks; other models and 1-token documents keep the one-document path. `/v1/embeddings`, `dotllm embed`
and `EmbeddingBench` get it through the existing `ForwardHiddenForPooling` call (the #1405 batcher already hands over the whole batch).

Per-document cost, Qwen3-Embedding-0.6B F16, iGPU 8060S, `gpu-lock` + `cpu-bench` held, paired off/on/on/off, min-max over reps (ms per document,
`benchmarks/EmbeddingBench/results/vulkan-pack-1437.jsonl`, `EmbeddingBench vkpack --device vulkan`). A forward costs a ~31 ms floor at <= 64 tokens
(weights streamed once, 28 layers of dispatches), so packing amortises it:

| doc tokens | one per submit | pack 4 | pack 8 | pack 16 (budget 1024) |
|---|---|---|---|---|
| 16 | 30.6-31.3 | 8.6 (3.6x) | 4.9 (6.3x) | 3.2 (9.7x) |
| 32 | 31.8-32.0 | 9.9 (3.2x) | 6.3 (5.1x) | 4.7-4.8 (6.7x) |
| 64 | 34.6-34.8 | 12.8 (2.7x) | 9.4-9.6 (3.6x) | 7.1-7.2 (4.9x) |
| 128 | 40.0-40.6 | 19.3-19.5 (2.1x) | 14.6-14.9 (2.7x) | 14.6-15.0 (2.7x, budget splits it into 2 x 8) |
| 256 (32 docs) | 54 | | | 31.6 at budget 1024, 29.7-30.0 at 2048+ (1.8x) |

Token budget sweep (32 x 64 tokens): 256 -> 12.7 ms/doc, 512 -> 9.4, 1024 -> 7.2, 2048 -> 6.7-6.8, 4096 -> 6.7; the default stays 1024 (scratch grows with
the budget; beyond it the win is <= 6 %). 64-token documents go from 29 docs/s to about 140 docs/s (4.9x).

Parity (`Qwen3EmbeddingPackedTests`, 10 documents of 2-110 tokens including EOS-only-ish short ones, unicode and a repeated long one): packed vs
one-at-a-time Vulkan and packed vs CPU last-token cosine >= 0.9999 for every document; reversed pack order gives the same per-document vectors;
mean pooling packed vs single >= 0.9999; a pack split by a 40-token budget equals single. **Control arm:** a test-only mutant turns the attention into
one causal dispatch over the concatenation (cross-document leak); the same comparison must then fail for the later documents (it does: the test
asserts at least N-2 of the N-1 later documents drop below 0.9999 and that document 0, which cannot see a leak, does not), so a wrong mask is detectable.
Not covered: BERT-class encoders (#741, bidirectional, would need the per-document slice with the Bidirectional mask), CUDA (no embedding path), other
decoder architectures (gated off until validated).
No CUDA run (T5500 needs the user's go-ahead). End-to-end `/v1/embeddings` on Vulkan returns 1024-d vectors (smoke-tested, 2 inputs, 10 prompt tokens).

**#741 (GPU BERT encoder), plan only, not started:** needs bidirectional attention (the Vulkan attention shader has a Bidirectional
mask mode from DiffusionGemma), LayerNorm + bias adds, absolute position / token-type embeddings, GELU, and mean/cls pooling; reuse the coopmat
prefill GEMM tiles for QKV/out/FFN, a padded-or-varlen batch of length-sorted documents (section 3 padding table) and the same
`ForwardHiddenForPooling` contract (mean = GPU-side reduction). Expect well above 5k tok/s on 110M-class encoders (unmeasured).

## 6. Ranked levers (expected wins, from the measurements above)

Win = change in end-to-end docs/s on CPU unless stated; "x" are estimates from the stage table, to be confirmed per PR with
the same harness, the cosine gate, and a discriminating shape.

1. (#1405) **Inter-sequence parallelism + batched `ForwardHidden` API** (biggest, ~4-8x at 64-256 tokens). 1-thread throughput is
   237 ms/doc (bge-base Q8, 64 tok) so 32 independent workers would give on the order of 60-130 docs/s (SMT and shared L3 will eat part of the ideal) versus 16 today, because attention, GELU,
   LN and bias then stop being Amdahl terms. Needs per-worker scratch and a server batcher (lever 5). Weight streaming (117 MB
   per document per worker) is far below the memory bandwidth at that rate.
2. (#1402, parallel part DONE) **Parallel + vectorised bidirectional attention** for short sequences (2.3x at 512, 1.6x at 256, 1.1x at 64 for bge-base Q8;
   lifts MiniLM/bge-small/nomic similarly). Today ~9-18 GFLOP/s on one thread; target >= 10x on that stage.
3. (#1403, DONE) **Vectorised/parallel GELU** (1.3-1.4x for bge/e5/gte/mxbai at 64-256 tokens: removes 18-31 % of the document).
4. (#1404) **F16 GEMM**: convert-once-per-tile and AVX-512 path, so F16 GGUFs match Q8_0 (>= 1.5-2x on F16 files; interim advice is to
   use the Q8_0 GGUF: 1.4-2x faster at 0.9996+ cosine).
5. (#1405) **Server dynamic batching** for `/v1/embeddings` (collect requests for a few ms, run lever 1 across clients; length-sort so a padded
   path never sees the 60-78 % waste above). Needed for any concurrency scaling; the current gate makes N clients no faster than 1.
6. (#1406, DONE) **Vulkan hidden-state path for decoder embedders** (Qwen3-Embedding-0.6B: 2.2x/5.6x/8x over the current CPU at 64/256/512 tokens, 29 docs/s at 64 tok),
   then the **GPU BERT encoder (#741)** reusing the coopmat prefill tiles (110M-class encoders are ~5x fewer FLOPs than the 0.6B
   decoder, so expect well above 5k tok/s; unmeasured).
7. **Correctness/coverage** (#1398 EOS: FIXED; #1401 XLM-R Unigram tokenizer: FIXED; #1399 gemma-embedding, #1400 nomic-bert-moe): Qwen3-Embedding needs `<|endoftext|>` appended before last-token pooling (llama.cpp does it; dotLLM
   mean cosine 0.80); add `gemma-embedding`, `nomic-bert-moe`, and XLM-R (SentencePiece) tokenisation for multilingual-e5/bge-m3.
8. Small/free: (#1407) p99 spikes on cold single-document F16 calls (876 ms at 16 tokens), content-hash doc cache at the server,
   int8/binary output formats, Matryoshka truncation is already supported (`--dimensions`), NPU background embedder (#1321, no measurement yet).

Not levers: tokenisation (<0.5 % of the time), pooling/normalise (<0.01 ms), embedding lookup (<0.2 ms).

## 7. Reproducing

```
dotnet build benchmarks/EmbeddingBench -c Release
EmbeddingBench throughput <gguf> [--threads N] [--batches 1,8,32,128] [--seqs 64,256,512] [--cell-seconds 5]
EmbeddingBench latency|profile|tokenizer|padding <gguf> ...
EmbeddingBench e2e <gguf> <texts.json>          # tokenise+embed docs/s on a fixed text set
EmbeddingBench dump <gguf> <texts.json> <out.json> [--pooling mean|cls|last] [--suffix <tokenId>] [--dims N]
EmbeddingBench llama <http://127.0.0.1:port> <gguf>   # token-id batches against llama-server /v1/embeddings
```

Take the `cpu-bench` lock for timing runs. Reference vectors: sentence-transformers fp32 (`max_seq_length = 512`) or llama-server
on the same GGUF (`-dev none`). Batch is "documents back to back"; see section 3.
