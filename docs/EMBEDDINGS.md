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
| multilingual-e5-small (XLM-R) | bert | 118M | 384 | mean | 512 | **no**: `BERT GGUF vocabulary is missing [CLS]/[SEP]/[UNK]` | - |
| EmbeddingGemma-300M | gemma-embedding | 300M | 768 | mean | 2048 | **no**: `Unsupported GGUF architecture: 'gemma-embedding'` | - |
| nomic-embed-text-v2-moe | nomic-bert-moe | 475M | 768 | mean | 512 | **no**: `Unsupported GGUF architecture: 'nomic-bert-moe'` | - |

Not attempted (same code path as a row above, or outside the 20-600M download rule): bge-large / e5-large / gte-large
(bert), Qwen3-Embedding-4B/8B (qwen3 decoder), gte-Qwen2, bge-m3 and jina-v3/v4 (XLM-R / custom code, expected to hit the
same tokenizer wall as multilingual-e5). Encoder = bidirectional BERT-style (`BertEncoderModel`, CPU only);
decoder-based embedders (Qwen3-Embedding, last-token pooling) run on `TransformerModel`, also CPU only.

GPU: **no Vulkan or CUDA model implements `IEmbeddingModel`** (`grep` finds only `BertEncoderModel` and `TransformerModel`);
the server returns 501 for a GPU-backed chat model and refuses encoders on a GPU device (#741 is the work item).

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
A retrieval-recall set (BEIR slice) is still missing; the cosine+NN-1 gate is what guards speed work for now.

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
versus 60-250 ms to embed a document). Server-side `/v1/embeddings` was not load-tested: the code path serialises all
requests on `ExecuteOnModelAsync` and loops sequences, so concurrent clients add latency, never throughput.

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
* Amdahl view at 32 threads, bge-base Q8 at 64 tokens: ~44 % of the document is the two serial stages (Attention + GELU).

## 5. Vulkan / CUDA

No GPU encoder (#741): BERT/nomic-bert on a GPU device is refused by design. For the decoder-embedder case,
`dotllm bench -d vulkan` on Qwen3-Embedding-0.6B F16 (prefill is the whole embedding cost; one extra token), 3 reps, gpu-lock held:
**pp64 1.66k tok/s, pp256 4.3k, pp512 5.2-5.4k** (38 / 59 / 96-99 ms per document) versus 163-182 tok/s on CPU: **9-30x**.
That is ~26 docs/s at 64 tokens and ~10 docs/s at 512 on the iGPU - but the Vulkan Qwen3 model has no hidden-state
output (`IEmbeddingModel`), so `/v1/embeddings` cannot use it yet (lever 6). No CUDA run (T5500 needs the user's go-ahead).

## 6. Ranked levers (expected wins, from the measurements above)

Win = change in end-to-end docs/s on CPU unless stated; "x" are estimates from the stage table, to be confirmed per PR with
the same harness, the cosine gate, and a discriminating shape.

1. (#1405) **Inter-sequence parallelism + batched `ForwardHidden` API** (biggest, ~4-8x at 64-256 tokens). 1-thread throughput is
   237 ms/doc (bge-base Q8, 64 tok) so 32 independent workers would give on the order of 60-130 docs/s (SMT and shared L3 will eat part of the ideal) versus 16 today, because attention, GELU,
   LN and bias then stop being Amdahl terms. Needs per-worker scratch and a server batcher (lever 5). Weight streaming (117 MB
   per document per worker) is far below the memory bandwidth at that rate.
2. (#1402) **Parallel + vectorised bidirectional attention** for short sequences (2.3x at 512, 1.6x at 256, 1.1x at 64 for bge-base Q8;
   lifts MiniLM/bge-small/nomic similarly). Today ~9-18 GFLOP/s on one thread; target >= 10x on that stage.
3. (#1403) **Vectorised/parallel GELU** (1.3-1.4x for bge/e5/gte/mxbai at 64-256 tokens: removes 18-31 % of the document).
4. (#1404) **F16 GEMM**: convert-once-per-tile and AVX-512 path, so F16 GGUFs match Q8_0 (>= 1.5-2x on F16 files; interim advice is to
   use the Q8_0 GGUF: 1.4-2x faster at 0.9996+ cosine).
5. (#1405) **Server dynamic batching** for `/v1/embeddings` (collect requests for a few ms, run lever 1 across clients; length-sort so a padded
   path never sees the 60-78 % waste above). Needed for any concurrency scaling; the current gate makes N clients no faster than 1.
6. (#1406) **Vulkan hidden-state path for decoder embedders** (Qwen3-Embedding-0.6B: 9-30x over CPU, ~26 docs/s at 64 tok today's kernels),
   then the **GPU BERT encoder (#741)** reusing the coopmat prefill tiles (110M-class encoders are ~5x fewer FLOPs than the 0.6B
   decoder, so expect well above 5k tok/s; unmeasured).
7. **Correctness/coverage** (#1398 EOS: FIXED; #1399 gemma-embedding, #1400 nomic-bert-moe, #1401 XLM-R tokenizer): Qwen3-Embedding needs `<|endoftext|>` appended before last-token pooling (llama.cpp does it; dotLLM
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
