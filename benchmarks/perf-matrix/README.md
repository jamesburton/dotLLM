# Performance matrix — model × host × runtime

Tracks whether dotLLM is **competitive on every machine and every model** against the
reference runtimes (llama.cpp, ollama). `results.csv` is the source of truth; append rows,
never rewrite history — trend analysis depends on the old rows staying put.

## Protocol (keep rows comparable)

- **Workload:** pp≈512 prefill + tg128 decode, greedy (temp 0), f16 KV, flash-attn on,
  full GPU offload, mmap on. Record the *actual* prompt token count and the context depth
  the decode ran at (`tg_ctx_depth` — llama-bench tg is depth 0; server-API tg is depth≈prompt).
- **Discipline:** quiet GPU, `scripts/gpu-lock.sh` held, back-to-back same-session A/B on
  UMA boxes (decode swings ~40% with CPU memory traffic — see LLMs.md). Median of ≥3,
  min also noted for decode when available.
- **Rows are point-in-time:** always fill `runtime_version` (llama.cpp build / ollama version /
  dotLLM commit) and `settings`. A new measurement = a new row with today's date.

## Columns

`date, host, device, backend, runtime, runtime_version, model, quant, pp_tok_s, tg_tok_s,
tg_ctx_depth, settings, notes`

Hosts: `strix-halo` (Radeon 8060S gfx1151, Vulkan), `t5500` (RTX 3060, CUDA),
`framework` (RTX 3060 + Intel Arc, Vulkan), `kaggle-t4` (dual Tesla T4, CUDA).

## Known coverage gaps (fill these)

- **t5500 / kaggle-t4 / framework have NO llama.cpp comparator rows** — dotLLM-only numbers
  can't prove competitiveness. Run llama-bench (same protocol) next time each box is used.
- ollama rows exist only on strix-halo.
- CPU-backend rows are functional smoke numbers, not tuned measurements.

## Reading the matrix (as of 2026-07-16, strix-halo Vulkan)

- Decode: dotLLM ≈ 0.55–0.72× llama.cpp (dense 0.55–0.57×; 26B MoE 0.72× after #137).
- Prefill: dotLLM ≈ 0.10× llama.cpp on dense — the dominant gap (#139 in flight).
- ollama ≈ 0.8× llama.cpp decode at equal settings (its own overhead); llama.cpp at
  optimum flags is the reference target everywhere.
