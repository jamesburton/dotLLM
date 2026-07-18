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

## Reading the matrix (as of 2026-07-18, strix-halo Vulkan)

- Decode: dotLLM ≈ 0.55–0.85× llama.cpp on dense (SmolLM now EXCEEDS fresh llama.cpp,
  3B 0.85×, 8B 0.70×); 26B MoE 0.72× after #137; 35B-A3B MoE ≈ 0.21–0.24× (resident-MoE,
  #372, 2026-07-18 fresh comparator).
- Prefill: dotLLM ≈ 0.71–0.79× llama.cpp on dense after #139/#366/#367/#378-380 (was
  0.10×); 35B-A3B MoE ≈ 0.022–0.023× — now the single biggest gap in the whole matrix.
  Root cause: MoE prefill has no dp4a indexed-MMQ kernel (scalar F32 only), unlike dense
  models' register-tiled MMQ or MoE decode's MMVQ path — the next campaign target.
- ollama ≈ 0.8× llama.cpp decode at equal settings (its own overhead); llama.cpp at
  optimum flags is the reference target everywhere.
- Qwen3.6-35B-A3B specifically: real end-to-end prefill+decode numbers only exist since
  2026-07-18 (issue #373's VK_ERROR_DEVICE_LOST at prefill>2 tokens closed
  cannot-reproduce — see `.docs/KERNEL_MAP.md` §7).
