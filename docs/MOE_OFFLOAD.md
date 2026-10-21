# MoE expert offload on CUDA (issue #1070, design #1063)

Nemotron 3.5 Bolt's routed experts take 15.9 GiB (NVIDIA's IQ2_XXS file) or 40.2 GiB (AD-Q4_K_M), which does not fit a 12 GB GPU.
On CUDA, a `nemotron_h_moe` model whose weights do not fit keeps its routed experts in host memory and runs them through a VRAM expert cache.
Everything else stays on the GPU: the SSM and attention layers, the router, `latent_down` / `latent_up`, the shared expert, the norms and the LM head.

## Options

| Option (`run`, `chat`, `serve`, `bench`) | Env fallback | Default | Meaning |
|---|---|---|---|
| `--moe-offload auto\|off\|on` | `DOTLLM_MOE_OFFLOAD` | `auto` | `auto`: keep the experts resident when the model fits, else offload them. `off`: refuse a model that does not fit (the previous behaviour). `on`: always offload, even when the model fits. |
| `--moe-cache-mib N` | `DOTLLM_MOE_CACHE_MIB` | `auto` | Size of the VRAM expert cache. `auto` uses the whole measured budget. A larger request is clamped to the budget, with a warning. |
| `--moe-cache-policy slru\|lru\|lfu` | `DOTLLM_MOE_CACHE_POLICY` | `slru` | Replacement policy. SLRU keeps a protected segment for experts that are hit again and a probation segment for new arrivals. |

Library callers pass `MoeOffloadOptions` to `CudaModelLoader.CreateFromGguf` / `CreateForGpuLayers`. The options record also has these settings:

- `PinBudgetFraction`: default 0.45 of physical RAM.
- `ReserveMib`: default 512.
- `ContextTokens`: the KV length the budget reserves for. The default is the model's maximum context.

`--device auto` does not yet consider offload, so it chooses CUDA only when the whole file fits. Pass `--device gpu` to run an oversized file on CUDA.

## How it works

- **Host store.** The routed banks are copied once into pinned (`cuMemHostAlloc`) memory when they fit the pin budget. Otherwise they stay in the memory-mapped GGUF and reach the GPU through a pinned staging ring of 4 × 32 MiB.
- **Device budget** (the #1063 addendum's hard rule; Windows WDDM spills over-committed VRAM instead of failing):
  - Every other allocation is made first: weights, forward state, the MoE scratch and the F16 GEMM scratch. Each is sized for one 512-token MoE block.
  - Free VRAM is then measured with `cuMemGetInfo`. The budget is that figure minus the reserve, minus the KV cache at the maximum context, minus a two-wave staging arena.
  - The cache is allocated **once** within that budget and never grows.
  - After allocation, free memory must still cover the reserve plus the KV cache. A spill probe then compares sampled reads of the pool with reads of a small buffer and fails on a 5× slowdown.
  - With an automatic size, a failed check shrinks the cache by 10 % and retries, up to 3 times. An explicit size fails.
- **Slot pools.** There is one pool per expert layout (up + down bytes), shared by every layer with that layout. Each pool gets the same fraction of its experts.
- **Copy-on-miss.** After host routing, every missed expert of the layer is copied on a dedicated copy stream. The compute stream waits on an event before running the expert kernels.
- **Prefill.** A 512-token block can touch more experts than a pool holds. Experts that find every slot taken by the same call stream through the staging arena in waves: wave i+1 copies while wave i computes.
- **Same numerics.** The expert kernels and the slot-ordered weighted sum are the ones the resident path uses, so offloaded logits are bit-identical to resident ones. Only the weight pointers and the order of the expert groups differ.
- **Prefill chunks.** An offloaded model prefers prefill chunks of one MoE block (`IModel.PreferredPrefillChunkTokens`). That keeps the forward state inside the budget. An explicit `--prefill-chunk-size` above 512 grows it past the budget.

Cold experts computed on the CPU or the iGPU (promote-only admission) are PR C / PR D of #1063.

## Telemetry

The `DotLLM.Moe` meter publishes the following instruments:

| Instrument | Kind |
|---|---|
| `dotllm.moe.expert_hits` | counter |
| `dotllm.moe.expert_misses` | counter |
| `dotllm.moe.evictions` | counter |
| `dotllm.moe.bytes_h2d` | counter |
| `dotllm.moe.copy_wait_us` | counter |
| `dotllm.moe.cache_slots` | gauge |
| `dotllm.moe.cache_bytes` | gauge |

`copy_wait_us` is the compute stream's stall on copy events, timed with event pairs.

`bench` prints a one-line summary for the measured run, and so does `run --verbose`:

```text
MoE offload: cache 6.1 GiB (812 slots, SLRU), hit 81.2%, H2D 71.3 MB/token, copy wait 9.8 ms/token, evictions 1234
```

Hits and misses count one lookup per expert per layer call: a prefill block that routes 30 tokens to one expert counts as one lookup.

## Tests

The GPU tests run on T5500.

- `CudaNemotronHMoeOffloadTests` (synthetic Bolt / Lightning fixture) runs a 1 MiB cache against a resident run and requires bit-identical logits.
  - It covers prefill and decode, waves of one expert, both store kinds and all three policies.
  - Its counters must match an offline `ExpertCache` replay of the recorded routing trace.
  - It also checks the budget.
- `CudaNemotronHBoltBlk9OffloadRealModelTests` runs the same comparison on the real 9-block truncation (`DOTLLM_BOLT_BLK9_GGUF`).

The CPU tests cover the cache bookkeeping, the budget arithmetic, the slot split and option parsing: `ExpertCacheTests` and `MoeOffloadPlannerTests`.
