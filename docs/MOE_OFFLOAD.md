# MoE expert offload on CUDA (issues #1070 and #1071, design #1063)

Nemotron 3.5 Bolt's routed experts take 15.9 GiB (NVIDIA's IQ2_XXS file) or 40.2 GiB (AD-Q4_K_M), which does not fit a 12 GB GPU.
On CUDA, a `nemotron_h_moe` model whose weights do not fit keeps its routed experts in host memory and runs them through a VRAM expert cache.
Everything else stays on the GPU: the SSM and attention layers, the router, `latent_down` / `latent_up`, the shared expert, the norms and the LM head.

## Options

| Option (`run`, `chat`, `serve`, `bench`) | Env fallback | Default | Meaning |
|---|---|---|---|
| `--moe-offload auto\|off\|on` | `DOTLLM_MOE_OFFLOAD` | `auto` | `auto`: keep the experts resident when the model fits, else offload them. `off`: refuse a model that does not fit (the previous behaviour). `on`: always offload, even when the model fits. |
| `--moe-cache-mib N` | `DOTLLM_MOE_CACHE_MIB` | `auto` | Size of the VRAM expert cache. `auto` uses the whole measured budget. A larger request is clamped to the budget, with a warning. |
| `--moe-cache-policy slru\|lru\|lfu` | `DOTLLM_MOE_CACHE_POLICY` | `slru` | Replacement policy. SLRU keeps a protected segment for experts that are hit again and a probation segment for new arrivals. |
| `--moe-cold auto\|copy\|cpu` | `DOTLLM_MOE_COLD` | `auto` | Where an offloaded expert that is not in the cache runs. `auto`: CPU or copy-to-GPU per expert, by the calibrated cost model. `copy`: copy every miss into the cache (the #1070 behaviour). `cpu`: compute every uncached expert on the CPU. In `auto` and `cpu`, an expert enters the cache only through promote-only admission. |
| `--moe-cpu-threads N` | `DOTLLM_MOE_CPU_THREADS` | physical cores − 1 | Threads of the CPU cold leg, including the forward's own thread. Each thread computes one expert at a time. |
| `--cpu-moe` | `DOTLLM_CPU_MOE=1` | off | llama.cpp's `--cpu-moe`: the routed experts of every MoE layer stay on the CPU. They are never cached and never copied. |
| `--n-cpu-moe N` | `DOTLLM_N_CPU_MOE` | 0 | llama.cpp's `--n-cpu-moe`: the same for the layers whose **index** is below N. On Bolt, layer 0 is a Mamba layer, so `--n-cpu-moe 2` places one MoE layer. `bench` also passes it to the Vulkan Qwen3MoeHybrid offload (#370). |
| `--moe-calibrate` | `DOTLLM_MOE_CALIBRATE=1` | off | Re-run the load-time calibration instead of reading the cached record. |
| `--ctx-size N` (`--ctx`) | `DOTLLM_MOE_CTX` | the model's maximum | Context length the device budget reserves KV cache for. Bolt's maximum is 262144 tokens, so `--ctx-size 8192` leaves that KV to the expert cache. `bench` defaults it to its own prompt + depth + generated tokens. |

Library callers pass `MoeOffloadOptions` to `CudaModelLoader.CreateFromGguf` / `CreateForGpuLayers`. The options record also has these settings:

- `PinBudgetFraction`: default 0.45 of physical RAM.
- `ReserveMib`: default 512.
- `ContextTokens`: the KV length the budget reserves for (`--ctx-size`). The default is the model's maximum context.
- `CalibrationPath`: the calibration cache file (default `%LOCALAPPDATA%\dotllm\moe-calibration.json`).
- `CostModelOverride`: a fixed `MoeCostModel` instead of the calibration.

`--cpu-moe` / `--n-cpu-moe` offload the experts even when the model would fit, and cannot be combined with `--moe-offload off`.

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

## Cold experts on the CPU (issue #1071)

Over a slow host link, copying every miss costs more than computing the expert where its weights already live. On the dev box (RTX 3060, PCIe 3.0 x4), Bolt AD-Q4_K_M with a 7 % cache misses about 554 MB per token, roughly 170 ms per token over the link.

- **The latent cut.** On a layer that may use the CPU, `latent_down` runs before the router read-back event. The router logits and `x_latent` (4 KB per token, F32) are both copied into pinned memory under that one event. A layer without a latent reads back the normed `h` instead.
- **Placement.** After host routing, each used expert of the layer goes to one of three places:
  - the device, when it is cached;
  - the device after a copy (admitted by the cache's policy, or streamed in waves);
  - the CPU.
- **The CPU leg.** The device launches its own groups first. Meanwhile the CPU workers compute the CPU set **expert-parallel**: one expert per worker at a time, each single-threaded.
  - The workers run the CPU model's own kernels on the memory-mapped bank slices (`NemotronHMoeExpertCompute`, which shares `NemotronHTransformerModel.GemmDispatch` with the CPU model).
  - The workers come from `Parallel.For` capped at `--moe-cpu-threads`, so a layer call creates no threads, and the forward's thread joins in. On the busy dev box an empty 9-expert dispatch cost 6–9 µs this way, while a dedicated spin-wait pool collapsed to milliseconds once other processes took its cores.
- **Merging.** Each CPU slot's output row goes into pinned memory. The rows are then copied to the device on the compute stream, after the device-computed rows of the same buffer. The existing gather kernel puts every row back in slot order.
- **Same summation order.** The weighted sum, `latent_up` and the shared-expert add are unchanged, so the summation order is the resident order. Only the CPU experts' GEMV rounding differs (the CPU kernels quantize activations to Q8_K).
- **No CPU experts, no extra cost.** A layer call with no CPU expert skips the copy and the gather.

### Assignment (`--moe-cold auto`)

Per layer call, the uncached experts are assigned by a pure function (`MoeColdAssignment`):

1. **Fiddler rule.** An expert goes to the CPU when `cpuUs(n_e) < copyUs(e) + gpuUs(n_e)`.
2. **Greedy balance (DALI-style).** Experts are visited in ascending order of `|cpu − (copy + gpu)|`. Each is moved to the other side when that lowers the layer's makespan, `max(CPU span, device span)`.
   - The CPU span is `round trip + max(Σcpu / threads, max cpu)`.
   - The device span is the cached experts' time plus `Σ(copy + gpu)` of the copied ones.
   - The pass starts from the best of the Fiddler assignment and the two one-sided assignments.

When the calibration cannot run, `auto` falls back to `copy`.

### Promote-only admission (`auto` and `cpu`)

An expert computed on the CPU is copied into the cache only if it scores higher than the weakest unpinned resident of its pool, or if a slot is free (`WindowedPromotionAdmission`).

- **Score.** Every routed token adds 1 to its expert's score. Every 64 tokens, all scores are halved.
- **Budget.** The copies of one forward may use about 25 % of the link time of the previous token, and always at least one expert.
- **Copy path.** The copy rides the copy stream behind every other copy of the call. A later hit on that slot waits for its event.
- **Trade-off.** In the #1063 section 4 simulation, this made 3.6–28× fewer copies than LRU for 5–7 points less hit rate.

### Calibration

When the experts are offloaded and the mode is `auto` or `cpu`, the load measures four things, in well under a second:

| Measurement | How |
|---|---|
| Pinned H2D bandwidth | A 16 MiB copy, four times |
| D2H + sync round trip | A 4 KB read-back and event sync, median of 50 |
| CPU time per expert layout | One expert at n = 1 and n = 8, single-threaded, median of 10, banks pre-touched |
| Device time per expert | Not measured: the expert bytes at 300 GB/s, one weight read per 8 tokens |

- **Cost model.** The CPU cost is fitted as `a + b·n` from the two CPU measurements.
- **Cache.** The record is stored in `%LOCALAPPDATA%\dotllm\moe-calibration.json`. The key is the GPU name, its PCI bus id, the CPU model, the model file (path, size and modification time) and the thread count.
- **Re-running.** Pass `--moe-calibrate` to measure again. The load log prints the numbers and the mode, for example:

```text
MoE cold experts: calibrated in 412 ms: H2D 5.39 GB/s, sync round trip 51 us; CPU us/expert (n=1 / n=8): [2.62 MB: 801 / 2553] ...
MoE cold experts: auto mode, 15 CPU thread(s).
```

Expected outcomes, still to be measured: T5500 (PCIe 2.0 x16, pre-AVX Xeons) is expected to choose copy, and the dev box (PCIe 3.0 x4, AVX2) the CPU.

## Other MoE models on CUDA: CPU-expert layers (issue #1187)

MoE models that load through the plain `CudaTransformerModel` (Recursive Flint, DeepSeek-V2, Qwen-MoE, ...) honour
`--cpu-moe`, `--n-cpu-moe N` and `--moe-offload` too. There is no VRAM expert cache on this host yet: a layer's routed
experts are either all resident or all on the CPU for decode; multi-token calls stream part of them to the GPU (#1202, below).

- **Placement.** Layers whose index is below N keep their routed experts on the CPU (`--cpu-moe` = every MoE layer,
  `--moe-offload on` = the same). Their router, shared expert and biases are still uploaded; the routed banks are not.
- **Auto.** With `--moe-offload auto` (the default) and no explicit N, a model whose weights do not fit picks the smallest
  N that leaves free VRAM for `ReserveMib` (default 1024 MiB on this host) plus an F16 KV cache of `--ctx-size` tokens
  (default 4096), plus the GPU leg's expert ring and its margin (376 MiB for Flint) once some experts go to the CPU
  anyway. The load log says what it chose, for example
  `MoE experts on CPU: 32 layer(s) (--n-cpu-moe 34), 15360 MiB read from the mmap'd GGUF, 15 CPU thread(s); 9423 MiB device weights.`.
  `auto` never moves the experts of a model with per-expert biases or `swiglu_oai` (gpt-oss): the CPU leg cannot run them,
  and an explicit `--cpu-moe` on such a model is refused.
- **The CPU leg** (`CudaMoeHostExperts`). Routing and top-k stay on the device. The top-k ids, the top-k weights and the
  F32 expert input (`seqLen x (hidden + 2 topK)` values) come back in one round trip: async copies into pinned memory and
  one stream sync (#1202; each synchronous copy cost a link round trip on the eGPU). The CPU model's own kernels
  (`MoeSwiGluMlp.ExecuteRoutedFromAssignments`, Q8_1 activations) compute the routed sum **straight from the
  memory-mapped GGUF**, and the sum (`seqLen x hidden` floats) is queued back asynchronously and added. The device shared
  expert is queued before the CPU leg starts, so the two overlap; `shared + routed` is one addition per element, so the
  order matches the resident path.
- **CPU leg threads.** `--moe-cpu-threads`, else physical cores - 1 from the detected topology (#1202: the old default
  halved the logical count, assuming SMT, which gave 7 threads on a 16-core Core Ultra 7 155H; 12-15 decode 20-40 % faster
  there). The n=1 decode dispatch hands out rows in small claims instead of a fixed 1/N slice, so a slower E-core or a
  preempted thread no longer holds up every dispatch.
- **The GPU leg for multi-token calls** (`CudaMoeExpertStream`, #1202). A prefill chunk routes to nearly every expert, and
  the CPU computing all of them was 22 of 25 s at pp512. Like llama.cpp's op offload for large batches, the experts of a
  CPU layer are copied to the GPU and run there, but split with the CPU so that both legs finish together:
  - experts `[0, split)` are copied into a VRAM ring of 8 slots x 8 experts (120 MiB for Flint) and run in waves by the
    same bucketed expert body as resident layers (below); experts `[split, E)` run on the CPU leg at the same time. The
    split minimises `max(link time of the GPU experts with rows, CPU time of the CPU assignments)` with the link rate
    measured once (2.25 GB/s on the 3060 eGPU) and an EMA of the CPU leg's ms per assignment;
  - the next CPU layer's first waves are copied ahead of its routing, so the link keeps working through its attention;
  - **no second copy of the weights**: the expert banks' mmap'd pages are page-locked in place (`cuMemHostRegister`,
    read-only) and DMA'd straight from the OS page cache, so the copies are asynchronous with no pinned or staged host
    copy (rule 3: the page cache is the source; a cold file is read once by the page-in, then every prefill chunk sends
    1.875 MiB per Flint expert over the link, ~0.83 ms at 2.25 GB/s; the first DMA of each page also pays for locking it,
    so the first prefill after load is slower);
  - the ring is admitted against free VRAM (with 256 MiB to spare, WDDM spills silently instead of failing) on the first
    multi-token call, and auto placement reserves it. Without room, or when page-locking fails, prefill keeps the CPU
    leg (logged). `DOTLLM_CUDA_MOE_STREAM=0` turns the GPU leg off;
  - the CPU part's routed sum is uploaded as F16, divided by a power of two when its max |x| exceeds 16384 (Flint's MoE
    output reaches 3.3e4, #1191) and multiplied back on the device;
  - decode (one token, 6 experts) stays on the CPU leg: 11 MiB of experts per layer would take 5 ms on the link.
- **Bucketed multi-token expert body** (#1202, every MoE model on this host, resident or streamed). A multi-token call
  buckets its assignments by expert once on the host, uploads the row maps in one copy, gathers all rows in one launch,
  runs gate / up / down per expert on its row range, one SwiGLU per range and one combine kernel that sums each token's
  rows in ascending expert order. The per-expert loop it replaces made one synchronous host copy per expert and one per
  row of mixed-slot buckets (365 ms per Flint layer at 512 tokens, now 27 ms); the results are bit-identical.
- **Device-routed decode** (#1182): a resident layer on the fully grouped Q4_1 decode body builds its six pointer arrays
  on the device from the top-k ids (`moe_build_grouped_ptrs`, experts in ascending id order, a per-layer device table of
  expert pointers), so it neither waits for the routing nor reads it back: three link round trips fewer per resident
  layer, bit-identical logits. CPU-expert layers still read the ids back (one round trip). `DOTLLM_CUDA_MOE_DEVICE_ROUTING=0`
  restores host routing.
- **Token embeddings on the host** (#1183): an F32 `token_embd` (Flint: vocab 150,144 x 2048, 1.15 GiB) stays in the
  mmap; the forward gathers the call's rows, converts them to F16 and uploads `hidden x 2` bytes per token. llama.cpp
  keeps the table on the host too. Auto placement uses the freed VRAM for three more resident MoE layers on Flint.
- **Routing top-k** (#1202): `moe_sigmoid_topk_f32` / `moe_softmax_topk_f32` picked the top-k on one thread with a
  "claimed" inner loop, 210 us per token for 256 experts / top-6 on the 3060 (10 ms per decoded Flint token across 47
  layers). They now run k rounds of a block-wide arg-max with the same tie rule (lower index wins), a few microseconds.
- **No second copy of the weights** (#1156). Unlike the Nemotron-H host store above, nothing is pinned or copied: the
  CPU reads the OS page cache's pages of the file, which every process mapping the file shares. Only activations cross
  the link.

### Measured: Recursive Flint Q4_1 (26 GB) on an RTX 3060 12 GB (eGPU, Core Ultra 7 155H, 96 GB RAM)

`dotllm bench -p 512 -n 64 -r 2 --warmup 1` vs `llama-bench -p 512 -n 64 -r 2` (llama.cpp master CUDA, default 16
threads), three interleaved rounds, median (min-max). Peak VRAM is device-wide `nvidia-smi` minus idle; host is the
process's peak private bytes minus its VRAM for dotLLM (WDDM counts its device allocations in private bytes), private
bytes for llama.cpp (it does not). "before" is `dev` at 132eaf4c (#1187 + #1191).

| engine, placement | pp512 tok/s | tg64 tok/s | peak VRAM | host |
|---|---|---|---|---|
| dotLLM before (#1187), `--n-cpu-moe 34` | 24.8 (24.5-24.9) | 13.6 (13.6-13.9) | 10.25 GiB | 1.1 GiB |
| **dotLLM now, `--n-cpu-moe 34`** | **83.7** (83.5-84.2) | 19.1 (18.0-22.2) | 9.23 GiB | 0.9-1.2 GiB |
| llama.cpp `-ncmoe 34` | 79.5 (78.9-79.8) | 21.3 (19.2-22.0) | 8.57 GiB | 0.8 GiB |
| **dotLLM now, `--n-cpu-moe 36`** (less VRAM than llama.cpp `-ncmoe 34`) | **80.0** (79.9-80.1) | 19.7 (19.7-20.9) | 8.23 GiB | 0.9-1.2 GiB |
| **dotLLM now, auto** (= `--n-cpu-moe 31`) | **88.3** (88.0-88.4) | **21.0** (19.7-22.0) | 10.73 GiB | 1.0-1.2 GiB |
| llama.cpp `-ncmoe 31` | 87.5 (87.3-87.7) | 19.5 (18.8-22.5) | 9.97 GiB | 0.8 GiB |

- Prefill is link-bound: the GPU leg sends ~190-225 of 256 experts per CPU layer over the 2.25 GB/s eGPU link while the
  CPU computes the rest. It now beats llama.cpp at every placement, also at lower VRAM.
- Decode is at parity within this machine's noise (the 155H is shared with other work and runs ThrottleStop; single
  runs vary 15-23 tok/s for both engines). Per token: CPU leg ~17-30 ms, kernel launches ~1,000 at 15-31 us each on WDDM
  (#1210). Runtime scratch keeps dotLLM ~0.65 GiB of VRAM above llama.cpp at the same placement (#1211).
- The first prefill after load runs at ~40 tok/s: page-locking the mapped experts is lazy on this driver, so the first
  DMA of each page pays for it.
- Quality vs the dotLLM CPU model (`CudaFlintCpuMoeRealProbe`, 5 chat prompts x 32 teacher-forced tokens, auto
  placement): top-1 157-158/160, KL mean 2.0-2.4e-3 (GPU leg off: 157/160, 1.5e-3). Against the CPU model with exact F32
  expert activations (`DOTLLM_Q4E_DIAG_F32ACT=moe`): 160/160 and KL 5.6e-4 with the GPU leg vs 7.7e-4 without: the
  streamed experts take F32 activations, so they move prefill away from the CPU leg's Q8_1 noise, towards the exact result.

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
| `dotllm.moe.cold_experts_cpu` | counter |
| `dotllm.moe.cpu_compute_us` | counter |
| `dotllm.moe.cpu_wait_us` | counter |
| `dotllm.moe.promotions` | counter |

`copy_wait_us` is the compute stream's stall on copy events, timed with event pairs.

The CPU-leg instruments (#1071):

- `cpu_wait_us`: the compute stream's stall between the end of its own expert groups and the arrival of the CPU rows.
- `cpu_compute_us`: the host wall time of the CPU legs.
- `cold_experts_cpu`: routed experts computed on the CPU.

`bench` prints a one-line summary for the measured run, and so does `run --verbose`:

```text
MoE offload: cache 6.1 GiB (812 slots, SLRU), hit 81.2%, H2D 71.3 MB/token, copy wait 9.8 ms/token, evictions 1234
```

When a CPU leg ran, the line continues:

```text
; cold: CPU 41% of routed, CPU compute 5.0 ms/token, CPU wait 3.2 ms/token, promotions 12
```

Hits and misses count one lookup per expert per layer call: a prefill block that routes 30 tokens to one expert counts as one lookup.

## Tests

`CudaFlintCpuMoeTests` (#1187, local RTX 3060) runs the synthetic Flint GGUF with `--cpu-moe`, `--n-cpu-moe 4` (a split
across MoE layers 2-3 / 4-5) and `--moe-offload on` against the CPU model, prefill plus 6 KV-cache decode steps, checks
that each decode token computes exactly `topK` experts per CPU layer, and unit-tests the auto placement. Each placement
runs with the prefill GPU leg (#1202) off, on with its cost model, and forced to split every CPU layer half / half in waves
of two experts (several waves per layer, ring slots reused across layers, the next layer prefetched, a CPU part; once
more with the CPU part's power-of-two F16 scale forced on). `PlanSplit_BalancesLinkAgainstCpu` unit-tests the split.
`CudaMoeFfnTests.MoeFfn_MultiToken_MixedSlots_BucketedMatchesLoopAndCpuOracle` checks the bucketed body bit-for-bit
against the per-expert loop (24 tokens, top-3 of 8, gated shared expert) and `CudaQ4_1KernelTests.SigmoidTopk_*` the
parallel top-k's tie rule within and across warps.

The Nemotron-H GPU tests run on T5500.

- `CudaNemotronHMoeOffloadTests` (synthetic Bolt / Lightning fixture) runs a 1 MiB cache against a resident run and requires bit-identical logits.
  - It covers prefill and decode, waves of one expert, both store kinds and all three policies.
  - Its counters must match an offline `ExpertCache` replay of the recorded routing trace.
  - It also checks the budget.
- `CudaNemotronHBoltBlk9OffloadRealModelTests` runs the same comparison on the real 9-block truncation (`DOTLLM_BOLT_BLK9_GGUF`).
- `CudaNemotronHMoeColdCpuTests` (#1071) covers the cold-expert paths on the same fixture:
  - `--moe-cold cpu` with a 1 MiB cache against a resident run, prefill and decode. The per-layer selection must be identical, and the logits must stay within the cross-backend bound and small against the selection-bias control.
  - `--n-cpu-moe` / `--cpu-moe`: those layers never reach the cache.
  - `auto` with a CPU that never wins is bit-identical to resident.
- `CudaNemotronHBoltBlk9ColdCpuRealModelTests` compares CPU-cold experts with resident ones on blk9: mean KL ≤ 0.02 and top-1 agreement ≥ 90 %.

The CPU tests cover:

- the cache bookkeeping, the budget arithmetic, the slot split and option parsing: `ExpertCacheTests` and `MoeOffloadPlannerTests`;
- the cost model and the assignment: `MoeColdAssignmentTests`;
- non-admitting lookups and promote-only admission: `ExpertAdmissionTests`;
- the calibration key, store and options: `MoeCalibrationTests`;
- the per-expert CPU API and the cold leg's workers and gather index: `NemotronHMoeExpertComputeTests`. The API must be bit-identical to the kernels the CPU model used before, with or without a pool, and the leg must give the same rows for any thread count.
