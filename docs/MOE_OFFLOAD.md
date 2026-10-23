# MoE expert offload on CUDA (issues #1070, #1071 and #1083, design #1063)

Nemotron 3.5 Bolt's routed experts take 15.9 GiB (NVIDIA's IQ2_XXS file) or 40.2 GiB (AD-Q4_K_M), which does not fit a 12 GB GPU.
On CUDA, a `nemotron_h_moe` model whose weights do not fit keeps its routed experts in host memory and runs them through a VRAM expert cache.
Everything else stays on the GPU: the SSM and attention layers, the router, `latent_down` / `latent_up`, the shared expert, the norms and the LM head.

## Options

| Option (`run`, `chat`, `serve`, `bench`) | Env fallback | Default | Meaning |
|---|---|---|---|
| `--moe-offload auto\|off\|on` | `DOTLLM_MOE_OFFLOAD` | `auto` | `auto`: keep the experts resident when the model fits, else offload them. `off`: refuse a model that does not fit (the previous behaviour). `on`: always offload, even when the model fits. |
| `--moe-cache-mib N` | `DOTLLM_MOE_CACHE_MIB` | `auto` | Size of the VRAM expert cache. `auto` uses the whole measured budget. A larger request is clamped to the budget, with a warning. |
| `--moe-cache-policy slru\|lru\|lfu` | `DOTLLM_MOE_CACHE_POLICY` | `slru` | Replacement policy. SLRU keeps a protected segment for experts that are hit again and a probation segment for new arrivals. |
| `--moe-cold auto\|copy\|cpu\|igpu\|cpu+igpu` | `DOTLLM_MOE_COLD` | `auto` | Where an offloaded expert that is not in the cache runs. `auto`: copy-to-GPU, CPU or (#1083) the integrated GPU per expert, by the calibrated cost model; the iGPU only when an integrated Vulkan GPU exists. `copy`: copy every miss into the cache (the #1070 behaviour). `cpu`: compute every uncached expert on the CPU. `igpu`: on the integrated GPU (a layer it cannot read falls back to the CPU). `cpu+igpu`: split between the CPU and the iGPU by the calibrated costs, never copying on a miss. In every mode but `copy`, an expert enters the cache only through promote-only admission. `igpu` and `cpu+igpu` fail the load when there is no integrated GPU. |
| `--moe-igpu-device N\|name` | `DOTLLM_MOE_IGPU_DEVICE` | first integrated GPU | The Vulkan device of the iGPU leg: an enumeration index or a case-insensitive name substring. An NVIDIA device is never used; a non-integrated pick is allowed with a warning (its banks are then read across its bus). |
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

## Cold experts on the integrated GPU (issue #1083, design 8c′)

The dev box's Core Ultra 7 155H has an Arc iGPU (Meteor Lake) that shares the 95 GB of system RAM, where the routed banks already are. The iGPU leg computes cold experts there, next to the CPU leg.

- **Device.** `VulkanIgpuDeviceSelector` picks the first integrated Vulkan GPU (never NVIDIA), or `--moe-igpu-device`. CUDA↔Vulkan interop across adapters is not supported (it needs a matching device UUID/LUID), so the leg is staged through host memory.
- **Banks, zero-copy.** `VulkanNemotronHMoeExpertService` imports every offloaded MoE layer's up and down banks from the memory-mapped GGUF (`VK_EXT_external_memory_host`, `VulkanWeightImportPolicy`), so the iGPU reads the pages the CPU leg reads. If the import is refused, the bank is staged into device memory within a budget (10 % of RAM, a second copy on UMA), and the log says so. A layer over the budget, or with a bank type the Vulkan expert kernels cannot read (anything but Q4_K, Q5_K, Q6_K, Q8_0, Q5_0/Q5_1 and the IQ family; IQ banks need integer dot product), is refused loudly and its cold experts stay on the CPU.
- **Kernels.** The service records the Vulkan Nemotron-H routed-expert kernels through the model's own (now static) recording code into its own scratch: the indexed dp4a MMVQ (or F32) arm below 16 rows, the expert-grouped MMQ / coopmat prefill from 16 rows, the IQ banks of #1058. It runs top-k 1 with unit weights, so every row comes back unweighted. No new shader.
- **Rows.** The iGPU groups come last in the layer's schedule, after the CPU's. Before the CPU workers start, the host gathers each iGPU row's `x_latent` row from the pinned read-back into the iGPU's host-visible input (an imported host allocation on UMA: a memcpy into the iGPU's own memory), records and submits without waiting. The CPU leg runs, then the host waits on the fence and copies the iGPU's rows into the pinned cold-row buffer right after the CPU's. One H2D and the existing `moe_gather_token_rows_f32` serve both legs (the gather index already covers every cold row in schedule order); no new CUDA kernel. The compute stream waits for both.
- **Not copied.** The output rows are copied once more on the host (iGPU host-visible output → pinned buffer) rather than written by the iGPU into the CUDA-pinned buffer: importing a sub-range of a `cuMemHostAlloc` allocation into Vulkan needs page alignment the buffer layout does not give. It is 4 KB per row.

### Three-way assignment

`MoeColdAssignment.Assign(costs, residentUs, cpuWorkers, coldLegOverheadUs, igpuSubmitUs, place)` extends #1071's to three sides:

- **Makespan** = `max(device span, cold span)`. The cold span is zero with no cold expert, else the round trip plus `max(CPU span, iGPU span)`, because the legs run concurrently. The iGPU span is `submit + Σigpu` (one queue) and is zero when its set is empty, so a layer call that puts nothing on the iGPU never pays the submission.
- **Fiddler** picks each expert's individually cheapest side (the device on a tie, then the CPU). The greedy pass starts from the best of Fiddler and the three one-sided assignments, visits experts in ascending order of the gap between their two cheapest sides, and moves each to whichever other side lowers the makespan most.
- **Without an iGPU** (`igpuSubmitUs` infinite) the result is exactly #1071's two-way assignment (pinned by a randomized test).
- `cpu+igpu` runs the same assignment with the copy cost set to infinity. `auto` uses the iGPU side only on layers whose banks the iGPU holds.

### iGPU calibration

When an iGPU leg exists, the calibration adds:

| Measurement | How |
|---|---|
| Submit + fence round trip | An empty submission (two barriers), median of 30 |
| iGPU time per expert layout | One expert in its own submission at n = 1 and n = 8 (gather, record, submit, fence, copy-out), median of 10 |

The per-expert iGPU cost is fitted from those two measurements **minus** the empty round trip, which becomes `MoeCostModel.IgpuSubmitUs`, paid once per layer call that uses the iGPU. The cache key gains `|igpu:<device>`; a key without an iGPU is unchanged, so a cached CPU-only record stays valid. A cached record without iGPU timings for the layouts the iGPU holds is re-measured.

`auto` opens the iGPU only when the cost model can use it: with the calibration, or with a `CostModelOverride` that has iGPU costs. When `auto` falls back to `copy`, the iGPU is closed again. `cpu+igpu` without a calibration becomes `igpu`.

Expected outcomes, to be measured in the quiet window: decode similar to the CPU (both read the same DRAM) minus the submit cost; prefill well ahead of AVX2, and the CPU cores freed.

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
| `dotllm.moe.cold_experts_igpu` | counter |
| `dotllm.moe.igpu_compute_us` | counter |
| `dotllm.moe.igpu_wait_us` | counter |

`copy_wait_us` is the compute stream's stall on copy events, timed with event pairs.

The CPU-leg instruments (#1071):

- `cpu_wait_us`: the compute stream's stall between the end of its own expert groups and the arrival of the CPU rows.
- `cpu_compute_us`: the host wall time of the CPU legs.
- `cold_experts_cpu`: routed experts computed on the CPU.

The iGPU-leg instruments (#1083):

- `cold_experts_igpu`: routed experts computed on the iGPU.
- `igpu_compute_us`: from the start of each iGPU submission (gather and record included) until its fence was observed.
- `igpu_wait_us`: how long the forward thread blocked on the fence after its own CPU-leg work. When both legs ran, `cpu_wait_us` is the compute stream's stall for both.

`bench` prints a one-line summary for the measured run, and so does `run --verbose`:

```text
MoE offload: cache 6.1 GiB (812 slots, SLRU), hit 81.2%, H2D 71.3 MB/token, copy wait 9.8 ms/token, evictions 1234
```

When a CPU leg ran, the line continues:

```text
; cold: CPU 41% of routed, CPU compute 5.0 ms/token, CPU wait 3.2 ms/token, promotions 12
```

and when an iGPU leg ran:

```text
; iGPU 30% of routed, iGPU compute 4.1 ms/token, iGPU wait 0.6 ms/token
```

Hits and misses count one lookup per expert per layer call: a prefill block that routes 30 tokens to one expert counts as one lookup.

## Tests

The GPU tests run on T5500.

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
- #1083, Vulkan only (an integrated GPU, no CUDA: Strix Halo or the dev-box Arc): `VulkanNemotronHMoeExpertServiceTests` holds the iGPU expert service to the CPU per-expert API on random Q4_K / Q5_K / Q6_K / Q8_0 / IQ2_XXS banks at n = 1, 3 and 17 tokens (2 to 34 rows: indexed and grouped arms) and on the synthetic Bolt fixture's banks, each against a double-precision reference within a per-element bound built from the actual activation rounding (block maxima) plus √K-scaled accumulation, propagated through relu² and the down projection.
- #1083, CUDA + an integrated GPU (the dev box, quiet window): `CudaNemotronHMoeColdIgpuTests` runs `igpu` and `cpu+igpu` with a 1 MiB cache against resident on the synthetic fixture (PR C's selection and logit checks, shared), `auto`'s iGPU gating, and blk9 (`DOTLLM_BOLT_BLK9_GGUF`) within mean KL ≤ 0.02 and top-1 ≥ 90 %.

The CPU tests cover:

- the cache bookkeeping, the budget arithmetic, the slot split and option parsing: `ExpertCacheTests` and `MoeOffloadPlannerTests`;
- the cost model and the assignment: `MoeColdAssignmentTests`;
- non-admitting lookups and promote-only admission: `ExpertAdmissionTests`;
- the calibration key, store and options: `MoeCalibrationTests`;
- the per-expert CPU API and the cold leg's workers and gather index: `NemotronHMoeExpertComputeTests`. The API must be bit-identical to the kernels the CPU model used before, with or without a pool, and the leg must give the same rows for any thread count.
- #1083: the three-way assignment, the iGPU calibration fields, the `igpu` / `cpu+igpu` options, the telemetry line and the iGPU device choice: `MoeIgpuColdTests`; the iGPU row list and the gather index over CPU + iGPU rows: `NemotronHMoeColdRowsTests`.
