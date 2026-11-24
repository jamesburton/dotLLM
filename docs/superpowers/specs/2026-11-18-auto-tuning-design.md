# `dotllm tune` and `--auto`: measured settings per model x device — design

Date: 2026-11-18. Refreshed 2026-11-24 against `dev` at 6c67922e. Issue #1902.
Owner decisions are marked **(James)**; designer calls are marked **(call)**; the 2026-11-24 rulings are collected in "Decisions".

## What changed on 2026-11-24, and why

Between 11-18 and 11-24 the perf-gap campaign merged (#1929, #1938, #1939, #1940, #1942, #1954, #1956, #1960), along with
GenerationSession and taps (#1931, #1932, #1937), chunk identity (#1951) and the upstream sync (#1946). Two issues from the magnitude
review now bind this design: #1936 (measurement rules) and #1933 (device memory ledger).

| # | Change | Why |
|---|---|---|
| 1 | The knob list is now three tiers: **tuned**, **device crossovers (later, opt-in)** and **settled, never touched**. | The campaign turned most of the old "backend switches" into defaults backed by rulings (F16 partials, step graph, staged TQ prefill, row-lane Q4_K, E4, the 2^15 P scale). Re-deciding them per machine would undo sign-offs and add search time for no gain. |
| 2 | Every knob has its current default, range, measurement and quality gate in one table. | The 11-18 list had names only. Several defaults moved (CUDA prefill chunk 1024/2048, decode v2 from 512 rows, TQ staged prefill). |
| 3 | Two kinds of quality gate. Lossy knobs (KV format) keep James's KL budget. Knobs that should not change quality get an **equivalence gate** whose bound comes from a same-run 1-ULP control, K = 2. | House style since #1928, #1919, #1924 and #1953. A fixed tolerance either saturates or cannot fail on models whose Q8 activations digitise ULP drift. |
| 4 | KL is scored on **teacher-forced decode steps at depth**, not on prefill windows. | Quantised KV, split-KV decode and the step graph only run on the decode path. A prefill-window perplexity cannot see them. `tools/tq-gate` (`decodedump`, `klcmp`) is the instrument #1926 and #1953 used. |
| 5 | The f16 "run-to-run floor" is redefined as the same-run control band, and the default budget needs a new ruling. | The engine is deterministic, so a run-to-run floor is 0. The measured f16 band is about 4e-7 (PRECISION.md E21); "2x the floor" would refuse every quantised format. Ruled: Decisions 1. |
| 6 | The key gains: a scope (one device, or the whole machine for `--device auto`), kernel assembly MVIDs, driver id and API version, an environment-override hash, a model-modifier digest, the tuned context, and schema and protocol versions. | The 11-18 key could not tell two builds apart when only a C# threshold changed (for example `DOTLLM_CUDA_DENSE_DECODE_V2_MIN_KV`), ignored active A/B variables, and had no file name for a result that chose between devices. |
| 7 | Measurement folds in #1936: weights-per-pass against the LLC, timing at the longest history, crash-safe probes cached per driver, and a fraction-of-floor column. | Accepted from the magnitude review. |
| 8 | New section on memory admission: candidates are sized before load, spill is detected and disqualifies, the result stores peak bytes, and `--auto` re-admits against free memory when it applies. | #1933 and the no-spill rule. Free memory at serve time is not free memory at tune time (the UMA limit drifts; other processes hold VRAM). |
| 9 | TurboQuant candidates are `tq8`, `tq4`, `tq3`, `tq2` in place; the refusal list is quoted from docs/CLI.md. | #1929 landed the in-place cache on CPU, CUDA and Vulkan. TQ still needs `--no-paged`, the same format on K and V, no MTP head and no draft model (#1893, #1895, #1901). |
| 10 | Tuned settings describe plain serving. A tap, a residual modifier or a mid-session constraint changes the path and is outside the measured result. | #1931/#1932: the CUDA step graph and the fused decode paths are bypassed while a tap or modifier is attached. |
| 11 | Model hashing reuses `TensorOverlay.BaseHash`. | It already caches SHA-256 by (path, length, mtime) under `~/.dotllm/hash-cache`. A second cache would be a duplicate. |

Nothing James decided on 11-18 was changed. Where a decision can no longer work as written, it is kept; the conflicts were ruled on
2026-11-24 (see "Decisions").

## Intent

A user should get the best settings for *their* model on *their* machine without knowing the knobs. Quantised KV must stay within a
quality limit they can state. The result is measured once, saved under a checksummed key of model, device and build, and reused until
something in that key changes or the user asks for a re-measure.

Why: dotLLM aims to be the best engine for every device x model cell. Reaching that still takes hand-tuning of KV format, prefill
chunk, placement, MTP, concurrency and threads. The defaults are measured on two reference devices (RTX 3060, Strix Halo gfx1151), and
a user on other hardware can't tell what their box would gain.

## Goals

1. **`dotllm tune <model>` runs the search and saves the winner (James: explicit command only).**
   - It prints the plan before starting: the placements to load, the sweeps, the memory each placement needs against what is free, an
     estimated time and a hard time cap.
   - It writes a human-readable report alongside the saved file.
2. **`--auto` on `serve`, `run`, `chat` and `bench` only reuses saved results.**
   - On a miss it starts with today's defaults and prints the exact `dotllm tune ...` command to run.
   - It never starts a long search on its own.
3. **Workload profiles define "best" (James).** A profile fixes three things:
   - a target context;
   - weights over decode tok/s, prefill tok/s and time to first token (TTFT);
   - a memory headroom.

   Built-in profiles:
   - `chat` (default; 8K, decode-heavy);
   - `agent` (32K, prefill/TTFT-heavy, prefix reuse on);
   - `long`;
   - `batch`.

   Profiles are JSON, and users can add their own. **(call: defaults and numbers are starting values, to be tuned with data.)**
   - **`long` (James):**
     - The score is a weighted blend of usable context and speed, measured at the context each candidate reaches.
     - `--min-context N` gives an optional floor: candidates that cannot reach N are dropped.
     - Context and speed weights live in the profile and can be overridden.
   - **`batch` (James):**
     - By default it sweeps concurrency (1, 4, 8, 16, bounded by memory) and shows a table of total throughput per concurrency.
     - It recommends the setting with the best total and lets the user choose. Non-interactive runs (`--json`, no TTY) take the
       recommendation.
     - `--concurrency N` tunes for exactly N.
     - The chosen concurrency is saved as the server's admission/batch hint.
   - **`--single-user` (James):** optimise for one active session. Concurrency is fixed at 1, the score leans on decode tok/s and TTFT,
     and the scheduler and batch settings that only help concurrency are excluded. It works with any profile.
4. **KV quality gate (James):**
   - Measure mean KL against f16 KV at the profile's context, on the quality datasets.
   - **Datasets (James):** the default is a short fixed slice of the wikitext fixture. `--quality-data <hf-dataset-id[:split]|path>...`
     adds or replaces them: Hugging Face datasets fetched into the default HF cache, or local text/JSONL files.
     - More data buys a more trustworthy gate at the cost of time.
     - The plan printout counts the tokens per dataset and the time.
     - The report gives KL per dataset and the worst case. A format passes only if it is within budget on every dataset.
   - Default budget **(James, 11-18)**: 2x the model's f16 run-to-run KL floor (measured once and saved), or 0.01 if the floor can't be
     measured. This needs a new ruling; ruled in Decisions 1 (max(2 x band, 0.01)).
   - `--kv-quality <budget>` overrides the default.
   - A format over budget is never chosen. It still appears in the report with its KL and speed.
5. **Only settings that really run are eligible.** `TurboQuantKvSupport.RefusalReason`, and every other "is this real here?" check,
   filters candidates before measuring. That way `tune` can never pick a silently substituted setting, and as gaps close (#1887 and its
   tickets) the search space grows on its own.
6. **Provenance.** Explicit flags always win over saved settings.
   - The startup log names the tuning file and its date.
   - `/v1/status` reports each setting's source: `flag`, `auto` or `default`.

## Backends

CPU, CUDA and Vulkan land together (the parity rule; deferring one needs the owner's OK). CUDA is validated on framework/T5500, with GPU
use coordinated through the owner.

## Non-goals

- Running a search automatically at serve start or in the background (James chose explicit `tune`).
- Tuning weight quantisation (the model file is the user's choice).
- Tuning kernel internals that have no documented safe switch.
- Re-deciding a precision or kernel default that has a recorded ruling (tier C below).
- Tuning the sampler, constraints, taps, residual modifiers or LoRA selection. These are per-request choices.
- Cross-machine sharing of results, beyond copying the file.

## Anti-goals (failure even if it "works")

- A saved result that silently applies to a different model file, device, driver or build.
- A winner chosen from cold-clock, contended, cache-warm or single-shot numbers (see Measurement).
- A search that spills device memory, or takes the GPU without the lock.
- A saved placement applied at serve time without re-checking that it still fits.
- `--auto` that hides which settings it applied.
- A quality gate with a fixed tolerance, or one whose control arm could not have moved.

## Approaches

- **C. Process per placement, in-process sweeps inside it. CHOSEN (James).**
  - `tune` (the orchestrator) launches one worker process per placement candidate (device, layer split, MoE offload), so each gets
    clean device memory and real cold-start behaviour.
  - The worker loads the weights once, then sweeps the settings that are swappable on a loaded model: KV format, prefill chunk, MTP
    on/off, scheduler mode and batch size, threads and decode threads.
  - It returns JSON results over stdout.
  - Cost: one weight load per placement.
- **A. All in one process.** The fastest option. Rejected: device memory and allocator state left over from earlier placements skew
  later measurements, and we have already seen about 40% run-to-run swing on UMA.
- **B. One process per setting.** Fully isolated, but it reloads the weights for every candidate, which means hours for a big model.
  Rejected.

The process boundary of approach C is also what makes a crashing candidate survivable (see "Crash-safe probes").

## Knob inventory

Defaults below were read from `docs/CLI.md` and the source on `dev` at 6c67922e. "Swappable" means the worker can change it on a loaded
model; anything else costs a placement (a new worker).

### Tier A: tuned by `dotllm tune`

| Knob (flag) | Default today | Range searched | How it is measured | Gate |
|---|---|---|---|---|
| Device (`--device`) | `auto`: CUDA if the model fits, else Vulkan, else CPU | every servable device that passes admission for the profile | one worker per placement; the profile score | equivalence |
| GPU layers (`--gpu-layers`, CUDA only; ignored on Vulkan) | unset = all layers; `auto` = most leading layers that fit | the auto pick and one neighbour each way | placement | equivalence |
| MoE offload (`--moe-offload`, `--n-cpu-moe`, `--cpu-moe`, `--moe-cache-mib`) | `auto`: experts in VRAM when the model fits, else host store plus a VRAM cache sized to the measured budget | auto; all-CPU experts; cache at 50% and 100% of the budget left after KV for the profile context | placement; decode timed on natural text, because a tiled prompt concentrates routing on a few experts (`DOTLLM_BENCH_SEED_FILE`, #1660) | equivalence |
| KV format (`--cache-type-k/-v`) | GPU: f16. CPU paged: q8_0. | f16, q8_0, q4_0, k8v4, k4v8, then `tq8`, `tq4`, `tq3`, `tq2` (K and V the same) where `RefusalReason` returns null | swappable; decode tok/s at the profile context, KV bytes per token | **KL budget** |
| Paged or contiguous KV (`--no-paged`) | `serve`: paged. `run`, `chat`: contiguous. | coupled to KV format: TurboQuant and the dense-CUDA chunk default both need contiguous | swappable | none (same K/V values); see Decisions 2 |
| Prefill chunk (`--prefill-chunk-size`) | `run`, `chat`, `bench`: 0 (one pass). `serve`: 1024 on dense CUDA with `--no-paged` and f16 KV, 2048 with q8_0/q4_0, one pass elsewhere. | 0, 256, 512, 1024, 2048, 4096, bounded by the prompt and by scratch admission | swappable; the in-process chunk rotation `BenchRunner` already has (`DOTLLM_BENCH_ALT_PREFILL_CHUNK`, #1660); prefill tok/s and TTFT at the profile context | equivalence |
| MTP (`--mtp` / `--no-mtp`), `--speculative-k` | on for `serve` when the GGUF carries a head, no draft model and expected concurrency below 5; K = 3 | off; on with K in 2, 3, 4 | swappable; decode tok/s **and acceptance rate** on natural text (acceptance on a tiled prompt is meaningless); skipped when a TQ format is the incumbent (#1901) | equivalence |
| CPU threads (`--threads`, `--decode-threads`, `--moe-cpu-threads`) | 0 = all cores; decode 0 = capped at the memory-channel count (#982); MoE = physical cores - 1 | physical cores, performance cores, logical cores; decode at 1x and 2x the channel count | swappable; CPU placements and the CPU tail of a split only | equivalence |
| Expected concurrency (`--expected-concurrency`) | 0 (serial path). At 5 or more, Vulkan hybrid models take the continuous-batch scheduler and default-on MTP is dropped. | `batch` only: 1, 4, 8, 16, bounded by KV memory for N sequences at the profile context | swappable; aggregate tok/s with N concurrent sessions on distinct prompts | none (a throughput choice) |
| Context (`--ctx-size`) | the checkpoint window; CUDA hybrid models fit it to free VRAM | `long` only: the largest context each candidate admits, down to `--min-context` | each candidate is timed at the context it reaches | none |

Search order and pruning **(call)**:
- KV formats are tried widest first within a family (q8_0, then q4_0; `tq8`, `tq4`, `tq3`, `tq2`). When a format fails the KL budget,
  narrower formats of the same family are skipped and reported as `skipped: wider format failed`.
- A knob with one eligible value is recorded as `fixed` with the reason, and costs no time.

### Tier B: device crossovers (opt-in, last slice)

These are integer thresholds and grid sizes whose defaults were measured on one device and baked in. They are the only kernel-level
settings a different GPU could plausibly want changed. They change no storage format, so no precision ruling is touched.

| Knob | Default today | Where the default came from | Range |
|---|---|---|---|
| CUDA dense decode v2 row threshold (`DOTLLM_CUDA_DENSE_DECODE_V2_MIN_KV`) | 512 | RTX 3060, #1939 | 128 to 2048 by powers of two |
| CUDA flash prefill crossover (`DOTLLM_CUDA_FLASH_ATTN_MINSEQ`) | measured default | RTX 3060 | by powers of two |
| CUDA split occupancy (`DOTLLM_CUDA_SPLIT_TARGET_BLOCKS`, `DOTLLM_CUDA_SPLIT_MIN_KV`) | derived from an RTX 3060 sweep on Bonsai-27B | #219 | 0.5x to 2x |
| Vulkan v2 split-KV grid (`DOTLLM_VULKAN_SPLITKV_GQA2_TARGET_WG`) | AMD: 2 x compute units, clamped to 256. Other vendors: 256. | Strix gfx1151 (#1956). **Not measured on any non-AMD Vulkan device** (#1926 bench plan B5). | 1x, 2x, 4x the CU or SM count, and 256 |
| Vulkan split floor (`DOTLLM_VULKAN_SPLIT_MIN_KV`; the v2 pass uses a constant 64) | 64 | Flint on Strix | 32, 64, 128 |
| Vulkan decode submit split (`DOTLLM_VULKAN_DECODE_SPLIT_LAYER`) | 8 | Strix | 0, 4, 8, 16 |
| Vulkan grouped MoE prefill threshold (`DOTLLM_VK_MOE_GROUPED_MIN_TOKENS`) | measured default | Strix | 0.5x to 2x |
| Vulkan scheduler crossover (`DOTLLM_VK_SCHEDULER`) | on at expected concurrency 5 or more | Tev1-4B on Strix | covered by the `batch` sweep |

Rules for tier B **(call)**:
- Off unless `tune --kernel-crossovers` is given, and never part of the default time estimate.
- A crossover is eligible only when it has a typed, in-process setter. Today these are read once from the environment into statics.
  The tier-B slice adds a `KernelCrossovers` record passed at model creation; the environment variable stays as the override and wins.
- Each change reassociates a reduction, so it takes the equivalence gate.
- A crossover measured on a reference device (RTX 3060 for CUDA, gfx1151 for Vulkan) is expected to come back `inert` or unchanged. A
  win there is a finding to report upstream as a new default, not just a per-machine setting.
- Tier B is built, as the last slice (Decisions 3).

### Tier C: settled, never touched by `tune`

Each has a recorded ruling or is bit-identical to its alternative. Their switches exist for A/B and bisection. `tune` records which of
them the environment overrides (see the key) and otherwise leaves them alone.

| Setting | State on dev | Ruling or evidence |
|---|---|---|
| CUDA split-KV decode v2 and the step graph (`DOTLLM_CUDA_DENSE_DECODE_V2`, `DOTLLM_CUDA_STEP_GRAPH`) | default on; kill switches | James 2026-11-21, #1939 |
| CUDA F16 flash prefill, staged q8/q4 loaders (`DOTLLM_CUDA_F16_FLASH_PREFILL`, `DOTLLM_CUDA_KVQ_FLASH_PREFILL`) | default on | #1938, relative KL gate accepted 2026-11-21 |
| CUDA E4 tile pipeline and its `_legacy` twins | chosen per launch by occupancy | #1954, bit-identical |
| F16 P operand scaled by 2^15 (CUDA, G3, Vulkan QSA and coopmat flash) | always | #1960 (#1953) |
| CUDA TurboQuant prefill entry (`DOTLLM_CUDA_TQ_PREFILL`) | staged at tq2/3/4/8 | James 2026-11-22 and 11-23, #1942 |
| TurboQuant in-place KV, F16 norms | the only TQ cache | #1929, PRECISION.md E17 |
| Vulkan v2 grouped split-KV: GROUP = 4 specialisation, F16 partials (`DOTLLM_VULKAN_SPLITKV_GROUP_SPEC`, `DOTLLM_VULKAN_SPLITKV_PART`, `DOTLLM_VULKAN_SPLITKV_GQA2`) | default on | James 2026-11-21, #1956, E21 |
| Vulkan split-KV decode and flash prefill on/off (`DOTLLM_VULKAN_DISABLE_SPLIT_DECODE`, `DOTLLM_VULKAN_DISABLE_FLASH_ATTENTION`) | on | #345/#346 |
| CPU split-KV decode, F16 partials (`DOTLLM_CPU_ATTN_SPLITKV`) | f16 | James 2026-11-21, #1940, E18 |
| CPU dense tile pack (`DOTLLM_CPU_ATTN_PACK`) | on | #1940, bit-identical |
| CPU Q4_K row-lane GEMM (`DOTLLM_CPU_Q4K_ROWLANE`) | on, batch-invariant | James 2026-11-22 and 11-23, #1940, E19 |
| CUDA graph capture for hybrid and MoE decode (`DOTLLM_CUDA_HYBRID_GRAPH`, `DOTLLM_CUDA_MOE_GRAPH`) | on | #1210 |
| Opt-in approximations (`DOTLLM_ATTN_SPLIT_KV`, `DOTLLM_ATTN_GQA_SPLIT`, `DOTLLM_ATTN_SPLIT_KV_HP`, `DOTLLM_GDN_SCAN_APPROX_SPLIT4`, the Int-QK experiments) | off | default-on only after James sees data |
| Crash-prone opt-ins (`DOTLLM_VULKAN_ENABLE_IQ2XXS_MMQ`) | off; AMD driver miscompile on gfx1151 | first user of the crash-safe probe cache, but still not enabled by `tune` without tier B |
| MoE cold-expert policy (`--moe-cold auto`) | per expert, from a cost model calibrated at load | it has its own calibration record; `tune` reuses it |
| Prompt and prefix cache policy, snapshots | `agent` policy since #1778 | a workload choice; the `agent` profile turns prefix reuse on and does not sweep it |

What `tune` assumes about sessions: results describe plain decode and prefill. An attached tap, residual modifier or per-request LoRA
takes the eager path (the step graph and fused decode are bypassed), so tuned figures do not describe those requests. The settings
still apply and are still correct. The report says so in one line.

## Quality gates

Every gate follows the house rule from #1928: **a bound is derived from a sensitivity control on the same instrument in the same run,
never from a fixed number, and a control that cannot move is rejected.**

### The instrument

- **Rows scored:** prefill to the profile's context, then teacher-forced single-token decode steps, keeping full log-softmax rows for
  the last 256 steps. This is the `tools/tq-gate` `decodedump` / `klcmp` protocol, with exact token ids fixed up front.
  - Why decode steps: quantised KV, split-KV decode and the step graph run only on the decode path, and a prefill-window perplexity
    cannot see them.
  - Why at depth: KV quantisation error and softmax-tail effects grow with context (#1953 appeared only at 32K).
- **Reference arm R:** the placement's defaults with f16 K and V, same token ids, same run.
- **Control arms C+ and C-:** R with one layer's residual moved by +1 and -1 ULP through the residual-modifier seam that exists on
  every backend (#1931). The reference for the control is R with a 0-ULP modifier attached, so both sides take the same path (the
  path-matched control from #1919).
  - The **band** is the larger of KL(R0 || C+) and KL(R0 || C-), with the matching top-1 disagreement count.
  - A control whose rows are bit-identical to R0 is a failed instrument: the gate reports `control inert` and the knob is not changed.
- **Shared evaluator:** #1903 factors the evaluator out of `PerplexityCommand` and gives it the decode-stepped mode and the missing KV
  formats (f16, mixed pairs, `tq*`) on CPU, CUDA and Vulkan. It scores in-process on the model the worker already loaded.
- **Cost:** 256 rows x vocabulary x 4 bytes of host RAM per arm per dataset (about 155 MB at a 151K vocabulary). This is a reference
  comparison, so the FP32 rule's standing exception applies. It is counted in the plan printout and in host-memory admission.

### Equivalence gate (knobs that should not change quality)

Applies to device, GPU layers, MoE offload, prefill chunk, MTP, threads and every tier-B crossover.

- Pass when KL(R || candidate) <= K x band with **K = 2**, and top-1 disagreements <= K x the control's count.
- When the band is exactly 0, the candidate must be bit-identical.
- A candidate that fails is not chosen, and is reported as `changes output beyond the 1-ULP band` with its KL. That is a finding worth
  a bug report, since these knobs are meant to be numerically neutral.
- MTP on GPU and CPU spec/MTP greedy can differ from plain greedy past 2K keys (accepted, James 2026-11-22, SPECULATIVE.md). MTP is
  therefore gated on teacher-forced rows, not on greedy text identity.

### KL budget gate (KV format)

- Mean KL(R || candidate) over the scored rows, per dataset. A format passes only if it is within budget on **every** dataset (James).
- The band is reported next to each format's KL, so a reader can tell format damage from digitisation noise. On models whose Q8
  activations digitise ULP drift the band itself is about 1e-2 (qwen4exp, Flint; PRECISION.md E14, E21).
- **Default budget.** James's 11-18 rule is "2x the f16 run-to-run floor, or 0.01 if the floor can't be measured". Two facts make it
  ruled in Decisions 1:
  - The engine is deterministic, so a literal run-to-run floor is 0. The measurable form of the floor is the control band.
  - On a model without digitisation the f16 band is about 4e-7 (Qwen3-4B Q4_K_M, E21). 2x that refuses q8_0 as well as q4_0. On the
    same model q4_0 KV measures 0.07 to 0.11 at 7.6K and 32K.
  - **Ruled (James 2026-11-24):** default budget = max(2 x band, 0.01). The band term keeps the gate meaningful on digitising
    models; 0.01 is James's own fallback number.
- `--kv-quality <budget>` overrides the default (James). The budget is part of the key.
- The band is measured once per placement and saved with the result.

### What has no gate

Paged against contiguous KV, expected concurrency and context change scheduling and capacity, not K/V values. They are covered by the
existing parity suites and get a greedy-identity smoke check in the worker, reported but not used as a bound.

## Measurement protocol

These are already-learned lessons plus #1936; each is a requirement.

### Exclusivity and contention

- **GPU exclusivity (call):** `tune` takes a built-in cross-process lock for the whole run (a named mutex on Windows, `flock` on a lock
  file under `~/.dotllm/locks/` elsewhere) and releases it in a `finally`.
  - ResourceLock is a dev-box agent coordinator and is not a product dependency. Agents on the dev box still take ResourceLock
    themselves, and confirm it, before launching `tune`.
- **Contention check (call):** before and after each placement, sample GPU utilisation and other processes' device memory
  (`VulkanDevice.ReadOtherProcessPressure`, NVML free memory, CPU load on the cores in use). Above a threshold, abort the placement
  and report it, rather than recording bad numbers. It runs on every OS.
- The power source (mains or battery) and power plan are recorded in the report. On battery `tune` warns before starting.

### Timing

- **Clock-ramp warm-up** before every timed configuration: cold-vs-warm alone gives 2-3x phantom deltas.
- **Repeats:** at least 3 per configuration; keep the median and range. "Better" means the median gain exceeds the larger of the two
  ranges. Report ranges, never single numbers.
- **Interleave** A and B within one session (on every device, not only UMA). Never compare numbers across sessions or across workers.
  Placements are compared through each worker's own same-session baseline.
- **Longest history (#1936):** every history-dependent figure is timed at the profile's target context, never extrapolated from a
  shallow run. Decode is also timed at 512 so the report shows the slope. `bench --depth-list` already measures several depths without
  a reload.
- **Cache state (#1936):** a timed pass must not find its weights in cache.
  - The worker computes weight bytes read per pass (for MoE, the routed experts only) and the device's last-level cache size (CPU LLC;
    on a GPU, L2).
  - When bytes per pass are at least 4x that cache, the real model satisfies the rule by construction, and the report states the ratio.
  - Below 4x (tiny models, or a tier-B kernel timed on one layer), timed passes rotate through weight copies totalling more than the
    cache. The copies are admitted like any other allocation. If they cannot be admitted, the figures are still reported but marked
    `cache-warm` and cannot win.
- **Natural text for data-dependent paths:** MoE routing and MTP acceptance are timed on the quality corpus, not a tiled prompt.
- **Prove the fast path ran.** When a setting is meant to switch kernels, the worker records which path ran, from the existing launch
  counters. A setting whose path didn't change is reported as `inert`, not as a win.

### Fraction of the bandwidth floor (#1936)

- Each decode cell in the report carries `floor_fraction = floor / measured`, where
  `floor = weight bytes read per token / stream rate + one launch`.
- The stream rate is measured on that device, at the thread count in use on CPU, once per device and driver, and cached beside the
  probe cache.
- It is a report column and a stop rule, not a score **(call)**: when the incumbent is at 0.9 of the floor or better, the remaining
  decode-only sweeps for that placement are skipped and reported as `at the bandwidth floor`.
- The same column goes into the perf-gap tables (the doc and script part of #1936, which needs no GPU run).

### Crash-safe probes (#1936)

- The worker writes a `starting <candidate>` line before each candidate. If the worker dies (access violation, `DEVICE_LOST`, a hang
  past the candidate's own timeout), the orchestrator knows which candidate did it.
- That candidate is stored as `unanswerable` in `~/.dotllm/tuning/probes/<device-key>.json`, keyed by device identity, driver version
  and kernel-set hash, with the exit code and the last log lines.
- An unanswerable candidate is not retried until the driver or the kernel set changes, or the user passes `--recheck`. The
  orchestrator starts a fresh worker for the same placement and continues with the remaining candidates.
- The first intended user is the IQ2_XXS MMQ kernel (`DOTLLM_VULKAN_ENABLE_IQ2XXS_MMQ`), in tier B. The mechanism lands with the
  worker, because any candidate can take a driver down.

### Time cap (call)

- The estimate is printed first. An interactive run that will exceed the cap asks before starting. Non-interactive runs (`--json`,
  no TTY) never prompt: the cap is enforced.
- At the cap, the best result so far is saved and marked `partial`, listing what was not measured. `--auto` applies a partial result
  with a warning that names the `tune` command to finish.

## Device memory admission and the no-spill rule

CLAUDE.md: an allocation is admitted against free device memory or refused; it never spills. #1933 will route every device allocation
through one `DeviceLedger` whose charges are the planner's own size functions. It is open, and #1857 lists CUDA allocations that are
still unadmitted. The design works before and after it lands.

1. **Before load, size and drop.** Each placement candidate is sized with the existing planners (`DeviceSelector.Plan`, the
   `--gpu-layers auto` fit, the MoE budget, `CudaAdmission`) for: weights, KV at the profile's context in the *widest* format the
   sweep will try, forward scratch at the *largest* chunk the sweep will try, split-KV partials, N sequences for `batch`, plus the
   profile's headroom. A candidate that does not fit is dropped and reported with required and free bytes. It is never loaded.
2. **Inside the worker, every swap is admitted.** Changing KV format, chunk or concurrency goes through the same admission as a normal
   load. A refusal (`DeviceMemoryAdmissionException`) removes that value from the sweep; it is not an error.
3. **Spill is detected, not assumed absent.** Until #1933 covers every site, the worker samples for spill after load and after each
   timed configuration: Windows shared-GPU usage (the `GpuSpillMonitor` PDH counter), the Vulkan memory-budget heap usage against its
   budget, and NVML used against total. A configuration seen spilling is disqualified and reported; its timings are discarded.
4. **UMA.** On Strix and Arc, device and host memory are one pool and the reported limit drifts with other consumers. Headroom is
   taken from the pool, host buffers of the tuner (reference log-prob rows, corpus) are counted against it, and the device budget is
   never part of the key.
5. **The result stores what it needed.** `peak_device_bytes` and `peak_host_bytes` per chosen setting, and the context they were
   measured at. After #1933 these come from the ledger's committed peak; before it, from the planners' figures and the samples in 3.
6. **`--auto` re-admits at apply time.** A saved placement is applied only if it fits free memory *now* at the context being served.
   If it does not, `--auto` falls back to the defaults' own fit logic and warns with required and free bytes. It does not shrink the
   tuned settings on its own, and it does not spill.
7. **Serving a larger context than was tuned** is allowed after the same admission check, with a warning that decode figures were
   not measured at that depth.
8. **No expansion.** The tuner adds no stored FP32 on a hot path and no decoded KV history. Its only FP32 buffers are the reference
   rows of the quality gate (the standing exception), on the host.
9. **Disk to device.** Each worker loads weights through the product's normal path. The tuner adds no bounce buffer of its own.

## Search (call)

1. **Placement candidates** come from the existing planners. Keep the auto pick, plus at most two neighbours (one layer split each
   way, iGPU against CPU tail, MoE experts on CPU). Each must pass step 1 of admission.
2. **Inside a worker, in this order:**
   1. Baseline at the defaults; the control band.
   2. KV formats: quality-gate each, then time the survivors at the target context.
   3. **Coordinate descent** over prefill chunk, MTP, threads, and (for `batch` and `agent`) concurrency. Start from the defaults,
      change one setting at a time, and keep a change only when it beats the incumbent by more than the measured noise.
3. **Score:** the profile-weighted geometric mean of each measured speed relative to the defaults' speed in the same session, so units
   cancel and a same-session baseline absorbs drift.

## Saved result

### Path

`~/.dotllm/tuning/<model-hash>/<scope>-<profile>.json` (`DOTLLM_TUNING_DIR` overrides the root).

- `<model-hash>`:
  - a GGUF file: its SHA-256, through `TensorOverlay.BaseHash.Compute`, which caches by (path, length, mtime) under
    `~/.dotllm/hash-cache`;
  - a file in the HF hub cache: `hf:<blob id>`, read from the blob name without hashing;
  - **a safetensors directory (in scope, call):** a SHA-256 over the sorted (name, size, SHA-256) list of its weight, config and
    tokenizer files.
- `<scope>` **(call)**: the device key for an explicit `--device`, or `auto-<machine key>` for `--device auto`.
  - An explicit-device result is keyed on that device's fingerprint.
  - An `auto` result is keyed on the fingerprints of every enumerated device, and its settings name the winning device. `--auto` then
    loads that device as if it had been given explicitly. If it can no longer load or be admitted, the defaults run, with a warning.
  - Why: under `--device auto` the device is only known after the load falls through CUDA, Vulkan, CPU. A per-device file name cannot
    be computed before the load, and the tuned placement is what should drive the load.

### Contents

Chosen settings; the measured score with ranges per setting; the runner-ups; KL and band per KV format; peak device and host bytes; the
floor fractions; the inert and unanswerable settings; the cache-state ratio; the time taken; `partial` and what was not measured.

### The key

| Field | Contents | On mismatch |
|---|---|---|
| schema version | the file layout | miss |
| model hash | as above | miss |
| model modifiers | digest of the tensor overlay, load-time ternary adapters and a separate MTP head or DFlash drafter file, using the load-time digests chunk identity already computes (#1951). Per-request LoRA is not in the key. | miss |
| device identity | backend; GPU name; PCI vendor and device id; total device memory (dedicated devices only); CPU model; physical and logical cores; RAM; for `auto` scope, every device | miss |
| profile | name and hash of its JSON | miss |
| KV quality budget | the value in force | miss |
| tuned context | the profile's target context, or the context reached for `long` | informational; see admission 7 |
| **driver identity** | Vulkan: `driverVersion`, `driverID` and `driverInfo` (`VkPhysicalDeviceDriverProperties`), API version. CUDA: the NVML driver version and `cuDriverGetVersion`. OS build. | stale |
| **dotLLM perf identity (call)** | release version; a hash over the shipped `.spv`, `.ptx` and native libraries **as loaded from the output directory**; the MVIDs of the kernel-bearing managed assemblies (`DotLLM.Cpu`, `DotLLM.Cuda`, `DotLLM.Vulkan`, `DotLLM.Engine`, `DotLLM.Models`); the .NET runtime version | stale |
| environment overrides | a hash of the sorted `name=value` list of every set `DOTLLM_*` tuning and A/B variable (the "Tuning, A/B and diagnostic variables" tables of docs/CLI.md) | stale; the warning names the variables that differ |
| tuner protocol version | bumped when the measurement or gate rules change | stale |

Notes:
- The perf identity is not the commit, so a dev rebuild that touches no kernel does not go stale. MVIDs are content-derived under
  deterministic builds, so they move only when the assembly's code does. They are needed because thresholds and dispatch live in C#
  (for example the 512-row decode v2 threshold), and CPU kernels live in `DotLLM.Cpu.dll`.
- The `.spv` hash reads the copies in the output directory, because those are what load.
- The .NET runtime is in the key because it changes CPU prefill by about 2x.

### Checksum and matching

- **Checksum:** SHA-256 over the canonical JSON of key and result, stored in the file.
- **Hard miss** (any `miss` field differs; the checksum fails; the file is corrupt or absent): ignored with a warning.
- **Stale** (only `stale` fields differ): `--auto` still applies it, warns with old and new values, and prints the
  `dotllm tune --recheck` command. **(call: stale settings beat slower defaults.)**
- **Partial:** applied, with a warning and the command to finish.
- In every applied case the settings still pass apply-time admission and `RefusalReason` for the current build. A setting the current
  build refuses is dropped with a warning; it is never substituted.
- `tune --recheck` re-measures even on a match, retries unanswerable candidates, and overwrites the file.
- **DTOs:** follow the CLAUDE.md JSON rules. Every optional member is nullable with no initializer, and the new JSON context gets the
  `JsonContextDefaultsTests`-style guard.

## Surfaces

- `dotllm tune <model> [--profile p] [--kv-quality x] [--quality-data ...] [--device d] [--min-context n] [--concurrency n]
  [--single-user] [--time-cap 20m] [--recheck] [--kernel-crossovers] [--json]`
- `--auto[=profile]` on serve, run, chat and bench.
- The tray gets an "auto-tuned" toggle later. **(call: v1 is CLI and server.)**
- The ollama `/api` surface passes `--auto` through like other options.
- `/v1/status` (and so `/props` `activity`) adds a `settings_source` map.
- `docs/CLI.md` gains the options (generated tables, guarded by `CliHelpTests`), `DOTLLM_TUNING_DIR`, and an "Auto-tuned settings"
  section.

## Prerequisites and related tickets

- #1903: shared evaluator with f16, mixed and `tq*` KV, plus the decode-stepped mode. Needed by the KV gate.
- #1904: safetensors models score on CPU only.
- #1925: `bench --kv-quant-k/-v` is ignored on Vulkan and absent on plain CUDA and CPU bench. The worker's KV sweep needs it fixed or
  must not go through that path.
- #1933 and #1857: the ledger and the remaining unadmitted allocations. Not blocking; they retire the spill sampling.
- #1936: the measurement rules, folded in above.
- #1893, #1894, #1895, #1901: TurboQuant gaps. As each closes, `RefusalReason` stops refusing and the search grows on its own.

## Tests

The CLAUDE.md "Prove the Test Can Fail" rule applies to each of these.

- **Key matching:** each key field changes in turn, and each must give a hard miss or stale exactly as the table says. Mutants: drop
  the driver from the key; skip the checksum check; drop the environment-override hash.
- **Scope:** an `auto` result is not found for an explicit device and the reverse. Mutant: ignore the scope.
- **Precedence:** an explicit flag beats the saved setting, and `/v1/status` reports the source. Mutant: apply saved settings after
  the flags.
- **Apply-time admission:** a saved placement that no longer fits a fake budget falls back to defaults with the bytes in the warning.
  Mutant: skip the re-check.
- **KL budget:** a format over budget is never chosen. Substitution: lower the budget to just under a format's measured KL and the
  winner changes.
- **Equivalence gate:** a synthetic arm at 3x the band is refused and one at 1x passes; an inert control refuses the change. Mutant:
  compare against a fixed tolerance.
- **Eligibility:** a refused setting is never measured or chosen. Mutant: bypass `RefusalReason`.
- **Noise rule:** a synthetic measurer with overlapping ranges keeps the incumbent; one with separated ranges switches.
- **Spill:** a configuration whose fake spill sample is positive cannot win. Mutant: ignore the sample.
- **Crash-safe probe:** a worker that exits on candidate X leaves X `unanswerable`, the run completes, and a second run skips X until
  the driver field changes. Mutant: key the probe cache without the driver.
- **Cache state:** below the 4x ratio without rotation a figure is marked `cache-warm` and cannot win.
- **Profiles:**
  - `--min-context` drops candidates below the floor.
  - `--single-user` excludes concurrency-only settings.
  - `batch`'s non-interactive run takes the recommended concurrency.
  - `--concurrency N` is honoured.
  - A partial result is saved, marked, and applied with a warning.
- **Datasets:** a format within budget on one dataset but over it on another is rejected. Mutant: gate on the mean across datasets
  instead of the worst case.
- **End to end:** `tune` on a tiny synthetic model (the `TinyQwen` fixture) on the CPU writes a valid file, and `--auto` reuses it.
  A GPU variant runs under the lock.

## Slices

1. Store, key and `--auto` reuse, with CLI help and docs (the slice-1 plan).
2. `dotllm tune` orchestrator and worker: placements, admission and spill sampling, the lock and contention check, the timing
   protocol, crash-safe probes, the equivalence gate, coordinate descent, the time cap, the report with floor fractions; the `chat`
   profile.
3. KV quality gate: the shared evaluator (#1903), the band, the budget, `--quality-data`, safetensors on GPU (#1904).
4. Profiles: `agent`, `long`, `batch`, `--single-user`, user profile files, the concurrency hint.
5. Tray toggle and the ollama `/api` pass-through.
6. Tier-B device crossovers (Decisions 3: yes).

## Decisions (James 2026-11-24)

1. **Default KV budget: max(2 x band, 0.01) (James).** The same-run control band replaces the run-to-run floor, and 0.01 KL is the floor of the budget. `--kv-quality` tightens or loosens it.
2. **`--no-paged` for `serve`: only under `--single-user` or for `run`/`chat` (James).** For multi-session serve profiles, TurboQuant stays ineligible. Tracked: once #1893 lands TurboQuant on the paged pools, lift this restriction (comment on #1893, 2026-11-24).
3. **Tier B: build it, as the last slice (James).**
4. **A changed tuned variable is `stale`, not a miss (James).** `--auto` applies the saved result with a warning that names the mismatch and the `dotllm tune --recheck` command.

## Left to the builder

- Contention thresholds.
- The interactive choice UI for `batch`.
- The exact HF dataset fetch (via `huggingface_hub`-compatible cache paths, never ad-hoc folders).
- The worker IPC framing (JSON lines over stdout).
- Exact profile numbers, which get revised with data.
- Neighbour selection for placements.
- Report formatting.
- How the stream rate is measured on each backend.
- How to read the last-level cache size on each OS, and the GPU L2 size (a per-vendor table is acceptable).
- How to read the driver version on each backend. Vulkan has `driverVersion` in `VulkanStructs.cs` but does not expose it on the
  device description. CUDA has neither `cuDriverGetVersion` nor an NVML driver-version binding yet. `DeviceInfoDto` needs the fields.
