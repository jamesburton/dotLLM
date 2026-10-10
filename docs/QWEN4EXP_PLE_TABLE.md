# Qwen4-Exp n-gram (PLE) table: prefetch / offload service (#822)

The released `qwen4exp` file carries one `per_layer_token_embd.weight` tensor, `[160, 320001536]` IQ4_NL (90 B/row, **28.8 GB**),
read 16 rows per token (hash heads). The row ids depend **only on token ids**, so every row of a chunk is known before layer 0 and can
be paged in while the earlier layers compute. Strata / llama.cpp measured the same lever: prefetching the table alone took llama.cpp
prefill 177 -> 383 tok/s on this box.

## What is in the tree

| piece | where |
|---|---|
| `PleRowPrefetcher` (page coalescing, OS hint, touch pool, clock row cache, stats) | `src/DotLLM.Cpu/Kernels/Qwen4ExpPleRowPrefetcher.cs` |
| `Qwen4ExpPleBranch.BeginPrefetch(tokens, state)` + `Apply` consuming it | `src/DotLLM.Cpu/Kernels/Qwen4ExpPle.cs` |
| CPU model call site (before layer 0) + `PleTableStats` | `Qwen4ExpTransformerModel.ForwardBody` |
| tests / real-file bench | `tests/DotLLM.Tests.Unit/Models/Qwen4Exp/Qwen4ExpPleTablePrefetchTests.cs`, `...PleTableRealBench.cs` |

**Correctness invariant.** The service changes *when* pages become resident and which copy of the same bytes is dequantised; every
row goes through the same `Dequantize.ToFloat32` from byte-identical data. A request that is late, abandoned, mismatched or never
issued cannot change a result.

## Configuration

| env | values | default |
|---|---|---|
| `DOTLLM_PLE_TABLE` | `mmap` rows read from the lazy mapping (page cache is the tier); `ram` also keeps used rows in a process-owned clock cache | `mmap` |
| `DOTLLM_PLE_PREFETCH` | `off`, `os` (`PrefetchVirtualMemory` / `madvise(WILLNEED)`), `touch` (thread pool reads a byte per page), `auto` (= os then touch) | `auto` |
| `DOTLLM_PLE_CACHE_MB` | row-cache budget (0 disables) | 0 mmap / 64 ram |
| `DOTLLM_PLE_TOUCH_THREADS` | touch / cache-fill threads | min(8, cores) |
| `DOTLLM_PLE_STATS=1` | sample the process page-fault counter around gathers (process-wide, cheap) | off |

Formats: anything `Dequantize` handles (IQ4_NL, BF16, F16, F32, Q8_0 are covered by tests). Stats: `PleRowPrefetcher.Stats` /
`Qwen4ExpTransformerModel.PleTableStats` (requests, rows, unique rows/pages, coalesced ranges, cache hits/misses, wait / worker /
gather microseconds, gather faults, hint failures) and the `DotLLM.Ple` `Meter` (`dotllm.ple.rows`, `.unique_pages`, `.cache_hits`,
`.cache_misses`, `.wait` (us), `.gather_faults`; observable counters, so zero cost without a listener).

## Measurements (CPU only, this Strix Halo box, UD-Q4_K_XL, 2026-10-09 ~19:00)

Genuine disk-cold evidence: 320M rows and ~8k rows per 512-token chunk means a fresh id set lands on essentially unread pages, so
"cold" = fresh random ids, "warm" = repeat. The probe confirms the table was *not* page-cached (431 us/row demand-faulted = NVMe
QD1; warm repeat 1 us/row). No eviction trick was needed or attempted.

Exposed time of one chunk's table gather with **no lead time at all** (Begin immediately followed by Collect), medians of two reps:

| chunk | demand (no prefetch) | OS hint | touch x16 | OS hint + touch x16 | ram + touch x16 |
|---|---|---|---|---|---|
| 512 tok (8192 rows, ~8.3k pages) | 3.56 s | 0.12 s | 0.31 s | 0.12 s | 0.41 s |
| 1024 tok (16384 rows, ~16.7k pages) | 6.77 s | 0.19 s | 0.50 s | 0.19 s | 0.55 s |
| warm repeat of the gather | 9-16 ms | 20-34 ms (pages standby, not yet in WS) | 8-16 ms | 8-16 ms | 6-12 ms |

* **`PrefetchVirtualMemory` is the lever**: ~75k pages/s vs ~2.6k pages/s for demand faults (a single thread faulting one 4 KiB page
  at a time = 381 us/page) and 28k pages/s for the best touch pool. 30-37x faster than demand. `auto` (= hint, then touch) adds the
  touch only to move the pages into the working set so the gather is 8-16 ms instead of 20-34 ms; the touch is hidden behind compute.
* **Random-fault scaling** (`FAULTS` bench, 40k fresh random pages): 1 thread 2.6k pages/s, 4: 10.2k, 8: 17.7k, 16: 27.9k, 32: 23.2k,
  64: 28.1k. The device saturates at ~28k random 4 KiB reads/s (~110 MB/s of useful data!) via faulting; the OS hint reaches ~75k/s
  because it issues the IO as one large queue. More touch threads than 16 hurt (they fight each other and the compute pool).
* **Decode** (1 token = 16 rows, fresh ids, 60 trials): cold demand gather median 4.36 ms (p90 5.9) vs 12 us warm; prefetch
  Begin..Collect with no lead median 0.59 ms, gather afterwards 11 us. With the lead of one decode layer (~1 ms on Vulkan) the
  cold-start penalty falls from ~4.4 ms/token (~7% of a 58 ms token at 17 tok/s) to ~0.
* **Lead time available**: the hint (0.12-0.19 s per chunk) is shorter than layer 0 of a chunk on the CPU oracle (seconds) and of the
  order of one layer of a 1K-token Vulkan prefill (~140 ms), so a prefetch started at forward entry is fully hidden on CPU and mostly
  hidden on Vulkan; chunked prefill can hide the *next* chunk entirely by prefetching it during the current one (ids and carried
  history are known - see the wiring notes).

### Windows mapped-file residency (documented risk)

* A page hinted/touched becomes **working-set resident immediately** (1308 -> 16717 of 16717 needed pages) and the unrequested
  neighbours stay at the baseline residency (1294 / 16326, 7.9%, same as before): **no read-around amplification** in the working set.
* `EmptyWorkingSet` (simulated trimming) drops the table pages to 0 working-set-resident and the working set 2678 -> 84 MB, but the
  pages stay on the **standby list**: re-gathering the same 16.7k pages cost 45 ms of soft faults (19.5k faults) vs 6.8 s cold. So
  trimming is cheap; **eviction** (standby repurposed under memory pressure - plausible next to a 59 GB GPU-agent working set on a
  UMA box) is what costs, and then `ram` mode only helps for repeated tokens (the clock cache), not for new ones.
* Hence mmap is the default: the page cache already is the table's tier and a 28.8 GB private copy would not fit next to a 69 GB
  Vulkan heap. `ram` is for hosts where mapped pages are trimmed/evicted aggressively.
* Soak (llama.cpp #28933 class, scaled 128 MiB table, 20 prefill chunks): table-resident pages track the touched footprint
  (3187 resident vs 3794 touched = 0.84x) and replaying already-seen chunks adds 0 pages. `PrivateMemorySize64` is blind to mapped
  pages, so the test counts the table's own pages with `QueryWorkingSetEx` (Windows; skipped elsewhere).

### CPU oracle, real file, 512 tokens (byte identity)

Both arms: `Qwen4ExpTransformerModel` CPU oracle on the UD-Q4_K_XL file, 8 threads (kept low: the machine was shared with a GPU
benchmark), 512 deterministic token ids containing EOS resets, all-row logits (508,559,360 bytes). `DOTLLM_PLE_PREFETCH=off` without a
cache (the pre-#822 path) vs the default service. **`cmp` says the two logits files are byte-identical** (md5 8b579a17...).
The `on` arm's stats prove the prefetch ran: 1 request, 8192 rows, 8381 pages in 8179 coalesced ranges, `WaitMicros = 0`.

Not claimed: a whole-forward speed-up. Off ran 391 s and on ran 497 s, but the `off` arm had already paid the cold gather for these
exact ids (so `on` read warm pages), the box was shared, and the PLE gather is ~1% of a CPU-oracle forward (cold demand gather of
512 tokens ~3.6 s of ~400 s). The ordering/contention noise dominates; the PLE-specific numbers above (wait, faults) are the
measurement. A repeat of the `off` arm as a determinism control was not run (on == off already shows the two agree).


## Wiring the Vulkan model (DONE in #820 stage 2; the notes below are what was wired)

**Status (2026-10-10).** Items 1, 2 and 4 are in: `VulkanQwen4ExpTransformerModel.ForwardCore` calls `BeginPrefetch` right after `Begin()` and before
`Q4RecordEmbedding`, the model disposes the prefetcher, `PleTableStats` is exposed, and the MTP draft loop re-issues the request with the verify
chunk's known prefix `[last token, d1 .. di]` after every draft step. Measured on the real file, cold 1K-token prefill of fresh random ids, A/B/B/A in
one process: **7.7-7.9 s with the prefetch, 14.4-14.6 s without** (-6.7 s per 1K chunk, 1.85x; the CPU measurement above predicted 6.8 s of demand
faults). `DOTLLM_VK_Q4E_PREFETCH=0` disables it for A/B. Item 3 (prefetching chunk k+1 during chunk k) is not done.

`VulkanQwen4ExpTransformerModel` builds the same `Qwen4ExpPleBranch` and runs it on the host at the PLE layer. Today it is inert
(mmap default, no `BeginPrefetch` call, `Gather(null, ...)` = the plain path plus counters). To get the win:

1. In `ForwardCore`, right after `Begin()` and **before** `_core.Q4RecordEmbedding`, call `_ple?.BeginPrefetch(tokenIds, state.Ple!)`
   (the hash window in `state.Ple` is still the chunk-start one there). The pages then stream in under layer 0's GPU work and
   `Apply` at the PLE layer collects them.
2. Dispose the service with the model (`_ple?.Prefetcher?.Dispose()`); only matters in `ram` mode (the cache slab).
3. Optional, larger win for chunked prefill: prefetch chunk *k+1* while chunk *k* runs.
   `Qwen4ExpPle.BuildRowIndices(nextTokens, historyAfterThisChunk, ...)` + `branch.Prefetcher.Begin(rows)` and hand the request to
   `Gather` later; `historyAfterThisChunk` is `Qwen4ExpPle.AdvanceHistory` over the current history.
4. MTP verification passes: the draft tokens are known before verify, so `BeginPrefetch` on the verify batch hides the (cold) rows of
   the speculative positions the same way.
