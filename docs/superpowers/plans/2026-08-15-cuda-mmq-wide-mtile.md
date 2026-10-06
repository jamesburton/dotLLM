# Batched-MMQ Q4_K Wide M-Tile Redesign (issue #367) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `quantized_gemv_q4_k_mmq_batched` (`native/kernels/quantized_gemv_mmq.cu`) so weight reads amortize across a wide M-tile (tens of tokens) via shared-memory weight staging, turning #349's correct-but-2x-51x-slower kernel into one with a real crossover where the quantized-GEMM path beats dequant→cuBLAS at some prefill length — then enable it by default above that measured crossover.

**Architecture:** #349's kernel pays decode-GEMV cost once per token (`MMQ_BATCH_M_TILE=2`, one warp per (row, token-pair), weight bytes re-read from global per pair — flat ~20ms/token at every seqLen). The redesign follows llama.cpp `ggml-cuda/mmq.cu`'s proven shape for the same Q4_K dp4a primitive: each block stages a WEIGHT tile (a slice of N output rows × one K-chunk of Q4_K blocks) into shared memory ONCE, and all warps in the block consume that staged copy against MANY tokens' Q8_1-quantized activations (also staged), accumulating per-(row, token) int32 dp4a partials in registers across the K-loop. Weight-read volume drops from O(rows × seqLen) to O(rows × seqLen / M_TILE_WIDE). The Q8_1 activation quantization pipeline from #349 (`quantize_x.cu` → `PreQ8_1BatchedScratch`) is reused unchanged; the dispatcher gate (`MmqBatchedMinSeqLen` + the 7-condition check incl. `PreQ8_1BatchedScratch != 0`) is reused with the threshold set from measurement instead of `int.MaxValue`.

**Tech Stack:** CUDA C (dp4a — `__dp4a`, sm_61+; repo targets compute_75), pinned toolkit `E:\CUDA_v12.8.1` → `.version 8.7` PTX; C# launcher in `CudaKernels.cs`; xUnit real-GPU tests; the #349 benchmark harness (already committed).

## Global Constraints

- Branch `issue/367-mmq-wide-mtile` from `dev`. Commits reference `(#367)`.
- **Prior art is committed and MUST be reused, not rewritten**: the #349 deliverables — kernel file `native/kernels/quantized_gemv_mmq.cu`, launcher `LaunchQuantizedGemvMmqBatchedQ4K` + `MmqBatchedMinSeqLen` + the 7-condition dispatcher gate in `CudaKernels.cs`/`CudaTransformerModel.cs`, scratch plumbing (`PreQ8_1BatchedScratch`, gated allocation), `quantize_x.cu`, the discriminating loop-of-M=1-oracle parity test (#349 Task 4), and the benchmark sweep methodology (#349 Task 6, documented in `docs/superpowers/plans/2026-08-12-cuda-prefill-batched-mmq.md` and `docs/perf/MMA_BATCHED_MMQ.md`). Read ALL of these before writing kernel code.
- llama.cpp's `ggml-cuda/mmq.cu` is the authoritative reference for the tile-and-cache structure (per CLAUDE.md: "CUDA: Reference llama.cpp ggml-cuda/ for proven kernels"). A local llama.cpp checkout may exist — search `E:\Development` and `C:\Development` for `llama.cpp`; if absent, fetch the single file from GitHub raw (network has been flaky — retry pattern).
- PTX rules: pinned `E:\CUDA_v12.8.1\bin\nvcc.exe`, `.version 8.7`, `.ptx` mtime newer than `.cu` (local MSBuild CompileCudaPtx is broken). Check whether `quantized_gemv_mmq` sits in `build_ptx.bat`'s NO_FMA/FAST_MATH lists and PRESERVE its current classification (integer dp4a math is unaffected by fmad; the float scale-combine follows the existing kernel's precedent).
- Correctness gates: the #349 parity test (loop-of-M=1 oracle) re-verified against the new kernel at the SAME tolerances, plus new M-tile-boundary shapes for the wider tile (seqLen exactly at, one-under, one-over the new tile width). Documented tolerances, never `SequenceEqual` (int dp4a core is exact; the per-block float scale combine is where drift lives — #349 measured 0.000-0.037%).
- Perf gate (the actual acceptance): the #349 Task 6 sweep methodology re-run — `-p 4..512` on `unsloth/Qwen3-4B-GGUF` Q4_K_M + the pure-Q4_K Llama-3.2-1B fixture, `-n 32 -r 5`, interleaved re-check for thermal drift. REQUIRED OUTCOME: a measured crossover point where the new path beats baseline; set `MmqBatchedMinSeqLen` to that crossover. IF NO CROSSOVER EXISTS after the redesign (possible — cuBLAS HGEMM on a 3060 is strong), the honest outcome is the #349 precedent: document the numbers, keep `int.MaxValue`, update the issue with the findings, and DO NOT ship a regression. Both outcomes are in-scope plan completions.
- Idle-GPU discipline for all benchmark runs (`nvidia-smi` check first); no other agents' test runs concurrent with sweeps.

---

### Task 1: Reference extraction — llama.cpp mmq tile geometry (bounded study, concrete deliverable)

**Files:**
- Create: `.docs/367-mmq-reference-notes.md` (git-ignored)

- [ ] **Step 1:** Read llama.cpp `ggml-cuda/mmq.cu` (+ `mmq.cuh` if split) for the Q4_K path and extract, with file:line citations: (a) tile dims (mmq_x = token-tile width, mmq_y = row-tile height, and the K-chunk depth per shared-staging iteration) for the sm_75-class config; (b) the shared-memory layout for staged Q4_K weight tiles (how qs/scales/dmin unpack into smem: raw block bytes vs pre-unpacked int8); (c) the Q8_1 activation tile layout and how it pairs with dp4a against the staged weights; (d) the register accumulator arrangement (how many (row, token) partials per thread) and the final scale-combine math (Q4_K's per-block d/dmin × Q8_1's per-block d/s — write the EXACT formula as llama.cpp computes it); (e) the loop structure (K-outer staging loop, warp-level work division). Deliverable: the notes file containing a self-contained pseudocode skeleton of the Q4_K mmq kernel with all constants for one chosen config — complete enough that Task 2 can be written from the notes alone.
- [ ] **Step 2:** Sanity-check against the EXISTING #349 kernel: its Q8_1 scratch format (what `quantize_x.cu` produces — block layout, scale placement) vs what the reference layout wants. If they differ, note the adapter needed (changing `quantize_x.cu`'s output format is allowed but pulls that kernel into scope — prefer consuming the existing format if the delta is only indexing). Commit nothing; report the notes path.

### Task 2: The wide-tile kernel

**Files:**
- Modify: `native/kernels/quantized_gemv_mmq.cu` (ADD `quantized_gemm_q4_k_mmq_wide` as a NEW entry point; the #349 narrow kernel stays — it is the parity oracle's subject and the fallback)
- Rebuild: `native/ptx/quantized_gemv_mmq.ptx`

**Interfaces:**
- Produces: `extern "C" __global__ void quantized_gemm_q4_k_mmq_wide(const void* w_q4k, const void* x_q8_1, float* y, int rows, int k, int seq_len)` — same logical contract as the batched kernel (`y[token*rows + row]` or the existing layout — MATCH the #349 kernel's output layout exactly; read it), wide M-tile internally.

- [ ] **Step 1:** Implement from the Task 1 notes: block = (WARP_SIZE × NWARPS) threads; grid = (ceil(rows / MMQ_Y), ceil(seq_len / MMQ_X)); K-outer loop stages one weight K-chunk (MMQ_Y rows) + one activation K-chunk (MMQ_X tokens) into shared, `__syncthreads()`, dp4a-accumulate all (row, token) partials in registers, `__syncthreads()`, next chunk; epilogue applies the exact scale-combine formula from Task 1(d) and writes y. Start with llama.cpp's sm_75 config constants from the notes (typical: MMQ_X 32-64, MMQ_Y 64-128); make them `#define`s for Task 4's tuning. Shared-memory budget check: staged bytes must fit 48KB (compute_75 default) — compute and assert in a comment.
- [ ] **Step 2:** Rebuild PTX (pinned toolkit, verify `.version 8.7`, all THREE entry symbols — old gemv, old batched, new wide — present, mtime rule).
- [ ] **Step 3:** Launcher `LaunchQuantizedGemvMmqWideQ4K` in `CudaKernels.cs`: Tier-2 optional loading against the SAME module (TryGetFunction for the new symbol — a stale PTX lacking it must not break the old paths), grid/block per Step 1, XML docs.
- [ ] **Step 4:** Build 0/0. Commit: `git commit -am "feat(cuda): wide M-tile Q4_K MMQ GEMM kernel (#367)"`

### Task 3: Parity gates

**Files:**
- Modify/extend: the #349 parity test file (grep `tests/` for the loop-of-M=1 oracle test from #349 Task 4)

- [ ] **Step 1:** Point the same oracle methodology at the NEW kernel: for each shape, wide-kernel output vs loop-of-M=1 `quantized_gemv_q4_k` oracle, #349's tolerances. Shapes: #349's original four PLUS new-tile boundary cases — seqLen ∈ {MMQ_X−1, MMQ_X, MMQ_X+1, 3×MMQ_X+5} and rows ∈ {MMQ_Y−1, MMQ_Y+3} crossings (partial-tile edge handling is where wide-tile kernels break; these discriminate).
- [ ] **Step 2:** Run on real GPU — all green at unchanged tolerances. Any drift beyond #349's 0.037% band = a scale-combine transcription bug; stop and fix against the Task 1 formula before proceeding.
- [ ] **Step 3:** Commit: `git commit -am "test(cuda): wide-MMQ parity vs M=1 oracle incl. tile-boundary shapes (#367)"`

### Task 4: Benchmark sweep, tuning, threshold decision

- [ ] **Step 1:** Re-run the #349 Task 6 sweep verbatim (same models, same `-p` grid, `-n 32 -r 5`, interleaved re-check, idle GPU) with the dispatcher temporarily routed to the wide kernel above `MmqBatchedMinSeqLen=4`. Record the full table in `.docs/367-sweep.md`.
- [ ] **Step 2:** If the wide kernel loses everywhere: one tuning iteration over the MMQ_X/MMQ_Y/NWARPS defines (at most 3 configs, chosen from the Task 1 notes' alternatives), re-sweep the best. Do not spiral — two sweeps total, then decide.
- [ ] **Step 3: Decision.** Crossover exists → set `MmqBatchedMinSeqLen` to the measured crossover (round up to the next power of two for stability), route the ≥threshold path to the WIDE kernel (narrow batched kernel stays for nothing — note it as dead-but-oracle-adjacent, or remove its dispatch reachability while keeping the entry point for the parity oracle's A/B), commit with the table in the message. No crossover → keep `int.MaxValue`, write the honest findings to `docs/perf/MMA_BATCHED_MMQ.md` (append a dated section), update issue #367 with the table and close-or-keep recommendation for the user. Either way: `git commit -am "perf(cuda): wide-MMQ sweep outcome + threshold decision (#367)"`

### Task 5: Docs + guard rails

- [ ] **Step 1:** Update `docs/perf/MMA_BATCHED_MMQ.md` with the redesign's architecture (tile dims, staging layout, measured table) regardless of outcome. If enabled: note the threshold and the env-var/config override story (whatever #349 shipped for `MmqBatchedMinSeqLen` configurability — read it).
- [ ] **Step 2:** Full CUDA test-filter regression; merge via superpowers:finishing-a-development-branch.

## Self-review checklist (author ran per the writing-plans skill)
- Spec coverage vs the issue's acceptance criteria: wide-tile redesign with shared staging ✓ (Task 2), same parity methodology re-verified ✓ (Task 3), same sweep methodology with crossover expectation ✓ (Task 4) — including the honest no-crossover outcome the issue's own history (#349) establishes as legitimate.
- No placeholders: the one unavoidable unknown (llama.cpp's exact tile constants/layouts) is structured as a bounded extraction task with a concrete deliverable (self-contained pseudocode skeleton + exact scale-combine formula) that Task 2 consumes — not a "TBD". Everything else (entry-point signature, grid/block scheme, loop structure, boundary-shape test grid, sweep protocol, both decision outcomes) is specified here.
- Type consistency: new entry `quantized_gemm_q4_k_mmq_wide` with the #349 kernel's exact output layout (Task 2 Step 1 pins it by reading, not assuming); launcher name `LaunchQuantizedGemvMmqWideQ4K` used consistently in Tasks 2-4.
