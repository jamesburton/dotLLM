# CUDA Prefill dp4a-Batched-MMQ for Q4_K (Issue #349) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give CUDA prefill (seqLen > 1) a genuine quantized-GEMM path for Q4_K weights — a batched-M dp4a MMQ kernel that reads weight bytes directly, instead of unconditionally dequantizing the whole weight matrix to FP16 and calling cuBLAS HGEMM on every prefill call regardless of quant type.

**Architecture:** Fork the existing `quantized_gemv_q4_k_mmq_preq` dp4a kernel (native/kernels/quantized_gemv_mmq.cu) into a new `quantized_gemv_q4_k_mmq_batched` kernel that adds an M (activation-row / prefill-token) tile alongside the existing N (output-row) tile: each CUDA block owns `MMQ_ROWS_PER_BLOCK`(4) × `MMQ_BATCH_M_TILE`(2) = 8 warps, one warp per (output-row, token) pair, each warp independently walking the full K dimension and warp-shuffle-reducing to one scalar. Warps in the same block assigned the same output row (different tokens) read the same 144-byte Q4_K superblocks within a few cycles of each other, so L1/L2 cache absorbs the repeat weight reads — the mechanism that amortizes weight bandwidth across the M tile instead of re-dequantizing the full matrix on every prefill call. A companion `quantize_x_to_q8_1_batched` kernel (native/kernels/quantize_x.cu) pre-quantizes all `seqLen` activation rows to INT8/Q8_1 in one 2D-grid launch, feeding the batched MMQ kernel exactly like the existing single-row `_preq` path feeds `quantized_gemv_q4_k_mmq_preq` today. A new seqLen-keyed dispatcher gate in `CudaTransformerModel.Project` picks this path over the dequant→cuBLAS HGEMM fallback, mirroring llama.cpp's `MMVQ_MAX_BATCH_SIZE`-style batch-size dispatch (docs/perf/MMA_BATCHED_MMQ.md §1d).

**Tech Stack:** CUDA C (`.cu` → PTX via `nvcc -ptx`, no CMake), C# .NET 10 CUDA Driver API wrapper (`CudaKernels.cs`, `cuLaunchKernel` via `CudaDriverApi`), xUnit (`Xunit.SkippableFact`) GPU-gated unit tests.

## Global Constraints

- .NET 10, `<Nullable>enable</Nullable>` project-wide, file-scoped namespaces, XML doc comments on all public APIs (per CLAUDE.md Code Style).
- GPU memory handles use `nint` (device pointer) per existing `CudaKernels.cs` / `CudaForwardState.cs` convention — this codebase's actual GPU interop pattern is the CUDA **Driver API** (`cuLaunchKernel` via a `CudaDriverApi` P/Invoke wrapper, PTX modules loaded at runtime with `CudaModule.LoadFromFile`/`TryGetFunction`), not a bespoke native shared library — follow that established pattern exactly, do not introduce a new native library entry point.
- No CMake / no C/C++ build system: `.cu` kernels compile via a single `nvcc -ptx` invocation through **`native/build_ptx.bat`** (Windows) / `native/build.sh` (Linux — kept in sync with the `.bat`, same `FAST_MATH`/`FAST_MATH_KERNELS` allow-list), output to `native/ptx/*.ptx`, one `.ptx` file per `.cu` file. **Do NOT use `native/build.ps1`** — despite the similar name, it is a separate, legacy script (its own comment self-describes as "the legacy fast-math path"): it defaults to `--use_fast_math` for every kernel except a small opt-out list, the opposite convention from `build_ptx.bat`'s small opt-**in** `FAST_MATH` list that the rest of this file's kernels (and every recently-added kernel elsewhere in the tree) actually build under, and it hardcodes an expected PTX ISA version (8.7 / CUDA 12.8) that does not match this environment's toolkit (13.1 emits 9.1) — running it would either throw on the version assert or silently rewrite every other committed `.ptx` file's fast-math flag. **Extend the two existing files** (`quantized_gemv_mmq.cu`, `quantize_x.cu`) — do not create new `.cu` files, so no changes to the build script or to `CudaKernels.cs`'s per-module `File.Exists`/`LoadFromFile` wiring are needed, only new `TryGetFunction` calls inside the already-existing `if (File.Exists(...))` blocks.
- PTX target is `compute_75`, **without** `--use_fast_math`: neither new kernel's base filename (`quantized_gemv_mmq`, `quantize_x`) appears in `build_ptx.bat`'s `FAST_MATH` allow-list (`add add_f32 swiglu swiglu_f32 convert bias_add bias_add_f32 embedding embedding_f32out dequant quant_kv`), so both files already compile with precise math today — matching the existing `_preq`/legacy MMQ kernels in the same two files, which are also not fast-math. This has effectively no numerical impact on the new kernels regardless (neither uses any transcendental function — pure `__dp4a`/multiply-add/warp-shuffle arithmetic), but the build-script correction above matters for the REST of the PTX tree.
- `*.ptx` files ARE tracked in git on purpose (216 compiled shaders/kernels are legitimate build artifacts per CLAUDE.md) — after regenerating PTX, `git add` the updated `native/ptx/quantized_gemv_mmq.ptx` and `native/ptx/quantize_x.ptx` alongside the `.cu` source changes.
- Model weights NEVER live in or get copied into the repository. The benchmark task (Task 6) must resolve a model via the existing GGUF path / HuggingFace repo-ID resolution built into the `bench` CLI command, never via a fixture copied into the working tree.
- Priority order per CLAUDE.md: Correctness then Performance then Extensibility — the discriminating parity test (Task 4) must pass before the dispatcher gate (Task 5) routes real forward-pass traffic through the new kernel.
- Cross-backend critical-bug propagation (CLAUDE.md's "Cross-Backend Critical Bugs" rule) does not apply to this plan: this is a new perf-only kernel path explicitly scoped to CUDA by the issue (no CPU/Vulkan/HIP logic is being changed, and no existing bug is being fixed).
- `[SkipLocalsInit]` / `[MethodImpl(AggressiveInlining)]` are CPU-hot-path rules from CLAUDE.md's SIMD section — not applicable to CUDA kernel code or to the Driver-API launcher methods in `CudaKernels.cs` (mirrors the file's existing launchers, none of which use them).
- Reference llama.cpp's `ggml-cuda/mmq.cu` / `mmvq.cuh` batch-size dispatch (`MMVQ_MAX_BATCH_SIZE=8`, `MMQ_DP4A_MAX_BATCH_SIZE=64`) as the authoritative shape for the new seqLen-keyed gate, already transcribed verbatim in `docs/perf/MMA_BATCHED_MMQ.md` §1d.

---

## File Structure

| File | Change |
|---|---|
| `native/kernels/quantized_gemv_mmq.cu` | Add `quantized_gemv_q4_k_mmq_batched` kernel (Task 1) |
| `native/kernels/quantize_x.cu` | Add `quantize_x_to_q8_1_batched` kernel (Task 2) |
| `src/DotLLM.Cuda/CudaForwardState.cs` | Add `PreQ8_1BatchedScratch` buffer, sized in `EnsureCapacity`/freed in `FreeSequenceBuffers` (Task 3) |
| `src/DotLLM.Cuda/CudaKernels.cs` | Add PTX symbol loading, `HasMmqBatchedQ4K`, `DisableMmqBatchedQ4K`, `MmqBatchedMinSeqLen`, `LaunchQuantizeXToQ8_1Batched`, `LaunchQuantizedGemvMmqBatchedQ4K` (Task 3) |
| `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs` | Add discriminating parity test (loop-of-M=1 oracle vs. batched kernel) (Task 4) |
| `src/DotLLM.Cuda/CudaTransformerModel.cs` | Wire the seqLen-keyed dispatcher gate into `Project` (Task 5) |
| `docs/CUDA.md` | Update kernel table + two future-work lines (Task 7) |

No new files are created — every change extends an existing module, matching the two existing `.cu` files' role as the home for MMQ/pre-quant kernel families.

---

### Task 1: Batched-M dp4a MMQ Q4_K prefill kernel (CUDA)

**Files:**
- Modify: `native/kernels/quantized_gemv_mmq.cu`

**Interfaces:**
- Consumes: nothing new — reuses the file-scope `#define MMQ_ROWS_PER_BLOCK 4` (line 50) and the `__dp4a` Q4_K math already used by `quantized_gemv_q4_k_mmq_preq` (lines 2285–2403).
- Produces: `extern "C" __global__ void quantized_gemv_q4_k_mmq_batched(const uint8_t* weight, const int8_t* xq_in, const half* dx_in, const half* sx2_in, half* y, int n, int k, int m)` — consumed by Task 3's C# launcher (`LaunchQuantizedGemvMmqBatchedQ4K`) and Task 4's test.

- [ ] **Step 1: Add the batched kernel right after `quantized_gemv_q4_k_mmq_preq`**

Open `native/kernels/quantized_gemv_mmq.cu` and locate the end of `quantized_gemv_q4_k_mmq_preq` (closing brace immediately before `extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q5_k_mmq_preq(`). Insert the new kernel between them:

```cuda
// ────────────────────────────────────────────────────────────────────────────
// Q4_K batched-M dp4a MMQ — prefill kernel (issue #349).
//
// Extends quantized_gemv_q4_k_mmq_preq to a BATCH of M activation rows (prefill
// tokens) processed in one launch, instead of one M=1 GEMV per token. The grid
// tiles BOTH output rows (N, MMQ_ROWS_PER_BLOCK=4, same tile as every other MMQ
// kernel in this file) and activation rows (M, MMQ_BATCH_M_TILE=2), so each
// block owns MMQ_ROWS_PER_BLOCK * MMQ_BATCH_M_TILE = 8 warps — one warp per
// (output row, token) pair. Each warp independently walks the full
// superblocks_per_row loop for its pair (lanes split the loop via a stride-32
// loop) and warp-shfl-reduces to one scalar — no shared memory at all.
//
// The amortization mechanism: all MMQ_BATCH_M_TILE warps assigned the SAME
// output row (different tokens) read the SAME 144-byte weight superblocks
// within a few cycles of each other on the same SM, so L1/L2 cache serves the
// repeat reads. This is what lets the batched path avoid re-dequantizing the
// whole [n,k] weight matrix to FP16 on every prefill call (the dequant->cuBLAS
// baseline pays that O(n*k) cost regardless of seqLen); see
// docs/perf/MMA_BATCHED_MMQ.md §3a step 2 for the design rationale.
//
// Consumes pre-quantized activations from quantize_x_to_q8_1_batched (int8 xq /
// half dx / half sx2, laid out as [m,k] / [m,k/32] / [m,k/16] row-major
// sections concatenated by section, NOT interleaved per token).
#define MMQ_BATCH_M_TILE 2

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q4_k_mmq_batched(
    const uint8_t* __restrict__ weight,
    const int8_t*  __restrict__ xq_in,   // [m, k] int8, row stride k
    const half*    __restrict__ dx_in,   // [m, k/32] half, row stride k/32
    const half*    __restrict__ sx2_in,  // [m, k/16] half (2 per chunk), row stride k/16
    half* __restrict__ y,                // [m, n] half, row stride n
    const int n,
    const int k,
    const int m)
{
    const int row_base = blockIdx.x * MMQ_ROWS_PER_BLOCK;
    const int m_base = blockIdx.y * MMQ_BATCH_M_TILE;
    if (row_base >= n || m_base >= m) return;

    const int superblocks_per_row = k / 256;
    const int num_chunks = k >> 5;   // k/32 chunks per activation row

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;                            // 0..7
    const int lane = tid & 31;
    const int row_in_tile = warp_id % MMQ_ROWS_PER_BLOCK;     // 0..3
    const int m_in_tile   = warp_id / MMQ_ROWS_PER_BLOCK;     // 0..1

    const int row = row_base + row_in_tile;
    const int token = m_base + m_in_tile;
    if (row >= n || token >= m) return;

    const int8_t* xq_row  = xq_in  + (size_t)token * k;
    const half*   dx_row  = dx_in  + (size_t)token * num_chunks;
    const half*   sx2_row = sx2_in + (size_t)token * num_chunks * 2;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 144;

    float row_acc = 0.0f;

    // Each lane handles a stride-32 subset of this (row, token) pair's
    // superblocks — identical per-superblock math to quantized_gemv_q4_k_mmq_preq,
    // just executed by one lane per superblock instead of scattered across
    // 256 threads with a shared-memory reduction.
    for (int sb = lane; sb < superblocks_per_row; sb += 32)
    {
        const uint8_t* block = w_row + sb * 144;
        float d    = __half2float(*reinterpret_cast<const half*>(block));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 2));
        const uint8_t* scales_raw = block + 4;
        const uint8_t* qs = block + 16;

        #pragma unroll
        for (int pair = 0; pair < 4; pair++)
        {
            int sb_even = pair * 2;
            int sb_odd  = pair * 2 + 1;

            int sc0, m0, sc1, m1;
            if (sb_even < 4)
            {
                sc0 = scales_raw[sb_even]     & 0x3F;
                m0  = scales_raw[sb_even + 4] & 0x3F;
                sc1 = scales_raw[sb_odd]      & 0x3F;
                m1  = scales_raw[sb_odd + 4]  & 0x3F;
            }
            else
            {
                sc0 = (scales_raw[sb_even + 4] & 0x0F) | ((scales_raw[sb_even - 4] >> 6) << 4);
                m0  = (scales_raw[sb_even + 4] >> 4)   | ((scales_raw[sb_even]     >> 6) << 4);
                sc1 = (scales_raw[sb_odd + 4]  & 0x0F) | ((scales_raw[sb_odd - 4]  >> 6) << 4);
                m1  = (scales_raw[sb_odd + 4]  >> 4)   | ((scales_raw[sb_odd]      >> 6) << 4);
            }

            const uint8_t* pair_qs = qs + pair * 32;
            int chunk_even = sb * 8 + sb_even;
            int chunk_odd  = sb * 8 + sb_odd;

            const int8_t* xq_even = xq_row + chunk_even * 32;
            const int8_t* xq_odd  = xq_row + chunk_odd  * 32;

            int dot0 = 0;
            int dot1 = 0;

            #pragma unroll
            for (int g = 0; g < 8; g++)
            {
                uint32_t qpacked = *reinterpret_cast<const uint32_t*>(pair_qs + g * 4);
                int lo = (int)(qpacked & 0x0F0F0F0F);
                int hi = (int)((qpacked >> 4) & 0x0F0F0F0F);

                int xq_e_packed = *reinterpret_cast<const int*>(xq_even + g * 4);
                int xq_o_packed = *reinterpret_cast<const int*>(xq_odd  + g * 4);

                dot0 = __dp4a(lo, xq_e_packed, dot0);
                dot1 = __dp4a(hi, xq_o_packed, dot1);
            }

            float dx_e = __half2float(dx_row[chunk_even]);
            float dx_o = __half2float(dx_row[chunk_odd]);
            float sx_e = __half2float(sx2_row[chunk_even * 2 + 0]) + __half2float(sx2_row[chunk_even * 2 + 1]);
            float sx_o = __half2float(sx2_row[chunk_odd  * 2 + 0]) + __half2float(sx2_row[chunk_odd  * 2 + 1]);

            row_acc += dx_e * (d * (float)sc0 * (float)dot0 - dmin * (float)m0 * sx_e);
            row_acc += dx_o * (d * (float)sc1 * (float)dot1 - dmin * (float)m1 * sx_o);
        }
    }

    // Warp-shfl reduction across the 32 lanes.
    for (int offset = 16; offset > 0; offset >>= 1)
        row_acc += __shfl_xor_sync(0xFFFFFFFF, row_acc, offset);

    if (lane == 0)
        y[(size_t)token * n + row] = __float2half(row_acc);
}
```

- [ ] **Step 2: Regenerate PTX and verify the new symbol compiles**

Run (Windows — from the repo root, NOT `native/build.ps1`, see Global Constraints for why):

```
native\build_ptx.bat
```

Expected output includes a line for `quantized_gemv_mmq.cu -> quantized_gemv_mmq.ptx` (or the script's equivalent per-file success line) with no `FAIL` marker (the script sets an error flag and reports failures at the end rather than throwing immediately).

- [ ] **Step 3: Verify the new kernel symbol is present in the regenerated PTX**

```powershell
Select-String -Path native/ptx/quantized_gemv_mmq.ptx -Pattern "quantized_gemv_q4_k_mmq_batched"
```

Expected: at least one match (the PTX `.visible .entry` line for the new kernel).

- [ ] **Step 4: Commit**

```bash
git add native/kernels/quantized_gemv_mmq.cu native/ptx/quantized_gemv_mmq.ptx
git commit -m "feat(cuda): add batched-M dp4a MMQ Q4_K prefill kernel (#349)"
```

---

### Task 2: Row-aware (batched) input quantization kernel (CUDA)

**Files:**
- Modify: `native/kernels/quantize_x.cu`

**Interfaces:**
- Consumes: nothing new — reuses `QX_THREADS_X`/`QX_WARPS_PER_BLOCK` (lines 29–31) and the per-chunk INT8 quantization math already in `quantize_x_to_q8_1` (lines 33–79).
- Produces: `extern "C" __global__ void quantize_x_to_q8_1_batched(const half* x, int8_t* xq, half* dx, half* sx2, int k)` — launched with a 2D grid (`blockIdx.y` = token index); consumed by Task 3's `LaunchQuantizeXToQ8_1Batched` and Task 4's test. Feeds Task 1's `quantized_gemv_q4_k_mmq_batched`.

- [ ] **Step 1: Append the batched quantization kernel**

Open `native/kernels/quantize_x.cu` and append after the existing `quantize_x_to_q8_1` kernel (end of file, after its closing brace):

```cuda

// Batched variant of quantize_x_to_q8_1 (issue #349): quantizes ALL m activation
// rows (prefill tokens) of x[m, k] in ONE launch instead of m separate launches.
// Adds a second grid dimension (blockIdx.y = token) on top of the identical
// per-chunk math; output layout is the single-row layout's THREE SECTIONS
// concatenated by row (int8_t xq[m,k] | half dx[m,k/32] | half sx2[m,k/16]),
// each section itself row-major — NOT interleaved per-token blocks. This
// mirrors quantize_x_to_q8_1's own single-row layout, scaled by m; see
// CudaForwardState.PreQ8_1ScratchBytes(k) for the per-row byte count (the
// batched scratch is m * PreQ8_1ScratchBytes(k) bytes total).
extern "C" __global__ void __launch_bounds__(QX_THREADS) quantize_x_to_q8_1_batched(
    const half* __restrict__ x,     // [m, k] half, row stride k
    int8_t* __restrict__ xq,        // [m, k] int8, row stride k
    half*   __restrict__ dx,        // [m, k/32] half, row stride k/32
    half*   __restrict__ sx2,       // [m, k/16] half, row stride k/16
    const int k)
{
    const int num_chunks = k >> 5;          // k / 32
    const int warp_id = threadIdx.y;        // 0..QX_WARPS_PER_BLOCK-1
    const int lane    = threadIdx.x;        // 0..31
    const int chunk   = blockIdx.x * QX_WARPS_PER_BLOCK + warp_id;
    const int token   = blockIdx.y;
    if (chunk >= num_chunks) return;

    const half* x_row     = x   + (size_t)token * k;
    int8_t* xq_row         = xq  + (size_t)token * k;
    half*   dx_row          = dx  + (size_t)token * num_chunks;
    half*   sx2_row         = sx2 + (size_t)token * num_chunks * 2;

    const int idx = chunk * 32 + lane;
    float v = __half2float(x_row[idx]);
    float a = fabsf(v);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
    {
        float other = __shfl_xor_sync(0xFFFFFFFF, a, offset);
        a = fmaxf(a, other);
    }

    float inv_scale = (a > 0.0f) ? (127.0f / a) : 0.0f;
    int qi = __float2int_rn(v * inv_scale);
    qi = qi > 127 ? 127 : (qi < -127 ? -127 : qi);
    xq_row[idx] = (int8_t)qi;

    int s = qi;
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
        s += __shfl_xor_sync(0xFFFFFFFF, s, offset);

    if (lane == 0)
    {
        dx_row[chunk] = __float2half(a / 127.0f);
        sx2_row[chunk * 2 + 0] = __float2half((float)s);
    }
    if (lane == 16)
    {
        sx2_row[chunk * 2 + 1] = __float2half((float)s);
    }
}
```

- [ ] **Step 2: Regenerate PTX**

Run (Windows — from the repo root, NOT `native/build.ps1`, see Global Constraints for why):

```
native\build_ptx.bat
```

Expected: a success line for `quantize_x.cu -> quantize_x.ptx` with no `FAIL` marker.

- [ ] **Step 3: Verify the new symbol is present**

```powershell
Select-String -Path native/ptx/quantize_x.ptx -Pattern "quantize_x_to_q8_1_batched"
```

Expected: at least one match.

- [ ] **Step 4: Commit**

```bash
git add native/kernels/quantize_x.cu native/ptx/quantize_x.ptx
git commit -m "feat(cuda): add row-aware batched Q8_1 input quantization kernel (#349)"
```

---

### Task 3: C# scratch buffer + launcher wiring

**Files:**
- Modify: `src/DotLLM.Cuda/CudaForwardState.cs`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`

**Interfaces:**
- Consumes: `quantized_gemv_q4_k_mmq_batched` and `quantize_x_to_q8_1_batched` PTX symbols from Tasks 1–2; `CudaForwardState.PreQ8_1ScratchBytes(int k)` (existing static method, `CudaForwardState.cs:259`).
- Produces:
  - `CudaForwardState.PreQ8_1BatchedScratch` (`nint` field) — grows with `EnsureCapacity(seqLen)`, freed in `FreeSequenceBuffers`.
  - `CudaKernels.HasMmqBatchedQ4K` (`bool` property).
  - `CudaKernels.DisableMmqBatchedQ4K` (`static bool` property, env `DOTLLM_DISABLE_MMQ_BATCHED_Q4K`).
  - `CudaKernels.MmqBatchedMinSeqLen` (`static int` property, env `DOTLLM_MMQ_BATCHED_MIN_SEQLEN`, default `8`).
  - `CudaKernels.LaunchQuantizeXToQ8_1Batched(nint x, nint scratch, int k, int m, nint stream)`.
  - `CudaKernels.LaunchQuantizedGemvMmqBatchedQ4K(nint quantWeight, nint preqScratch, nint y, int n, int k, int m, nint stream)`.
  All four consumed by Task 4's test and Task 5's dispatcher gate.

- [ ] **Step 1: Add the batched scratch buffer to `CudaForwardState`**

In `src/DotLLM.Cuda/CudaForwardState.cs`, add a new field right after `PreQ8_1ScratchK` (after line 85):

```csharp
    public nint PreQ8_1Scratch;
    public int  PreQ8_1ScratchK;        // capacity in elements (must be a multiple of 32)

    // Batched pre-Q8_1 input-quantization scratch for prefill (issue #349): holds ALL
    // seqLen activation rows quantized in one shot, feeding the batched-MMQ prefill
    // kernel. Grows with EnsureCapacity like the other per-token buffers below (unlike
    // PreQ8_1Scratch, which is a fixed one-row buffer sized once in the constructor).
    // Per-row byte count uses the same PreQ8_1ScratchBytes(k) formula as the decode
    // scratch, with k = PreQ8_1ScratchK (the largest GEMV input dim across call sites).
    public nint PreQ8_1BatchedScratch;
```

Then in `EnsureCapacity` (`CudaForwardState.cs`), add a line right after the existing `LoraTmp = AllocDevice(...)` line (around line 188):

```csharp
        LoraTmp = AllocDevice((long)newCapacity * MaxLoraRank * half);
        PreQ8_1BatchedScratch = AllocDevice((long)newCapacity * PreQ8_1ScratchBytes(PreQ8_1ScratchK));
```

Then in `FreeSequenceBuffers` (`CudaForwardState.cs`), add a line right after `FreeIfNonZero(ref LoraTmp);` (around line 234):

```csharp
        FreeIfNonZero(ref LoraTmp);
        FreeIfNonZero(ref PreQ8_1BatchedScratch);
```

- [ ] **Step 2: Verify `CudaForwardState.cs` builds**

```powershell
dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj -c Release
```

Expected: build succeeds (no compile errors). This is a pure field-addition change — no behavior to test yet.

- [ ] **Step 3: Add the two new PTX function fields to `CudaKernels`**

In `src/DotLLM.Cuda/CudaKernels.cs`, add a field right after `_quantizeXToQ8_1Func` (line 99):

```csharp
    private readonly nint _quantizeXToQ8_1Func;
    private readonly nint _quantizeXToQ8_1BatchedFunc;
```

And add a field right after `_quantizedGemvIQ4_XSMmvqLargePreqFunc` (line 119, immediately before the `_maxDynamicSharedBytesOptIn` doc comment):

```csharp
    private readonly nint _quantizedGemvIQ4_XSMmvqLargePreqFunc;
    // Batched-M dp4a MMQ prefill kernel (issue #349) — amortizes weight reads across
    // MmqBatchedMinSeqLen prefill tokens per block via L1/L2 cache reuse (see
    // native/kernels/quantized_gemv_mmq.cu for the kernel body). Q4_K only for this PoC.
    private readonly nint _quantizedGemvQ4_KMmqBatchedFunc;
```

- [ ] **Step 4: Load the two new symbols in the constructor**

Inside the existing `if (File.Exists(mmqPath))` block, add a line right after `_quantizedGemvIQ4_XSMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_xs_mmvq_large_preq");` (line 582):

```csharp
            _quantizedGemvIQ4_XSMmvqLargePreqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_iq4_xs_mmvq_large_preq");
            // Batched-M prefill kernel (issue #349) — TryGetFunction so a stale PTX
            // without the new symbol still loads; HasMmqBatchedQ4K reports false.
            _quantizedGemvQ4_KMmqBatchedFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q4_k_mmq_batched");
```

Inside the existing `if (File.Exists(quantizeXPath))` block, add a line right after `_quantizeXToQ8_1Func = _quantizeXModule.GetFunction("quantize_x_to_q8_1");` (line 614):

```csharp
            _quantizeXToQ8_1Func = _quantizeXModule.GetFunction("quantize_x_to_q8_1");
            _quantizeXToQ8_1BatchedFunc = _quantizeXModule.TryGetFunction("quantize_x_to_q8_1_batched");
```

- [ ] **Step 5: Add `HasMmqBatchedQ4K`, `DisableMmqBatchedQ4K`, `MmqBatchedMinSeqLen`**

Insert right after the `HasPreQ8_1` property (`CudaKernels.cs`, after line 4311, before `DisableMmqQ2K`):

```csharp
    /// <summary>
    /// True when the batched-M dp4a MMQ Q4_K prefill kernel AND its batched input-quantization
    /// companion are both loaded (issue #349 proof of concept — Q4_K only). Gates the prefill
    /// dispatcher in <c>CudaTransformerModel.Project</c>: when true (and <see cref="MmqBatchedMinSeqLen"/>
    /// is met), prefill skips dequant→cuBLAS HGEMM entirely for Q4_K weights.
    /// </summary>
    public bool HasMmqBatchedQ4K => _quantizedGemvQ4_KMmqBatchedFunc != 0 && _quantizeXToQ8_1BatchedFunc != 0
        && !DisableQuantizedGemv && !DisableMmqBatchedQ4K;

```

Insert right after `DisablePreQ8_1` (`CudaKernels.cs`, after line 4353, before `HasMmq(QuantizationType qt)`):

```csharp
    /// <summary>Test/benchmark hook to force the dequant→cuBLAS prefill fallback even when the
    /// batched-MMQ Q4_K kernel is loaded. Override: <c>DOTLLM_DISABLE_MMQ_BATCHED_Q4K=1</c>.</summary>
    public static bool DisableMmqBatchedQ4K { get; set; } =
        Environment.GetEnvironmentVariable("DOTLLM_DISABLE_MMQ_BATCHED_Q4K") == "1";

    /// <summary>
    /// Minimum prefill seqLen (inclusive) at which the batched-MMQ Q4_K kernel is preferred over
    /// the dequant→cuBLAS HGEMM fallback — mirrors llama.cpp's MMVQ_MAX_BATCH_SIZE /
    /// MMQ_DP4A_MAX_BATCH_SIZE crossover gating (docs/perf/MMA_BATCHED_MMQ.md §1d). Default set
    /// from the Task 6 benchmark sweep in
    /// docs/superpowers/plans/2026-08-12-cuda-prefill-batched-mmq.md; override with
    /// <c>DOTLLM_MMQ_BATCHED_MIN_SEQLEN</c> for A/B comparison.
    /// </summary>
    public static int MmqBatchedMinSeqLen { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_MMQ_BATCHED_MIN_SEQLEN"), out int v) ? v : 8;

```

- [ ] **Step 6: Add the two launcher methods**

Insert right after `LaunchQuantizedGemvMmq`'s closing brace (`CudaKernels.cs`, after line 4645, before the `CheckDynamicSharedBudget(uint dynShmem, QuantizationType qt, int k)` overload):

```csharp
    /// <summary>
    /// Batched pre-Q8_1 input quantization for prefill (issue #349). Quantizes
    /// <paramref name="x"/>[m, k] (row-major, row stride k) to INT8 in ONE launch covering all
    /// m activation rows, instead of m separate <see cref="LaunchQuantizeXToQ8_1"/> calls.
    /// Scratch layout is the single-row layout's sections concatenated by row (NOT interleaved
    /// per-row blocks): <c>int8_t xq[m,k] | half dx[m,k/32] | half sx2[m,k/16]</c>, each section
    /// itself row-major with row stride k, k/32, k/16 respectively. Size the scratch as
    /// <c>m * CudaForwardState.PreQ8_1ScratchBytes(k)</c> bytes. Consumed by
    /// <see cref="LaunchQuantizedGemvMmqBatchedQ4K"/>.
    /// </summary>
    public void LaunchQuantizeXToQ8_1Batched(nint x, nint scratch, int k, int m, nint stream)
    {
        if (_quantizeXToQ8_1BatchedFunc == 0)
            throw new InvalidOperationException(
                "Batched pre-Q8_1 quantization kernel not available. Compile native/kernels/quantize_x.cu to PTX.");
        if ((k & 31) != 0)
            throw new ArgumentException($"k must be a multiple of 32 (got {k}).", nameof(k));

        int numChunks = k >> 5;
        nint xqPtr  = scratch;
        nint dxPtr  = scratch + (nint)((long)m * k);
        nint sx2Ptr = dxPtr + (nint)((long)m * numChunks * 2);

        nint xArg = x, xqArg = xqPtr, dxArg = dxPtr, sx2Arg = sx2Ptr;
        int kArg = k;
        void** args = stackalloc void*[] { &xArg, &xqArg, &dxArg, &sx2Arg, &kArg };

        // Must mirror QX_THREADS_X / QX_WARPS_PER_BLOCK in quantize_x.cu (32 × 8 = 256).
        const uint QxThreadsX = 32;
        const uint QxWarpsPerBlock = 8;
        uint gridX = (uint)((numChunks + QxWarpsPerBlock - 1) / QxWarpsPerBlock);
        CudaDriverApi.cuLaunchKernel(_quantizeXToQ8_1BatchedFunc,
                gridX, (uint)m, 1, QxThreadsX, QxWarpsPerBlock, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

    /// <summary>
    /// Batched-M dp4a MMQ prefill GEMV for Q4_K (issue #349 proof of concept). Computes
    /// <c>Y[m, n] = X[m, k] × W[n, k]^T</c> for Q4_K-quantized W in a single launch — replacing
    /// the dequant→cuBLAS HGEMM prefill fallback. Requires <paramref name="preqScratch"/> to
    /// already hold the output of <see cref="LaunchQuantizeXToQ8_1Batched"/> for the same x/k/m.
    /// Gate calls with <see cref="HasMmqBatchedQ4K"/>.
    /// </summary>
    public void LaunchQuantizedGemvMmqBatchedQ4K(nint quantWeight, nint preqScratch,
                                                   nint y, int n, int k, int m, nint stream)
    {
        if (_quantizedGemvQ4_KMmqBatchedFunc == 0)
            throw new InvalidOperationException(
                "Batched MMQ GEMV kernel not available. Compile native/kernels/quantized_gemv_mmq.cu to PTX.");

        int numChunks = k >> 5;
        nint xqPtr  = preqScratch;
        nint dxPtr  = preqScratch + (nint)((long)m * k);
        nint sx2Ptr = dxPtr + (nint)((long)m * numChunks * 2);

        nint wArg = quantWeight, xqArg = xqPtr, dxArg = dxPtr, sx2Arg = sx2Ptr, yArg = y;
        int nArg = n, kArg = k, mArg = m;
        void** args = stackalloc void*[] { &wArg, &xqArg, &dxArg, &sx2Arg, &yArg, &nArg, &kArg, &mArg };

        // Must mirror MMQ_ROWS_PER_BLOCK / MMQ_BATCH_M_TILE in quantized_gemv_mmq.cu.
        const int MmqRowsPerBlock = 4;
        const int MmqBatchMTile = 2;
        uint gridX = (uint)((n + MmqRowsPerBlock - 1) / MmqRowsPerBlock);
        uint gridY = (uint)((m + MmqBatchMTile - 1) / MmqBatchMTile);

        // No dynamic shmem — pure register + warp-shfl accumulation, no budget check needed.
        CudaDriverApi.cuLaunchKernel(_quantizedGemvQ4_KMmqBatchedFunc,
                gridX, gridY, 1, BlockSize, 1, 1,
                0, stream, (nint)args, 0).ThrowOnError();
    }

```

- [ ] **Step 7: Build to verify**

```powershell
dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj -c Release
```

Expected: build succeeds with no errors (unused-symbol warnings for the not-yet-called launchers are fine — Task 4/5 consume them next).

- [ ] **Step 8: Commit**

```bash
git add src/DotLLM.Cuda/CudaForwardState.cs src/DotLLM.Cuda/CudaKernels.cs
git commit -m "feat(cuda): wire batched-MMQ Q4_K prefill scratch + launchers (#349)"
```

---

### Task 4: Discriminating correctness test

**Files:**
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs`

**Interfaces:**
- Consumes: `CudaKernels.HasMmqBatchedQ4K`, `LaunchQuantizeXToQ8_1`, `LaunchQuantizedGemvMmq` (existing, single-row, used as the oracle), `LaunchQuantizeXToQ8_1Batched`, `LaunchQuantizedGemvMmqBatchedQ4K` (new, from Task 3), `CudaForwardState.PreQ8_1ScratchBytes(int k)` (existing static method), `SynthesiseQ4KBlock` (existing private helper in this file, `CudaMmqKernelTests.cs:566`).
- Produces: `MmqBatchedQ4K_MatchesLoopOfM1_WithinTolerance` xUnit theory — the discriminating parity gate Task 5's dispatcher wiring depends on.

- [ ] **Step 1: Write the failing test**

Insert a new region into `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs` right after `SynthesiseIQ4_XSBlock` (ends at line 634) and before `FindPtxDir` (line 636):

```csharp
    // ── Batched-M prefill equivalence tests (issue #349) ─────────────────────
    // Oracle: loop of M=1 launches via the already-validated single-row preq path
    // (quantize_x_to_q8_1 + quantized_gemv_q4_k_mmq_preq), one token at a time.
    // Under test: ONE quantize_x_to_q8_1_batched launch + ONE
    // quantized_gemv_q4_k_mmq_batched launch covering all m tokens. Reusing the
    // preq kernel as ground truth (rather than re-deriving Q4_K math a third
    // time) mirrors RunPreqEquivalence's structure.

    [SkippableTheory]
    [InlineData(2, 4, 256)]      // MMQ_BATCH_M_TILE boundary: m=2 exactly fills one M-tile
    [InlineData(3, 8, 512)]      // odd m: second M-tile is partially out of range
    [InlineData(8, 64, 1024)]    // larger batch, moderate n/k
    [InlineData(32, 4096, 4096)] // Qwen3-8B-class shape at a real prefill-length batch
    public void MmqBatchedQ4K_MatchesLoopOfM1_WithinTolerance(int m, int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqBatchedEquivalence(m, n, k);
    }

    private unsafe void RunMmqBatchedEquivalence(int m, int n, int k, float peakRelativeTolerance = 0.01f)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasMmq(QuantizationType.Q4_K), "MMQ kernel for Q4_K not loaded (PTX may be stale)");
        Skip.IfNot(kernels.HasMmqBatchedQ4K, "Batched-MMQ Q4_K kernel not loaded (PTX may be stale)");

        var rng = new Random(9012 ^ m ^ n ^ k);
        int superblocksPerRow = k / 256;
        int rowBytes = superblocksPerRow * 144;
        long weightBytes = (long)n * rowBytes;

        byte[] hostWeight = new byte[weightBytes];
        var weightSpan = hostWeight.AsSpan();
        for (int row = 0; row < n; row++)
        for (int sb = 0; sb < superblocksPerRow; sb++)
            SynthesiseQ4KBlock(rng, weightSpan.Slice(row * rowBytes + sb * 144, 144));

        Half[] hostX = new Half[m * k];
        for (int i = 0; i < hostX.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            hostX[i] = (Half)(g * 0.4);
        }

        long xBytes = (long)m * k * sizeof(ushort);
        long yBytes = (long)m * n * sizeof(ushort);
        long batchedScratchBytes = (long)m * CudaForwardState.PreQ8_1ScratchBytes(k);
        long singleScratchBytes = CudaForwardState.PreQ8_1ScratchBytes(k);

        nint devW = 0, devX = 0, devYLoop = 0, devYBatched = 0, devScratchBatched = 0, devScratchSingle = 0;
        Half[] yLoop = new Half[m * n];
        Half[] yBatched = new Half[m * n];

        try
        {
            CudaDriverApi.cuMemAlloc_v2(out devW, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devYLoop, (nuint)yBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devYBatched, (nuint)yBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devScratchBatched, (nuint)batchedScratchBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devScratchSingle, (nuint)singleScratchBytes).ThrowOnError();

            fixed (byte* pW = hostWeight)
                CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)pW, (nuint)weightBytes).ThrowOnError();
            fixed (Half* pX = hostX)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();

            // Oracle: one M=1 launch pair per token, via the already-validated preq path.
            for (int token = 0; token < m; token++)
            {
                nint xRow = devX + (nint)((long)token * k * sizeof(ushort));
                nint yRow = devYLoop + (nint)((long)token * n * sizeof(ushort));
                kernels.LaunchQuantizeXToQ8_1(xRow, devScratchSingle, k, stream.Handle);
                kernels.LaunchQuantizedGemvMmq(devW, QuantizationType.Q4_K, xRow, yRow, n, k, devScratchSingle, stream.Handle);
            }

            // Under test: one batched-quantize + one batched-MMQ launch for all m tokens.
            kernels.LaunchQuantizeXToQ8_1Batched(devX, devScratchBatched, k, m, stream.Handle);
            kernels.LaunchQuantizedGemvMmqBatchedQ4K(devW, devScratchBatched, devYBatched, n, k, m, stream.Handle);

            stream.Synchronize();

            fixed (Half* pY = yLoop)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYLoop, (nuint)yBytes).ThrowOnError();
            fixed (Half* pY = yBatched)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devYBatched, (nuint)yBytes).ThrowOnError();
        }
        finally
        {
            if (devW != 0) CudaDriverApi.cuMemFree_v2(devW);
            if (devX != 0) CudaDriverApi.cuMemFree_v2(devX);
            if (devYLoop != 0) CudaDriverApi.cuMemFree_v2(devYLoop);
            if (devYBatched != 0) CudaDriverApi.cuMemFree_v2(devYBatched);
            if (devScratchBatched != 0) CudaDriverApi.cuMemFree_v2(devScratchBatched);
            if (devScratchSingle != 0) CudaDriverApi.cuMemFree_v2(devScratchSingle);
        }

        float maxAbs = 0f, refMax = 0f;
        double sumAbs = 0.0;
        int total = m * n;
        for (int i = 0; i < total; i++)
        {
            float a = (float)yLoop[i];
            float b = (float)yBatched[i];
            float diff = MathF.Abs(a - b);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
            if (MathF.Abs(a) > refMax) refMax = MathF.Abs(a);
        }
        float meanAbs = (float)(sumAbs / total);
        float peakRelMax = refMax > 0 ? maxAbs / refMax : 0f;
        float peakRelMean = refMax > 0 ? meanAbs / refMax : 0f;

        _out.WriteLine($"BATCHED Q4_K m={m} n={n} k={k}: ref|max|={refMax:F3} |batched-loop| max={maxAbs:F4} mean={meanAbs:F4} peak-rel max={peakRelMax:P3} mean={peakRelMean:P3}");

        Assert.True(peakRelMax < peakRelativeTolerance,
            $"Peak-relative max diff {peakRelMax:P3} exceeds {peakRelativeTolerance:P1} (max={maxAbs}, refMax={refMax})");
    }
```

- [ ] **Step 2: Run the test to confirm it compiles and either passes or (without a GPU) skips cleanly**

```powershell
dotnet test tests/DotLLM.Tests.Unit/ --filter "FullyQualifiedName~MmqBatchedQ4K_MatchesLoopOfM1_WithinTolerance"
```

Expected on a machine WITHOUT a CUDA GPU: all 4 cases report `Skipped` with reason `"No CUDA GPU available"` (proves the test compiles and the skip path works). Expected on a machine WITH a CUDA GPU (e.g. the RTX 3060 T1 tier): all 4 cases `Passed`, with `_out.WriteLine` peak-relative drift under 1% printed for each case. If any case reports `Skipped: "Batched-MMQ Q4_K kernel not loaded (PTX may be stale)"` on a GPU machine, re-run Task 1/2 Step 2 (`native\build_ptx.bat`) — the PTX in `native/ptx/` is stale relative to the `.cu` source.

- [ ] **Step 3: Commit**

```bash
git add tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs
git commit -m "test(cuda): add batched-MMQ Q4_K prefill parity test vs loop-of-M=1 oracle (#349)"
```

---

### Task 5: Dispatcher gate — route real prefill traffic through the batched kernel

**Files:**
- Modify: `src/DotLLM.Cuda/CudaTransformerModel.cs`

**Interfaces:**
- Consumes: `CudaKernels.HasMmqBatchedQ4K`, `CudaKernels.MmqBatchedMinSeqLen`, `CudaKernels.LaunchQuantizeXToQ8_1Batched`, `CudaKernels.LaunchQuantizedGemvMmqBatchedQ4K` (all from Task 3), `CudaForwardState.PreQ8_1BatchedScratch` / `PreQ8_1ScratchK` (from Task 3 and existing code).
- Produces: updated `Project` method — every prefill call site (Q, K, V, O, Gate, Up, Down, LmHead — all flow through this single method per the research citation that fused QKV/GateUp decode shortcuts are hard-gated to `seqLen == 1`) now conditionally routes Q4_K through the batched kernel.

- [ ] **Step 1: Replace the prefill branch in `Project`**

In `src/DotLLM.Cuda/CudaTransformerModel.cs`, update the doc comment above `Project` (currently lines 3123–3127):

Old:
```csharp
    /// <summary>
    /// Dispatches projection as cuBLAS HGEMM (prefill) or quantized/cuBLAS GEMV (decode).
    /// For quantized weights with no persistent FP16 copy (<paramref name="fp16Weight"/> == 0),
    /// dequantizes on-the-fly into <see cref="CudaForwardState.DequantScratch"/> before calling cuBLAS.
    /// </summary>
```

New:
```csharp
    /// <summary>
    /// Dispatches projection as batched-MMQ / cuBLAS HGEMM (prefill) or quantized/cuBLAS GEMV
    /// (decode). Prefill (seqLen &gt; 1) uses the dp4a-batched-MMQ kernel for Q4_K weights once
    /// seqLen clears <see cref="CudaKernels.MmqBatchedMinSeqLen"/> (issue #349); otherwise — and
    /// for every other quant type — falls back to dequant→cuBLAS HGEMM. For quantized weights with
    /// no persistent FP16 copy (<paramref name="fp16Weight"/> == 0), the fallback dequantizes
    /// on-the-fly into <see cref="CudaForwardState.DequantScratch"/> before calling cuBLAS.
    /// </summary>
```

Then replace the `if (seqLen > 1)` branch (currently lines 3133–3153):

Old:
```csharp
        if (seqLen > 1) // Prefill: cuBLAS HGEMM
        {
            nint w = fp16Weight;
            if (w == 0)
            {
                // Quantized: dequant into scratch, then GEMM
                if (qt == QuantizationType.I2_S && inputDim % I2SBlockSize128 != 0)
                    // Ragged K (issue #206): the aligned dequant kernel's blocks_per_row=k/128
                    // integer division would silently drop each row's tail elements.
                    _kernels.LaunchDequantI2_SToF16Ragged(quantWeight, _state.DequantScratch, outputDim, inputDim, s);
                else if (qt == QuantizationType.I2_S)
                    _kernels.LaunchDequantI2_SToF16(quantWeight, _state.DequantScratch, outputDim, inputDim, s);
                else if (qt == QuantizationType.PQ2_0)
                    _kernels.LaunchDequantPQ2_0ToF16(quantWeight, _state.DequantScratch, outputDim, inputDim, s);
                else
                    _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                        outputDim * inputDim, s);
                w = _state.DequantScratch;
            }
            CudaGemm.LinearF16(_cublas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
        }
```

New:
```csharp
        if (seqLen > 1) // Prefill
        {
            // dp4a-batched-MMQ prefill (issue #349 PoC): skip dequant+cuBLAS entirely for Q4_K
            // once seqLen clears the crossover gate (mirrors llama.cpp's MMVQ_MAX_BATCH_SIZE /
            // MMQ_DP4A_MAX_BATCH_SIZE-style batch-size dispatch — docs/perf/MMA_BATCHED_MMQ.md §1d).
            if (quantWeight != 0 && qt == QuantizationType.Q4_K
                && seqLen >= CudaKernels.MmqBatchedMinSeqLen
                && (inputDim & 31) == 0 && inputDim <= _state.PreQ8_1ScratchK
                && _kernels.HasMmqBatchedQ4K)
            {
                _kernels.LaunchQuantizeXToQ8_1Batched(input, _state.PreQ8_1BatchedScratch, inputDim, seqLen, s);
                _kernels.LaunchQuantizedGemvMmqBatchedQ4K(quantWeight, _state.PreQ8_1BatchedScratch,
                    output, outputDim, inputDim, seqLen, s);
            }
            else // Fallback: dequant into scratch, then cuBLAS HGEMM
            {
                nint w = fp16Weight;
                if (w == 0)
                {
                    // Quantized: dequant into scratch, then GEMM
                    if (qt == QuantizationType.I2_S && inputDim % I2SBlockSize128 != 0)
                        // Ragged K (issue #206): the aligned dequant kernel's blocks_per_row=k/128
                        // integer division would silently drop each row's tail elements.
                        _kernels.LaunchDequantI2_SToF16Ragged(quantWeight, _state.DequantScratch, outputDim, inputDim, s);
                    else if (qt == QuantizationType.I2_S)
                        _kernels.LaunchDequantI2_SToF16(quantWeight, _state.DequantScratch, outputDim, inputDim, s);
                    else if (qt == QuantizationType.PQ2_0)
                        _kernels.LaunchDequantPQ2_0ToF16(quantWeight, _state.DequantScratch, outputDim, inputDim, s);
                    else
                        _kernels.LaunchDequantToF16(quantWeight, qt, _state.DequantScratch,
                            outputDim * inputDim, s);
                    w = _state.DequantScratch;
                }
                CudaGemm.LinearF16(_cublas.Handle, input, w, output, seqLen, inputDim, outputDim, s);
            }
        }
```

- [ ] **Step 2: Build**

```powershell
dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj -c Release
```

Expected: build succeeds.

- [ ] **Step 3: Run the full MMQ kernel test suite (regression check) plus the new batched test**

```powershell
dotnet test tests/DotLLM.Tests.Unit/ --filter "FullyQualifiedName~CudaMmqKernelTests"
```

Expected: all existing `CudaMmqKernelTests` cases still pass/skip exactly as before (this task did not touch any of those kernels or launchers), and the Task 4 `MmqBatchedQ4K_MatchesLoopOfM1_WithinTolerance` cases pass/skip as in Task 4 Step 2.

- [ ] **Step 4: Run the real-model end-to-end regression gate (best-effort — requires GPU + fixture)**

```powershell
dotnet test tests/DotLLM.Tests.Unit/ --filter "FullyQualifiedName~SmolLM135M_Q4KM_LogitsMatchPyTorchReference_Cuda"
```

This is `CudaLogitsMatchPyTorchReferenceTests`'s real Q4_K_M GGUF end-to-end forward-pass parity gate (see `tests/DotLLM.Tests.Unit/Cuda/CudaLogitsMatchPyTorchReferenceTests.cs`) — it exercises the FULL prompt forward (prefill), so it is the most direct check that the new dispatcher branch didn't regress real-model output. Expected: `Passed` if the CUDA GPU and the SmolLM-135M-GGUF fixture (`docs/QUANT_FIXTURES.md`) are both present on this machine; `Skipped` otherwise (not a failure — note this in the task's completion report either way).

- [ ] **Step 5: Commit**

```bash
git add src/DotLLM.Cuda/CudaTransformerModel.cs
git commit -m "feat(cuda): route Q4_K prefill through batched-MMQ dispatcher gate (#349)"
```

---

### Task 6: Before/after benchmark and crossover threshold tuning

**Files:**
- Modify: `src/DotLLM.Cuda/CudaKernels.cs` (only the `MmqBatchedMinSeqLen` default literal, if the measured crossover differs from the interim default of `8`)
- No other files created — results are recorded in a `benchmarks/perf-matrix/results.csv` row (existing convention, per `BenchCommand.cs`'s doc comment) and in Task 7's `docs/CUDA.md` update.

**Interfaces:**
- Consumes: the `bench` CLI command (`src/DotLLM.Cli/Commands/BenchCommand.cs`), `CudaKernels.DisableMmqBatchedQ4K` (env `DOTLLM_DISABLE_MMQ_BATCHED_Q4K`, from Task 3) as the A/B toggle.
- Produces: measured prefill tok/s before/after, and a justified final value for `CudaKernels.MmqBatchedMinSeqLen`'s default.

- [ ] **Step 1: Identify a locally available Q4_K_M GGUF model**

```powershell
huggingface-cli scan-cache
```

Look for any cached repo with a `Q4_K_M` (or `Q4_K_S`) GGUF quant — e.g. an existing SmolLM/Qwen2.5/TinyLlama GGUF already used by the project's other CUDA tests (check `docs/QUANT_FIXTURES.md` and `E:\.cache\huggingface\hub` for what's already resolvable). If nothing is cached, use a small public GGUF repo directly with the `bench` command below — the `Model` argument accepts a HuggingFace repo ID and resolves/downloads it automatically (per `BenchCommand.Settings.Model`'s description); `QuantFactory/SmolLM-135M-GGUF` (already referenced throughout the CUDA test suite) with `--quant Q4_K_M` is a safe default if no larger cached model is available. Do not copy any GGUF into the working tree — the `bench` command resolves it into the existing HF cache / `~/.dotllm` model cache per CLAUDE.md's Model & Fixture Storage Rules.

- [ ] **Step 2: Baseline sweep (batched path forced OFF)**

Run at several prompt lengths spanning below/above the interim `MmqBatchedMinSeqLen=8` default, and well past it to characterize the large-seqLen regime:

```powershell
$env:DOTLLM_DISABLE_MMQ_BATCHED_Q4K = "1"
foreach ($p in 4, 8, 16, 32, 64, 128, 256, 512) {
    dotnet run --project src/DotLLM.Cli -c Release -- bench <model-or-repo> --device cuda -p $p -n 32 -r 5 --quant Q4_K_M
}
Remove-Item Env:\DOTLLM_DISABLE_MMQ_BATCHED_Q4K
```

Record the reported prefill tokens/sec (median + best) for each `-p` value.

- [ ] **Step 3: New-path sweep (batched path enabled — the default)**

```powershell
foreach ($p in 4, 8, 16, 32, 64, 128, 256, 512) {
    dotnet run --project src/DotLLM.Cli -c Release -- bench <model-or-repo> --device cuda -p $p -n 32 -r 5 --quant Q4_K_M
}
```

Record the reported prefill tokens/sec (median + best) for each `-p` value.

- [ ] **Step 4: Determine the crossover and finalize `MmqBatchedMinSeqLen`**

Compare Step 2 vs Step 3 prefill tok/s at each `-p`. Identify the smallest `-p` at which the batched path is consistently (median AND best) faster than the dequant→cuBLAS baseline. If that smallest favorable `-p` differs materially from the interim default of `8`, update the literal in `CudaKernels.cs`:

```csharp
    public static int MmqBatchedMinSeqLen { get; set; } =
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_MMQ_BATCHED_MIN_SEQLEN"), out int v) ? v : 8;
```

Replace the trailing `8` with the measured crossover value, and update the property's doc comment to state the measured number and the date. If any `-p` value shows the batched path regressing relative to baseline even above the chosen threshold (e.g. due to L2-cache-thrash at very large `n*k` on the RTX 3060's 3 MB L2), note that as a known limitation in the same doc comment rather than silently shipping a regression range.

- [ ] **Step 5: Rebuild and re-run the parity test to confirm the threshold change didn't break anything**

```powershell
dotnet build src/DotLLM.Cuda/DotLLM.Cuda.csproj -c Release
dotnet test tests/DotLLM.Tests.Unit/ --filter "FullyQualifiedName~MmqBatchedQ4K_MatchesLoopOfM1_WithinTolerance"
```

Expected: same pass/skip result as Task 4 Step 2 (the threshold change only affects `CudaTransformerModel.Project`'s dispatch, not the kernels or the test, which calls the launchers directly).

- [ ] **Step 6: Record results**

Append the `bench` command's printed `benchmarks/perf-matrix/results.csv` row for the representative before/after runs (per `BenchCommand`'s own doc comment: "prints a ready-to-paste `benchmarks/perf-matrix/results.csv` row") to that file for both the baseline and new-path measurements at the model/quant/prompt-length used.

- [ ] **Step 7: Commit**

```bash
git add src/DotLLM.Cuda/CudaKernels.cs benchmarks/perf-matrix/results.csv
git commit -m "perf(cuda): tune batched-MMQ Q4_K prefill crossover threshold from measured RTX 3060 data (#349)"
```

(If Step 4 concluded no change to the interim default was warranted, skip staging `CudaKernels.cs` in this commit and commit only the results CSV, or fold this step into Task 7's commit.)

---

### Task 7: Update `docs/CUDA.md`

**Files:**
- Modify: `docs/CUDA.md`

**Interfaces:**
- Consumes: the measured results from Task 6, the final `MmqBatchedMinSeqLen` value.
- Produces: an accurate `docs/CUDA.md` that no longer claims prefill is dequant→cuBLAS-only for every quant type.

- [ ] **Step 1: Add a kernel-table row**

In the "Quantized GEMV (decode path...)" table (`docs/CUDA.md`, around line 643–657), add a new row after the existing `Q{4,5,6}_K MMQ/MMVQ _preq` row (line 655):

Old:
```
| Q{4,5,6}_K MMQ/MMVQ `_preq` | `quantized_gemv_mmq.cu` | `*_preq` suffix | Read pre-quantized x from scratch; skips Stage 1 input quant |
| Pre-Q8_1 input quant | `quantize_x.cu` | `quantize_x_to_q8_1` | Quantizes activation x[k] once per fused-GEMV-input — feeds the `_preq` variants. Auto-engages for k≥1024 |
```

New:
```
| Q{4,5,6}_K MMQ/MMVQ `_preq` | `quantized_gemv_mmq.cu` | `*_preq` suffix | Read pre-quantized x from scratch; skips Stage 1 input quant |
| Q4_K batched-MMQ (prefill) | `quantized_gemv_mmq.cu` | `quantized_gemv_q4_k_mmq_batched` | dp4a, MMQ_ROWS_PER_BLOCK(4) × MMQ_BATCH_M_TILE(2) warps/block; genuine quantized-GEMM prefill path (issue #349), no dequant. Gated by seqLen ≥ `CudaKernels.MmqBatchedMinSeqLen` |
| Pre-Q8_1 input quant | `quantize_x.cu` | `quantize_x_to_q8_1` | Quantizes activation x[k] once per fused-GEMV-input — feeds the `_preq` variants. Auto-engages for k≥1024 |
| Pre-Q8_1 input quant (batched) | `quantize_x.cu` | `quantize_x_to_q8_1_batched` | Quantizes all seqLen activation rows in one 2D-grid launch — feeds the Q4_K batched-MMQ prefill kernel |
```

- [ ] **Step 2: Correct the blanket cuBLAS-only prefill statement**

Around `docs/CUDA.md:676`:

Old:
```
GEMM/GEMV for prefill use cuBLAS (`cublasHgemm` / `cublasGemmEx`) directly — no custom PTX kernel needed.
```

New:
```
GEMM/GEMV for prefill use cuBLAS (`cublasHgemm` / `cublasGemmEx`) directly for most quant types. Q4_K is the
exception (issue #349): prefill routes through a custom dp4a-batched-MMQ PTX kernel
(`quantized_gemv_q4_k_mmq_batched`) once seqLen clears `CudaKernels.MmqBatchedMinSeqLen`, falling back to
cuBLAS below that threshold or for any other quant type.
```

- [ ] **Step 3: Update the future-work entry (append a DONE note, matching the file's existing convention of appending completion notes to a bullet rather than deleting it — see the BitNet decode CUDA-graph bullet at line 1015 for the established style)**

Around `docs/CUDA.md:1020`, replace:

Old:
```
- **Fused quantized GEMM for prefill**: Marlin-style dequant-in-register. Decode is now MMQ + MMVQ-large + pre-Q8_1 (Qwen3-8B Q4_K_M decode hits 33 tok/s eager on RTX 3060 — inside llama.cpp's reported range); prefill still uses dequant→cuBLAS HGEMM.
```

New (fill in the `<measured X>` / `<measured Y>` placeholders with Task 6's actual recorded numbers before committing — do not leave them as literal text):
```
- **Fused quantized GEMM for prefill.** Decode is MMQ + MMVQ-large + pre-Q8_1 (Qwen3-8B Q4_K_M decode hits 33 tok/s eager on RTX 3060 — inside llama.cpp's reported range). **Q4_K prefill DONE for the proof-of-concept tier (issue #349, 2026-08-12)**: a dp4a-batched-MMQ kernel (`quantized_gemv_q4_k_mmq_batched`, forked from the existing `quantized_gemv_q4_k_mmq_preq` dp4a primitive with an added M-tile — see `docs/perf/MMA_BATCHED_MMQ.md` §3a step 2 for the design rationale this issue follows) now handles Q4_K prefill directly from the quantized bytes once seqLen clears `CudaKernels.MmqBatchedMinSeqLen` (measured crossover: `<measured N>`; below that, or for any other quant type, prefill still falls back to dequant→cuBLAS HGEMM). Measured on the RTX 3060 T1 baseline: `<measured before tok/s>` → `<measured after tok/s>` prefill tok/s at p=`<measured p>` (see `benchmarks/perf-matrix/results.csv`). Marlin-style dequant-in-register and coverage beyond Q4_K (Q5_K, Q6_K, per the issue's own prioritization) remain open follow-ups.
```

- [ ] **Step 4: Verify the edits render correctly**

```powershell
Select-String -Path docs/CUDA.md -Pattern "quantized_gemv_q4_k_mmq_batched|MmqBatchedMinSeqLen" 
```

Expected: matches in all three edited locations (table row, line ~676 area, line ~1020 area).

- [ ] **Step 5: Commit**

```bash
git add docs/CUDA.md
git commit -m "docs(cuda): document batched-MMQ Q4_K prefill kernel, close out future-work line (#349)"
```

---

## Self-Review

**Spec coverage against issue #349's acceptance criteria:**
- "dp4a-batched-MMQ prefill kernel for at least Q4_K (proof of concept), with a discriminating parity test against the CPU reference" — Task 1 (kernel) + Task 4 (parity test). Note: the parity test uses the existing, already-validated single-row MMQ/preq kernel as its oracle (per the research report's explicit recommendation in §4) rather than re-deriving against the CPU scalar reference directly — the single-row MMQ kernel's own parity against the CPU reference (`DequantizeQ4_KScalar`) is already covered by the pre-existing `MmqQ4K_MatchesLegacyWithinTolerance`-style tests in the same file, which this plan does not duplicate. This satisfies the spirit of "discriminating parity test" (a test that fails if either kernel regresses) while avoiding redundant CPU-reference re-derivation.
- "Real-model prefill throughput measured before/after on this project's standard RTX 3060 baseline" — Task 6.
- "Dispatcher gate (mirrors llama.cpp's MMVQ_MAX_BATCH_SIZE) picking dp4a-MMQ vs the existing dequant→cuBLAS fallback by seqLen, so short prefills aren't penalized by kernel-launch overhead if the crossover isn't favorable there" — Task 5 (gate) + Task 6 (empirical crossover tuning).
- "docs/CUDA.md's 'Fused quantized GEMM for prefill' future-work line updated to point here once scoped/landed" — Task 7 Step 3.
- Issue scope note ("Prioritize by what's actually hot: Q4_K/Q5_K/Q6_K... first") — this plan intentionally covers Q4_K ONLY, matching the acceptance criteria's "at least Q4_K (proof of concept)" floor; Q5_K/Q6_K are explicitly called out as open follow-ups in Task 7 Step 3's doc update, not silently dropped.
- Issue scope note ("Leave the mma tensor-core variant and the continuous-batching-specific analysis in docs/perf/MMA_BATCHED_MMQ.md as-is") — no task touches that file; Task 7 only edits `docs/CUDA.md`.

**Placeholder scan:** All CUDA and C# code blocks in Tasks 1–5 are complete, compilable, verified-against-the-actual-current-file insertions (not sketches) — every insertion point was located via direct `Read`/`Grep` of the current repository state during planning, not derived from the research report's line numbers. Task 6/7 contain two intentional, clearly-labeled fill-in points (`<measured N>`, `<measured before/after tok/s>`) because those values are the output of Task 6's own benchmark run and cannot be known before that run executes — this is not a placeholder for un-designed work, it's the expected data-dependent last step of a benchmark task, explicitly flagged with "do not leave them as literal text."

**Type/signature consistency check:**
- `quantized_gemv_q4_k_mmq_batched(weight, xq_in, dx_in, sx2_in, y, n, k, m)` (Task 1) ↔ `LaunchQuantizedGemvMmqBatchedQ4K(quantWeight, preqScratch, y, n, k, m, stream)` (Task 3) — argument order and count match (`quantWeight`→`weight`, `preqScratch` split into `xqArg,dxArg,sx2Arg`→`xq_in,dx_in,sx2_in`, then `y,n,k,m`).
- `quantize_x_to_q8_1_batched(x, xq, dx, sx2, k)` with `blockIdx.y`=token (Task 2) ↔ `LaunchQuantizeXToQ8_1Batched(x, scratch, k, m, stream)` launching `gridX, (uint)m, 1, ...` (Task 3) — `m` maps to `gridDim.y`, matching the kernel's `blockIdx.y` = token read.
- Scratch layout formula `m * CudaForwardState.PreQ8_1ScratchBytes(k)` is used identically in: Task 3's `EnsureCapacity` allocation, Task 3's two launchers' `dxPtr`/`sx2Ptr` offset arithmetic, and Task 4's test's scratch sizing — all three derive the same three-section (xq/dx/sx2) layout scaled by `m`.
- `HasMmqBatchedQ4K`, `DisableMmqBatchedQ4K`, `MmqBatchedMinSeqLen` are defined once in Task 3 and referenced by identical names in Task 4 (test skip gating), Task 5 (dispatcher gate), Task 6 (env var + literal update), and Task 7 (doc text) — no renaming drift.
- `MMQ_BATCH_M_TILE` (CUDA `#define`, Task 1) = `MmqBatchMTile` (C# local `const int`, Task 3's launcher) = `2` in both places, and the launcher's `gridDim.y` computation (`ceil(m / MmqBatchMTile)`) matches the kernel's `m_base = blockIdx.y * MMQ_BATCH_M_TILE` addressing.
