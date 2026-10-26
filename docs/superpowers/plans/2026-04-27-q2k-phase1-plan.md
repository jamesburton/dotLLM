# Q2_K Quantization Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add full L3 support (CPU dequant + GPU dequant + per-call GEMV + MMQ + MMQ_preq + MMVQ-large + MMVQ-large_preq + grouped-MoE-GEMV) for the Q2_K GGUF quantization type so V2-Lite-Q2_K models load and run end-to-end on dotLLM CUDA.

**Architecture:** Q2_K is a port-style addition — structurally identical to existing Q3_K/Q4_K/Q5_K/Q6_K K-quants (256-element superblocks, linear quantization, dp4a-friendly inner loops). Each new kernel is built by copying the corresponding Q4_K kernel and substituting the per-superblock decode body (16 sub-blocks of 16 elements with 4-bit scale + 4-bit min coef per sub-block, vs Q4_K's 8 sub-blocks of 32 elements with 6-bit packed scales). No new abstractions; all extension via switch cases on `QuantizationType.Q2_K`.

**Tech Stack:** C# / .NET 10 (CPU side), CUDA C++ → PTX (GPU kernels), xUnit + SkippableFact (tests), `cuLaunchKernel` driver API, GGUF binary format (Q2_K = type ID 10).

**Spec reference:** `docs/superpowers/specs/2026-04-27-q2k-iquant-coverage-design.md` — Section 1, 2, 3, 4, 5, 6 cover the architecture; this plan implements Phase 1 only.

**Effort estimate:** ~1 week of focused work, ~27 new tests, 1 PR.

---

## File Structure

| Action | Path | Responsibility |
|---|---|---|
| Modify | `src/DotLLM.Core/Configuration/QuantizationType.cs` | Add `Q2_K = 10` enum entry |
| Modify | `src/DotLLM.Cpu/Kernels/Dequantize.cs` | Add `Q2_K_BlockBytes = 84` constant; switch cases in `RowByteSize` + `ToFloat32` |
| Modify | `src/DotLLM.Cpu/Kernels/DequantizeKQuants.cs` | Add `DequantizeQ2_K` scalar impl |
| Modify | `tests/DotLLM.Tests.Unit/Cpu/Kernels/DequantizeKQuantTests.cs` | Add Q2_K hand-calc + RowByteSize + non-aligned-throws tests |
| Modify | `native/kernels/dequant.cu` | Add `dequant_q2_k_f16` kernel + `Q2_K_BLOCK_BYTES` macro |
| Modify | `native/kernels/quantized_gemv.cu` | Add `quantized_gemv_q2_k` kernel |
| Modify | `native/kernels/quantized_gemv_mmq.cu` | Add `quantized_gemv_q2_k_mmq` + `_preq` + `_mmvq_large` + `_mmvq_large_preq` kernels |
| Modify | `native/kernels/moe_grouped_gemv.cu` | Add `moe_grouped_gemv_q2_k_f16` kernel |
| Modify | `src/DotLLM.Cuda/CudaKernels.cs` | Module function fields, `Has*Q2K` properties, `MinKAlignmentFor` case, dispatch switches in `LaunchDequantToF16`, `LaunchQuantizedGemv`, `LaunchQuantizedGemvMmq`, `LaunchMoeGroupedGemv` |
| Modify | `tests/DotLLM.Tests.Unit/Cuda/CudaQuantizedGemvAlignmentTests.cs` | Add Q2_K theory entries (M=2048, K=2048; M=1408, K=2048) |
| Modify | `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs` | Add `MmqQ2K_MatchesLegacyWithinTolerance` + `MmvqLargeQ2K_MatchesLegacy` + pre-Q8_1 theories |
| Modify | `tests/DotLLM.Tests.Unit/Cuda/CudaMoeGroupedGemvTests.cs` | Add `GroupedQ2K_MatchesPerCallWithinFp16Tolerance` theory |
| Modify | `tests/DotLLM.Tests.Unit/Models/Architectures/DeepSeekV2GgufLoadTests.cs` | Add `RealGguf_Q2K_FullModel_27LayerSmoke` |
| Modify | `.continue-here.md` | Round 16 entry documenting Q2_K landing |

No new files created in this phase; all additions extend existing K-quant infrastructure. Phase 2 (IQ-family) is the phase that introduces new files.

---

## Q2_K Block Layout Reference

Pinning the encoding so all kernels agree (verified against `ggml-quants.h` `block_q2_K`):

```
Offset  Size  Field           Description
─────────────────────────────────────────────────────────────────────
0       16    scales[16]      Per-sub-block scale (low nibble) + min coef (high nibble), 4 bits each
16      64    qs[64]          2-bit quant values (4 elements per byte, packed low-to-high)
80      2     d (half)        Super-block scale
82      2     dmin (half)     Super-block min
─────────────────────────────────────────────────────────────────────
Total: 84 bytes per 256 elements (16 sub-blocks of 16 elements)
```

**Decode**: for element index `t` (0..255):
- `sub = t / 16` (0..15) → which sub-block
- `j = t % 16` (0..15) → position within sub-block
- `byte_idx = (sub * 16 + j) / 4 = t / 4`
- `bit_off = ((sub * 16 + j) % 4) * 2 = (t % 4) * 2`
- `q2 = (qs[byte_idx] >> bit_off) & 0x3` (0..3, unsigned)
- `scale = scales[sub] & 0xF` (0..15, unsigned)
- `dmin_coef = (scales[sub] >> 4) & 0xF` (0..15, unsigned)
- `value = d * scale * q2 - dmin * dmin_coef`

The ½-byte scale + ½-byte min-coef packing per sub-block is what differs from Q3_K (12-byte 6-bit packed scales for 16 sub-blocks). The 2-bits-per-element packing is the same density as Q3_K's qs but without the hmask high-bit.

---

### Task 1: Q2_K enum + block-bytes constant + CPU dequant

**Files:**
- Modify: `src/DotLLM.Core/Configuration/QuantizationType.cs`
- Modify: `src/DotLLM.Cpu/Kernels/Dequantize.cs`
- Modify: `src/DotLLM.Cpu/Kernels/DequantizeKQuants.cs`
- Test: `tests/DotLLM.Tests.Unit/Cpu/Kernels/DequantizeKQuantTests.cs`

- [ ] **Step 1: Write the failing tests for Q2_K dequant**

Add to `tests/DotLLM.Tests.Unit/Cpu/Kernels/DequantizeKQuantTests.cs` (place after the Q3_K block at line ~274; mirrors the Q3_K hand-calc pattern):

```csharp
    // ──────────────────── Q2_K dequant ────────────────────

    private const int Q2_K_BlockBytes = 84;

    [Fact]
    public void Q2_K_SingleBlock_HandCalculated()
    {
        // Block layout: scales[16] + qs[64] + d[2] + dmin[2] = 84 bytes.
        nuint totalBytes = Q2_K_BlockBytes;
        nint ptr = (nint)NativeMemory.AlignedAlloc(totalBytes, 64);
        try
        {
            NativeMemory.Clear((void*)ptr, totalBytes);
            byte* block = (byte*)ptr;

            // d = 1.0, dmin = 0.5
            Unsafe.WriteUnaligned(block + 80, (Half)1.0f);
            Unsafe.WriteUnaligned(block + 82, (Half)0.5f);

            // scales[0]: low nibble = scale (we want scale = 3), high nibble = dmin coef (we want 2).
            // Packed as: (dmin_coef << 4) | scale = (2 << 4) | 3 = 0x23
            block[0] = 0x23;

            // qs[0] (offset 16): set element 0's 2 low bits to 0b10 (= 2).
            // qs encoding: 4 elements per byte, low-to-high.
            //   byte 0, bits 0-1 → element 0
            //   byte 0, bits 2-3 → element 1
            //   byte 0, bits 4-5 → element 2
            //   byte 0, bits 6-7 → element 3
            block[16 + 0] = 0x02;  // element 0 = 2, elements 1-3 = 0

            // Element 0: q2 = 2, scale = 3, dmin_coef = 2
            //   value = d * scale * q2 - dmin * dmin_coef
            //         = 1.0 * 3 * 2 - 0.5 * 2
            //         = 6 - 1 = 5
            float[] dest = new float[KQuantGroupSize];
            Dequantize.ToFloat32(ptr, KQuantGroupSize, QuantizationType.Q2_K, dest);

            Assert.Equal(5.0f, dest[0], 0.01f);

            // Element 1: q2 = 0, scale = 3, dmin_coef = 2
            //   value = 1.0 * 3 * 0 - 0.5 * 2 = -1
            Assert.Equal(-1.0f, dest[1], 0.01f);

            // Sub-block 1 (elements 16..31): scale = 0, dmin_coef = 0 (all-zero scales[1..15])
            //   value = 1.0 * 0 * 0 - 0.5 * 0 = 0
            Assert.Equal(0.0f, dest[16], 0.01f);
        }
        finally
        {
            NativeMemory.AlignedFree((void*)ptr);
        }
    }

    [Fact]
    public void Q2_K_RowByteSize_Matches()
    {
        // 256 elements = 1 super-block = 84 bytes.
        Assert.Equal(84L, Dequantize.RowByteSize(256, QuantizationType.Q2_K));
        // 1024 elements = 4 super-blocks = 336 bytes.
        Assert.Equal(336L, Dequantize.RowByteSize(1024, QuantizationType.Q2_K));
    }

    [Fact]
    public void Q2_K_NonAlignedCount_Throws()
    {
        float[] dest = new float[100];
        Assert.Throws<ArgumentException>(() =>
            Dequantize.ToFloat32(nint.Zero, 100, QuantizationType.Q2_K, dest));
    }
```

- [ ] **Step 2: Run tests to verify they fail (compile error)**

Run: `dotnet build src/DotLLM.Cpu/DotLLM.Cpu.csproj -c Release -nologo 2>&1 | tail -10`

Expected: build error referencing `QuantizationType.Q2_K` (enum value doesn't exist yet).

- [ ] **Step 3: Add Q2_K to the QuantizationType enum**

Edit `src/DotLLM.Core/Configuration/QuantizationType.cs`. Find the line with `Q8_0 = 8,` and add `Q2_K = 10` after it (matching GGUF type ID and slotting between Q8_0 and Q3_K which is type 11):

```csharp
    /// <summary>8-bit quantization, group size 32.</summary>
    Q8_0 = 8,

    /// <summary>2-bit K-quant, super-block of 256.</summary>
    Q2_K = 10,

    /// <summary>3-bit K-quant, super-block of 256.</summary>
    Q3_K = 11,
```

- [ ] **Step 4: Add Q2_K_BlockBytes constant + RowByteSize + ToFloat32 dispatch**

Edit `src/DotLLM.Cpu/Kernels/Dequantize.cs`. Find the K-quant block-bytes constants block; add `Q2_K_BlockBytes = 84` (place between Q3_K_BlockBytes=110 and Q4_K_BlockBytes=144 or wherever the alphabetical/numeric ordering puts it):

```csharp
    /// <summary>Q2_K super-block byte size: scales[16] + qs[64] + d[2] + dmin[2] = 84.</summary>
    public const int Q2_K_BlockBytes = 84;
```

In the same file, find `RowByteSize` and add the Q2_K case before the Q3_K case:

```csharp
        QuantizationType.Q2_K => elementCount / KQuantGroupSize * Q2_K_BlockBytes,
```

In `ToFloat32`, find the K-quant case block and add Q2_K:

```csharp
            case QuantizationType.Q2_K:
                if (elementCount % KQuantGroupSize != 0)
                    throw new ArgumentException(
                        $"Q2_K requires elementCount to be a multiple of {KQuantGroupSize}.",
                        nameof(elementCount));
                DequantizeKQuants.DequantizeQ2_K(src, dest.Slice(0, (int)elementCount), elementCount);
                return;
```

- [ ] **Step 5: Add DequantizeQ2_K scalar implementation**

Edit `src/DotLLM.Cpu/Kernels/DequantizeKQuants.cs`. Add (after `DequantizeQ3_K` impl):

```csharp
    /// <summary>
    /// Dequantizes Q2_K-quantized data to float32. Block layout:
    /// scales[16] (4-bit scale + 4-bit dmin coef per sub-block, packed) +
    /// qs[64] (2-bit elements, 4 per byte) + d (half) + dmin (half) = 84 bytes per 256 elements.
    /// Per-element decode: <c>value = d × scale × q2 − dmin × dmin_coef</c>.
    /// </summary>
    public static unsafe void DequantizeQ2_K(nint src, Span<float> dest, long elementCount)
    {
        if (elementCount % KQuantGroupSize != 0)
            throw new ArgumentException(
                $"Q2_K requires elementCount to be a multiple of {KQuantGroupSize}.", nameof(elementCount));

        long superBlocks = elementCount / KQuantGroupSize;
        byte* basePtr = (byte*)src;

        for (long sb = 0; sb < superBlocks; sb++)
        {
            byte* block = basePtr + sb * Dequantize.Q2_K_BlockBytes;
            byte* scales = block;          // 16 bytes
            byte* qs = block + 16;         // 64 bytes
            float d = (float)Unsafe.ReadUnaligned<Half>(block + 80);
            float dmin = (float)Unsafe.ReadUnaligned<Half>(block + 82);

            int outOffset = (int)(sb * KQuantGroupSize);
            for (int t = 0; t < KQuantGroupSize; t++)
            {
                int sub = t >> 4;          // t / 16
                int byteIdx = t >> 2;      // t / 4
                int bitOff = (t & 0x3) << 1; // (t % 4) * 2
                int q2 = (qs[byteIdx] >> bitOff) & 0x3;
                int scale = scales[sub] & 0xF;
                int dmCoef = (scales[sub] >> 4) & 0xF;
                dest[outOffset + t] = d * scale * q2 - dmin * dmCoef;
            }
        }
    }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~Q2_K" -nologo`

Expected: 3 tests passed (`Q2_K_SingleBlock_HandCalculated`, `Q2_K_RowByteSize_Matches`, `Q2_K_NonAlignedCount_Throws`).

- [ ] **Step 7: Commit**

```bash
git add src/DotLLM.Core/Configuration/QuantizationType.cs \
        src/DotLLM.Cpu/Kernels/Dequantize.cs \
        src/DotLLM.Cpu/Kernels/DequantizeKQuants.cs \
        tests/DotLLM.Tests.Unit/Cpu/Kernels/DequantizeKQuantTests.cs
git commit -m "$(cat <<'EOF'
CPU: Q2_K dequantization

Adds Q2_K (GGUF type ID 10, K-quant family, 84-byte 256-element
super-blocks) to the CPU dequantization path. Block layout:
scales[16] + qs[64] + d (half) + dmin (half).

Per-element decode: value = d × (scales[sub] & 0xF) × q_2bit
                          − dmin × ((scales[sub] >> 4) & 0xF)

Matches the encoding spec in ggml-quants.h block_q2_K. Single-block
hand-calc test pins the math; RowByteSize + non-aligned-throws tests
mirror the existing K-quant pattern.

Phase 1 of Spec 1 (Q2_K + IQ-family quantization coverage).
EOF
)"
```

---

### Task 2: Q2_K GPU dequant kernel

**Files:**
- Modify: `native/kernels/dequant.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaKernelComparisonTests.cs` OR add a new lean test class

- [ ] **Step 1: Add Q2_K GPU dequant kernel to dequant.cu**

Edit `native/kernels/dequant.cu`. Add (place after `dequant_q3_k_f16`, before `dequant_q4_k_f16`):

```cuda
// ── Q2_K: 84 bytes per 256 values ──────────────────────────────────
// struct block_q2_K { uint8_t scales[16]; uint8_t qs[64]; half d; half dmin; };
//   - scales[i]: low nibble = sub-block i scale, high nibble = sub-block i dmin coef
//   - qs[i]: 2-bit elements packed 4 per byte (low-to-high)
//
// 256 threads/block, one element per thread, FP16 store.

#define Q2_K_SUPER_BLOCK_SIZE 256
#define Q2_K_BLOCK_BYTES 84

extern "C" __global__ void __launch_bounds__(256) dequant_q2_k_f16(
    const uint8_t* __restrict__ src,
    half* __restrict__ dst,
    const int total_superblocks)
{
    int t = threadIdx.x; // 0..255

    for (int sb_idx = blockIdx.x; sb_idx < total_superblocks; sb_idx += gridDim.x)
    {
        const uint8_t* block = src + (size_t)sb_idx * Q2_K_BLOCK_BYTES;
        const uint8_t* scales = block;        // 16 bytes
        const uint8_t* qs = block + 16;       // 64 bytes
        float d = __half2float(*reinterpret_cast<const half*>(block + 80));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 82));

        int sub = t >> 4;            // t / 16
        int byte_idx = t >> 2;       // t / 4
        int bit_off = (t & 0x3) << 1; // (t % 4) * 2
        int q2 = (qs[byte_idx] >> bit_off) & 0x3;
        int scale = scales[sub] & 0xF;
        int dm_coef = (scales[sub] >> 4) & 0xF;

        float result = d * (float)scale * (float)q2 - dmin * (float)dm_coef;
        dst[(size_t)sb_idx * Q2_K_SUPER_BLOCK_SIZE + t] = __float2half(result);
    }
}
```

- [ ] **Step 2: Build PTX**

Run: `bash scripts/gpu-lock.sh acquire q2k-build "build dequant.ptx" 600 && bash native/build_ptx.sh dequant 2>&1 | tail -10; bash scripts/gpu-lock.sh release q2k-build`

(Or `native/build_ptx.bat dequant` on Windows-only shells.)

Expected: clean compilation, `native/ptx/dequant.ptx` updated.

- [ ] **Step 3: Add Q2_K dispatch case in CudaKernels.cs**

Edit `src/DotLLM.Cuda/CudaKernels.cs`. Find the dequant module function fields (around line 340); add:

```csharp
        _dequantQ2_KFunc = _dequantModule.TryGetFunction("dequant_q2_k_f16");
```

Find `LaunchDequantToF16` (around line 1599); add the Q2_K case before Q3_K:

```csharp
            case QuantizationType.Q2_K:
            {
                int totalSuperblocks = totalElements / 256;
                int tsbArg = totalSuperblocks;
                void** args = stackalloc void*[] {&srcArg, &dstArg, &tsbArg};
                uint gridDim = (uint)Math.Min(totalSuperblocks, MaxDequantGridSize);
                CudaDriverApi.cuLaunchKernel(_dequantQ2_KFunc,
                        gridDim, 1, 1, BlockSize, 1, 1,
                        0, stream, (nint)args, 0).ThrowOnError();
                return;
            }
```

Add the `_dequantQ2_KFunc` field declaration in the class field section.

- [ ] **Step 4: Write GPU dequant parity test**

Add to `tests/DotLLM.Tests.Unit/Cuda/CudaKernelComparisonTests.cs` (or extend the existing dequant comparison area). The test synthesizes random Q2_K block bytes, dequants on CPU + GPU, asserts max-abs-diff < 1e-3.

```csharp
    [SkippableFact]
    public void DequantQ2K_GpuMatchesCpu()
    {
        Skip.IfNot(_available, "No CUDA GPU available or PTX missing");

        const int superBlocks = 16;  // 16 × 256 = 4096 elements
        const int elementCount = superBlocks * 256;
        const int blockBytes = 84;
        long totalBytes = (long)superBlocks * blockBytes;

        var rng = new Random(0xC0FFEE);
        byte[] hostBytes = new byte[totalBytes];
        rng.NextBytes(hostBytes);

        // Make d / dmin reasonable halves at offset 80 / 82 of each super-block
        unsafe {
            fixed (byte* p = hostBytes) {
                for (int sb = 0; sb < superBlocks; sb++) {
                    byte* block = p + sb * blockBytes;
                    *(Half*)(block + 80) = (Half)((rng.NextDouble() - 0.5) * 0.04);
                    *(Half*)(block + 82) = (Half)((rng.NextDouble() - 0.5) * 0.02);
                }
            }
        }

        // CPU reference
        float[] cpuRef = new float[elementCount];
        unsafe {
            fixed (byte* p = hostBytes) {
                Dequantize.ToFloat32((nint)p, elementCount, QuantizationType.Q2_K, cpuRef);
            }
        }

        // GPU path
        Half[] gpuOut = new Half[elementCount];
        nint devSrc = 0, devDst = 0;
        try {
            CudaDriverApi.cuMemAlloc_v2(out devSrc, (nuint)totalBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devDst, (nuint)((long)elementCount * sizeof(ushort))).ThrowOnError();
            unsafe {
                fixed (byte* p = hostBytes)
                    CudaDriverApi.cuMemcpyHtoD_v2(devSrc, (nint)p, (nuint)totalBytes).ThrowOnError();
            }
            _kernels!.LaunchDequantToF16(devSrc, QuantizationType.Q2_K, devDst, elementCount, _stream!.Handle);
            _stream.Synchronize();
            unsafe {
                fixed (Half* p = gpuOut)
                    CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devDst, (nuint)((long)elementCount * sizeof(ushort))).ThrowOnError();
            }
        }
        finally {
            if (devSrc != 0) CudaDriverApi.cuMemFree_v2(devSrc);
            if (devDst != 0) CudaDriverApi.cuMemFree_v2(devDst);
        }

        float maxAbs = 0f;
        for (int i = 0; i < elementCount; i++) {
            float diff = MathF.Abs(cpuRef[i] - (float)gpuOut[i]);
            if (diff > maxAbs) maxAbs = diff;
        }
        _output.WriteLine($"Q2_K dequant max-abs-diff (GPU vs CPU): {maxAbs:F6}");
        Assert.True(maxAbs < 1e-3f, $"Q2_K GPU dequant diverges from CPU (max-abs-diff={maxAbs}).");
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
bash scripts/gpu-lock.sh acquire q2k-test "Q2_K dequant parity" 600
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1"  # or rely on Round-15 fallback
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~DequantQ2K_GpuMatchesCpu" -nologo
bash scripts/gpu-lock.sh release q2k-test
```

Expected: 1 test passed.

- [ ] **Step 6: Commit**

```bash
git add native/kernels/dequant.cu native/ptx/dequant.ptx \
        src/DotLLM.Cuda/CudaKernels.cs \
        tests/DotLLM.Tests.Unit/Cuda/CudaKernelComparisonTests.cs
git commit -m "$(cat <<'EOF'
CUDA: Q2_K GPU dequant kernel

Adds dequant_q2_k_f16 mirroring the existing dequant_q4_k_f16 pattern:
one CUDA block per super-block, 256 threads (one element per thread),
FP16 store. Substituted decode body for Q2_K's 16 sub-blocks of
16 elements with 4-bit scale + 4-bit min coef per sub-block.

GPU vs CPU parity test passes within FP16 rounding (1e-3 max-abs-diff).
EOF
)"
```

---

### Task 3: Q2_K per-call quantized GEMV

**Files:**
- Modify: `native/kernels/quantized_gemv.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaQuantizedGemvAlignmentTests.cs`

- [ ] **Step 1: Add quantized_gemv_q2_k kernel**

Edit `native/kernels/quantized_gemv.cu`. Add (after `quantized_gemv_q4_k`):

```cuda
// ── Q2_K: 84 bytes per 256 values ──────────────────────────────────

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q2_k(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    int row = blockIdx.x;
    if (row >= n) return;

    const int superblocks_per_row = k / 256;
    const uint8_t* w_row = weight + (size_t)row * superblocks_per_row * 84;

    float acc = 0.0f;

    for (int sb = threadIdx.x; sb < superblocks_per_row; sb += blockDim.x)
    {
        const uint8_t* block = w_row + sb * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        float d = __half2float(*reinterpret_cast<const half*>(block + 80));
        float dmin = __half2float(*reinterpret_cast<const half*>(block + 82));

        // 16 sub-blocks of 16 elements
        for (int sub = 0; sub < 16; sub++) {
            int sc = scales[sub] & 0xF;
            int dm = (scales[sub] >> 4) & 0xF;

            float sub_acc = 0.0f;
            float xsum_sub = 0.0f;
            #pragma unroll 16
            for (int j = 0; j < 16; j++) {
                int t = sub * 16 + j;
                int byte_idx = t >> 2;
                int bit_off = (t & 0x3) << 1;
                int q2 = (qs[byte_idx] >> bit_off) & 0x3;
                float xv = __half2float(x[sb * 256 + t]);
                sub_acc += (float)q2 * xv;
                xsum_sub += xv;
            }
            acc += d * (float)sc * sub_acc - dmin * (float)dm * xsum_sub;
        }
    }

    // Warp reduction
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);

    __shared__ float warp_sums[32];
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) warp_sums[warp_id] = acc;
    __syncthreads();

    if (warp_id == 0) {
        int num_warps = (blockDim.x + warpSize - 1) / warpSize;
        acc = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (threadIdx.x == 0) y[row] = __float2half(acc);
}
```

- [ ] **Step 2: Build PTX**

Run: `bash scripts/gpu-lock.sh acquire q2k-build "build quantized_gemv.ptx" 600 && bash native/build_ptx.sh quantized_gemv 2>&1 | tail -10; bash scripts/gpu-lock.sh release q2k-build`

- [ ] **Step 3: Wire dispatch in CudaKernels.cs**

Find the per-call GEMV module fields:

```csharp
        _quantizedGemvQ2_KFunc = _quantizedGemvModule.TryGetFunction("quantized_gemv_q2_k");
```

Add the field declaration. Then update `LaunchQuantizedGemv` switch:

```csharp
            QuantizationType.Q2_K => _quantizedGemvQ2_KFunc,
```

Update `HasQuantizedGemv`:

```csharp
    public static bool HasQuantizedGemv(QuantizationType qt) =>
        qt is QuantizationType.Q8_0 or QuantizationType.Q4_K or QuantizationType.Q5_0
            or QuantizationType.Q5_K or QuantizationType.Q6_K
            or QuantizationType.Q2_K;  // NEW
```

Update `MinKAlignmentFor` — add Q2_K to the K-quant 256-alignment branch:

```csharp
        QuantizationType.Q3_K or QuantizationType.Q4_K
            or QuantizationType.Q5_K or QuantizationType.Q6_K
            or QuantizationType.Q2_K => 256,  // NEW
```

- [ ] **Step 4: Write Q2_K GEMV parity test**

Add to `tests/DotLLM.Tests.Unit/Cuda/CudaQuantizedGemvAlignmentTests.cs`. Add a new theory:

```csharp
    [SkippableTheory]
    [InlineData(QuantizationType.Q2_K, 2048, 2048, 84)]
    [InlineData(QuantizationType.Q2_K, 2048, 1408, 84)]
    [InlineData(QuantizationType.Q2_K, 256, 256, 84)]
    public void Q2K_GemvMatchesScalarReference(
        QuantizationType qt, int M, int K, int blockBytes)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        // Q2_K requires K%256==0. Tests with K=1408 are *expected* to throw via
        // the K-alignment gate (1408 is not multiple of 256). Skip those for Q2_K.
        if (qt == QuantizationType.Q2_K && (K % 256) != 0)
            Skip.If(true, "Q2_K requires K%256==0; this shape exercises a different path.");

        RunGemvVsScalar(qt, M, K, blockBytes);
    }
```

Extend `RunGemvVsScalar` (already exists in this file) to handle Q2_K — add a branch in the synthesis loop and the `ComputeScalarReference` switch:

```csharp
                else if (qt == QuantizationType.Q2_K)
                {
                    // Q2_K block: scales[16] + qs[64] + d + dmin = 84 bytes
                    Half d = (Half)((rng.NextDouble() - 0.5) * 0.04);
                    Half dmin = (Half)((rng.NextDouble() - 0.5) * 0.02);
                    fixed (byte* pBlk = blk) {
                        *(Half*)(pBlk + 80) = d;
                        *(Half*)(pBlk + 82) = dmin;
                    }
                    for (int j = 0; j < 16; j++) blk[j] = (byte)rng.Next(0, 256);
                    for (int j = 16; j < 80; j++) blk[j] = (byte)rng.Next(0, 256);
                }
```

And in `ComputeScalarReference`:

```csharp
                else if (qt == QuantizationType.Q2_K)
                {
                    Half d, dmin;
                    fixed (byte* pW = weight) {
                        d = *(Half*)(pW + blkOff + 80);
                        dmin = *(Half*)(pW + blkOff + 82);
                    }
                    float dF = (float)d, dminF = (float)dmin;
                    // Walk the 256 elements of this block (we're inside the per-block loop).
                    // ... we need to refactor — for Q2_K the inner loop walks 256 elements per superblock,
                    // not 32 (Q8_0/Q5_0). The existing test infrastructure was for block_size=32 quants.
                    throw new NotImplementedException("Q2_K scalar reference goes here — see plan");
                }
```

NOTE: the existing `CudaQuantizedGemvAlignmentTests` was built for block_size=32 quants (Q8_0, Q5_0). Q2_K is block_size=256. **Either extend `RunGemvVsScalar` to handle both block sizes, OR add a new `RunGemvVsScalarKQuant` helper.** The cleaner choice is the latter:

```csharp
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunGemvVsScalarKQuant(QuantizationType qt, int M, int K, int blockBytes)
    {
        // Test infrastructure for K-quants (block_size=256). Mirrors RunGemvVsScalar
        // structurally but iterates over superblocks (K/256) per row.
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(CudaKernels.HasQuantizedGemv(qt), $"No GEMV kernel for {qt}");
        Assert.Equal(0, K % 256);

        int sbPerRow = K / 256;
        long rowBytes = (long)sbPerRow * blockBytes;
        long weightBytes = (long)M * rowBytes;
        var rng = new Random(0xBEEF ^ (int)qt ^ M ^ K);

        byte[] hostW = new byte[weightBytes];
        rng.NextBytes(hostW);
        // Per-super-block: write d/dmin halves at known offset (Q2_K: +80/+82)
        unsafe {
            fixed (byte* p = hostW) {
                for (int row = 0; row < M; row++) {
                    for (int sb = 0; sb < sbPerRow; sb++) {
                        byte* blk = p + row * rowBytes + sb * blockBytes;
                        if (qt == QuantizationType.Q2_K) {
                            *(Half*)(blk + 80) = (Half)((rng.NextDouble() - 0.5) * 0.04);
                            *(Half*)(blk + 82) = (Half)((rng.NextDouble() - 0.5) * 0.02);
                        }
                    }
                }
            }
        }

        Half[] hostX = new Half[K];
        for (int i = 0; i < K; i++) {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            hostX[i] = (Half)(g * 0.4);
        }

        // Scalar reference: dequant whole row → F32 dot.
        float[] yRef = new float[M];
        float[] xF32 = new float[K];
        for (int i = 0; i < K; i++) xF32[i] = (float)hostX[i];
        unsafe {
            fixed (byte* p = hostW) {
                float[] rowDequant = new float[K];
                for (int row = 0; row < M; row++) {
                    Dequantize.ToFloat32((nint)(p + row * rowBytes), K, qt, rowDequant);
                    float acc = 0;
                    for (int i = 0; i < K; i++) acc += rowDequant[i] * xF32[i];
                    yRef[row] = acc;
                }
            }
        }

        // GPU GEMV
        long xBytes = (long)K * sizeof(ushort);
        long yBytes = (long)M * sizeof(ushort);
        nint devW = 0, devX = 0, devY = 0;
        Half[] yGpu = new Half[M];
        try {
            CudaDriverApi.cuMemAlloc_v2(out devW, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devY, (nuint)yBytes).ThrowOnError();
            unsafe {
                fixed (byte* pW = hostW) CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)pW, (nuint)weightBytes).ThrowOnError();
                fixed (Half* pX = hostX) CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();
            }
            kernels.LaunchQuantizedGemv(devW, qt, devX, devY, M, K, stream.Handle);
            stream.Synchronize();
            unsafe { fixed (Half* p = yGpu) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devY, (nuint)yBytes).ThrowOnError(); }
        }
        finally {
            if (devW != 0) CudaDriverApi.cuMemFree_v2(devW);
            if (devX != 0) CudaDriverApi.cuMemFree_v2(devX);
            if (devY != 0) CudaDriverApi.cuMemFree_v2(devY);
        }

        float maxAbs = 0f, refMax = 0f;
        for (int i = 0; i < M; i++) {
            float diff = MathF.Abs((float)yGpu[i] - yRef[i]);
            if (diff > maxAbs) maxAbs = diff;
            if (MathF.Abs(yRef[i]) > refMax) refMax = MathF.Abs(yRef[i]);
        }
        _out.WriteLine($"{qt} M={M} K={K}: ref|max|={refMax:F3} max-abs-diff={maxAbs:F5}");
        Assert.True(maxAbs < 0.05f,
            $"Q2_K GEMV diverges from scalar reference (max-abs-diff={maxAbs}, refMax={refMax}).");
    }
```

Then change the Q2_K theory to call `RunGemvVsScalarKQuant`:

```csharp
    [SkippableTheory]
    [InlineData(QuantizationType.Q2_K, 2048, 2048, 84)]
    [InlineData(QuantizationType.Q2_K, 2048, 1024, 84)]
    [InlineData(QuantizationType.Q2_K, 256, 256, 84)]
    public void GemvKQuantMatchesScalarReference(
        QuantizationType qt, int M, int K, int blockBytes)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunGemvVsScalarKQuant(qt, M, K, blockBytes);
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run:
```bash
bash scripts/gpu-lock.sh acquire q2k-gemv "Q2_K GEMV parity" 600
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~Q2K|FullyQualifiedName~GemvKQuant" -nologo
bash scripts/gpu-lock.sh release q2k-gemv
```

Expected: 3 new tests passed.

- [ ] **Step 6: Commit**

```bash
git add native/kernels/quantized_gemv.cu native/ptx/quantized_gemv.ptx \
        src/DotLLM.Cuda/CudaKernels.cs \
        tests/DotLLM.Tests.Unit/Cuda/CudaQuantizedGemvAlignmentTests.cs
git commit -m "$(cat <<'EOF'
CUDA: Q2_K per-call quantized GEMV

Adds quantized_gemv_q2_k mirroring quantized_gemv_q4_k pattern (one
CUDA block per output row, FP32 accumulation, FP16 store, warp+block
reduction). Inner loop walks 16 sub-blocks of 16 elements per
super-block with 4-bit scale + 4-bit dmin coef each.

HasQuantizedGemv + MinKAlignmentFor extended to recognize Q2_K
(K%256 alignment requirement, same as other K-quants).

GEMV vs scalar reference parity within 5e-2 max-abs-diff at
V2-Lite-class shapes (M=2048, K=2048).
EOF
)"
```

---

### Task 4: Q2_K MMQ kernel (port from Q4_K MMQ)

**Files:**
- Modify: `native/kernels/quantized_gemv_mmq.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs`

- [ ] **Step 1: Read the existing Q4_K MMQ kernel as the porting source**

Run: `sed -n '47,219p' native/kernels/quantized_gemv_mmq.cu`

Read it carefully — note especially:
- Pre-Q8_1 input scratch contract (fixed format from `quantize_x_to_q8_1`)
- Per-warp 4-rows-per-block tile layout
- dp4a inner loop structure (`__dp4a(W_int8x4, X_int8x4, acc)`)
- Per-super-block scale extraction (Q4_K's 6-bit packed → for Q2_K we need 4-bit scale + 4-bit dmin coef)
- Final FP16 store with d × scale - dmin × dmin_coef × xsum applied

- [ ] **Step 2: Add quantized_gemv_q2_k_mmq kernel**

Edit `native/kernels/quantized_gemv_mmq.cu`. Add (after the Q4_K MMQ kernel — find `quantized_gemv_q4_k_mmq` and place Q2_K's MMQ before it OR add after the Q6_K MMQ before the MMVQ-large kernels, your choice based on file flow). Substitute the Q4_K decode body for Q2_K's:

```cuda
// ── Q2_K MMQ: 84 bytes per 256 values, 4-bit scale + 4-bit dmin coef per sub-block ──

extern "C" __global__ void __launch_bounds__(256, 2) quantized_gemv_q2_k_mmq(
    const uint8_t* __restrict__ weight,
    const half* __restrict__ x,
    half* __restrict__ y,
    const int n,
    const int k)
{
    // Layout mirrors Q4_K MMQ:
    //   - blockDim.x = 256 threads, block_y = 1
    //   - 32 warps × 8 lanes per warp tile = 4 output rows × 32 elements per warp
    //   - W_tile[4][32] in shared mem stores int8 weight bytes
    //   - X_tile[32] in shared mem stores int8 input bytes (pre-Q8_1)
    //   - dp4a inner loop accumulates int32 per (row, K-position)
    //
    // Per-super-block decode:
    //   scales[sub] (1 byte): low nibble = sc, high nibble = dm_coef
    //   qs[byte_idx]: 4 elements packed 2-bit
    //   d, dmin: half values at offset +80, +82

    // Read the Q4_K MMQ kernel and copy structure here, with these substitutions:
    //   Q4_K_BLOCK_BYTES (144) → Q2_K_BLOCK_BYTES (84)
    //   8 sub-blocks of 32 → 16 sub-blocks of 16
    //   6-bit packed scales (12 bytes from offset 4) → 16 bytes of (sc|dm) at offset 0
    //   nibble decode (4-bit) → 2-bit decode (4 elements per byte)
    //   q4 sign convention: nibble × scale → q2 × scale - dmin × dm_coef × xsum
    //
    // The dp4a fma stays identical — we only change how W_tile is filled
    // (decode body) and how the final FP32 → FP16 cast is performed (Q2_K
    // applies BOTH d-scale-positive AND dmin-coef-negative terms).

    /* PASTE Q4_K MMQ kernel body here, then apply substitutions per
       comment block above. ~170 lines total. Pin the Q4_K reference at
       quantized_gemv_mmq.cu:47-219 in the commit message. */
}
```

**This task does NOT include the full ~170-line MMQ kernel inline because the porting work is mechanical given the existing Q4_K reference.** The implementing engineer reads the Q4_K MMQ source, applies the documented substitutions, and produces a new ~150-180-line kernel. The output is a Q2_K MMQ that:
- Consumes the same Q8_1-style FP16 input (calls `quantize_x_to_q8_1` per call to convert input).
- Lays out the same 4-rows × 32-elements warp tiles in shared memory.
- Runs the same dp4a accumulation chain.
- Differs only in the per-superblock decode (16 sub-blocks of 16 elements + 4-bit scale + 4-bit dmin coef) and the final FP32 → FP16 cast formula (`d*sc*sub_acc - dmin*dm*xsum_sub`).

If during implementation a structural difference emerges that prevents direct porting (e.g., the 16-sub-block × 16-element layout creates a warp-tile mismatch), the engineer falls back to the same approach Q3_K uses — which currently has dequant-only support — and this task expands to also adding a Q3_K MMQ kernel as a sibling. Pin that decision in the PR description.

- [ ] **Step 3: Build PTX**

Run: `bash scripts/gpu-lock.sh acquire q2k-mmq "build q2_k mmq" 600 && bash native/build_ptx.sh quantized_gemv_mmq 2>&1 | tail -10; bash scripts/gpu-lock.sh release q2k-mmq`

- [ ] **Step 4: Wire dispatch**

In `CudaKernels.cs`:
- Add field: `private readonly nint _quantizedGemvQ2_KMmqFunc;`
- Load at ctor: `_quantizedGemvQ2_KMmqFunc = _quantizedGemvMmqModule.TryGetFunction("quantized_gemv_q2_k_mmq");`
- Add property: `public bool HasMmqQ2K => _quantizedGemvQ2_KMmqFunc != 0 && !DisableMmqQ2K;` plus `static bool DisableMmqQ2K`.
- Update `HasMmq(qt)`: `Q2_K => HasMmqQ2K,`
- Update `LaunchQuantizedGemvMmq` switch: `QuantizationType.Q2_K => _quantizedGemvQ2_KMmqFunc,`

- [ ] **Step 5: Add MMQ parity test**

Add to `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs`. Mirror the Q4_K theory pattern:

```csharp
    [SkippableTheory]
    [InlineData(4, 256)]
    [InlineData(8, 512)]
    [InlineData(64, 1024)]
    [InlineData(2048, 2048)]
    public void MmqQ2K_MatchesLegacyWithinTolerance(int n, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunMmqEquivalence(QuantizationType.Q2_K, n, k, blockBytes: 84,
            (rng, span) => SynthesizeQ2KBlock(rng, span));
    }

    private static unsafe void SynthesizeQ2KBlock(Random rng, Span<byte> block)
    {
        Assert.Equal(84, block.Length);
        Half d = (Half)((rng.NextDouble() - 0.5) * 0.04);
        Half dmin = (Half)((rng.NextDouble() - 0.5) * 0.02);
        fixed (byte* pBlk = block) {
            *(Half*)(pBlk + 80) = d;
            *(Half*)(pBlk + 82) = dmin;
        }
        for (int i = 0; i < 80; i++) block[i] = (byte)rng.Next(0, 256);
    }
```

- [ ] **Step 6: Run tests**

Run:
```bash
bash scripts/gpu-lock.sh acquire q2k-mmq-test "Q2_K MMQ parity" 600
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~MmqQ2K" -nologo
bash scripts/gpu-lock.sh release q2k-mmq-test
```

Expected: 4 tests passed within 3% peak-relative tolerance (matches existing Q4_K MMQ test bar).

- [ ] **Step 7: Commit**

```bash
git add native/kernels/quantized_gemv_mmq.cu native/ptx/quantized_gemv_mmq.ptx \
        src/DotLLM.Cuda/CudaKernels.cs \
        tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs
git commit -m "$(cat <<'EOF'
CUDA: Q2_K MMQ kernel (dp4a fast path)

Ports quantized_gemv_q4_k_mmq pattern to Q2_K with substituted
per-super-block decode body: 16 sub-blocks of 16 elements with
4-bit scale + 4-bit dmin coef per sub-block, vs Q4_K's 8 sub-blocks
of 32 elements with 6-bit packed scales.

Inner dp4a loop unchanged — int8 W_tile × int8 X_tile (pre-Q8_1
quantized input). Only the W_tile fill (decode prefix) and final
FP16 store (d*sc*acc - dmin*dm*xsum) differ from Q4_K.

MMQ vs legacy GEMV parity within 3% peak-relative on synthetic
Llama70B-class shapes (n=2048, k=2048).

Q4_K reference: native/kernels/quantized_gemv_mmq.cu:47-219.
EOF
)"
```

---

### Task 5: Q2_K MMQ pre-Q8_1 variant

**Files:**
- Modify: `native/kernels/quantized_gemv_mmq.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs`

- [ ] **Step 1: Add quantized_gemv_q2_k_mmq_preq kernel**

The `_preq` variant skips the per-call input quantization step; the input scratch is pre-quantized once per MoE/MLA projection (existing `quantize_x_to_q8_1` kernel writes the scratch; subsequent kernel calls read it). Body is identical to `quantized_gemv_q2_k_mmq` minus the per-warp input convert lane.

Reference the Q4_K _preq pattern at `native/kernels/quantized_gemv_mmq.cu:1110-1228`. Substitute the Q2_K decode body. Same structural rule as Task 4.

- [ ] **Step 2: Build PTX, wire dispatch, add test, commit**

Repeat the Task 4 step pattern:
- Build PTX
- Add field `_quantizedGemvQ2_KMmqPreqFunc`, load via `TryGetFunction`
- Update `LaunchQuantizedGemvMmqPreq` switch
- Add test theory `MmqQ2K_PreQ8_1_MatchesOnTheFly` mirroring existing pre-Q8_1 tests
- Run tests
- Commit with message `CUDA: Q2_K MMQ pre-Q8_1 variant`

---

### Task 6: Q2_K MMVQ-large + MMVQ-large pre-Q8_1

**Files:**
- Modify: `native/kernels/quantized_gemv_mmq.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaMmqKernelTests.cs`

- [ ] **Step 1: Add quantized_gemv_q2_k_mmvq_large kernel**

Reference Q4_K MMVQ-large pattern at `native/kernels/quantized_gemv_mmq.cu:630-784`. The MMVQ-large layout is 1 CUDA block per output row, multiple warps span the K-reduction, optimized for M ≥ 1024. Substitute Q2_K decode body.

- [ ] **Step 2: Add MMVQ-large pre-Q8_1 variant + dispatch + test**

Mirrors Task 5 pattern.

Run tests:
```bash
bash scripts/gpu-lock.sh acquire q2k-mmvq "Q2_K MMVQ-large parity" 600
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~MmvqLargeQ2K" -nologo
bash scripts/gpu-lock.sh release q2k-mmvq
```

Expected: 4 new tests passed. Commit with `CUDA: Q2_K MMVQ-large + pre-Q8_1 variants`.

---

### Task 7: Q2_K grouped-MoE-GEMV

**Files:**
- Modify: `native/kernels/moe_grouped_gemv.cu`
- Modify: `src/DotLLM.Cuda/CudaKernels.cs`
- Modify: `tests/DotLLM.Tests.Unit/Cuda/CudaMoeGroupedGemvTests.cs`

- [ ] **Step 1: Add moe_grouped_gemv_q2_k_f16 kernel**

Edit `native/kernels/moe_grouped_gemv.cu`. The grouped variant is structurally identical to the per-call `quantized_gemv_q2_k`, except the per-expert weight pointer is loaded from a device array indexed by `blockIdx.y`:

```cuda
extern "C" __global__ void moe_grouped_gemv_q2_k_f16(
    const half* __restrict__ x,
    const uintptr_t* __restrict__ weights_ptrs,
    uintptr_t* __restrict__ outputs_ptrs,
    const int M, const int K, const int K_active)
{
    int m = blockIdx.x, e = blockIdx.y;
    if (m >= M || e >= K_active) return;

    const uint8_t* W = (const uint8_t*)weights_ptrs[e];
    half* y = (half*)outputs_ptrs[e];

    // Body identical to quantized_gemv_q2_k inner loop; just substitute
    // weight_ptr-from-array and write result to per-expert output.

    // ... 16 sub-blocks decode, dp4a-free FP32 accumulation,
    //     warp + block reduction, single FP16 store at y[m].
}
```

(Full kernel ~80-100 lines. Reference moe_grouped_gemv_q4_k_f16 in the same file as the porting source.)

- [ ] **Step 2: Build PTX, wire dispatch, add test**

```csharp
// In CudaKernels.cs:
_moeGroupedGemvQ2_KFunc = _moeGroupedGemvModule.TryGetFunction("moe_grouped_gemv_q2_k_f16");

public bool HasMoeGroupedGemv(QuantizationType qt) => qt switch {
    /* existing entries */
    Q2_K => _moeGroupedGemvQ2_KFunc != 0,
    /* ... */
};

// Add to LaunchMoeGroupedGemv switch:
QuantizationType.Q2_K => _moeGroupedGemvQ2_KFunc,
```

Add test theory in `CudaMoeGroupedGemvTests.cs`:

```csharp
    [SkippableTheory]
    [InlineData(4, 256, 256)]
    [InlineData(4, 1408, 2048)]
    [InlineData(2, 256, 512)]
    public void GroupedQ2K_MatchesPerCallWithinFp16Tolerance(int kActive, int M, int K)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunGroupedEquivalence(QuantizationType.Q2_K, kActive, M, K, blockBytes: 84,
            (rng, span) => SynthesizeQ2KBlock(rng, span));
    }
```

(Reuses the existing `RunGroupedEquivalence` helper. Add `SynthesizeQ2KBlock` if not already present from Task 4.)

- [ ] **Step 3: Run tests**

```bash
bash scripts/gpu-lock.sh acquire q2k-grouped "Q2_K grouped MoE GEMV" 600
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~GroupedQ2K" -nologo
bash scripts/gpu-lock.sh release q2k-grouped
```

Expected: 3 new tests passed within 1e-3 max-abs-diff (matches existing grouped tests).

- [ ] **Step 4: Commit**

```bash
git add native/kernels/moe_grouped_gemv.cu native/ptx/moe_grouped_gemv.ptx \
        src/DotLLM.Cuda/CudaKernels.cs \
        tests/DotLLM.Tests.Unit/Cuda/CudaMoeGroupedGemvTests.cs
git commit -m "CUDA: Q2_K grouped-MoE-GEMV variant"
```

---

### Task 8: Real-GGUF V2-Lite Q2_K end-to-end smoke

**Files:**
- Modify: `tests/DotLLM.Tests.Unit/Models/Architectures/DeepSeekV2GgufLoadTests.cs`

- [ ] **Step 1: Cache the V2-Lite Q2_K GGUF**

Manual step (~10 min one-time):
```
Download bartowski/DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf (~5 GB)
Place at: ~/.dotllm/models/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf
```

If absent, the test added in Step 2 skips gracefully via `Skip.If`.

- [ ] **Step 2: Add the smoke test**

Edit `tests/DotLLM.Tests.Unit/Models/Architectures/DeepSeekV2GgufLoadTests.cs`. Add (mirroring `RealGguf_Q3KM_FullModel_27LayerSmoke` at line ~301):

```csharp
    /// <summary>
    /// Full 27-layer V2-Lite at Q2_K. Smaller than Q3_K_M (~5 GB on disk vs 8 GB),
    /// so the full model fits more comfortably in 12 GB GPU. Asserts: 27-layer load +
    /// prefill on 4 tokens + 3 decode steps, every step's logits finite. Skips if
    /// the GGUF isn't cached.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "GPU")]
    public void RealGguf_Q2K_FullModel_27LayerSmoke()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string path = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "models", "bartowski", "DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf");
        Skip.If(!File.Exists(path), $"Q2_K GGUF not cached at {path}");

        using var gguf = GgufFile.Open(path);
        var fullConfig = GgufModelConfigExtractor.Extract(gguf.Metadata);
        Assert.Equal(27, fullConfig.NumLayers);
        Assert.NotNull(fullConfig.MlaConfig);
        Assert.NotNull(fullConfig.Moe);

        // Trim KV cache horizon to 16 to fit smoke (matches Q3_K_M smoke pattern).
        var config = fullConfig with { MaxSequenceLength = 16 };

        using var model = CudaTransformerModel.LoadFromGguf(gguf, config);

        // Prefill 4 tokens, decode 3 more.
        int[] tokenIds = [100000, 261, 1559, 11];
        int[] positions = [0, 1, 2, 3];

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: 0, kvCache: null);
        AssertAllFinite(logits, "Q2_K prefill");

        // ... continue with 3 decode steps, asserting finite logits each time
        // (mirror the Q3_K_M smoke pattern verbatim).
    }
```

- [ ] **Step 3: Run smoke**

```bash
bash scripts/gpu-lock.sh acquire q2k-smoke "V2-Lite Q2_K 27-layer smoke" 1800
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~RealGguf_Q2K" -nologo
bash scripts/gpu-lock.sh release q2k-smoke
```

Expected: 1 smoke passed (or skipped if GGUF not cached) within ~30-60 s. The Q3_K_M baseline is 31 s; Q2_K is smaller so we expect similar or faster.

- [ ] **Step 4: Commit**

```bash
git add tests/DotLLM.Tests.Unit/Models/Architectures/DeepSeekV2GgufLoadTests.cs
git commit -m "$(cat <<'EOF'
CUDA: V2-Lite Q2_K full-27-layer real-GGUF smoke

Asserts the full chain — Q2_K dequant + GEMV + MMQ + grouped-MoE-GEMV
— produces finite logits end-to-end on the cached
bartowski/DeepSeek-Coder-V2-Lite-Instruct-Q2_K.gguf (5 GB) at all
27 layers (1 dense + 26 MoE).

Decisive evidence Phase 1 of Spec 1 ships V2-Lite-Q2_K on a single
RTX 3060.
EOF
)"
```

---

### Task 9: Final regression sweep + .continue-here.md update

**Files:**
- Modify: `.continue-here.md`

- [ ] **Step 1: Run the full critical-path regression sweep**

```bash
bash scripts/gpu-lock.sh acquire q2k-regression "full Q2_K regression" 1200
dotnet test tests/DotLLM.Tests.Unit/DotLLM.Tests.Unit.csproj -c Release --filter "FullyQualifiedName~CudaMoeGroupedGemvTests|FullyQualifiedName~CudaTransformerMlaForwardTests|FullyQualifiedName~CudaQuantizedGemvAlignmentTests|FullyQualifiedName~RealGguf_QuantizedMlaMoe|FullyQualifiedName~RealGguf_Q3KM|FullyQualifiedName~RealGguf_Q2K|FullyQualifiedName~Q2K|FullyQualifiedName~MmqQ2K|FullyQualifiedName~MmvqLargeQ2K|FullyQualifiedName~GroupedQ2K" -nologo --no-build
bash scripts/gpu-lock.sh release q2k-regression
```

Expected:
- All previous tests (24 critical CUDA + Q3_K_M smoke) still pass.
- 27+ new Q2_K tests pass.
- 0 failures.

- [ ] **Step 2: Update .continue-here.md with Round 16 entry**

Insert at the top of the file (replacing the date in the front-matter and prepending a Round 16 section):

```markdown
last_commit: <commit-sha-of-task-8>
last_updated: 2026-MM-DD (Round 16 — Q2_K full L3, Phase 1 of Spec 1)

## Round 16: Q2_K full L3 (Phase 1 of Spec 1)

The Q2_K K-quant family quantization type is now fully supported on CUDA:
CPU dequant + GPU dequant + per-call GEMV + MMQ (legacy + pre-Q8_1) +
MMVQ-large (legacy + pre-Q8_1) + grouped-MoE-GEMV. V2-Lite-Q2_K loads
end-to-end and produces finite logits across all 27 layers on the
12 GB RTX 3060.

### Commits this round (`<task1-sha>..<task8-sha>`)

```
<task1-sha> CPU: Q2_K dequantization
<task2-sha> CUDA: Q2_K GPU dequant kernel
<task3-sha> CUDA: Q2_K per-call quantized GEMV
<task4-sha> CUDA: Q2_K MMQ kernel (dp4a fast path)
<task5-sha> CUDA: Q2_K MMQ pre-Q8_1 variant
<task6-sha> CUDA: Q2_K MMVQ-large + pre-Q8_1 variants
<task7-sha> CUDA: Q2_K grouped-MoE-GEMV variant
<task8-sha> CUDA: V2-Lite Q2_K full-27-layer real-GGUF smoke
```

### What landed

(Filled in by engineer during Task 9 Step 2 from the regression-sweep output:
new Q2_K test count, V2-Lite Q2_K smoke timing vs the 31 s Q3_K_M baseline,
critical-tests-still-green count, total CPU pass count, total failure count.)

### Next: Plan 1.2 (IQ4_NL + IQ4_XS)

Phase 2 of Spec 1. Brings the IQ-family infrastructure online —
codebook constants in IQuantGrids.cs, new dequant_iquants.cu PTX,
shared MMQ pattern with codebook expansion before dp4a fires.
```

- [ ] **Step 3: Final commit**

```bash
git add .continue-here.md
git commit -m "$(cat <<'EOF'
.continue-here.md: Round 16 — Q2_K full L3 lands (Phase 1 of Spec 1)

Documents the Q2_K phase: 8 commits across CPU dequant, GPU dequant,
per-call GEMV, MMQ + pre-Q8_1, MMVQ-large + pre-Q8_1,
grouped-MoE-GEMV, real-GGUF V2-Lite Q2_K 27-layer smoke.

Phase 2 (IQ4_NL + IQ4_XS) is next.
EOF
)"
```

- [ ] **Step 4: Optional — open the Phase 1 PR**

```bash
git push origin feature/mamba-3-cuda
gh pr create --base feature/mamba-3 --title "CUDA: Q2_K full L3 (Phase 1 of quantization-coverage spec)" --body "$(cat <<'EOF'
## Summary
- Phase 1 of Spec 1 (`docs/superpowers/specs/2026-04-27-q2k-iquant-coverage-design.md`).
- Adds Q2_K K-quant family support: CPU + GPU dequant, per-call GEMV, MMQ + MMVQ-large (legacy + pre-Q8_1), grouped-MoE-GEMV.
- V2-Lite-Q2_K loads + decodes end-to-end on RTX 3060 (27 layers, 5 GB GGUF, ~30-60 s smoke).

## Validation
- 27+ new Q2_K-specific tests, 0 failures.
- 24 critical CUDA tests (3 MLA + 12 grouped GEMV + 4 V2-Lite Q4_K_M smokes + 5 K=1408 alignment) still green.
- 1467+ CPU tests still green.
- Q3_K_M 27-layer smoke unchanged at ~31 s.
- Build: 0 warnings, 0 errors.

## Next steps
Phase 2 (IQ4_NL + IQ4_XS) — separate plan, separate PR.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:** Each section of `2026-04-27-q2k-iquant-coverage-design.md` Phase 1 row maps to a task:
- Section 1 (architecture) ↔ implicit in every task
- Section 2 (CPU dequant) ↔ Task 1
- Section 3 (GPU dequant) ↔ Task 2
- Section 4 (per-call GEMV) ↔ Task 3
- Section 5 (MMQ + MMVQ-large + pre-Q8_1) ↔ Tasks 4, 5, 6
- Section 6 (grouped-MoE-GEMV) ↔ Task 7
- Section 7 (testing strategy: kernel parity + real-GGUF smoke + regression) ↔ inline in each task + Task 8 + Task 9

**Placeholder scan:** Tasks 4 / 5 / 6 reference Q4_K source-line numbers (`quantized_gemv_mmq.cu:47-219` etc.) and instruct the implementer to mechanically port the kernel body. This is intentional: the kernel bodies are 150-200 lines each and the porting work is largely mechanical given the existing reference. Embedding 600+ lines of CUDA inline would obscure the core changes (decode body substitution + final-cast formula). The plan documents the decode delta precisely; the implementing engineer reads the reference, applies the delta, and produces the new kernel. This is a deliberate granularity tradeoff.

If the implementing agent prefers more granular per-line guidance for the MMQ kernels, expand Tasks 4-6 in-session by reading the Q4_K kernel and writing a per-section diff inline.

**Type consistency:**
- `_quantizedGemvQ2_KFunc` (Task 3), `_quantizedGemvQ2_KMmqFunc` (Task 4), `_quantizedGemvQ2_KMmqPreqFunc` (Task 5), `_quantizedGemvQ2_KMmvqLargeFunc` (Task 6), `_moeGroupedGemvQ2_KFunc` (Task 7) — naming matches existing `_quantizedGemvQ4_KFunc` etc. patterns.
- `HasMmqQ2K` (Task 4), `HasMmvqLargeQ2K` (Task 6), `DisableMmqQ2K`, `DisableMmvqLargeQ2K` — match existing `HasMmqQ4K` / `DisableMmvqLargeQ4K` patterns.
- `QuantizationType.Q2_K` consistent across all task references.
- `MinKAlignmentFor(Q2_K) = 256` consistent (it's a K-quant).

**Scope:** This plan is 8 tasks covering a single quantization type (Q2_K) end-to-end. ~1 week of work; one shippable PR. Right granularity.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-27-q2k-phase1-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration. Each subagent gets the task content + relevant spec context; the parent agent reviews diffs + test output before approving the next task.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints for review.

Which approach?
