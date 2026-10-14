using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Tests CUDA KV-cache quantization kernels (quant_kv.cu) with random data that
/// discriminates between correct and permuted element orderings within 32-element blocks.
/// <para>
/// Background (issue #974): ramp data like <c>pos*0.01 + i*0.001</c> produces monotone
/// sequences within each 32-block whose adjacent elements quantize to the same nibble/byte.
/// A layout mismatch (e.g. Q4_0 lo/hi nibble swap) permutes elements inside each block
/// without changing block statistics, so the ramp-based tests pass despite bugs.
/// Random data ensures that adjacent elements differ enough to detect any within-block
/// permutation.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
public sealed unsafe class CudaKvQuantRoundtripTests : IDisposable
{
    private const int BlockSize = 32;
    private const int Q8_0BlockBytes = 34;
    private const int Q4_0BlockBytes = 18;

    private readonly ITestOutputHelper _output;
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;
    private readonly bool _available;

    public CudaKvQuantRoundtripTests(ITestOutputHelper output)
    {
        _output = output;
        if (!CudaDevice.IsAvailable()) return;

        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();

        string? ptxDir = FindPtxDir();
        if (ptxDir != null)
            _kernels = new CudaKernels(ptxDir);

        _available = _kernels != null;
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };

        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    // ─────────────────── Q8_0 round-trip ───────────────────

    /// <summary>
    /// FP16 → Q8_0 (GPU quant) → Q8_0 → FP16 (GPU dequant) round-trip with random data.
    /// Asserts per-element error within Q8_0 tolerance (maxAbs / 127 + FP16 rounding).
    /// </summary>
    [SkippableFact]
    public void Q8_0_RandomData_RoundTrip_PreservesElementOrder()
    {
        SkipIfUnavailable();

        int numBlocks = 8; // 256 elements — multiple of 32, tests several blocks
        int n = numBlocks * BlockSize;
        nint s = _stream!.Handle;

        // Generate random FP16 data on host (range [-5, 5])
        var rng = new Random(42);
        ushort[] srcFp16 = new ushort[n];
        float[] srcF32 = new float[n];
        for (int i = 0; i < n; i++)
        {
            float val = (float)(rng.NextDouble() * 10.0 - 5.0);
            Half h = (Half)val;
            srcFp16[i] = BitConverter.HalfToUInt16Bits(h);
            srcF32[i] = (float)h; // capture actual FP16 value for comparison
        }

        long fp16Bytes = (long)n * sizeof(ushort);
        long quantBytes = (long)numBlocks * Q8_0BlockBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Src, (nuint)fp16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQuant, (nuint)quantBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

        try
        {
            // Upload FP16 source
            fixed (ushort* p = srcFp16)
                CudaDriverApi.cuMemcpyHtoD_v2(devFp16Src, (nint)p, (nuint)fp16Bytes).ThrowOnError();

            // Quantize: FP16 → Q8_0 on GPU
            _kernels!.LaunchQuantKv(devFp16Src, devQuant, n, KvCacheDType.Q8_0, s);

            // Dequantize: Q8_0 → FP16 on GPU
            _kernels.LaunchDequantToF16(devQuant, QuantizationType.Q8_0, devFp16Dst, n, s);
            _stream!.Synchronize();

            // Download result
            ushort[] dstFp16 = new ushort[n];
            fixed (ushort* p = dstFp16)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

            // Per-element comparison with Q8_0 tolerance
            int mismatches = 0;
            float maxErr = 0;
            for (int i = 0; i < n; i++)
            {
                float original = srcF32[i];
                float roundTripped = (float)BitConverter.UInt16BitsToHalf(dstFp16[i]);

                // Per-block tolerance: maxAbs/127 * 1.1 + FP16 rounding
                int block = i / BlockSize;
                float blockMaxAbs = 0;
                for (int j = block * BlockSize; j < (block + 1) * BlockSize; j++)
                    blockMaxAbs = MathF.Max(blockMaxAbs, MathF.Abs(srcF32[j]));

                float tolerance = blockMaxAbs / 127.0f * 1.15f + 0.002f; // Q8_0 step + FP16 rounding
                float err = MathF.Abs(original - roundTripped);
                if (err > maxErr) maxErr = err;
                if (err > tolerance)
                {
                    if (mismatches < 10)
                        _output.WriteLine($"  [{i}] orig={original:F4} rt={roundTripped:F4} err={err:E3} tol={tolerance:E3}");
                    mismatches++;
                }
            }

            _output.WriteLine($"[Q8_0 RoundTrip] n={n} maxErr={maxErr:E4} mismatches={mismatches}/{n}");
            Assert.True(mismatches == 0,
                $"Q8_0 round-trip: {mismatches}/{n} elements exceed tolerance. maxErr={maxErr:E4}");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devFp16Src);
            CudaDriverApi.cuMemFree_v2(devQuant);
            CudaDriverApi.cuMemFree_v2(devFp16Dst);
        }
    }

    // ─────────────────── Q4_0 round-trip ───────────────────

    /// <summary>
    /// FP16 → Q4_0 (GPU quant) → Q4_0 → FP16 (GPU dequant) round-trip with random data.
    /// Asserts per-element error within Q4_0 tolerance (maxAbs / 7 + FP16 rounding).
    /// </summary>
    [SkippableFact]
    public void Q4_0_RandomData_RoundTrip_PreservesElementOrder()
    {
        SkipIfUnavailable();

        int numBlocks = 8;
        int n = numBlocks * BlockSize;
        nint s = _stream!.Handle;

        var rng = new Random(42);
        ushort[] srcFp16 = new ushort[n];
        float[] srcF32 = new float[n];
        for (int i = 0; i < n; i++)
        {
            float val = (float)(rng.NextDouble() * 10.0 - 5.0);
            Half h = (Half)val;
            srcFp16[i] = BitConverter.HalfToUInt16Bits(h);
            srcF32[i] = (float)h;
        }

        long fp16Bytes = (long)n * sizeof(ushort);
        long quantBytes = (long)numBlocks * Q4_0BlockBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Src, (nuint)fp16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQuant, (nuint)quantBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

        try
        {
            fixed (ushort* p = srcFp16)
                CudaDriverApi.cuMemcpyHtoD_v2(devFp16Src, (nint)p, (nuint)fp16Bytes).ThrowOnError();

            _kernels!.LaunchQuantKv(devFp16Src, devQuant, n, KvCacheDType.Q4_0, s);
            _kernels.LaunchDequantToF16(devQuant, QuantizationType.Q4_0, devFp16Dst, n, s);
            _stream!.Synchronize();

            ushort[] dstFp16 = new ushort[n];
            fixed (ushort* p = dstFp16)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

            int mismatches = 0;
            float maxErr = 0;
            for (int i = 0; i < n; i++)
            {
                float original = srcF32[i];
                float roundTripped = (float)BitConverter.UInt16BitsToHalf(dstFp16[i]);

                int block = i / BlockSize;
                float blockMaxAbs = 0;
                for (int j = block * BlockSize; j < (block + 1) * BlockSize; j++)
                    blockMaxAbs = MathF.Max(blockMaxAbs, MathF.Abs(srcF32[j]));

                float tolerance = blockMaxAbs / 7.0f * 1.15f + 0.01f; // Q4_0 step + FP16 rounding
                float err = MathF.Abs(original - roundTripped);
                if (err > maxErr) maxErr = err;
                if (err > tolerance)
                {
                    if (mismatches < 10)
                        _output.WriteLine($"  [{i}] orig={original:F4} rt={roundTripped:F4} err={err:E3} tol={tolerance:E3}");
                    mismatches++;
                }
            }

            _output.WriteLine($"[Q4_0 RoundTrip] n={n} maxErr={maxErr:E4} mismatches={mismatches}/{n}");
            Assert.True(mismatches == 0,
                $"Q4_0 round-trip: {mismatches}/{n} elements exceed tolerance. maxErr={maxErr:E4}");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devFp16Src);
            CudaDriverApi.cuMemFree_v2(devQuant);
            CudaDriverApi.cuMemFree_v2(devFp16Dst);
        }
    }

    // ─────────────────── Cross-backend: CUDA quant matches CPU dequant ───────────────────

    /// <summary>
    /// Quantize on GPU (FP16 → Q8_0), download the raw Q8_0 bytes, dequantize on CPU.
    /// Compare CPU dequant result against the GPU dequant result.
    /// This catches layout mismatches between GPU quant_kv.cu and CPU KvQuantize.
    /// </summary>
    [SkippableFact]
    public void Q8_0_CudaQuant_CpuDequant_MatchesGpuDequant()
    {
        SkipIfUnavailable();

        int numBlocks = 4;
        int n = numBlocks * BlockSize;
        nint s = _stream!.Handle;

        var rng = new Random(42);
        ushort[] srcFp16 = new ushort[n];
        float[] srcF32 = new float[n];
        for (int i = 0; i < n; i++)
        {
            float val = (float)(rng.NextDouble() * 10.0 - 5.0);
            Half h = (Half)val;
            srcFp16[i] = BitConverter.HalfToUInt16Bits(h);
            srcF32[i] = (float)h;
        }

        long fp16Bytes = (long)n * sizeof(ushort);
        long quantBytes = (long)numBlocks * Q8_0BlockBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Src, (nuint)fp16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQuant, (nuint)quantBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

        try
        {
            fixed (ushort* p = srcFp16)
                CudaDriverApi.cuMemcpyHtoD_v2(devFp16Src, (nint)p, (nuint)fp16Bytes).ThrowOnError();

            // GPU quantize
            _kernels!.LaunchQuantKv(devFp16Src, devQuant, n, KvCacheDType.Q8_0, s);

            // GPU dequantize
            _kernels.LaunchDequantToF16(devQuant, QuantizationType.Q8_0, devFp16Dst, n, s);
            _stream!.Synchronize();

            // Download quantized bytes for CPU dequant
            byte[] quantBytes_host = new byte[quantBytes];
            fixed (byte* p = quantBytes_host)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devQuant, (nuint)quantBytes).ThrowOnError();

            // Download GPU-dequantized FP16 result
            ushort[] gpuDequant = new ushort[n];
            fixed (ushort* p = gpuDequant)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

            // CPU dequant from the GPU-produced Q8_0 bytes
            float[] cpuDequant = new float[n];
            fixed (byte* qp = quantBytes_host)
            fixed (float* op = cpuDequant)
            {
                DotLLM.Cpu.Kernels.KvQuantize.Q8_0ToF32(qp, op, n);
            }

            // Compare GPU and CPU dequant results
            int mismatches = 0;
            for (int i = 0; i < n; i++)
            {
                float gpuVal = (float)BitConverter.UInt16BitsToHalf(gpuDequant[i]);
                float cpuVal = cpuDequant[i];
                // Both should agree within FP16 representation tolerance
                float diff = MathF.Abs(gpuVal - cpuVal);
                if (diff > 0.002f) // FP16 smallest step for values around 5
                {
                    if (mismatches < 10)
                        _output.WriteLine($"  [{i}] gpu={gpuVal:F4} cpu={cpuVal:F4} diff={diff:E3}");
                    mismatches++;
                }
            }

            _output.WriteLine($"[Q8_0 Cross-Backend] n={n} mismatches={mismatches}/{n}");
            Assert.True(mismatches == 0,
                $"Q8_0 GPU-quant/CPU-dequant mismatch: {mismatches}/{n} elements differ. " +
                $"This indicates a layout difference between CUDA quant_kv.cu and CPU KvQuantize.");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devFp16Src);
            CudaDriverApi.cuMemFree_v2(devQuant);
            CudaDriverApi.cuMemFree_v2(devFp16Dst);
        }
    }

    /// <summary>
    /// Same cross-backend validation for Q4_0: GPU quantize, CPU dequant, compare.
    /// </summary>
    [SkippableFact]
    public void Q4_0_CudaQuant_CpuDequant_MatchesGpuDequant()
    {
        SkipIfUnavailable();

        int numBlocks = 4;
        int n = numBlocks * BlockSize;
        nint s = _stream!.Handle;

        var rng = new Random(42);
        ushort[] srcFp16 = new ushort[n];
        float[] srcF32 = new float[n];
        for (int i = 0; i < n; i++)
        {
            float val = (float)(rng.NextDouble() * 10.0 - 5.0);
            Half h = (Half)val;
            srcFp16[i] = BitConverter.HalfToUInt16Bits(h);
            srcF32[i] = (float)h;
        }

        long fp16Bytes = (long)n * sizeof(ushort);
        long quantBytes_val = (long)numBlocks * Q4_0BlockBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Src, (nuint)fp16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQuant, (nuint)quantBytes_val).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

        try
        {
            fixed (ushort* p = srcFp16)
                CudaDriverApi.cuMemcpyHtoD_v2(devFp16Src, (nint)p, (nuint)fp16Bytes).ThrowOnError();

            _kernels!.LaunchQuantKv(devFp16Src, devQuant, n, KvCacheDType.Q4_0, s);
            _kernels.LaunchDequantToF16(devQuant, QuantizationType.Q4_0, devFp16Dst, n, s);
            _stream!.Synchronize();

            byte[] quantBytes_host = new byte[quantBytes_val];
            fixed (byte* p = quantBytes_host)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devQuant, (nuint)quantBytes_val).ThrowOnError();

            ushort[] gpuDequant = new ushort[n];
            fixed (ushort* p = gpuDequant)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

            float[] cpuDequant = new float[n];
            fixed (byte* qp = quantBytes_host)
            fixed (float* op = cpuDequant)
            {
                DotLLM.Cpu.Kernels.KvQuantize.Q4_0ToF32(qp, op, n);
            }

            int mismatches = 0;
            for (int i = 0; i < n; i++)
            {
                float gpuVal = (float)BitConverter.UInt16BitsToHalf(gpuDequant[i]);
                float cpuVal = cpuDequant[i];
                float diff = MathF.Abs(gpuVal - cpuVal);
                if (diff > 0.01f) // Q4_0 is coarser, so tolerance for FP16 rounding is wider
                {
                    if (mismatches < 10)
                        _output.WriteLine($"  [{i}] gpu={gpuVal:F4} cpu={cpuVal:F4} diff={diff:E3}");
                    mismatches++;
                }
            }

            _output.WriteLine($"[Q4_0 Cross-Backend] n={n} mismatches={mismatches}/{n}");
            Assert.True(mismatches == 0,
                $"Q4_0 GPU-quant/CPU-dequant mismatch: {mismatches}/{n} elements differ. " +
                $"This indicates a layout difference between CUDA quant_kv.cu and CPU KvQuantize.");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devFp16Src);
            CudaDriverApi.cuMemFree_v2(devQuant);
            CudaDriverApi.cuMemFree_v2(devFp16Dst);
        }
    }

    // ─────────────────── Negative controls ───────────────────

    /// <summary>
    /// Negative control for Q4_0: GPU-quantize random FP16 data, download the Q4_0 bytes,
    /// then dequant on CPU with lo/hi nibbles deliberately swapped.
    /// The permuted result MUST differ from the correct dequant at many elements.
    /// If it doesn't, the random data is degenerate and cannot catch nibble-swap bugs.
    /// </summary>
    [SkippableFact]
    public void Q4_0_NegativeControl_PermutedDequant_DetectedByRandomData()
    {
        SkipIfUnavailable();

        int numBlocks = 8;
        int n = numBlocks * BlockSize;
        nint s = _stream!.Handle;

        var rng = new Random(42);
        ushort[] srcFp16 = new ushort[n];
        for (int i = 0; i < n; i++)
        {
            float val = (float)(rng.NextDouble() * 10.0 - 5.0);
            srcFp16[i] = BitConverter.HalfToUInt16Bits((Half)val);
        }

        long fp16Bytes = (long)n * sizeof(ushort);
        long quantBytesVal = (long)numBlocks * Q4_0BlockBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Src, (nuint)fp16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQuant, (nuint)quantBytesVal).ThrowOnError();

        try
        {
            fixed (ushort* p = srcFp16)
                CudaDriverApi.cuMemcpyHtoD_v2(devFp16Src, (nint)p, (nuint)fp16Bytes).ThrowOnError();

            _kernels!.LaunchQuantKv(devFp16Src, devQuant, n, KvCacheDType.Q4_0, s);
            _stream!.Synchronize();

            // Download Q4_0 bytes
            byte[] quantHost = new byte[quantBytesVal];
            fixed (byte* p = quantHost)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devQuant, (nuint)quantBytesVal).ThrowOnError();

            // Correct CPU dequant
            float[] correct = new float[n];
            fixed (byte* qp = quantHost)
            fixed (float* op = correct)
            {
                DotLLM.Cpu.Kernels.KvQuantize.Q4_0ToF32Scalar(qp, op, n);
            }

            // Permuted dequant: swap lo/hi nibbles within each block
            float[] permuted = new float[n];
            fixed (byte* qp = quantHost)
            {
                for (int block = 0; block < numBlocks; block++)
                {
                    byte* blockSrc = qp + block * Q4_0BlockBytes;
                    float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
                    byte* qs = blockSrc + 2;
                    for (int j = 0; j < 16; j++)
                    {
                        byte packed = qs[j];
                        int lo = (packed & 0x0F) - 8;
                        int hi = (packed >> 4) - 8;
                        permuted[block * BlockSize + 2 * j] = d * hi;       // SWAPPED
                        permuted[block * BlockSize + 2 * j + 1] = d * lo;   // SWAPPED
                    }
                }
            }

            // Count how many elements differ between correct and permuted
            int mismatches = 0;
            for (int i = 0; i < n; i++)
            {
                if (MathF.Abs(correct[i] - permuted[i]) > 1e-6f)
                    mismatches++;
            }

            _output.WriteLine($"[Q4_0 Negative Control] lo/hi swap: {mismatches}/{n} elements differ");

            // With well-distributed random data, swapping lo/hi should affect > 25% of elements
            Assert.True(mismatches > n / 4,
                $"Permuted Q4_0 dequant (lo/hi swap) only differed at {mismatches}/{n} elements. " +
                $"The test data is too degenerate to catch permutation bugs.");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devFp16Src);
            CudaDriverApi.cuMemFree_v2(devQuant);
        }
    }

    /// <summary>
    /// Negative control for Q8_0: GPU-quantize random FP16 data, download Q8_0 bytes,
    /// dequant on CPU with adjacent pairs swapped. Must detect the permutation.
    /// </summary>
    [SkippableFact]
    public void Q8_0_NegativeControl_PermutedDequant_DetectedByRandomData()
    {
        SkipIfUnavailable();

        int numBlocks = 8;
        int n = numBlocks * BlockSize;
        nint s = _stream!.Handle;

        var rng = new Random(42);
        ushort[] srcFp16 = new ushort[n];
        for (int i = 0; i < n; i++)
        {
            float val = (float)(rng.NextDouble() * 10.0 - 5.0);
            srcFp16[i] = BitConverter.HalfToUInt16Bits((Half)val);
        }

        long fp16Bytes = (long)n * sizeof(ushort);
        long quantBytesVal = (long)numBlocks * Q8_0BlockBytes;

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Src, (nuint)fp16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQuant, (nuint)quantBytesVal).ThrowOnError();

        try
        {
            fixed (ushort* p = srcFp16)
                CudaDriverApi.cuMemcpyHtoD_v2(devFp16Src, (nint)p, (nuint)fp16Bytes).ThrowOnError();

            _kernels!.LaunchQuantKv(devFp16Src, devQuant, n, KvCacheDType.Q8_0, s);
            _stream!.Synchronize();

            byte[] quantHost = new byte[quantBytesVal];
            fixed (byte* p = quantHost)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devQuant, (nuint)quantBytesVal).ThrowOnError();

            // Correct CPU dequant
            float[] correct = new float[n];
            fixed (byte* qp = quantHost)
            fixed (float* op = correct)
            {
                DotLLM.Cpu.Kernels.KvQuantize.Q8_0ToF32Scalar(qp, op, n);
            }

            // Permuted dequant: swap adjacent pairs within each block
            float[] permuted = new float[n];
            fixed (byte* qp = quantHost)
            {
                for (int block = 0; block < numBlocks; block++)
                {
                    byte* blockSrc = qp + block * Q8_0BlockBytes;
                    float d = (float)Unsafe.ReadUnaligned<Half>(blockSrc);
                    sbyte* qs = (sbyte*)(blockSrc + 2);
                    for (int i = 0; i < BlockSize; i += 2)
                    {
                        permuted[block * BlockSize + i] = d * qs[i + 1];       // SWAPPED
                        permuted[block * BlockSize + i + 1] = d * qs[i];       // SWAPPED
                    }
                }
            }

            int mismatches = 0;
            for (int i = 0; i < n; i++)
            {
                if (MathF.Abs(correct[i] - permuted[i]) > 1e-6f)
                    mismatches++;
            }

            _output.WriteLine($"[Q8_0 Negative Control] pair swap: {mismatches}/{n} elements differ");

            Assert.True(mismatches > n / 4,
                $"Permuted Q8_0 dequant (pair swap) only differed at {mismatches}/{n} elements. " +
                $"The test data is too degenerate to catch permutation bugs.");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devFp16Src);
            CudaDriverApi.cuMemFree_v2(devQuant);
        }
    }

    // ─────────────────── Multi-row: simulates KV stride ───────────────────

    /// <summary>
    /// Round-trip test with kvStride = numKvHeads * headDim = 3 * 64 = 192 (6 blocks per row)
    /// to match realistic KV-cache dimensions. Tests multiple rows to verify no off-by-one
    /// in row addressing.
    /// </summary>
    [SkippableTheory]
    [InlineData(KvCacheDType.Q8_0)]
    [InlineData(KvCacheDType.Q4_0)]
    public void MultiRow_RandomData_RoundTrip_PreservesElementOrder(KvCacheDType dtype)
    {
        SkipIfUnavailable();

        int numKvHeads = 3;
        int headDim = 64;
        int kvStride = numKvHeads * headDim; // 192
        int numRows = 4;
        int totalElements = numRows * kvStride;
        int numBlocks = totalElements / BlockSize;
        nint s = _stream!.Handle;

        var rng = new Random(42);
        ushort[] srcFp16 = new ushort[totalElements];
        float[] srcF32 = new float[totalElements];
        for (int i = 0; i < totalElements; i++)
        {
            float val = (float)(rng.NextDouble() * 10.0 - 5.0);
            Half h = (Half)val;
            srcFp16[i] = BitConverter.HalfToUInt16Bits(h);
            srcF32[i] = (float)h;
        }

        long fp16Bytes = (long)totalElements * sizeof(ushort);
        int blockBytes = dtype == KvCacheDType.Q8_0 ? Q8_0BlockBytes : Q4_0BlockBytes;
        long quantBytesVal = (long)numBlocks * blockBytes;
        QuantizationType quantType = dtype == KvCacheDType.Q8_0
            ? QuantizationType.Q8_0
            : QuantizationType.Q4_0;

        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Src, (nuint)fp16Bytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devQuant, (nuint)quantBytesVal).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

        try
        {
            fixed (ushort* p = srcFp16)
                CudaDriverApi.cuMemcpyHtoD_v2(devFp16Src, (nint)p, (nuint)fp16Bytes).ThrowOnError();

            // Quantize row by row (matching CudaQuantizedKvCache.UpdateDevice pattern)
            int quantRowBytes = (kvStride / BlockSize) * blockBytes;
            for (int row = 0; row < numRows; row++)
            {
                nint rowSrc = devFp16Src + (nint)(row * kvStride * sizeof(ushort));
                nint rowDst = devQuant + (nint)(row * quantRowBytes);
                _kernels!.LaunchQuantKv(rowSrc, rowDst, kvStride, dtype, s);
            }

            // Dequantize all at once (matching PrepareAttentionScratch pattern)
            _kernels!.LaunchDequantToF16(devQuant, quantType, devFp16Dst, totalElements, s);
            _stream!.Synchronize();

            ushort[] dstFp16 = new ushort[totalElements];
            fixed (ushort* p = dstFp16)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devFp16Dst, (nuint)fp16Bytes).ThrowOnError();

            int mismatches = 0;
            float maxErr = 0;
            float quantStep = dtype == KvCacheDType.Q8_0 ? 127.0f : 7.0f;

            for (int i = 0; i < totalElements; i++)
            {
                float original = srcF32[i];
                float roundTripped = (float)BitConverter.UInt16BitsToHalf(dstFp16[i]);

                int block = i / BlockSize;
                float blockMaxAbs = 0;
                for (int j = block * BlockSize; j < (block + 1) * BlockSize; j++)
                    blockMaxAbs = MathF.Max(blockMaxAbs, MathF.Abs(srcF32[j]));

                float fpMargin = dtype == KvCacheDType.Q8_0 ? 0.002f : 0.01f;
                float tolerance = blockMaxAbs / quantStep * 1.15f + fpMargin;
                float err = MathF.Abs(original - roundTripped);
                if (err > maxErr) maxErr = err;
                if (err > tolerance)
                {
                    if (mismatches < 10)
                        _output.WriteLine($"  [{i}] row={i / kvStride} orig={original:F4} rt={roundTripped:F4} err={err:E3} tol={tolerance:E3}");
                    mismatches++;
                }
            }

            _output.WriteLine($"[{dtype} MultiRow] rows={numRows} kvStride={kvStride} maxErr={maxErr:E4} mismatches={mismatches}/{totalElements}");
            Assert.True(mismatches == 0,
                $"{dtype} multi-row round-trip: {mismatches}/{totalElements} elements exceed tolerance. maxErr={maxErr:E4}");
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devFp16Src);
            CudaDriverApi.cuMemFree_v2(devQuant);
            CudaDriverApi.cuMemFree_v2(devFp16Dst);
        }
    }

    // ─────────────────── Helpers ───────────────────

    private void SkipIfUnavailable()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(!_available, "PTX files not found or KV quant kernels not compiled");
    }

    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }
}
