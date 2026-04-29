using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates that <see cref="CudaKernels.LaunchQuantizedGemv"/> for block-32
/// quants (Q8_0, Q5_0) handles non-256-aligned K values like K=1408 — the
/// V2-Lite <c>ffn_down_exps</c> shape (M=hidden=2048, K=intermediate=1408,
/// stored as Q8_0 in Q4_K_M / Q5_0 in Q3_K_M).
/// </summary>
/// <remarks>
/// Pre-fix: <c>ProjectF32OrQuant</c> gated the GEMV fast path on
/// <c>K % 256 == 0</c>, which locked Q8_0/Q5_0 down_proj into the
/// dequant-then-GEMM fallback (an order-of-magnitude more bandwidth per call).
/// Post-fix: the gate uses <see cref="CudaKernels.MinKAlignmentFor"/> (32 for
/// block-32, 256 for K-quants), unlocking the fast path for down_proj.
/// </remarks>
[Trait("Category", "GPU")]
public class CudaQuantizedGemvAlignmentTests
{
    private readonly ITestOutputHelper _out;
    public CudaQuantizedGemvAlignmentTests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    [Fact]
    public void MinKAlignment_BlockQuants_32_KQuants_256()
    {
        Assert.Equal(32, CudaKernels.MinKAlignmentFor(QuantizationType.Q8_0));
        Assert.Equal(32, CudaKernels.MinKAlignmentFor(QuantizationType.Q5_0));
        Assert.Equal(32, CudaKernels.MinKAlignmentFor(QuantizationType.Q4_0));
        Assert.Equal(32, CudaKernels.MinKAlignmentFor(QuantizationType.Q4_1));
        Assert.Equal(32, CudaKernels.MinKAlignmentFor(QuantizationType.Q5_1));
        Assert.Equal(32, CudaKernels.MinKAlignmentFor(QuantizationType.IQ4_NL));
        Assert.Equal(256, CudaKernels.MinKAlignmentFor(QuantizationType.Q4_K));
        Assert.Equal(256, CudaKernels.MinKAlignmentFor(QuantizationType.Q5_K));
        Assert.Equal(256, CudaKernels.MinKAlignmentFor(QuantizationType.Q6_K));
        Assert.Equal(256, CudaKernels.MinKAlignmentFor(QuantizationType.Q3_K));
        Assert.Equal(256, CudaKernels.MinKAlignmentFor(QuantizationType.Q2_K));
        Assert.Equal(256, CudaKernels.MinKAlignmentFor(QuantizationType.IQ4_XS));
    }

    [Theory]
    [InlineData(QuantizationType.IQ4_NL)]
    [InlineData(QuantizationType.IQ4_XS)]
    public void HasLoadedQuantizedGemv_DoesNotTreatStaticIQ4SupportAsRuntimeCapability(QuantizationType qt)
    {
        Assert.True(CudaKernels.HasQuantizedGemv(qt));

        // Simulates stale PTX: the type is supported by metadata, but no runtime
        // GEMV function pointer was resolved. This avoids checking in stale PTX.
        var kernels = (CudaKernels)RuntimeHelpers.GetUninitializedObject(typeof(CudaKernels));

        Assert.False(kernels.HasQuantizedGemvKernel(qt));
        Assert.False(kernels.HasLoadedQuantizedGemv(qt));
    }

    /// <summary>
    /// Q8_0 GEMV at K=1408 (V2-Lite down_proj shape) must produce results that
    /// match a scalar reference (dequant-each-block × FP32 dot) within FP16
    /// rounding tolerance. Previously this path was blocked by the K%256 gate;
    /// this test pins the kernel-level correctness so a future regression is
    /// caught even before the model-level smoke runs.
    /// </summary>
    [SkippableTheory]
    [InlineData(QuantizationType.Q8_0, 2048, 1408, 34)]  // V2-Lite Q4_K_M down_proj
    [InlineData(QuantizationType.Q5_0, 2048, 1408, 22)]  // V2-Lite Q3_K_M down_proj
    [InlineData(QuantizationType.Q8_0, 256, 1408, 34)]   // narrow M
    [InlineData(QuantizationType.Q5_0, 256, 1408, 22)]   // narrow M
    public void GemvAtK1408_BlockQuants_MatchesScalarReference(
        QuantizationType qt, int M, int K, int blockBytes)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunGemvVsScalar(qt, M, K, blockBytes);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunGemvVsScalar(QuantizationType qt, int M, int K, int blockBytes)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasQuantizedGemvKernel(qt), $"No GEMV kernel loaded for {qt}");
        Assert.Equal(0, K % CudaKernels.MinKAlignmentFor(qt));

        int blocksPerRow = K / 32;
        long rowBytes = (long)blocksPerRow * blockBytes;
        long weightBytes = (long)M * rowBytes;
        var rng = new Random(0xC0FFEE ^ (int)qt ^ M ^ K);

        // Synthesise per-block: half d, then qs[32] (Q8_0 = int8, Q5_0 = nibble + 5th bits).
        byte[] hostW = new byte[weightBytes];
        var ws = hostW.AsSpan();
        for (int row = 0; row < M; row++)
        for (int b = 0; b < blocksPerRow; b++)
        {
            var blk = ws.Slice(row * (int)rowBytes + b * blockBytes, blockBytes);
            Half d = (Half)((rng.NextDouble() - 0.5) * 0.04);
            fixed (byte* pBlk = blk) *(Half*)pBlk = d;
            for (int j = 2; j < blockBytes; j++) blk[j] = (byte)rng.Next(0, 256);
        }

        Half[] hostX = new Half[K];
        for (int i = 0; i < K; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            hostX[i] = (Half)(g * 0.4);
        }

        // Scalar reference: dequant each block on host, FP32 dot.
        float[] yRef = new float[M];
        ComputeScalarReference(qt, hostW, hostX, M, K, blockBytes, yRef);

        long xBytes = (long)K * sizeof(ushort);
        long yBytes = (long)M * sizeof(ushort);
        nint devW = 0, devX = 0, devY = 0;
        Half[] yGpu = new Half[M];
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out devW, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devY, (nuint)yBytes).ThrowOnError();
            fixed (byte* pW = hostW)
                CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)pW, (nuint)weightBytes).ThrowOnError();
            fixed (Half* pX = hostX)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();
            kernels.LaunchQuantizedGemv(devW, qt, devX, devY, M, K, stream.Handle);
            stream.Synchronize();
            fixed (Half* pY = yGpu)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, devY, (nuint)yBytes).ThrowOnError();
        }
        finally
        {
            if (devW != 0) CudaDriverApi.cuMemFree_v2(devW);
            if (devX != 0) CudaDriverApi.cuMemFree_v2(devX);
            if (devY != 0) CudaDriverApi.cuMemFree_v2(devY);
        }

        // Compare. FP16 output rounding + tree-vs-linear FP32 reduction allow ~1e-2.
        float maxAbs = 0f, refMax = 0f;
        for (int i = 0; i < M; i++)
        {
            float diff = MathF.Abs((float)yGpu[i] - yRef[i]);
            if (diff > maxAbs) maxAbs = diff;
            if (MathF.Abs(yRef[i]) > refMax) refMax = MathF.Abs(yRef[i]);
        }
        _out.WriteLine($"{qt} M={M} K={K}: ref|max|={refMax:F3}  max-abs-diff={maxAbs:F5}");
        Assert.True(maxAbs < 0.05f,
            $"GEMV at K={K} for {qt} diverges from scalar reference (max-abs-diff={maxAbs}, refMax={refMax}).");
    }

    private static unsafe void ComputeScalarReference(
        QuantizationType qt, byte[] weight, Half[] x, int M, int K, int blockBytes, float[] y)
    {
        int blocksPerRow = K / 32;
        long rowBytes = (long)blocksPerRow * blockBytes;
        for (int row = 0; row < M; row++)
        {
            float acc = 0f;
            for (int b = 0; b < blocksPerRow; b++)
            {
                int blkOff = row * (int)rowBytes + b * blockBytes;
                Half d;
                fixed (byte* pW = weight) d = *(Half*)(pW + blkOff);
                float dF = (float)d;
                if (qt == QuantizationType.Q8_0)
                {
                    for (int j = 0; j < 32; j++)
                    {
                        sbyte q = (sbyte)weight[blkOff + 2 + j];
                        acc += dF * q * (float)x[b * 32 + j];
                    }
                }
                else // Q5_0: 6 byte half d + 4 byte qh + 16 byte qs (nibbles, 5th bit from qh)
                {
                    uint qh;
                    fixed (byte* pW = weight) qh = *(uint*)(pW + blkOff + 2);
                    for (int j = 0; j < 16; j++)
                    {
                        byte qs = weight[blkOff + 2 + 4 + j];
                        int xl = (qs & 0x0F) | (int)((qh >> j) & 1) << 4;
                        int xh = (qs >> 4)   | (int)((qh >> (j + 16)) & 1) << 4;
                        // Q5_0 stores 5-bit values in [0, 31], symmetric quant: signed = q - 16.
                        int sl = xl - 16;
                        int sh = xh - 16;
                        acc += dF * sl * (float)x[b * 32 + j];
                        acc += dF * sh * (float)x[b * 32 + j + 16];
                    }
                }
            }
            y[row] = acc;
        }
    }

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

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunGemvVsScalarKQuant(QuantizationType qt, int M, int K, int blockBytes)
    {
        // Test infrastructure for K-quants (block_size=256). Mirrors RunGemvVsScalar
        // structurally but iterates over superblocks (K/256) per row, dequants the
        // whole row to F32 via Dequantize.ToFloat32 as the scalar oracle.
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasQuantizedGemvKernel(qt), $"No GEMV kernel loaded for {qt}");
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
                    DotLLM.Cpu.Kernels.Dequantize.ToFloat32((nint)(p + row * rowBytes), K, qt, rowDequant);
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
            $"K-quant GEMV diverges from scalar reference (max-abs-diff={maxAbs}, refMax={refMax}).");
    }

    [SkippableTheory]
    [InlineData(QuantizationType.IQ4_NL, 512, 1408, 18)]
    [InlineData(QuantizationType.IQ4_NL, 256, 32, 18)]
    [InlineData(QuantizationType.IQ4_XS, 512, 1024, 136)]
    [InlineData(QuantizationType.IQ4_XS, 256, 256, 136)]
    [InlineData(QuantizationType.IQ4_XS, 128256, 4096, 136)]
    public void GemvIQ4MatchesScalarReference(
        QuantizationType qt, int M, int K, int blockBytes)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        RunGemvVsDequantScalar(qt, M, K, blockBytes);
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private unsafe void RunGemvVsDequantScalar(QuantizationType qt, int M, int K, int blockBytes)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");
        using var kernels = new CudaKernels(ptxDir!);
        Skip.IfNot(kernels.HasQuantizedGemvKernel(qt), $"No GEMV kernel loaded for {qt}");
        Assert.Equal(0, K % CudaKernels.MinKAlignmentFor(qt));

        int blockSize = qt == QuantizationType.IQ4_NL ? 32 : 256;
        int blocksPerRow = K / blockSize;
        long rowBytes = (long)blocksPerRow * blockBytes;
        long weightBytes = (long)M * rowBytes;
        var rng = new Random(0x1A4D ^ (int)qt ^ M ^ K);

        byte[] hostW = new byte[weightBytes];
        rng.NextBytes(hostW);
        fixed (byte* p = hostW)
        {
            for (int row = 0; row < M; row++)
            for (int b = 0; b < blocksPerRow; b++)
            {
                byte* blk = p + row * rowBytes + b * blockBytes;
                *(Half*)blk = qt == QuantizationType.IQ4_NL
                    ? (Half)((rng.NextDouble() - 0.5) * 0.04)
                    : (Half)((rng.NextDouble() - 0.5) * 0.002);
            }
        }

        Half[] hostX = new Half[K];
        for (int i = 0; i < K; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            hostX[i] = (Half)(g * 0.4);
        }

        float[] yRef = new float[M];
        float[] xF32 = new float[K];
        for (int i = 0; i < K; i++) xF32[i] = (float)hostX[i];
        fixed (byte* p = hostW)
        {
            float[] rowDequant = new float[K];
            for (int row = 0; row < M; row++)
            {
                Dequantize.ToFloat32((nint)(p + row * rowBytes), K, qt, rowDequant);
                float acc = 0;
                for (int i = 0; i < K; i++) acc += rowDequant[i] * xF32[i];
                yRef[row] = acc;
            }
        }

        long xBytes = (long)K * sizeof(ushort);
        long yBytes = (long)M * sizeof(ushort);
        nint devW = 0, devX = 0, devY = 0;
        Half[] yGpu = new Half[M];
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out devW, (nuint)weightBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out devY, (nuint)yBytes).ThrowOnError();
            fixed (byte* pW = hostW)
                CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)pW, (nuint)weightBytes).ThrowOnError();
            fixed (Half* pX = hostX)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)pX, (nuint)xBytes).ThrowOnError();
            kernels.LaunchQuantizedGemv(devW, qt, devX, devY, M, K, stream.Handle);
            stream.Synchronize();
            fixed (Half* p = yGpu)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, devY, (nuint)yBytes).ThrowOnError();
        }
        finally
        {
            if (devW != 0) CudaDriverApi.cuMemFree_v2(devW);
            if (devX != 0) CudaDriverApi.cuMemFree_v2(devX);
            if (devY != 0) CudaDriverApi.cuMemFree_v2(devY);
        }

        float maxAbs = 0f, refMax = 0f;
        for (int i = 0; i < M; i++)
        {
            float diff = MathF.Abs((float)yGpu[i] - yRef[i]);
            if (diff > maxAbs) maxAbs = diff;
            if (MathF.Abs(yRef[i]) > refMax) refMax = MathF.Abs(yRef[i]);
        }

        _out.WriteLine($"{qt} M={M} K={K}: ref|max|={refMax:F3} max-abs-diff={maxAbs:F5}");
        Assert.True(maxAbs < 0.1f,
            $"IQ GEMV diverges from scalar reference (max-abs-diff={maxAbs}, refMax={refMax}).");
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
}
