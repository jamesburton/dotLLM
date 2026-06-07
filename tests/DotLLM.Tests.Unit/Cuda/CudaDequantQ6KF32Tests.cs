using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness tests for the GPU Q6_K → F32 dequant path that the kernel agent
/// just added to <c>LaunchDequantToF32</c>. Validates bit-for-bit equality
/// against the CPU <see cref="Dequantize.ToFloat32"/> oracle for both a single
/// synthetic super-block and a larger M×K tensor of random super-blocks.
///
/// <para>
/// <b>Bit-exact tolerance.</b> Q6_K dequant performs only multiplications of
/// scaled FP32 values (<c>d * scales[isc] * q</c>) — no transcendentals, no
/// FMA-eligible accumulations (the kernel uses three separate multiplies, not a
/// fused multiply-add). FP32 multiplication of independent operands is exactly
/// associative on the same hardware regardless of the parenthesisation, so the
/// CUDA and CPU paths must agree byte-for-byte. We assert ULP distance == 0.
/// </para>
/// </summary>
[Trait("Category", "GPU")]
public class CudaDequantQ6KF32Tests
{
    private const int Q6_K_BlockBytes = 210;
    private const int Q6_K_GroupSize = 256;

    private readonly ITestOutputHelper _out;
    public CudaDequantQ6KF32Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

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

    /// <summary>
    /// Single super-block (256 elements) round-trip: synthesise random Q6_K
    /// bytes, dequant on both CPU and GPU, assert byte-for-byte equality.
    /// </summary>
    [SkippableFact]
    public unsafe void Q6KF32_SingleBlock_BitExactWithCpuOracle()
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        const int ElementCount = Q6_K_GroupSize;
        const int BlockBytes = Q6_K_BlockBytes;

        byte[] packed = new byte[BlockBytes];
        SynthesiseQ6KBlock(new Random(0x12345), packed);

        // CPU oracle
        float[] cpu = new float[ElementCount];
        fixed (byte* pSrc = packed)
        {
            Dequantize.ToFloat32((nint)pSrc, ElementCount, QuantizationType.Q6_K, cpu);
        }

        // GPU
        nint dSrc = 0, dDst = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)BlockBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)(ElementCount * sizeof(float))).ThrowOnError();

            fixed (byte* pSrc = packed)
                CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)pSrc, (nuint)BlockBytes).ThrowOnError();

            kernels.LaunchDequantToF32(dSrc, QuantizationType.Q6_K, dDst, ElementCount, stream.Handle);
            stream.Synchronize();

            float[] gpu = new float[ElementCount];
            fixed (float* p = gpu)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)(ElementCount * sizeof(float))).ThrowOnError();

            AssertBitExact(cpu, gpu, "Q6_K → F32 (single block)");
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDst != 0) CudaDriverApi.cuMemFree_v2(dDst);
        }
    }

    /// <summary>
    /// Larger M×K-shape tensor (8 super-blocks per row, 64 rows = 16 384 elements,
    /// 64 super-blocks total). Exercises the kernel's grid-stride loop with a
    /// realistic distribution of super-blocks and proves the per-block formula
    /// is applied independently with no cross-block contamination.
    /// </summary>
    [SkippableTheory]
    [InlineData(64, 2048)]    // 64 rows × (2048/256 = 8) super-blocks = 512 SB
    [InlineData(128, 4096)]   // 2048 SB
    public unsafe void Q6KF32_FullTensor_BitExactWithCpuOracle(int rows, int k)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        Assert.Equal(0, k % Q6_K_GroupSize);
        int superblocksPerRow = k / Q6_K_GroupSize;
        long elementCount = (long)rows * k;
        long superblockCount = (long)rows * superblocksPerRow;
        long byteCount = superblockCount * Q6_K_BlockBytes;
        Assert.True(byteCount < int.MaxValue / 2, "Test budget too large");

        byte[] packed = new byte[byteCount];
        var rng = new Random(unchecked((int)0xDEADBEEF) ^ rows ^ (k << 8));
        for (long sb = 0; sb < superblockCount; sb++)
        {
            SynthesiseQ6KBlock(rng, packed.AsSpan((int)(sb * Q6_K_BlockBytes), Q6_K_BlockBytes));
        }

        // CPU oracle
        float[] cpu = new float[elementCount];
        fixed (byte* pSrc = packed)
        {
            Dequantize.ToFloat32((nint)pSrc, elementCount, QuantizationType.Q6_K, cpu);
        }

        // GPU
        nint dSrc = 0, dDst = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)byteCount).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)(elementCount * sizeof(float))).ThrowOnError();

            fixed (byte* pSrc = packed)
                CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)pSrc, (nuint)byteCount).ThrowOnError();

            // LaunchDequantToF32 takes int totalElements — verify we stay in range
            Assert.True(elementCount < int.MaxValue, "elementCount overflows int");
            kernels.LaunchDequantToF32(dSrc, QuantizationType.Q6_K, dDst, (int)elementCount, stream.Handle);
            stream.Synchronize();

            float[] gpu = new float[elementCount];
            fixed (float* p = gpu)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)(elementCount * sizeof(float))).ThrowOnError();

            AssertBitExact(cpu, gpu, $"Q6_K → F32 ({rows}×{k})");
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDst != 0) CudaDriverApi.cuMemFree_v2(dDst);
        }
    }

    /// <summary>
    /// Build a random but valid Q6_K block. Layout matches
    /// <c>DequantizeKQuants.DequantizeQ6_KScalar</c>:
    ///   ql[128] @ 0   — packed lower nibbles
    ///   qh[64]  @ 128 — packed upper 2-bit fragments
    ///   scales[16] @ 192 — per-sub-block signed int8 scales
    ///   d (Half) @ 208 — super-block delta
    /// </summary>
    private static unsafe void SynthesiseQ6KBlock(Random rng, Span<byte> block)
    {
        Assert.Equal(Q6_K_BlockBytes, block.Length);
        // d in a modest range to avoid FP16 overflow in scales[isc]*q*d (q ∈ [-32, 31],
        // scales ∈ [-128, 127], so up to ~4096 × |d|).
        Half d = (Half)(rng.NextDouble() * 0.02 + 0.005);
        for (int i = 0; i < 128; i++) block[i] = (byte)rng.Next(0, 256);           // ql
        for (int i = 128; i < 192; i++) block[i] = (byte)rng.Next(0, 256);         // qh
        for (int i = 192; i < 208; i++) block[i] = (byte)rng.Next(0, 256);         // scales — any byte reinterpreted as int8
        fixed (byte* p = block)
            *(Half*)(p + 208) = d;
    }

    /// <summary>
    /// Pure-multiply Q6_K → F32 dequant should be bit-identical between CUDA
    /// and the CPU scalar reference. Print the worst observed delta on failure
    /// so the maintainer can see whether it's a rounding inconsistency (≤ 1 ULP)
    /// versus a structural bug.
    /// </summary>
    private void AssertBitExact(float[] cpu, float[] gpu, string label)
    {
        Assert.Equal(cpu.Length, gpu.Length);
        long mismatchCount = 0;
        int firstMismatch = -1;
        float worstAbs = 0f;
        long worstUlp = 0;
        for (int i = 0; i < cpu.Length; i++)
        {
            int ci = BitConverter.SingleToInt32Bits(cpu[i]);
            int gi = BitConverter.SingleToInt32Bits(gpu[i]);
            if (ci != gi)
            {
                if (firstMismatch < 0) firstMismatch = i;
                mismatchCount++;
                float abs = MathF.Abs(cpu[i] - gpu[i]);
                if (abs > worstAbs) worstAbs = abs;
                // Sign-magnitude ULP distance.
                int ai = ci, bi = gi;
                if (ai < 0) ai = int.MinValue - ai;
                if (bi < 0) bi = int.MinValue - bi;
                long u = Math.Abs((long)ai - (long)bi);
                if (u > worstUlp) worstUlp = u;
            }
        }

        _out.WriteLine($"{label}: n={cpu.Length}, mismatches={mismatchCount}/{cpu.Length}, worst-abs={worstAbs:E3}, worst-ulp={worstUlp}");

        Assert.True(mismatchCount == 0,
            $"{label}: {mismatchCount} of {cpu.Length} elements differ. " +
            $"First mismatch at [{firstMismatch}]: cpu={(firstMismatch < 0 ? 0f : cpu[firstMismatch])}, " +
            $"gpu={(firstMismatch < 0 ? 0f : gpu[firstMismatch])}. worst-ulp={worstUlp}, worst-abs={worstAbs:E3}.");
    }
}
