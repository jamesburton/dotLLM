using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Correctness test for the CUDA <c>dequant_q3_k_f16</c> kernel, against the CPU
/// <see cref="Dequantize.ToFloat32"/> oracle, over <b>random raw block bytes</b>.
/// </summary>
/// <remarks>
/// <para>
/// Issue #311 found the Q3_K bit layout was transposed in every backend, in two
/// independent ways (the 6-bit scale high-bit byte/shift pair, and the element →
/// <c>qs</c>/<c>hmask</c> bit mapping). The kernel-level tests that existed then all
/// ran through a fixture that <i>encoded</i> with the same transposed layout it
/// decoded with — a closed loop that could never fail.
/// </para>
/// <para>
/// This test avoids that trap by never encoding anything: it feeds the same
/// <b>random</b> 110-byte blocks to both the CPU scalar reference and the GPU kernel.
/// Random bytes populate every bit position, so the two transpositions produce
/// grossly different values (they permute elements into the wrong sub-block scale)
/// and the test cannot pass unless the GPU layout matches the CPU one bit for bit.
/// </para>
/// <para>
/// <b>Bit-exactness.</b> Both paths evaluate <c>(d * signedScale) * signed3</c> —
/// pure FP32 multiplication in the same order, no FMA-eligible accumulation and no
/// transcendentals (so <c>--use_fast_math</c> on this translation unit is
/// irrelevant), then round the FP32 result to FP16 with round-to-nearest-even. The
/// results must therefore be identical FP16 bit patterns.
/// </para>
/// <para>
/// Issue #318: this also serves as the on-hardware check that the committed
/// <c>native/ptx/dequant.ptx</c> carries the #311-fixed kernel, not a stale build.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public class CudaDequantQ3KF16Tests
{
    private const int Q3_K_BlockBytes = 110;
    private const int Q3_K_GroupSize = 256;

    private readonly ITestOutputHelper _out;
    public CudaDequantQ3KF16Tests(ITestOutputHelper output) => _out = output;

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

    private static string? FindPtxDir()
    {
        string[] candidates =
        [
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        ];
        foreach (string dir in candidates)
        {
            string full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    [SkippableTheory]
    [InlineData(1)]     // single super-block
    [InlineData(64)]    // grid-stride over many super-blocks
    [InlineData(517)]   // non-power-of-two count
    public unsafe void Q3KF16_RandomBlocks_MatchesCpuOracle(int superblockCount)
    {
        Skip.IfNot(IsCudaDriverPresent(), "No CUDA GPU available");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir == null, "PTX files not found");

        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);

        int elementCount = superblockCount * Q3_K_GroupSize;
        int byteCount = superblockCount * Q3_K_BlockBytes;

        byte[] packed = new byte[byteCount];
        var rng = new Random(0x03180311 ^ superblockCount);
        for (int sb = 0; sb < superblockCount; sb++)
            SynthesiseQ3KBlock(rng, packed.AsSpan(sb * Q3_K_BlockBytes, Q3_K_BlockBytes));

        float[] cpu = new float[elementCount];
        fixed (byte* pSrc = packed)
            Dequantize.ToFloat32((nint)pSrc, elementCount, QuantizationType.Q3_K, cpu);

        nint dSrc = 0, dDst = 0;
        try
        {
            CudaDriverApi.cuMemAlloc_v2(out dSrc, (nuint)byteCount).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out dDst, (nuint)(elementCount * sizeof(ushort))).ThrowOnError();

            fixed (byte* pSrc = packed)
                CudaDriverApi.cuMemcpyHtoD_v2(dSrc, (nint)pSrc, (nuint)byteCount).ThrowOnError();

            kernels.LaunchDequantToF16(dSrc, QuantizationType.Q3_K, dDst, elementCount, stream.Handle);
            stream.Synchronize();

            ushort[] gpuBits = new ushort[elementCount];
            fixed (ushort* p = gpuBits)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dDst, (nuint)(elementCount * sizeof(ushort))).ThrowOnError();

            AssertMatchesOracle(cpu, gpuBits, $"Q3_K → F16 ({superblockCount} super-blocks)");
        }
        finally
        {
            if (dSrc != 0) CudaDriverApi.cuMemFree_v2(dSrc);
            if (dDst != 0) CudaDriverApi.cuMemFree_v2(dDst);
        }
    }

    /// <summary>
    /// Fill a 110-byte Q3_K block with random bytes. Every byte pattern is a valid
    /// block (<c>hmask[32] | qs[64] | scales[12]</c> are raw bit fields), so no
    /// encoder — and therefore no shared-layout assumption — is involved. Only the
    /// FP16 super-block delta is constrained, to a modest positive magnitude, so
    /// that <c>d * scale * q</c> (|scale| ≤ 32, |q| ≤ 4) stays well inside FP16
    /// range and the comparison tests the layout rather than overflow behaviour.
    /// </summary>
    private static unsafe void SynthesiseQ3KBlock(Random rng, Span<byte> block)
    {
        Assert.Equal(Q3_K_BlockBytes, block.Length);
        for (int i = 0; i < 108; i++)
            block[i] = (byte)rng.Next(0, 256);
        Half d = (Half)((rng.NextDouble() * 0.02) + 0.005);
        fixed (byte* p = block)
            *(Half*)(p + 108) = d;
    }

    /// <summary>
    /// Compares the GPU FP16 output against the CPU FP32 oracle rounded to FP16.
    /// Also reports Pearson correlation, which is the statistic that exposed the
    /// #311 transposition (0.006 broken vs 0.988 fixed) — so a failure message
    /// distinguishes "wrong layout" from "1 ULP rounding drift" at a glance.
    /// </summary>
    private void AssertMatchesOracle(float[] cpu, ushort[] gpuBits, string label)
    {
        Assert.Equal(cpu.Length, gpuBits.Length);

        long mismatchCount = 0;
        int firstMismatch = -1;
        float worstAbs = 0f;
        double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;

        for (int i = 0; i < cpu.Length; i++)
        {
            ushort expectedBits = BitConverter.HalfToUInt16Bits((Half)cpu[i]);
            if (expectedBits != gpuBits[i])
            {
                if (firstMismatch < 0) firstMismatch = i;
                mismatchCount++;
                worstAbs = MathF.Max(worstAbs, MathF.Abs(cpu[i] - (float)BitConverter.UInt16BitsToHalf(gpuBits[i])));
            }

            double x = cpu[i];
            double y = (float)BitConverter.UInt16BitsToHalf(gpuBits[i]);
            sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
        }

        int n = cpu.Length;
        double cov = (sxy / n) - ((sx / n) * (sy / n));
        double varX = (sxx / n) - ((sx / n) * (sx / n));
        double varY = (syy / n) - ((sy / n) * (sy / n));
        double corr = (varX > 0 && varY > 0) ? cov / Math.Sqrt(varX * varY) : double.NaN;

        _out.WriteLine($"{label}: n={n}, mismatches={mismatchCount}, worst-abs={worstAbs:E3}, corr={corr:F6}");

        Assert.True(mismatchCount == 0,
            $"{label}: {mismatchCount} of {n} elements differ from the CPU oracle. " +
            $"First at [{firstMismatch}]: cpu={(firstMismatch < 0 ? 0f : cpu[firstMismatch])}, " +
            $"gpu={(firstMismatch < 0 ? 0f : (float)BitConverter.UInt16BitsToHalf(gpuBits[firstMismatch]))}. " +
            $"worst-abs={worstAbs:E3}, correlation={corr:F6}. " +
            "A correlation far below 1 means the GPU Q3_K bit layout disagrees with the CPU one " +
            "(see #311) — most likely native/ptx/dequant.ptx is a stale build of native/kernels/dequant.cu.");
    }
}
