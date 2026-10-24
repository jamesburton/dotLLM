using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #403: the expert-grouped GEMV (<c>moe_eg_gemv_*</c>) and the slot-ordered weighted sum
/// (<c>moe_weighted_slot_sum_f32</c>) of <c>nemotron_moe.ptx</c>, against a CPU reference that dequantises
/// the same random blocks with dotLLM's CPU dequantiser (<see cref="Dequantize.ToFloat32"/>) and accumulates
/// in double.
/// </summary>
/// <remarks>
/// Every case uses three groups of 1, 3 and 17 slots over a four-expert bank whose group pointers are out of
/// order (experts 2, 0, 3; expert 1 unused), a shuffled slot list, and an <c>x_row</c> map that is not the slot
/// id, so a kernel that ignores the group pointer, the slot order or the row map writes the wrong row.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronMoeKernelTests(ITestOutputHelper output)
{
    private static readonly int[] GroupSizes = [1, 3, 17];
    private static readonly int[] GroupExperts = [2, 0, 3];
    private const int BankExperts = 4;
    private const int XRows = 5;

    public static IEnumerable<object[]> Cases()
    {
        QuantizationType[] types =
            [QuantizationType.Q4_K, QuantizationType.Q5_K, QuantizationType.Q6_K, QuantizationType.Q8_0, QuantizationType.IQ2_XXS];
        foreach (var qt in types)
            foreach (int k in new[] { 256, 1024, 2048 })
                foreach (int m in new[] { 1, 37, 2048 })
                    yield return [qt, m, k];
    }

    /// <summary>Each bank type at K in {256, 1024, 2048} and M in {1, 37, 2048}, with and without the fused relu².</summary>
    [SkippableTheory]
    [MemberData(nameof(Cases))]
    public void ExpertGroupedGemv_MatchesCpuDequantDoubleReference(QuantizationType qt, int m, int k)
    {
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        var rng = new Random(403 + (int)qt * 7919 + m * 31 + k);
        long rowBytes = Dequantize.RowByteSize(k, qt);
        long expertBytes = rowBytes * m;
        byte[] bank = RandomBlocks(rng, qt, checked((int)(expertBytes * BankExperts)));

        int slots = GroupSizes.Sum();
        int[] slotIds = Enumerable.Range(0, slots).OrderBy(_ => rng.Next()).ToArray(); // ordered by group, ids shuffled
        int[] groupStart = new int[GroupSizes.Length + 1];
        for (int g = 0; g < GroupSizes.Length; g++) groupStart[g + 1] = groupStart[g] + GroupSizes[g];
        int[] xRow = new int[slots];
        for (int s = 0; s < slots; s++) xRow[s] = (s * 3 + 1) % XRows; // not the slot id
        Assert.Contains(Enumerable.Range(0, slots), s => xRow[s] != s);
        float[] x = new float[XRows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        // CPU reference: dequantise each used expert's rows with the CPU dequantiser, dot in double.
        var refPre = new double[slots * m];
        var l2 = new double[slots * m];
        for (int g = 0; g < GroupSizes.Length; g++)
        {
            float[] w = DequantExpert(bank, GroupExperts[g], expertBytes, rowBytes, qt, m, k);
            for (int j = groupStart[g]; j < groupStart[g + 1]; j++)
            {
                int s = slotIds[j];
                for (int r = 0; r < m; r++)
                {
                    double acc = 0, sq = 0;
                    for (int c = 0; c < k; c++)
                    {
                        double term = (double)w[(long)r * k + c] * x[xRow[s] * k + c];
                        acc += term;
                        sq += term * term;
                    }
                    refPre[s * m + r] = acc;
                    l2[s * m + r] = Math.Sqrt(sq);
                }
            }
        }

        foreach (bool relu2 in new[] { false, true })
        {
            float[] gpu = RunGroupedGemv(ptxDir!, qt, bank, expertBytes, slotIds, groupStart, xRow, x, m, k, slots, relu2);

            // F32 accumulation in a different order than the double reference: bound by sqrt(K) times the L2 norm of the
            // products (the error grows with the term magnitudes, not with the possibly cancelling sum). A layout bug moves
            // an output by O(L2), far above this bound.
            double worstRatio = 0;
            for (int i = 0; i < slots * m; i++)
            {
                double pre = refPre[i];
                double expected = relu2 ? Math.Max(pre, 0) * Math.Max(pre, 0) : pre;
                double tolPre = 1e-6 * Math.Sqrt(k) * l2[i] + 1e-6;
                double tol = relu2 ? 2 * Math.Abs(pre) * tolPre + tolPre * tolPre : tolPre;
                double diff = Math.Abs(gpu[i] - expected);
                Assert.False(float.IsNaN(gpu[i]), $"slot row {i} not written (NaN sentinel) for {qt} m={m} k={k}");
                Assert.True(diff <= tol,
                    $"{qt} m={m} k={k} relu2={relu2}: Y[slot {i / m}, row {i % m}] = {gpu[i]:G9}, expected {expected:G9} (|diff| {diff:E3} > {tol:E3})");
                worstRatio = Math.Max(worstRatio, diff / tol);
            }
            output.WriteLine($"{qt} m={m} k={k} relu2={relu2}: worst |diff|/tol = {worstRatio:F4}");
        }
    }

    /// <summary>The down-projection form: no <c>x_row</c> map, so each slot reads its own row of X.</summary>
    [SkippableFact]
    public void ExpertGroupedGemv_NullRowMap_ReadsTheSlotsOwnRow()
    {
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        const QuantizationType qt = QuantizationType.Q5_K;
        const int m = 37, k = 512;
        var rng = new Random(1403);
        long rowBytes = Dequantize.RowByteSize(k, qt);
        long expertBytes = rowBytes * m;
        byte[] bank = RandomBlocks(rng, qt, checked((int)(expertBytes * BankExperts)));
        int slots = GroupSizes.Sum();
        int[] slotIds = Enumerable.Range(0, slots).Reverse().ToArray();
        int[] groupStart = [0, 1, 4, slots];
        float[] x = new float[slots * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        float[] gpu = RunGroupedGemv(ptxDir!, qt, bank, expertBytes, slotIds, groupStart, xRow: null, x, m, k, slots, relu2: false);
        for (int g = 0; g < 3; g++)
        {
            float[] w = DequantExpert(bank, GroupExperts[g], expertBytes, rowBytes, qt, m, k);
            for (int j = groupStart[g]; j < groupStart[g + 1]; j++)
            {
                int s = slotIds[j];
                for (int r = 0; r < m; r++)
                {
                    double acc = 0, sq = 0;
                    for (int c = 0; c < k; c++) { double t = (double)w[r * k + c] * x[s * k + c]; acc += t; sq += t * t; }
                    double tol = 1e-6 * Math.Sqrt(k) * Math.Sqrt(sq) + 1e-6;
                    Assert.True(Math.Abs(gpu[s * m + r] - acc) <= tol, $"slot {s} row {r}: {gpu[s * m + r]:G9} vs {acc:G9}");
                }
            }
        }
    }

    /// <summary>The weighted slot sum: acc[t, d] = sum over k in slot order of w[t*topK+k] * Y[t*topK+k, d], bit-exact against the
    /// same float operations on the CPU (the kernel forbids FMA contraction).</summary>
    [SkippableFact]
    public unsafe void WeightedSlotSum_MatchesCpuSlotOrderBitExact()
    {
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");

        const int nTok = 7, topK = 3, d = 300;
        var rng = new Random(2403);
        float[] y = new float[nTok * topK * d];
        float[] w = new float[nTok * topK];
        for (int i = 0; i < y.Length; i++) y[i] = (float)(rng.NextDouble() * 2 - 1) * 3f;
        for (int i = 0; i < w.Length; i++) w[i] = (float)rng.NextDouble() * 2.5f;

        float[] expected = new float[nTok * d];
        for (int t = 0; t < nTok; t++)
            for (int i = 0; i < d; i++)
            {
                float a = 0f;
                for (int k = 0; k < topK; k++)
                {
                    float prod = w[t * topK + k] * y[(t * topK + k) * d + i];
                    a += prod;
                }
                expected[t * d + i] = a;
            }

        float[] gpu = new float[nTok * d];
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir!);
        nint dy = Upload(y), dw = Upload(w), dacc = Alloc((long)gpu.Length * sizeof(float));
        try
        {
            kernels.LaunchMoeWeightedSlotSumF32(dy, dw, dacc, nTok, topK, d, stream.Handle);
            stream.Synchronize();
            fixed (float* p = gpu)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dacc, (nuint)(gpu.Length * sizeof(float))).ThrowOnError();
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dy); CudaDriverApi.cuMemFree_v2(dw); CudaDriverApi.cuMemFree_v2(dacc);
        }
        for (int i = 0; i < gpu.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(gpu[i]),
                $"acc[{i / d}, {i % d}]: cpu {expected[i]:G9} gpu {gpu[i]:G9}");
    }

    [Fact]
    public void ExpertGroupedGemv_SupportedTypesAndKMultiples()
    {
        // Static capability table only: no device work.
        Assert.True(CudaKernels.SupportsMoeExpertGroupedGemv(QuantizationType.IQ2_XXS));
        Assert.False(CudaKernels.SupportsMoeExpertGroupedGemv(QuantizationType.IQ4_XS));
        Assert.False(CudaKernels.SupportsMoeExpertGroupedGemv(QuantizationType.F16));
        Assert.Equal(32, CudaKernels.MoeExpertGroupedKMultiple(QuantizationType.Q8_0));
        Assert.Equal(256, CudaKernels.MoeExpertGroupedKMultiple(QuantizationType.Q4_K));
    }

    // ───────────────────────── helpers ─────────────────────────

    private unsafe float[] RunGroupedGemv(string ptxDir, QuantizationType qt, byte[] bank, long expertBytes,
        int[] slotIds, int[] groupStart, int[]? xRow, float[] x, int m, int k, int slots, bool relu2)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);

        var allocs = new List<nint>();
        try
        {
            nint dBank = Alloc(bank.Length); allocs.Add(dBank);
            fixed (byte* p = bank) CudaDriverApi.cuMemcpyHtoD_v2(dBank, (nint)p, (nuint)bank.Length).ThrowOnError();
            long[] ptrs = GroupExperts.Select(e => (long)dBank + e * expertBytes).ToArray();
            nint dPtrs = Upload(ptrs); allocs.Add(dPtrs);
            nint dGs = Upload(groupStart); allocs.Add(dGs);
            nint dSlots = Upload(slotIds); allocs.Add(dSlots);
            nint dXRow = 0;
            if (xRow is not null) { dXRow = Upload(xRow); allocs.Add(dXRow); }
            nint dX = Upload(x); allocs.Add(dX);
            long yBytes = (long)slots * m * sizeof(float);
            nint dY = Alloc(yBytes); allocs.Add(dY);
            CudaDriverApi.cuMemsetD8_v2(dY, 0xFF, (nuint)yBytes).ThrowOnError(); // NaN sentinel: every slot row must be written

            kernels.LaunchMoeExpertGroupedGemvF32(qt, dPtrs, dGs, dSlots, dXRow, dX, dY, m, k, GroupSizes.Length, relu2, stream.Handle);
            stream.Synchronize();

            float[] y = new float[slots * m];
            fixed (float* p = y) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dY, (nuint)yBytes).ThrowOnError();
            return y;
        }
        finally
        {
            foreach (nint p in allocs) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    private static unsafe float[] DequantExpert(byte[] bank, int expert, long expertBytes, long rowBytes,
        QuantizationType qt, int m, int k)
    {
        float[] w = new float[(long)m * k];
        fixed (byte* p = bank)
            for (int r = 0; r < m; r++)
                Dequantize.ToFloat32((nint)(p + expert * expertBytes + r * rowBytes), k, qt, w.AsSpan(r * k, k));
        return w;
    }

    /// <summary>
    /// Random but valid blocks: random quant/scale bytes, with each block's f16 super-scale(s) overwritten by a sane
    /// positive value (random f16 bytes would include NaN/Inf and 1e4-sized scales). IQ2_XXS random bytes are valid.
    /// </summary>
    internal static byte[] RandomBlocks(Random rng, QuantizationType qt, int bytes)
    {
        byte[] b = new byte[bytes];
        rng.NextBytes(b);
        (int blockBytes, int[] halfOffsets) = qt switch
        {
            QuantizationType.Q4_K => (144, new[] { 0, 2 }),
            QuantizationType.Q5_K => (176, new[] { 0, 2 }),
            QuantizationType.Q6_K => (210, new[] { 208 }),
            QuantizationType.Q8_0 => (34, new[] { 0 }),
            QuantizationType.IQ2_XXS => (66, new[] { 0 }),
            _ => throw new ArgumentOutOfRangeException(nameof(qt)),
        };
        Assert.Equal(0, bytes % blockBytes);
        for (int blk = 0; blk < bytes; blk += blockBytes)
            foreach (int off in halfOffsets)
            {
                // d in [0.002, 0.02]; dmin (second Q4_K/Q5_K half) in [0.001, 0.01].
                float v = off == 2 ? 0.001f + (float)rng.NextDouble() * 0.009f : 0.002f + (float)rng.NextDouble() * 0.018f;
                ushort h = BitConverter.HalfToUInt16Bits((Half)v);
                b[blk + off] = (byte)(h & 0xFF);
                b[blk + off + 1] = (byte)(h >> 8);
            }
        return b;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)Math.Max(bytes, 1)).ThrowOnError();
        return p;
    }

    private static unsafe nint Upload<T>(T[] data) where T : unmanaged
    {
        long bytes = (long)data.Length * sizeof(T);
        nint p = Alloc(bytes);
        fixed (T* s = data) CudaDriverApi.cuMemcpyHtoD_v2(p, (nint)s, (nuint)bytes).ThrowOnError();
        return p;
    }
}

/// <summary>Shared CUDA probes for the #403 Nemotron MoE tests.</summary>
internal static class CudaNemotronMoeTestSupport
{
    /// <summary>True when a CUDA driver loads and reports a device.</summary>
    public static bool IsCudaAvailable()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return Probe();
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool Probe() => CudaDevice.IsAvailable();

    /// <summary>The PTX directory next to the test binaries, or the repo's <c>native/ptx</c>; null when neither has PTX.</summary>
    public static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && File.Exists(Path.Combine(full, "nemotron_moe.ptx"))) return full;
        }
        return null;
    }
}
