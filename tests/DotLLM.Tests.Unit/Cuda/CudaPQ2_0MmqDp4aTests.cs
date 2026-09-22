using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Kernel parity for the packed PQ2_0 prefill GEMM (issue #490,
/// <c>pq2_0_mmq_dp4a_f32y_bn32</c> / <c>_bn16</c> in <c>native/kernels/pq2_0_mmq_dp4a.cu</c>).
/// </summary>
/// <remarks>
/// <para>
/// <b>Three oracles.</b> (A) exact: the int8 activations are the GPU's own, so every 32-block dot is
/// an exact integer and the only freedom left is FP32 summation order — bound <c>1e-5·Σ|term|</c> per
/// output. A dropped K group moves an output by about <c>Σ|term|/gpr</c>, two to three orders above
/// that. (B) the two tile instantiations, which share no tiling parameter, must agree to the same
/// bound on the same data. (C) the path this replaces — dequant-to-F16 + cuBLAS HGEMM — on
/// Gaussian-with-outliers data: relative RMS error and argmax over a wide output, the same
/// calibration the #485 GEMV test uses (this is a numerics change, the CPU W2A8 tier's).
/// </para>
/// <para>
/// <b>Shapes.</b> Bonsai 2's widths (k = 5120 hidden, 17408 ffn, and the 248k-row lm_head, which
/// takes this path at perplexity widths) and deliberately ragged row/column counts — 9, 13, 37, 256,
/// 512 tokens against n not a multiple of either tile's 128/256 rows. A tile-aligned shape cannot see
/// a tail bug, and the S tail is the one a 32-wide tile hits on almost every real prompt.
/// </para>
/// <para>
/// <b>Mutants this must kill</b> (the same set <see cref="PQ2_0MmqLayoutEmulationTests"/> kills on the
/// CPU, which is the point of keeping that emulation): a shifted activation-tile offset, a dropped
/// K-tile, a transposed shared-memory weight index, and an off-by-one column tail mask.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaPQ2_0MmqDp4aTests
{
    private readonly ITestOutputHelper _out;
    public CudaPQ2_0MmqDp4aTests(ITestOutputHelper output) => _out = output;

    public static TheoryData<int, int, int> Shapes()
        => new()
        {
            { 9, 37, 5120 },        // just past the GEMV's range; narrow tile, every dimension a tail
            { 13, 129, 5120 },
            { 16, 513, 5120 },      // widest narrow-tile column count
            { 17, 257, 5120 },      // first wide-tile width
            { 37, 129, 17408 },     // Bonsai 2's ffn width
            { 256, 257, 5120 },
            { 512, 37, 17408 },     // more columns than rows
        };

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Mmq_MatchesExactIntegerReference(int columns, int n, int k)
    {
        string ptxDir = SkipUnlessMmqKernel();
        var rng = new Random(4900 + 31 * columns + n + k);
        byte[] packed = CudaPQ2_0GemvDp4aTests.RandomPackedPQ2_0WithCode3(rng, n, k);
        float[] x = CudaPQ2_0GemvDp4aTests.RealisticActivations(rng, columns * k);

        MmqRun run = RunMmq(ptxDir, packed, x, n, k, columns, withF16Oracle: false);

        AssertMatchesReference("oracle A (exact integer reference), default tile",
            packed, run.Q, run.D, run.Y, n, k, columns);
        AssertMatchesReference("oracle B (the other tile)",
            packed, run.Q, run.D, run.YOtherTile!, n, k, columns);
    }

    /// <summary>
    /// The lm_head shape: Bonsai 2 runs a 248k-row projection at prefill width, which is where the
    /// path this replaces materialises a ~2.5 GB F16 scratch. Reference on sampled rows only.
    /// </summary>
    [SkippableFact]
    public void Mmq_MatchesExactIntegerReference_AtVocabScale()
    {
        string ptxDir = SkipUnlessMmqKernel();
        const int n = 248832, k = 5120, columns = 13;
        var rng = new Random(49001);
        byte[] packed = CudaPQ2_0GemvDp4aTests.RandomPackedPQ2_0WithCode3(rng, n, k);
        float[] x = CudaPQ2_0GemvDp4aTests.RealisticActivations(rng, columns * k);

        MmqRun run = RunMmq(ptxDir, packed, x, n, k, columns, withF16Oracle: false);
        AssertMatchesReference("oracle A (exact integer reference), lm_head shape",
            packed, run.Q, run.D, run.Y, n, k, columns);
    }

    [SkippableTheory]
    [InlineData(9)]
    [InlineData(37)]
    [InlineData(256)]
    public void Mmq_AgreesWithDequantCublasPath_OnRealisticData(int columns)
    {
        string ptxDir = SkipUnlessMmqKernel();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "dequant_pq2_0.ptx")), "dequant_pq2_0.ptx not generated");
        const int n = 4096, k = 5120;
        var rng = new Random(49010 + columns);
        byte[] packed = CudaPQ2_0GemvDp4aTests.RandomPackedPQ2_0WithCode3(rng, n, k);
        float[] x = CudaPQ2_0GemvDp4aTests.RealisticActivations(rng, columns * k);

        MmqRun run = RunMmq(ptxDir, packed, x, n, k, columns, withF16Oracle: true);

        for (int c = 0; c < columns; c++)
        {
            double se = 0, sr = 0, maxDiff = 0;
            for (int r = 0; r < n; r++)
            {
                double f = run.YF16![(long)c * n + r], d = run.Y[(long)c * n + r];
                Assert.True(float.IsFinite(run.Y[(long)c * n + r]), $"column {c} row {r} is not finite");
                se += (f - d) * (f - d);
                sr += f * f;
                maxDiff = Math.Max(maxDiff, Math.Abs(f - d));
            }
            double relRms = Math.Sqrt(se / sr);
            (int argF16, double gap) = ArgMaxWithGap(run.YF16!, (long)c * n, n);
            (int argMmq, _) = ArgMaxWithGap(run.Y, (long)c * n, n);
            _out.WriteLine($"S={columns} col {c}: rel RMS |mmq-f16| {relRms:E3}, max diff {maxDiff:E3}, " +
                           $"argmax f16={argF16} mmq={argMmq}, f16 top-2 gap {gap:E3}");
            // Same calibration as the #485 GEMV test: W2A8 activation error on these activations is
            // ~0.7-1.3% of the output RMS. Correctness is oracle A; this bounds the numerics change.
            Assert.True(relRms <= 2e-2, $"column {c}: relative RMS error {relRms:E3} vs dequant+cuBLAS exceeds 2e-2");
            if (gap > 2 * maxDiff)
                Assert.Equal(argF16, argMmq);
            else
                _out.WriteLine($"  col {c}: F16 top-2 gap within 2x the observed error — near-tie, argmax not asserted");
        }
    }

    private void AssertMatchesReference(string label, byte[] packed, sbyte[] q, float[] d, float[] y,
        int n, int k, int columns)
    {
        int gpr = k / 128, bpr = k / 32;
        int[] rows = SampleRows(n);
        double worst = 0;
        int worstRow = 0, worstCol = 0;
        foreach (int r in rows)
        {
            for (int c = 0; c < columns; c++)
            {
                double acc = 0, abs = 0;
                for (int b = 0; b < bpr; b++)
                {
                    int g = b / 4;
                    long gb = ((long)r * gpr + g) * 34;
                    float ws = (float)BitConverter.UInt16BitsToHalf((ushort)(packed[gb] | (packed[gb + 1] << 8)));
                    int isum = 0;
                    for (int i = 0; i < 32; i++)
                    {
                        int e = (b % 4) * 32 + i;                  // element within the group
                        int code = (packed[gb + 2 + e / 4] >> (2 * (e % 4))) & 3;
                        isum += (code - 1) * q[(long)c * k + (long)b * 32 + i];
                    }
                    double term = (double)isum * (float)(d[(long)c * bpr + b] * ws);
                    acc += term;
                    abs += Math.Abs(term);
                }
                double tol = 1e-5 * abs + 1e-6;
                double ratio = Math.Abs(acc - y[(long)c * n + r]) / tol;
                if (ratio > worst) { worst = ratio; worstRow = r; worstCol = c; }
            }
        }
        _out.WriteLine($"{label}: S={columns} n={n} k={k}, {rows.Length} rows sampled: worst |diff|/tol = {worst:F4} " +
                       $"(row {worstRow}, col {worstCol})");
        Assert.True(worst <= 1.0,
            $"{label}: S={columns} n={n} k={k} row {worstRow} column {worstCol}: got {y[(long)worstCol * n + worstRow]:R}, " +
            $"|diff|/tol = {worst:F3}");
    }

    /// <summary>Every row when there are few, otherwise a spread of 64 (the reference is O(n·S·k) on the CPU).</summary>
    private static int[] SampleRows(int n)
    {
        if (n <= 64)
        {
            var all = new int[n];
            for (int i = 0; i < n; i++) all[i] = i;
            return all;
        }
        var rows = new int[64];
        for (int i = 0; i < 64; i++) rows[i] = (int)((long)i * (n - 1) / 63);
        return rows;
    }

    private static (int Index, double Gap) ArgMaxWithGap(float[] y, long offset, int n)
    {
        int best = 0;
        double second = double.NegativeInfinity;
        for (int r = 1; r < n; r++)
        {
            if (y[offset + r] > y[offset + best]) { second = y[offset + best]; best = r; }
            else if (y[offset + r] > second) second = y[offset + r];
        }
        return (best, y[offset + best] - second);
    }

    // ───────────────────────── GPU plumbing ─────────────────────────

    private sealed record MmqRun(float[] Y, float[]? YOtherTile, float[]? YF16, sbyte[] Q, float[] D);

    /// <summary>
    /// Stages exactly as <c>CudaQwen3HybridDenseTransformerModel.Gemm</c>'s #490 branch does
    /// (repack → quantize → MMQ), then repeats the GEMM on the other tile, and optionally runs the
    /// dequant-to-F16 + cuBLAS path this replaces.
    /// </summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static unsafe MmqRun RunMmq(string ptxDir, byte[] packed, float[] x, int n, int k, int columns,
        bool withF16Oracle)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);
        Assert.True(kernels.HasPQ2_0MmqDp4a, kernels.PQ2_0MmqDp4aUnavailableReason);
        Assert.True(kernels.HasPQ2_0GemvDp4a, kernels.PQ2_0GemvDp4aUnavailableReason);
        nint s = stream.Handle;

        int elems = x.Length, blocks = elems / 32;
        int tile = CudaSmallSGemvDispatch.MmqTileColumns(columns);
        int other = tile == CudaKernels.Pq2_0MmqTileColumnsWide
            ? CudaKernels.Pq2_0MmqTileColumnsNarrow
            : CudaKernels.Pq2_0MmqTileColumnsWide;

        nint devW = Alloc(packed.LongLength), devWSplit = Alloc(CudaKernels.PQ2_0SplitLayoutBytes(n, k));
        nint devX = Alloc((long)elems * sizeof(float));
        nint devQ = Alloc(elems), devMeta = Alloc((long)blocks * CudaKernels.Pq2_0Dp4aMetaBytesPerBlock);
        nint devY = Alloc((long)columns * n * sizeof(float));
        nint devYOther = Alloc((long)columns * n * sizeof(float));
        nint devWF16 = withF16Oracle ? Alloc((long)n * k * sizeof(ushort)) : 0;
        nint devXh = withF16Oracle ? Alloc((long)elems * sizeof(ushort)) : 0;
        nint devYh = withF16Oracle ? Alloc((long)columns * n * sizeof(ushort)) : 0;
        nint devYF16 = withF16Oracle ? Alloc((long)columns * n * sizeof(float)) : 0;
        try
        {
            fixed (byte* w = packed)
                CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packed.LongLength).ThrowOnError();
            fixed (float* px = x)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)elems * sizeof(float))).ThrowOnError();
            kernels.LaunchPQ2_0RepackSplitF16(devW, devWSplit, n, k, s);
            kernels.LaunchPQ2_0Dp4aQuantizeX(devX, devQ, devMeta, elems, s);
            kernels.LaunchPQ2_0MmqDp4a(devWSplit, devQ, devMeta, devY, n, k, columns, tile, s);
            kernels.LaunchPQ2_0MmqDp4a(devWSplit, devQ, devMeta, devYOther, n, k, columns, other, s);

            if (withF16Oracle)
            {
                using var cublas = CudaCublasHandle.Create();
                kernels.LaunchDequantPQ2_0ToF16(devWSplit, devWF16, n, k, s);
                kernels.LaunchConvertF32ToF16(devX, devXh, elems, s);
                CudaGemm.LinearF16(cublas.Handle, devXh, devWF16, devYh, columns, k, n, s);
                kernels.LaunchConvertF16ToF32(devYh, devYF16, columns * n, s);
            }
            stream.Synchronize();

            float[] y = Download<float>(devY, (long)columns * n);
            float[] yOther = Download<float>(devYOther, (long)columns * n);
            float[]? yF16 = withF16Oracle ? Download<float>(devYF16, (long)columns * n) : null;
            (sbyte[] q, float[] d) = DownloadQuant(devQ, devMeta, elems);
            return new MmqRun(y, yOther, yF16, q, d);
        }
        finally
        {
            foreach (nint p in new[] { devW, devWSplit, devX, devQ, devMeta, devY, devYOther, devWF16, devXh, devYh, devYF16 })
                if (p != 0) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    /// <summary>Downloads the quantizer output and undoes the in-chunk permutation.</summary>
    private static (sbyte[] Q, float[] D) DownloadQuant(nint devQ, nint devMeta, int elems)
    {
        sbyte[] permuted = Download<sbyte>(devQ, elems);
        int[] meta = Download<int>(devMeta, elems / 32 * 2);
        var q = new sbyte[elems];
        for (int p = 0; p < elems; p++)
            q[CudaPQ2_0GemvDp4aTests.Dp4aPermutedToElement(p)] = permuted[p];
        var d = new float[elems / 32];
        for (int b = 0; b < d.Length; b++) d[b] = BitConverter.Int32BitsToSingle(meta[2 * b]);
        return (q, d);
    }

    private static unsafe T[] Download<T>(nint dev, long count) where T : unmanaged
    {
        var host = new T[count];
        fixed (T* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)(count * sizeof(T))).ThrowOnError();
        return host;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        return p;
    }

    /// <summary>
    /// Skips without a CUDA GPU or without <c>pq2_0_mmq_dp4a.ptx</c>; a present-but-stale PTX is a
    /// real failure (asserted inside the tests).
    /// </summary>
    internal static string SkipUnlessMmqKernel()
    {
        string ptxDir = CudaPQ2_0GemvDp4aTests.SkipUnlessDp4aKernel();   // the MMQ reuses its quantizer
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "pq2_0_mmq_dp4a.ptx")),
            "pq2_0_mmq_dp4a.ptx not generated (run native/build_ptx.bat on a CUDA box)");
        return ptxDir;
    }
}
