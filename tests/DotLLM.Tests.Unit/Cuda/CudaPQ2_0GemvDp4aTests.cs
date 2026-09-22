using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Kernel parity for the int8-activation dp4a PQ2_0 GEMV (issue #485):
/// <c>pq2_0_dp4a_quantize_x</c> and <c>pq2_0_gemv_dp4a_f32y_{S}</c> in
/// <c>native/kernels/pq2_0_gemv_dp4a.cu</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Quantizer — byte-exact.</b> The GPU quantizer claims the CPU W2A8 activation quantizer's
/// rounding exactly (<see cref="MatMul.QuantizeF32ToQ8_0"/>: per-32 amax/127, round-half-even,
/// half-rounded scale). After undoing the in-chunk permutation, every int8 byte, every scale's
/// bits and every block sum must match the CPU — both its dispatched SIMD tier and its scalar
/// reference.
/// </para>
/// <para>
/// <b>GEMV — three oracles.</b> (A) near-exact: from the int8 activations the GPU itself produced,
/// the integer dot of every 32-block is exact, so the only freedom is FP32 summation order and the
/// bound is <c>1e-5·Σ|term|</c> per output. A dropped block moves an output by about
/// <c>Σ|term|/(k/32)</c> ≥ 1.8e-3·Σ|term| — two orders above the bound. (B) the production CPU W2A8
/// GEMV (<see cref="MatMul.GemvPQ2_0"/>, SSSE3/AVX2 tier) on the same F32 input at
/// <c>2e-5·Σ|term|</c>, so a GPU quantizer and GEMV that agreed on a wrong answer would still fail.
/// (C) against the #482 F16-activation kernel on Gaussian-with-outliers data: relative RMS error and
/// argmax over a 4096-row output (a stand-in for logits) must agree wherever the F16 top-2 gap is
/// clear of the observed error.
/// </para>
/// <para>
/// <b>Mutants this must kill</b> (see the issue #485 report): a wrong activation-scale index in the
/// GEMV (<c>xmeta + s*bpr + blk</c> → <c>+ blk + 1</c>), a dropped column (skip <c>s == S-1</c>),
/// not subtracting the block sum (<c>isum = -md.y</c> → <c>0</c>), natural-order activations in the
/// quantizer (oracle A/B fail, the quantizer test still passes after un-permute — which is why both
/// exist), and the float scale instead of the half-rounded one (only the bit-exact scale check sees
/// it).
/// </para>
/// <para>
/// Weights include code 3 (value +2), which the #482 test generator never emits. Shapes:
/// k ∈ {5120, 17408} (Bonsai 2 hidden / ffn) for every S = 1..8, plus k = 640 (5 groups: the last
/// warp step has one live group of eight) and n ∈ {37, 513} (not multiples of the 16-row block).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaPQ2_0GemvDp4aTests
{
    private readonly ITestOutputHelper _out;
    public CudaPQ2_0GemvDp4aTests(ITestOutputHelper output) => _out = output;

    public static TheoryData<int, int, int> Shapes()
    {
        var data = new TheoryData<int, int, int>();
        for (int s = 1; s <= CudaKernels.Pq2_0GemvMultiMaxColumns; s++)
        {
            data.Add(s, 37, 5120);
            data.Add(s, 513, 17408);
        }
        data.Add(3, 513, 5120);
        data.Add(4, 37, 640);
        data.Add(7, 513, 640);
        return data;
    }

    public static TheoryData<int, int> QuantShapes()
    {
        var data = new TheoryData<int, int>();
        foreach (int s in new[] { 1, 3, 8 })
        {
            data.Add(s, 5120);
            data.Add(s, 17408);
        }
        return data;
    }

    // ───────────────────────── quantizer ─────────────────────────

    [SkippableTheory]
    [MemberData(nameof(QuantShapes))]
    public void Quantizer_MatchesCpuW2A8ActivationQuantizer_ByteExact(int columns, int k)
    {
        string ptxDir = SkipUnlessDp4aKernel();
        var rng = new Random(485 + 7 * columns + k);
        float[] x = RealisticActivations(rng, columns * k);
        // Edge blocks: one all-zero block (scale 0 -> all-zero ints) and one with a lone outlier.
        Array.Clear(x, 64, 32);
        x[200] = 37.5f;

        (sbyte[] gpuQ, float[] gpuD, int[] gpuSum) = RunQuantizer(ptxDir, x);

        (sbyte[] cpuQ, float[] cpuD) = CpuQuantize(x, scalar: false);
        (sbyte[] refQ, float[] refD) = CpuQuantize(x, scalar: true);
        AssertQuantMatches("CPU dispatched tier", cpuQ, cpuD, gpuQ, gpuD, gpuSum);
        AssertQuantMatches("CPU scalar reference", refQ, refD, gpuQ, gpuD, gpuSum);
        _out.WriteLine($"S={columns} k={k}: {x.Length} int8 values, {gpuD.Length} scales and sums byte-exact");
    }

    private static void AssertQuantMatches(string label, sbyte[] cpuQ, float[] cpuD, sbyte[] gpuQ, float[] gpuD, int[] gpuSum)
    {
        for (int b = 0; b < cpuD.Length; b++)
        {
            Assert.True(BitConverter.SingleToInt32Bits(cpuD[b]) == BitConverter.SingleToInt32Bits(gpuD[b]),
                $"{label}: block {b} scale cpu={cpuD[b]:R} gpu={gpuD[b]:R}");
            int sum = 0;
            for (int i = 0; i < 32; i++)
            {
                int e = b * 32 + i;
                Assert.True(cpuQ[e] == gpuQ[e], $"{label}: element {e} (block {b}) cpu={cpuQ[e]} gpu={gpuQ[e]}");
                sum += cpuQ[e];
            }
            Assert.True(sum == gpuSum[b], $"{label}: block {b} sum cpu={sum} gpu={gpuSum[b]}");
        }
    }

    // ───────────────────────── GEMV ─────────────────────────

    [SkippableTheory]
    [MemberData(nameof(Shapes))]
    public void Dp4a_MatchesExactIntegerReference_AndCpuW2A8(int columns, int n, int k)
    {
        string ptxDir = SkipUnlessDp4aKernel();
        var rng = new Random(4850 + 31 * columns + n + k);
        byte[] packed = RandomPackedPQ2_0WithCode3(rng, n, k);
        float[] x = RealisticActivations(rng, columns * k);

        GpuRun run = RunGemv(ptxDir, packed, x, n, k, columns, withF16Oracle: false);

        // The GPU's own int8 must be the CPU's (the quantizer test covers this in depth).
        (sbyte[] cpuQ, float[] cpuD) = CpuQuantize(x, scalar: false);
        Assert.Equal(cpuQ, run.Q);

        // Oracle A — exact integer dot per block, float scale product mirrored, double accumulation.
        int gpr = k / 128, bpr = k / 32;
        double[] refY = new double[columns * n];
        double[] absSum = new double[columns * n];
        for (int r = 0; r < n; r++)
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
                        int e = (b % 4) * 32 + i;                        // element within the group
                        int code = (packed[gb + 2 + e / 4] >> (2 * (e % 4))) & 3;
                        isum += (code - 1) * run.Q[c * k + b * 32 + i];
                    }
                    double term = (double)isum * (float)(run.D[c * bpr + b] * ws);
                    acc += term;
                    abs += Math.Abs(term);
                }
                refY[c * n + r] = acc;
                absSum[c * n + r] = abs;
            }
        }
        AssertWithin("oracle A (exact integer reference)", refY, absSum, run.Y, columns, n, k, relToAbsSum: 1e-5);

        // Oracle B — the production CPU W2A8 GEMV on the same F32 input.
        if (CpuW2A8Active())
        {
            double[] cpuY = new double[columns * n];
            float[] row = new float[n];
            unsafe
            {
                fixed (byte* w = packed)
                fixed (float* px = x, py = row)
                {
                    for (int c = 0; c < columns; c++)
                    {
                        MatMul.GemvPQ2_0(w, px + (long)c * k, py, n, k, threadPool: null);
                        for (int r = 0; r < n; r++) cpuY[c * n + r] = row[r];
                    }
                }
            }
            AssertWithin("oracle B (CPU W2A8 GEMV)", cpuY, absSum, run.Y, columns, n, k, relToAbsSum: 2e-5);
        }
        else
        {
            _out.WriteLine("oracle B skipped: CPU W2A8 tier inactive (no SSSE3 or DOTLLM_PQ2_W2A8=0)");
        }
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    public void Dp4a_AgreesWithF16ActivationKernel_OnRealisticData(int columns)
    {
        string ptxDir = SkipUnlessDp4aKernel();
        Skip.IfNot(File.Exists(Path.Combine(ptxDir, "pq2_0_gemv_multi.ptx")), "pq2_0_gemv_multi.ptx not generated");
        const int n = 4096, k = 5120;
        var rng = new Random(48500 + columns);
        byte[] packed = RandomPackedPQ2_0WithCode3(rng, n, k);
        float[] x = RealisticActivations(rng, columns * k);

        GpuRun run = RunGemv(ptxDir, packed, x, n, k, columns, withF16Oracle: true);

        for (int c = 0; c < columns; c++)
        {
            double se = 0, sr = 0, maxDiff = 0;
            for (int r = 0; r < n; r++)
            {
                double f = run.YF16![c * n + r], d = run.Y[c * n + r];
                Assert.True(float.IsFinite(run.Y[c * n + r]), $"column {c} row {r} is not finite");
                se += (f - d) * (f - d);
                sr += f * f;
                maxDiff = Math.Max(maxDiff, Math.Abs(f - d));
            }
            double relRms = Math.Sqrt(se / sr);
            (int argF16, double gap) = ArgMaxWithGap(run.YF16!, c * n, n);
            (int argDp4a, _) = ArgMaxWithGap(run.Y, c * n, n);
            _out.WriteLine($"S={columns} col {c}: rel RMS |dp4a-f16| {relRms:E3}, max diff {maxDiff:E3}, " +
                           $"argmax f16={argF16} dp4a={argDp4a}, f16 top-2 gap {gap:E3}");
            // Estimated W2A8 error on these activations (~15% of blocks carry an outlier) is ~0.7-1.3% of
            // the output RMS. This bound is calibration, not physics — oracles A/B in the parity test are
            // the correctness signal; a lost block or column is O(100%) on the hit outputs.
            Assert.True(relRms <= 2e-2, $"column {c}: relative RMS error {relRms:E3} vs the F16-activation kernel exceeds 2e-2");
            if (gap > 2 * maxDiff)
                Assert.Equal(argF16, argDp4a);
            else
                _out.WriteLine($"  col {c}: F16 top-2 gap within 2x the observed error — near-tie, argmax not asserted");
        }
    }

    private void AssertWithin(string label, double[] expected, double[] absSum, float[] actual,
        int columns, int n, int k, double relToAbsSum)
    {
        for (int c = 0; c < columns; c++)
        {
            double worstRatio = 0;
            int worst = 0;
            for (int r = 0; r < n; r++)
            {
                int i = c * n + r;
                double tol = relToAbsSum * absSum[i] + 1e-6;
                double ratio = Math.Abs(expected[i] - actual[i]) / tol;
                if (ratio > worstRatio) { worstRatio = ratio; worst = r; }
            }
            int wi = c * n + worst;
            _out.WriteLine($"{label}: S={columns} n={n} k={k} col {c}: worst |diff|/tol = {worstRatio:F3} (row {worst})");
            Assert.True(worstRatio <= 1.0,
                $"{label}: S={columns} n={n} k={k} column {c} row {worst}: expected {expected[wi]:R}, got {actual[wi]:R}, " +
                $"|diff| {Math.Abs(expected[wi] - actual[wi]):E3} > {relToAbsSum:E1}·Σ|term| ({absSum[wi]:E3})");
        }
    }

    private static (int Index, double Gap) ArgMaxWithGap(float[] y, int offset, int n)
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

    private sealed record GpuRun(float[] Y, float[]? YF16, sbyte[] Q, float[] D, int[] Sum);

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static unsafe GpuRun RunGemv(string ptxDir, byte[] packed, float[] x, int n, int k, int columns, bool withF16Oracle)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);
        Assert.True(kernels.HasPQ2_0GemvDp4a, kernels.PQ2_0GemvDp4aUnavailableReason);
        if (withF16Oracle) Assert.True(kernels.HasPQ2_0GemvMulti, kernels.PQ2_0GemvMultiUnavailableReason);
        nint s = stream.Handle;

        int elems = x.Length, blocks = elems / 32;
        long splitLen = CudaKernels.PQ2_0SplitLayoutBytes(n, k);
        nint devW = Alloc(packed.LongLength), devWSplit = Alloc(splitLen);
        nint devX = Alloc((long)elems * sizeof(float));
        nint devQ = Alloc(elems), devMeta = Alloc((long)blocks * CudaKernels.Pq2_0Dp4aMetaBytesPerBlock);
        nint devY = Alloc((long)columns * n * sizeof(float));
        nint devXh = withF16Oracle ? Alloc((long)elems * sizeof(ushort)) : 0;
        nint devYF16 = withF16Oracle ? Alloc((long)columns * n * sizeof(float)) : 0;
        try
        {
            fixed (byte* w = packed)
                CudaDriverApi.cuMemcpyHtoD_v2(devW, (nint)w, (nuint)packed.LongLength).ThrowOnError();
            fixed (float* px = x)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)elems * sizeof(float))).ThrowOnError();
            kernels.LaunchPQ2_0RepackSplitF16(devW, devWSplit, n, k, s);

            // Same staging as CudaQwen3HybridDenseTransformerModel.Gemm's dp4a branch.
            kernels.LaunchPQ2_0Dp4aQuantizeX(devX, devQ, devMeta, elems, s);
            kernels.LaunchPQ2_0GemvDp4a(devWSplit, devQ, devMeta, devY, n, k, columns, s);
            if (withF16Oracle)
            {
                kernels.LaunchConvertF32ToF16(devX, devXh, elems, s);
                kernels.LaunchPQ2_0GemvMulti(devWSplit, devXh, devYF16, n, k, columns, s);
            }
            stream.Synchronize();

            float[] y = Download<float>(devY, columns * n);
            float[]? yF16 = withF16Oracle ? Download<float>(devYF16, columns * n) : null;
            (sbyte[] q, float[] d, int[] sum) = DownloadQuant(devQ, devMeta, elems);
            return new GpuRun(y, yF16, q, d, sum);
        }
        finally
        {
            foreach (nint p in new[] { devW, devWSplit, devX, devQ, devMeta, devY, devXh, devYF16 })
                if (p != 0) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static unsafe (sbyte[] Q, float[] D, int[] Sum) RunQuantizer(string ptxDir, float[] x)
    {
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptxDir);
        Assert.True(kernels.HasPQ2_0GemvDp4a, kernels.PQ2_0GemvDp4aUnavailableReason);

        int elems = x.Length, blocks = elems / 32;
        nint devX = Alloc((long)elems * sizeof(float));
        nint devQ = Alloc(elems), devMeta = Alloc((long)blocks * CudaKernels.Pq2_0Dp4aMetaBytesPerBlock);
        try
        {
            fixed (float* px = x)
                CudaDriverApi.cuMemcpyHtoD_v2(devX, (nint)px, (nuint)((long)elems * sizeof(float))).ThrowOnError();
            kernels.LaunchPQ2_0Dp4aQuantizeX(devX, devQ, devMeta, elems, stream.Handle);
            stream.Synchronize();
            return DownloadQuant(devQ, devMeta, elems);
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(devX);
            CudaDriverApi.cuMemFree_v2(devQ);
            CudaDriverApi.cuMemFree_v2(devMeta);
        }
    }

    /// <summary>Downloads the quantizer output and undoes the in-chunk permutation.</summary>
    private static (sbyte[] Q, float[] D, int[] Sum) DownloadQuant(nint devQ, nint devMeta, int elems)
    {
        sbyte[] permuted = Download<sbyte>(devQ, elems);
        int[] meta = Download<int>(devMeta, elems / 32 * 2);
        var q = new sbyte[elems];
        for (int p = 0; p < elems; p++)
            q[Dp4aPermutedToElement(p)] = permuted[p];
        var d = new float[elems / 32];
        var sum = new int[elems / 32];
        for (int b = 0; b < d.Length; b++)
        {
            d[b] = BitConverter.Int32BitsToSingle(meta[2 * b]);
            sum[b] = meta[2 * b + 1];
        }
        return (q, d, sum);
    }

    /// <summary>
    /// Permuted byte position → element index (the contract in pq2_0_gemv_dp4a.cu): within each
    /// 16-element chunk, byte 4i + j holds element 4j + i.
    /// </summary>
    internal static int Dp4aPermutedToElement(int p)
    {
        int chunkBase = p & ~15, r = p & 15;
        return chunkBase + 4 * (r & 3) + (r >> 2);
    }

    private static unsafe T[] Download<T>(nint dev, int count) where T : unmanaged
    {
        var host = new T[count];
        fixed (T* p = host)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dev, (nuint)((long)count * sizeof(T))).ThrowOnError();
        return host;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint p, (nuint)bytes).ThrowOnError();
        return p;
    }

    // ───────────────────────── CPU side ─────────────────────────

    /// <summary>The CPU W2A8 activation quantizer's output, split into int8 values and float(Half) scales.</summary>
    private static unsafe (sbyte[] Q, float[] D) CpuQuantize(float[] x, bool scalar)
    {
        int blocks = x.Length / 32;
        byte[] q8 = new byte[blocks * 34];
        fixed (float* px = x)
        fixed (byte* pq = q8)
        {
            if (scalar) MatMul.QuantizeF32ToQ8_0Scalar(px, pq, x.Length);
            else MatMul.QuantizeF32ToQ8_0(px, pq, x.Length);
        }
        var q = new sbyte[x.Length];
        var d = new float[blocks];
        for (int b = 0; b < blocks; b++)
        {
            d[b] = (float)BitConverter.UInt16BitsToHalf((ushort)(q8[b * 34] | (q8[b * 34 + 1] << 8)));
            for (int i = 0; i < 32; i++) q[b * 32 + i] = (sbyte)q8[b * 34 + 2 + i];
        }
        return (q, d);
    }

    /// <summary>Mirrors <c>MatMul.PQ2_0UseW2A8</c>: the SIMD W2A8 tier runs on SSSE3+ unless disabled.</summary>
    private static bool CpuW2A8Active()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_PQ2_W2A8");
        if (env is "0" or "false" or "off") return false;
        return Ssse3.IsSupported;
    }

    /// <summary>
    /// Roughly post-RMSNorm activations: N(0, 1) with ~0.5% outliers up to ±12 (the channel outliers
    /// that make per-block scaling matter).
    /// </summary>
    internal static float[] RealisticActivations(Random rng, int count)
    {
        var x = new float[count];
        for (int i = 0; i < count; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
            if (rng.Next(200) == 0) g *= 6;
            x[i] = (float)g;
        }
        return x;
    }

    /// <summary>
    /// Random interleaved (on-disk) PQ2_0 rows — per 128-element group an fp16 scale then 32 code
    /// bytes — with codes drawn from {0, 1, 2, 3} (30/30/30/10%), so value +2 is exercised.
    /// </summary>
    internal static byte[] RandomPackedPQ2_0WithCode3(Random rng, int n, int k)
    {
        int groupsPerRow = k / 128;
        byte[] buf = new byte[(long)n * groupsPerRow * 34];
        for (long g = 0; g < (long)n * groupsPerRow; g++)
        {
            long gb = g * 34;
            ushort bits = BitConverter.HalfToUInt16Bits((Half)(0.01f + rng.NextSingle() * 0.05f));
            buf[gb] = (byte)bits;
            buf[gb + 1] = (byte)(bits >> 8);
            for (int b = 0; b < 32; b++)
            {
                int v = 0;
                for (int i = 0; i < 4; i++)
                {
                    int u = rng.Next(10);
                    int code = u < 3 ? 0 : u < 6 ? 1 : u < 9 ? 2 : 3;
                    v |= code << (2 * i);
                }
                buf[gb + 2 + b] = (byte)v;
            }
        }
        return buf;
    }

    /// <summary>
    /// Skips without a CUDA GPU or without <c>pq2_0_gemv_dp4a.ptx</c>; a present-but-stale PTX is a
    /// real failure (asserted inside the tests).
    /// </summary>
    internal static string SkipUnlessDp4aKernel()
    {
        string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
        bool driver = NativeLibrary.TryLoad(lib, out nint h);
        if (driver) NativeLibrary.Free(h);
        Skip.IfNot(driver && CudaDevice.IsAvailable(), "No CUDA GPU available");

        string? ptxDir = null;
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) { ptxDir = full; break; }
        }
        Skip.If(ptxDir is null, "PTX files not found");
        Skip.IfNot(File.Exists(Path.Combine(ptxDir!, "pq2_0_gemv_dp4a.ptx")),
            "pq2_0_gemv_dp4a.ptx not generated (run native/build_ptx.bat on a CUDA box)");
        return ptxDir!;
    }
}
