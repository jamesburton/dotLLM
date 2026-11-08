using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Rows4 GEMV family for Q2_K / Q4_K / Q5_K / Q6_K / IQ4_XS / IQ4_NL at 1..8 activation columns
/// (<c>quantized_gemv_&lt;qt&gt;_mmvq_rows4_preq</c> and <c>_c2.._c8</c>, issues #1316 / #1375).
/// </summary>
/// <remarks>
/// <para>
/// <b>Oracle.</b> The weights are dequantised straight from the llama.cpp reference layout
/// (<c>dequantize_row_q2_K / q4_K / q5_K / q6_K / iq4_xs / iq4_nl</c>, written out here per type) and multiplied by
/// the dequantised Q8_1 activation (<c>dx * xq</c>, requantised exactly as <c>quantize_x_to_q8_1(_batched)</c> does),
/// so the oracle is independent of every kernel-side sub-block / nibble / 5th-bit / codebook / scale index: a wrong
/// scale slot, a swapped nibble half, a dropped min term or a wrong codebook word moves whole rows by O(|y|), not by
/// the tolerance. Weights are random bytes (distinct scales per sub-block, full-range packed scales) and shapes are
/// not tile-friendly (odd n, odd superblock counts, n % 4 != 0).
/// </para>
/// <para>
/// <b>Columns.</b> Column c of the multi-column kernel runs the same instructions on column c's Q8_1 data as the
/// single-column kernel does on that column alone, so the two must be <i>bit-identical</i>; a column-stride bug
/// (reading column 0's scales for every column) is caught by that identity, and n = 1 / columns = 1 / columns = 8 cover
/// the launch edges. columns = 9 must be refused by the launcher and never claimed by the dispatch policy.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaKQuantRows4MultiTests
{
    private readonly ITestOutputHelper _out;
    public CudaKQuantRows4MultiTests(ITestOutputHelper output) => _out = output;

    /// <summary>Bytes per 256 elements (IQ4_NL: 8 blocks of 18).</summary>
    internal static int SuperblockBytes(QuantizationType qt) => qt switch
    {
        QuantizationType.Q2_K => 84,
        QuantizationType.Q4_K => 144,
        QuantizationType.Q5_K => 176,
        QuantizationType.Q6_K => 210,
        QuantizationType.IQ4_XS => 136,
        QuantizationType.IQ4_NL => 144,
        _ => throw new ArgumentOutOfRangeException(nameof(qt)),
    };

    public static TheoryData<QuantizationType, int, int, int> DecodeShapes()
    {
        var d = new TheoryData<QuantizationType, int, int, int>();
        foreach (var qt in CudaKernels.Rows4Types)
        {
            d.Add(qt, 1, 1, 1024);        // n = 1: one live warp in the block
            d.Add(qt, 1, 3, 1280);        // 5 superblocks: odd count, half-empty tail step
            d.Add(qt, 1, 37, 1792);       // 7 superblocks, n % 4 = 1
            d.Add(qt, 1, 1537, 1536);
        }
        return d;
    }

    public static TheoryData<QuantizationType, int, int, int> MultiShapes()
    {
        var d = new TheoryData<QuantizationType, int, int, int>();
        foreach (var qt in CudaKernels.Rows4Types)
        {
            d.Add(qt, 2, 37, 1280);
            d.Add(qt, 3, 1, 1024);        // n = 1 with several columns
            d.Add(qt, 5, 37, 1792);
            d.Add(qt, 8, 1537, 1536);     // the cap
        }
        d.Add(QuantizationType.Q4_K, 4, 513, 8960);
        d.Add(QuantizationType.Q5_K, 7, 129, 2048);
        d.Add(QuantizationType.Q2_K, 6, 129, 768);     // below MmvqLargeKThreshold: the multi path still covers k % 256 == 0
        d.Add(QuantizationType.IQ4_NL, 4, 65, 512);
        return d;
    }

    [SkippableTheory]
    [MemberData(nameof(DecodeShapes))]
    public void Rows4Decode_MatchesDequantOracle(QuantizationType qt, int columns, int n, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (w, x) = Synthesise(qt, n, k, columns, seed: 1316 ^ n ^ k ^ (int)qt);
        float[] expected = Oracle(qt, w, x, n, k, columns);
        Half[] actual = RunDecode(qt, w, x, n, k);
        Compare(qt, n, k, 1, expected, actual, maxRel: 1e-3);
    }

    [SkippableTheory]
    [MemberData(nameof(MultiShapes))]
    public void Rows4Multi_MatchesOracle_AndIsBitIdenticalToSingleColumnPerColumn(QuantizationType qt, int columns, int n, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (w, x) = Synthesise(qt, n, k, columns, seed: 1375 ^ n ^ k ^ columns ^ (int)qt);
        float[] expected = Oracle(qt, w, x, n, k, columns);
        Half[] multi = RunMulti(qt, w, x, n, k, columns);
        Compare(qt, n, k, columns, expected, multi, maxRel: 1e-3);

        // Column c alone through the 1-column instantiation must reproduce the multi kernel's column c bit for bit.
        for (int c = 0; c < columns; c++)
        {
            Half[] single = RunMulti(qt, w, x.AsSpan(c * k, k).ToArray(), n, k, 1);
            for (int i = 0; i < n; i++)
                Assert.True(single[i] == multi[c * n + i],
                    $"{qt} column {c} row {i}: single {single[i]} vs multi {multi[c * n + i]} (column stride / lane mapping)");
        }
    }

    [SkippableFact]
    public void Rows4Multi_RefusesNineColumns()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var kernels = new CudaKernels(ptx);
        Assert.False(kernels.HasMmvqRows4Multi(QuantizationType.Q4_K, 9));
        Assert.False(kernels.HasMmvqRows4Multi(QuantizationType.Q4_K, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            kernels.LaunchQuantizedGemvMmvqRows4Multi(0, QuantizationType.Q4_K, 0, 0, 4, 1024, 9, 0));
    }

    /// <summary>Opt-in microbench (<c>DOTLLM_KQUANT_BENCH=1</c>): the F16-dequant + HGEMM small-S path vs the rows4 multi kernel.</summary>
    [SkippableTheory]
    [InlineData(QuantizationType.Q5_K, 2)]
    [InlineData(QuantizationType.Q5_K, 4)]
    [InlineData(QuantizationType.Q5_K, 8)]
    [InlineData(QuantizationType.Q4_K, 4)]
    public void Bench_SmallS_RowsMulti_vs_DequantHgemm(QuantizationType qt, int columns)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_KQUANT_BENCH") == "1", "set DOTLLM_KQUANT_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        const int n = 8960, k = 1536;
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        using var cublas = CudaCublasHandle.Create();
        cublas.SetStream(stream);
        nint s = stream.Handle;
        long wBytes = (long)n * (k / 256) * SuperblockBytes(qt);
        Alloc(wBytes, out nint dW); Alloc((long)columns * k * 2, out nint dX); Alloc((long)columns * n * 2, out nint dY);
        Alloc((long)n * k * 2, out nint dF16); Alloc((long)columns * CudaForwardState.PreQ8_1ScratchBytes(k), out nint dS);
        CudaDriverApi.cuEventCreate(out nint e0, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint e1, 0).ThrowOnError();
        try
        {
            CudaDriverApi.cuMemsetD8_v2(dW, 0x11, (nuint)wBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(dX, 0x3C, (nuint)((long)columns * k * 2)).ThrowOnError();
            double Time(Action body, int reps)
            {
                for (int i = 0; i < 3; i++) body();
                CudaDriverApi.cuEventRecord(e0, s).ThrowOnError();
                for (int i = 0; i < reps; i++) body();
                CudaDriverApi.cuEventRecord(e1, s).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(e1).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float ms, e0, e1).ThrowOnError();
                return ms * 1000.0 / reps;
            }
            double dequant = Time(() =>
            {
                kernels.LaunchDequantToF16(dW, qt, dF16, n * k, s);
                CudaGemm.LinearF16(cublas.Handle, dX, dF16, dY, columns, k, n, s);
            }, 50);
            double rows = Time(() =>
            {
                kernels.LaunchQuantizeXToQ8_1Batched(dX, dS, k, columns, s);
                kernels.LaunchQuantizedGemvMmvqRows4Multi(dW, qt, dS, dY, n, k, columns, s);
            }, 50);
            _out.WriteLine($"{qt} {n}x{k} S={columns}: dequant+HGEMM {dequant:F1} us | rows4 _c{columns} {rows:F1} us ({wBytes / rows / 1e3:F0} GB/s weights) | {dequant / rows:F2}x");
        }
        finally
        {
            foreach (nint p in new[] { dW, dX, dY, dF16, dS }) CudaDriverApi.cuMemFree_v2(p);
            CudaDriverApi.cuEventDestroy_v2(e0); CudaDriverApi.cuEventDestroy_v2(e1);
        }
    }

    private static void Alloc(long bytes, out nint ptr) => CudaDriverApi.cuMemAlloc_v2(out ptr, (nuint)bytes).ThrowOnError();

    // ── helpers ───────────────────────────────────────────────────────────────

    private void Compare(QuantizationType qt, int n, int k, int columns, float[] expected, Half[] actual, double maxRel)
    {
        double peak = expected.Max(MathF.Abs);
        double maxErr = 0;
        for (int i = 0; i < n * columns; i++) maxErr = Math.Max(maxErr, Math.Abs((float)actual[i] - expected[i]));
        _out.WriteLine($"{qt} n={n} k={k} S={columns}: peak|y|={peak:F3} max|err|={maxErr:E2} peak-rel={maxErr / peak:E2}");
        Assert.True(maxErr / peak < maxRel, $"peak-relative error {maxErr / peak:E2} (max abs {maxErr})");
    }

    /// <summary>Random blocks with FP16 scales in a sane range; <paramref name="columns"/> Gaussian activation rows with per-chunk outliers.</summary>
    internal static (byte[] Weights, Half[] X) Synthesise(QuantizationType qt, int n, int k, int columns, int seed)
    {
        var rng = new Random(seed);
        int sbb = SuperblockBytes(qt), nsb = k / 256;
        byte[] w = new byte[(long)n * nsb * sbb];
        rng.NextBytes(w);
        Half Scale() => (Half)(0.0005 + rng.NextDouble() * 0.004);
        for (long b = 0; b < (long)n * nsb; b++)
        {
            int off = (int)(b * sbb);
            switch (qt)
            {
                case QuantizationType.Q4_K:
                case QuantizationType.Q5_K:
                    { Half d = Scale(), dmin = Scale(); MemoryMarshal.Write(w.AsSpan(off, 2), in d); MemoryMarshal.Write(w.AsSpan(off + 2, 2), in dmin); break; }
                case QuantizationType.Q6_K:
                    { Half d = Scale(); MemoryMarshal.Write(w.AsSpan(off + 208, 2), in d); break; }
                case QuantizationType.Q2_K:
                    { Half d = Scale(), dmin = Scale(); MemoryMarshal.Write(w.AsSpan(off + 80, 2), in d); MemoryMarshal.Write(w.AsSpan(off + 82, 2), in dmin); break; }
                case QuantizationType.IQ4_XS:
                    { Half d = Scale(); MemoryMarshal.Write(w.AsSpan(off, 2), in d); break; }
                case QuantizationType.IQ4_NL:
                    for (int blk = 0; blk < 8; blk++) { Half d = Scale(); MemoryMarshal.Write(w.AsSpan(off + blk * 18, 2), in d); }
                    break;
            }
        }
        Half[] x = new Half[columns * k];
        for (int i = 0; i < x.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double outlier = ((i % k) / 32 + i / k) % 5 == 2 ? 6.0 : 1.0;
            x[i] = (Half)(g * 0.4 * outlier);
        }
        return (w, x);
    }

    private static void ScaleMinK4(int j, ReadOnlySpan<byte> q, out int sc, out int m)
    {
        if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
        else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
    }

    private static readonly sbyte[] KValuesIq4Nl = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

    /// <summary>llama.cpp <c>dequantize_row_*</c> for one 256-element superblock (IQ4_NL: 8 blocks) into <paramref name="y"/>[256].</summary>
    internal static void DequantBlock(QuantizationType qt, ReadOnlySpan<byte> blk, Span<float> y)
    {
        switch (qt)
        {
            case QuantizationType.Q4_K:
            {
                float d = (float)MemoryMarshal.Read<Half>(blk[..2]), dmin = (float)MemoryMarshal.Read<Half>(blk.Slice(2, 2));
                var scales = blk.Slice(4, 12); var qs = blk.Slice(16, 128);
                int isb = 0, qo = 0, yo = 0;
                for (int j = 0; j < 256; j += 64)
                {
                    ScaleMinK4(isb, scales, out int sc1, out int m1); ScaleMinK4(isb + 1, scales, out int sc2, out int m2);
                    float d1 = d * sc1, mm1 = dmin * m1, d2 = d * sc2, mm2 = dmin * m2;
                    for (int l = 0; l < 32; l++) y[yo + l] = d1 * (qs[qo + l] & 0xF) - mm1;
                    for (int l = 0; l < 32; l++) y[yo + 32 + l] = d2 * (qs[qo + l] >> 4) - mm2;
                    qo += 32; yo += 64; isb += 2;
                }
                break;
            }
            case QuantizationType.Q5_K:
            {
                float d = (float)MemoryMarshal.Read<Half>(blk[..2]), dmin = (float)MemoryMarshal.Read<Half>(blk.Slice(2, 2));
                var scales = blk.Slice(4, 12); var qh = blk.Slice(16, 32); var qs = blk.Slice(48, 128);
                int isb = 0, qo = 0, yo = 0, u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64)
                {
                    ScaleMinK4(isb, scales, out int sc1, out int m1); ScaleMinK4(isb + 1, scales, out int sc2, out int m2);
                    float d1 = d * sc1, mm1 = dmin * m1, d2 = d * sc2, mm2 = dmin * m2;
                    for (int l = 0; l < 32; l++) y[yo + l] = d1 * ((qs[qo + l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - mm1;
                    for (int l = 0; l < 32; l++) y[yo + 32 + l] = d2 * ((qs[qo + l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - mm2;
                    qo += 32; yo += 64; isb += 2; u1 <<= 2; u2 <<= 2;
                }
                break;
            }
            case QuantizationType.Q6_K:
            {
                var ql = blk[..128]; var qh = blk.Slice(128, 64); var sc = blk.Slice(192, 16);
                float d = (float)MemoryMarshal.Read<Half>(blk.Slice(208, 2));
                int qlo = 0, qho = 0, sco = 0, yo = 0;
                for (int h = 0; h < 2; h++)
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int is_ = l / 16;
                        int q1 = ((ql[qlo + l] & 0xF) | (((qh[qho + l] >> 0) & 3) << 4)) - 32;
                        int q2 = ((ql[qlo + l + 32] & 0xF) | (((qh[qho + l] >> 2) & 3) << 4)) - 32;
                        int q3 = ((ql[qlo + l] >> 4) | (((qh[qho + l] >> 4) & 3) << 4)) - 32;
                        int q4 = ((ql[qlo + l + 32] >> 4) | (((qh[qho + l] >> 6) & 3) << 4)) - 32;
                        y[yo + l] = d * (sbyte)sc[sco + is_] * q1;
                        y[yo + l + 32] = d * (sbyte)sc[sco + is_ + 2] * q2;
                        y[yo + l + 64] = d * (sbyte)sc[sco + is_ + 4] * q3;
                        y[yo + l + 96] = d * (sbyte)sc[sco + is_ + 6] * q4;
                    }
                    yo += 128; qlo += 64; qho += 32; sco += 8;
                }
                break;
            }
            case QuantizationType.Q2_K:
            {
                var scales = blk[..16]; var qs = blk.Slice(16, 64);
                float d = (float)MemoryMarshal.Read<Half>(blk.Slice(80, 2)), dmin = (float)MemoryMarshal.Read<Half>(blk.Slice(82, 2));
                int is_ = 0, yo = 0, qo = 0;
                for (int nn = 0; nn < 256; nn += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        byte s0 = scales[is_++];
                        float dl = d * (s0 & 0xF), ml = dmin * (s0 >> 4);
                        for (int l = 0; l < 16; l++) y[yo++] = dl * ((qs[qo + l] >> shift) & 3) - ml;
                        byte s1 = scales[is_++];
                        dl = d * (s1 & 0xF); ml = dmin * (s1 >> 4);
                        for (int l = 0; l < 16; l++) y[yo++] = dl * ((qs[qo + l + 16] >> shift) & 3) - ml;
                        shift += 2;
                    }
                    qo += 32;
                }
                break;
            }
            case QuantizationType.IQ4_XS:
            {
                float d = (float)MemoryMarshal.Read<Half>(blk[..2]);
                int scalesH = MemoryMarshal.Read<ushort>(blk.Slice(2, 2));
                var scalesL = blk.Slice(4, 4); var qs = blk.Slice(8, 128);
                for (int ib = 0; ib < 8; ib++)
                {
                    int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                    float dl = d * (ls - 32);
                    for (int j = 0; j < 16; j++)
                    {
                        byte q = qs[ib * 16 + j];
                        y[ib * 32 + j] = dl * KValuesIq4Nl[q & 0xF];
                        y[ib * 32 + j + 16] = dl * KValuesIq4Nl[q >> 4];
                    }
                }
                break;
            }
            case QuantizationType.IQ4_NL:
                for (int b = 0; b < 8; b++)
                {
                    var block = blk.Slice(b * 18, 18);
                    float d = (float)MemoryMarshal.Read<Half>(block[..2]);
                    for (int j = 0; j < 16; j++)
                    {
                        byte q = block[2 + j];
                        y[b * 32 + j] = d * KValuesIq4Nl[q & 0xF];
                        y[b * 32 + j + 16] = d * KValuesIq4Nl[q >> 4];
                    }
                }
                break;
            default: throw new ArgumentOutOfRangeException(nameof(qt));
        }
    }

    /// <summary>y[c*n + r] = sum_i W[r, i] * dequant_q8_1(x[c, i]), accumulated in double.</summary>
    internal static float[] Oracle(QuantizationType qt, byte[] w, Half[] x, int n, int k, int columns)
    {
        int bpr = k / 32, nsb = k / 256, sbb = SuperblockBytes(qt);
        float[] xd = new float[columns * k];   // dequantised Q8_1 activation = dx * xq, per column
        for (int c = 0; c < columns * bpr; c++)
        {
            float a = 0f;
            for (int j = 0; j < 32; j++) a = MathF.Max(a, MathF.Abs((float)x[c * 32 + j]));
            float inv = a > 0f ? 127.0f / a : 0f;
            float dx = (float)(Half)(a / 127.0f);
            for (int j = 0; j < 32; j++)
            {
                int q = Math.Clamp((int)MathF.Round((float)x[c * 32 + j] * inv, MidpointRounding.ToEven), -127, 127);
                xd[c * 32 + j] = dx * q;
            }
        }
        float[] y = new float[columns * n];
        float[] deq = new float[256];
        double[] acc = new double[columns];
        for (int r = 0; r < n; r++)
        {
            Array.Clear(acc);
            for (int sb = 0; sb < nsb; sb++)
            {
                DequantBlock(qt, w.AsSpan((int)(((long)r * nsb + sb) * sbb), sbb), deq);
                for (int c = 0; c < columns; c++)
                    for (int i = 0; i < 256; i++) acc[c] += (double)deq[i] * xd[c * k + sb * 256 + i];
            }
            for (int c = 0; c < columns; c++) y[c * n + r] = (float)acc[c];
        }
        return y;
    }

    /// <summary>The production decode route: pre-quantise + <see cref="CudaKernels.LaunchQuantizedGemvMmq"/> with the rows4 rule forced.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static Half[] RunDecode(QuantizationType qt, byte[] w, Half[] x, int n, int k)
    {
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        Skip.IfNot(kernels.HasMmvqRows4Preq(qt), $"rows4 kernel for {qt} not loaded (stale PTX?)");
        string prev = CudaKernels.KQuantGemvMode;
        CudaKernels.KQuantGemvMode = "rows4";
        try
        {
            return Launch(kernels, stream, qt, w, x, n, k, 1,
                (dW, dX, dS, dY) =>
                {
                    kernels.LaunchQuantizeXToQ8_1(dX, dS, k, stream.Handle);
                    kernels.LaunchQuantizedGemvMmq(dW, qt, dX, dY, n, k, dS, stream.Handle);
                });
        }
        finally { CudaKernels.KQuantGemvMode = prev; }
    }

    /// <summary>The small-S route: batched pre-quantise + <see cref="CudaKernels.LaunchQuantizedGemvMmvqRows4Multi"/>.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static Half[] RunMulti(QuantizationType qt, byte[] w, Half[] x, int n, int k, int columns)
    {
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        Skip.IfNot(kernels.HasMmvqRows4Multi(qt, columns), $"rows4 _c{columns} kernel for {qt} not loaded (stale PTX?)");
        return Launch(kernels, stream, qt, w, x, n, k, columns,
            (dW, dX, dS, dY) =>
            {
                kernels.LaunchQuantizeXToQ8_1Batched(dX, dS, k, columns, stream.Handle);
                kernels.LaunchQuantizedGemvMmvqRows4Multi(dW, qt, dS, dY, n, k, columns, stream.Handle);
            });
    }

    private static Half[] Launch(CudaKernels kernels, CudaStream stream, QuantizationType qt, byte[] w, Half[] x, int n, int k,
                                 int columns, Action<nint, nint, nint, nint> body)
    {
        long wBytes = w.Length, xBytes = (long)columns * k * 2, yBytes = (long)columns * n * 2;
        CudaDriverApi.cuMemAlloc_v2(out nint dW, (nuint)wBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dX, (nuint)xBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dY, (nuint)yBytes).ThrowOnError();
        CudaDriverApi.cuMemAlloc_v2(out nint dS, (nuint)((long)columns * CudaForwardState.PreQ8_1ScratchBytes(k))).ThrowOnError();
        try
        {
            fixed (byte* pW = w) CudaDriverApi.cuMemcpyHtoD_v2(dW, (nint)pW, (nuint)wBytes).ThrowOnError();
            fixed (Half* pX = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)pX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(dY, 0xFF, (nuint)yBytes).ThrowOnError();   // poison: an unwritten row cannot pass as 0
            body(dW, dX, dS, dY);
            stream.Synchronize();
            Half[] y = new Half[columns * n];
            fixed (Half* pY = y) CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, dY, (nuint)yBytes).ThrowOnError();
            return y;
        }
        finally
        {
            CudaDriverApi.cuMemFree_v2(dW); CudaDriverApi.cuMemFree_v2(dX);
            CudaDriverApi.cuMemFree_v2(dY); CudaDriverApi.cuMemFree_v2(dS);
        }
    }

    internal static string? FindPtxDir()
    {
        foreach (var dir in new[]
                 {
                     Path.Combine(AppContext.BaseDirectory, "ptx"),
                     Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                 })
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
        }
        return null;
    }
}

/// <summary>
/// The #1316/#1375 small-S policy for K-quant / IQ4 projections (<see cref="CudaSmallSGemvDispatch.CoversKQuant"/>).
/// Pure policy, no GPU; joins the serialized CUDA collection because the override is read per projection on the live path.
/// </summary>
[Collection(CudaCollection.Name)]
public sealed class CudaKQuantSmallSDispatchTests : IDisposable
{
    public void Dispose() => CudaSmallSGemvDispatch.KQuantMaxColumnsOverride = null;

    [Theory]
    [InlineData(1, false)]    // decode: LaunchQuantizedGemvMmq's rule, not this policy
    [InlineData(2, true)]
    [InlineData(8, true)]
    [InlineData(9, false)]    // past the _c8 instantiation: dequant + HGEMM
    [InlineData(512, false)]
    public void CoversKQuant_IsTheTwoToEightRange(int seqLen, bool expected)
    {
        CudaSmallSGemvDispatch.KQuantMaxColumnsOverride = null;
        Assert.Equal(expected, CudaSmallSGemvDispatch.CoversKQuant(seqLen));
    }

    [Fact]
    public void CoversKQuant_OverrideCapsAndDisables()
    {
        CudaSmallSGemvDispatch.KQuantMaxColumnsOverride = 4;
        Assert.True(CudaSmallSGemvDispatch.CoversKQuant(4));
        Assert.False(CudaSmallSGemvDispatch.CoversKQuant(5));
        CudaSmallSGemvDispatch.KQuantMaxColumnsOverride = 0;
        Assert.False(CudaSmallSGemvDispatch.CoversKQuant(2));
    }

    [Fact]
    public void Rows4ShapeRule_NeedsSuperblockMultiplesAtOrAboveTheLargeThreshold()
    {
        Assert.True(CudaKernels.Rows4ShapeOk(1024));
        Assert.True(CudaKernels.Rows4ShapeOk(8960));
        Assert.False(CudaKernels.Rows4ShapeOk(768));      // below MmvqLargeKThreshold (decode keeps MMQ-4-rows)
        Assert.False(CudaKernels.Rows4ShapeOk(1408));     // 32-multiple but not a 256-multiple
        Assert.Equal(8, CudaKernels.KQuantRows4MaxColumns);
        Assert.Equal(6, CudaKernels.Rows4Types.Length);
    }
}
