using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Q4_K / Q6_K decode GEMV on the warp-per-row kernels (<c>quantized_gemv_{q4_k,q6_k}_mmvq_rows4_preq</c>, #1298).
/// The oracle dequantises the weights straight from the llama.cpp reference layout (dequantize_row_q4_K / q6_K) and
/// multiplies by the dequantised Q8_1 activation (dx * xq, requantised exactly as <c>quantize_x_to_q8_1</c> does), so it
/// is independent of the kernel's own sub-block / nibble / scale indexing: a swapped nibble half, a wrong scale slot or
/// a dropped min term shows up at O(1), not at the tolerance. Weights are random bytes (distinct scales per sub-block,
/// full-range packed scales) and shapes are not tile-friendly (odd n, odd superblock counts, n % 4 != 0).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaKQuantRows4Tests
{
    private readonly ITestOutputHelper _out;
    public CudaKQuantRows4Tests(ITestOutputHelper output) => _out = output;

    private static int BlockBytes(QuantizationType qt) => qt == QuantizationType.Q4_K ? 144 : 210;

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K, 1, 1024)]
    [InlineData(QuantizationType.Q4_K, 3, 1280)]      // 5 superblocks: odd count, half-empty tail step
    [InlineData(QuantizationType.Q4_K, 37, 1792)]     // 7 superblocks, n % 4 = 1
    [InlineData(QuantizationType.Q4_K, 1537, 1536)]
    [InlineData(QuantizationType.Q4_K, 1536, 8960)]
    [InlineData(QuantizationType.Q4_K, 17920, 1536)]
    [InlineData(QuantizationType.Q6_K, 1, 1024)]
    [InlineData(QuantizationType.Q6_K, 5, 1280)]
    [InlineData(QuantizationType.Q6_K, 39, 1792)]
    [InlineData(QuantizationType.Q6_K, 1537, 1536)]
    [InlineData(QuantizationType.Q6_K, 1536, 8960)]
    [InlineData(QuantizationType.Q6_K, 32000, 1536)]
    public void Rows4_MatchesDequantOracle(QuantizationType qt, int n, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (w, x) = Synthesise(qt, n, k, seed: 1298 ^ n ^ k ^ (int)qt);
        float[] expected = Oracle(qt, w, x, n, k);
        Half[] actual = Run(qt, w, x, n, k, "rows4", out bool ran);
        Assert.True(ran, "rows4 kernel not loaded (stale PTX?)");
        Compare(qt, n, k, expected, actual, maxRel: 1e-3);
    }

    /// <summary>The legacy kernel must agree with the same oracle, so the A/B switch compares two correct kernels.</summary>
    [SkippableTheory]
    [InlineData(QuantizationType.Q4_K, 1537, 1536)]
    [InlineData(QuantizationType.Q6_K, 1537, 1536)]
    public void Large_MatchesDequantOracle_AndSwitchReroutes(QuantizationType qt, int n, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (w, x) = Synthesise(qt, n, k, seed: 99 ^ n ^ k);
        float[] expected = Oracle(qt, w, x, n, k);
        Half[] large = Run(qt, w, x, n, k, "large", out _);
        Compare(qt, n, k, expected, large, maxRel: 1e-3);
        Half[] rows4 = Run(qt, w, x, n, k, "rows4", out _);
        // Different summation order => FP16 results must differ somewhere, else the switch is inert.
        Assert.Contains(Enumerable.Range(0, n), i => large[i] != rows4[i]);
    }

    /// <summary>Opt-in microbench (<c>DOTLLM_KQUANT_BENCH=1</c>): large vs rows4 on the Bielik-1.5B decode shapes.</summary>
    [SkippableFact]
    public void Bench_KQuant_Decode_Shapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_KQUANT_BENCH") == "1", "set DOTLLM_KQUANT_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        nint s = stream.Handle;
        CudaDriverApi.cuEventCreate(out nint e0, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint e1, 0).ThrowOnError();
        var allocs = new List<nint>();
        nint Alloc(long bytes, byte fill = 0x11)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(d, fill, (nuint)bytes).ThrowOnError();
            allocs.Add(d);
            return d;
        }
        double TimeUs(Action launch, int reps)
        {
            for (int i = 0; i < 10; i++) launch();
            CudaDriverApi.cuEventRecord(e0, s).ThrowOnError();
            for (int i = 0; i < reps; i++) launch();
            CudaDriverApi.cuEventRecord(e1, s).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(e1).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, e0, e1).ThrowOnError();
            return ms * 1000.0 / reps;
        }
        string prevMode = CudaKernels.KQuantGemvMode;
        try
        {
            _out.WriteLine("path: large = quantized_gemv_*_mmvq_large_preq (1 row/block), rows4 = quantized_gemv_*_mmvq_rows4_preq (warp/row)");
            _out.WriteLine("shape                       | MB/call | large us (GB/s)   | rows4 us (GB/s)  | speedup");
            foreach (var (qt, n, k, label) in new[]
                     {
                         (QuantizationType.Q4_K, 17920, 1536, "gate+up fused"), (QuantizationType.Q4_K, 8960, 1536, "gate or up"),
                         (QuantizationType.Q4_K, 1536, 1536, "O / small sq"), (QuantizationType.Q4_K, 2048, 1536, "QKV-ish"),
                         (QuantizationType.Q4_K, 1536, 8960, "down (Q4_K)"),
                         (QuantizationType.Q6_K, 1536, 8960, "down (Q6_K)"), (QuantizationType.Q6_K, 32000, 1536, "LM head"),
                         (QuantizationType.Q6_K, 1536, 1536, "Q6_K small sq"),
                     })
            {
                long bytes = (long)n * (k / 256) * BlockBytes(qt);
                int copies = (int)Math.Max(1, 96L * 1024 * 1024 / bytes + 1);
                var ws = Enumerable.Range(0, copies).Select(_ => Alloc(bytes)).ToArray();
                nint xh = Alloc((long)k * 2, 0x3C), yh = Alloc((long)n * 2), scratch = Alloc(CudaForwardState.PreQ8_1ScratchBytes(k));
                kernels.LaunchQuantizeXToQ8_1(xh, scratch, k, s);
                int rep = 0;
                CudaKernels.KQuantGemvMode = "large";
                double large = TimeUs(() => kernels.LaunchQuantizedGemvMmq(ws[rep++ % copies], qt, xh, yh, n, k, scratch, s), 200);
                rep = 0;
                CudaKernels.KQuantGemvMode = "rows4";
                double rows4 = TimeUs(() => kernels.LaunchQuantizedGemvMmq(ws[rep++ % copies], qt, xh, yh, n, k, scratch, s), 200);
                _out.WriteLine($"{qt} {n,6}x{k,-5} {label,-13} | {bytes / 1e6,7:F1} | {large,8:F1} ({bytes / large / 1e3,5:F0}) | {rows4,8:F1} ({bytes / rows4 / 1e3,5:F0}) | {large / rows4:F2}x");
                foreach (nint p in ws.Append(xh).Append(yh).Append(scratch)) { CudaDriverApi.cuMemFree_v2(p); allocs.Remove(p); }
            }
        }
        finally
        {
            CudaKernels.KQuantGemvMode = prevMode;
            foreach (nint p in allocs) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private void Compare(QuantizationType qt, int n, int k, float[] expected, Half[] actual, double maxRel)
    {
        double peak = expected.Max(MathF.Abs);
        double maxErr = 0;
        for (int i = 0; i < n; i++) maxErr = Math.Max(maxErr, Math.Abs((float)actual[i] - expected[i]));
        _out.WriteLine($"{qt} n={n} k={k}: peak|y|={peak:F3} max|err|={maxErr:E2} peak-rel={maxErr / peak:E2}");
        Assert.True(maxErr / peak < maxRel, $"peak-relative error {maxErr / peak:E2} (max abs {maxErr})");
    }

    private static (byte[] Weights, Half[] X) Synthesise(QuantizationType qt, int n, int k, int seed)
    {
        var rng = new Random(seed);
        int bb = BlockBytes(qt), nsb = k / 256;
        byte[] w = new byte[(long)n * nsb * bb];
        rng.NextBytes(w);
        for (long b = 0; b < (long)n * nsb; b++)
        {
            int off = (int)(b * bb);
            if (qt == QuantizationType.Q4_K)
            {
                Half d = (Half)(0.0005 + rng.NextDouble() * 0.004), dmin = (Half)(0.0005 + rng.NextDouble() * 0.004);
                MemoryMarshal.Write(w.AsSpan(off, 2), in d);
                MemoryMarshal.Write(w.AsSpan(off + 2, 2), in dmin);
            }
            else
            {
                Half d = (Half)(0.0005 + rng.NextDouble() * 0.004);
                MemoryMarshal.Write(w.AsSpan(off + 208, 2), in d);
            }
        }
        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double outlier = (i / 32) % 5 == 2 ? 6.0 : 1.0;
            x[i] = (Half)(g * 0.4 * outlier);
        }
        return (w, x);
    }

    private static void ScaleMinK4(int j, ReadOnlySpan<byte> q, out int sc, out int m)
    {
        if (j < 4) { sc = q[j] & 63; m = q[j + 4] & 63; }
        else { sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
    }

    /// <summary>llama.cpp dequantize_row_q4_K / dequantize_row_q6_K for one superblock into <paramref name="y"/>[256].</summary>
    private static void DequantBlock(QuantizationType qt, ReadOnlySpan<byte> blk, Span<float> y)
    {
        if (qt == QuantizationType.Q4_K)
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
        }
        else
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
        }
    }

    private static float[] Oracle(QuantizationType qt, byte[] w, Half[] x, int n, int k)
    {
        int bpr = k / 32, nsb = k / 256, bb = BlockBytes(qt);
        float[] xd = new float[k];   // dequantised Q8_1 activation = dx * xq
        for (int c = 0; c < bpr; c++)
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
        float[] y = new float[n];
        float[] deq = new float[256];
        for (int r = 0; r < n; r++)
        {
            double acc = 0;
            for (int sb = 0; sb < nsb; sb++)
            {
                DequantBlock(qt, w.AsSpan((int)(((long)r * nsb + sb) * bb), bb), deq);
                for (int i = 0; i < 256; i++) acc += (double)deq[i] * xd[sb * 256 + i];
            }
            y[r] = (float)acc;
        }
        return y;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static Half[] Run(QuantizationType qt, byte[] w, Half[] x, int n, int k, string mode, out bool ranRows4)
    {
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        string prev = CudaKernels.KQuantGemvMode;
        CudaKernels.KQuantGemvMode = mode;
        try
        {
            ranRows4 = mode == "rows4" && kernels.HasMmvqRows4Preq(qt);
            Skip.If(mode == "rows4" && !ranRows4, "rows4 kernel not loaded (stale PTX?)");
            long wBytes = w.Length, xBytes = (long)k * 2, yBytes = (long)n * 2;
            CudaDriverApi.cuMemAlloc_v2(out nint dW, (nuint)wBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint dX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint dY, (nuint)yBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint dS, (nuint)CudaForwardState.PreQ8_1ScratchBytes(k)).ThrowOnError();
            try
            {
                fixed (byte* pW = w) CudaDriverApi.cuMemcpyHtoD_v2(dW, (nint)pW, (nuint)wBytes).ThrowOnError();
                fixed (Half* pX = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)pX, (nuint)xBytes).ThrowOnError();
                CudaDriverApi.cuMemsetD8_v2(dY, 0xFF, (nuint)yBytes).ThrowOnError();   // poison: an unwritten row cannot pass as 0
                kernels.LaunchQuantizeXToQ8_1(dX, dS, k, stream.Handle);
                kernels.LaunchQuantizedGemvMmq(dW, qt, dX, dY, n, k, dS, stream.Handle);
                stream.Synchronize();
                Half[] y = new Half[n];
                fixed (Half* pY = y) CudaDriverApi.cuMemcpyDtoH_v2((nint)pY, dY, (nuint)yBytes).ThrowOnError();
                return y;
            }
            finally
            {
                CudaDriverApi.cuMemFree_v2(dW); CudaDriverApi.cuMemFree_v2(dX);
                CudaDriverApi.cuMemFree_v2(dY); CudaDriverApi.cuMemFree_v2(dS);
            }
        }
        finally { CudaKernels.KQuantGemvMode = prev; }
    }

    private static string? FindPtxDir()
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
