using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Q8_0 decode GEMV on the dp4a MMVQ-large kernel (<c>quantized_gemv_q8_0_mmvq_large_preq</c>, #1297).
/// The oracle is a scalar CPU model of the kernel's contract, NOT the legacy kernel: it requantises x exactly as
/// <c>quantize_x_to_q8_1</c> does (127/absmax, round-to-nearest-even, clamp +-127, FP16 scale) and takes an exact
/// integer dot per 32-block times the FP16 block scales. A 1-3% tolerance against the legacy kernel can hide an
/// indexing bug; this one cannot (only F32 add-order and the final FP16 rounding are left). Weights get a different
/// FP16 scale and random int8 payload per block, x has outlier chunks, and shapes are deliberately not tile-friendly
/// (odd n, 32*odd k) so a swapped row/block stride, a dropped tail block or a misaligned 32-bit load shows up.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaQ8_0MmvqTests
{
    private const int BlockBytes = 34;
    private readonly ITestOutputHelper _out;

    public CudaQ8_0MmvqTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(1, 1024)]       // single row, boundary k
    [InlineData(3, 1056)]       // 33 blocks per row: k = 32*odd, not a multiple of 256 or of the 32-block step
    [InlineData(37, 2080)]      // 65 blocks per row, prime-ish n
    [InlineData(2049, 1536)]    // odd n
    [InlineData(2048, 2048)]    // Llama-3.2-1B O projection
    [InlineData(3072, 2048)]    // fused QKV
    [InlineData(2048, 8192)]    // down
    [InlineData(16384, 2048)]   // fused gate+up
    public void MmvqLargePreq_MatchesScalarOracle(int n, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (w, x) = Synthesise(n, k, seed: 1297 ^ n ^ k);
        float[] expected = Oracle(w, x, n, k);
        Half[] actual = RunKernel(w, x, n, k, disableNewKernel: false, out bool ranNewKernel);
        Assert.True(ranNewKernel, "HasMmq(Q8_0) is false: the new kernel did not run (stale PTX?)");

        double peak = expected.Max(MathF.Abs);
        double maxErr = 0;
        for (int i = 0; i < n; i++) maxErr = Math.Max(maxErr, Math.Abs((float)actual[i] - expected[i]));
        _out.WriteLine($"Q8_0 MMVQ n={n} k={k}: peak|y|={peak:F3} max|err|={maxErr:E2} peak-rel={maxErr / peak:E2}");
        // FP16 output rounding is 4.9e-4 of the value; allow 2x for the F32 reduction-order difference.
        Assert.True(maxErr / peak < 1e-3, $"peak-relative error {maxErr / peak:E2} (max abs {maxErr})");
    }

    /// <summary>
    /// Below the pre-quantise threshold there is no scratch, so the dispatcher must hand the call to the legacy FP
    /// kernel instead of throwing (SmolLM-135M has k=576). The legacy kernel keeps FP16 x, so compare it to the
    /// oracle at the activation-quantisation noise level only.
    /// </summary>
    [SkippableTheory]
    [InlineData(5, 576)]
    [InlineData(64, 224)]
    public void BelowThreshold_FallsBackToLegacy(int n, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (w, x) = Synthesise(n, k, seed: 4242 ^ n ^ k);
        float[] expected = Oracle(w, x, n, k);
        Half[] actual = RunKernel(w, x, n, k, disableNewKernel: false, out _, allowPreq: false);
        double peak = expected.Max(MathF.Abs);
        double maxErr = 0;
        for (int i = 0; i < n; i++) maxErr = Math.Max(maxErr, Math.Abs((float)actual[i] - expected[i]));
        Assert.True(maxErr / peak < 0.03, $"peak-relative error {maxErr / peak:E2}");
    }

    /// <summary>The kill switch must actually reroute: proves A/B runs compare two different kernels.</summary>
    [SkippableFact]
    public void DisableSwitch_RoutesToLegacy()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var (w, x) = Synthesise(256, 2048, seed: 7);
        Half[] legacy = RunKernel(w, x, 256, 2048, disableNewKernel: true, out bool ranNew);
        Half[] fast = RunKernel(w, x, 256, 2048, disableNewKernel: false, out bool ranNew2);
        Assert.False(ranNew);
        Assert.True(ranNew2);
        // Legacy keeps FP16 x, the new kernel quantises it: the outputs must differ somewhere (else the switch is inert).
        Assert.Contains(Enumerable.Range(0, 256), i => legacy[i] != fast[i]);
    }

    /// <summary>
    /// Opt-in microbench (<c>DOTLLM_Q8_0_BENCH=1</c>): legacy vs MMVQ-large on the Llama-3.2-1B decode shapes, CUDA events
    /// over back-to-back launches, weight GB/s = weight bytes per launch / time. Each shape rotates over enough weight
    /// copies to exceed the 3 MB L2, so this is DRAM bandwidth, not cache bandwidth.
    /// </summary>
    [SkippableFact]
    public void Bench_Q8_0_Decode_Shapes()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_Q8_0_BENCH") == "1", "set DOTLLM_Q8_0_BENCH=1");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        Skip.IfNot(kernels.HasMmqQ8_0, "new Q8_0 kernel not loaded (stale PTX)");
        nint s = stream.Handle;
        CudaDriverApi.cuEventCreate(out nint e0, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out nint e1, 0).ThrowOnError();
        var allocs = new List<nint>();
        nint Alloc(long bytes)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
            CudaDriverApi.cuMemsetD8_v2(d, 0x11, (nuint)bytes).ThrowOnError();
            allocs.Add(d);
            return d;
        }
        double TimeUs(Action launch, int reps)
        {
            for (int i = 0; i < 5; i++) launch();
            CudaDriverApi.cuEventRecord(e0, s).ThrowOnError();
            for (int i = 0; i < reps; i++) launch();
            CudaDriverApi.cuEventRecord(e1, s).ThrowOnError();
            CudaDriverApi.cuEventSynchronize(e1).ThrowOnError();
            CudaDriverApi.cuEventElapsedTime(out float ms, e0, e1).ThrowOnError();
            return ms * 1000.0 / reps;
        }
        try
        {
            _out.WriteLine("shape                    | MB/call | legacy us (GB/s)  | mmvq us (GB/s)   | speedup");
            foreach (var (n, k, label) in new[]
                     {
                         (16384, 2048, "gate+up fused"), (3072, 2048, "QKV fused"), (2048, 2048, "O"),
                         (2048, 8192, "down"), (128256, 2048, "LM head"),
                     })
            {
                long bytes = (long)n * (k / 32) * BlockBytes;
                int copies = (int)Math.Max(1, 64L * 1024 * 1024 / bytes + 1);
                var ws = Enumerable.Range(0, copies).Select(_ => Alloc(bytes)).ToArray();
                nint xh = Alloc((long)k * 2), yh = Alloc((long)n * 2), scratch = Alloc(CudaForwardState.PreQ8_1ScratchBytes(k));
                int rep = 0;
                double legacy = TimeUs(() => kernels.LaunchQuantizedGemv(ws[rep++ % copies], QuantizationType.Q8_0, xh, yh, n, k, s), 100);
                rep = 0;
                double fast = TimeUs(() =>
                {
                    kernels.LaunchQuantizeXToQ8_1(xh, scratch, k, s);
                    kernels.LaunchQuantizedGemvMmq(ws[rep++ % copies], QuantizationType.Q8_0, xh, yh, n, k, scratch, s);
                }, 100);
                _out.WriteLine($"{n,6}x{k,-5} {label,-13} | {bytes / 1e6,7:F1} | {legacy,8:F1} ({bytes / legacy / 1e3,5:F0}) | {fast,8:F1} ({bytes / fast / 1e3,5:F0}) | {legacy / fast:F2}x");
            }
        }
        finally
        {
            foreach (nint p in allocs) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    private static (byte[] Weights, Half[] X) Synthesise(int n, int k, int seed)
    {
        var rng = new Random(seed);
        int bpr = k / 32;
        byte[] w = new byte[(long)n * bpr * BlockBytes];
        for (int r = 0; r < n; r++)
        for (int b = 0; b < bpr; b++)
        {
            int off = (r * bpr + b) * BlockBytes;
            // Distinct scale per block (also catches a scale read from the wrong block); full int8 range incl. -128.
            Half d = (Half)(0.002 + rng.NextDouble() * 0.03);
            MemoryMarshal.Write(w.AsSpan(off, 2), in d);
            for (int j = 0; j < 32; j++) w[off + 2 + j] = (byte)rng.Next(-128, 128);
        }
        Half[] x = new Half[k];
        for (int i = 0; i < k; i++)
        {
            double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
            double g = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
            double outlier = (i / 32) % 5 == 2 ? 6.0 : 1.0;   // a few chunks with a much larger absmax
            x[i] = (Half)(g * 0.4 * outlier);
        }
        return (w, x);
    }

    /// <summary>Scalar model of the kernel: requantise x like quantize_x_to_q8_1, exact int dot, FP16 scales.</summary>
    private static float[] Oracle(byte[] w, Half[] x, int n, int k)
    {
        int bpr = k / 32;
        sbyte[] xq = new sbyte[k];
        float[] dx = new float[bpr];
        for (int c = 0; c < bpr; c++)
        {
            float a = 0f;
            for (int j = 0; j < 32; j++) a = MathF.Max(a, MathF.Abs((float)x[c * 32 + j]));
            float inv = a > 0f ? 127.0f / a : 0f;
            for (int j = 0; j < 32; j++)
            {
                int q = (int)MathF.Round((float)x[c * 32 + j] * inv, MidpointRounding.ToEven);
                xq[c * 32 + j] = (sbyte)Math.Clamp(q, -127, 127);
            }
            dx[c] = (float)(Half)(a / 127.0f);
        }
        float[] y = new float[n];
        for (int r = 0; r < n; r++)
        {
            double acc = 0;
            for (int b = 0; b < bpr; b++)
            {
                int off = (r * bpr + b) * BlockBytes;
                float d = (float)MemoryMarshal.Read<Half>(w.AsSpan(off, 2));
                long dot = 0;
                for (int j = 0; j < 32; j++) dot += (sbyte)w[off + 2 + j] * xq[b * 32 + j];
                acc += (double)d * dx[b] * dot;
            }
            y[r] = (float)acc;
        }
        return y;
    }

    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static Half[] RunKernel(byte[] w, Half[] x, int n, int k, bool disableNewKernel, out bool ranNewKernel,
                                    bool allowPreq = true)
    {
        string ptx = FindPtxDir() ?? throw new InvalidOperationException("ptx dir not found");
        using var ctx = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ptx);
        bool prev = CudaKernels.DisableMmqQ8_0;
        CudaKernels.DisableMmqQ8_0 = disableNewKernel;
        try
        {
            ranNewKernel = kernels.HasMmq(QuantizationType.Q8_0);
            Skip.If(!disableNewKernel && !ranNewKernel, "Q8_0 MMVQ kernel not loaded (stale PTX?)");
            long wBytes = w.Length, xBytes = (long)k * 2, yBytes = (long)n * 2;
            CudaDriverApi.cuMemAlloc_v2(out nint dW, (nuint)wBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint dX, (nuint)xBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint dY, (nuint)yBytes).ThrowOnError();
            CudaDriverApi.cuMemAlloc_v2(out nint dS, (nuint)CudaForwardState.PreQ8_1ScratchBytes(k)).ThrowOnError();
            try
            {
                fixed (byte* pW = w) CudaDriverApi.cuMemcpyHtoD_v2(dW, (nint)pW, (nuint)wBytes).ThrowOnError();
                fixed (Half* pX = x) CudaDriverApi.cuMemcpyHtoD_v2(dX, (nint)pX, (nuint)xBytes).ThrowOnError();
                // Poison the output so an unwritten row cannot pass as zero.
                CudaDriverApi.cuMemsetD8_v2(dY, 0xFF, (nuint)yBytes).ThrowOnError();
                nint scratch = 0;
                if (allowPreq && kernels.HasPreQ8_1)
                {
                    kernels.LaunchQuantizeXToQ8_1(dX, dS, k, stream.Handle);
                    scratch = dS;
                }
                if (kernels.HasMmq(QuantizationType.Q8_0))
                    kernels.LaunchQuantizedGemvMmq(dW, QuantizationType.Q8_0, dX, dY, n, k, scratch, stream.Handle);
                else
                    kernels.LaunchQuantizedGemv(dW, QuantizationType.Q8_0, dX, dY, n, k, stream.Handle);
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
        finally { CudaKernels.DisableMmqQ8_0 = prev; }
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
