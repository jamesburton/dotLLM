using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Tests.Unit.Models.Flint;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// The in-place Q4_1 kernels (<c>q4_1_gemv.cu</c>, #1172) and the sigmoid router (<c>moe_sigmoid_topk_f32</c>) against
/// host references: dense GEMV, multi-token GEMM (FP16 and FP32 activations, token tails), the MoE grouped GEMV and the
/// sigmoid top-k. The references dequantise the same Q4_1 bytes on the CPU (<see cref="Dequantize.ToFloat32"/>) and
/// accumulate in double, so a nibble-order, min-term or row-stride bug shows up as an O(1) error, not noise. Shapes
/// include rows not divisible by the 8-row tile and k = 96 (3 blocks: fewer than one 8-block warp pass).
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaQ4_1KernelTests
{
    private readonly ITestOutputHelper _out;

    public CudaQ4_1KernelTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(2048, 2048)]   // Flint attention / shared-expert shape
    [InlineData(512, 2048)]    // routed expert gate/up
    [InlineData(2048, 512)]    // routed expert down
    [InlineData(37, 96)]       // ragged rows, k below one warp pass
    public void Gemv_F16_MatchesDequantReference(int n, int k)
    {
        using var g = Gpu.Open();
        var w = RandomWeights(n, k, seed: n ^ k);
        var x = RandomActivations(1, k, seed: 7 + k, scale: 1f);
        nint dW = g.Upload(w.Bytes), dX = g.UploadHalf(x), dY = g.Alloc(n * 2);
        g.K.LaunchQuantizedGemv(dW, QuantizationType.Q4_1, dX, dY, n, k, g.S.Handle);
        g.S.Synchronize();
        AssertClose(Reference(w, RoundHalf(x), 1, n, k), g.DownloadHalf(dY, n), relTol: 3e-3, $"gemv f16 {n}x{k}");
    }

    [SkippableTheory]
    [InlineData(64, 2048, 2)]
    [InlineData(64, 2048, 8)]
    [InlineData(100, 96, 13)]   // token tail (13 = 8 + 5) and ragged rows
    [InlineData(512, 2048, 33)]
    public void Gemm_F16_MatchesDequantReference(int n, int k, int m)
    {
        using var g = Gpu.Open();
        Skip.IfNot(g.K.HasQ4_1Gemm, "Q4_1 GEMM not loaded");
        var w = RandomWeights(n, k, seed: n ^ k ^ m);
        var x = RandomActivations(m, k, seed: 11 + m, scale: 1f);
        nint dW = g.Upload(w.Bytes), dX = g.UploadHalf(x), dY = g.Alloc(m * n * 2);
        g.K.LaunchQuantizedGemmQ4_1F16(dW, dX, dY, n, k, m, g.S.Handle);
        g.S.Synchronize();
        AssertClose(Reference(w, RoundHalf(x), m, n, k), g.DownloadHalf(dY, m * n), relTol: 3e-3, $"gemm f16 {n}x{k} m={m}");
    }

    /// <summary>
    /// F32 activations, including a row of ~2e5 values (beyond F16): the MoE down projection consumes Flint's routed SwiGLU
    /// output, which reaches ~1.3e4; the F32-in kernel must stay finite and exact where an F16 round trip would be inf.
    /// </summary>
    [SkippableTheory]
    [InlineData(2048, 512, 1)]
    [InlineData(512, 2048, 5)]
    [InlineData(100, 96, 16)]
    public void Gemm_F32_MatchesDequantReference_IncludingValuesBeyondF16(int n, int k, int m)
    {
        using var g = Gpu.Open();
        Skip.IfNot(g.K.HasQ4_1Gemm, "Q4_1 GEMM not loaded");
        var w = RandomWeights(n, k, seed: 3 * n + k + m);
        var x = RandomActivations(m, k, seed: 5 + m, scale: 1f);
        for (int i = 0; i < k; i++) x[i] *= 2e5f;   // row 0 overflows F16
        nint dW = g.Upload(w.Bytes), dX = g.UploadFloat(x), dY = g.Alloc(m * n * 4);
        g.K.LaunchQuantizedGemmQ4_1F32(dW, dX, dY, n, k, m, g.S.Handle);
        g.S.Synchronize();
        float[] y = g.DownloadFloat(dY, m * n);
        Assert.All(y, v => Assert.True(float.IsFinite(v)));
        AssertClose(Reference(w, x, m, n, k), y, relTol: 2e-5, $"gemm f32 {n}x{k} m={m}");
    }

    [SkippableTheory]
    [InlineData(6, 512, 2048)]   // Flint gate/up: top-6, 512 rows, hidden 2048
    [InlineData(3, 32, 64)]      // the synthetic Flint GGUF's shape
    public void GroupedGemv_MatchesPerExpertGemvBitExact(int kActive, int m, int k)
    {
        using var g = Gpu.Open();
        Skip.IfNot(g.K.HasMoeGroupedGemv(QuantizationType.Q4_1), "Q4_1 grouped GEMV not loaded");
        var x = RandomActivations(1, k, seed: 99, scale: 1f);
        nint dX = g.UploadHalf(x);
        var dW = new nint[kActive];
        var dYg = new nint[kActive];
        var dYp = new nint[kActive];
        var weights = new Q4_1Matrix[kActive];
        for (int e = 0; e < kActive; e++)
        {
            weights[e] = RandomWeights(m, k, seed: 1000 + e);
            dW[e] = g.Upload(weights[e].Bytes);
            dYg[e] = g.Alloc(m * 2);
            dYp[e] = g.Alloc(m * 2);
        }
        long* ptrs = stackalloc long[2 * kActive];
        for (int e = 0; e < kActive; e++) { ptrs[e] = dW[e]; ptrs[kActive + e] = dYg[e]; }
        nint dPtrs = g.Alloc(2 * kActive * sizeof(long));
        CudaDriverApi.cuMemcpyHtoD_v2(dPtrs, (nint)ptrs, (nuint)(2 * kActive * sizeof(long))).ThrowOnError();

        g.K.LaunchMoeGroupedGemv(dPtrs, dPtrs + kActive * sizeof(long), dX, QuantizationType.Q4_1, m, k, kActive, g.S.Handle);
        for (int e = 0; e < kActive; e++)
            g.K.LaunchQuantizedGemv(dW[e], QuantizationType.Q4_1, dX, dYp[e], m, k, g.S.Handle);
        g.S.Synchronize();

        for (int e = 0; e < kActive; e++)
        {
            float[] grouped = g.DownloadHalf(dYg[e], m), perCall = g.DownloadHalf(dYp[e], m);
            Assert.Equal(perCall, grouped);   // same row body, same reduction order
            AssertClose(Reference(weights[e], RoundHalf(x), 1, m, k), grouped, relTol: 3e-3, $"grouped e={e}");
        }
    }

    /// <summary>Sigmoid top-6 of 256 with renormalisation, three tokens, against the CPU routing rule (sum clamp, lower index on ties).</summary>
    [SkippableFact]
    public void SigmoidTopk_MatchesHostReference()
    {
        using var g = Gpu.Open();
        Skip.IfNot(g.K.HasMoeSigmoidTopk, "moe_sigmoid_topk_f32 not loaded");
        const int tokens = 3, experts = 256, topK = 6;
        var logits = RandomActivations(tokens, experts, seed: 4242, scale: 2f);
        logits[experts + 17] = logits[experts + 18] = 9f;   // exact tie: index 17 must win the first slot of token 1
        nint dL = g.UploadFloat(logits), dIdx = g.Alloc(tokens * topK * 4), dW = g.Alloc(tokens * topK * 4);
        g.K.LaunchMoeSigmoidTopk(dL, dIdx, dW, tokens, experts, topK, normalize: true, g.S.Handle);
        g.S.Synchronize();
        var idx = new int[tokens * topK];
        fixed (int* p = idx) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dIdx, (nuint)(idx.Length * 4)).ThrowOnError();
        float[] wts = g.DownloadFloat(dW, tokens * topK);

        for (int t = 0; t < tokens; t++)
        {
            var sig = new float[experts];
            for (int e = 0; e < experts; e++) sig[e] = 1f / (1f + MathF.Exp(-logits[t * experts + e]));
            var chosen = new List<int>();
            for (int s = 0; s < topK; s++)
            {
                int best = -1;
                for (int e = 0; e < experts; e++)
                    if (!chosen.Contains(e) && (best < 0 || sig[e] > sig[best])) best = e;
                chosen.Add(best);
            }
            float sum = MathF.Max(chosen.Sum(e => sig[e]), 6.103515625e-5f);
            for (int s = 0; s < topK; s++)
            {
                Assert.Equal(chosen[s], idx[t * topK + s]);
                Assert.Equal(sig[chosen[s]] / sum, wts[t * topK + s], 5e-6f);
            }
        }
        Assert.Equal(17, idx[topK]);
        Assert.Equal(18, idx[topK + 1]);
    }

    /// <summary>
    /// The CUDA mirror of the CPU's F16-safe MoE down input (#1158): the LFM2 packed Q4_0 x Q8_1 GEMM stores each block
    /// sum in F16, so a ~2e5 SwiGLU row turns into inf without the power-of-two rescale and matches the reference with it.
    /// Also pins the threshold (1024 untouched, 1025 halved) and that the round trip is exact.
    /// </summary>
    [SkippableFact]
    public void Pow2DownRescale_KeepsPackedQ4_0Q8_1Finite_AndIsExact()
    {
        using var g = Gpu.Open();
        const int rows = 64, k = 512;
        var rng = new Random(17);
        var wf = new float[rows * k];
        for (int i = 0; i < wf.Length; i++) wf[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f;
        byte[] q4 = EncodeQ4_0(wf);
        var deq = new float[wf.Length];
        fixed (byte* p = q4) Dequantize.ToFloat32((nint)p, wf.Length, QuantizationType.Q4_0, deq);
        var x = RandomActivations(1, k, seed: 3, scale: 2e5f);

        using var lfm = new DotLLM.Cuda.Architectures.CudaLfm2Kernels(g.PtxDir);
        nint dW = g.Upload(q4), dQ = g.Alloc(k / 32 * 36), dY = g.Alloc(rows * 4), dF = g.Alloc(4);

        nint dX = g.UploadFloat(x);
        lfm.QuantizeQ8(dX, dQ, k / 32, g.S.Handle);
        lfm.GemmQ4Q8(dW, dQ, dY, rows, k, 1, g.S.Handle);
        g.S.Synchronize();
        Assert.Contains(g.DownloadFloat(dY, rows), v => !float.IsFinite(v));   // the hazard is real

        dX = g.UploadFloat(x);
        g.K.LaunchMoePow2RescaleRows(dX, dF, 1, k, g.S.Handle);
        lfm.QuantizeQ8(dX, dQ, k / 32, g.S.Handle);
        lfm.GemmQ4Q8(dW, dQ, dY, rows, k, 1, g.S.Handle);
        g.K.LaunchMoeScaleRows(dY, dF, 1, rows, g.S.Handle);
        g.S.Synchronize();
        float[] y = g.DownloadFloat(dY, rows);
        Assert.All(y, v => Assert.True(float.IsFinite(v)));
        AssertClose(Reference(new Q4_1Matrix(q4, deq), x, 1, rows, k), y, relTol: 2e-2, "q4_0 x q8_1 with rescale");

        // Threshold and exactness: 1024 stays, 1025 -> /2, and scaling back reproduces the input bit for bit.
        float[] probe = [1024f, -3.5f, 1025f, 0.1f];
        nint dP = g.UploadFloat(probe), dPf = g.Alloc(8);
        g.K.LaunchMoePow2RescaleRows(dP, dPf, 2, 2, g.S.Handle);
        g.S.Synchronize();
        Assert.Equal([1f, 2f], g.DownloadFloat(dPf, 2));
        Assert.Equal([1024f, -3.5f, 512.5f, 0.05f], g.DownloadFloat(dP, 4));
        g.K.LaunchMoeScaleRows(dP, dPf, 2, 2, g.S.Handle);
        g.S.Synchronize();
        Assert.Equal(probe, g.DownloadFloat(dP, 4));
    }

    // ── helpers ──

    /// <summary>ggml block_q4_0: f16 d, 16 bytes of nibbles (element j low, j + 16 high); x = (q - 8) * d.</summary>
    private static byte[] EncodeQ4_0(float[] data)
    {
        var bytes = new byte[data.Length / 32 * 18];
        for (int b = 0; b < data.Length / 32; b++)
        {
            var x = data.AsSpan(b * 32, 32);
            float amax = 0, max = 0;
            foreach (float v in x) if (MathF.Abs(v) > amax) { amax = MathF.Abs(v); max = v; }
            float d = max / -8f, id = d != 0 ? 1f / d : 0f;
            BitConverter.TryWriteBytes(bytes.AsSpan(b * 18, 2), (Half)d);
            for (int j = 0; j < 16; j++)
            {
                int lo = Math.Min(15, (int)(x[j] * id + 8.5f)), hi = Math.Min(15, (int)(x[j + 16] * id + 8.5f));
                bytes[b * 18 + 2 + j] = (byte)(lo | (hi << 4));
            }
        }
        return bytes;
    }

    private sealed record Q4_1Matrix(byte[] Bytes, float[] Dequant);

    private static Q4_1Matrix RandomWeights(int n, int k, int seed)
    {
        var rng = new Random(seed);
        var f = new float[(long)n * k];
        for (long i = 0; i < f.Length; i++) f[i] = (float)(rng.NextDouble() * 2 - 1) * 0.05f + 0.01f;   // non-zero mean: the m term matters
        byte[] bytes = SyntheticFlintGguf.EncodeQ4_1(f);
        var deq = new float[f.Length];
        fixed (byte* p = bytes) Dequantize.ToFloat32((nint)p, f.Length, QuantizationType.Q4_1, deq);
        return new Q4_1Matrix(bytes, deq);
    }

    private static float[] RandomActivations(int m, int k, int seed, float scale)
    {
        var rng = new Random(seed);
        var x = new float[(long)m * k];
        for (long i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return x;
    }

    private static float[] RoundHalf(float[] x) => x.Select(v => (float)(Half)v).ToArray();

    private static float[] Reference(Q4_1Matrix w, float[] x, int m, int n, int k)
    {
        var y = new float[m * n];
        for (int t = 0; t < m; t++)
            for (int r = 0; r < n; r++)
            {
                double acc = 0;
                for (int i = 0; i < k; i++) acc += (double)w.Dequant[(long)r * k + i] * x[(long)t * k + i];
                y[t * n + r] = (float)acc;
            }
        return y;
    }

    private void AssertClose(float[] expected, float[] actual, double relTol, string what)
    {
        double rms = Math.Sqrt(expected.Average(v => (double)v * v));
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = Math.Abs(expected[i] - actual[i]) / (Math.Abs(expected[i]) + rms);
            worst = Math.Max(worst, err);
        }
        _out.WriteLine($"{what}: worst scaled error {worst:E2} (rms {rms:E2})");
        Assert.True(worst <= relTol, $"{what}: worst scaled error {worst:E3} > {relTol:E1}");
    }

    /// <summary>Context, stream, kernels and tracked device allocations for one test.</summary>
    private sealed class Gpu : IDisposable
    {
        private readonly List<nint> _allocs = new();
        private CudaContext? _ctx;
        public CudaStream S { get; private set; } = null!;
        public CudaKernels K { get; private set; } = null!;
        public string PtxDir { get; private set; } = "";

        public static Gpu Open()
        {
            Skip.IfNot(DriverPresent(), "No CUDA GPU available");
            string? ptx = FindPtxDir();
            Skip.If(ptx is null, "PTX files not found");
            var g = new Gpu { _ctx = CudaContext.Create(0), PtxDir = ptx! };
            g.S = CudaStream.Create();
            g.K = new CudaKernels(ptx!);
            Skip.IfNot(g.K.HasQuantizedGemvKernel(QuantizationType.Q4_1), "Q4_1 GEMV not loaded (PTX may be stale)");
            return g;
        }

        public nint Alloc(long bytes)
        {
            CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
            _allocs.Add(d);
            return d;
        }

        public nint Upload(byte[] data)
        {
            nint d = Alloc(data.Length);
            fixed (byte* p = data) CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)data.Length).ThrowOnError();
            return d;
        }

        public nint UploadHalf(float[] x) => Upload(MemoryMarshal.AsBytes(x.Select(v => (Half)v).ToArray().AsSpan()).ToArray());

        public nint UploadFloat(float[] x) => Upload(MemoryMarshal.AsBytes(x.AsSpan()).ToArray());

        public float[] DownloadHalf(nint d, int count)
        {
            var h = new Half[count];
            fixed (Half* p = h) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(count * 2)).ThrowOnError();
            return h.Select(v => (float)v).ToArray();
        }

        public float[] DownloadFloat(nint d, int count)
        {
            var f = new float[count];
            fixed (float* p = f) CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(count * 4)).ThrowOnError();
            return f;
        }

        public void Dispose()
        {
            foreach (nint d in _allocs) CudaDriverApi.cuMemFree_v2(d);
            K?.Dispose();
            S?.Dispose();
            _ctx?.Dispose();
        }

        private static bool DriverPresent()
        {
            string lib = OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1";
            if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
            NativeLibrary.Free(h);
            return CudaDevice.IsAvailable();
        }

        private static string? FindPtxDir()
        {
            foreach (var dir in new[]
                     {
                         Path.Combine(AppContext.BaseDirectory, "ptx"),
                         Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
                     })
            {
                string full = Path.GetFullPath(dir);
                if (Directory.Exists(full) && File.Exists(Path.Combine(full, "q4_1_gemv.ptx")))
                    return full;
            }
            return null;
        }
    }
}
