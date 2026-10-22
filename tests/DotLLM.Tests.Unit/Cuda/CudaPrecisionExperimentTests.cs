using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1109 precision-study seams: the F16-accumulating flash prefill variants
/// (<c>attention_flash_mma_prefill_hd256_xacc.ptx</c>) against the shipped F32-accumulate kernels, and the
/// F16-accumulate GEMM with F32 promotion every K chunk against cuBLAS COMPUTE_16F / COMPUTE_32F.
/// Each test asserts the variant actually ran (its output differs from the F32 kernel) and that its error stays
/// inside a measured bound; the two "drifts more" tests discriminate a whole-range F16 accumulator from a promoted one.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaPrecisionExperimentTests : IDisposable
{
    private const int NumHeads = 24;
    private const int NumKvHeads = 4;
    private const int HeadDim = 256;

    private readonly ITestOutputHelper _out;
    private readonly CudaContext? _ctx;
    private readonly CudaStream? _stream;
    private readonly CudaKernels? _kernels;

    /// <summary>Creates the CUDA context, stream and kernel set when a device is available.</summary>
    /// <param name="output">xUnit output sink.</param>
    public CudaPrecisionExperimentTests(ITestOutputHelper output)
    {
        _out = output;
        if (!CudaDevice.IsAvailable()) return;
        _ctx = CudaContext.Create(0);
        _stream = CudaStream.Create();
        string? ptxDir = FindPtxDir();
        if (ptxDir != null) _kernels = new CudaKernels(ptxDir);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        _kernels?.Dispose();
        _stream?.Dispose();
        _ctx?.Dispose();
    }

    // ── flash prefill accumulators ──

    /// <summary>Upper bound of each variant's relative RMS error vs the F32 kernel at 4,224 keys (measured, see the test).</summary>
    /// <param name="mode">Variant.</param>
    /// <returns>The bound.</returns>
    private static double FlashTol(FlashAccumulatorMode mode) => mode switch
    {
        FlashAccumulatorMode.Qk16 => 2e-2,
        FlashAccumulatorMode.Pv16 => 2e-2,
        FlashAccumulatorMode.Pv16Tile => 2e-3,
        _ => 3e-2,
    };

    /// <summary>
    /// Every F16-KV variant runs (differs from the shipped kernel), stays finite and inside its bound at a 4k-key
    /// history.
    /// </summary>
    /// <param name="variant">Variant (DOTLLM_CUDA_FLASH_ACC value).</param>
    [SkippableTheory]
    [InlineData("qk16")]
    [InlineData("pv16")]
    [InlineData("pv16t")]
    [InlineData("qk16pv16")]
    public void F16KvVariant_IsEngagedAndNearF32Kernel(string variant)
    {
        FlashAccumulatorMode mode = CudaFlashAccumulatorExperiment.Parse(variant);
        SkipUnlessFlash();
        double rel = FlashVariantError(mode, seqQ: 128, posOffset: 4096, kvq: false);
        Assert.True(rel > 0, $"{mode} output is bit-identical to the F32 kernel: the variant did not run.");
        Assert.True(rel < FlashTol(mode), $"{mode}: relRms {rel:E3} >= {FlashTol(mode):E1}");
    }

    /// <summary>The Q4_0/Q4_0 entries run and stay inside the same bounds as the F16-KV ones.</summary>
    /// <param name="variant">Variant (DOTLLM_CUDA_FLASH_ACC value).</param>
    [SkippableTheory]
    [InlineData("qk16")]
    [InlineData("pv16")]
    [InlineData("pv16t")]
    [InlineData("qk16pv16")]
    public void Q4Q4Variant_IsEngagedAndNearF32Kernel(string variant)
    {
        FlashAccumulatorMode mode = CudaFlashAccumulatorExperiment.Parse(variant);
        SkipUnlessFlash();
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillHd256Kvq(KvCacheDType.Q4_0, KvCacheDType.Q4_0), "kvq prefill kernel not loaded");
        double rel = FlashVariantError(mode, seqQ: 128, posOffset: 4096, kvq: true);
        Assert.True(rel > 0, $"{mode} output is bit-identical to the F32 kernel: the variant did not run.");
        Assert.True(rel < FlashTol(mode), $"{mode}: relRms {rel:E3} >= {FlashTol(mode):E1}");
    }

    /// <summary>
    /// Discriminates the two PV forms: an F16 O accumulator carried over the whole key range loses far more than one
    /// promoted to F32 every 16 keys, and its error grows with the history.
    /// </summary>
    [SkippableFact]
    public void PvF16WholeRange_DriftsMoreThanPerTilePromotion_AndGrowsWithHistory()
    {
        SkipUnlessFlash();
        double whole4k = FlashVariantError(FlashAccumulatorMode.Pv16, 64, 4096, kvq: false);
        double tile4k = FlashVariantError(FlashAccumulatorMode.Pv16Tile, 64, 4096, kvq: false);
        double whole256 = FlashVariantError(FlashAccumulatorMode.Pv16, 64, 192, kvq: false);
        Assert.True(whole4k > 4 * tile4k, $"pv16 {whole4k:E2} is not clearly worse than pv16t {tile4k:E2}");
        Assert.True(whole4k > 2 * whole256, $"pv16 error does not grow with the key range ({whole256:E2} -> {whole4k:E2})");
    }

    private void SkipUnlessFlash()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        Skip.If(_kernels == null, "PTX kernels not available");
        Skip.IfNot(_kernels!.HasAttentionFlashPrefillHd256, "flash prefill hd256 kernel not loaded (pre-Ampere or stale PTX)");
    }

    /// <summary>Relative RMS of the variant's output against the shipped F32-accumulate kernel on the same inputs.</summary>
    private double FlashVariantError(FlashAccumulatorMode mode, int seqQ, int posOffset, bool kvq)
    {
        int seqKv = posOffset + seqQ;
        int qStride = NumHeads * HeadDim, kvStride = NumKvHeads * HeadDim;
        var rng = new Random(1109 + seqQ * 7 + posOffset);
        // Post-norm-like magnitudes: |q.k|/16 spans a few units, so softmax is neither flat nor one-hot.
        nint dQ = Upload(RandomHalf(rng, (long)seqQ * qStride, 1.7f));
        nint dK, dV;
        if (kvq)
        {
            dK = Upload(RandomQ4Blocks(rng, (long)seqKv * kvStride));
            dV = Upload(RandomQ4Blocks(rng, (long)seqKv * kvStride));
        }
        else
        {
            dK = Upload(RandomHalf(rng, (long)seqKv * kvStride, 1.7f));
            dV = Upload(RandomHalf(rng, (long)seqKv * kvStride, 1.0f));
        }
        long outElems = (long)seqQ * qStride;
        nint dRef = Alloc(outElems * 4), dGot = Alloc(outElems * 4);
        try
        {
            if (kvq)
                _kernels!.LaunchAttentionFlashPrefillHd256Kvq(dQ, dK, dV, KvCacheDType.Q4_0, KvCacheDType.Q4_0, dRef,
                    seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream!.Handle);
            else
                _kernels!.LaunchAttentionFlashPrefillHd256(dQ, dK, dV, dRef, seqQ, seqKv, NumHeads, NumKvHeads, posOffset, _stream!.Handle);
            _kernels.LaunchAttentionFlashPrefillHd256Xacc(mode, kvq, dQ, dK, dV, dGot, seqQ, seqKv, NumHeads, NumKvHeads,
                posOffset, _stream.Handle);
            _stream.Synchronize();
            float[] want = Download(dRef, outElems), got = Download(dGot, outElems);
            double sd = 0, sr = 0;
            for (long i = 0; i < outElems; i++)
            {
                Assert.True(float.IsFinite(got[i]), $"{mode}: non-finite output at {i}");
                double d = got[i] - want[i];
                sd += d * d;
                sr += (double)want[i] * want[i];
            }
            double rel = Math.Sqrt(sd / sr);
            _out.WriteLine($"{mode} kvq={kvq} seqQ={seqQ} seqKv={seqKv}: relRms vs F32 kernel = {rel:E3}");
            return rel;
        }
        finally
        {
            foreach (nint p in new[] { dQ, dK, dV, dRef, dGot }) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    // ── F16-accumulate GEMM with F32 promotion ──

    /// <summary>
    /// At Bonsai's longest reduction (K = 17,408, ffn_down) promoting the F16 accumulator to F32 every 256 elements
    /// recovers most of COMPUTE_32F's accuracy, while plain COMPUTE_16F does not.
    /// </summary>
    [SkippableFact]
    public void PromotedChunks_RecoverF32Accuracy_AtLongK()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        const int m = 32, k = 17408, n = 256, chunk = 256;
        var rng = new Random(1109);
        // Positive-mean activations (post-SwiGLU-like) make the running sum grow, which is where F16 accumulation loses.
        var x = new ushort[(long)m * k];
        for (long i = 0; i < x.Length; i++)
            x[i] = BitConverter.HalfToUInt16Bits((Half)(float)(rng.NextDouble() * 0.5 + 0.05));
        var w = RandomHalf(rng, (long)n * k, 0.05f);
        using var cublas = CudaCublasHandle.Create();
        nint dX = Upload(x), dW = Upload(w);
        nint dY32 = Alloc((long)m * n * 2), dY16 = Alloc((long)m * n * 2), dYp = Alloc((long)m * n * 2);
        nint dPart = Alloc((long)(k / chunk) * m * n * 2);
        var ones = new ushort[k / chunk];
        Array.Fill(ones, BitConverter.HalfToUInt16Bits(Half.One));
        nint dOnes = Upload(ones);
        try
        {
            nint s = _stream!.Handle;
            CudaGemm.LinearF16WithCompute(cublas.Handle, dX, dW, dY32, m, k, n, compute16F: false, s);
            CudaGemm.LinearF16WithCompute(cublas.Handle, dX, dW, dY16, m, k, n, compute16F: true, s);
            CudaGemm.LinearF16PromotedChunks(cublas.Handle, dX, dW, dYp, m, k, n, chunk, dPart, dOnes, s);
            _stream.Synchronize();
            Half[] y32 = DownloadHalf(dY32, m * n), y16 = DownloadHalf(dY16, m * n), yp = DownloadHalf(dYp, m * n);
            double e16 = RelErr(y16, y32), ep = RelErr(yp, y32);
            _out.WriteLine($"K={k}: COMPUTE_16F rel err {e16:E3}; promoted every {chunk}: {ep:E3}");
            Assert.True(e16 > 0, "COMPUTE_16F matched COMPUTE_32F exactly: the toggle did not engage.");
            Assert.True(ep < e16 / 3, $"promotion did not recover accuracy ({ep:E3} vs {e16:E3})");
        }
        finally
        {
            foreach (nint p in new[] { dX, dW, dY32, dY16, dYp, dPart, dOnes }) CudaDriverApi.cuMemFree_v2(p);
        }
    }

    // ── host-side probe math (no GPU) ──

    /// <summary>The Q8_1 round trip errs by at most half a step per element, and keeps an all-zero block exact.</summary>
    [Fact]
    public void QuantizeDequantizeQ8_1_ErrorIsWithinHalfAStep()
    {
        var rng = new Random(7);
        var x = new Half[4 * GemmPrecisionProbe.QuantBlock];
        for (int i = GemmPrecisionProbe.QuantBlock; i < x.Length; i++) x[i] = (Half)(float)(rng.NextDouble() * 20 - 10);
        x[^1] = (Half)300f;   // an outlier dominates its block's scale
        var y = new Half[x.Length];
        GemmPrecisionProbe.QuantizeDequantizeQ8_1(x, y);
        for (int b = 0; b < x.Length; b += GemmPrecisionProbe.QuantBlock)
        {
            float amax = 0;
            for (int i = 0; i < GemmPrecisionProbe.QuantBlock; i++) amax = MathF.Max(amax, MathF.Abs((float)x[b + i]));
            float d = (float)(Half)(amax / 127f);
            for (int i = 0; i < GemmPrecisionProbe.QuantBlock; i++)
            {
                float err = MathF.Abs((float)y[b + i] - (float)x[b + i]);
                Assert.True(err <= d * 0.5f + MathF.Abs((float)x[b + i]) * 2e-3f, $"block {b / 32} elem {i}: err {err} > d/2 ({d / 2})");
            }
        }
        for (int i = 0; i < GemmPrecisionProbe.QuantBlock; i++) Assert.Equal(Half.Zero, y[i]);
    }

    /// <summary>Environment values map to the documented variants; anything else keeps the shipped kernel.</summary>
    [Fact]
    public void FlashAccumulatorMode_ParsesEnvValues()
    {
        Assert.Equal(FlashAccumulatorMode.Qk16, CudaFlashAccumulatorExperiment.Parse("qk16"));
        Assert.Equal(FlashAccumulatorMode.Pv16, CudaFlashAccumulatorExperiment.Parse("PV16"));
        Assert.Equal(FlashAccumulatorMode.Pv16Tile, CudaFlashAccumulatorExperiment.Parse("pv16t"));
        Assert.Equal(FlashAccumulatorMode.Qk16Pv16, CudaFlashAccumulatorExperiment.Parse("qk16pv16"));
        Assert.Equal(FlashAccumulatorMode.F32, CudaFlashAccumulatorExperiment.Parse(null));
        Assert.Equal(FlashAccumulatorMode.F32, CudaFlashAccumulatorExperiment.Parse("1"));
        Assert.Equal("pv16t", CudaFlashAccumulatorExperiment.Suffix(FlashAccumulatorMode.Pv16Tile));
        Assert.Throws<ArgumentOutOfRangeException>(() => CudaFlashAccumulatorExperiment.Suffix(FlashAccumulatorMode.F32));
    }

    // ── helpers ──

    private static double RelErr(Half[] got, Half[] want)
    {
        double sd = 0, sr = 0;
        for (int i = 0; i < got.Length; i++)
        {
            double d = (double)(float)got[i] - (float)want[i];
            sd += d * d;
            sr += (double)(float)want[i] * (float)want[i];
        }
        return Math.Sqrt(sd / sr);
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

    private static ushort[] RandomHalf(Random rng, long n, float magnitude)
    {
        var h = new ushort[n];
        for (long i = 0; i < n; i++)
            h[i] = BitConverter.HalfToUInt16Bits((Half)(float)((rng.NextDouble() * 2 - 1) * magnitude));
        return h;
    }

    /// <summary>Random Q4_0 blocks (F16 scale, any nibble pattern is valid).</summary>
    private static byte[] RandomQ4Blocks(Random rng, long elems)
    {
        long blocks = elems / 32;
        var buf = new byte[blocks * 18];
        for (long b = 0; b < blocks; b++)
        {
            ushort sc = BitConverter.HalfToUInt16Bits((Half)(0.05f + (float)rng.NextDouble() * 0.25f));
            buf[b * 18] = (byte)(sc & 0xFF);
            buf[b * 18 + 1] = (byte)(sc >> 8);
            for (int i = 2; i < 18; i++) buf[b * 18 + i] = (byte)rng.Next(256);
        }
        return buf;
    }

    private static nint Upload<T>(T[] host) where T : unmanaged
    {
        nuint bytes = (nuint)((long)host.Length * sizeof(T));
        CudaDriverApi.cuMemAlloc_v2(out nint d, bytes).ThrowOnError();
        fixed (T* p = host)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, bytes).ThrowOnError();
        return d;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static float[] Download(nint d, long n)
    {
        var h = new float[n];
        fixed (float* p = h)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(n * 4)).ThrowOnError();
        return h;
    }

    private static Half[] DownloadHalf(nint d, long n)
    {
        var h = new Half[n];
        fixed (Half* p = h)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(n * 2)).ThrowOnError();
        return h;
    }
}
