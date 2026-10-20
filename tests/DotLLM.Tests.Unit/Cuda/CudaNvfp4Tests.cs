using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using Xunit;
using Variant = DotLLM.Tests.Unit.Models.Architectures.NemotronHNvfp4SsmTests.Variant;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CUDA NVFP4 (issue #1038): the dequant (F16/F32), decode GEMV and in-place scale kernels against the
/// CPU dequantizer (pinned bit-exact to llama.cpp b11739 in <c>Nvfp4Tests</c>), and the Nemotron-H GGUF
/// path end to end — prefill (dequant + cuBLAS + scale) and decode (GEMV with scale) — against the CPU
/// model on the F32 <c>scale · dequant(W)</c> twin of the same fixture.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed unsafe class CudaNvfp4Tests : IDisposable
{
    private readonly string _scratch;

    public CudaNvfp4Tests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-nvfp4-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(5)]       // more blocks than one 256-thread tile (4 blocks)
    [InlineData(1037)]    // many tiles, grid-stride loop
    public void Dequant_F32AndF16_MatchCpu(int blocks)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        byte[] w = RandomBlocks(new Random(blocks), blocks);
        float[] cpu = CpuDequant(w);

        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        Assert.True(kernels.HasNvfp4, "nvfp4.ptx not loaded");

        nint dSrc = Upload(w), dF32 = Alloc(cpu.Length * 4L), dF16 = Alloc(cpu.Length * 2L);
        try
        {
            kernels.LaunchDequantToF32(dSrc, QuantizationType.NVFP4, dF32, cpu.Length, stream.Handle);
            kernels.LaunchDequantToF16(dSrc, QuantizationType.NVFP4, dF16, cpu.Length, stream.Handle);
            stream.Synchronize();

            float[] f32 = DownloadF32(dF32, cpu.Length);
            ushort[] f16 = new ushort[cpu.Length];
            fixed (ushort* p = f16)
                CudaDriverApi.cuMemcpyDtoH_v2((nint)p, dF16, (nuint)(f16.Length * 2)).ThrowOnError();

            for (int i = 0; i < cpu.Length; i++)
            {
                Assert.True(BitConverter.SingleToUInt32Bits(cpu[i]) == BitConverter.SingleToUInt32Bits(f32[i]),
                    $"F32 element {i}: cpu {cpu[i]}, cuda {f32[i]}");
                Assert.Equal((float)(Half)cpu[i], (float)BitConverter.UInt16BitsToHalf(f16[i]));
            }
        }
        finally
        {
            Free(dSrc); Free(dF32); Free(dF16);
        }
    }

    [SkippableTheory]
    [InlineData(1, 64, 1.0f)]
    [InlineData(70, 192, 0.37f)]
    [InlineData(300, 4096, 1.3e-4f)]   // Bolt-like K and scale
    [InlineData(64, 8192, 2.5f)]       // more blocks per row than threads (128)
    public void Gemv_MatchesCpuDequantReference_WithScale(int m, int k, float scale)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var rng = new Random(m * 17 + k);
        byte[] w = RandomBlocks(rng, m * (k / 64));
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] wf = CpuDequant(w);
        float[] expected = new float[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[c];
            expected[r] = (float)(acc * scale);
        }

        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        nint dW = Upload(w), dX = UploadF32(x), dY = Alloc(m * 4L);
        try
        {
            kernels.LaunchNvfp4GemvF32(dW, dX, dY, m, k, scale, stream.Handle);
            stream.Synchronize();
            float[] actual = DownloadF32(dY, m);
            float maxAbs = expected.Max(MathF.Abs);
            for (int i = 0; i < m; i++)
                Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-5f * maxAbs + 1e-6f * MathF.Abs(expected[i]),
                    $"m={m} k={k} row {i}: expected {expected[i]}, cuda {actual[i]}");
        }
        finally
        {
            Free(dW); Free(dX); Free(dY);
        }
    }

    /// <summary>
    /// The dp4a GEMV (#1061) on activations quantized by the house F32 → int8 quantizer, against the CPU's
    /// NVFP4 × Q8_0 GEMM (<see cref="MatMul.GemmNvfp4"/>, llama.cpp's <c>ggml_vec_dot_nvfp4_q8_0</c>) times
    /// the scale. The activation quantization is byte-exact and each sub-block's int32 sum is exact, so only
    /// FP32 summation order differs: the bar is the FP32 GEMV's 1e-5 · max|y|. Covers K tails (fewer units
    /// than lanes; a partial last warp step), row tails (odd m: half a warp's row pair, a partial CUDA block),
    /// all-zero activation blocks and the UE4M3 edge bytes.
    /// </summary>
    [SkippableTheory]
    [InlineData(1, 64, 1.0f)]          // one block: 2 units, 30 idle lanes
    [InlineData(70, 192, 0.37f)]       // 6 units
    [InlineData(301, 1088, 0.8f)]      // 34 units: one full warp step + 2; 301 rows: 37 full blocks of 8 + 5
    [InlineData(300, 4096, 1.3e-4f)]   // Bolt ssm_in K and scale
    [InlineData(67, 8192, 2.5f)]       // Bolt ssm_out K
    public void Dp4aGemv_MatchesCpuNvfp4Q8_0_WithScale(int m, int k, float scale)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var rng = new Random(m * 31 + k);
        byte[] w = RandomBlocks(rng, m * (k / 64));
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (i % 97 == 0 ? 40f : 1f);
        if (k >= 128) Array.Clear(x, 64, 32);   // an all-zero activation block (scale 0)
        float[] expected = CpuGemmNvfp4Q8_0(w, x, m, k, scale);

        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        Assert.True(kernels.HasNvfp4Dp4a, "NVFP4 dp4a GEMV or the pq2_0 activation quantizer not loaded");
        nint dW = Upload(w), dX = UploadF32(x), dQ = Alloc(k), dMeta = Alloc(k / 32 * 8L), dY = Alloc(m * 4L);
        try
        {
            kernels.LaunchPQ2_0Dp4aQuantizeX(dX, dQ, dMeta, k, stream.Handle);
            CudaDriverApi.cuMemsetD8_v2(dY, 0xFF, (nuint)(m * 4L)).ThrowOnError();   // NaN: every row must be written
            kernels.LaunchNvfp4GemvQ8(dW, dQ, dMeta, dY, m, k, scale, stream.Handle);
            stream.Synchronize();
            float[] actual = DownloadF32(dY, m);
            float maxAbs = expected.Max(MathF.Abs);
            Assert.True(maxAbs > 0, "degenerate fixture");
            for (int i = 0; i < m; i++)
                Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-5f * maxAbs + 1e-6f * MathF.Abs(expected[i]),
                    $"m={m} k={k} row {i}: cpu {expected[i]}, cuda {actual[i]}");
        }
        finally
        {
            Free(dW); Free(dX); Free(dQ); Free(dMeta); Free(dY);
        }
    }

    [SkippableFact]
    public void ScaleInplace_MultipliesEveryElement()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        float[] v = Enumerable.Range(0, 70_001).Select(i => i * 0.25f - 3f).ToArray();
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        nint d = UploadF32(v);
        try
        {
            kernels.LaunchScaleF32Inplace(d, v.Length, 1.3e-4f, stream.Handle);
            stream.Synchronize();
            float[] got = DownloadF32(d, v.Length);
            for (int i = 0; i < v.Length; i++)
                Assert.Equal(v[i] * 1.3e-4f, got[i]);
        }
        finally
        {
            Free(d);
        }
    }

    /// <summary>
    /// The Bolt-shaped NVFP4 fixture (two Mamba-2 layers, NVFP4 <c>ssm_in</c>/<c>ssm_out</c>, per-tensor
    /// scales 0.05..3) through <see cref="CudaModelLoader.CreateFromGguf"/>: a 5-token prefill (dequant +
    /// cuBLAS + scale kernel) and then one decode step (NVFP4 GEMV with the scale), each against the CPU
    /// model on the F32 twin. A dropped scale on either path is off by a multiple of the logits. On the dp4a
    /// GEMV (#1061) the decode step carries the Q8_0 activation rounding the CPU NVFP4 model itself carries
    /// against this twin, so it gets that model's bar (see <see cref="Dp4aDecodeVsF32TwinTolerance"/>);
    /// <see cref="NemotronH_Nvfp4Ssm_TokenByTokenDecodeMatchesCpuNvfp4"/> is the tight check of that path.
    /// </summary>
    [SkippableTheory]
    [InlineData(Variant.Nvfp4Scaled, true)]   // decode on the dp4a GEMV (#1061)
    [InlineData(Variant.Nvfp4Scaled, false)]  // decode on #1038's FP32-activation GEMV
    [InlineData(Variant.F32Reference, true)]  // control: the same model with F32 SSM weights on CUDA
    [InlineData(Variant.Nvfp4ScaledBoltMagnitude, true)]  // unscaled prefill GEMM output beyond FP16 range, as in Bolt
    [InlineData(Variant.Nvfp4ScaledBoltMagnitude, false)]
    public void NemotronH_Nvfp4Ssm_PrefillAndDecodeMatchCpuF32Twin(Variant cudaVariant, bool dp4a)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        if (dp4a)
        {
            using var probeContext = CudaContext.Create(0);
            using var probe = new CudaKernels(ResolvePtxDir());
            Assert.True(probe.HasNvfp4Dp4a, "dp4a decode requested but the NVFP4 dp4a kernels are not loaded");
        }

        float decodeTolerance = dp4a && cudaVariant != Variant.F32Reference ? Dp4aDecodeVsF32TwinTolerance : 1e-2f;
        WithNvfp4Dp4a(dp4a, () => RunNemotronHNvfp4Fixture(cudaVariant, decodeTolerance));
    }

    /// <summary>
    /// Bar for a dp4a decode step against the F32 twin: the 0.06 · max|logit| that
    /// <c>NemotronHNvfp4SsmTests</c> holds the CPU NVFP4 model to (measured 0.035 there, all of it Q8_0
    /// activation rounding). Measured on CUDA: 0.026.
    /// </summary>
    private const float Dp4aDecodeVsF32TwinTolerance = 0.06f;

    /// <summary>
    /// The dp4a decode path end to end against the CPU model on the SAME NVFP4 weights (#1061): every token
    /// of the fixture is fed alone, so every SSM projection on both sides is an NVFP4 × Q8_0 GEMV with the
    /// same activation rounding, and the logits of every step must agree far more tightly than either does with
    /// the F32 twin. #1038's FP32-activation GEMV misses this bar by the Q8_0 rounding (see the PR).
    /// </summary>
    [SkippableFact]
    public void NemotronH_Nvfp4Ssm_TokenByTokenDecodeMatchesCpuNvfp4()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        int[] ids = NemotronHNvfp4SsmTests.Ids;
        int[] pos = NemotronHNvfp4SsmTests.Pos;
        int vocab = NemotronHNvfp4SsmTests.Vocab;
        string fixture = NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.Nvfp4Scaled);

        var cpu = new List<float[]>();
        using (var gguf = GgufFile.Open(fixture))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
            for (int t = 0; t < ids.Length; t++)
                using (ITensor l = model.Forward(ids[t..(t + 1)], pos[t..(t + 1)], deviceId: -1)) cpu.Add(LastRow(l, vocab));
        }

        using (var probeContext = CudaContext.Create(0))
        using (var probe = new CudaKernels(ResolvePtxDir()))
            Assert.True(probe.HasNvfp4Dp4a, "NVFP4 dp4a kernels not loaded");

        var cuda = new List<float[]>();
        WithNvfp4Dp4a(true, () =>
        {
            using var gguf = GgufFile.Open(fixture);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir: ResolvePtxDir());
            using (model)
            {
                using var kv = kvFactory(16);
                for (int t = 0; t < ids.Length; t++)
                    using (ITensor l = model.Forward(ids[t..(t + 1)], pos[t..(t + 1)], deviceId: -1, kv)) cuda.Add(LastRow(l, vocab));
            }
        });

        for (int t = 0; t < ids.Length; t++)
            AssertClose(cpu[t], cuda[t], $"token {t} (CPU NVFP4 x Q8_0 vs CUDA dp4a)", TokenByTokenTolerance);
    }

    /// <summary>
    /// Bar for <see cref="NemotronH_Nvfp4Ssm_TokenByTokenDecodeMatchesCpuNvfp4"/>: both sides round the
    /// activations identically, so what is left is FP32 summation order through two layers.
    /// </summary>
    private const float TokenByTokenTolerance = 1e-4f;

    /// <summary>
    /// Runs <paramref name="body"/> with every NVFP4 decode GEMV forced onto the dp4a kernel or onto the FP32 one
    /// (the fixture's weights are far below <see cref="CudaKernels.Nvfp4Dp4aMinWeightBytes"/>, so
    /// <see cref="Nvfp4DecodeGemv.Auto"/> would never pick dp4a), then restores the mode.
    /// </summary>
    private static void WithNvfp4Dp4a(bool enabled, Action body)
    {
        Nvfp4DecodeGemv was = CudaKernels.Nvfp4DecodeGemvMode;
        CudaKernels.Nvfp4DecodeGemvMode = enabled ? Nvfp4DecodeGemv.Dp4a : Nvfp4DecodeGemv.F32;
        try
        {
            body();
        }
        finally
        {
            CudaKernels.Nvfp4DecodeGemvMode = was;
        }
    }

    private void RunNemotronHNvfp4Fixture(Variant cudaVariant, float decodeTolerance)
    {
        int[] ids = NemotronHNvfp4SsmTests.Ids;
        int[] pos = NemotronHNvfp4SsmTests.Pos;
        int vocab = NemotronHNvfp4SsmTests.Vocab;

        float[] cpuPrefill, cpuDecode;
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.F32Reference)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            // Same prefill(4) + decode(1) sequence as the CUDA side. The fixture has no attention layers,
            // and the model's recurrent SSM state carries from one call to the next, so the second call
            // is a true decode step (two independent full forwards would need a ResetSequenceState()).
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using (ITensor a = model.Forward(ids[..4], pos[..4], deviceId: -1)) cpuPrefill = LastRow(a, vocab);
            using (ITensor b = model.Forward(ids[4..], pos[4..], deviceId: -1)) cpuDecode = LastRow(b, vocab);
        }

        float[] cudaPrefill, cudaDecode;
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, cudaVariant)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir: ResolvePtxDir());
            using (model)
            {
                using var kv = kvFactory(16);
                using (ITensor a = model.Forward(ids[..4], pos[..4], deviceId: -1, kv)) cudaPrefill = LastRow(a, vocab);
                using (ITensor b = model.Forward(ids[4..], pos[4..], deviceId: -1, kv)) cudaDecode = LastRow(b, vocab);
            }
        }

        AssertClose(cpuPrefill, cudaPrefill, "prefill (dequant F16 + cuBLAS + scale)", 1e-2f);
        AssertClose(cpuDecode, cudaDecode, "decode (NVFP4 GEMV with scale)", decodeTolerance);
    }

    /// <summary>
    /// Decode GEMV bandwidth (#1061): #1038's FP32-activation GEMV against the dp4a GEMV and its per-token
    /// activation quantize, on Bolt's SSM shapes and a sweep, and which one <see cref="Nvfp4DecodeGemv.Auto"/>
    /// picks. Times are a host loop of back-to-back launches (best of 3 x 300): GPU-bound from a few MB of
    /// weights up, launch-bound below (T5500 floors at ~16 us per launch; per-launch CUDA-event timing there
    /// carries a ~20 us WDDM overhead, so it is not used). GB/s counts NVFP4 weight bytes only; shapes under
    /// ~3 MB of weights sit in the RTX 3060's L2. Prints numbers only.
    /// </summary>
    [SkippableTheory]
    [Trait("Category", "Benchmark")]
    [InlineData(18560, 4096)]   // Bolt ssm_in
    [InlineData(4096, 8192)]    // Bolt ssm_out
    [InlineData(1024, 1024)]
    [InlineData(4096, 1024)]
    [InlineData(1024, 4096)]
    [InlineData(2048, 4096)]
    [InlineData(4096, 4096)]
    [InlineData(8192, 4096)]
    [InlineData(2048, 8192)]
    [InlineData(18560, 8192)]
    [InlineData(4096, 18560)]
    [InlineData(32768, 4096)]
    public void Benchmark_Dp4aGemvSweep(int m, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        var rng = new Random(2);
        byte[] w = RandomBlocks(rng, m * (k / 64));
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        Assert.True(kernels.HasNvfp4Dp4a, "NVFP4 dp4a kernels not loaded");
        nint dW = Upload(w), dX = UploadF32(x), dQ = Alloc(k), dMeta = Alloc(k / 32 * 8L), dY = Alloc(m * 4L);
        try
        {
            double Loop(Action a)
            {
                const int Iters = 300;
                for (int i = 0; i < 10; i++) a();
                stream.Synchronize();
                double best = double.MaxValue;
                for (int rep = 0; rep < 3; rep++)
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < Iters; i++) a();
                    stream.Synchronize();
                    best = Math.Min(best, sw.Elapsed.TotalMilliseconds * 1000 / Iters);
                }
                return best;   // microseconds
            }
            string Gbs(double us) => $"{w.Length / us / 1e3,5:F0}";
            void Quant() => kernels.LaunchPQ2_0Dp4aQuantizeX(dX, dQ, dMeta, k, stream.Handle);
            void Dp4a() => kernels.LaunchNvfp4GemvQ8(dW, dQ, dMeta, dY, m, k, 1.3e-4f, stream.Handle);

            double f32 = Loop(() => kernels.LaunchNvfp4GemvF32(dW, dX, dY, m, k, 1.3e-4f, stream.Handle));
            double quant = Loop(Quant), gemv = Loop(Dp4a), both = Loop(() => { Quant(); Dp4a(); });
            bool auto;
            Nvfp4DecodeGemv was = CudaKernels.Nvfp4DecodeGemvMode;
            CudaKernels.Nvfp4DecodeGemvMode = Nvfp4DecodeGemv.Auto;
            try { auto = kernels.UseNvfp4Dp4a(m, k); }
            finally { CudaKernels.Nvfp4DecodeGemvMode = was; }
            Console.WriteLine(
                $"NVFP4 {m,5}x{k,5} ({w.Length / 1e6,4:F1} MB): f32 {f32,6:F1} us {Gbs(f32)} GB/s | "
                + $"quant {quant,4:F1} us | dp4a {gemv,6:F1} us {Gbs(gemv)} GB/s | quant+dp4a {both,6:F1} us {Gbs(both)} GB/s | "
                + $"x{f32 / both,4:F2} | auto: {(auto ? "dp4a" : "f32")}");
        }
        finally
        {
            Free(dW); Free(dX); Free(dQ); Free(dMeta); Free(dY);
        }
    }

    [SkippableTheory]
    [Trait("Category", "Benchmark")]
    [InlineData(18560, 4096)]
    [InlineData(4096, 8192)]
    public void Benchmark_BoltSsmShapes(int m, int k)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
        byte[] w = RandomBlocks(new Random(1), m * (k / 64));
        using var context = CudaContext.Create(0);
        using var stream = CudaStream.Create();
        using var kernels = new CudaKernels(ResolvePtxDir());
        nint dW = Upload(w), dX = UploadF32(new float[k]), dY = Alloc(m * 4L), dF16 = Alloc((long)m * k * 2);
        try
        {
            const int Iters = 200;
            double Time(Action a)
            {
                a(); stream.Synchronize();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < Iters; i++) a();
                stream.Synchronize();
                return sw.Elapsed.TotalMilliseconds / Iters;
            }
            double gemv = Time(() => kernels.LaunchNvfp4GemvF32(dW, dX, dY, m, k, 1.3e-4f, stream.Handle));
            double deq = Time(() => kernels.LaunchDequantToF16(dW, QuantizationType.NVFP4, dF16, m * k, stream.Handle));
            Console.WriteLine($"NVFP4 {m}x{k}: GEMV {gemv * 1000:F1} us ({w.Length / gemv / 1e6:F1} GB/s of weights); "
                + $"dequant->F16 {deq * 1000:F1} us ({(w.Length + 2.0 * m * k) / deq / 1e6:F1} GB/s read+write)");
        }
        finally
        {
            Free(dW); Free(dX); Free(dY); Free(dF16);
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    /// <summary>Asserts every logit is within <paramref name="tolerance"/> · max|cpu logit| (the prefill path's F16 bar is 1e-2).</summary>
    private static void AssertClose(float[] cpu, float[] cuda, string what, float tolerance)
    {
        float maxAbs = cpu.Max(MathF.Abs);
        Assert.True(maxAbs > 0.5f, $"degenerate fixture: max |logit| {maxAbs}");
        float maxDiff = cpu.Zip(cuda, (a, b) => MathF.Abs(a - b)).Max();
        Console.WriteLine($"{what}: max |diff| {maxDiff} = {maxDiff / maxAbs:E2} x max|logit| (bar {tolerance:E1})");
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - cuda[i]) <= tolerance * maxAbs,
                $"{what} logit {i}: cpu {cpu[i]}, cuda {cuda[i]} (max |logit| {maxAbs}, bar {tolerance} x)");
    }

    private static float[] LastRow(ITensor logits, int vocab)
    {
        int rows = (int)(logits.Shape.ElementCount / vocab);
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (rows - 1) * vocab, vocab).ToArray();
    }

    private static byte[] RandomBlocks(Random rng, int blocks)
    {
        byte[] w = new byte[blocks * 36];
        rng.NextBytes(w);
        byte[] edge = [0x00, 0x7F, 0xFF, 0x01, 0x80, 0x7E];
        for (int b = 0; b < blocks; b++)
            for (int s = 0; s < 4; s++)
                w[b * 36 + s] = rng.Next(8) == 0 ? edge[rng.Next(edge.Length)] : (byte)rng.Next(0x20, 0x50);
        return w;
    }

    /// <summary>The CPU NVFP4 × Q8_0 GEMV (<see cref="MatMul.GemmNvfp4"/>) for one token, times <paramref name="scale"/>.</summary>
    private static float[] CpuGemmNvfp4Q8_0(byte[] w, float[] x, int m, int k, float scale)
    {
        float[] y = new float[m];
        fixed (byte* pw = w)
        fixed (float* px = x, py = y)
            MatMul.GemmNvfp4(pw, px, py, m, k, 1, pool: null);
        for (int i = 0; i < m; i++) y[i] *= scale;
        return y;
    }

    private static float[] CpuDequant(byte[] w)
    {
        float[] y = new float[w.Length / 36 * 64];
        fixed (byte* p = w)
            Dequantize.ToFloat32((nint)p, y.Length, QuantizationType.NVFP4, y);
        return y;
    }

    private static nint Alloc(long bytes)
    {
        CudaDriverApi.cuMemAlloc_v2(out nint d, (nuint)bytes).ThrowOnError();
        return d;
    }

    private static nint Upload(byte[] data)
    {
        nint d = Alloc(data.Length);
        fixed (byte* p = data)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)data.Length).ThrowOnError();
        return d;
    }

    private static nint UploadF32(float[] data)
    {
        nint d = Alloc(data.Length * 4L);
        fixed (float* p = data)
            CudaDriverApi.cuMemcpyHtoD_v2(d, (nint)p, (nuint)(data.Length * 4L)).ThrowOnError();
        return d;
    }

    private static float[] DownloadF32(nint d, int n)
    {
        float[] r = new float[n];
        fixed (float* p = r)
            CudaDriverApi.cuMemcpyDtoH_v2((nint)p, d, (nuint)(n * 4L)).ThrowOnError();
        return r;
    }

    private static void Free(nint d)
    {
        if (d != 0) CudaDriverApi.cuMemFree_v2(d);
    }

    private static string ResolvePtxDir()
    {
        string? envDir = Environment.GetEnvironmentVariable("DOTLLM_PTX_DIR");
        if (envDir is not null && Directory.Exists(envDir))
            return envDir;
        string dir = Path.GetDirectoryName(typeof(CudaNvfp4Tests).Assembly.Location)!;
        while (dir.Length > 3 && !File.Exists(Path.Combine(dir, "dotLLM.slnx")))
            dir = Path.GetDirectoryName(dir)!;
        return Path.Combine(dir, "native", "ptx");
    }
}
