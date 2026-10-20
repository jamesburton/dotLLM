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
    /// model on the F32 twin. A dropped scale on either path is off by a multiple of the logits.
    /// </summary>
    [SkippableTheory]
    [InlineData(Variant.Nvfp4Scaled)]
    [InlineData(Variant.F32Reference)] // control: the same model with F32 SSM weights on CUDA
    [InlineData(Variant.Nvfp4ScaledBoltMagnitude)] // unscaled prefill GEMM output beyond FP16 range, as in Bolt
    public void NemotronH_Nvfp4Ssm_PrefillAndDecodeMatchCpuF32Twin(Variant cudaVariant)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available");
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

        AssertClose(cpuPrefill, cudaPrefill, "prefill (dequant F16 + cuBLAS + scale)");
        AssertClose(cpuDecode, cudaDecode, "decode (NVFP4 GEMV with scale)");
    }

    /// <summary>
    /// Bandwidth of the NVFP4 decode GEMV and the prefill dequant on Bolt's SSM shapes (ssm_in 4096 -> 18560,
    /// ssm_out 8192 -> 4096). Prints numbers only; nothing is asserted beyond the kernels running.
    /// </summary>
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

    private static void AssertClose(float[] cpu, float[] cuda, string what)
    {
        float maxAbs = cpu.Max(MathF.Abs);
        Assert.True(maxAbs > 0.5f, $"degenerate fixture: max |logit| {maxAbs}");
        // The prefill path runs F16 weights/activations through cuBLAS, so the bar is F16-sized.
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - cuda[i]) <= 1e-2f * maxAbs,
                $"{what} logit {i}: cpu(F32 twin) {cpu[i]}, cuda(NVFP4) {cuda[i]} (max |logit| {maxAbs})");
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
