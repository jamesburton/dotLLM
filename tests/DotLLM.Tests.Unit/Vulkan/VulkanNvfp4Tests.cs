using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Architectures;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Variant = DotLLM.Tests.Unit.Models.Architectures.NemotronHNvfp4SsmTests.Variant;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Vulkan NVFP4 (issue #1038): the packed GEMV/GEMM against the CPU dequantizer (itself pinned
/// bit-exact to llama.cpp b11739 in <c>Nvfp4Tests</c>), and the Nemotron-H GGUF path end to end
/// against the CPU model on the F32 <c>scale · dequant(W)</c> twin of the same fixture.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNvfp4Tests : IDisposable
{
    private readonly string _scratch;

    public VulkanNvfp4Tests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-vk-nvfp4-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableTheory]
    [InlineData(1, 64, 1, 1.0f)]       // GEMV, one block
    [InlineData(70, 192, 1, 0.37f)]    // GEMV, 3 blocks, scale
    [InlineData(300, 4096, 1, 1.3e-4f)] // GEMV, Bolt-like K and scale
    [InlineData(70, 192, 5, 2.5f)]     // GEMM, partial M and N tiles
    [InlineData(33, 128, 17, 1.0f)]    // GEMM, N crosses one tile
    [InlineData(64, 4096, 32, 1.3e-4f)] // GEMM, Bolt-like K
    public void MatMul_MatchesCpuDequantReference(int m, int k, int n, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var (w, x, expected) = MakeCase(m, k, n, scale);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulNvfp4F32Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k, n, scale);
        float[] actual = new float[n * m];
        device.Download(bufY, actual);

        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 1e-4f * maxAbs + 1e-6f * MathF.Abs(expected[i]),
                $"m={m} k={k} n={n} [{i}]: expected {expected[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
    }

    /// <summary>
    /// #1060: the NVFP4 to F16 dequant is exact (E2M1 x UE4M3 needs at most 6 mantissa bits, inside F16's normal range), so it must
    /// equal the CPU dequantizer bit for bit, UE4M3 edge bytes included.
    /// </summary>
    [SkippableFact]
    public void DequantF16_IsBitExactVsCpu()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int m = 37, k = 320;
        var (w, _, _) = MakeCase(m, k, n: 1, scale: 1f);
        float[] cpu = Dequant(w, m * k);

        using var device = VulkanDevice.Create();
        using var kernel = Nvfp4DequantF16Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(w.Length);
        using var bufY = device.Allocate((long)m * k * sizeof(ushort));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        kernel.Launch(bufW, bufY, m * k / 64);
        float[] raw = new float[m * k / 2];   // two F16 per float word; Download reads float spans only
        device.Download(bufY, raw);
        ushort[] bits = System.Runtime.InteropServices.MemoryMarshal.Cast<float, ushort>(raw).ToArray();

        for (int i = 0; i < cpu.Length; i++)
        {
            Assert.True((float)(Half)cpu[i] == cpu[i], $"[{i}]: CPU value {cpu[i]} is not exact in F16, so the premise is wrong");
            Assert.True(BitConverter.HalfToUInt16Bits((Half)cpu[i]) == bits[i] || (cpu[i] == 0 && (bits[i] & 0x7FFF) == 0),
                $"[{i}]: cpu {cpu[i]}, vulkan {BitConverter.UInt16BitsToHalf(bits[i])}");
        }
    }

    /// <summary>
    /// #1060: <see cref="IqF16PrefillMatmul"/>'s NVFP4 arm (exact F16 dequant, F16 coopmat GEMM with F32 accumulation, then the
    /// per-tensor scale on the F32 output) against the double-precision reference. The activations round to F16, so the bound is
    /// looser than the F32 GEMM's.
    /// </summary>
    [SkippableTheory]
    [InlineData(70, 192, 32, 0.37f)]     // partial tiles, scale
    [InlineData(64, 4096, 40, 1.3e-4f)]  // Bolt-like K and scale
    [InlineData(33, 128, 64, 1.0f)]      // no scale
    public void F16Prefill_MatchesCpuDequantReference(int m, int k, int n, float scale)
    {
        var (w, x, expected) = MakeCase(m, k, n, scale);
        float[] actual = RunF16Prefill(w, x, m, k, n, scale);

        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(MathF.Abs(actual[i] - expected[i]) <= 4e-3f * maxAbs,
                $"m={m} k={k} n={n} [{i}]: expected {expected[i]}, vulkan {actual[i]} (max |y| {maxAbs})");
    }

    /// <summary>
    /// #1060 regression guard for the #1059 trap (a tiny per-tensor scale applied after an F16 intermediate): at Bolt's scale (~1e-4) the
    /// UNSCALED products are ~1e4 x the real outputs, so here every unscaled dot product lies far beyond F16's 65504 (large sub-block
    /// scales, same-sign weights and activations) while the scaled result is ordinary. Passes only while the GEMM accumulates and writes
    /// F32 and the scale multiplies that F32 output; an F16 accumulator, an F16 output or an F16 copy before the scale is inf/NaN here.
    /// </summary>
    [SkippableFact]
    public void F16Prefill_BoltScale_UnscaledSumsBeyondF16Range_StayFinite()
    {
        const int m = 48, k = 4096, n = IqF16PrefillMatmul.DefaultMinTokens + 8; // at or above the F16-prefill threshold
        const float scale = 1.0e-4f;
        var rng = new Random(1060);
        int blocks = m * (k / 64);
        byte[] w = new byte[blocks * 36];
        for (int b = 0; b < blocks; b++)
        {
            for (int s = 0; s < 4; s++) w[b * 36 + s] = (byte)rng.Next(0x70, 0x7F);   // UE4M3 64..224 (0x7F would decode to 0)
            for (int q = 4; q < 36; q++) w[b * 36 + q] = (byte)(rng.Next(5, 8) | (rng.Next(5, 8) << 4)); // positive E2M1 6, 8, 12
        }
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(0.25 + 0.75 * rng.NextDouble());
        float[] expected = Reference(w, x, m, k, n, scale);

        float minUnscaled = expected.Min(MathF.Abs) / scale;
        Assert.True(minUnscaled > 10 * 65504f, $"premise: the smallest unscaled dot product {minUnscaled:E3} is not far beyond F16's range");

        float[] actual = RunF16Prefill(w, x, m, k, n, scale);
        for (int i = 0; i < actual.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && MathF.Abs(actual[i] - expected[i]) <= 4e-3f * MathF.Abs(expected[i]),
                $"[{i}]: expected {expected[i]}, vulkan {actual[i]}");
    }

    /// <summary>Runs <see cref="IqF16PrefillMatmul"/>'s NVFP4 arm once and checks the path counter.</summary>
    /// <param name="w">Packed NVFP4 weights, <c>[m, k]</c>.</param>
    /// <param name="x">F32 activations, <c>[n, k]</c>.</param>
    /// <param name="m">Output rows.</param>
    /// <param name="k">Input width.</param>
    /// <param name="n">Tokens.</param>
    /// <param name="scale">Per-tensor scale.</param>
    /// <returns>The F32 output, <c>[n, m]</c>.</returns>
    private static float[] RunF16Prefill(byte[] w, float[] x, int m, int k, int n, float scale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "no cooperative-matrix F16 GEMM on this device");
        using var gemm = MatMulF16GemmCoopmatKernel.Create(device, spvDir);
        using var prefill = IqF16PrefillMatmul.TryCreate(device, spvDir, gemm, (long)m * k)
            ?? throw new InvalidOperationException("F16 prefill helper unavailable");
        using var bufW = device.Allocate(w.Length);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        IqF16PrefillMatmul.Counters.Reset();
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            Assert.True(prefill.TryRecord(ctx.CommandBuffer, QuantizationType.NVFP4, bufW, bufX, bufY, m, k, n, scale),
                "the F16 prefill declined an NVFP4 matmul");
            ctx.SubmitAndWait();
        }
        Assert.Equal(1, Interlocked.Read(ref IqF16PrefillMatmul.Counters.Nvfp4));
        float[] actual = new float[n * m];
        device.Download(bufY, actual);
        return actual;
    }

    /// <summary>Random NVFP4 blocks (sub-block scales sprinkled with the UE4M3 edge bytes), activations and the double-precision product.</summary>
    private static (byte[] W, float[] X, float[] Expected) MakeCase(int m, int k, int n, float scale)
    {
        var rng = new Random(m * 131 + k * 7 + n);
        int blocks = m * (k / 64);
        byte[] w = new byte[blocks * 36];
        rng.NextBytes(w);
        byte[] edge = [0x00, 0x7F, 0xFF, 0x01, 0x80, 0x7E];
        for (int b = 0; b < blocks; b++)
            for (int s = 0; s < 4; s++)
                w[b * 36 + s] = rng.Next(8) == 0 ? edge[rng.Next(edge.Length)] : (byte)rng.Next(0x20, 0x50);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        return (w, x, Reference(w, x, m, k, n, scale));
    }

    /// <summary>Double-precision <c>scale * (x . dequant(w)^T)</c>, <c>[n, m]</c>.</summary>
    private static float[] Reference(byte[] w, float[] x, int m, int k, int n, float scale)
    {
        float[] wf = Dequant(w, m * k);
        float[] expected = new float[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[t * k + c];
                expected[t * m + r] = (float)(acc * scale);
            }
        return expected;
    }

    private static unsafe float[] Dequant(byte[] w, int elements)
    {
        float[] wf = new float[elements];
        fixed (byte* pw = w)
            Dequantize.ToFloat32((nint)pw, wf.Length, QuantizationType.NVFP4, wf);
        return wf;
    }

    /// <summary>
    /// The Bolt-shaped fixture (NVFP4 ssm_in/ssm_out with per-tensor scales 0.05..3) through
    /// <see cref="VulkanModelLoader.CreateFromGguf"/>, against the CPU model on the F32 twin whose SSM weights are
    /// <c>scale · dequant(W)</c>. Vulkan uses FP32 activations, so this is tight; a dropped scale, a wrong
    /// nibble order or a widened-and-unscaled upload is off by a multiple of the logits.
    /// </summary>
    /// <param name="f16Prefill">#1060: lower the F16-prefill threshold so the 5-token prompt's NVFP4 projections take the F16
    /// dequant + coopmat GEMM (activations rounded to F16: a looser bound); otherwise the packed NVFP4 GEMM runs.</param>
    [SkippableTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void NemotronH_Nvfp4Ssm_ForwardMatchesCpuF32Twin(bool f16Prefill)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        if (f16Prefill)
        {
            using var probe = VulkanDevice.Create();
            Skip.IfNot(probe.HasCooperativeMatrix, "no cooperative-matrix F16 GEMM on this device");
        }
        using var scope = f16Prefill ? new F16PrefillTestScope() : null;
        IqF16PrefillMatmul.Counters.Reset();

        float[] cpu;
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.F32Reference)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var model = ModelLoader.CreateCpuModelFromGguf(gguf, config);
            using ITensor logits = model.Forward(NemotronHNvfp4SsmTests.Ids, NemotronHNvfp4SsmTests.Pos, deviceId: -1);
            cpu = LastRow(logits);
        }

        float[] vk;
        using (var gguf = GgufFile.Open(NemotronHNvfp4SsmTests.WriteFixture(_scratch, Variant.Nvfp4Scaled)))
        {
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            using var device = VulkanDevice.Create();
            var (model, _) = VulkanModelLoader.CreateFromGguf(device, gguf, config, spvDir);
            using (model)
            {
                Assert.IsType<VulkanNemotronHTransformerModel>(model);
                using ITensor logits = model.Forward(NemotronHNvfp4SsmTests.Ids, NemotronHNvfp4SsmTests.Pos, deviceId: -1);
                vk = LastRow(logits);
            }
        }

        long viaF16 = Interlocked.Read(ref IqF16PrefillMatmul.Counters.Nvfp4);
        if (f16Prefill) Assert.True(viaF16 > 0, "the NVFP4 projections did not take the F16 prefill path");
        else Assert.Equal(0, viaF16);

        float maxAbs = cpu.Max(MathF.Abs);
        float tol = f16Prefill ? 1e-2f : 2e-3f;
        Assert.True(maxAbs > 0.5f, $"degenerate fixture: max |logit| {maxAbs}");
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - vk[i]) <= tol * maxAbs,
                $"logit {i}: cpu(F32 twin) {cpu[i]}, vulkan(NVFP4) {vk[i]} (max |logit| {maxAbs})");
    }

    [Fact]
    public void ScaleOnNonNvfp4Projection_IsRefused()
    {
        VulkanNemotronHWeights.ThrowIfScaleUnsupported("ssm_in", 1.0f, QuantizationType.F32);
        VulkanNemotronHWeights.ThrowIfScaleUnsupported("ssm_in", 0.5f, QuantizationType.NVFP4);
        Assert.Throws<NotSupportedException>(() =>
            VulkanNemotronHWeights.ThrowIfScaleUnsupported("ssm_in", 0.5f, QuantizationType.F32));
    }

    private static unsafe float[] LastRow(ITensor logits)
    {
        int vocab = NemotronHNvfp4SsmTests.Vocab;
        int rows = (int)(logits.Shape.ElementCount / vocab);
        return new ReadOnlySpan<float>((float*)logits.DataPointer + (rows - 1) * vocab, vocab).ToArray();
    }
}
