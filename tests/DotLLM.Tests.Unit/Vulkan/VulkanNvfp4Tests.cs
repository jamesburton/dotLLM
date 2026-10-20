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

        float[] wf = new float[m * k];
        unsafe
        {
            fixed (byte* pw = w)
                Dequantize.ToFloat32((nint)pw, wf.Length, QuantizationType.NVFP4, wf);
        }

        float[] expected = new float[n * m];
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[r * k + c] * x[t * k + c];
                expected[t * m + r] = (float)(acc * scale);
            }

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
    /// The Bolt-shaped fixture (NVFP4 ssm_in/ssm_out with per-tensor scales 0.05..3) through
    /// <see cref="VulkanModelLoader.CreateFromGguf"/>, against the CPU model on the F32 twin whose SSM weights are
    /// <c>scale · dequant(W)</c>. Vulkan uses FP32 activations, so this is tight; a dropped scale, a wrong
    /// nibble order or a widened-and-unscaled upload is off by a multiple of the logits.
    /// </summary>
    [SkippableFact]
    public void NemotronH_Nvfp4Ssm_ForwardMatchesCpuF32Twin()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

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

        float maxAbs = cpu.Max(MathF.Abs);
        Assert.True(maxAbs > 0.5f, $"degenerate fixture: max |logit| {maxAbs}");
        for (int i = 0; i < cpu.Length; i++)
            Assert.True(MathF.Abs(cpu[i] - vk[i]) <= 2e-3f * maxAbs,
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
