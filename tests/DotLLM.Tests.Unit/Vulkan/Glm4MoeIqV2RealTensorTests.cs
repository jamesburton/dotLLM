using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1678: the V2 indexed IQ4_XS / IQ4_NL MMVQ kernels against the V1 kernels and the CPU dequant oracle on REAL expert tensors (the first
/// eight experts of the glm4moe REAP-82B <c>blk.1</c> gate and down banks; the file's only IQ bank formats). Opt-in via <c>DOTLLM_GLM4MOE_GGUF</c>;
/// reads ~50 MB of the mmap, no model load. The random-byte kernel tests remain the layout oracle; this one proves the kernels on the
/// value distribution the model actually has (fitted scales, sparse codebook use), where a reordering bug would hide behind random bytes.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Glm4MoeIqV2RealTensorTests
{
    private readonly ITestOutputHelper _out;
    public Glm4MoeIqV2RealTensorTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData("blk.1.ffn_gate_exps.weight", QuantizationType.IQ4_XS)]   // M 1408, K 4096 (one 16-block chunk per lane step)
    [InlineData("blk.1.ffn_down_exps.weight", QuantizationType.IQ4_NL)]   // M 4096, K 1408 (44 blocks: partial U=2 chunk)
    public void RealExpertBank_V2_MatchesV1AndOracle(string tensor, QuantizationType qt)
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_GLM4MOE_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_GLM4MOE_GGUF not set");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "Device does not advertise VK_KHR_shader_integer_dot_product.");

        using var gguf = GgufFile.Open(path!);
        var desc = gguf.TensorsByName[tensor];
        Assert.Equal(qt, desc.QuantizationType);
        int k = (int)desc.Shape[0], m = (int)desc.Shape[1], fileExperts = (int)desc.Shape[2];
        const int experts = 8, n = 16;
        var iq = MoeIndexedMatmulIqMmvqKernel.FromQuantizationType(qt)!.Value;
        var (blockBytes, group) = MoeIndexedMatmulIqMmvqKernel.Describe(iq);
        long rowBytes = (long)(k / group) * blockBytes;
        long bankBytes = experts * m * rowBytes;
        Assert.True(experts <= fileExperts);
        var bank = new byte[bankBytes];
        unsafe { new ReadOnlySpan<byte>((void*)gguf.TensorDataPointer(tensor), (int)bankBytes).CopyTo(bank); }

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var v1 = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, iq, v2: false) ?? throw new Xunit.Sdk.XunitException("V1 spv missing.");
        using var v2 = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, iq, v2: true) ?? throw new Xunit.Sdk.XunitException("V2 spv missing.");
        Assert.False(v1.IsV2);
        Assert.True(v2.IsV2);

        var rng = new Random(0x1678 + (int)qt);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        int[] indices = MoeMmvqParitySupport.RandomIndices(rng, n, experts, experts);

        float[] y1 = VulkanMoeIndexedMatmulIqMmvqKernelTests.Launch(device, quant, v1, bank, x, indices, m, k, n, experts, out sbyte[] xq, out float[] xds);
        float[] y2 = VulkanMoeIndexedMatmulIqMmvqKernelTests.Launch(device, quant, v2, bank, x, indices, m, k, n, experts, out _, out _);
        float[] oracle = VulkanMoeIndexedMatmulIqMmvqKernelTests.CpuOracle(bank, qt, xq, xds, indices, m, k, n, experts, rowBytes);

        double ss = 0, d12 = 0, maxRel12 = 0;
        for (int i = 0; i < y1.Length; i++)
        {
            ss += (double)oracle[i] * oracle[i];
            double d = Math.Abs(y1[i] - y2[i]);
            d12 = Math.Max(d12, d);
            maxRel12 = Math.Max(maxRel12, d / (Math.Abs(oracle[i]) + 1e-3));
        }
        double rms = Math.Sqrt(ss / y1.Length);
        _out.WriteLine($"{tensor} {qt} M={m} K={k} E={experts} n={n}: output rms {rms:E3}; V1 vs V2 max |diff| {d12:E3} ({d12 / rms:E2} of rms), max rel {maxRel12:E2}");
        MoeMmvqParitySupport.AssertSameTierParity(oracle, y1, m, n, qt + " V1 real bank");
        MoeMmvqParitySupport.AssertSameTierParity(oracle, y2, m, n, qt + " V2 real bank");
        // Reordering-only drift between the two kernels: well under 1e-4 of the output scale (an index / scale / sign bug is 1e-1..1).
        Assert.True(d12 <= 2e-4 * rms, $"V1 vs V2 differ by {d12} (rms {rms})");
    }
}
