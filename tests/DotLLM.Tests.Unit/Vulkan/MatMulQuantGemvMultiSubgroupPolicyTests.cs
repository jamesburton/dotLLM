using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Hardware-independent guard for #1104: a multi-column GEMV shader that reduces a 32-lane workgroup with a single <c>subgroupAdd</c> (no shared
/// memory between subgroups) is only correct when the workgroup is exactly one subgroup. Intel Arc compiles such a workgroup as several 8/16-lane
/// subgroups, each storing its partial sum, so the IQ4_XS multi GEMV returned about a quarter of the dot product there and every 2..8-row qwen4exp
/// forward (MTP verify) was garbage. Such pipelines must be pinned to wave32 or refused. These tests read the shader sources, so they discriminate
/// on any machine (the GPU parity test only fails on a device whose driver picks a narrow subgroup).
/// </summary>
public class MatMulQuantGemvMultiSubgroupPolicyTests
{
    private static string ShaderDir()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "native", "vulkan", "shaders");
            if (Directory.Exists(candidate)) return candidate;
        }

        throw new DirectoryNotFoundException("native/vulkan/shaders not found above the test binaries.");
    }

    private static string ShaderSource(QuantGemvMultiFormat format) => File.ReadAllText(Path.Combine(ShaderDir(), format switch
    {
        QuantGemvMultiFormat.Q4_K => "matmul_q4_k_gemv_f32_multi.comp",
        QuantGemvMultiFormat.Q5_K => "matmul_q5_k_gemv_f32_multi.comp",
        QuantGemvMultiFormat.Q6_K => "matmul_q6_k_gemv_f32_multi.comp",
        QuantGemvMultiFormat.Q6_K_R2 => "matmul_q6_k_gemv_f32_multi_r2.comp",
        QuantGemvMultiFormat.IQ4_XS => "matmul_iq4_xs_f32_gemv_multi.comp",
        QuantGemvMultiFormat.IQ3_S => "matmul_iq3_s_f32_gemv_multi.comp",
        _ => throw new ArgumentOutOfRangeException(nameof(format)),
    }));

    /// <summary>True when the shader's only cross-lane reduction is subgroup arithmetic (no workgroup shared memory, no subgroup-id bookkeeping).</summary>
    private static bool ReducesWithOneSubgroupOnly(string src)
        => src.Contains("subgroupAdd", StringComparison.Ordinal)
           && !src.Contains("\nshared ", StringComparison.Ordinal)
           && !src.Contains("gl_SubgroupID", StringComparison.Ordinal)
           && !src.Contains("gl_NumSubgroups", StringComparison.Ordinal);

    [Theory]
    [InlineData(QuantGemvMultiFormat.Q4_K)]
    [InlineData(QuantGemvMultiFormat.Q5_K)]
    [InlineData(QuantGemvMultiFormat.Q6_K)]
    [InlineData(QuantGemvMultiFormat.Q6_K_R2)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS)]
    [InlineData(QuantGemvMultiFormat.IQ3_S)]
    public void SingleSubgroupShaders_ArePinnedToWave32(QuantGemvMultiFormat format)
    {
        bool singleSubgroup = ReducesWithOneSubgroupOnly(ShaderSource(format));
        Assert.Equal(singleSubgroup, MatMulQuantGemvMultiKernel.RequiresWave32(format));
    }

    [Fact]
    public void Iq4Xs_IsASingleSubgroupShader()
    {
        // The case that broke on Intel Arc: keeps the theory above from passing vacuously if the shader detection drifts.
        Assert.True(ReducesWithOneSubgroupOnly(ShaderSource(QuantGemvMultiFormat.IQ4_XS)));
    }

    [Theory]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 32u, 32u)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 0u, null)]          // no pin available (or DOTLLM_VULKAN_DISABLE_WAVE32=1): refuse, fall back to the GEMM
    [InlineData(QuantGemvMultiFormat.Q4_K, 0u, 0u)]              // cross-subgroup reduction: driver default is fine
    [InlineData(QuantGemvMultiFormat.Q4_K, 32u, 0u)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 32u, 0u)]
    public void PipelineSubgroupSize_PinsOrRefuses(QuantGemvMultiFormat format, uint wave32Pin, uint? expected)
        => Assert.Equal(expected, MatMulQuantGemvMultiKernel.PipelineSubgroupSize(format, wave32Pin));
}
