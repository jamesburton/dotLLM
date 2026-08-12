using DotLLM.Core.Configuration;
using DotLLM.Cuda;
using Xunit;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Validates <see cref="CudaKernels.EnsureQuantExpansionAllowed"/>: the opt-in gate that stops
/// the CUDA backend from silently expanding a quant type it has no native kernel for into a
/// full, model-lifetime-resident FP16/F32 copy. Pure logic test — no CUDA device required.
/// </summary>
public sealed class CudaQuantExpansionGateTests
{
    /// <summary>
    /// Default (gate disabled) must throw for a type CUDA has no native kernel for, and the
    /// message must be actionable: name the quant type, the tensor, both sizes, and the
    /// opt-in env var.
    /// </summary>
    [Fact]
    public void EnsureQuantExpansionAllowed_DefaultDisabled_ThrowsWithActionableMessage()
    {
        bool prev = CudaKernels.AllowQuantExpansion;
        CudaKernels.AllowQuantExpansion = false;
        try
        {
            var ex = Assert.Throws<InvalidOperationException>(() =>
                CudaKernels.EnsureQuantExpansionAllowed(
                    QuantizationType.Q4_0, "layer 3 Gate projection",
                    compactBytes: 10_000_000, expandedBytes: 40_000_000));

            Assert.Contains("Q4_0", ex.Message);
            Assert.Contains("layer 3 Gate projection", ex.Message);
            Assert.Contains("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION", ex.Message);
            // Both sizes should be human-readable, not raw byte counts.
            Assert.Contains("MiB", ex.Message);
        }
        finally
        {
            CudaKernels.AllowQuantExpansion = prev;
        }
    }

    /// <summary>Every currently-gap-listed type throws by default, not just Q4_0.</summary>
    [Theory]
    [InlineData(QuantizationType.Q4_0)]
    [InlineData(QuantizationType.Q4_1)]
    [InlineData(QuantizationType.Q5_1)]
    [InlineData(QuantizationType.Q3_K)]
    [InlineData(QuantizationType.IQ1_S)]
    [InlineData(QuantizationType.IQ3_XXS)]
    [InlineData(QuantizationType.IQ3_S)]
    [InlineData(QuantizationType.MXFP4)]
    public void EnsureQuantExpansionAllowed_DefaultDisabled_ThrowsForEveryGapType(QuantizationType qt)
    {
        bool prev = CudaKernels.AllowQuantExpansion;
        CudaKernels.AllowQuantExpansion = false;
        try
        {
            Assert.Throws<InvalidOperationException>(() =>
                CudaKernels.EnsureQuantExpansionAllowed(qt, "test tensor", 1000, 4000));
        }
        finally
        {
            CudaKernels.AllowQuantExpansion = prev;
        }
    }

    /// <summary>Opting in via the flag (mirrors DOTLLM_CUDA_ALLOW_QUANT_EXPANSION=1) restores
    /// the old silent-fallback behavior.</summary>
    [Fact]
    public void EnsureQuantExpansionAllowed_Enabled_DoesNotThrow()
    {
        bool prev = CudaKernels.AllowQuantExpansion;
        CudaKernels.AllowQuantExpansion = true;
        try
        {
            CudaKernels.EnsureQuantExpansionAllowed(
                QuantizationType.Q3_K, "layer 0 Q projection",
                compactBytes: 10_000_000, expandedBytes: 40_000_000);
            // No exception — success.
        }
        finally
        {
            CudaKernels.AllowQuantExpansion = prev;
        }
    }

    /// <summary>Env var default resolution: unset/anything-but-"1" means disabled.</summary>
    [Fact]
    public void AllowQuantExpansion_DefaultsFromEnvVar_RequiresExactly1()
    {
        string? prevEnv = Environment.GetEnvironmentVariable("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION");
        bool prevFlag = CudaKernels.AllowQuantExpansion;
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION", "true");
            Assert.False(Environment.GetEnvironmentVariable("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION") == "1");

            Environment.SetEnvironmentVariable("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION", "1");
            Assert.True(Environment.GetEnvironmentVariable("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION") == "1");
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_CUDA_ALLOW_QUANT_EXPANSION", prevEnv);
            CudaKernels.AllowQuantExpansion = prevFlag;
        }
    }
}
