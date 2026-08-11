using DotLLM.Core.Configuration;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tests for <see cref="VulkanResidencyReport"/> — the accounting that makes weight
/// expansion (a type with no Vulkan kernel being widened to F32 on upload) observable
/// instead of silent.
/// </summary>
public class VulkanResidencyReportTests
{
    [Fact]
    public void Report_CountsExpansion_WhenTypeHasNoVulkanKernel()
    {
        var report = new VulkanResidencyReport();
        // Q5_0: 22 bytes per 32 elements; expanded to F32 = 128 bytes per 32.
        report.Add("blk.0.ffn_down.weight", QuantizationType.Q5_0, QuantizationType.F32,
            packedBytes: 22 * 64, uploadedBytes: 128 * 64);
        report.Add("blk.0.attn_q.weight", QuantizationType.Q8_0, QuantizationType.Q8_0,
            packedBytes: 34 * 64, uploadedBytes: 34 * 64);

        Assert.Equal(1, report.ExpandedTensorCount);
        Assert.Equal(22 * 64 + 34 * 64, report.PackedBytes);
        Assert.Equal(128 * 64 + 34 * 64, report.UploadedBytes);
        Assert.Contains("blk.0.ffn_down.weight", report.Describe());
        Assert.Contains("Q5_0", report.Describe());
        // The tensor that stayed packed must NOT be listed as expanded.
        Assert.DoesNotContain("blk.0.attn_q.weight", report.Describe());
    }

    [Fact]
    public void Report_IsClean_WhenNothingExpanded()
    {
        var report = new VulkanResidencyReport();
        report.Add("blk.0.attn_q.weight", QuantizationType.Q4_K, QuantizationType.Q4_K,
            packedBytes: 144 * 16, uploadedBytes: 144 * 16);

        Assert.Equal(0, report.ExpandedTensorCount);
        Assert.Equal(report.PackedBytes, report.UploadedBytes);
    }
}
