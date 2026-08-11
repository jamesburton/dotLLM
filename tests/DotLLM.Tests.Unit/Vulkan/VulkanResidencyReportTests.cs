using System.Linq;
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

    /// <summary>
    /// #327: a routed-MoE layer's three banks must be resolved independently, not
    /// ANDed model-globally. Uses the SAME per-bank contraction dims the real loader
    /// does (<c>VulkanWeights.CanSkipMoeF32HostDequant</c>'s DeepSeek-V2 loop): gate/up
    /// contract along <c>hiddenSize</c>, down contracts along the MoE intermediate size.
    /// </summary>
    /// <remarks>
    /// The brief's original 3-bank example passed a single <c>inputDim: 1408</c> to
    /// every bank, including the two Q4_K gate/up banks — but Q4_K requires 256-alignment
    /// and 1408 % 256 != 0, so that literal example can never pass against
    /// <c>CanKeepBankResident</c>'s real (K-quant super-block-aligned) implementation.
    /// This version keeps the scenario (2 resident-capable siblings + 1 not, #327) but
    /// gives gate/up their production dimension (hiddenSize, 256-aligned) instead.
    /// </remarks>
    [Fact]
    public void MoeSkip_IsPerBank_NotModelGlobal()
    {
        const int hiddenSize = 2048;       // gate/up contraction axis — 256-aligned.
        const int moeIntermediate = 1408;  // down contraction axis — 32- but not 256-aligned.

        // 3 banks: two Q4_K (resident-capable), one Q5_0 (not — no Vulkan Q5_0 kernel
        // in this worktree; #344 Unit 1 adds it).
        var banks = new[]
        {
            (Name: "blk.0.ffn_gate_exps.weight", Qt: QuantizationType.Q4_K, InputDim: hiddenSize),
            (Name: "blk.0.ffn_up_exps.weight",   Qt: QuantizationType.Q4_K, InputDim: hiddenSize),
            (Name: "blk.0.ffn_down_exps.weight", Qt: QuantizationType.Q5_0, InputDim: moeIntermediate),
        };

        var resident = banks
            .Where(b => VulkanWeights.CanKeepBankResident(b.Qt, b.InputDim))
            .Select(b => b.Name)
            .ToArray();

        Assert.Equal(2, resident.Length);
        Assert.DoesNotContain("blk.0.ffn_down_exps.weight", resident);
    }
}
