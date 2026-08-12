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
    /// <remarks>
    /// The non-resident exemplar is Q4_K at <c>moeIntermediate</c> (1408), not Q5_0:
    /// Q5_0 only failed to resolve resident here because no Vulkan Q5_0 kernel existed yet
    /// in this worktree, and #344 added one (<c>KeepQ5_0OnDevice</c>, gated only on
    /// 32-alignment — 1408 % 32 == 0, so Q5_0 at this dim IS resident-capable now). Picking
    /// a non-resident exemplar whose non-residency is contingent on "kernel not implemented
    /// yet" is exactly the kind of thing this project keeps fixing, which rotted this test
    /// once already. Q4_K's 256-super-block alignment gate is structural — 1408 % 256 == 128
    /// != 0 forever, regardless of what kernels get added — so it stays a valid "not
    /// resident-capable" exemplar for the life of the K-quant format.
    /// </remarks>
    [Fact]
    public void MoeSkip_IsPerBank_NotModelGlobal()
    {
        const int hiddenSize = 2048;       // gate/up contraction axis — 256-aligned.
        const int moeIntermediate = 1408;  // down contraction axis — 32-aligned but not 256-aligned.

        // 3 banks: two Q5_0 (resident-capable, 32-alignment only), one Q4_K (not — K-quants
        // require 256-super-block alignment and moeIntermediate is only 32-aligned).
        var banks = new[]
        {
            (Name: "blk.0.ffn_gate_exps.weight", Qt: QuantizationType.Q5_0, InputDim: hiddenSize),
            (Name: "blk.0.ffn_up_exps.weight",   Qt: QuantizationType.Q5_0, InputDim: hiddenSize),
            (Name: "blk.0.ffn_down_exps.weight", Qt: QuantizationType.Q4_K, InputDim: moeIntermediate),
        };

        var resident = banks
            .Where(b => VulkanWeights.CanKeepBankResident(b.Qt, b.InputDim))
            .Select(b => b.Name)
            .ToArray();

        Assert.Equal(2, resident.Length);
        Assert.DoesNotContain("blk.0.ffn_down_exps.weight", resident);
    }
}
