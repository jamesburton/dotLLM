using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Decode-step restructurings of the Vulkan qwen4exp model (#885): (1) the gated-residual read folds its residual copy into an out-of-place
/// norm and overlaps the inject GEMV with the down GEMV - same kernels and operands, so the logits must be EXACTLY equal to the unfused
/// sequence; (2) single-token MoE layers run on the fused chain (one quantize + xDiv-broadcast indexed MMVQ, SwiGLU fused with the down
/// quantize, shared expert in the routed barrier slots, 8 barriers instead of ~17) - oracle-close. Counters prove each new path ran, and a
/// switched-off run must record nothing.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpDecodeOverlapTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpDecodeOverlapTests(ITestOutputHelper output) => _out = output;

    private static float[] Chunked(Q4eRig rig, int[] ids, int rows)
    {
        using var state = rig.Vk.CreateState();
        float[] last = [];
        for (int i = 0; i < ids.Length; i += rows)
        {
            int n = Math.Min(rows, ids.Length - i);
            last = Q4eRig.Row(rig.Vk.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), -1, state), 0);
        }
        return last;
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    public void FusedGatedResidualRead_IsBitIdenticalToTheUnfusedSequence(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int T = 24;
        var ids = VulkanQwen4ExpParityTests.Ids(T, 128, seed: 57);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        bool prior = VulkanQwen4ExpTransformerModel.GrFused;
        try
        {
            VulkanQwen4ExpTransformerModel.GrFused = true;
            long gr0 = rig.Vk.GrFusedReads;
            var fast = Chunked(rig, ids, rows);
            long gr = rig.Vk.GrFusedReads - gr0;
            _out.WriteLine($"rows {rows}: fused GR reads {gr}");
            Assert.True(gr > 0, "the fused gated-residual read never ran");

            VulkanQwen4ExpTransformerModel.GrFused = false;
            long gr1 = rig.Vk.GrFusedReads;
            var slow = Chunked(rig, ids, rows);
            Assert.Equal(gr1, rig.Vk.GrFusedReads);   // switched off: nothing recorded

            int differing = 0;
            for (int i = 0; i < slow.Length; i++) if (BitConverter.SingleToInt32Bits(slow[i]) != BitConverter.SingleToInt32Bits(fast[i])) differing++;
            Assert.True(differing == 0, $"{differing} of {slow.Length} logits differ between the fused and the unfused read (must be bit-identical)");
        }
        finally { VulkanQwen4ExpTransformerModel.GrFused = prior; }
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(3)]
    public void SingleTokenMoeChain_RunsFused_AndStaysOracleClose(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int T = 24;
        var ids = VulkanQwen4ExpParityTests.Ids(T, 128, seed: 63);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, Enumerable.Range(0, T).ToArray(), -1), T - 1);
        bool prior = VulkanQwen4ExpTransformerModel.MoeDecodeFused;
        float[] fast, slow;
        try
        {
            VulkanQwen4ExpTransformerModel.MoeDecodeFused = true;
            long f0 = rig.Vk.MoeFusedLayers;
            fast = Chunked(rig, ids, rows);
            long fused = rig.Vk.MoeFusedLayers - f0;
            _out.WriteLine($"rows {rows}: {fused} MoE layers on the fused chain");
            if (rows == 1) Assert.True(fused > 0, "the fused single-token MoE chain never ran on decode");
            else Assert.Equal(0, fused);   // several rows keep the general path

            VulkanQwen4ExpTransformerModel.MoeDecodeFused = false;
            long f1 = rig.Vk.MoeFusedLayers;
            slow = Chunked(rig, ids, rows);
            Assert.Equal(f1, rig.Vk.MoeFusedLayers);   // switched off: nothing recorded
        }
        finally { VulkanQwen4ExpTransformerModel.MoeDecodeFused = prior; }

        var (fRel, fKl, _) = VulkanQwen4ExpParityTests.Compare(cpu, fast);
        var (sRel, sKl, _) = VulkanQwen4ExpParityTests.Compare(cpu, slow);
        var (aRel, aKl, _) = VulkanQwen4ExpParityTests.Compare(slow, fast);
        _out.WriteLine($"rows {rows}: fused vs CPU relL2 {fRel:E3} KL {fKl:E3}; general vs CPU relL2 {sRel:E3} KL {sKl:E3}; fused vs general relL2 {aRel:E3} KL {aKl:E3}");
        Assert.True(fRel < 0.08 && fKl < 0.02, $"fused MoE chain off the oracle: relL2 {fRel:E3}, KL {fKl:E3}");
        Assert.True(aRel < 0.05 && aKl < 0.01, $"fused MoE chain diverges from the general path: relL2 {aRel:E3}, KL {aKl:E3}");
    }

    [SkippableFact]
    public void FusedCombine_IsBitIdenticalToScatterThenGatedAdd()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int T = 24;
        var ids = VulkanQwen4ExpParityTests.Ids(T, 128, seed: 71);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        bool prior = VulkanQwen4ExpTransformerModel.CombineFused;
        try
        {
            VulkanQwen4ExpTransformerModel.CombineFused = true;
            long c0 = rig.Vk.CombineFusedLayers;
            var fast = Chunked(rig, ids, 1);
            long used = rig.Vk.CombineFusedLayers - c0;
            _out.WriteLine($"decode: {used} layers used the fused combine kernel");
            Assert.True(used > 0, "the fused scatter + gated-add kernel never ran on decode");

            VulkanQwen4ExpTransformerModel.CombineFused = false;
            long c1 = rig.Vk.CombineFusedLayers;
            var slow = Chunked(rig, ids, 1);
            Assert.Equal(c1, rig.Vk.CombineFusedLayers);   // switched off: nothing recorded

            int differing = 0;
            for (int i = 0; i < slow.Length; i++) if (BitConverter.SingleToInt32Bits(slow[i]) != BitConverter.SingleToInt32Bits(fast[i])) differing++;
            Assert.True(differing == 0, $"{differing} of {slow.Length} logits differ between the fused and the two-pass combine (must be bit-identical)");
        }
        finally { VulkanQwen4ExpTransformerModel.CombineFused = prior; }
    }
}
