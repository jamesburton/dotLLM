using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The 2..8-row fast paths of the Vulkan qwen4exp model (#876): the multi-column Q8_0 / F16 / F32 GEMVs for the dense projections and the
/// multi-row indexed MoE MMVQs. Fixture 11 is the released shape class: 512 experts / top-10 / intermediate 640 (not a multiple of 256),
/// Q4_K gate/up + Q5_1 down experts, Q8_0 projections, a GDN with NK = 2 != NV = 4 and a GQA ratio of 2 (no degenerate head counts).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpSmallRowTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpSmallRowTests(ITestOutputHelper output) => _out = output;

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
    [InlineData(2)]
    [InlineData(5)]
    [InlineData(8)]
    [InlineData(9)]
    public void ChunkedRows_TakeTheSmallRowPaths_AndStayOracleClose(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int T = 27;   // chunks of `rows` plus a ragged tail (27 = exactly 3 x 9, so the 9-row case has no short tail chunk)
        var ids = VulkanQwen4ExpParityTests.Ids(T, 128, seed: 33);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        long F32() => rig.Vk.SmallRowPathCount(VulkanQwen3MoeHybridTransformerModel.SmallRowPath.F32Multi);
        long F16() => rig.Vk.SmallRowPathCount(VulkanQwen3MoeHybridTransformerModel.SmallRowPath.F16Multi);
        long Q8() => rig.Vk.SmallRowPathCount(VulkanQwen3MoeHybridTransformerModel.SmallRowPath.Q8Multi);
        long Q4() => rig.Vk.SmallRowPathCount(VulkanQwen3MoeHybridTransformerModel.SmallRowPath.MoeQ4KMr);
        long Q51() => rig.Vk.SmallRowPathCount(VulkanQwen3MoeHybridTransformerModel.SmallRowPath.MoeQ5_1Mr);

        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, Enumerable.Range(0, T).ToArray(), -1), T - 1);
        bool prior = VulkanQwen4ExpTransformerModel.SmallRowGemv;
        float[] fast, slow;
        try
        {
            VulkanQwen4ExpTransformerModel.SmallRowGemv = true;
            fast = Chunked(rig, ids, rows);
            long f32 = F32(), f16 = F16(), q8 = Q8(), q4 = Q4(), q51 = Q51();
            _out.WriteLine($"rows {rows}: F32Multi {f32} F16Multi {f16} Q8Multi {q8} MoeQ4KMr {q4} MoeQ5_1Mr {q51}");
            if (rows == 1)
            {
                // single-token decode must be untouched: not one small-row dispatch was recorded
                Assert.Equal(0, f32 + f16 + q8 + q4 + q51);
            }
            else if (rows <= 8)
            {
                Assert.True(f32 > 0 && f16 > 0 && q8 > 0 && q4 > 0 && q51 > 0, "a 2..8-row fast path never ran");
            }
            else
            {
                // 9 rows: no multi-column GEMV (n > 8 -> GEMM) but still below the grouped threshold -> multi-row MMVQ
                Assert.Equal(0, f32 + f16 + q8);
                Assert.True(q4 > 0 && q51 > 0, "multi-row MoE MMVQ never ran at 9 rows");
            }

            VulkanQwen4ExpTransformerModel.SmallRowGemv = false;
            long before = Q8() + F32() + F16() + Q4() + Q51();
            slow = Chunked(rig, ids, rows);
            Assert.Equal(before, Q8() + F32() + F16() + Q4() + Q51());   // disabled: nothing recorded
        }
        finally { VulkanQwen4ExpTransformerModel.SmallRowGemv = prior; }

        var (fastRel, fastKl, _) = VulkanQwen4ExpParityTests.Compare(cpu, fast);
        var (slowRel, slowKl, _) = VulkanQwen4ExpParityTests.Compare(cpu, slow);
        var (armRel, armKl, _) = VulkanQwen4ExpParityTests.Compare(slow, fast);
        _out.WriteLine($"rows {rows}: fast vs CPU relL2 {fastRel:E3} KL {fastKl:E3}; old path vs CPU relL2 {slowRel:E3} KL {slowKl:E3}; fast vs old relL2 {armRel:E3} KL {armKl:E3}");
        Assert.True(fastRel < 0.08 && fastKl < 0.02, $"fast path off the oracle: relL2 {fastRel:E3}, KL {fastKl:E3}");
        Assert.True(armRel < 0.05 && armKl < 0.01, $"fast path diverges from the previous multi-row path: relL2 {armRel:E3}, KL {armKl:E3}");
        if (rows is >= 2 and <= 8)
            Assert.True(armRel > 0, "perturbation inert: the fast path produced bit-identical logits to the old kernels");
    }
}
