using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The 2..8-row (speculative verify) forward of the Vulkan qwen4exp model attends its dense full-attention layers with the per-row split-KV kernel
/// (<c>VulkanSplitKvRowsAttentionKernel</c>) instead of the f16 coopmat flash attention. This pins the WIRING: the kernel is dispatched once per dense QSA layer of
/// a 2..8-row forward, never for a 1-row or 9+-row forward or with the switch off, and the logits stay on the CPU oracle.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpVerifyRowsAttentionTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpVerifyRowsAttentionTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(5)]
    [InlineData(8)]
    public void VerifyRows_TakeThePerRowSplitKvKernel_PerDenseQsaLayer_AndStayOracleClose(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int ctx = 40;   // > 16 so the first verify row has >= 17 keys and splits like a decode step would
        var ids = VulkanQwen4ExpParityTests.Ids(ctx + rows, 128, seed: 91);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        int qsaLayers = rig.Vk.Qsa.Ordinal.Count(o => o >= 0);
        Assert.True(qsaLayers > 0);

        float[] Run(bool rowsKernel, out long dispatches)
        {
            bool prior = VulkanQwen3MoeHybridTransformerModel.Q4SplitKvRows;
            VulkanQwen3MoeHybridTransformerModel.Q4SplitKvRows = rowsKernel;
            try
            {
                using var state = rig.Vk.CreateState();
                using (rig.Vk.Forward(ids.AsSpan(0, ctx), Enumerable.Range(0, ctx).ToArray(), -1, state)) { }
                long before = rig.Vk.SplitKvRowsDispatches;
                var last = Q4eRig.Row(rig.Vk.Forward(ids.AsSpan(ctx, rows), Enumerable.Range(ctx, rows).ToArray(), -1, state), 0);
                dispatches = rig.Vk.SplitKvRowsDispatches - before;
                return last;
            }
            finally { VulkanQwen3MoeHybridTransformerModel.Q4SplitKvRows = prior; }
        }

        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, Enumerable.Range(0, ctx + rows).ToArray(), -1), ctx + rows - 1);
        var fast = Run(true, out long fastDispatches);
        var flash = Run(false, out long flashDispatches);

        Assert.Equal(qsaLayers, fastDispatches);   // one dispatch per dense full-attention layer
        Assert.Equal(0, flashDispatches);          // switched off: nothing recorded (the mutant "always flash" fails the line above)
        var (relOracle, klOracle, _) = VulkanQwen4ExpParityTests.Compare(cpu, fast);
        var (relFlash, klFlash, _) = VulkanQwen4ExpParityTests.Compare(flash, fast);
        _out.WriteLine($"rows {rows}: dispatches {fastDispatches} ({qsaLayers} QSA layers); rows-kernel vs CPU relL2 {relOracle:E3} KL {klOracle:E3}; vs flash path relL2 {relFlash:E3} KL {klFlash:E3}");
        Assert.True(relOracle < 0.08 && klOracle < 0.02, $"per-row split-KV forward off the oracle: relL2 {relOracle:E3}, KL {klOracle:E3}");
        Assert.True(relFlash < 0.05 && klFlash < 0.01, $"per-row split-KV forward diverges from the flash path: relL2 {relFlash:E3}, KL {klFlash:E3}");
        Assert.True(relFlash > 0, "perturbation inert: the rows kernel produced bit-identical logits to the f16 flash attention");
    }

    [SkippableFact]
    public void OneRowAndWideForwards_DoNotTakeThePerRowKernel()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int ctx = 40;
        var ids = VulkanQwen4ExpParityTests.Ids(ctx + 9, 128, seed: 92);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        using var state = rig.Vk.CreateState();
        using (rig.Vk.Forward(ids.AsSpan(0, ctx), Enumerable.Range(0, ctx).ToArray(), -1, state)) { }   // a 40-row prefill: wide
        Assert.Equal(0, rig.Vk.SplitKvRowsDispatches);
        using (rig.Vk.Forward(ids.AsSpan(ctx, 1), [ctx], -1, state)) { }                                   // a decode step: the 1-row kernel's job
        Assert.Equal(0, rig.Vk.SplitKvRowsDispatches);
        using (rig.Vk.Forward(ids.AsSpan(ctx + 1, 8), Enumerable.Range(ctx + 1, 8).ToArray(), -1, state)) { }
        Assert.True(rig.Vk.SplitKvRowsDispatches > 0);   // 8 rows is the widest the kernel serves
    }
}
