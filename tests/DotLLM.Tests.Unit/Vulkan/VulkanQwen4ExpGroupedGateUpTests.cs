using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Tests.Unit.Models.Qwen4Exp;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #821: a layer whose gate/up experts are Q5_K with a legacy-quant (Q8_0) down - layer 2 of the real UD-Q4_K_XL file - used to fail the
/// "gate/up must be Q4_K" test of the grouped coopmat prefill and ran the scalar indexed kernels (~half of a 1K prefill on that one layer).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpGroupedGateUpTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpGroupedGateUpTests(ITestOutputHelper output) => _out = output;

    [Fact]
    public unsafe void TestQ5KEncoder_RoundTripsThroughTheCpuDequantizer()
    {
        var rng = new Random(5);
        var x = new float[1024];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1) * (1 + (i / 256));
        var bytes = Qwen4ExpRandomGguf.EncodeQ5K(x);
        Assert.Equal(4 * 176, bytes.Length);
        var back = new float[x.Length];
        fixed (byte* p = bytes)
            Dequantize.ToFloat32((nint)p, x.Length, QuantizationType.Q5_K, back);
        double err = 0, sig = 0;
        for (int i = 0; i < x.Length; i++) { err += (x[i] - back[i]) * (x[i] - back[i]); sig += x[i] * x[i]; }
        Assert.True(Math.Sqrt(err / sig) < 0.03, $"Q5_K encoder layout is off: relL2 {Math.Sqrt(err / sig):E3}");
    }

    private static byte[] Build(int budgetAndContext = 0)
    {
        var g = Qwen4ExpRandomGguf.Real512x640;
        if (budgetAndContext > 0) g = g with { Budget = budgetAndContext, Context = budgetAndContext + 64 };
        return Qwen4ExpRandomGguf.Build(g, Q4eQuant.RealMixQ5KQ80);
    }

    [SkippableTheory]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(64)]
    public void Q5KGateUp_Q80Down_TakesTheGroupedArm_AndMatchesOracle(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(Build(), spvDir);
        Assert.All(rig.Vk.ExpertBankDeviceTypes, t =>
        {
            Assert.Equal(QuantizationType.Q5_K, t.Gate);
            Assert.Equal(QuantizationType.Q5_K, t.Up);
            Assert.Equal(QuantizationType.Q8_0, t.Down);
        });
        var ids = VulkanQwen4ExpParityTests.Ids(rows, 128, seed: 31);
        var pos = Enumerable.Range(0, rows).ToArray();
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, pos, -1), rows - 1);
        var grouped = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
        var kind = VulkanQwen3MoeHybridTransformerModel.MoePath.GroupedGateUpNotQ4K;
        Assert.True(rig.Vk.MoePathCount(kind) > 0, "Q5_K gate/up never took the grouped arm");
        Assert.True(rig.Vk.MoePathCount(VulkanQwen3MoeHybridTransformerModel.MoePath.GroupedLegacyDown) > 0);

        // Perturbation: the same resident banks through the scalar indexed kernels must give different bits.
        rig.Vk.MoeGroupedMinTokens = int.MaxValue;
        long before = rig.Vk.MoePathCount(kind);
        var scalar = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
        Assert.Equal(before, rig.Vk.MoePathCount(kind));
        var (gRel, gKl, _) = VulkanQwen4ExpParityTests.Compare(cpu, grouped);
        var (sRel, _, _) = VulkanQwen4ExpParityTests.Compare(cpu, scalar);
        var (armRel, _, _) = VulkanQwen4ExpParityTests.Compare(scalar, grouped);
        _out.WriteLine($"rows {rows}: grouped vs CPU relL2 {gRel:E3} KL {gKl:E3}; scalar vs CPU {sRel:E3}; grouped vs scalar {armRel:E3}");
        Assert.True(gRel < 0.08 && gKl < 0.02, $"grouped Q5_K gate/up is off the oracle: relL2 {gRel:E3}, KL {gKl:E3}");
        Assert.True(armRel > 0, "perturbation inert: the grouped and scalar arms produced identical logits");
    }

    [SkippableTheory]
    [InlineData(512)]
    [InlineData(1000)]
    public void Q5KGateUp_LargeRowCounts_AgreeWithTheScalarArm(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var rig = new Q4eRig(Build(budgetAndContext: 1100), spvDir);
        var ids = VulkanQwen4ExpParityTests.Ids(rows, 128, seed: 33);
        var pos = Enumerable.Range(0, rows).ToArray();
        var grouped = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
        Assert.True(rig.Vk.MoePathCount(VulkanQwen3MoeHybridTransformerModel.MoePath.GroupedGateUpNotQ4K) > 0);
        rig.Vk.MoeGroupedMinTokens = int.MaxValue;
        var scalar = Q4eRig.Row(rig.Vk.Forward(ids, pos, -1), 0);
        var (rel, kl, _) = VulkanQwen4ExpParityTests.Compare(scalar, grouped);
        _out.WriteLine($"rows {rows}: grouped vs scalar relL2 {rel:E3} KL {kl:E3}");
        Assert.True(rel < 0.08 && kl < 0.02 && rel > 0, $"relL2 {rel:E3} KL {kl:E3}");
    }
}
