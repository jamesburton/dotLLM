using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The GPU-side PLE key/value projections of short qwen4exp forwards (#885): the token-only half of the n-gram branch runs ahead of layer 0
/// on the GPU instead of as a CPU GEMM between layer 0 and layer 1. Must stay oracle-close, must measurably take the path, and the
/// opt-out switch must restore the all-host branch.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen4ExpPleGpuTests
{
    private readonly ITestOutputHelper _out;
    public VulkanQwen4ExpPleGpuTests(ITestOutputHelper output) => _out = output;

    private static float[] Chunked(Q4eRig rig, int[] ids, int rows, out int forwards)
    {
        using var state = rig.Vk.CreateState();
        float[] last = [];
        forwards = 0;
        for (int i = 0; i < ids.Length; i += rows)
        {
            int n = Math.Min(rows, ids.Length - i);
            last = Q4eRig.Row(rig.Vk.Forward(ids.AsSpan(i, n), Enumerable.Range(i, n).ToArray(), -1, state), 0);
            forwards++;
        }
        return last;
    }

    [SkippableTheory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(9)]
    public void ShortForwards_ProjectPleOnGpu_AndStayOracleClose(int rows)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int T = 27;
        var ids = VulkanQwen4ExpParityTests.Ids(T, 128, seed: 41);
        using var rig = new Q4eRig(VulkanQwen4ExpParityTests.Build(11), spvDir);
        var cpu = Q4eRig.Row(rig.Cpu.Forward(ids, Enumerable.Range(0, T).ToArray(), -1), T - 1);
        bool prior = VulkanQwen4ExpTransformerModel.PleOnGpu;
        float[] gpu, host;
        try
        {
            VulkanQwen4ExpTransformerModel.PleOnGpu = true;
            long b0 = rig.Vk.PleGpuForwards;
            gpu = Chunked(rig, ids, rows, out int fwds);
            long taken = rig.Vk.PleGpuForwards - b0;
            _out.WriteLine($"rows {rows}: {taken}/{fwds} forwards projected PLE on the GPU");
            if (rows <= 8) Assert.Equal(fwds, taken);   // every short forward takes it
            else Assert.Equal(0, taken);                // 9-row chunks are the all-host branch

            VulkanQwen4ExpTransformerModel.PleOnGpu = false;
            long b1 = rig.Vk.PleGpuForwards;
            host = Chunked(rig, ids, rows, out _);
            Assert.Equal(b1, rig.Vk.PleGpuForwards);    // switched off: nothing recorded
        }
        finally { VulkanQwen4ExpTransformerModel.PleOnGpu = prior; }

        var (gRel, gKl, _) = VulkanQwen4ExpParityTests.Compare(cpu, gpu);
        var (hRel, hKl, _) = VulkanQwen4ExpParityTests.Compare(cpu, host);
        var (aRel, aKl, _) = VulkanQwen4ExpParityTests.Compare(host, gpu);
        _out.WriteLine($"rows {rows}: gpu-PLE vs CPU relL2 {gRel:E3} KL {gKl:E3}; host-PLE vs CPU relL2 {hRel:E3} KL {hKl:E3}; gpu vs host relL2 {aRel:E3} KL {aKl:E3}");
        Assert.True(gRel < 0.08 && gKl < 0.02, $"GPU-PLE path off the oracle: relL2 {gRel:E3}, KL {gKl:E3}");
        Assert.True(aRel < 0.02 && aKl < 0.005, $"GPU-PLE path diverges from the host branch: relL2 {aRel:E3}, KL {aKl:E3}");
        if (rows <= 8)
            Assert.True(aRel > 0, "perturbation inert: the GPU projections produced bit-identical logits to the host GEMM");
    }
}
