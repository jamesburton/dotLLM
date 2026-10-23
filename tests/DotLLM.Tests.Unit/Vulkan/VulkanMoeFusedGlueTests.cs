using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Fused glue of the grouped MoE prefill: (1) <see cref="MoeExpandGatherGroupF32Kernel"/> replaces broadcast + expand-group and also emits the
/// inverse permutation; (2) <see cref="MoeWeightedScatterGroupedF32Kernel"/> replaces ungroup + weighted scatter. The gather's slot order inside
/// an expert group is atomic arrival order (as in the unfused kernel), so it is checked by invariants; the scatter is bit-identical to the
/// two-kernel chain it replaces.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeFusedGlueTests
{
    [SkippableTheory]
    [InlineData(37, 4, 8, 64)]     // ragged token count, small expert pool
    [InlineData(512, 8, 256, 2048)]  // the Qwen3.6-35B-A3B prefill shape
    [InlineData(1, 8, 16, 128)]
    public void ExpandGather_PacksRowsByExpertAndRecordsBothPermutations(int seqLen, int topK, int numExperts, int hidden)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(MoeExpandGatherGroupF32Kernel.IsSupportedOn(spvDir), "fused gather SPIR-V missing.");
        using var device = VulkanDevice.Create();

        int rows = seqLen * topK;
        var rng = new Random(0x676 + seqLen);
        int[] indices = new int[rows];
        for (int t = 0; t < seqLen; t++)       // distinct experts per token, like a real top-k
        {
            var picks = Enumerable.Range(0, numExperts).OrderBy(_ => rng.Next()).Take(Math.Min(topK, numExperts)).ToArray();
            for (int k = 0; k < topK; k++) indices[t * topK + k] = picks[k % picks.Length];
        }
        float[] x = Q4KFixture.RandomFloats(rng, seqLen * hidden, 1f);

        using var bx = device.Allocate((long)seqLen * hidden * 4);
        using var bidx = device.Allocate((long)rows * 4);
        using var boff = device.Allocate((long)(numExperts + 1) * 4);
        using var bcnt = device.Allocate((long)numExperts * 4);
        using var bcounters = device.Allocate((long)numExperts * 4);
        using var bpacked = device.Allocate((long)rows * hidden * 4);
        using var bperm = device.Allocate((long)rows * 4);
        using var binv = device.Allocate((long)rows * 4);
        device.Upload(x, bx);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), bidx);

        using var offsetsK = MoeExpertOffsetsKernel.Create(device, spvDir);
        using var gather = MoeExpandGatherGroupF32Kernel.Create(device, spvDir);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            offsetsK.Record(ctx.CommandBuffer, bidx, bcnt, boff, bcounters, rows, numExperts);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            gather.Record(ctx.CommandBuffer, bx, bidx, boff, bcounters, bpacked, bperm, binv, rows, hidden, numExperts, topK);
            ctx.SubmitAndWait();
        }

        float[] packed = new float[(long)rows * hidden];
        float[] permF = new float[rows], invF = new float[rows], offF = new float[numExperts + 1];
        device.Download(bpacked, packed);
        device.Download(bperm, permF);
        device.Download(binv, invF);
        device.Download(boff, offF);
        uint U(float f) => BitConverter.SingleToUInt32Bits(f);

        var seen = new bool[rows];
        for (int g = 0; g < rows; g++)
        {
            uint row = U(permF[g]);
            Assert.True(row < rows, $"perm[{g}] = {row} out of range");
            Assert.False(seen[row], $"routed row {row} appears twice in the permutation");
            seen[row] = true;
            Assert.Equal((uint)g, U(invF[row]));

            int expert = indices[row];
            Assert.True(g >= U(offF[expert]) && g < U(offF[expert + 1]), $"packed row {g} (routed row {row}) is outside expert {expert}'s group");
            int token = (int)row / topK;
            for (int c = 0; c < hidden; c++)
                Assert.True(packed[(long)g * hidden + c] == x[(long)token * hidden + c], $"packed[{g}, {c}] != x[{token}, {c}]");
        }
    }

    [SkippableTheory]
    [InlineData(37, 4, 64)]
    [InlineData(512, 8, 2048)]
    [InlineData(1, 8, 128)]
    public void WeightedScatterGrouped_EqualsUngroupThenWeightedScatter(int seqLen, int topK, int hidden)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(MoeWeightedScatterGroupedF32Kernel.IsSupportedOn(spvDir), "fused scatter SPIR-V missing.");
        using var device = VulkanDevice.Create();

        int rows = seqLen * topK;
        var rng = new Random(0x677 + seqLen);
        uint[] inv = Enumerable.Range(0, rows).Select(i => (uint)i).OrderBy(_ => rng.Next()).ToArray();   // routed row -> grouped row
        uint[] perm = new uint[rows];                                                                    // grouped row -> routed row
        for (int r = 0; r < rows; r++) perm[inv[r]] = (uint)r;
        float[] grouped = Q4KFixture.RandomFloats(rng, rows * hidden, 1f);
        float[] weights = Q4KFixture.RandomFloats(rng, rows, 1f);

        using var bg = device.Allocate((long)rows * hidden * 4);
        using var binv = device.Allocate((long)rows * 4);
        using var bperm = device.Allocate((long)rows * 4);
        using var bw = device.Allocate((long)rows * 4);
        using var bUngrouped = device.Allocate((long)rows * hidden * 4);
        using var bOutChain = device.Allocate((long)seqLen * hidden * 4);
        using var bOutFused = device.Allocate((long)seqLen * hidden * 4);
        device.Upload(grouped, bg);
        device.Upload(MemoryMarshal.AsBytes<uint>(inv), binv);
        device.Upload(MemoryMarshal.AsBytes<uint>(perm), bperm);
        device.Upload(weights, bw);

        using var ungroup = MoeUngroupScatterF32Kernel.Create(device, spvDir);
        using var scatter = MoeWeightedScatterF32Kernel.Create(device, spvDir);
        using var fused = MoeWeightedScatterGroupedF32Kernel.Create(device, spvDir);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            ungroup.Record(ctx.CommandBuffer, bg, bperm, bUngrouped, rows, hidden);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            scatter.Record(ctx.CommandBuffer, bUngrouped, bw, bOutChain, seqLen, topK, hidden);
            fused.Record(ctx.CommandBuffer, bg, binv, bw, bOutFused, seqLen, topK, hidden);
            ctx.SubmitAndWait();
        }

        float[] a = new float[(long)seqLen * hidden], b = new float[(long)seqLen * hidden];
        device.Download(bOutChain, a);
        device.Download(bOutFused, b);
        for (int i = 0; i < a.Length; i++)
            Assert.True(a[i] == b[i], $"idx {i} (token {i / hidden}, col {i % hidden}): chain {a[i]:G9} != fused {b[i]:G9}");
    }
}
