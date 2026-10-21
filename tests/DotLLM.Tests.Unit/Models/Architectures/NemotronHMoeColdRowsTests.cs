using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #1083: the host-side bookkeeping that joins the iGPU cold leg to the CPU leg (pure CPU code; no GPU is touched).
/// A schedule with device, CPU and iGPU groups (in that order): the iGPU row list (<see cref="VulkanNemotronHMoeExpertService.BuildRows"/>)
/// names each row's expert and token in schedule order; the CPU leg writes its rows first and the iGPU's rows follow them in the same
/// buffer (as <c>RunColdLegs</c> places them); the gather index (<see cref="CudaNemotronHTransformerModel.BuildColdGatherIndex"/>)
/// then sends every slot to its own row. The iGPU's output is stood in for by the CPU per-expert API, so the test checks the indexing,
/// not the kernels (those are the Vulkan-only tests' job).
/// </summary>
public sealed unsafe class NemotronHMoeColdRowsTests : IDisposable
{
    private const int ExpertDim = 256, Inter = 256;
    private readonly List<nint> _allocs = [];

    public void Dispose()
    {
        foreach (nint p in _allocs) NativeMemory.Free((void*)p);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    public void CpuAndIgpuRows_LandWhereTheGatherIndexReadsThem(int cpuThreads)
    {
        var moe = Layer(experts: 7);
        const int N = 6, TopK = 3, Slots = N * TopK;
        float* x = Floats(N * ExpertDim, seed: 83);

        // Schedule: 2 device groups, 2 CPU groups, 3 iGPU groups; slot ids ascending within each group; every slot exactly once.
        int[] groupExperts = [0, 3, 5, 1, 6, 2, 4];
        int[][] groupSlots = [[0, 4], [1, 5, 9], [2, 7, 10], [3, 6, 11, 14], [8, 12], [13, 15, 16], [17]];
        int firstCpu = 2, firstIgpu = 4;
        int[] groupStart = new int[groupExperts.Length + 1];
        int[] slotIds = new int[Slots];
        for (int g = 0, pos = 0; g < groupExperts.Length; g++)
        {
            groupStart[g] = pos;
            foreach (int s in groupSlots[g]) slotIds[pos++] = s;
            groupStart[g + 1] = pos;
        }
        Assert.Equal(Slots, groupStart[^1]);
        Assert.Equal(Slots, slotIds.Distinct().Count());
        int coldStart = groupStart[firstCpu], coldRows = Slots - coldStart, igpuRowsAt = groupStart[firstIgpu] - coldStart;

        // The iGPU row list: schedule order, expert per group, token = slot / topK.
        int[] rowExperts = new int[Slots], rowTokens = new int[Slots];
        int igpuRows;
        fixed (int* gs = groupStart, ids = slotIds)
            igpuRows = VulkanNemotronHMoeExpertService.BuildRows(groupExperts, firstIgpu, groupExperts.Length, gs, ids, TopK, rowExperts, rowTokens);
        Assert.Equal(Slots - groupStart[firstIgpu], igpuRows);
        Assert.Equal([6, 6, 2, 2, 2, 4], rowExperts[..igpuRows]);
        Assert.Equal([8 / TopK, 12 / TopK, 13 / TopK, 15 / TopK, 16 / TopK, 17 / TopK], rowTokens[..igpuRows]);

        // The cold rows buffer: the CPU leg's rows first, the "iGPU" rows (one per-expert call per row) after them.
        float* rows = Floats(coldRows * ExpertDim);
        using (var leg = new CudaNemotronHTransformerModel.CpuColdLeg(cpuThreads))
        {
            fixed (int* gs = groupStart, ids = slotIds)
                leg.Run(moe, ExpertDim, TopK, groupExperts, firstCpu, firstIgpu, gs, ids, x, rows);
        }
        for (int r = 0; r < igpuRows; r++)
            NemotronHMoeExpertCompute.Compute(moe, rowExperts[r], ExpertDim, x + (long)rowTokens[r] * ExpertDim, 1, Floats(Inter),
                rows + (long)(igpuRowsAt + r) * ExpertDim);

        int[] idx = new int[Slots];
        fixed (int* ids = slotIds, ix = idx)
            CudaNemotronHTransformerModel.BuildColdGatherIndex(ids, coldStart, Slots, ix);

        // Device slots read their own rows; every cold slot reads a cold row holding its expert applied to its token.
        for (int g = 0; g < firstCpu; g++)
            foreach (int s in groupSlots[g]) Assert.Equal(s, idx[s]);
        Assert.Equal(Slots, idx.Distinct().Count());
        for (int g = firstCpu; g < groupExperts.Length; g++)
        {
            // The expected rows are computed the way each leg computed them: the CPU leg runs a group as one n-row call, the
            // stand-in iGPU one row at a time (a K-quant GEMM and GEMV may round differently).
            int cnt = groupSlots[g].Length;
            bool onIgpu = g >= firstIgpu;
            float* gIn = Floats(cnt * ExpertDim), want = Floats(cnt * ExpertDim);
            for (int j = 0; j < cnt; j++)
                new ReadOnlySpan<float>(x + (long)(groupSlots[g][j] / TopK) * ExpertDim, ExpertDim)
                    .CopyTo(new Span<float>(gIn + (long)j * ExpertDim, ExpertDim));
            if (onIgpu)
                for (int j = 0; j < cnt; j++)
                    NemotronHMoeExpertCompute.Compute(moe, groupExperts[g], ExpertDim, gIn + (long)j * ExpertDim, 1, Floats(Inter),
                        want + (long)j * ExpertDim);
            else
                NemotronHMoeExpertCompute.Compute(moe, groupExperts[g], ExpertDim, gIn, cnt, Floats(cnt * Inter), want);

            for (int j = 0; j < cnt; j++)
            {
                int s = groupSlots[g][j];
                int row = idx[s] - Slots;
                Assert.InRange(row, 0, coldRows - 1);
                Assert.Equal(onIgpu, row >= igpuRowsAt);   // the iGPU's slots read the iGPU's rows
                for (int i = 0; i < ExpertDim; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(want[(long)j * ExpertDim + i]) ==
                        BitConverter.SingleToInt32Bits(rows[(long)row * ExpertDim + i]),
                        $"slot {s} (group {g}, expert {groupExperts[g]}), element {i}");
            }
        }
    }

    [Fact]
    public void BuildRows_OfAnEmptyRange_IsEmpty_AndOfOneGroup_IsThatGroup()
    {
        int[] groupExperts = [4, 9];
        int[] groupStart = [0, 2, 5];
        int[] slotIds = [1, 3, 0, 2, 5];
        int[] e = new int[5], t = new int[5];
        fixed (int* gs = groupStart, ids = slotIds)
        {
            Assert.Equal(0, VulkanNemotronHMoeExpertService.BuildRows(groupExperts, 2, 2, gs, ids, 2, e, t));
            Assert.Equal(3, VulkanNemotronHMoeExpertService.BuildRows(groupExperts, 1, 2, gs, ids, 2, e, t));
        }
        Assert.Equal([9, 9, 9], e[..3]);
        Assert.Equal([0, 1, 2], t[..3]);
    }

    private NemotronHMoeWeights Layer(int experts)
    {
        var rng = new Random(1083);
        long upPer = Dequantize.RowByteSize(ExpertDim, QuantizationType.Q4_K) * Inter;
        long downPer = Dequantize.RowByteSize(Inter, QuantizationType.Q8_0) * ExpertDim;
        float[] up = Rand(rng, experts * Inter * ExpertDim), down = Rand(rng, experts * ExpertDim * Inter);
        nint upBank = Copy(Q4KFixture.QuantizeRows(up, experts * Inter, ExpertDim));
        nint downBank = Copy(Quantize.FromFloat32(down, down.Length, QuantizationType.Q8_0));
        Assert.Equal(upPer * experts, (long)Q4KFixture.QuantizeRows(up, experts * Inter, ExpertDim).Length);
        return new NemotronHMoeWeights
        {
            GateInpWeight = 0, GateInpQuantType = QuantizationType.F32,
            SelectionBias = new float[experts],
            UpExpsWeight = upBank, UpExpsQuantType = QuantizationType.Q4_K, UpPerExpertBytes = upPer,
            DownExpsWeight = downBank, DownExpsQuantType = QuantizationType.Q8_0, DownPerExpertBytes = downPer,
            UpShexpWeight = 0, UpShexpQuantType = QuantizationType.F32,
            DownShexpWeight = 0, DownShexpQuantType = QuantizationType.F32,
            NumExperts = experts, NumExpertsPerTok = 3, MoeIntermediateSize = Inter, SharedIntermediateSize = 1,
            NormalizeWeights = true, WeightsScale = 1f, LatentSize = ExpertDim,
        };
    }

    private static float[] Rand(Random rng, int count)
    {
        var w = new float[count];
        for (int i = 0; i < count; i++) w[i] = (float)(rng.NextDouble() * 2 - 1) * 0.1f;
        return w;
    }

    private float* Floats(int count, int seed = -1)
    {
        float* p = (float*)NativeMemory.AllocZeroed((nuint)Math.Max(count, 1), sizeof(float));
        _allocs.Add((nint)p);
        if (seed >= 0)
        {
            var rng = new Random(seed);
            for (int i = 0; i < count; i++) p[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        return p;
    }

    private nint Copy(byte[] src)
    {
        byte* p = (byte*)NativeMemory.Alloc((nuint)src.Length);
        _allocs.Add((nint)p);
        src.CopyTo(new Span<byte>(p, src.Length));
        return (nint)p;
    }
}
