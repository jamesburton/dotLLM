using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// The grouped-by-expert MoE GEMMs launched a grid of (mTiles, maxRowsPerExpert / 16, numExperts) workgroups, ~30x more than the real
/// (expert, row tile) pairs at serving prompt sizes. <see cref="MoeBuildTileListKernel"/> enumerates only the real tiles on the GPU and
/// the GEMMs consume them through <c>vkCmdDispatchIndirect</c>. These tests pin (1) the tile list itself against a CPU build and (2) that
/// the indirect launch produces EXACTLY the legacy launch's output (same arithmetic, different grid).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeGroupedIndirectTests
{
    private static uint[] OffsetsOf(int[] counts)
    {
        var o = new uint[counts.Length + 1];
        for (int i = 0; i < counts.Length; i++) o[i + 1] = o[i] + (uint)counts[i];
        return o;
    }

    [SkippableTheory]
    [InlineData("3,0,17,40")]
    [InlineData("0,0,0,5")]
    [InlineData("16,16,16,16,16,16,16,16")]
    [InlineData("1")]
    [InlineData("33,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1")]
    public void TileList_MatchesCpuBuild(string countsCsv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(MoeBuildTileListKernel.IsSupportedOn(spvDir), "tile-list SPIR-V missing.");
        using var device = VulkanDevice.Create();

        int[] counts = Array.ConvertAll(countsCsv.Split(','), int.Parse);
        int E = counts.Length, rows = counts.Sum();
        long uints = MoeBuildTileListKernel.OffsetsBufferUints(E, Math.Max(rows, 1));
        using var data = device.Allocate(uints * sizeof(uint));
        using var args = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        device.Upload(MemoryMarshal.AsBytes<uint>(OffsetsOf(counts)), data);

        using var build = MoeBuildTileListKernel.Create(device, spvDir);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            build.Record(ctx.CommandBuffer, data, args, E, mTiles0: 7, mTiles1: 9);
            ctx.SubmitAndWait();
        }

        var expect = new List<(uint Expert, uint Row)>();
        for (int e = 0; e < E; e++)
            for (int r = 0; r < counts[e]; r += MoeBuildTileListKernel.TileRows) expect.Add(((uint)e, (uint)r));

        float[] rawData = new float[uints];
        float[] rawArgs = new float[6];
        device.Download(data, rawData);
        device.Download(args, rawArgs);
        uint U(float f) => BitConverter.SingleToUInt32Bits(f);

        Assert.Equal(7u, U(rawArgs[0]));
        Assert.Equal((uint)expect.Count, U(rawArgs[1]));
        Assert.Equal(1u, U(rawArgs[2]));
        Assert.Equal(9u, U(rawArgs[3]));
        Assert.Equal((uint)expect.Count, U(rawArgs[4]));
        Assert.Equal(1u, U(rawArgs[5]));
        for (int t = 0; t < expect.Count; t++)
        {
            Assert.Equal(expect[t].Expert, U(rawData[E + 1 + 2 * t]));
            Assert.Equal(expect[t].Row, U(rawData[E + 1 + 2 * t + 1]));
        }
    }

    [SkippableTheory]
    [InlineData(MoeGroupedKQuant.Q4_K, 64, 512, "0,3,0,17,40,16,15,1", 64)]   // expert 0 empty: a shifted list cannot alias tile (0, 0)
    [InlineData(MoeGroupedKQuant.Q4_K, 48, 256, "0,3,0,17,40", 16)]
    [InlineData(MoeGroupedKQuant.Q5_K, 64, 768, "0,16,15,17,1,33,0,2", 64)]
    [InlineData(MoeGroupedKQuant.Q5_K, 64, 768, "16,15,17,1", 16)]
    [InlineData(MoeGroupedKQuant.Q6_K, 128, 512, "0,16,15,17,1,0,9,48", 64)]
    [InlineData(MoeGroupedKQuant.Q6_K, 64, 512, "16,15,17,1,0,9", 16)]
    public void IndirectLaunch_EqualsLegacyLaunch(MoeGroupedKQuant quant, int m, int k, string countsCsv, int tile)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulKQuantCoopmatKernel.IsSupportedOn(device, spvDir, quant) && MoeBuildTileListKernel.IsSupportedOn(spvDir),
            "coopmat or SPIR-V unavailable.");

        int[] counts = Array.ConvertAll(countsCsv.Split(','), int.Parse);
        int E = counts.Length, rows = counts.Sum();
        uint[] offsets = OffsetsOf(counts);

        var rng = new Random(0x674 + m * 31 + k + rows);
        byte[] bank = Enumerable.Range(0, E).SelectMany(_ =>
        {
            float[] w = Q4KFixture.RandomFloats(rng, m * k, 0.5f);
            return quant switch
            {
                MoeGroupedKQuant.Q4_K => Q4KFixture.QuantizeRows(w, m, k),
                MoeGroupedKQuant.Q5_K => Q5KFixture.QuantizeRows(w, m, k),
                _ => Q6KFixture.QuantizeRows(w, m, k),
            };
        }).ToArray();
        float[] x = Q4KFixture.RandomFloats(rng, rows * k, 1f);

        using var kernel = MoeGroupedMatmulKQuantCoopmatKernel.Create(device, spvDir, quant, tile);
        using var build = MoeBuildTileListKernel.Create(device, spvDir);
        using var bufW = device.Allocate(bank.Length);
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufOff = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
        using var bufArgs = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        using var yLegacy = device.Allocate((long)rows * m * sizeof(float));
        using var yIndirect = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(bank, bufW);
        device.Upload(x, bufX);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bufOff);

        kernel.Launch(bufW, bufX, bufOff, yLegacy, m, k, rows, E, maxRowsPerExpert: counts.Max());
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            build.Record(ctx.CommandBuffer, bufOff, bufArgs, E, kernel.MTiles(m), kernel.MTiles(m));
            KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
            kernel.RecordIndirect(ctx.CommandBuffer, bufW, bufX, bufOff, yIndirect, bufArgs, 0, m, k, rows, E);
            ctx.SubmitAndWait();
        }

        float[] a = new float[(long)rows * m], b = new float[(long)rows * m];
        device.Download(yLegacy, a);
        device.Download(yIndirect, b);
        for (int i = 0; i < a.Length; i++)
            Assert.True(a[i] == b[i], $"{quant} tile{tile} idx {i} (row {i / m}, col {i % m}) counts={countsCsv}: legacy {a[i]:G9} != indirect {b[i]:G9}");
    }
}
