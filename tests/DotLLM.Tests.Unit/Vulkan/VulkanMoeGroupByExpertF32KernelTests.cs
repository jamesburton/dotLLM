using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>Parity coverage for MoE group/ungroup data-movement kernels.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeGroupByExpertF32KernelTests
{
    [SkippableFact]
    public void ExpandGroupThenUngroup_RoundTripsAndGroupsByExpert()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int rows = 9;
        const int hidden = 7;
        const int numExperts = 4;

        int[] indices = [2, 0, 2, 1, 3, 1, 0, 3, 1];
        uint[] offsets = ComputeOffsets(indices, numExperts);
        uint[] counters = new uint[numExperts];
        float[] x = new float[rows * hidden];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < hidden; c++)
                x[r * hidden + c] = r * 100 + c + 0.25f;

        using var device = VulkanDevice.Create();
        using var offsetsKernel = MoeExpertOffsetsKernel.Create(device, spvDir);
        using var groupKernel = MoeExpandGroupByExpertF32Kernel.Create(device, spvDir);
        using var ungroupKernel = MoeUngroupScatterF32Kernel.Create(device, spvDir);

        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var countsBuf = device.Allocate((long)numExperts * sizeof(uint));
        using var offsetsBuf = device.Allocate((long)offsets.Length * sizeof(uint));
        using var countersBuf = device.Allocate((long)counters.Length * sizeof(uint));
        using var packedBuf = device.Allocate((long)x.Length * sizeof(float));
        using var permBuf = device.Allocate((long)rows * sizeof(uint));
        using var roundTripBuf = device.Allocate((long)x.Length * sizeof(float));

        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
        device.Upload(MemoryMarshal.AsBytes<uint>(Enumerable.Repeat(0xDEADBEEFu, numExperts).ToArray()), countsBuf);
        device.Upload(MemoryMarshal.AsBytes<uint>(Enumerable.Repeat(0xA5A5A5A5u, offsets.Length).ToArray()), offsetsBuf);
        device.Upload(MemoryMarshal.AsBytes<uint>(Enumerable.Repeat(0xCAFEBABEu, counters.Length).ToArray()), countersBuf);

        offsetsKernel.Launch(idxBuf, countsBuf, offsetsBuf, countersBuf, rows, numExperts);

        groupKernel.Launch(xBuf, idxBuf, offsetsBuf, countersBuf, packedBuf, permBuf,
            rows, hidden, numExperts);

        float[] packed = new float[x.Length];
        uint[] permutation = new uint[rows];
        uint[] gpuOffsets = new uint[numExperts + 1];
        uint[] gpuCounts = new uint[numExperts];
        uint[] finalCounters = new uint[numExperts];
        device.Download(packedBuf, packed);
        DownloadUInt32(device, permBuf, permutation);
        DownloadUInt32(device, offsetsBuf, gpuOffsets);
        DownloadUInt32(device, countsBuf, gpuCounts);
        DownloadUInt32(device, countersBuf, finalCounters);

        Assert.Equal(offsets, gpuOffsets);
        AssertGroupedPermutationAndPackedRows(indices, x, packed, permutation, offsets, rows, hidden, numExperts);
        for (int expert = 0; expert < numExperts; expert++)
        {
            Assert.Equal(offsets[expert + 1] - offsets[expert], gpuCounts[expert]);
            Assert.Equal(offsets[expert + 1] - offsets[expert], finalCounters[expert]);
        }

        ungroupKernel.Launch(packedBuf, permBuf, roundTripBuf, rows, hidden);
        float[] roundTrip = new float[x.Length];
        device.Download(roundTripBuf, roundTrip);
        Assert.Equal(x, roundTrip);
    }

    [SkippableFact]
    public void GroupedF16CoopmatMatmul_MatchesCpuReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int rows = 11;
        const int hidden = 32;
        const int outputDim = 19;
        const int numExperts = 4;

        int[] indices = [2, 0, 2, 1, 3, 1, 0, 3, 1, 2, 0];
        uint[] offsets = ComputeOffsets(indices, numExperts);

        var rng = new Random(0x5EED);
        float[] x = F16Bf16Fixture.RandomFloats(rng, rows * hidden, range: 0.5f);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, numExperts * outputDim * hidden, range: 0.1f);
        byte[] weightsF16 = F16Bf16Fixture.QuantizeRowsF16(weightsF32, numExperts * outputDim, hidden);
        float[] expected = CpuGroupedMatmulF16(weightsF16, x, offsets, outputDim, hidden, rows, numExperts);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "VK_KHR_cooperative_matrix not available on this Vulkan device.");
        using var kernel = MoeGroupedMatmulF16CoopmatKernel.Create(device, spvDir);

        using var weightsBuf = device.Allocate((long)weightsF16.Length);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var offsetsBuf = device.Allocate((long)offsets.Length * sizeof(uint));
        using var yBuf = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(weightsF16, weightsBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), offsetsBuf);

        kernel.Launch(weightsBuf, xBuf, offsetsBuf, yBuf,
            outputDim, hidden, rows, numExperts);

        float[] actual = new float[expected.Length];
        device.Download(yBuf, actual);

        F16Bf16Fixture.AssertClose(expected, actual, outputDim, hidden, absTol: 5e-3f, relTol: 1e-3f);
    }

    private static uint[] ComputeOffsets(int[] indices, int numExperts)
    {
        var counts = new uint[numExperts];
        foreach (int idx in indices)
            counts[Math.Clamp(idx, 0, numExperts - 1)]++;

        var offsets = new uint[numExperts + 1];
        for (int i = 0; i < numExperts; i++)
            offsets[i + 1] = offsets[i] + counts[i];
        return offsets;
    }

    private static void AssertGroupedPermutationAndPackedRows(
        int[] indices, float[] x, float[] packed, uint[] permutation,
        uint[] offsets, int rows, int hidden, int numExperts)
    {
        var seen = new bool[rows];
        for (int expert = 0; expert < numExperts; expert++)
        {
            for (int packedRow = (int)offsets[expert]; packedRow < offsets[expert + 1]; packedRow++)
            {
                int sourceRow = checked((int)permutation[packedRow]);
                Assert.InRange(sourceRow, 0, rows - 1);
                Assert.False(seen[sourceRow]);
                seen[sourceRow] = true;
                Assert.Equal(expert, Math.Clamp(indices[sourceRow], 0, numExperts - 1));

                for (int col = 0; col < hidden; col++)
                    Assert.Equal(x[sourceRow * hidden + col], packed[packedRow * hidden + col]);
            }
        }

        Assert.All(seen, Assert.True);
    }

    private static unsafe float[] CpuGroupedMatmulF16(
        byte[] weightsF16, float[] x, uint[] offsets,
        int m, int k, int rows, int numExperts)
    {
        var y = new float[rows * m];
        fixed (byte* wBase = weightsF16)
        {
            for (int expert = 0; expert < numExperts; expert++)
            {
                int start = (int)offsets[expert];
                int end = (int)offsets[expert + 1];
                for (int row = start; row < end; row++)
                {
                    for (int outCol = 0; outCol < m; outCol++)
                    {
                        Half* w = (Half*)wBase + ((expert * m + outCol) * k);
                        float acc = 0f;
                        for (int col = 0; col < k; col++)
                            acc += (float)w[col] * x[row * k + col];
                        y[row * m + outCol] = acc;
                    }
                }
            }
        }

        return y;
    }

    private static void DownloadUInt32(VulkanDevice device, VulkanDevice.Buffer src, Span<uint> dst)
    {
        float[] tmp = new float[dst.Length];
        device.Download(src, tmp);
        MemoryMarshal.Cast<float, uint>(tmp.AsSpan()).CopyTo(dst);
    }
}
