using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #637: parity of the grouped Q4_K / Q5_K MoE coopmat kernels against a CPU GEMM over the same packed bytes. Ragged expert counts
/// (empty, smaller and larger than the 16-row tile) are the point: the row bound and the per-expert weight slab are what can go wrong.
/// </summary>
public sealed class VulkanMoeGroupedMatmulKQuantCoopmatTests
{
    [SkippableTheory]
    [InlineData(MoeGroupedKQuant.Q4_K, 48, 256, "3,0,17,40")]
    [InlineData(MoeGroupedKQuant.Q4_K, 33, 512, "5,7,1,0,20")]
    [InlineData(MoeGroupedKQuant.Q5_K, 48, 256, "3,0,17,40")]
    [InlineData(MoeGroupedKQuant.Q5_K, 64, 768, "16,15,17,1")]
    public void MatchesCpuReference(MoeGroupedKQuant quant, int m, int k, string counts)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulKQuantCoopmatKernel.IsSupportedOn(device, spvDir, quant), "coopmat or SPIR-V unavailable.");

        int[] expertRows = Array.ConvertAll(counts.Split(','), int.Parse);
        int numExperts = expertRows.Length;
        uint[] offsets = new uint[numExperts + 1];
        int rows = 0;
        for (int e = 0; e < numExperts; e++) { offsets[e] = (uint)rows; rows += expertRows[e]; }
        offsets[numExperts] = (uint)rows;

        var rng = new Random(0x637 + m * 31 + k + rows);
        var expertBytes = new byte[numExperts][];
        for (int e = 0; e < numExperts; e++)
        {
            float[] w = Q4KFixture.RandomFloats(rng, m * k, 0.5f);
            expertBytes[e] = quant == MoeGroupedKQuant.Q4_K ? Q4KFixture.QuantizeRows(w, m, k) : Q5KFixture.QuantizeRows(w, m, k);
        }
        byte[] bank = expertBytes.SelectMany(b => b).ToArray();
        float[] x = Q4KFixture.RandomFloats(rng, rows * k, 1f);

        float[] expected = new float[(long)rows * m];
        for (int e = 0; e < numExperts; e++)
        {
            int n = expertRows[e];
            if (n == 0) continue;
            float[] xe = x.AsSpan((int)offsets[e] * k, n * k).ToArray();
            float[] ye = quant == MoeGroupedKQuant.Q4_K
                ? Q4KFixture.CpuGemmQ4K(expertBytes[e], xe, m, k, n)
                : Q5KFixture.CpuGemmQ5K(expertBytes[e], xe, m, k, n);
            ye.CopyTo(expected, (long)offsets[e] * m);
        }

        using var kernel = MoeGroupedMatmulKQuantCoopmatKernel.Create(device, spvDir, quant);
        using var bufW = device.Allocate(bank.Length);
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufOff = device.Allocate(offsets.Length * sizeof(uint));
        using var bufY = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(bank, bufW);
        device.Upload(x, bufX);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(offsets), bufOff);
        kernel.Launch(bufW, bufX, bufOff, bufY, m, k, rows, numExperts, maxRowsPerExpert: expertRows.Max());

        float[] actual = new float[(long)rows * m];
        device.Download(bufY, actual);
        for (int i = 0; i < actual.Length; i++)
        {
            float tol = 2e-2f + 1e-2f * MathF.Abs(expected[i]);
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"{quant} idx {i} (row {i / m}, col {i % m}) counts={counts}: expected {expected[i]:G9}, got {actual[i]:G9}");
        }
    }
}
