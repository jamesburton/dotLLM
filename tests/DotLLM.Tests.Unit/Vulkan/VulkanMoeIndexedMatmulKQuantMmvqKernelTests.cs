using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity of the generic indexed K-quant MMVQ decode GEMV (<see cref="MoeIndexedMatmulKQuantMmvqKernel"/>, Q4_K / Q5_K / Q6_K) -- what
/// Qwen3.6-35B-A3B runs for its routed experts at decode. Two assertions per quant: (1) agreement with a CPU GEMM over the same packed
/// bytes within the int8-activation tolerance; (2) the batched launch equals n single-row launches BIT-EXACTLY, which discriminates a
/// dropped expert stride or every row taking <c>indices[0]</c>.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulKQuantMmvqKernelTests
{
    [SkippableTheory]
    [InlineData(MoeGroupedKQuant.Q4_K, 8, 16, 48, 512)]
    [InlineData(MoeGroupedKQuant.Q5_K, 8, 16, 96, 512)]
    [InlineData(MoeGroupedKQuant.Q5_K, 5, 7, 12, 768)]
    [InlineData(MoeGroupedKQuant.Q6_K, 8, 16, 96, 512)]
    [InlineData(MoeGroupedKQuant.Q6_K, 3, 4, 9, 1024)]
    public void MatchesCpuReferenceAndSingleRowLaunches(MoeGroupedKQuant quantType, int n, int numExperts, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulKQuantMmvqKernel.TryCreate(device, spvDir, quantType)
            ?? throw new Xunit.Sdk.XunitException("indexed MMVQ spv missing or unsupported.");

        var rng = new Random(0x640 + n * 31 + numExperts * 17 + m * 11 + k * 7 + (int)quantType);
        var expertBytes = new byte[numExperts][];
        for (int e = 0; e < numExperts; e++)
        {
            float[] w = Q4KFixture.RandomFloats(rng, m * k, 0.1f);
            expertBytes[e] = quantType switch
            {
                MoeGroupedKQuant.Q4_K => Q4KFixture.QuantizeRows(w, m, k),
                MoeGroupedKQuant.Q5_K => Q5KFixture.QuantizeRows(w, m, k),
                _ => Q6KFixture.QuantizeRows(w, m, k),
            };
        }
        byte[] bank = expertBytes.SelectMany(b => b).ToArray();
        float[] x = Q4KFixture.RandomFloats(rng, n * k, 1f);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = (i * 3 + 1) % numExperts;

        float[] expected = new float[(long)n * m];
        for (int r = 0; r < n; r++)
        {
            float[] xr = x.AsSpan(r * k, k).ToArray();
            byte[] eb = expertBytes[indices[r]];
            float[] yr = quantType switch
            {
                MoeGroupedKQuant.Q4_K => Q4KFixture.CpuGemmQ4K(eb, xr, m, k, 1),
                MoeGroupedKQuant.Q5_K => Q5KFixture.CpuGemmQ5K(eb, xr, m, k, 1),
                _ => Q6KFixture.CpuGemmQ6K(eb, xr, m, k, 1),
            };
            yr.CopyTo(expected, (long)r * m);
        }

        float[] batched = Launch(device, quant, kernel, bank, x, indices, m, k, n, numExperts);
        float maxAbs = expected.Max(MathF.Abs);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(MathF.Abs(expected[i] - batched[i]) <= 0.03f * maxAbs,
                $"{quantType} cell {i}: expected {expected[i]:G6} got {batched[i]:G6} (maxAbs {maxAbs:G6}).");

        for (int r = 0; r < n; r++)
        {
            float[] single = Launch(device, quant, kernel, bank, x.AsSpan(r * k, k).ToArray(), new[] { indices[r] }, m, k, 1, numExperts);
            for (int c = 0; c < m; c++)
                Assert.True(batched[r * m + c].Equals(single[c]), $"{quantType} row {r} col {c}: batched != single-row launch.");
        }
    }

    /// <summary>
    /// The decode MoE path quantizes ONE activation row and has every topK slot read it (<c>xDiv = topK</c>): bit-exact against replicating that
    /// row topK times and launching with <c>xDiv = 1</c>.
    /// </summary>
    [SkippableTheory]
    [InlineData(MoeGroupedKQuant.Q4_K, 8, 16, 48, 512)]
    [InlineData(MoeGroupedKQuant.Q5_K, 8, 16, 96, 512)]
    [InlineData(MoeGroupedKQuant.Q6_K, 8, 16, 96, 1024)]
    public void BroadcastActivationRow_MatchesReplicatedRows(MoeGroupedKQuant quantType, int topK, int numExperts, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulKQuantMmvqKernel.TryCreate(device, spvDir, quantType) ?? throw new Xunit.Sdk.XunitException("spv missing.");

        var rng = new Random(0x647 + (int)quantType);
        byte[] bank = Enumerable.Range(0, numExperts).SelectMany(_ =>
        {
            float[] w = Q4KFixture.RandomFloats(rng, m * k, 0.1f);
            return quantType switch
            {
                MoeGroupedKQuant.Q4_K => Q4KFixture.QuantizeRows(w, m, k),
                MoeGroupedKQuant.Q5_K => Q5KFixture.QuantizeRows(w, m, k),
                _ => Q6KFixture.QuantizeRows(w, m, k),
            };
        }).ToArray();
        float[] row = Q4KFixture.RandomFloats(rng, k, 1f);
        float[] replicated = Enumerable.Range(0, topK).SelectMany(_ => row).ToArray();
        int[] indices = Enumerable.Range(0, topK).Select(i => (i * 5 + 2) % numExperts).ToArray();

        float[] viaBroadcast = Launch(device, quant, kernel, bank, row, indices, m, k, topK, numExperts, xDiv: topK);
        float[] viaReplicated = Launch(device, quant, kernel, bank, replicated, indices, m, k, topK, numExperts, xDiv: 1);
        for (int i = 0; i < viaBroadcast.Length; i++)
            Assert.True(viaBroadcast[i].Equals(viaReplicated[i]), $"{quantType} cell {i}: broadcast {viaBroadcast[i]:G9} != replicated {viaReplicated[i]:G9}.");
    }

    private static float[] Launch(VulkanDevice device, QuantizeQ8_1RowsKernel quant, MoeIndexedMatmulKQuantMmvqKernel kernel,
        byte[] bank, float[] x, int[] indices, int m, int k, int n, int numExperts, int xDiv = 1)
    {
        int xRows = (n + xDiv - 1) / xDiv;
        using var bankBuf = device.Allocate(((long)bank.Length + 3) & ~3L);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(xRows, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(xRows, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(bank), bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, xBuf, xqBuf, xdsBuf, xRows, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kernel.Record(ctx.CommandBuffer, bankBuf, xqBuf, xdsBuf, idxBuf, yBuf, m, k, n, numExperts, xDiv);
            ctx.SubmitAndWait();
        }
        float[] y = new float[(long)n * m];
        device.Download(yBuf, y);
        return y;
    }
}
