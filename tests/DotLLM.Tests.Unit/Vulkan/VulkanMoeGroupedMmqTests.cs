using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1351: parity of the grouped dp4a MMQ MoE kernels (#986) against a CPU GEMM over the same packed bytes with UNQUANTIZED activations.
/// The kernel reads Q8_1 activations, so the expected gap is the 8-bit activation rounding (~0.5% relL2); a kernel or layout defect shows as
/// several percent. Ragged expert counts exercise the row bound and the per-expert weight slab for both row-tile variants.
/// </summary>
public sealed class VulkanMoeGroupedMmqTests
{
    [SkippableTheory]
    [InlineData(MoeGroupedKQuant.Q4_K, 32, 128, 256, "3,0,17,40", false)]
    [InlineData(MoeGroupedKQuant.Q4_K, 32, 128, 256, "3,0,17,40", true)]
    [InlineData(MoeGroupedKQuant.Q4_K, 32, 256, 512, "5,7,1,0,20", false)]
    [InlineData(MoeGroupedKQuant.Q4_K, 32, 256, 512, "5,7,1,0,20", true)]
    [InlineData(MoeGroupedKQuant.Q4_K, 64, 96, 256, "70,33,2", false)]
    [InlineData(MoeGroupedKQuant.Q4_K, 64, 96, 256, "70,33,2", true)]
    [InlineData(MoeGroupedKQuant.Q5_K, 32, 128, 256, "3,0,17,40", false)]
    [InlineData(MoeGroupedKQuant.Q5_K, 32, 128, 256, "3,0,17,40", true)]
    [InlineData(MoeGroupedKQuant.Q5_K, 64, 256, 768, "16,15,17,1", false)]
    [InlineData(MoeGroupedKQuant.Q5_K, 64, 256, 768, "16,15,17,1", true)]
    public void MatchesCpuReference(MoeGroupedKQuant quant, int tileN, int m, int k, string counts, bool indirect)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulKQuantMmqKernel.IsSupportedOn(device, spvDir, quant), "MMQ SPIR-V unavailable.");
        using var quantRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir);
        Skip.If(quantRows is null, "quantize_q8_1_rows.spv unavailable.");
        using var kernel = MoeGroupedMatmulKQuantMmqKernel.Create(device, spvDir, quant);
        Skip.IfNot(kernel.IsAdaptive || kernel.SelectRowTile(1) == tileN, $"{tileN}-row variant not built.");

        int[] expertRows = Array.ConvertAll(counts.Split(','), int.Parse);
        int numExperts = expertRows.Length;
        uint[] offsets = new uint[numExperts + 1];
        int rows = 0;
        for (int e = 0; e < numExperts; e++) { offsets[e] = (uint)rows; rows += expertRows[e]; }
        offsets[numExperts] = (uint)rows;

        var rng = new Random(0x1351 + m * 31 + k + rows);
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
            float[] ye = quant == MoeGroupedKQuant.Q4_K ? Q4KFixture.CpuGemmQ4K(expertBytes[e], xe, m, k, n) : Q5KFixture.CpuGemmQ5K(expertBytes[e], xe, m, k, n);
            ye.CopyTo(expected, (long)offsets[e] * m);
        }

        using var bufW = device.Allocate(bank.Length);
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(rows, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(rows, k));
        using var bufOff = device.Allocate(Math.Max(offsets.Length, MoeBuildTileListKernel.OffsetsBufferUints(numExperts, rows)) * sizeof(uint));
        using var bufArgs = device.Allocate(64);
        using var bufY = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(bank, bufW);
        device.Upload(x, bufX);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes<uint>(offsets), bufOff);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quantRows!.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, rows, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            if (indirect)
            {
                using var build = MoeBuildTileListKernel.Create(device, spvDir);
                build.Record(ctx.CommandBuffer, bufOff, bufArgs, numExperts, kernel.MTiles(m), kernel.MTiles(m), tileRows: tileN);
                KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
                kernel.RecordIndirect(ctx.CommandBuffer, bufW, bufXq, bufXds, bufOff, bufY, bufArgs, 0, m, k, rows, numExperts, tileN);
                ctx.SubmitAndWait();
            }
            else
            {
                kernel.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufOff, bufY, m, k, rows, numExperts, tileN, maxRowsPerExpert: expertRows.Max());
                ctx.SubmitAndWait();
            }
        }

        float[] actual = new float[(long)rows * m];
        device.Download(bufY, actual);
        double num = 0, den = 0;
        for (int i = 0; i < actual.Length; i++) { double d = actual[i] - expected[i]; num += d * d; den += (double)expected[i] * expected[i]; }
        double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
        Assert.True(rel < 0.02, $"{quant} tile {tileN} counts={counts} indirect={indirect}: relL2 {rel:E3} vs unquantized-activation CPU GEMM");
    }
}
