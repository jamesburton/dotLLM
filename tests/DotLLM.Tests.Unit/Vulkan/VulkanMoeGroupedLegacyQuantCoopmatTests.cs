using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #773: grouped-by-expert coopmat GEMMs for the packed Q5_1 / Q8_0 expert banks (Gemma-4 <c>ffn_down_exps</c>). Each case runs the
/// grouped kernel (legacy grid AND indirect tile list) and the SAME-GPU scalar indexed kernel on identical random banks and compares them.
/// The two differ only in operand precision (F16 dequantised A / activations vs F32), so the envelope is a small multiple of 2^-11 of the
/// accumulated magnitude; a broadcast / addressing / nibble-plane / scale bug is O(1). Shapes are discriminating: E != topK-style
/// degenerate sizes, per-expert scales all different, experts with 0, 1, 15, 16, 17, 33 and 64 rows, m not a multiple of the 64-row tile.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeGroupedLegacyQuantCoopmatTests
{
    private static byte[] RandomBank(Random rng, MoeGroupedLegacyQuant quant, int experts, int m, int k)
    {
        int blocksPerRow = k / 32;
        int blockBytes = quant == MoeGroupedLegacyQuant.Q5_1 ? 24 : 34;
        byte[] bank = new byte[(long)experts * m * blocksPerRow * blockBytes];
        rng.NextBytes(bank);
        // Overwrite the fp16 header fields with sane magnitudes (random bytes would produce inf/NaN halves).
        for (long off = 0; off < bank.Length; off += blockBytes)
        {
            WriteHalf(bank, off, (float)(rng.NextDouble() * 0.04 + 0.002));
            if (quant == MoeGroupedLegacyQuant.Q5_1) WriteHalf(bank, off + 2, (float)(rng.NextDouble() * 0.2 - 0.1));
        }
        return bank;
    }

    private static void WriteHalf(byte[] dst, long off, float v)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)v);
        dst[off] = (byte)bits;
        dst[off + 1] = (byte)(bits >> 8);
    }

    [SkippableTheory]
    [InlineData(MoeGroupedLegacyQuant.Q5_1, 128, 704, "0,3,0,17,40,16,15,1,33,64,2,0", true)]
    [InlineData(MoeGroupedLegacyQuant.Q5_1, 100, 192, "5,0,1,40", true)]        // m not a multiple of 64: invalid-row path
    [InlineData(MoeGroupedLegacyQuant.Q8_0, 128, 704, "0,3,0,17,40,16,15,1,33,64,2,0", false)]
    [InlineData(MoeGroupedLegacyQuant.Q8_0, 100, 192, "5,0,1,40", false)]
    [InlineData(MoeGroupedLegacyQuant.Q8_0, 64, 64, "16,15,17,1", false)]       // single staging round
    [InlineData(MoeGroupedLegacyQuant.Q5_1, 100, 640, "5,0,1,40,17,0,3", false)] // #849: K = 640 (not a multiple of 256), scale off as the resident MoE path runs it
    [InlineData(MoeGroupedLegacyQuant.Q8_0, 100, 640, "5,0,1,40,17,0,3", false)]
    public void Grouped_MatchesScalarIndexed(MoeGroupedLegacyQuant quant, int m, int k, string countsCsv, bool useScale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulLegacyQuantCoopmatKernel.IsSupportedOn(device, spvDir, quant) && MoeBuildTileListKernel.IsSupportedOn(spvDir),
            "coopmat / wave64 / SPIR-V unavailable.");

        int[] counts = Array.ConvertAll(countsCsv.Split(','), int.Parse);
        int E = counts.Length, rows = counts.Sum();
        uint[] offsets = new uint[E + 1];
        int[] expertOfRow = new int[rows];
        for (int e = 0, r = 0; e < E; e++)
        {
            offsets[e + 1] = offsets[e] + (uint)counts[e];
            for (int i = 0; i < counts[e]; i++) expertOfRow[r++] = e;
        }

        var rng = new Random(0x773 + m * 31 + k + rows);
        byte[] bank = RandomBank(rng, quant, E, m, k);
        float[] x = new float[(long)rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] scale = new float[E];
        for (int e = 0; e < E; e++) scale[e] = useScale ? 0.25f + 0.37f * e : 1f;

        using var bufW = device.Allocate(bank.Length);
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufOff = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
        using var bufArgs = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        using var bufIdx = device.Allocate((long)rows * sizeof(int));
        using var bufScale = device.Allocate((long)E * sizeof(float));
        using var yRef = device.Allocate((long)rows * m * sizeof(float));
        using var yLegacy = device.Allocate((long)rows * m * sizeof(float));
        using var yIndirect = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(bank, bufW);
        device.Upload(x, bufX);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bufOff);
        device.Upload(MemoryMarshal.AsBytes<int>(expertOfRow), bufIdx);
        device.Upload(scale, bufScale);

        // Reference: the production scalar indexed kernels (Q5_1 folds the per-expert scale; Q8_0 has none).
        if (quant == MoeGroupedLegacyQuant.Q5_1)
        {
            using var refKernel = MoeIndexedMatmulQ5_1F32Kernel.Create(device, spvDir);
            refKernel.Launch(bufW, bufX, bufIdx, yRef, bufScale, m, k, rows, E);
        }
        else
        {
            using var refKernel = MoeIndexedMatmulQ8_0F32Kernel.Create(device, spvDir);
            refKernel.Launch(bufW, bufX, bufIdx, yRef, m, k, rows, E);
        }

        using var kernel = MoeGroupedMatmulLegacyQuantCoopmatKernel.Create(device, spvDir, quant);
        using var build = MoeBuildTileListKernel.Create(device, spvDir);
        kernel.Launch(bufW, bufX, bufOff, yLegacy, bufScale, useScale, m, k, rows, E, maxRowsPerExpert: counts.Max());
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            build.Record(ctx.CommandBuffer, bufOff, bufArgs, E, kernel.MTiles(m), kernel.MTiles(m), kernel.RowTile);
            KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
            kernel.RecordIndirect(ctx.CommandBuffer, bufW, bufX, bufOff, yIndirect, bufScale, useScale, bufArgs, 0, m, k, rows, E);
            ctx.SubmitAndWait();
        }

        float[] r0 = new float[(long)rows * m], a = new float[(long)rows * m], b = new float[(long)rows * m];
        device.Download(yRef, r0);
        device.Download(yLegacy, a);
        device.Download(yIndirect, b);

        float maxAbs = r0.Max(MathF.Abs);
        Assert.True(maxAbs > 1e-3f, "reference output is degenerate (all ~0): the comparison would be vacuous.");
        float tol = 6e-3f * maxAbs;
        float worst = 0;
        for (int i = 0; i < r0.Length; i++)
        {
            Assert.True(a[i] == b[i], $"{quant} idx {i} (row {i / m}, col {i % m}): legacy grid {a[i]:G9} != indirect {b[i]:G9}");
            float d = MathF.Abs(a[i] - r0[i]);
            worst = MathF.Max(worst, d);
            Assert.True(d <= tol, $"{quant} idx {i} (row {i / m}, col {i % m}, expert {expertOfRow[i / m]}): grouped {a[i]:G9} vs scalar {r0[i]:G9} (tol {tol:G4}, maxAbs {maxAbs:G4})");
        }
        Assert.True(worst > 0 || r0.Length == 0, "bit-identical to the F32 scalar path is implausible with F16 operands; the grouped kernel may not have run.");
    }
}
