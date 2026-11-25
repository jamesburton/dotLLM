using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0 G3 (#1981): the record-STRIDED bank kernels read an array of fixed-size expert records (gate Q3_K | up Q3_K | down IQ4_NL, the pack's layout) in place.
/// The oracle is the CONTIGUOUS kernel of the same format run on banks split out of the same bytes: same shaders but for the row base, so the results must be
/// BIT-IDENTICAL (indexed and grouped). The records have the real geometry (K 1792, ff 640: 1,630,720 B) with distinct random bytes per expert and projection, so a
/// dropped projection offset, a dropped expert stride, or a stride taken from the projection size instead of the record each read different weights.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanStridedBankKernelTests
{
    private const int L = 1792, F = 640, E = 24;
    private static readonly long GateBytes = (long)F * (L / 256) * 110, DownBytes = (long)L * (F / 32) * 18, Record = 2 * GateBytes + DownBytes;

    private static byte[] Records(Random rng)
    {
        var b = new byte[E * Record];
        rng.NextBytes(b);
        for (int e = 0; e < E; e++)
        {
            long r0 = e * Record;
            for (int p = 0; p < 2; p++)        // Q3_K: fp16 d at offset 108 of each 110-byte block
                for (long o = r0 + p * GateBytes; o + 110 <= r0 + (p + 1) * GateBytes; o += 110) SetHalf(b, o + 108, rng);
            for (long o = r0 + 2 * GateBytes; o + 18 <= r0 + Record; o += 18) SetHalf(b, o, rng);   // IQ4_NL: fp16 d first
        }
        return b;
    }

    private static void SetHalf(byte[] b, long o, Random rng)
    {
        ushort bits = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.02f));
        b[o] = (byte)bits; b[o + 1] = (byte)(bits >> 8);
    }

    private static byte[] Split(byte[] rec, long offset, long bytes)
    {
        var o = new byte[E * bytes];
        for (int e = 0; e < E; e++) Buffer.BlockCopy(rec, (int)(e * Record + offset), o, (int)(e * bytes), (int)bytes);
        return o;
    }

    [SkippableTheory]
    [InlineData(MoeIqQuant.Q3_K, 0, 5, 3)]
    [InlineData(MoeIqQuant.Q3_K, 1, 10, 10)]      // up projection: base offset = one gate projection
    [InlineData(MoeIqQuant.IQ4_NL, 2, 10, 10)]    // down projection: base offset = two gate projections
    [InlineData(MoeIqQuant.IQ4_NL, 2, 3, 3)]
    public void IndexedStrided_IsBitIdenticalToTheContiguousKernelOnSplitBanks(MoeIqQuant quant, int proj, int n, int active)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var q8 = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var strided = MoeIndexedStridedMmvqKernel.TryCreate(device, spvDir, quant) ?? throw new Xunit.Sdk.XunitException("strided spv missing.");
        using var plain = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, quant) ?? throw new Xunit.Sdk.XunitException("contiguous spv missing.");
        bool down = proj == 2;
        int m = down ? L : F, k = down ? F : L;
        long off = proj * GateBytes, bytes = down ? DownBytes : GateBytes;
        var rng = new Random(0x5DE0 + proj * 31 + n);
        byte[] rec = Records(rng);
        byte[] bankSplit = Split(rec, off, bytes);
        float[] x = new float[n * k]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] idx = MoeMmvqParitySupport.RandomIndices(rng, n, E, active);

        using var bRec = device.Allocate((rec.Length + 3L) & ~3L); using var bSplit = device.Allocate((bankSplit.Length + 3L) & ~3L);
        using var bX = device.Allocate(x.Length * 4L); using var bXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k)); using var bXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bIdx = device.Allocate(n * 4L); using var yA = device.Allocate((long)n * m * 4); using var yB = device.Allocate((long)n * m * 4);
        device.Upload(rec, bRec); device.Upload(bankSplit, bSplit); device.Upload(x, bX); device.Upload(MemoryMarshal.AsBytes<int>(idx), bIdx);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            q8.Record(ctx.CommandBuffer, bX, bXq, bXds, n, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            strided.Record(ctx.CommandBuffer, bRec, bXq, bXds, bIdx, yA, m, k, n, E, Record, off);
            plain.Record(ctx.CommandBuffer, bSplit, bXq, bXds, bIdx, yB, m, k, n, E);
            ctx.SubmitAndWait();
        }
        float[] a = new float[n * m], b = new float[n * m]; device.Download(yA, a); device.Download(yB, b);
        Assert.True(b.Max(MathF.Abs) > 1e-3f, "reference degenerate");
        for (int i = 0; i < a.Length; i++) Assert.True(a[i].Equals(b[i]), $"{quant} proj {proj} idx {i}: strided {a[i]:G9} vs contiguous {b[i]:G9}");
    }

    [SkippableTheory]
    [InlineData(MoeIqQuant.Q3_K, 0, "5,0,1,40,17,0,3,16,15,17,33,64,2,1")]
    [InlineData(MoeIqQuant.Q3_K, 1, "5,0,1,40,17,0,3,16,15,17,33,64,2,1")]
    [InlineData(MoeIqQuant.IQ4_NL, 2, "5,0,1,40,17,0,3,16,15,17,33,64,2,1")]
    public void GroupedStrided_IsBitIdenticalToTheContiguousKernelOnSplitBanks(MoeIqQuant quant, int proj, string countsCsv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedStridedCoopmatKernel.IsSupportedOn(device, spvDir, quant) && MoeGroupedMatmulIqCoopmatKernel.IsSupportedOn(device, spvDir, quant) && MoeBuildTileListKernel.IsSupportedOn(spvDir),
            "coopmat / native wave64 / SPIR-V unavailable.");
        int[] counts = Array.ConvertAll(countsCsv.Split(','), int.Parse);
        Assert.Equal(E - 10, counts.Length);   // counts cover experts 0..13; the rest are empty
        int[] full = new int[E]; Array.Copy(counts, full, counts.Length);
        int rows = full.Sum();
        uint[] offsets = new uint[E + 1];
        for (int e = 0; e < E; e++) offsets[e + 1] = offsets[e] + (uint)full[e];
        bool down = proj == 2;
        int m = down ? L : F, k = down ? F : L;
        long off = proj * GateBytes, bytes = down ? DownBytes : GateBytes;
        var rng = new Random(0x5DE1 + proj);
        byte[] rec = Records(rng);
        byte[] bankSplit = Split(rec, off, bytes);
        float[] x = new float[(long)rows * k]; for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        using var bRec = device.Allocate((rec.Length + 3L) & ~3L); using var bSplit = device.Allocate((bankSplit.Length + 3L) & ~3L);
        using var bX = device.Allocate(x.Length * 4L);
        using var bOff = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint)); using var bOff2 = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
        using var bArgs = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes); using var bArgs2 = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        using var yLegacy = device.Allocate((long)rows * m * 4); using var yInd = device.Allocate((long)rows * m * 4); using var yPlain = device.Allocate((long)rows * m * 4);
        device.Upload(rec, bRec); device.Upload(bankSplit, bSplit); device.Upload(x, bX);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bOff); device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bOff2);
        using var strided = MoeGroupedStridedCoopmatKernel.Create(device, spvDir, quant);
        using var plain = MoeGroupedMatmulIqCoopmatKernel.Create(device, spvDir, quant);
        using var build = MoeBuildTileListKernel.Create(device, spvDir);
        strided.Launch(bRec, bX, bOff, yLegacy, m, k, rows, E, Record, off, maxRowsPerExpert: full.Max());
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            build.Record(ctx.CommandBuffer, bOff, bArgs, E, strided.MTiles(m), strided.MTiles(m), strided.RowTile);
            build.Record(ctx.CommandBuffer, bOff2, bArgs2, E, plain.MTiles(m), plain.MTiles(m), plain.RowTile);
            KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
            strided.RecordIndirect(ctx.CommandBuffer, bRec, bX, bOff, yInd, bArgs, 0, m, k, rows, E, Record, off);
            plain.RecordIndirect(ctx.CommandBuffer, bSplit, bX, bOff2, yPlain, bArgs2, 0, m, k, rows, E);
            ctx.SubmitAndWait();
        }
        float[] a = new float[(long)rows * m], b = new float[(long)rows * m], c = new float[(long)rows * m];
        device.Download(yLegacy, a); device.Download(yInd, b); device.Download(yPlain, c);
        Assert.True(c.Max(MathF.Abs) > 1e-3f, "reference degenerate");
        for (int i = 0; i < c.Length; i++)
        {
            Assert.True(b[i].Equals(c[i]), $"{quant} proj {proj} idx {i} (row {i / m}, col {i % m}): strided indirect {b[i]:G9} vs contiguous {c[i]:G9}");
            Assert.True(a[i].Equals(c[i]), $"{quant} proj {proj} idx {i}: strided legacy grid {a[i]:G9} vs contiguous {c[i]:G9}");
        }
    }
}
