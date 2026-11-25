using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Edge0 G3 (#1981): resident Q3_K MoE expert banks on Vulkan (indexed MMVQ for decode, grouped coopmat for prefill), read in place - no widening.
/// Q3_K has a bug history (#311: the dequant was wrong in every backend), so three oracles are used and one of them is independent of dotLLM:
/// (1) a scalar port of llama.cpp <c>dequantize_row_q3_K</c> written in this file ("F32-decoded control"), which must equal the production
/// <see cref="Dequantize.ToFloat32"/> on random blocks; (2) the kernel against that decoder dotted (double) with the kernel's OWN Q8_1 activations
/// (same tier: only fp32 accumulation order differs); (3) the kernel against the unquantised activations (the int8 activation error only).
/// Banks are RANDOM BLOCK BYTES with just the fp16 <c>d</c> (offset 108) forced sane, so every hmask/qs/scale bit feeds the result.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQ3KMoeBankTests
{
    private const int BlockBytes = 110;

    private static byte[] RandomBank(Random rng, long rows, int k)
    {
        int rowBytes = (k / 256) * BlockBytes;
        var bytes = new byte[rows * rowBytes];
        rng.NextBytes(bytes);
        for (long off = 0; off + BlockBytes <= bytes.Length; off += BlockBytes)
        {
            ushort bits = BitConverter.HalfToUInt16Bits((Half)(0.002f + (float)rng.NextDouble() * 0.02f));
            bytes[off + 108] = (byte)bits;
            bytes[off + 109] = (byte)(bits >> 8);
        }
        return bytes;
    }

    /// <summary>Scalar port of llama.cpp <c>dequantize_row_q3_K</c> (ggml-quants.c): independent of <see cref="Dequantize"/>.</summary>
    internal static float[] LlamaCppDecodeRow(ReadOnlySpan<byte> row, int k)
    {
        var y = new float[k];
        int o = 0;
        for (int b = 0; b < k / 256; b++)
        {
            var blk = row.Slice(b * BlockBytes, BlockBytes);
            ReadOnlySpan<byte> hm = blk.Slice(0, 32), qs = blk.Slice(32, 64), sc = blk.Slice(96, 12);
            float dAll = (float)BitConverter.UInt16BitsToHalf((ushort)(blk[108] | (blk[109] << 8)));
            uint a0 = BitConverter.ToUInt32(sc.Slice(0, 4)), a1 = BitConverter.ToUInt32(sc.Slice(4, 4)),
                 a2 = BitConverter.ToUInt32(sc.Slice(8, 4));
            const uint k1 = 0x03030303u, k2 = 0x0f0f0f0fu;
            uint tmp = a2;
            uint s2 = ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            uint s3 = ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            uint s0 = (a0 & k2) | (((tmp >> 0) & k1) << 4);
            uint s1 = (a1 & k2) | (((tmp >> 2) & k1) << 4);
            var scales = new sbyte[16];
            uint[] aux = { s0, s1, s2, s3 };
            for (int i = 0; i < 16; i++) scales[i] = unchecked((sbyte)((aux[i / 4] >> (8 * (i % 4))) & 0xFF));
            int isc = 0, qoff = 0;
            byte m = 1;
            for (int n = 0; n < 256; n += 128)
            {
                int shift = 0;
                for (int j = 0; j < 4; j++)
                {
                    float dl = dAll * (scales[isc++] - 32);
                    for (int l = 0; l < 16; l++) y[o++] = dl * (((qs[qoff + l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4));
                    dl = dAll * (scales[isc++] - 32);
                    for (int l = 0; l < 16; l++) y[o++] = dl * (((qs[qoff + l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4));
                    shift += 2;
                    m <<= 1;
                }
                qoff += 32;
            }
        }
        return y;
    }

    [Fact]
    public unsafe void ProductionDequant_EqualsIndependentLlamaCppDecoder()
    {
        var rng = new Random(0x3311);
        const int k = 1792, rows = 24;
        byte[] bank = RandomBank(rng, rows, k);
        int rowBytes = (k / 256) * BlockBytes;
        var w = new float[k];
        fixed (byte* p = bank)
            for (int r = 0; r < rows; r++)
            {
                Dequantize.ToFloat32((nint)(p + (long)r * rowBytes), k, QuantizationType.Q3_K, w);
                float[] indep = LlamaCppDecodeRow(bank.AsSpan(r * rowBytes, rowBytes), k);
                for (int i = 0; i < k; i++)
                    Assert.True(w[i] == indep[i], $"row {r} elem {i}: production {w[i]:G9} vs llama.cpp decode {indep[i]:G9}");
            }
    }

    private static double[] RefDot(byte[] bank, int[] indices, int m, int k, int n, int xDiv, Func<int, int, double> x)
    {
        int rowBytes = (k / 256) * BlockBytes;
        var y = new double[(long)n * m];
        var cache = new Dictionary<(int, int), float[]>();
        for (int r = 0; r < n; r++)
            for (int c = 0; c < m; c++)
            {
                if (!cache.TryGetValue((indices[r], c), out var w))
                    cache[(indices[r], c)] = w = LlamaCppDecodeRow(bank.AsSpan((int)(((long)indices[r] * m + c) * rowBytes), rowBytes), k);
                double acc = 0;
                for (int i = 0; i < k; i++) acc += (double)w[i] * x(r / xDiv, i);
                y[(long)r * m + c] = acc;
            }
        return y;
    }

    [SkippableTheory]
    [InlineData(5, 7, 12, 256, 4)]          // 1 super-block per row, n != m != E
    [InlineData(6, 8, 12, 768, 5)]          // 3 super-blocks
    [InlineData(10, 64, 40, 1792, 10)]      // Edge0 gate/up K = 1792 (7 super-blocks), top-10, 64 of the 512 experts
    [InlineData(10, 64, 48, 1792, 3)]       // few active experts: repeated expert rows
    public void IndexedMmvq_MatchesIndependentDecoder_SameTierAndF32Control(int n, int numExperts, int m, int k, int active)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, MoeIqQuant.Q3_K)
            ?? throw new Xunit.Sdk.XunitException(MoeIndexedMatmulIqMmvqKernel.ShaderName(MoeIqQuant.Q3_K) + ".spv missing or unsupported.");

        var rng = new Random(0x3312 + n * 31 + numExperts * 17 + m * 11 + k * 7);
        byte[] bank = RandomBank(rng, (long)numExperts * m, k);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        int[] indices = MoeMmvqParitySupport.RandomIndices(rng, n, numExperts, active);

        // The shared IQ launcher is bound to the Q3_K kernel through the common MoeIndexedMatmulIqMmvqKernel type.
        float[] actual = VulkanMoeIndexedMatmulIqMmvqKernelTests.Launch(device, quant, kernel, bank, x, indices, m, k, n, numExperts,
            out sbyte[] xq, out float[] xds);

        // (2) same tier: independent decoder x the kernel's own int8 activations.
        int bpr = k / 32;
        double[] sameTier = RefDot(bank, indices, m, k, n, 1, (r, i) => (double)xq[(long)r * k + i] * xds[(r * bpr + i / 32) * 2]);
        MoeMmvqParitySupport.AssertSameTierParity(Array.ConvertAll(sameTier, d => (float)d), actual, m, n, "Q3_K indexed MMVQ vs llama.cpp decoder");

        // (3) F32 control: unquantised activations; only the int8 activation error remains.
        double[] control = RefDot(bank, indices, m, k, n, 1, (r, i) => x[(long)r * k + i]);
        double ss = 0, es = 0;
        for (int i = 0; i < control.Length; i++) { ss += control[i] * control[i]; double d = control[i] - actual[i]; es += d * d; }
        double rel = Math.Sqrt(es / ss);
        Assert.True(rel < 0.03, $"Q3_K MMVQ vs F32-decoded control: relative RMS error {rel:G4} (int8 activations alone should stay below ~1%).");
    }

    [SkippableFact]
    public void IndexedMmvq_PerRowExpertIndex_MatchesSingleRowLaunchesBitExactly()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulIqMmvqKernel.TryCreate(device, spvDir, MoeIqQuant.Q3_K) ?? throw new Xunit.Sdk.XunitException("spv missing.");
        const int n = 9, numExperts = 16, m = 20, k = 1792;
        var rng = new Random(0x3313);
        byte[] bank = RandomBank(rng, (long)numExperts * m, k);
        float[] x = new float[n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = (i * 3 + 1) % numExperts;
        float[] batched = VulkanMoeIndexedMatmulIqMmvqKernelTests.Launch(device, quant, kernel, bank, x, indices, m, k, n, numExperts, out _, out _);
        for (int row = 0; row < n; row++)
        {
            float[] single = VulkanMoeIndexedMatmulIqMmvqKernelTests.Launch(device, quant, kernel, bank, x.AsSpan(row * k, k).ToArray(),
                new[] { indices[row] }, m, k, 1, numExperts, out _, out _);
            for (int c = 0; c < m; c++)
                Assert.True(batched[row * m + c].Equals(single[c]), $"row {row} (expert {indices[row]}) col {c}: batched != single-row launch.");
        }
    }

    [SkippableTheory]
    [InlineData(100, 512, "5,0,1,40,17,0,3")]
    [InlineData(130, 1792, "0,3,0,17,40,16,15,1,33,64,2,0")]    // Edge0 gate/up K = 1792 (7 super-blocks), m not a multiple of 64
    [InlineData(64, 256, "16,15,17,1")]
    public void Grouped_MatchesIndependentDecoder_LegacyGridAndIndirectBitIdentical(int m, int k, string countsCsv)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulIqCoopmatKernel.IsSupportedOn(device, spvDir, MoeIqQuant.Q3_K) && MoeBuildTileListKernel.IsSupportedOn(spvDir),
            "coopmat / native wave64 / SPIR-V unavailable.");
        int[] counts = Array.ConvertAll(countsCsv.Split(','), int.Parse);
        int E = counts.Length, rows = counts.Sum();
        uint[] offsets = new uint[E + 1];
        int[] expertOfRow = new int[rows];
        for (int e = 0, r = 0; e < E; e++)
        {
            offsets[e + 1] = offsets[e] + (uint)counts[e];
            for (int i = 0; i < counts[e]; i++) expertOfRow[r++] = e;
        }
        var rng = new Random(0x3314 + m * 31 + k + rows);
        byte[] bank = RandomBank(rng, (long)E * m, k);
        float[] x = new float[(long)rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        using var bufW = device.Allocate((bank.Length + 3L) & ~3L);
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufOff = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
        using var bufArgs = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        using var yLegacy = device.Allocate((long)rows * m * sizeof(float));
        using var yIndirect = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(bank, bufW);
        device.Upload(x, bufX);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bufOff);
        using var kernel = MoeGroupedMatmulIqCoopmatKernel.Create(device, spvDir, MoeIqQuant.Q3_K);
        using var build = MoeBuildTileListKernel.Create(device, spvDir);
        kernel.Launch(bufW, bufX, bufOff, yLegacy, m, k, rows, E, maxRowsPerExpert: counts.Max());
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            build.Record(ctx.CommandBuffer, bufOff, bufArgs, E, kernel.MTiles(m), kernel.MTiles(m), kernel.RowTile);
            KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
            kernel.RecordIndirect(ctx.CommandBuffer, bufW, bufX, bufOff, yIndirect, bufArgs, 0, m, k, rows, E);
            ctx.SubmitAndWait();
        }
        float[] a = new float[(long)rows * m], b = new float[(long)rows * m];
        device.Download(yLegacy, a);
        device.Download(yIndirect, b);

        double[] r0 = RefDot(bank, expertOfRow, m, k, rows, 1, (r, i) => x[(long)r * k + i]);
        float maxAbs = (float)r0.Max(Math.Abs);
        Assert.True(maxAbs > 1e-3f, "reference output is degenerate (all ~0): the comparison would be vacuous.");
        float tol = 6e-3f * maxAbs;
        for (int i = 0; i < r0.Length; i++)
        {
            Assert.True(a[i] == b[i], $"idx {i} (row {i / m}, col {i % m}): legacy grid {a[i]:G9} != indirect {b[i]:G9}");
            Assert.True(MathF.Abs(a[i] - (float)r0[i]) <= tol,
                $"idx {i} (row {i / m}, col {i % m}, expert {expertOfRow[i / m]}): grouped {a[i]:G9} vs llama.cpp-decoder {r0[i]:G9} (tol {tol:G4}, maxAbs {maxAbs:G4})");
        }
    }
}
