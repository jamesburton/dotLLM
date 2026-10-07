using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #789: grouped-by-expert coopmat GEMM for packed MXFP4 expert banks (gpt-oss), with the optional per-expert BIAS epilogue. Each case
/// runs the grouped kernel (legacy grid AND indirect tile list) and the SAME-GPU scalar indexed MXFP4 kernel (+ a host-side per-expert bias add,
/// exactly what the scalar path's separate expert-bias pass computes) on identical random banks. They differ only in operand precision (F16
/// dequantised A / activations vs F32), so the envelope is a small multiple of 2^-11 of the accumulated magnitude; a nibble-plane / scale / addressing /
/// bias-indexing bug is O(1). Expert row counts cover 0, 1, 15, 16, 17, 33 and 64 (tile boundaries), m is not always a multiple of the 64-row tile.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeGroupedMxfp4CoopmatTests
{
    private static byte[] RandomBank(Random rng, int experts, int m, int k)
    {
        int blocksPerRow = k / 32;
        byte[] bank = new byte[(long)experts * m * blocksPerRow * 17];
        rng.NextBytes(bank);
        // Byte 0 of every 17-byte block is the E8M0 scale: keep it in a sane window (2^-9 .. 2^-6 after the halving) instead of 2^+127 garbage.
        for (long off = 0; off < bank.Length; off += 17) bank[off] = (byte)(rng.Next(119, 123));
        return bank;
    }

    [SkippableTheory]
    [InlineData(128, 704, "0,3,0,17,40,16,15,1,33,64,2,0", false)]
    [InlineData(128, 704, "0,3,0,17,40,16,15,1,33,64,2,0", true)]
    [InlineData(100, 192, "5,0,1,40", true)]          // m not a multiple of 64: invalid-row path (bias must not index past m)
    [InlineData(64, 64, "16,15,17,1", true)]          // single staging round
    [InlineData(128, 2880, "20,0,1,16,33", true)]     // gpt-oss hidden size: 45 staging rounds, 17-byte blocks at every alignment
    public void Grouped_MatchesScalarIndexed(int m, int k, string countsCsv, bool useBias)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulLegacyQuantCoopmatKernel.IsSupportedOn(device, spvDir, MoeGroupedLegacyQuant.Mxfp4) && MoeBuildTileListKernel.IsSupportedOn(spvDir),
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

        var rng = new Random(0x789 + m * 31 + k + rows);
        byte[] bank = RandomBank(rng, E, m, k);
        float[] x = new float[(long)rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] bias = new float[(long)E * m];
        for (int i = 0; i < bias.Length; i++) bias[i] = useBias ? (float)(rng.NextDouble() * 4 - 2) + 0.5f * (i / m) : 0f;

        using var bufW = device.Allocate(bank.Length);
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufOff = device.Allocate(MoeBuildTileListKernel.OffsetsBufferUints(E, rows) * sizeof(uint));
        using var bufArgs = device.Allocate(2 * MoeBuildTileListKernel.ArgsStrideBytes);
        using var bufIdx = device.Allocate((long)rows * sizeof(int));
        using var bufBias = device.Allocate((long)E * m * sizeof(float));
        using var yRef = device.Allocate((long)rows * m * sizeof(float));
        using var yLegacy = device.Allocate((long)rows * m * sizeof(float));
        using var yIndirect = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(bank, bufW);
        device.Upload(x, bufX);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bufOff);
        device.Upload(MemoryMarshal.AsBytes<int>(expertOfRow), bufIdx);
        device.Upload(bias, bufBias);

        using (var refKernel = MoeIndexedMatmulMxfp4F32Kernel.Create(device, spvDir))
            refKernel.Launch(bufW, bufX, bufIdx, yRef, m, k, rows, E);

        using var kernel = MoeGroupedMatmulLegacyQuantCoopmatKernel.Create(device, spvDir, MoeGroupedLegacyQuant.Mxfp4);
        using var build = MoeBuildTileListKernel.Create(device, spvDir);
        kernel.Launch(bufW, bufX, bufOff, yLegacy, bufBias, false, m, k, rows, E, maxRowsPerExpert: counts.Max(), applyBias: useBias);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            build.Record(ctx.CommandBuffer, bufOff, bufArgs, E, kernel.MTiles(m), kernel.MTiles(m), kernel.RowTile);
            KernelSupport.ComputeToIndirectAndComputeBarrier(ctx.CommandBuffer);
            kernel.RecordIndirect(ctx.CommandBuffer, bufW, bufX, bufOff, yIndirect, bufBias, false, bufArgs, 0, m, k, rows, E, applyBias: useBias);
            ctx.SubmitAndWait();
        }

        float[] r0 = new float[(long)rows * m], a = new float[(long)rows * m], b = new float[(long)rows * m];
        device.Download(yRef, r0);
        device.Download(yLegacy, a);
        device.Download(yIndirect, b);
        // Scalar path = matmul, then the expert-bias pass: y[r, :] += bias[expert(r), :].
        for (int i = 0; i < r0.Length; i++) r0[i] += bias[(long)expertOfRow[i / m] * m + i % m];

        float maxAbs = r0.Max(MathF.Abs);
        Assert.True(maxAbs > 1e-3f, "reference output is degenerate (all ~0): the comparison would be vacuous.");
        float tol = 6e-3f * maxAbs;
        float worst = 0;
        for (int i = 0; i < r0.Length; i++)
        {
            Assert.True(a[i] == b[i], $"idx {i} (row {i / m}, col {i % m}): legacy grid {a[i]:G9} != indirect {b[i]:G9}");
            float d = MathF.Abs(a[i] - r0[i]);
            worst = MathF.Max(worst, d);
            Assert.True(d <= tol, $"idx {i} (row {i / m}, col {i % m}, expert {expertOfRow[i / m]}): grouped {a[i]:G9} vs scalar {r0[i]:G9} (tol {tol:G4}, maxAbs {maxAbs:G4})");
        }
        Assert.True(worst > 0 || r0.Length == 0, "bit-identical to the F32 scalar path is implausible with F16 operands; the grouped kernel may not have run.");
    }

    /// <summary>Control: the bias epilogue must actually be what produces the bias (output differs from the no-bias run by exactly bias[expert, m]).</summary>
    [SkippableFact]
    public void BiasEpilogue_AddsExactlyPerExpertBias()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MoeGroupedMatmulLegacyQuantCoopmatKernel.IsSupportedOn(device, spvDir, MoeGroupedLegacyQuant.Mxfp4), "coopmat / wave64 / SPIR-V unavailable.");

        const int m = 100, k = 128;
        int[] counts = [0, 5, 17, 1];
        int E = counts.Length, rows = counts.Sum();
        uint[] offsets = new uint[E + 1];
        int[] expertOfRow = new int[rows];
        for (int e = 0, r = 0; e < E; e++)
        {
            offsets[e + 1] = offsets[e] + (uint)counts[e];
            for (int i = 0; i < counts[e]; i++) expertOfRow[r++] = e;
        }
        var rng = new Random(789);
        byte[] bank = RandomBank(rng, E, m, k);
        float[] x = new float[(long)rows * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] bias = new float[(long)E * m];
        for (int i = 0; i < bias.Length; i++) bias[i] = 8f * (i / m + 1) + (i % m) * 0.125f;   // exactly representable, expert- and column-dependent

        using var bufW = device.Allocate(bank.Length);
        using var bufX = device.Allocate((long)rows * k * sizeof(float));
        using var bufOff = device.Allocate((E + 1) * sizeof(uint));
        using var bufBias = device.Allocate((long)E * m * sizeof(float));
        using var y0 = device.Allocate((long)rows * m * sizeof(float));
        using var y1 = device.Allocate((long)rows * m * sizeof(float));
        device.Upload(bank, bufW);
        device.Upload(x, bufX);
        device.Upload(MemoryMarshal.AsBytes<uint>(offsets), bufOff);
        device.Upload(bias, bufBias);
        using var kernel = MoeGroupedMatmulLegacyQuantCoopmatKernel.Create(device, spvDir, MoeGroupedLegacyQuant.Mxfp4);
        kernel.Launch(bufW, bufX, bufOff, y0, bufBias, false, m, k, rows, E, maxRowsPerExpert: counts.Max(), applyBias: false);
        kernel.Launch(bufW, bufX, bufOff, y1, bufBias, false, m, k, rows, E, maxRowsPerExpert: counts.Max(), applyBias: true);
        float[] a = new float[(long)rows * m], b = new float[(long)rows * m];
        device.Download(y0, a);
        device.Download(y1, b);
        for (int i = 0; i < a.Length; i++)
            Assert.Equal(a[i] + bias[(long)expertOfRow[i / m] * m + i % m], b[i], 4);
    }
}
