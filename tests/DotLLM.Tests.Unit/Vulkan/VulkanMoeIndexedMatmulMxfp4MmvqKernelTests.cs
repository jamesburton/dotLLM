using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #912: parity of the indexed MXFP4 MMVQ decode GEMV (<see cref="MoeIndexedMatmulMxfp4MmvqKernel"/>, gpt-oss routed experts at decode).
/// Three oracles per case: (1) a host reference computed from the SAME Q8_1 activation bytes the kernel reads (downloaded after the quantize pass),
/// so the only difference is float association -- this pins the nibble/element mapping and the table-free |kvalue| decode exactly; (2) the
/// scalar F32 indexed kernel within the int8-activation tolerance; (3) bit-exact batched-vs-single-row and broadcast-vs-replicated launches,
/// which discriminate a dropped expert stride or every row taking <c>indices[0]</c>. K = 2880 (gpt-oss hidden, 90 blocks: not a multiple of the
/// 32 lanes) and a 17-byte block stride put weight blocks at every dword alignment.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeIndexedMatmulMxfp4MmvqKernelTests
{
    private static readonly int[] KValues = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

    private static byte[] RandomBank(Random rng, int experts, int m, int k)
    {
        int blocksPerRow = k / 32;
        byte[] bank = new byte[(long)experts * m * blocksPerRow * 17 + 16];   // slack: the shader's 5-dword block load runs past the last block
        rng.NextBytes(bank);
        for (long off = 0; off < (long)experts * m * blocksPerRow * 17; off += 17) bank[off] = (byte)rng.Next(119, 123);   // sane E8M0 window
        return bank;
    }

    [SkippableTheory]
    [InlineData(4, 32, 96, 2880, false)]
    [InlineData(4, 32, 96, 2880, true)]
    [InlineData(7, 8, 13, 96, true)]      // odd m / small k: 3 blocks only (29 idle lanes)
    [InlineData(1, 32, 64, 2880, false)]
    [InlineData(15, 32, 50, 1024, true)]  // 15 rows, k = 32 blocks exactly one lane window
    public void MatchesHostReferenceScalarKernelAndSingleRowLaunches(int n, int numExperts, int m, int k, bool useBias)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulMxfp4MmvqKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("moe_indexed_matmul_mxfp4_mmvq.spv missing.");

        var rng = new Random(0x912 + n * 31 + numExperts * 17 + m * 11 + k);
        byte[] bank = RandomBank(rng, numExperts, m, k);
        float[] x = new float[(long)n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = (i * 5 + 3) % numExperts;   // numExperts > n for the small cases: not every expert is hit
        float[] bias = new float[(long)numExperts * m];
        for (int i = 0; i < bias.Length; i++) bias[i] = useBias ? (float)(rng.NextDouble() * 4 - 2) : 0f;

        (float[] got, uint[] xq, Vector2[] xds) = Launch(device, quant, kernel, bank, x, indices, useBias ? bias : null, m, k, n, numExperts, 1);

        // (1) host reference on the kernel's own quantized activations.
        int blocksPerRow = k / 32;
        float[] hostRef = new float[(long)n * m];
        for (int r = 0; r < n; r++)
            for (int c = 0; c < m; c++)
            {
                long rowOff = ((long)indices[r] * m + c) * blocksPerRow * 17;
                double acc = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    long off = rowOff + b * 17L;
                    int e = bank[off];
                    double dW = Math.ScaleB(1.0, e - 128);   // kvalues are the doubled E2M1 values: 0.5 * 2^(e-127)
                    int sum = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        byte nib = bank[off + 1 + j];
                        int lo = KValues[nib & 0xF], hi = KValues[nib >> 4];
                        sum += lo * XqByte(xq, r, b * 32 + j, k) + hi * XqByte(xq, r, b * 32 + j + 16, k);
                    }
                    acc += dW * xds[r * blocksPerRow + b].X * sum;
                }
                hostRef[(long)r * m + c] = (float)acc + (useBias ? bias[(long)indices[r] * m + c] : 0f);
            }
        float maxAbs = hostRef.Max(MathF.Abs);
        Assert.True(maxAbs > 1e-3f, "reference is degenerate (all ~0): the comparison would be vacuous.");
        for (int i = 0; i < hostRef.Length; i++)
            Assert.True(MathF.Abs(hostRef[i] - got[i]) <= 2e-4f * maxAbs + 1e-5f,
                $"cell {i} (row {i / m}, col {i % m}, expert {indices[i / m]}): kernel {got[i]:G9} vs host-on-Q8_1 {hostRef[i]:G9} (maxAbs {maxAbs:G5}).");

        // (2) scalar F32 kernel (+ host bias): differs only by the int8 activation quantization.
        float[] scalar = RunScalar(device, spvDir, bank, x, indices, m, k, n, numExperts);
        for (int i = 0; i < scalar.Length; i++) scalar[i] += useBias ? bias[(long)indices[i / m] * m + i % m] : 0f;
        for (int i = 0; i < scalar.Length; i++)
            Assert.True(MathF.Abs(scalar[i] - got[i]) <= 0.03f * maxAbs, $"cell {i}: scalar F32 {scalar[i]:G6} vs mmvq {got[i]:G6} (maxAbs {maxAbs:G5}).");

        // (3a) batched == n single-row launches, bit-exact.
        for (int r = 0; r < n; r++)
        {
            var (single, _, _) = Launch(device, quant, kernel, bank, x.AsSpan(r * k, k).ToArray(), [indices[r]], useBias ? bias : null, m, k, 1, numExperts, 1);
            for (int c = 0; c < m; c++)
                Assert.True(got[r * m + c].Equals(single[c]), $"row {r} col {c}: batched != single-row launch.");
        }
    }

    /// <summary>The decode path quantizes ONE row per token and every topK slot reads it (xDiv = topK): bit-exact vs replicating the row.</summary>
    [SkippableFact]
    public void BroadcastActivationRow_MatchesReplicatedRows()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulMxfp4MmvqKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("spv missing.");
        const int topK = 4, numExperts = 32, m = 64, k = 2880;
        var rng = new Random(0x9121);
        byte[] bank = RandomBank(rng, numExperts, m, k);
        float[] row = new float[k];
        for (int i = 0; i < k; i++) row[i] = (float)(rng.NextDouble() * 2 - 1);
        float[] replicated = Enumerable.Range(0, topK).SelectMany(_ => row).ToArray();
        int[] indices = [3, 17, 5, 30];
        float[] bias = new float[(long)numExperts * m];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() - 0.5);

        var (viaBroadcast, _, _) = Launch(device, quant, kernel, bank, row, indices, bias, m, k, topK, numExperts, xDiv: topK);
        var (viaReplicated, _, _) = Launch(device, quant, kernel, bank, replicated, indices, bias, m, k, topK, numExperts, xDiv: 1);
        Assert.True(viaBroadcast.Max(MathF.Abs) > 1e-3f);
        for (int i = 0; i < viaBroadcast.Length; i++)
            Assert.True(viaBroadcast[i].Equals(viaReplicated[i]), $"cell {i}: broadcast {viaBroadcast[i]:G9} != replicated {viaReplicated[i]:G9}.");
    }

    /// <summary>Control: the bias epilogue adds exactly bias[expert, m].</summary>
    [SkippableFact]
    public void BiasEpilogue_AddsExactlyPerExpertBias()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "integer-dot-product unavailable.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulMxfp4MmvqKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("spv missing.");
        const int numExperts = 8, m = 40, k = 288, n = 4;
        var rng = new Random(0x9122);
        byte[] bank = RandomBank(rng, numExperts, m, k);
        float[] x = new float[(long)n * k];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] indices = [1, 6, 2, 7];
        float[] bias = new float[(long)numExperts * m];
        for (int i = 0; i < bias.Length; i++) bias[i] = (float)(rng.NextDouble() * 6 - 3);
        var (with, _, _) = Launch(device, quant, kernel, bank, x, indices, bias, m, k, n, numExperts, 1);
        var (without, _, _) = Launch(device, quant, kernel, bank, x, indices, null, m, k, n, numExperts, 1);
        for (int i = 0; i < with.Length; i++)
            Assert.Equal(without[i] + bias[(long)indices[i / m] * m + i % m], with[i], 4);
    }

    private readonly record struct Vector2(float X, float Y);

    private static int XqByte(uint[] xq, int row, int elem, int k)
    {
        uint w = xq[(long)row * (k / 4) + elem / 4];
        return (sbyte)((w >> (8 * (elem & 3))) & 0xFF);
    }

    private static float[] RunScalar(VulkanDevice device, string spvDir, byte[] bank, float[] x, int[] indices, int m, int k, int n, int numExperts)
    {
        using var refKernel = MoeIndexedMatmulMxfp4F32Kernel.Create(device, spvDir);
        using var bufW = device.Allocate(((long)bank.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufIdx = device.Allocate((long)indices.Length * sizeof(int));
        using var bufY = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(bank), bufW);
        device.Upload(x, bufX);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), bufIdx);
        refKernel.Launch(bufW, bufX, bufIdx, bufY, m, k, n, numExperts);
        float[] y = new float[(long)n * m];
        device.Download(bufY, y);
        return y;
    }

    private static (float[] Y, uint[] Xq, Vector2[] Xds) Launch(VulkanDevice device, QuantizeQ8_1RowsKernel quant, MoeIndexedMatmulMxfp4MmvqKernel kernel,
        byte[] bank, float[] x, int[] indices, float[]? bias, int m, int k, int n, int numExperts, int xDiv)
    {
        int xRows = (n + xDiv - 1) / xDiv;
        using var bankBuf = device.Allocate(((long)bank.Length + 3) & ~3L);
        using var xBuf = device.Allocate((long)x.Length * sizeof(float));
        using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(xRows, k));
        using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(xRows, k));
        using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
        using var yBuf = device.Allocate((long)n * m * sizeof(float));
        using var biasBuf = bias is null ? null : device.Allocate((long)bias.Length * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(bank), bankBuf);
        device.Upload(x, xBuf);
        device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
        if (bias is not null) device.Upload(bias, biasBuf!);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, xBuf, xqBuf, xdsBuf, xRows, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kernel.Record(ctx.CommandBuffer, bankBuf, xqBuf, xdsBuf, idxBuf, yBuf, biasBuf, m, k, n, numExperts, xDiv);
            ctx.SubmitAndWait();
        }
        float[] y = new float[(long)n * m];
        device.Download(yBuf, y);
        uint[] xq = new uint[xRows * (k / 4)];
        float[] xdsRaw = new float[xRows * (k / 32) * 2];
        device.Download(xqBuf, MemoryMarshal.Cast<uint, float>(xq.AsSpan()));
        device.Download(xdsBuf, xdsRaw);
        var xds = new Vector2[xRows * (k / 32)];
        for (int i = 0; i < xds.Length; i++) xds[i] = new Vector2(xdsRaw[2 * i], xdsRaw[2 * i + 1]);
        return (y, xq, xds);
    }
}
