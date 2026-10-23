using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #1018: the native Q4_0 dense kernels (dp4a decode MMVQ, 128x128x4 coopmat prefill GEMM) over RANDOM block bytes.
/// Random nibbles exercise both nibble planes, the -8 offset and the fp16 scale; the oracle is the production CPU dequant of the same
/// bytes (<see cref="Dequantize.ToFloat32"/>, the oracle the Q4_0 -> Q8_0 transcode is also bit-checked against).
/// Issue #1171: every case also runs on Q4_1 (20-byte block, value d * nibble + m), served by the same kernel classes.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNativeQ4_0DenseKernelTests
{
    private const int Group = 32;

    private static int BlockBytes(QuantizationType qt) => qt == QuantizationType.Q4_1 ? QuantFormat.Q4_1BlockBytes : QuantFormat.Q4_0BlockBytes;

    private static byte[] RandomWeights(Random rng, int m, int k, QuantizationType qt)
    {
        int blockBytes = BlockBytes(qt);
        int header = qt == QuantizationType.Q4_1 ? 4 : 2;
        int blocks = m * (k / Group);
        byte[] w = new byte[(long)blocks * blockBytes];
        for (int b = 0; b < blocks; b++)
        {
            int o = b * blockBytes;
            float d = 0.002f + rng.NextSingle() * 0.02f;
            BitConverter.TryWriteBytes(w.AsSpan(o, 2), (Half)d);
            if (qt == QuantizationType.Q4_1)
                BitConverter.TryWriteBytes(w.AsSpan(o + 2, 2), (Half)(-d * (4f + 8f * rng.NextSingle())));   // min: distinct per block
            rng.NextBytes(w.AsSpan(o + header, blockBytes - header));
        }
        return w;
    }

    private static float[] Dequant(byte[] w, int m, int k, QuantizationType qt)
    {
        float[] wf = new float[(long)m * k];
        int rowBytes = k / Group * BlockBytes(qt);
        var pin = GCHandle.Alloc(w, GCHandleType.Pinned);
        try
        {
            for (int r = 0; r < m; r++)
                Dequantize.ToFloat32(pin.AddrOfPinnedObject() + r * rowBytes, k, qt, wf.AsSpan(r * k, k));
        }
        finally { pin.Free(); }
        return wf;
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, 1, 32, true)]
    [InlineData(QuantizationType.Q4_1, 1, 32, true)]
    [InlineData(QuantizationType.Q4_0, 7, 96, true)]            // 3 blocks: one partial 32-block step
    [InlineData(QuantizationType.Q4_1, 7, 96, true)]
    [InlineData(QuantizationType.Q4_0, 33, 288, true)]          // 9 blocks
    [InlineData(QuantizationType.Q4_1, 33, 288, true)]
    [InlineData(QuantizationType.Q4_0, 5, 2080, true)]          // 65 blocks: one full 64-block U=2 chunk + a 1-block tail
    [InlineData(QuantizationType.Q4_1, 5, 2080, true)]
    [InlineData(QuantizationType.Q4_0, 6144, 2048, true)]       // LFM2.5 conv in_proj
    [InlineData(QuantizationType.Q4_0, 2048, 2048, true)]       // conv out_proj / attention q,o
    [InlineData(QuantizationType.Q4_1, 2048, 2048, true)]
    [InlineData(QuantizationType.Q4_0, 11776, 2048, true)]      // dense FFN gate/up
    [InlineData(QuantizationType.Q4_0, 2048, 11776, true)]      // dense FFN down (368 blocks/row)
    [InlineData(QuantizationType.Q4_0, 7, 96, false)]
    [InlineData(QuantizationType.Q4_0, 33, 288, false)]
    [InlineData(QuantizationType.Q4_0, 2048, 2048, false)]
    [InlineData(QuantizationType.Q4_0, 2048, 11776, false)]
    public void Mmvq_MatchesSameTierCpuOracle(QuantizationType qt, int m, int k, bool v2)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer-dot-product support.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmvq = MatMulQ4_0MmvqKernel.TryCreate(device, spvDir, v2, qt) ?? throw new Xunit.Sdk.XunitException($"{qt} mmvq spv missing or unsupported.");
        Assert.Equal(v2 || qt == QuantizationType.Q4_1, mmvq.IsV2);

        var rng = new Random(0x1018 + m * 7 + k * 11);
        byte[] w = RandomWeights(rng, m, k, qt);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;

        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, k));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, 1, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufY, m, k);
            ctx.SubmitAndWait();
        }
        float[] actual = new float[m];
        device.Download(bufY, actual);
        sbyte[] xq = MoeMmvqParitySupport.DownloadActivationBytes(device, bufXq, 1, k);
        float[] xds = MoeMmvqParitySupport.DownloadActivationScales(device, bufXds, 1, k);

        float[] wf = Dequant(w, m, k, qt);
        float[] expected = new float[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int i = 0; i < k; i++) acc += (double)wf[(long)r * k + i] * xq[i] * xds[(i / Group) * 2];
            expected[r] = (float)acc;
        }
        MoeMmvqParitySupport.AssertSameTierParity(expected, actual, m, 1, $"{qt} dense MMVQ " + (v2 ? "V2" : "V1"));
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, 1, 1, 32)]
    [InlineData(QuantizationType.Q4_1, 1, 1, 32)]
    [InlineData(QuantizationType.Q4_0, 2, 4, 96)]
    [InlineData(QuantizationType.Q4_1, 2, 4, 96)]
    [InlineData(QuantizationType.Q4_0, 17, 33, 64)]
    [InlineData(QuantizationType.Q4_1, 17, 33, 64)]
    [InlineData(QuantizationType.Q4_0, 129, 129, 256)]
    [InlineData(QuantizationType.Q4_1, 129, 129, 256)]
    [InlineData(QuantizationType.Q4_0, 5, 1000, 2048)]
    [InlineData(QuantizationType.Q4_1, 5, 1000, 2048)]
    [InlineData(QuantizationType.Q4_0, 64, 6144, 2048)]    // LFM2.5 conv in_proj width
    [InlineData(QuantizationType.Q4_1, 64, 6144, 2048)]
    public void Gemm_MatchesCpuReference(QuantizationType qt, int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spvDir, qt), $"{qt} coopmat kernel unsupported here (needs cooperative matrix, wave64, SPIR-V).");

        var rng = new Random(0x1018 + n * 31 + m * 17 + k);
        byte[] w = RandomWeights(rng, m, k, qt);
        float[] b32 = new float[(long)n * k];
        for (int i = 0; i < b32.Length; i++) b32[i] = rng.NextSingle() * 2f - 1f;
        float[] wf = Dequant(w, m, k, qt);
        double[] expected = new double[(long)n * m];
        Parallel.For(0, n, t =>
        {
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int c = 0; c < k; c++) acc += (double)wf[(long)r * k + c] * b32[(long)t * k + c];
                expected[(long)t * m + r] = acc;
            }
        });

        using var kernel = MatMulQ4_0GemmCoopmatKernel.Create(device, spvDir, qt);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(b32, bufB);
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        float[] actual = new float[(long)n * m];
        device.Download(bufC, actual);

        // F16 operands: ~2^-11 relative per product; the tolerance scales with the output RMS.
        double rms = Math.Sqrt(expected.Sum(v => v * v) / expected.Length);
        double tol = Math.Max(1e-3, 2.4e-2 * rms);
        for (long i = 0; i < actual.LongLength; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= tol,
                $"t={i / m} row={i % m}: gpu={actual[i]:F6} vs ref={expected[i]:F6} (tol {tol:G4}, rms {rms:G4})");
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, 1, 32)]
    [InlineData(QuantizationType.Q4_1, 1, 32)]
    [InlineData(QuantizationType.Q4_0, 5, 96)]
    [InlineData(QuantizationType.Q4_1, 5, 96)]
    [InlineData(QuantizationType.Q4_0, 300, 2080)]    // 65 blocks, ragged rows
    [InlineData(QuantizationType.Q4_1, 300, 2080)]
    [InlineData(QuantizationType.Q4_0, 2048, 2048)]
    [InlineData(QuantizationType.Q4_1, 2048, 2048)]
    public void ScalarGemv_MatchesCpuReference(QuantizationType qt, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ4_0GemvF32Kernel.Create(device, spvDir, qt);
        var rng = new Random(0x2018 + m * 7 + k);
        byte[] w = RandomWeights(rng, m, k, qt);
        float[] x = new float[k];
        for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;
        float[] wf = Dequant(w, m, k, qt);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(x, bufX);
        kernel.Launch(bufW, bufX, bufY, m, k);
        float[] actual = new float[m];
        device.Download(bufY, actual);
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int i = 0; i < k; i++) acc += (double)wf[(long)r * k + i] * x[i];
            Assert.True(Math.Abs(actual[r] - acc) <= 1e-3 + 1e-4 * Math.Abs(acc), $"row {r}: gpu={actual[r]:G9} ref={acc:G9}");
        }
    }

    [SkippableTheory]
    [InlineData(QuantizationType.Q4_0, 1, 1, 32)]
    [InlineData(QuantizationType.Q4_1, 1, 1, 32)]
    [InlineData(QuantizationType.Q4_0, 17, 33, 96)]
    [InlineData(QuantizationType.Q4_1, 17, 33, 96)]
    [InlineData(QuantizationType.Q4_0, 40, 130, 2080)]
    [InlineData(QuantizationType.Q4_1, 40, 130, 2080)]
    public void ScalarGemm_MatchesCpuReference(QuantizationType qt, int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ4_0GemmF32Kernel.Create(device, spvDir, qt);
        var rng = new Random(0x3018 + n * 31 + m * 17 + k);
        byte[] w = RandomWeights(rng, m, k, qt);
        float[] b = new float[(long)n * k];
        for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
        float[] wf = Dequant(w, m, k, qt);
        using var bufW = device.Allocate(((long)w.Length + 3) & ~3L);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));
        device.Upload(new ReadOnlySpan<byte>(w), bufW);
        device.Upload(b, bufB);
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        float[] actual = new float[(long)n * m];
        device.Download(bufC, actual);
        for (int t = 0; t < n; t++)
            for (int r = 0; r < m; r++)
            {
                double acc = 0;
                for (int i = 0; i < k; i++) acc += (double)wf[(long)r * k + i] * b[(long)t * k + i];
                Assert.True(Math.Abs(actual[(long)t * m + r] - acc) <= 1e-3 + 1e-4 * Math.Abs(acc), $"t={t} row {r}: gpu={actual[(long)t * m + r]:G9} ref={acc:G9}");
            }
    }
}
