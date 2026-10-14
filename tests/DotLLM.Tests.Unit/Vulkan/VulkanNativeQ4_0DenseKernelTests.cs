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
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNativeQ4_0DenseKernelTests
{
    private const int BlockBytes = 18;
    private const int Group = 32;

    private static byte[] RandomWeights(Random rng, int m, int k)
    {
        int blocks = m * (k / Group);
        byte[] w = new byte[(long)blocks * BlockBytes];
        for (int b = 0; b < blocks; b++)
        {
            int o = b * BlockBytes;
            BitConverter.TryWriteBytes(w.AsSpan(o, 2), (Half)(0.002f + rng.NextSingle() * 0.02f));
            rng.NextBytes(w.AsSpan(o + 2, BlockBytes - 2));
        }
        return w;
    }

    private static float[] Dequant(byte[] w, int m, int k)
    {
        float[] wf = new float[(long)m * k];
        int rowBytes = k / Group * BlockBytes;
        var pin = GCHandle.Alloc(w, GCHandleType.Pinned);
        try
        {
            for (int r = 0; r < m; r++)
                Dequantize.ToFloat32(pin.AddrOfPinnedObject() + r * rowBytes, k, QuantizationType.Q4_0, wf.AsSpan(r * k, k));
        }
        finally { pin.Free(); }
        return wf;
    }

    [SkippableTheory]
    [InlineData(1, 32)]
    [InlineData(7, 96)]            // 3 blocks, window tail
    [InlineData(33, 288)]          // 9 blocks: window + 1 block
    [InlineData(6144, 2048)]       // LFM2.5 conv in_proj
    [InlineData(2048, 2048)]       // conv out_proj / attention q,o
    [InlineData(11776, 2048)]      // dense FFN gate/up
    [InlineData(2048, 11776)]      // dense FFN down (368 blocks/row)
    public void Mmvq_MatchesSameTierCpuOracle(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No integer-dot-product support.");
        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmvq = MatMulQ4_0MmvqKernel.TryCreate(device, spvDir) ?? throw new Xunit.Sdk.XunitException("matmul_q4_0_mmvq.spv missing or unsupported.");

        var rng = new Random(0x1018 + m * 7 + k * 11);
        byte[] w = RandomWeights(rng, m, k);
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

        float[] wf = Dequant(w, m, k);
        float[] expected = new float[m];
        for (int r = 0; r < m; r++)
        {
            double acc = 0;
            for (int i = 0; i < k; i++) acc += (double)wf[(long)r * k + i] * xq[i] * xds[(i / Group) * 2];
            expected[r] = (float)acc;
        }
        MoeMmvqParitySupport.AssertSameTierParity(expected, actual, m, 1, "Q4_0 dense MMVQ");
    }

    [SkippableTheory]
    [InlineData(1, 1, 32)]
    [InlineData(2, 4, 96)]
    [InlineData(17, 33, 64)]
    [InlineData(129, 129, 256)]
    [InlineData(5, 1000, 2048)]
    [InlineData(64, 6144, 2048)]    // LFM2.5 conv in_proj width
    public void Gemm_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(MatMulQ4_0GemmCoopmatKernel.IsSupportedOn(device, spvDir), "Q4_0 coopmat kernel unsupported here (needs cooperative matrix, wave64, SPIR-V).");

        var rng = new Random(0x1018 + n * 31 + m * 17 + k);
        byte[] w = RandomWeights(rng, m, k);
        float[] b32 = new float[(long)n * k];
        for (int i = 0; i < b32.Length; i++) b32[i] = rng.NextSingle() * 2f - 1f;
        float[] wf = Dequant(w, m, k);
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

        using var kernel = MatMulQ4_0GemmCoopmatKernel.Create(device, spvDir);
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
}
