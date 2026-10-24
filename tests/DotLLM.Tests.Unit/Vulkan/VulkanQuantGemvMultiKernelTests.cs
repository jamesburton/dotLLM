using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity for the multi-column K-quant / IQ GEMVs (#902): every column of <see cref="MatMulQuantGemvMultiKernel"/> must equal the single-row
/// decode GEMV run on that column alone (same per-column accumulation order; only FMA contraction may differ). Weights are random raw
/// super-blocks with finite scale fields, which exercises every code path of the decode without needing a quantiser for each format.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQuantGemvMultiKernelTests
{
    private static void AssertClose(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= 2e-3f + 1e-4f * Math.Abs(expected[i]),
                $"{what}[{i}]: expected {expected[i]:G9} actual {actual[i]:G9}");
    }

    private static byte[] RandomBlocks(Random rng, int m, int k, int blockBytes, params int[] halfOffsets)
    {
        int blocks = m * (k / 256);
        var raw = new byte[(long)blocks * blockBytes];
        rng.NextBytes(raw);
        for (int b = 0; b < blocks; b++)
            foreach (int off in halfOffsets)
                BitConverter.TryWriteBytes(raw.AsSpan(b * blockBytes + off, 2), (Half)(0.002f + 0.004f * (float)rng.NextDouble()));
        return raw;
    }

    private static void RunParity(QuantGemvMultiFormat format, int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x902 + m * 7 + k * 11 + n + (int)format);
        (int blockBytes, int[] halves) = format switch
        {
            QuantGemvMultiFormat.Q4_K => (144, new[] { 0, 2 }),
            QuantGemvMultiFormat.Q5_K => (176, new[] { 0, 2 }),
            QuantGemvMultiFormat.Q6_K or QuantGemvMultiFormat.Q6_K_R2 => (210, new[] { 208 }),
            QuantGemvMultiFormat.IQ4_XS => (136, new[] { 0 }),
            _ => (110, new[] { 0 }),
        };
        byte[] wq = RandomBlocks(rng, m, k, blockBytes, halves);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);

        using var device = VulkanDevice.Create();
        using var multi = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, format);
        Skip.If(multi is null, $"{format} multi shader unavailable on this device.");
        using var bw = device.Allocate(((long)wq.Length + 3) & ~3L);
        using var bx = device.Allocate((long)n * k * 4);
        using var by = device.Allocate((long)n * m * 4);
        using var bx1 = device.Allocate((long)k * 4);
        using var by1 = device.Allocate((long)m * 4);
        device.Upload(new ReadOnlySpan<byte>(wq), bw);
        device.Upload(x, bx);

        // The single-row reference kernel for the format.
        Action<VulkanDevice.Buffer, VulkanDevice.Buffer, VulkanDevice.Buffer> single;
        VulkanDevice.Buffer? codebook = null;
        IDisposable? refKernel;
        switch (format)
        {
            case QuantGemvMultiFormat.Q4_K: { var r = MatMulQ4KGemvF32Kernel.Create(device, spvDir); refKernel = r; single = (w, xx, yy) => r.Launch(w, xx, yy, m, k); break; }
            case QuantGemvMultiFormat.Q5_K: { var r = MatMulQ5KGemvF32Kernel.Create(device, spvDir); refKernel = r; single = (w, xx, yy) => r.Launch(w, xx, yy, m, k); break; }
            case QuantGemvMultiFormat.Q6_K or QuantGemvMultiFormat.Q6_K_R2: { var r = MatMulQ6KGemvF32Kernel.Create(device, spvDir); refKernel = r; single = (w, xx, yy) => r.Launch(w, xx, yy, m, k); break; }
            case QuantGemvMultiFormat.IQ4_XS: { var r = MatMulIq4XsGemvF32Kernel.Create(device, spvDir); refKernel = r; single = (w, xx, yy) => r.Launch(w, xx, yy, m, k); break; }
            default: { var r = MatMulIq3SGemvF32Kernel.Create(device, spvDir); refKernel = r; codebook = r.GridBuffer; single = (w, xx, yy) => r.Launch(w, xx, yy, m, k); break; }
        }
        using (refKernel)
        {
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                multi!.Record(ctx.CommandBuffer, bw, bx, by, m, k, n, codebook);
                ctx.SubmitAndWait();
            }
            float[] actual = new float[n * m];
            device.Download(by, actual);
            for (int c = 0; c < n; c++)
            {
                device.Upload(x.AsSpan(c * k, k).ToArray(), bx1);
                single(bw, bx1, by1);
                float[] exp = new float[m];
                device.Download(by1, exp);
                Assert.Contains(exp, v => v != 0f);
                AssertClose(exp, actual.AsSpan(c * m, m).ToArray(), $"{format} col {c}");
            }
        }
    }

    [SkippableTheory]
    [InlineData(QuantGemvMultiFormat.Q4_K, 64, 1024, 2)]
    [InlineData(QuantGemvMultiFormat.Q4_K, 40, 2560, 5)]     // K = 10 super-blocks: window tail (two blocks per 64-lane step)
    [InlineData(QuantGemvMultiFormat.Q4_K, 17, 256, 8)]
    [InlineData(QuantGemvMultiFormat.Q5_K, 64, 1024, 3)]
    [InlineData(QuantGemvMultiFormat.Q5_K, 40, 2560, 8)]
    [InlineData(QuantGemvMultiFormat.Q6_K, 64, 1024, 2)]
    [InlineData(QuantGemvMultiFormat.Q6_K, 40, 2560, 5)]     // 4 super-blocks per 64-lane step: 10 blocks leave a partial window
    // Two output rows per workgroup (#996): odd M (the lone last row), a single super-block, partial windows, every column count the dispatcher can pick.
    [InlineData(QuantGemvMultiFormat.Q6_K_R2, 64, 1024, 3)]
    [InlineData(QuantGemvMultiFormat.Q6_K_R2, 33, 2560, 4)]
    [InlineData(QuantGemvMultiFormat.Q6_K_R2, 17, 256, 8)]
    [InlineData(QuantGemvMultiFormat.Q6_K_R2, 40, 2560, 2)]
    [InlineData(QuantGemvMultiFormat.Q6_K_R2, 1, 4096, 4)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 64, 1024, 4)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 48, 2560, 7)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 64, 1024, 2)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 40, 2560, 6)]
    public void Multi_EachColumnMatchesSingleRowGemv(QuantGemvMultiFormat format, int m, int k, int n) => RunParity(format, m, k, n);
}
