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
    // #1377: 9..16 columns. Odd M, a single super-block, partial windows, the widest pipeline, every column a distinct activation row.
    [InlineData(QuantGemvMultiFormat.Q5_K, 37, 2560, 9)]
    [InlineData(QuantGemvMultiFormat.Q5_K, 65, 1024, 16)]
    [InlineData(QuantGemvMultiFormat.Q5_K, 17, 256, 13)]
    [InlineData(QuantGemvMultiFormat.Q6_K, 37, 2560, 12)]
    [InlineData(QuantGemvMultiFormat.Q6_K, 65, 1024, 16)]
    [InlineData(QuantGemvMultiFormat.Q6_K, 17, 256, 9)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 37, 2560, 10)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 65, 1024, 16)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 17, 256, 15)]
    // #1389: IQ3_S 9..16 columns (coalesced K-position layout, no longer capped at 8).
    [InlineData(QuantGemvMultiFormat.IQ3_S, 37, 2560, 9)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 65, 1024, 16)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 17, 256, 13)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 33, 2560, 12)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 40, 2560, 8)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 64, 1024, 3)]
    public void Multi_EachColumnMatchesSingleRowGemv(QuantGemvMultiFormat format, int m, int k, int n) => RunParity(format, m, k, n);

    /// <summary>
    /// #1377: a wide (9..16) dispatch and the 8-column dispatch agree to ULP level on the columns they share (not bit-equal: per-NC
    /// specialisation lets the driver contract FMAs differently, unlike the NR-parameterised Q4_K / Q6_K_Rows shaders). Also pins the per-format column cap (the R2 Q6_K stays at 8).
    /// </summary>
    [SkippableTheory]
    [InlineData(QuantGemvMultiFormat.Q5_K, 37, 2560, 13)]
    [InlineData(QuantGemvMultiFormat.Q6_K, 37, 2560, 16)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 37, 2560, 11)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 37, 2560, 16)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 37, 2560, 12)]
    public void Wide_SharedColumnsMatchEightColumnDispatch(QuantGemvMultiFormat format, int m, int k, int n)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var rng = new Random(0x1377 + m * 7 + k * 11 + n + (int)format);
        (int blockBytes, int[] halves) = format switch
        {
            QuantGemvMultiFormat.Q5_K => (176, new[] { 0, 2 }),
            QuantGemvMultiFormat.Q6_K => (210, new[] { 208 }),
            QuantGemvMultiFormat.IQ4_XS => (136, new[] { 0 }),
            _ => (110, new[] { 0 }),
        };
        byte[] wq = RandomBlocks(rng, m, k, blockBytes, halves);
        float[] x = F16Bf16Fixture.RandomFloats(rng, n * k, 1.0f);
        using var device = VulkanDevice.Create();
        using var multi = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, format);
        Skip.If(multi is null, $"{format} multi shader unavailable on this device.");
        Assert.Equal(16, multi!.MaxColumnsSupported);
        Assert.True(multi.AcceptsColumns(16, 256));
        Assert.False(multi.AcceptsColumns(17, 256));
        VulkanDevice.Buffer? codebook = null;
        using var gridOwner = format == QuantGemvMultiFormat.IQ3_S ? MatMulIq3SGemvF32Kernel.Create(device, spvDir) : null;
        codebook = gridOwner?.GridBuffer;
        using var bw = device.Allocate(((long)wq.Length + 3) & ~3L);
        using var bx = device.Allocate((long)n * k * 4);
        using var byWide = device.Allocate((long)n * m * 4);
        using var by8 = device.Allocate(8L * m * 4);
        device.Upload(new ReadOnlySpan<byte>(wq), bw);
        device.Upload(x, bx);
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            multi.Record(ctx.CommandBuffer, bw, bx, byWide, m, k, n, codebook);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            multi.Record(ctx.CommandBuffer, bw, bx, by8, m, k, 8, codebook);
            ctx.SubmitAndWait();
        }
        float[] wide = new float[n * m], eight = new float[8 * m];
        device.Download(byWide, wide);
        device.Download(by8, eight);
        Assert.Contains(eight, v => v != 0f);
        // Not bit-equal: the non-Rows shaders are specialised per NC, so the driver may contract FMAs differently (ULP-level, e.g. 49.449379 vs
        // 49.4493713). The accumulation ORDER per column is the same; the bound below is far tighter than the 2e-3 + 1e-4 rel parity vs the single-row kernel.
        float[] w8 = wide.AsSpan(0, 8 * m).ToArray();
        for (int i = 0; i < w8.Length; i++)
            Assert.True(Math.Abs(eight[i] - w8[i]) <= 1e-4f + 1e-5f * Math.Abs(eight[i]), $"[{i}]: 8-col {eight[i]:G9} vs wide {w8[i]:G9}");
    }

    /// <summary>The Q6_K two-rows-per-workgroup variant has no wide pipelines: it stops at eight and the dispatcher must not hand it more. IQ3_S serves 16 since #1389.</summary>
    [SkippableFact]
    public void Q6KR2_StopsAtEightColumns_Iq3SServesSixteen()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var r2 = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.Q6_K_R2);
        Skip.If(r2 is null, "Q6_K_R2 unavailable");
        Assert.Equal(8, r2!.MaxColumnsSupported);
        Assert.False(r2.AcceptsColumns(9, 256));
        using var iq3 = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, QuantGemvMultiFormat.IQ3_S);
        if (iq3 is not null) { Assert.Equal(16, iq3.MaxColumnsSupported); Assert.True(iq3.AcceptsColumns(16, 256)); Assert.False(iq3.AcceptsColumns(17, 256)); }
    }

    /// <summary>
    /// Opt-in timing (<c>DOTLLM_GEMV_WIDE_BENCH=1</c>, #1377): per format at an FFN shape, one 16-column dispatch against two 8-column dispatches
    /// (the alternative that needs no wide shader) and a single 8-column one. Minimum of 5 rounds x 20 back-to-back dispatches; device-local weights.
    /// </summary>
    [SkippableTheory]
    [InlineData(QuantGemvMultiFormat.Q5_K, 176, 17408, 5120)]
    [InlineData(QuantGemvMultiFormat.Q6_K, 210, 5120, 6144)]
    [InlineData(QuantGemvMultiFormat.IQ4_XS, 136, 17408, 5120)]
    [InlineData(QuantGemvMultiFormat.IQ3_S, 110, 17408, 5120)]   // #1389
    public void Bench_WideColumns(QuantGemvMultiFormat format, int blockBytes, int m, int k)
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_GEMV_WIDE_BENCH") == "1", "DOTLLM_GEMV_WIDE_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var multi = MatMulQuantGemvMultiKernel.TryCreate(device, spvDir, format);
        Skip.If(multi is null, "unavailable");
        using var gridOwner = format == QuantGemvMultiFormat.IQ3_S ? MatMulIq3SGemvF32Kernel.Create(device, spvDir) : null;
        var rng = new Random(m + k);
        byte[] wq = RandomBlocks(rng, m, k, blockBytes, format == QuantGemvMultiFormat.Q6_K ? new[] { 208 } : new[] { 0, 2 });
        using var staging = device.Allocate(((long)wq.Length + 3) & ~3L);
        device.Upload(new ReadOnlySpan<byte>(wq), staging);
        using var bw = device.AllocateDeviceLocal(((long)wq.Length + 3) & ~3L);
        device.CopyBufferSynchronous(staging, bw, (ulong)wq.Length);
        using var bx = device.Allocate(16L * k * 4);
        device.Upload(F16Bf16Fixture.RandomFloats(rng, 16 * k, 1.0f), bx);
        using var by = device.AllocateDeviceLocal(16L * m * 4);
        var cb = gridOwner?.GridBuffer;
        double[] t = VulkanDenseGemvRowsTests.Time(device, [
            cmd => multi!.Record(cmd, bw, bx, by, m, k, 8, cb),
            cmd => { multi!.Record(cmd, bw, bx, by, m, k, 8, cb); multi.Record(cmd, bw, bx, by, m, k, 8, cb); },
            cmd => multi!.Record(cmd, bw, bx, by, m, k, 16, cb),
            cmd => multi!.Record(cmd, bw, bx, by, m, k, 12, cb)]);
        Console.WriteLine($"wide-bench {format} m={m} k={k}: 1x8 {t[0] * 1000:F0} us | 2x8 {t[1] * 1000:F0} us | 1x12 {t[3] * 1000:F0} us | 1x16 {t[2] * 1000:F0} us");
    }
}
