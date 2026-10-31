using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Integration.Vulkan;

/// <summary>
/// #1473: the in-place F16-activation encoder GEMM (<see cref="MatMulEncoderGemmKernel"/>) against a double-precision reference for the F32 and
/// F16 weight formats, on shapes that are not multiples of the 128x128 tile (boundary store path) with K spanning several chunks. The Q8_0 format
/// is covered end to end by <see cref="BertEncoderVulkanParityTests"/>.
/// </summary>
[Trait("Category", "GPU")]
[Collection(GpuCollection.Name)]
public sealed class EncoderGemmKernelTests
{
    [SkippableTheory]
    [InlineData(EncoderWeightFormat.F32, 200, 96, 150)]
    [InlineData(EncoderWeightFormat.F32, 384, 384, 7)]
    [InlineData(EncoderWeightFormat.F16, 200, 96, 150)]
    public unsafe void Matches_reference_and_discriminates(EncoderWeightFormat fmt, int m, int k, int n)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan device.");
        string spv = VulkanModelLoader.ResolveSpvDir();
        using var dev = VulkanDevice.Create();
        Skip.IfNot(MatMulEncoderGemmKernel.IsSupportedOn(dev, spv, fmt), "needs a wave64 cooperative-matrix device");
        var rng = new Random(1473);
        var w = new float[m * k]; var b = new float[n * k];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        var bh = new Half[b.Length]; for (int i = 0; i < b.Length; i++) bh[i] = (Half)b[i];
        byte[] wBytes;
        float[] wRef = new float[w.Length];
        if (fmt == EncoderWeightFormat.F32) { wBytes = MemoryMarshal.AsBytes<float>(w).ToArray(); for (int i = 0; i < w.Length; i++) wRef[i] = w[i]; }
        else
        {
            var wh = new Half[w.Length]; for (int i = 0; i < w.Length; i++) { wh[i] = (Half)w[i]; wRef[i] = (float)wh[i]; }
            wBytes = MemoryMarshal.AsBytes<Half>(wh).ToArray();
        }
        using var wb = dev.AllocateDeviceLocal(wBytes.Length); dev.Upload(wBytes, wb);
        using var bb = dev.AllocateDeviceLocal(bh.Length * 2); dev.Upload(MemoryMarshal.AsBytes<Half>(bh), bb);
        using var cb = dev.AllocateDeviceLocal((long)n * m * 2);
        using (var kern = MatMulEncoderGemmKernel.Create(dev, spv, fmt))
        {
            using var ctx = dev.CreateSubmitContext();
            ctx.Begin();
            kern.Record(ctx.CommandBuffer, wb, bb, cb, m, k, n);
            ctx.SubmitAndWait();
        }
        int words = (n * m + 1) / 2;
        var raw = new float[words];
        dev.Download(cb, raw);
        var c = MemoryMarshal.Cast<float, Half>(raw.AsSpan());
        double worst = 0, scale = 0;
        for (int t = 0; t < n; t++)
            for (int j = 0; j < m; j++)
            {
                double r = 0;
                for (int kk = 0; kk < k; kk++) r += (double)wRef[j * k + kk] * (double)(float)bh[t * k + kk];
                worst = Math.Max(worst, Math.Abs(r - (double)(float)c[t * m + j]));
                scale = Math.Max(scale, Math.Abs(r));
            }
        // Discriminating bound: F16 output rounding is ~2^-11 relative (4.9e-4); an unscaled / mis-indexed weight read is O(1).
        Assert.True(worst <= 2e-3 * scale + 1e-3, $"{fmt} m={m} k={k} n={n}: max |err| {worst:E3} vs max |ref| {scale:F3}");
        Assert.True(scale > 0.5, "reference must be non-trivial");
    }
}
