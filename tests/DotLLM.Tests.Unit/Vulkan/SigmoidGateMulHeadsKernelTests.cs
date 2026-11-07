using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1572: LFM3's per-head sigmoid gate (<c>attnOut[t, h, d] *= sigmoid(gate[t, h])</c>) against a CPU reference, including the
/// long-prefill sizes (LFM3 1.7B: 16 heads x 128) where one element per thread needs more than 65535 workgroups.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class SigmoidGateMulHeadsKernelTests(ITestOutputHelper output)
{
    [SkippableTheory]
    [InlineData(3, 4, 32)]            // tiny, partial workgroup
    [InlineData(8192, 16, 128)]       // 65536 groups of 256: the old 1-D dispatch's last fitting size
    [InlineData(8193, 16, 128)]       // first size past 65535 groups
    [InlineData(32768, 16, 128)]      // 32K prefill
    public void MatchesCpu(int seqLen, int numHeads, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        using var kernel = SigmoidGateMulHeadsF32Kernel.Create(device, spvDir);

        int n = seqLen * numHeads * headDim, g = seqLen * numHeads;
        var rng = new Random(0x1572 + seqLen);
        float[] x = new float[n], gate = new float[g];
        for (int i = 0; i < n; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < g; i++) gate[i] = (float)(rng.NextDouble() * 8 - 4);

        using var xB = device.Allocate(n * 4L);
        using var gB = device.Allocate(g * 4L);
        device.Upload(x, xB);
        device.Upload(gate, gB);
        long t0 = Stopwatch.GetTimestamp();
        kernel.Launch(xB, gB, n, headDim);
        output.WriteLine($"seqLen={seqLen} n={n}: {Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F1} ms");
        float[] actual = new float[n];
        device.Download(xB, actual);

        // Every element must be gated by ITS (token, head) gate: an unwritten tail keeps x, a wrong gate index changes the value.
        for (int i = 0; i < n; i++)
        {
            float expected = x[i] * (1f / (1f + MathF.Exp(-gate[i / headDim])));
            Assert.True(MathF.Abs(expected - actual[i]) <= 1e-6f + 1e-5f * MathF.Abs(expected),
                $"i={i} (row {i / (numHeads * headDim)}): cpu {expected:G7} vs gpu {actual[i]:G7}");
        }
    }
}
