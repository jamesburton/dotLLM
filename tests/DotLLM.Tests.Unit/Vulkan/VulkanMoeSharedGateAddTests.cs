using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #693: <see cref="MoeSharedGateAddF32Kernel"/> fuses the shared-expert gate dot, its sigmoid and the gated add. Checked against a double-
/// precision CPU reference (the dot is reduced in a different order than the GEMM it replaces, so not bit-identical).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanMoeSharedGateAddTests
{
    [SkippableTheory]
    [InlineData(1, 2048)]
    [InlineData(5, 2048)]
    [InlineData(37, 260)]     // 65 vec4s: fewer than a workgroup's 256 threads, ragged
    [InlineData(64, 4)]       // one vec4
    [InlineData(9, 3000)]     // 750 vec4s: more than 256, ragged tail
    public void Launch_MatchesCpuReference(int seqLen, int hidden)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "moe_shared_gate_add_f32.spv")), "SPIR-V missing.");
        using var device = VulkanDevice.Create();
        using var kernel = MoeSharedGateAddF32Kernel.Create(device, spvDir);

        var rng = new Random(694 + seqLen * 31 + hidden);
        float[] Rand(int n, float s) => Enumerable.Range(0, n).Select(_ => (float)((rng.NextDouble() * 2 - 1) * s)).ToArray();
        float[] x = Rand(seqLen * hidden, 1f), w = Rand(hidden, 0.2f), b = Rand(seqLen * hidden, 1f), o = Rand(seqLen * hidden, 1f);
        float[] expect = new float[o.Length];
        for (int t = 0; t < seqLen; t++)
        {
            double dot = 0;
            for (int h = 0; h < hidden; h++) dot += (double)x[t * hidden + h] * w[h];
            double scale = 1.0 / (1.0 + Math.Exp(-dot));
            for (int h = 0; h < hidden; h++) expect[t * hidden + h] = (float)(o[t * hidden + h] + scale * b[t * hidden + h]);
        }

        using var bo = device.Allocate((long)o.Length * 4);
        using var bb = device.Allocate((long)b.Length * 4);
        using var bx = device.Allocate((long)x.Length * 4);
        using var bw = device.Allocate((long)w.Length * 4);
        device.Upload(o, bo); device.Upload(b, bb); device.Upload(x, bx); device.Upload(w, bw);
        kernel.Launch(bo, bb, bx, bw, seqLen, hidden);
        float[] act = new float[o.Length];
        device.Download(bo, act);
        for (int i = 0; i < act.Length; i++)
            Assert.True(Math.Abs(act[i] - expect[i]) <= 2e-5 + 1e-5 * Math.Abs(expect[i]), $"idx {i}: {act[i]:G9} vs {expect[i]:G9}");
    }
}
