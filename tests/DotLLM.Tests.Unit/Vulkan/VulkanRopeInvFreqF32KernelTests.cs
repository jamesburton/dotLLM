using System.Runtime.InteropServices;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #743 — the inverse-frequency RoPE kernel (llama.cpp <c>rope_freqs.weight</c> factors +
/// Mistral-3 attention temperature) must match the CPU oracle
/// (<see cref="RoPE.PrecomputeFrequencyTableWithFactors"/> + <c>log(floor(pos/floor)+1)*scale+1</c> on Q),
/// and must DIFFER from the plain rope kernel (otherwise the test could not see an ignored tensor).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanRopeInvFreqF32KernelTests
{
    [SkippableTheory]
    [InlineData(false, 0f, 0)]
    [InlineData(true, 0f, 0)]
    [InlineData(false, 0.1f, 8)]
    [InlineData(true, 0.1f, 8)]
    public void Launch_MatchesCpuOracle_AndDiffersFromPlainRope(bool useFactors, float tempScale, int tempFloor)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        const int seqLen = 40, numHeads = 8, numKvHeads = 2, headDim = 64;
        const float theta = 500000f;
        int half = headDim / 2;

        var factors = new float[half];
        for (int i = 0; i < half; i++) factors[i] = useFactors ? 1f + 31f * i / (half - 1) : 1f; // 1..32 like Llama-3.2

        var rng = new Random(743);
        float[] q = Rand(rng, seqLen * numHeads * headDim), k = Rand(rng, seqLen * numKvHeads * headDim);
        int[] pos = Enumerable.Range(0, seqLen).ToArray();

        var cos = new float[seqLen * half]; var sin = new float[seqLen * half];
        RoPE.PrecomputeFrequencyTableWithFactors(seqLen, headDim, theta, factors, cos, sin);
        float[] qExp = (float[])q.Clone(), kExp = (float[])k.Clone();
        RoPE.ExecuteScalar(qExp, kExp, pos, numHeads, numKvHeads, headDim, headDim, cos, sin);
        if (tempScale != 0f)
            for (int t = 0; t < seqLen; t++)
            {
                float f = (float)(Math.Log(Math.Floor((double)pos[t] / tempFloor) + 1.0) * tempScale + 1.0);
                for (int j = 0; j < numHeads * headDim; j++) qExp[t * numHeads * headDim + j] *= f;
            }

        var inv = new float[half];
        for (int i = 0; i < half; i++) inv[i] = 1f / (MathF.Pow(theta, 2f * i / headDim) * factors[i]);

        using var device = VulkanDevice.Create();
        using var kernel = RopeInvFreqF32Kernel.Create(device, spvDir);
        using var plain = RopeF32Kernel.Create(device, spvDir);
        using var bQ = device.Allocate(q.Length * 4L); using var bK = device.Allocate(k.Length * 4L);
        using var bQp = device.Allocate(q.Length * 4L); using var bKp = device.Allocate(k.Length * 4L);
        using var bPos = device.Allocate(seqLen * 4L); using var bInv = device.Allocate(half * 4L);
        device.Upload(q.AsSpan(), bQ); device.Upload(k.AsSpan(), bK);
        device.Upload(q.AsSpan(), bQp); device.Upload(k.AsSpan(), bKp);
        device.Upload(MemoryMarshal.AsBytes(pos.AsSpan()), bPos); device.Upload(inv.AsSpan(), bInv);

        kernel.Launch(bQ, bK, bPos, bInv, seqLen, numHeads, numKvHeads, headDim, headDim, theta,
            RopeF32Kernel.Variant.Norm, tempScale: tempScale, tempFloor: tempFloor);
        plain.Launch(bQp, bKp, bPos, seqLen, numHeads, numKvHeads, headDim, headDim, theta, RopeF32Kernel.Variant.Norm);

        var qa = new float[q.Length]; var ka = new float[k.Length]; var qp = new float[q.Length];
        device.Download(bQ, qa); device.Download(bK, ka); device.Download(bQp, qp);

        Assert.True(MaxDiff(qExp, qa) < 2e-3f, $"Q diff {MaxDiff(qExp, qa)}");
        Assert.True(MaxDiff(kExp, ka) < 2e-3f, $"K diff {MaxDiff(kExp, ka)}");
        if (useFactors || tempScale != 0f)
            Assert.True(MaxDiff(qp, qa) > 1e-2f, "kernel output equals the plain rope kernel — feature ignored");
        else
            Assert.True(MaxDiff(qp, qa) < 2e-3f);
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        float m = 0; for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i])); return m;
    }

    private static float[] Rand(Random r, int n)
    {
        var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(r.NextDouble() * 2 - 1); return a;
    }
}
