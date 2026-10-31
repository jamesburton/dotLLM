using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// #1278: the Vulkan MLA decoupled-rope kernels must apply DeepSeek-V2/V3's YaRN-ramped inverse frequencies (and the cos/sin multiplier) exactly
/// as the CPU table does. They used plain <c>1 / theta^(2i/d)</c>, which shifted every DeepSeek-V2-Lite logit (L-inf 4.8 against the CPU).
/// Reference: <see cref="RoPE.ComputeYarnInverseFrequencies"/> + interleaved (2i, 2i+1) rotation, the CPU convention.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class MlaRopeYarnParityTests
{
    private const int NumHeads = 4, Nope = 8, Rope = 64, Tokens = 7;
    private const float Theta = 10000f;

    private static MlaYarnRope Yarn(float mscale = 1.0f)
    {
        var (low, high) = RoPE.YarnRampBounds(Rope, Theta, 4096, 32f, 1f);
        return new MlaYarnRope(40f, low, high, mscale);
    }

    private static float[] Reference(float[] x, int[] positions, int rowStride, int ropeOffset, int rows, int perTokenRows, MlaYarnRope y)
    {
        var inv = new float[Rope / 2];
        if (y.Factor > 1f) RoPE.ComputeYarnInverseFrequencies(Rope, Theta, y.Factor, 4096, 32f, 1f, inv);
        else for (int i = 0; i < inv.Length; i++) inv[i] = 1f / MathF.Pow(Theta, 2f * i / Rope);
        var o = (float[])x.Clone();
        for (int r = 0; r < rows; r++)
        {
            int t = r / perTokenRows;
            for (int i = 0; i < Rope / 2; i++)
            {
                double ang = positions[t] * (double)inv[i];
                float c = (float)Math.Cos(ang) * y.Mscale, s = (float)Math.Sin(ang) * y.Mscale;
                int i0 = r * rowStride + ropeOffset + 2 * i;
                float v0 = x[i0], v1 = x[i0 + 1];
                o[i0] = v0 * c - v1 * s;
                o[i0 + 1] = v0 * s + v1 * c;
            }
        }
        return o;
    }

    private static (float QErr, float KErr) Run(MlaYarnRope kernelYarn, MlaYarnRope refYarn, string spvDir)
    {
        var rng = new Random(1278);
        int qHead = Nope + Rope;
        var q = new float[Tokens * NumHeads * qHead];
        var kPe = new float[Tokens * Rope];
        for (int i = 0; i < q.Length; i++) q[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < kPe.Length; i++) kPe[i] = (float)(rng.NextDouble() * 2 - 1);
        int[] positions = [0, 1, 17, 255, 1000, 4095, 5000];

        float[] qRef = Reference(q, positions, qHead, Nope, Tokens * NumHeads, NumHeads, refYarn);
        float[] kRef = Reference(kPe, positions, Rope, 0, Tokens, 1, refYarn);

        using var device = VulkanDevice.Create();
        using var kernel = RopeMlaF32Kernel.Create(device, spvDir);
        kernel.Yarn = kernelYarn;
        using var bq = device.Allocate((long)q.Length * 4);
        using var bk = device.Allocate((long)kPe.Length * 4);
        using var bp = device.Allocate((long)positions.Length * 4);
        device.Upload(q.AsSpan(), bq);
        device.Upload(kPe.AsSpan(), bk);
        device.Upload(System.Runtime.InteropServices.MemoryMarshal.AsBytes(positions.AsSpan()), bp);
        kernel.Launch(bq, bk, bp, Tokens, NumHeads, Nope, Rope, Theta);
        var qOut = new float[q.Length];
        var kOut = new float[kPe.Length];
        device.Download(bq, qOut);
        device.Download(bk, kOut);
        float qe = 0, ke = 0;
        for (int i = 0; i < q.Length; i++) qe = MathF.Max(qe, MathF.Abs(qOut[i] - qRef[i]));
        for (int i = 0; i < kPe.Length; i++) ke = MathF.Max(ke, MathF.Abs(kOut[i] - kRef[i]));
        return (qe, ke);
    }

    [SkippableTheory]
    [InlineData(1.0f)]
    [InlineData(1.2607f)]
    public void RopeMla_YarnKernel_MatchesCpuTable(float mscale)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (qe, ke) = Run(Yarn(mscale), Yarn(mscale), spvDir);
        Assert.True(qe < 2e-3f && ke < 2e-3f, $"YaRN MLA rope diverges from the CPU table: q max err {qe:E3}, k_pe max err {ke:E3}.");
    }

    [SkippableFact]
    public void RopeMla_PlainThetaKernel_IsDetectablyDifferentFromYarnReference()
    {
        // Sensitivity of the check above: the pre-#1278 behaviour (YaRN off in the kernel) must be far from the YaRN reference at these positions.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (qe, ke) = Run(MlaYarnRope.Off, Yarn(), spvDir);
        Assert.True(qe > 0.05f || ke > 0.05f, $"Plain-theta kernel is not distinguishable from the YaRN reference (q {qe:E3}, k {ke:E3}); the parity test above is blind.");
    }

    [SkippableFact]
    public void RopeMla_YarnOff_StillMatchesPlainThetaReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (qe, ke) = Run(MlaYarnRope.Off, MlaYarnRope.Off, spvDir);
        Assert.True(qe < 2e-3f && ke < 2e-3f, $"Legacy plain-theta path regressed: q {qe:E3}, k {ke:E3}.");
    }
}
