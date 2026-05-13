using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for the Vulkan multi-token GDN scan against the CPU oracle
/// <see cref="GatedDeltaNetScan.Execute"/>.
/// </summary>
/// <remarks>
/// <para>
/// The shader walks the seqLen loop internally — exactly the same access
/// pattern as the CPU reference. Reductions inside each token are scalar
/// row-outer (no parallel reduction across rows), so the rounding order is
/// preserved exactly. Tolerance is therefore tight: ≤4 ULP, accommodating
/// only the ≤4 ULP drift of GLSL <c>sqrt</c> in the per-token <c>1/√d</c>
/// scale (and even that is small).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanGdnScanMultiTokenF32KernelTests
{
    [SkippableTheory]
    [InlineData(1, 16, 4, 32)]              // single-token sanity
    [InlineData(4, 16, 4, 32)]
    [InlineData(8, 16, 4, 32)]
    [InlineData(16, 32, 8, 64)]             // power-of-two heads
    [InlineData(7, 12, 3, 32)]              // odd seqLen, non-power-of-two heads
    public void Launch_MatchesCpuReference(int seqLen, int nVHead, int nKHead, int dState)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_scan_multi_token_f32.spv")),
            "gdn_scan_multi_token_f32.spv not compiled (glslc / Vulkan SDK required).");

        // ── Synthetic inputs ────────────────────────────────────────────────
        var rng = new Random((seqLen * 31 + nVHead) * 1009 + dState);
        float[] state0 = RandomFloats(rng, nVHead * dState * dState, 0.1f);
        float[] q = RandomFloats(rng, seqLen * nKHead * dState, 1.0f);
        float[] k = RandomFloats(rng, seqLen * nKHead * dState, 1.0f);
        float[] v = RandomFloats(rng, seqLen * nVHead * dState, 1.0f);
        // g in (0, 1] and beta in [0, 1] — what the CPU oracle expects post
        // softplus/sigmoid. Use deterministic small ranges.
        float[] g = new float[seqLen * nVHead];
        float[] beta = new float[seqLen * nVHead];
        for (int i = 0; i < g.Length; i++)
        {
            g[i] = 0.5f + 0.5f * (float)rng.NextDouble();
            beta[i] = (float)rng.NextDouble();
        }

        // ── CPU oracle ──────────────────────────────────────────────────────
        float[] cpuState = (float[])state0.Clone();
        float[] cpuOut = new float[seqLen * nVHead * dState];
        GatedDeltaNetScan.Execute(cpuState, q, k, v, g, beta, cpuOut,
            nVHead, nKHead, dState, seqLen);

        // ── GPU dispatch ────────────────────────────────────────────────────
        using var device = VulkanDevice.Create();
        using var kernel = GdnScanMultiTokenF32Kernel.Create(device, spvDir);

        using var stateBuf = device.Allocate((long)state0.Length * sizeof(float));
        using var qBuf = device.Allocate((long)q.Length * sizeof(float));
        using var kBuf = device.Allocate((long)k.Length * sizeof(float));
        using var vBuf = device.Allocate((long)v.Length * sizeof(float));
        using var gBuf = device.Allocate((long)g.Length * sizeof(float));
        using var betaBuf = device.Allocate((long)beta.Length * sizeof(float));
        using var outBuf = device.Allocate((long)cpuOut.Length * sizeof(float));
        device.Upload(state0.AsSpan(), stateBuf);
        device.Upload(q.AsSpan(), qBuf);
        device.Upload(k.AsSpan(), kBuf);
        device.Upload(v.AsSpan(), vBuf);
        device.Upload(g.AsSpan(), gBuf);
        device.Upload(beta.AsSpan(), betaBuf);

        kernel.Launch(stateBuf, qBuf, kBuf, vBuf, gBuf, betaBuf, outBuf,
            seqLen, nVHead, nKHead, dState);

        float[] gpuState = new float[state0.Length];
        float[] gpuOut = new float[cpuOut.Length];
        device.Download(stateBuf, gpuState);
        device.Download(outBuf, gpuOut);

        AssertCloseUlp(cpuState, gpuState, "state");
        AssertCloseUlp(cpuOut, gpuOut, "output");
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static unsafe int UlpDiff(float a, float b)
    {
        if (float.IsNaN(a) || float.IsNaN(b)) return int.MaxValue;
        if (a == b) return 0;
        int ai = *(int*)&a;
        int bi = *(int*)&b;
        if (ai < 0) ai = unchecked((int)(0x80000000u - (uint)ai));
        if (bi < 0) bi = unchecked((int)(0x80000000u - (uint)bi));
        return Math.Abs(ai - bi);
    }

    private static void AssertCloseUlp(float[] expected, float[] actual, string label)
    {
        const int MaxUlp = 4;
        Assert.Equal(expected.Length, actual.Length);
        int maxUlp = 0;
        int violations = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            int diff = UlpDiff(expected[i], actual[i]);
            if (MathF.Abs(expected[i]) < 1e-30f && MathF.Abs(actual[i]) < 1e-30f) diff = 0;
            if (diff > maxUlp) maxUlp = diff;
            if (diff > MaxUlp) violations++;
        }
        Assert.True(violations == 0,
            $"GdnScanMultiToken {label} drift exceeded {MaxUlp} ULP: violations={violations}/{expected.Length}, maxUlp={maxUlp}");
    }
}
