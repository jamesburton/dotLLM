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
    [InlineData(4, 32, 16, 128)]            // Qwen3.6-A3B production shape (NVHead/NKHead=2, dState=128)
    [InlineData(3, 4, 2, 16)]               // tiny tiled discriminator — NVHead/NKHead=2 with distinct per-kh inputs;
                                            // the random k/q per token are different for kh=0 and kh=1, so
                                            // an interleaved-vs-tiled (vh/2 vs vh%2) head-broadcast bug would
                                            // produce a numerically different result. Matches the CPU regression
                                            // template `GatedDeltaNetScanTests.HeadBroadcast_IsTiledNotInterleaved_QwenMoeStyle`
                                            // referenced in .planning/.continue-here.md CONSTRAINT 2.
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

    /// <summary>
    /// #445: every factorial arm must reproduce the shipping kernel's output <b>bit for bit</b>,
    /// not merely within the 4 ULP the CPU-oracle test allows.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The arms change only where the state matrix lives (global SSBO vs an LDS column slab)
    /// and how many times each element is re-read per token (six vs four). Every reduction
    /// keeps its order and its <c>precise</c> qualifier, and each fusion reuses from a register
    /// the exact f32 the preceding store rounded to. So the correct assertion is equality, and
    /// a 4 ULP tolerance here would be <i>too loose to discriminate</i> — it would pass an arm
    /// that had quietly let the driver contract a multiply-add into an FMA, which is precisely
    /// the failure the shipping shader's <c>precise</c> discipline exists to prevent.
    /// </para>
    /// <para>
    /// Run at Bonsai 2's real GDN shape (nVHead 48, nKHead 16, dState 128) because the LDS arms
    /// split columns 32 at a time: a dState of 32 would put every head in a single workgroup and
    /// leave the split itself untested. seqLen 12 is enough for the recurrence to carry state
    /// across many tokens while keeping the test quick.
    /// </para>
    /// </remarks>
    [SkippableTheory]
    [InlineData(GdnScanMultiTokenF32Kernel.Variant.Fused)]
    [InlineData(GdnScanMultiTokenF32Kernel.Variant.Lds)]
    [InlineData(GdnScanMultiTokenF32Kernel.Variant.LdsFused)]
    public void Variant_IsBitIdenticalToShippingKernel(GdnScanMultiTokenF32Kernel.Variant variant)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_scan_multi_token_f32.spv")),
            "gdn_scan_multi_token_f32.spv not compiled (glslc / Vulkan SDK required).");

        const int seqLen = 12, nVHead = 48, nKHead = 16, dState = 128;

        var rng = new Random(445);
        float[] state0 = RandomFloats(rng, nVHead * dState * dState, 0.1f);
        float[] q = RandomFloats(rng, seqLen * nKHead * dState, 1.0f);
        float[] k = RandomFloats(rng, seqLen * nKHead * dState, 1.0f);
        float[] v = RandomFloats(rng, seqLen * nVHead * dState, 1.0f);
        float[] g = new float[seqLen * nVHead];
        float[] beta = new float[seqLen * nVHead];
        for (int i = 0; i < g.Length; i++)
        {
            g[i] = 0.5f + 0.5f * (float)rng.NextDouble();
            beta[i] = (float)rng.NextDouble();
        }

        using var device = VulkanDevice.Create();
        Skip.IfNot(File.Exists(Path.Combine(spvDir, VariantSpv(variant))),
            $"{VariantSpv(variant)} not compiled (glslc / Vulkan SDK required).");

        var (baseState, baseOut) = RunArm(
            device, spvDir, GdnScanMultiTokenF32Kernel.Variant.Baseline,
            state0, q, k, v, g, beta, seqLen, nVHead, nKHead, dState);
        var (armState, armOut) = RunArm(
            device, spvDir, variant,
            state0, q, k, v, g, beta, seqLen, nVHead, nKHead, dState);

        AssertBitIdentical(baseState, armState, $"{variant} state");
        AssertBitIdentical(baseOut, armOut, $"{variant} output");
    }

    private static string VariantSpv(GdnScanMultiTokenF32Kernel.Variant v) => v switch
    {
        GdnScanMultiTokenF32Kernel.Variant.Fused => "gdn_scan_multi_token_fused_f32.spv",
        GdnScanMultiTokenF32Kernel.Variant.Lds => "gdn_scan_multi_token_lds_f32.spv",
        GdnScanMultiTokenF32Kernel.Variant.LdsFused => "gdn_scan_multi_token_lds_fused_f32.spv",
        _ => "gdn_scan_multi_token_f32.spv",
    };

    private static (float[] State, float[] Out) RunArm(
        VulkanDevice device, string spvDir, GdnScanMultiTokenF32Kernel.Variant variant,
        float[] state0, float[] q, float[] k, float[] v, float[] g, float[] beta,
        int seqLen, int nVHead, int nKHead, int dState)
    {
        using var kernel = GdnScanMultiTokenF32Kernel.Create(device, spvDir, variant);
        // Allocated fresh per arm and never reused across iterations: a recycled handle can
        // land back in the handle-keyed DescriptorSetCache and hand the second arm the first
        // arm's descriptor set, which reads as a correct-then-zeros kernel bug.
        using var stateBuf = device.Allocate((long)state0.Length * sizeof(float));
        using var qBuf = device.Allocate((long)q.Length * sizeof(float));
        using var kBuf = device.Allocate((long)k.Length * sizeof(float));
        using var vBuf = device.Allocate((long)v.Length * sizeof(float));
        using var gBuf = device.Allocate((long)g.Length * sizeof(float));
        using var betaBuf = device.Allocate((long)beta.Length * sizeof(float));
        using var outBuf = device.Allocate((long)seqLen * nVHead * dState * sizeof(float));
        device.Upload(state0.AsSpan(), stateBuf);
        device.Upload(q.AsSpan(), qBuf);
        device.Upload(k.AsSpan(), kBuf);
        device.Upload(v.AsSpan(), vBuf);
        device.Upload(g.AsSpan(), gBuf);
        device.Upload(beta.AsSpan(), betaBuf);

        kernel.Launch(stateBuf, qBuf, kBuf, vBuf, gBuf, betaBuf, outBuf,
            seqLen, nVHead, nKHead, dState);

        float[] outState = new float[state0.Length];
        float[] outVals = new float[seqLen * nVHead * dState];
        device.Download(stateBuf, outState);
        device.Download(outBuf, outVals);
        return (outState, outVals);
    }

    private static void AssertBitIdentical(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        int mismatches = 0;
        int firstIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]))
                continue;
            if (firstIdx < 0) firstIdx = i;
            mismatches++;
        }

        Assert.True(mismatches == 0,
            $"{label}: {mismatches}/{expected.Length} elements differ from the shipping kernel; " +
            (firstIdx >= 0
                ? $"first at [{firstIdx}] expected {expected[firstIdx]:R} actual {actual[firstIdx]:R}"
                : string.Empty));
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
