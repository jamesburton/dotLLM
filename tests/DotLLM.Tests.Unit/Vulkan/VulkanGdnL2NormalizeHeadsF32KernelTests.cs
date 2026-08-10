using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <c>gdn_l2_normalize_heads_f32</c> — the in-place per-head L2
/// normalisation applied to Q and K before the GDN delta-rule scan. Closes the
/// coverage hole reported in issue #309.
/// </summary>
/// <remarks>
/// <para>
/// Oracle is the production CPU routine
/// <see cref="GatedDeltaNetScan.L2NormalizeHeads"/> — same numerical tier (both
/// pure F32), so the only admissible divergence is that the shader reduces
/// <c>Σx²</c> with a shared-memory tree while the CPU sums sequentially.
/// Re-association over ≤128 terms bounds the relative error of <c>Σx²</c> at
/// roughly <c>log2(128)·ε ≈ 4e-7</c>, and that error passes through
/// <c>1/(√Σ + eps)</c> essentially halved, so a 1e-5 relative bar carries ~25×
/// margin without being able to hide a structural bug (which moves results by
/// whole percent or more).
/// </para>
/// <para>
/// Two discriminating aspects beyond the random sweep:
/// </para>
/// <list type="bullet">
///   <item><b>Per-head isolation</b> — head magnitudes are deliberately
///     spread over ~3 decades, so a kernel that normalised the whole buffer (or
///     used the wrong head base) produces grossly wrong values instead of
///     drifting within tolerance.</item>
///   <item><b>Epsilon placement</b> — the contract is
///     <c>1/(√Σx² + eps)</c>, NOT the RMSNorm-style <c>1/√(Σx² + eps)</c>. A
///     dedicated case uses a large eps against a small-magnitude head, where the
///     two forms differ by tens of percent.</item>
/// </list>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGdnL2NormalizeHeadsF32KernelTests
{
    private const float AbsTol = 1e-7f;
    private const float RelTol = 1e-5f;

    [SkippableTheory]
    [InlineData(1, 16)]      // single head, dState < workgroup
    [InlineData(5, 32)]      // odd head count
    [InlineData(7, 24)]      // dState not a power of two
    [InlineData(64, 128)]    // Qwen3.6-A3B dState, many heads
    [InlineData(33, 64)]     // odd head count, mid dState
    public void Launch_MatchesCpuReference(int totalHeads, int dState)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_l2_normalize_heads_f32.spv")),
            "gdn_l2_normalize_heads_f32.spv not compiled (glslc / Vulkan SDK required).");

        const float Eps = 1e-6f;
        var rng = new Random(0x12A0 ^ (totalHeads * 1009) ^ (dState << 12));
        float[] input = PerHeadScaledRandom(rng, totalHeads, dState);

        float[] expected = (float[])input.Clone();
        GatedDeltaNetScan.L2NormalizeHeads(expected, dState, Eps);

        using var device = VulkanDevice.Create();
        using var kernel = GdnL2NormalizeHeadsF32Kernel.Create(device, spvDir);
        using var buf = device.Allocate((long)input.Length * sizeof(float));
        device.Upload(input.AsSpan(), buf);

        kernel.Launch(buf, totalHeads, dState, Eps);

        float[] actual = new float[input.Length];
        device.Download(buf, actual);

        AssertClose(expected, actual, $"totalHeads={totalHeads}, dState={dState}");
    }

    /// <summary>
    /// Epsilon-placement contract: <c>invNorm = 1/(√Σx² + eps)</c>. With a large
    /// eps and a small-magnitude head the RMSNorm-style <c>1/√(Σx² + eps)</c>
    /// differs by tens of percent, so this case fails outright if the shader ever
    /// "tidies up" to <c>inversesqrt(sumSq + eps)</c>.
    /// </summary>
    [SkippableFact]
    public void Launch_AddsEpsilonOutsideTheSqrt()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_l2_normalize_heads_f32.spv")),
            "gdn_l2_normalize_heads_f32.spv not compiled (glslc / Vulkan SDK required).");

        const int totalHeads = 4;
        const int dState = 32;
        const float Eps = 0.5f;

        var rng = new Random(0xE95);
        float[] input = new float[totalHeads * dState];
        for (int i = 0; i < input.Length; i++)
            input[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.02); // ‖head‖ ≈ 0.065 ≪ eps

        float[] expected = (float[])input.Clone();
        GatedDeltaNetScan.L2NormalizeHeads(expected, dState, Eps);

        using var device = VulkanDevice.Create();
        using var kernel = GdnL2NormalizeHeadsF32Kernel.Create(device, spvDir);
        using var buf = device.Allocate((long)input.Length * sizeof(float));
        device.Upload(input.AsSpan(), buf);

        kernel.Launch(buf, totalHeads, dState, Eps);

        float[] actual = new float[input.Length];
        device.Download(buf, actual);

        // Guard the guard: confirm the two epsilon placements really do diverge
        // far beyond tolerance for this input, so the case is discriminating.
        float sumSq = 0;
        for (int i = 0; i < dState; i++) sumSq += input[i] * input[i];
        float outside = 1f / (MathF.Sqrt(sumSq) + Eps);
        float inside = 1f / MathF.Sqrt(sumSq + Eps);
        Assert.True(MathF.Abs(outside - inside) / outside > 0.1f,
            $"Test input is not discriminating: invNorm outside={outside:G6} vs inside={inside:G6}.");

        AssertClose(expected, actual, "epsilon-outside-sqrt contract");
    }

    /// <summary>Head magnitudes spread over ~3 decades — isolates per-head normalisation.</summary>
    private static float[] PerHeadScaledRandom(Random rng, int totalHeads, int dState)
    {
        var arr = new float[totalHeads * dState];
        for (int h = 0; h < totalHeads; h++)
        {
            float scale = MathF.Pow(10f, (h % 7) - 3); // 1e-3 … 1e3
            for (int i = 0; i < dState; i++)
                arr[h * dState + i] = (float)((rng.NextDouble() * 2.0 - 1.0) * scale);
        }
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"GdnL2NormalizeHeads drift exceeded tolerance ({label}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
