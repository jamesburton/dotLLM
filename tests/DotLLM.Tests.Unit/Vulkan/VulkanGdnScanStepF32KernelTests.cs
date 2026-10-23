using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <c>gdn_scan_step_f32</c> — the SINGLE-TOKEN Gated DeltaNet
/// recurrence step, i.e. the DECODE path of Qwen3MoeHybrid / Qwen3HybridDense
/// GDN layers. Closes the coverage hole reported in issue #309.
/// </summary>
/// <remarks>
/// <para>
/// The multi-token scan (<see cref="VulkanGdnScanMultiTokenF32KernelTests"/>)
/// was already covered; the single-step kernel was not. That asymmetry is
/// exactly where a bug reaches generation but never prefill, so both directions
/// are pinned here:
/// </para>
/// <list type="number">
///   <item><b>vs the CPU oracle</b> — N consecutive steps driven exactly as
///     <c>ForwardGdnBody</c> drives them, checking BOTH the output and the
///     carried recurrence state after every step (a state bug that only shows up
///     on step 2+ is the classic decode-only failure).</item>
///   <item><b>vs the multi-token scan</b> — N single steps over the same inputs
///     must reproduce the multi-token kernel's whole-sequence result. This is a
///     GPU-to-GPU cross-check that needs no oracle at all.</item>
/// </list>
/// <para>
/// Tolerance: see <see cref="AssertCloseUlp"/> — an RMS-scaled absolute bound, not
/// a ULP bound. Both implementations fold
/// rows in the same order (thread <c>tid</c> owns column <c>tid</c> and iterates
/// <c>row = 0..dState-1</c>, which is the CPU's inner col-loop order), so the
/// admissible drift is only FMA contraction / <c>sqrt</c> rounding.
/// </para>
/// <para>
/// Shapes include <c>nVHead/nKHead == 2</c> with distinct per-<c>kh</c> q/k so a
/// regression from the TILED head broadcast (<c>kh = vh % nKHead</c>) to the
/// interleaved form (<c>vh / vHeadsPerKHead</c>) — the bug called out in
/// CLAUDE.md's cross-backend rule — is discriminated rather than aliased away.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanGdnScanStepF32KernelTests
{
    [SkippableTheory]
    [InlineData(4, 2, 16, 5)]     // tiled discriminator: NVHead/NKHead = 2, several steps
    [InlineData(6, 3, 32, 4)]     // non-power-of-two head counts
    [InlineData(12, 4, 24, 3)]    // NVHead/NKHead = 3, dState not a power of two
    [InlineData(32, 16, 128, 4)]  // Qwen3.6-A3B production shape
    public void Step_MatchesCpuReference_AcrossMultipleSteps(int nVHead, int nKHead, int dState, int steps)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_scan_step_f32.spv")),
            "gdn_scan_step_f32.spv not compiled (glslc / Vulkan SDK required).");

        var rng = new Random(0x6D45 ^ (nVHead * 1009) ^ (nKHead << 8) ^ (dState << 16) ^ steps);
        int qkPerToken = nKHead * dState;
        int vPerToken = nVHead * dState;

        float[] cpuState = RandomFloats(rng, nVHead * dState * dState, 0.1f);
        float[] gpuState0 = (float[])cpuState.Clone();

        using var device = VulkanDevice.Create();
        using var kernel = GdnScanStepF32Kernel.Create(device, spvDir);

        using var stateBuf = device.Allocate((long)cpuState.Length * sizeof(float));
        using var qBuf = device.Allocate((long)qkPerToken * sizeof(float));
        using var kBuf = device.Allocate((long)qkPerToken * sizeof(float));
        using var vBuf = device.Allocate((long)vPerToken * sizeof(float));
        using var gBuf = device.Allocate((long)nVHead * sizeof(float));
        using var betaBuf = device.Allocate((long)nVHead * sizeof(float));
        using var outBuf = device.Allocate((long)vPerToken * sizeof(float));
        device.Upload(gpuState0.AsSpan(), stateBuf);

        float[] cpuOut = new float[vPerToken];

        for (int step = 0; step < steps; step++)
        {
            // Distinct random q/k per kh — an interleaved-vs-tiled head-broadcast
            // bug changes which kh a given vh reads and therefore the result.
            float[] q = RandomFloats(rng, qkPerToken, 1.0f);
            float[] k = RandomFloats(rng, qkPerToken, 1.0f);
            float[] v = RandomFloats(rng, vPerToken, 1.0f);
            float[] g = new float[nVHead];
            float[] beta = new float[nVHead];
            for (int i = 0; i < nVHead; i++) g[i] = 0.5f + 0.5f * (float)rng.NextDouble();
            for (int i = 0; i < nVHead; i++) beta[i] = (float)rng.NextDouble();

            GatedDeltaNetScan.Execute(cpuState, q, k, v, g, beta, cpuOut,
                nVHead, nKHead, dState, seqLen: 1);

            device.Upload(q.AsSpan(), qBuf);
            device.Upload(k.AsSpan(), kBuf);
            device.Upload(v.AsSpan(), vBuf);
            device.Upload(g.AsSpan(), gBuf);
            device.Upload(beta.AsSpan(), betaBuf);

            kernel.Launch(stateBuf, qBuf, kBuf, vBuf, gBuf, betaBuf, outBuf, nVHead, nKHead, dState);

            float[] gpuOut = new float[vPerToken];
            float[] gpuState = new float[cpuState.Length];
            device.Download(outBuf, gpuOut);
            device.Download(stateBuf, gpuState);

            AssertCloseUlp(cpuOut, gpuOut, $"output @ step {step} (nVHead={nVHead},nKHead={nKHead},dState={dState})");
            AssertCloseUlp(cpuState, gpuState, $"state @ step {step} (nVHead={nVHead},nKHead={nKHead},dState={dState})");
        }
    }

    /// <summary>
    /// The decode path stepped <c>seqLen</c> times must reproduce the prefill
    /// path run once over the whole sequence. A GPU-to-GPU cross-check with no
    /// oracle involved: it fails if either kernel drifts from the other, which
    /// is precisely the prefill/decode divergence issue #309 flags.
    /// </summary>
    [SkippableTheory]
    [InlineData(4, 2, 16, 5)]
    [InlineData(6, 3, 32, 4)]
    [InlineData(32, 16, 128, 3)]
    public void NSingleSteps_MatchMultiTokenScan(int nVHead, int nKHead, int dState, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_scan_step_f32.spv")),
            "gdn_scan_step_f32.spv not compiled (glslc / Vulkan SDK required).");
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "gdn_scan_multi_token_f32.spv")),
            "gdn_scan_multi_token_f32.spv not compiled (glslc / Vulkan SDK required).");

        var rng = new Random(0x5EE9 ^ (nVHead * 1009) ^ (nKHead << 8) ^ (dState << 16) ^ seqLen);
        int qkPerToken = nKHead * dState;
        int vPerToken = nVHead * dState;

        float[] state0 = RandomFloats(rng, nVHead * dState * dState, 0.1f);
        float[] q = RandomFloats(rng, seqLen * qkPerToken, 1.0f);
        float[] k = RandomFloats(rng, seqLen * qkPerToken, 1.0f);
        float[] v = RandomFloats(rng, seqLen * vPerToken, 1.0f);
        float[] g = new float[seqLen * nVHead];
        float[] beta = new float[seqLen * nVHead];
        for (int i = 0; i < g.Length; i++) g[i] = 0.5f + 0.5f * (float)rng.NextDouble();
        for (int i = 0; i < beta.Length; i++) beta[i] = (float)rng.NextDouble();

        using var device = VulkanDevice.Create();

        // ── Prefill: one multi-token dispatch over the whole sequence ────────
        float[] multiOut = new float[seqLen * vPerToken];
        float[] multiState = new float[state0.Length];
        using (var multi = GdnScanMultiTokenF32Kernel.Create(device, spvDir))
        using (var sBuf = device.Allocate((long)state0.Length * sizeof(float)))
        using (var qBuf = device.Allocate((long)q.Length * sizeof(float)))
        using (var kBuf = device.Allocate((long)k.Length * sizeof(float)))
        using (var vBuf = device.Allocate((long)v.Length * sizeof(float)))
        using (var gBuf = device.Allocate((long)g.Length * sizeof(float)))
        using (var bBuf = device.Allocate((long)beta.Length * sizeof(float)))
        using (var oBuf = device.Allocate((long)multiOut.Length * sizeof(float)))
        {
            device.Upload(state0.AsSpan(), sBuf);
            device.Upload(q.AsSpan(), qBuf);
            device.Upload(k.AsSpan(), kBuf);
            device.Upload(v.AsSpan(), vBuf);
            device.Upload(g.AsSpan(), gBuf);
            device.Upload(beta.AsSpan(), bBuf);
            multi.Launch(sBuf, qBuf, kBuf, vBuf, gBuf, bBuf, oBuf, seqLen, nVHead, nKHead, dState);
            device.Download(oBuf, multiOut);
            device.Download(sBuf, multiState);
        }

        // ── Decode: seqLen single-step dispatches over the same inputs ───────
        float[] stepOut = new float[seqLen * vPerToken];
        float[] stepState = new float[state0.Length];
        using (var step = GdnScanStepF32Kernel.Create(device, spvDir))
        using (var sBuf = device.Allocate((long)state0.Length * sizeof(float)))
        using (var qBuf = device.Allocate((long)qkPerToken * sizeof(float)))
        using (var kBuf = device.Allocate((long)qkPerToken * sizeof(float)))
        using (var vBuf = device.Allocate((long)vPerToken * sizeof(float)))
        using (var gBuf = device.Allocate((long)nVHead * sizeof(float)))
        using (var bBuf = device.Allocate((long)nVHead * sizeof(float)))
        using (var oBuf = device.Allocate((long)vPerToken * sizeof(float)))
        {
            device.Upload(state0.AsSpan(), sBuf);
            for (int t = 0; t < seqLen; t++)
            {
                device.Upload(q.AsSpan(t * qkPerToken, qkPerToken), qBuf);
                device.Upload(k.AsSpan(t * qkPerToken, qkPerToken), kBuf);
                device.Upload(v.AsSpan(t * vPerToken, vPerToken), vBuf);
                device.Upload(g.AsSpan(t * nVHead, nVHead), gBuf);
                device.Upload(beta.AsSpan(t * nVHead, nVHead), bBuf);

                step.Launch(sBuf, qBuf, kBuf, vBuf, gBuf, bBuf, oBuf, nVHead, nKHead, dState);

                device.Download(oBuf, stepOut.AsSpan(t * vPerToken, vPerToken));
            }
            device.Download(sBuf, stepState);
        }

        AssertCloseUlp(multiOut, stepOut,
            $"multi-token vs {seqLen} single steps, output (nVHead={nVHead},nKHead={nKHead},dState={dState})");
        AssertCloseUlp(multiState, stepState,
            $"multi-token vs {seqLen} single steps, state (nVHead={nVHead},nKHead={nKHead},dState={dState})");
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// Magnitude-scaled comparison.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A per-element ULP or relative bar is the WRONG metric for this kernel and
    /// would have to be widened absurdly to pass. Every output is a
    /// <c>dState</c>-long dot product <c>Σ_row S[row,col]·q[row]</c> over terms of
    /// magnitude <c>O(‖S‖·‖q‖)</c> whose sum frequently cancels to near zero. The
    /// rounding error of such a sum scales with the TERM magnitude, not with the
    /// result, so a result that happens to land at 1e-5 while its summands are
    /// O(1) carries thousands of ULPs of perfectly benign error.
    /// </para>
    /// <para>
    /// The bar is therefore absolute, sized to the RMS of the tensor being
    /// compared (a proxy for the summand magnitude) at <c>1e-5</c> relative —
    /// i.e. ~170× the drift a single FMA-contraction/re-association difference
    /// produces over 128 F32 accumulations (<c>√128·ε ≈ 7e-7</c> relative),
    /// and orders of magnitude below any structural error, which shifts results
    /// by whole percent. Elements are also allowed to pass on a 1e-5 per-element
    /// relative bound, which is the binding constraint for the large outputs.
    /// </para>
    /// </remarks>
    private static void AssertCloseUlp(float[] expected, float[] actual, string label)
    {
        const float RelTol = 1e-5f;

        Assert.Equal(expected.Length, actual.Length);

        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, expected.Length));
        float absTol = MathF.Max(rms, 1e-20f) * RelTol;

        int violations = 0, worst = -1;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-20f);
            if (diff > maxAbs) { maxAbs = diff; worst = i; }
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > RelTol) violations++;
        }
        Assert.True(violations == 0,
            $"GdnScanStep {label} drift exceeded tolerance: violations={violations}/{expected.Length}, " +
            $"maxAbs={maxAbs:G9} at [{worst}] (expected={(worst >= 0 ? expected[worst] : 0):G9}, " +
            $"actual={(worst >= 0 ? actual[worst] : 0):G9}), maxRel={maxRel:G9}, absTol={absTol:G9}");
    }
}
