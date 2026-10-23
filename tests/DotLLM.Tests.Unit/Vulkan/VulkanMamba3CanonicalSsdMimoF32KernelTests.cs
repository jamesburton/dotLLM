using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Mamba-3 canonical SSD scan (MIMO) Vulkan kernel.
/// </summary>
/// <remarks>
/// <para>
/// Reference is <see cref="Mamba3CanonicalSsd.ExecuteMimo"/> (scalar CPU).
/// Tolerance is abs 1e-3 / rel 1e-3 — exp + the rank-summed n-loop recurrence
/// accumulate F32 rounding faster than the pointwise kernels; the per-thread
/// loop order matches CPU but FMA emission can shift the last bits.
/// </para>
/// <para>
/// Inputs are generated with bounded magnitudes so the scan stays numerically
/// tame. <c>adt</c> is forced negative (so <c>decay = exp(adt) ∈ (0, 1]</c>),
/// and qRoped/kRoped sit in <c>[-0.5, 0.5]</c> (post-RoPE values are bounded
/// by the unit RoPE rotation).
/// </para>
/// <para>
/// State persistence: a single <c>seqLen=8</c> call must produce
/// bit-identical output to two consecutive <c>seqLen=4</c> calls on the
/// same state buffer.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMamba3CanonicalSsdMimoF32KernelTests
{
    private const float AbsTol = 1e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 2, 2, 4, 8, false)]    // smallest MIMO, no gate
    [InlineData(4, 2, 2, 4, 8, true)]     // multi-token, with gate
    [InlineData(1, 4, 4, 8, 16, true)]    // larger rank, decode
    [InlineData(4, 4, 4, 8, 16, true)]    // larger rank, prefill
    [InlineData(1, 2, 8, 64, 128, true)]  // Mamba-3 realistic decode
    public void Launch_MatchesCpuReference(int seqLen, int nRank, int nHead, int headDim, int dState, bool hasZ)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x3A22 ^ (seqLen * 131) ^ (nRank * 97) ^ (nHead * 71)
                             ^ (headDim * 53) ^ (dState * 23) ^ (hasZ ? 7 : 0));

        float[] state0      = SmallRandom(rng, nHead * headDim * dState);
        float[] v           = SmallRandom(rng, seqLen * nHead * headDim);
        float[] qRoped      = MidRandom(rng, seqLen * nRank * nHead * dState);     // [-0.5, 0.5]
        float[] kRoped      = MidRandom(rng, seqLen * nRank * nHead * dState);
        float[] qkPreDotSum = SmallRandom(rng, seqLen * nHead);
        float[] scale       = SmallRandom(rng, seqLen * nHead);
        float[] gamma       = SmallRandom(rng, seqLen * nHead);
        float[] adt         = NegativeRandom(rng, seqLen * nHead);
        float[] d           = SmallRandom(rng, nHead);
        float[] z           = SmallRandom(rng, seqLen * nHead * headDim);
        float[] mimoZ       = SmallRandom(rng, nHead * nRank * headDim);
        float[] mimoO       = SmallRandom(rng, nHead * nRank * headDim);

        // CPU reference. yPerRank is OUT OF SCOPE for this kernel — pass empty.
        float[] stateCpu = (float[])state0.Clone();
        float[] yCpu = new float[seqLen * nHead * headDim];
        Mamba3CanonicalSsd.ExecuteMimo(
            stateCpu, v, qRoped, kRoped, qkPreDotSum,
            scale, gamma, adt, d,
            hasZ ? z : ReadOnlySpan<float>.Empty,
            mimoZ, mimoO,
            yCpu,
            Span<float>.Empty,
            seqLen, nRank, nHead, headDim, dState);

        // GPU run.
        using var device = VulkanDevice.Create();
        using var kernel = Mamba3CanonicalSsdMimoF32Kernel.Create(device, spvDir);

        using var bufState       = device.Allocate((long)state0.Length      * sizeof(float));
        using var bufV           = device.Allocate((long)v.Length           * sizeof(float));
        using var bufQRoped      = device.Allocate((long)qRoped.Length      * sizeof(float));
        using var bufKRoped      = device.Allocate((long)kRoped.Length      * sizeof(float));
        using var bufQkPreDotSum = device.Allocate((long)qkPreDotSum.Length * sizeof(float));
        using var bufScale       = device.Allocate((long)scale.Length       * sizeof(float));
        using var bufGamma       = device.Allocate((long)gamma.Length       * sizeof(float));
        using var bufAdt         = device.Allocate((long)adt.Length         * sizeof(float));
        using var bufD           = device.Allocate((long)d.Length           * sizeof(float));
        using var bufZ           = device.Allocate((long)z.Length           * sizeof(float));
        using var bufMimoZ       = device.Allocate((long)mimoZ.Length       * sizeof(float));
        using var bufMimoO       = device.Allocate((long)mimoO.Length       * sizeof(float));
        using var bufY           = device.Allocate((long)yCpu.Length        * sizeof(float));

        device.Upload(state0,      bufState);
        device.Upload(v,           bufV);
        device.Upload(qRoped,      bufQRoped);
        device.Upload(kRoped,      bufKRoped);
        device.Upload(qkPreDotSum, bufQkPreDotSum);
        device.Upload(scale,       bufScale);
        device.Upload(gamma,       bufGamma);
        device.Upload(adt,         bufAdt);
        device.Upload(d,           bufD);
        device.Upload(z,           bufZ);
        device.Upload(mimoZ,       bufMimoZ);
        device.Upload(mimoO,       bufMimoO);

        kernel.Launch(bufState, bufV, bufQRoped, bufKRoped, bufQkPreDotSum,
                      bufScale, bufGamma, bufAdt, bufD, bufZ,
                      bufMimoZ, bufMimoO, bufY,
                      seqLen, nRank, nHead, headDim, dState, hasZ);

        float[] yGpu = new float[yCpu.Length];
        float[] stateGpu = new float[stateCpu.Length];
        device.Download(bufY, yGpu);
        device.Download(bufState, stateGpu);

        // y parity.
        for (int i = 0; i < yCpu.Length; i++)
        {
            float diff = MathF.Abs(yCpu[i] - yGpu[i]);
            float bar = AbsTol + RelTol * MathF.Abs(yCpu[i]);
            Assert.True(diff <= bar,
                $"y[{i}]: cpu={yCpu[i]:F6} vs vulkan={yGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        // State parity (in-place RW path).
        for (int i = 0; i < stateCpu.Length; i++)
        {
            float diff = MathF.Abs(stateCpu[i] - stateGpu[i]);
            float bar = AbsTol + RelTol * MathF.Abs(stateCpu[i]);
            Assert.True(diff <= bar,
                $"state[{i}]: cpu={stateCpu[i]:F6} vs vulkan={stateGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void Launch_StatePersistsAcrossCalls()
    {
        // Splitting a seqLen=8 scan into two seqLen=4 calls on the same state
        // buffer must produce BIT-IDENTICAL y rows as a single seqLen=8 call.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int nRank = 2, nHead = 4, headDim = 8, dState = 16, seqLen = 8;
        const bool hasZ = true;

        var rng = new Random(unchecked((int)0xC0FFEE3B));
        float[] state0      = SmallRandom(rng, nHead * headDim * dState);
        float[] v           = SmallRandom(rng, seqLen * nHead * headDim);
        float[] qRoped      = MidRandom(rng, seqLen * nRank * nHead * dState);
        float[] kRoped      = MidRandom(rng, seqLen * nRank * nHead * dState);
        float[] qkPreDotSum = SmallRandom(rng, seqLen * nHead);
        float[] scale       = SmallRandom(rng, seqLen * nHead);
        float[] gamma       = SmallRandom(rng, seqLen * nHead);
        float[] adt         = NegativeRandom(rng, seqLen * nHead);
        float[] d           = SmallRandom(rng, nHead);
        float[] z           = SmallRandom(rng, seqLen * nHead * headDim);
        float[] mimoZ       = SmallRandom(rng, nHead * nRank * headDim);
        float[] mimoO       = SmallRandom(rng, nHead * nRank * headDim);

        using var device = VulkanDevice.Create();
        using var kernel = Mamba3CanonicalSsdMimoF32Kernel.Create(device, spvDir);

        using var bufState       = device.Allocate((long)state0.Length      * sizeof(float));
        using var bufV           = device.Allocate((long)v.Length           * sizeof(float));
        using var bufQRoped      = device.Allocate((long)qRoped.Length      * sizeof(float));
        using var bufKRoped      = device.Allocate((long)kRoped.Length      * sizeof(float));
        using var bufQkPreDotSum = device.Allocate((long)qkPreDotSum.Length * sizeof(float));
        using var bufScale       = device.Allocate((long)scale.Length       * sizeof(float));
        using var bufGamma       = device.Allocate((long)gamma.Length       * sizeof(float));
        using var bufAdt         = device.Allocate((long)adt.Length         * sizeof(float));
        using var bufD           = device.Allocate((long)d.Length           * sizeof(float));
        using var bufZ           = device.Allocate((long)z.Length           * sizeof(float));
        using var bufMimoZ       = device.Allocate((long)mimoZ.Length       * sizeof(float));
        using var bufMimoO       = device.Allocate((long)mimoO.Length       * sizeof(float));
        using var bufY           = device.Allocate((long)seqLen * nHead * headDim * sizeof(float));

        // 1. One-shot seqLen=8 baseline.
        device.Upload(state0,      bufState);
        device.Upload(v,           bufV);
        device.Upload(qRoped,      bufQRoped);
        device.Upload(kRoped,      bufKRoped);
        device.Upload(qkPreDotSum, bufQkPreDotSum);
        device.Upload(scale,       bufScale);
        device.Upload(gamma,       bufGamma);
        device.Upload(adt,         bufAdt);
        device.Upload(d,           bufD);
        device.Upload(z,           bufZ);
        device.Upload(mimoZ,       bufMimoZ);
        device.Upload(mimoO,       bufMimoO);

        kernel.Launch(bufState, bufV, bufQRoped, bufKRoped, bufQkPreDotSum,
                      bufScale, bufGamma, bufAdt, bufD, bufZ,
                      bufMimoZ, bufMimoO, bufY,
                      seqLen, nRank, nHead, headDim, dState, hasZ);
        float[] yOneShot = new float[seqLen * nHead * headDim];
        float[] stateOneShot = new float[state0.Length];
        device.Download(bufY, yOneShot);
        device.Download(bufState, stateOneShot);

        // 2. Two seqLen=4 calls on the same state buffer.
        device.Upload(state0, bufState);

        const int half = 4;
        const int bcRowPerToken = nRank * nHead * dState;
        const int hdrPerToken   = nHead;
        const int vRowPerToken  = nHead * headDim;

        using var bufVHalf           = device.Allocate((long)half * vRowPerToken  * sizeof(float));
        using var bufQRopedHalf      = device.Allocate((long)half * bcRowPerToken * sizeof(float));
        using var bufKRopedHalf      = device.Allocate((long)half * bcRowPerToken * sizeof(float));
        using var bufQkPreDotSumHalf = device.Allocate((long)half * hdrPerToken   * sizeof(float));
        using var bufScaleHalf       = device.Allocate((long)half * hdrPerToken   * sizeof(float));
        using var bufGammaHalf       = device.Allocate((long)half * hdrPerToken   * sizeof(float));
        using var bufAdtHalf         = device.Allocate((long)half * hdrPerToken   * sizeof(float));
        using var bufZHalf           = device.Allocate((long)half * vRowPerToken  * sizeof(float));
        using var bufYHalf           = device.Allocate((long)half * vRowPerToken  * sizeof(float));

        // First half: tokens 0..3.
        device.Upload(v.AsSpan(0, half * vRowPerToken).ToArray(),                bufVHalf);
        device.Upload(qRoped.AsSpan(0, half * bcRowPerToken).ToArray(),          bufQRopedHalf);
        device.Upload(kRoped.AsSpan(0, half * bcRowPerToken).ToArray(),          bufKRopedHalf);
        device.Upload(qkPreDotSum.AsSpan(0, half * hdrPerToken).ToArray(),       bufQkPreDotSumHalf);
        device.Upload(scale.AsSpan(0, half * hdrPerToken).ToArray(),             bufScaleHalf);
        device.Upload(gamma.AsSpan(0, half * hdrPerToken).ToArray(),             bufGammaHalf);
        device.Upload(adt.AsSpan(0, half * hdrPerToken).ToArray(),               bufAdtHalf);
        device.Upload(z.AsSpan(0, half * vRowPerToken).ToArray(),                bufZHalf);

        kernel.Launch(bufState, bufVHalf, bufQRopedHalf, bufKRopedHalf, bufQkPreDotSumHalf,
                      bufScaleHalf, bufGammaHalf, bufAdtHalf, bufD, bufZHalf,
                      bufMimoZ, bufMimoO, bufYHalf,
                      half, nRank, nHead, headDim, dState, hasZ);
        float[] yFirstHalf = new float[half * vRowPerToken];
        device.Download(bufYHalf, yFirstHalf);

        // Second half: tokens 4..7.
        device.Upload(v.AsSpan(half * vRowPerToken, half * vRowPerToken).ToArray(),           bufVHalf);
        device.Upload(qRoped.AsSpan(half * bcRowPerToken, half * bcRowPerToken).ToArray(),    bufQRopedHalf);
        device.Upload(kRoped.AsSpan(half * bcRowPerToken, half * bcRowPerToken).ToArray(),    bufKRopedHalf);
        device.Upload(qkPreDotSum.AsSpan(half * hdrPerToken, half * hdrPerToken).ToArray(),   bufQkPreDotSumHalf);
        device.Upload(scale.AsSpan(half * hdrPerToken, half * hdrPerToken).ToArray(),         bufScaleHalf);
        device.Upload(gamma.AsSpan(half * hdrPerToken, half * hdrPerToken).ToArray(),         bufGammaHalf);
        device.Upload(adt.AsSpan(half * hdrPerToken, half * hdrPerToken).ToArray(),           bufAdtHalf);
        device.Upload(z.AsSpan(half * vRowPerToken, half * vRowPerToken).ToArray(),           bufZHalf);

        kernel.Launch(bufState, bufVHalf, bufQRopedHalf, bufKRopedHalf, bufQkPreDotSumHalf,
                      bufScaleHalf, bufGammaHalf, bufAdtHalf, bufD, bufZHalf,
                      bufMimoZ, bufMimoO, bufYHalf,
                      half, nRank, nHead, headDim, dState, hasZ);
        float[] ySecondHalf = new float[half * vRowPerToken];
        float[] stateSplit = new float[state0.Length];
        device.Download(bufYHalf, ySecondHalf);
        device.Download(bufState, stateSplit);

        // 3. Compare: bit-identical (no tolerance).
        for (int i = 0; i < half * vRowPerToken; i++)
            Assert.Equal(yOneShot[i], yFirstHalf[i]);
        for (int i = 0; i < half * vRowPerToken; i++)
            Assert.Equal(yOneShot[half * vRowPerToken + i], ySecondHalf[i]);
        for (int i = 0; i < state0.Length; i++)
            Assert.Equal(stateOneShot[i], stateSplit[i]);
    }

    /// <summary>U(-0.1, 0.1) — small magnitudes for state/scalars.</summary>
    private static float[] SmallRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return arr;
    }

    /// <summary>U(-0.5, 0.5) — bounded post-RoPE qRoped/kRoped magnitudes.</summary>
    private static float[] MidRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 1.0 - 0.5);
        return arr;
    }

    /// <summary>U(-0.5, -0.05) — adt = _A·DT is negative, so decay = exp(adt) ∈ (0, 1].</summary>
    private static float[] NegativeRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(-(rng.NextDouble() * 0.45 + 0.05));
        return arr;
    }
}
