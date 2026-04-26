using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Mamba-3 canonical SSD scan (SISO) Vulkan kernel.
/// </summary>
/// <remarks>
/// <para>
/// Reference is <see cref="Mamba3CanonicalSsd.ExecuteSiso"/> (scalar CPU).
/// Tolerance is abs 1e-3 / rel 1e-3 — exp + the inner n-loop recurrence
/// accumulate F32 rounding faster than the pointwise kernels; the per-thread
/// loop order is identical to the CPU reference, but exp / 1/(1+exp(-z)) fast
/// paths and FMA emission can shift the last bits of the running state.
/// </para>
/// <para>
/// Inputs are generated with small magnitudes so the scan stays numerically
/// tame. <c>adt</c> is forced negative (the GGUF <c>_A</c> parameter is stored
/// negative and <c>DT</c> is positive, so <c>adt = _A * DT</c> ends up
/// negative-clamped, yielding <c>decay = exp(adt) ∈ (0, 1]</c> and a
/// state magnitude that stays bounded over multi-token sequences).
/// </para>
/// <para>
/// State persistence: a single <c>seqLen=8</c> call must produce
/// bit-identical output to two consecutive <c>seqLen=4</c> calls on the
/// same state buffer (the kernel does the exact same sequence of ops in
/// either case).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMamba3CanonicalSsdSisoF32KernelTests
{
    private const float AbsTol = 1e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 2, 4, 8, false)]           // smallest, no gate
    [InlineData(4, 4, 8, 16, false)]          // multi-token, no gate
    [InlineData(1, 2, 4, 8, true)]            // smallest, with silu(z) gate
    [InlineData(4, 4, 8, 16, true)]           // multi-token, with silu(z) gate
    [InlineData(1, 8, 64, 128, true)]         // Mamba-3 realistic decode
    [InlineData(4, 8, 64, 128, true)]         // Mamba-3 realistic prefill
    public void Launch_MatchesCpuReference(int seqLen, int nHead, int headDim, int dState, bool hasZ)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x3A11 ^ (seqLen * 131) ^ (nHead * 71)
                             ^ (headDim * 53) ^ (dState * 23) ^ (hasZ ? 7 : 0));

        float[] state0   = SmallRandom(rng, nHead * headDim * dState);
        float[] v        = SmallRandom(rng, seqLen * nHead * headDim);
        float[] qRoped   = SmallRandom(rng, seqLen * nHead * dState);
        float[] kRoped   = SmallRandom(rng, seqLen * nHead * dState);
        float[] qkPreDot = SmallRandom(rng, seqLen * nHead);
        float[] scale    = SmallRandom(rng, seqLen * nHead);
        float[] gamma    = SmallRandom(rng, seqLen * nHead);
        float[] adt      = NegativeRandom(rng, seqLen * nHead); // _A * DT is negative-clamped
        float[] d        = SmallRandom(rng, nHead);
        float[] z        = SmallRandom(rng, seqLen * nHead * headDim);

        // CPU reference — operates on a copy of state0 so we can re-use the
        // same initial state for the GPU run. Empty-z span signals "no gate".
        float[] stateCpu = (float[])state0.Clone();
        float[] yCpu = new float[seqLen * nHead * headDim];
        Mamba3CanonicalSsd.ExecuteSiso(
            stateCpu, v, qRoped, kRoped, qkPreDot,
            scale, gamma, adt, d,
            hasZ ? z : ReadOnlySpan<float>.Empty,
            yCpu,
            seqLen, nHead, headDim, dState);

        // GPU run.
        using var device = VulkanDevice.Create();
        using var kernel = Mamba3CanonicalSsdSisoF32Kernel.Create(device, spvDir);

        using var bufState    = device.Allocate((long)state0.Length    * sizeof(float));
        using var bufV        = device.Allocate((long)v.Length         * sizeof(float));
        using var bufQRoped   = device.Allocate((long)qRoped.Length    * sizeof(float));
        using var bufKRoped   = device.Allocate((long)kRoped.Length    * sizeof(float));
        using var bufQkPreDot = device.Allocate((long)qkPreDot.Length  * sizeof(float));
        using var bufScale    = device.Allocate((long)scale.Length     * sizeof(float));
        using var bufGamma    = device.Allocate((long)gamma.Length     * sizeof(float));
        using var bufAdt      = device.Allocate((long)adt.Length       * sizeof(float));
        using var bufD        = device.Allocate((long)d.Length         * sizeof(float));
        using var bufZ        = device.Allocate((long)z.Length         * sizeof(float));
        using var bufY        = device.Allocate((long)yCpu.Length      * sizeof(float));

        device.Upload(state0,   bufState);
        device.Upload(v,        bufV);
        device.Upload(qRoped,   bufQRoped);
        device.Upload(kRoped,   bufKRoped);
        device.Upload(qkPreDot, bufQkPreDot);
        device.Upload(scale,    bufScale);
        device.Upload(gamma,    bufGamma);
        device.Upload(adt,      bufAdt);
        device.Upload(d,        bufD);
        device.Upload(z,        bufZ);

        kernel.Launch(bufState, bufV, bufQRoped, bufKRoped, bufQkPreDot,
                      bufScale, bufGamma, bufAdt, bufD, bufZ, bufY,
                      seqLen, nHead, headDim, dState, hasZ);

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

        // State parity (checks the in-place RW path is correct).
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
        // buffer must produce the BIT-IDENTICAL y rows as a single seqLen=8
        // call (no tolerance). This is the property the decode loop relies on:
        // each token's call leaves state in the right shape for the next.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int nHead = 4, headDim = 8, dState = 16, seqLen = 8;
        const bool hasZ = true;

        var rng = new Random(unchecked((int)0xC0FFEE3A));
        float[] state0   = SmallRandom(rng, nHead * headDim * dState);
        float[] v        = SmallRandom(rng, seqLen * nHead * headDim);
        float[] qRoped   = SmallRandom(rng, seqLen * nHead * dState);
        float[] kRoped   = SmallRandom(rng, seqLen * nHead * dState);
        float[] qkPreDot = SmallRandom(rng, seqLen * nHead);
        float[] scale    = SmallRandom(rng, seqLen * nHead);
        float[] gamma    = SmallRandom(rng, seqLen * nHead);
        float[] adt      = NegativeRandom(rng, seqLen * nHead);
        float[] d        = SmallRandom(rng, nHead);
        float[] z        = SmallRandom(rng, seqLen * nHead * headDim);

        using var device = VulkanDevice.Create();
        using var kernel = Mamba3CanonicalSsdSisoF32Kernel.Create(device, spvDir);

        using var bufState    = device.Allocate((long)state0.Length   * sizeof(float));
        using var bufV        = device.Allocate((long)v.Length        * sizeof(float));
        using var bufQRoped   = device.Allocate((long)qRoped.Length   * sizeof(float));
        using var bufKRoped   = device.Allocate((long)kRoped.Length   * sizeof(float));
        using var bufQkPreDot = device.Allocate((long)qkPreDot.Length * sizeof(float));
        using var bufScale    = device.Allocate((long)scale.Length    * sizeof(float));
        using var bufGamma    = device.Allocate((long)gamma.Length    * sizeof(float));
        using var bufAdt      = device.Allocate((long)adt.Length      * sizeof(float));
        using var bufD        = device.Allocate((long)d.Length        * sizeof(float));
        using var bufZ        = device.Allocate((long)z.Length        * sizeof(float));
        using var bufY        = device.Allocate((long)seqLen * nHead * headDim * sizeof(float));

        // 1. One-shot seqLen=8 baseline.
        device.Upload(state0,   bufState);
        device.Upload(v,        bufV);
        device.Upload(qRoped,   bufQRoped);
        device.Upload(kRoped,   bufKRoped);
        device.Upload(qkPreDot, bufQkPreDot);
        device.Upload(scale,    bufScale);
        device.Upload(gamma,    bufGamma);
        device.Upload(adt,      bufAdt);
        device.Upload(d,        bufD);
        device.Upload(z,        bufZ);

        kernel.Launch(bufState, bufV, bufQRoped, bufKRoped, bufQkPreDot,
                      bufScale, bufGamma, bufAdt, bufD, bufZ, bufY,
                      seqLen, nHead, headDim, dState, hasZ);
        float[] yOneShot = new float[seqLen * nHead * headDim];
        float[] stateOneShot = new float[state0.Length];
        device.Download(bufY, yOneShot);
        device.Download(bufState, stateOneShot);

        // 2. Two seqLen=4 calls on the same state buffer.
        device.Upload(state0, bufState);

        const int half = 4;
        using var bufVHalf        = device.Allocate((long)half * nHead * headDim * sizeof(float));
        using var bufQRopedHalf   = device.Allocate((long)half * nHead * dState  * sizeof(float));
        using var bufKRopedHalf   = device.Allocate((long)half * nHead * dState  * sizeof(float));
        using var bufQkPreDotHalf = device.Allocate((long)half * nHead           * sizeof(float));
        using var bufScaleHalf    = device.Allocate((long)half * nHead           * sizeof(float));
        using var bufGammaHalf    = device.Allocate((long)half * nHead           * sizeof(float));
        using var bufAdtHalf      = device.Allocate((long)half * nHead           * sizeof(float));
        using var bufZHalf        = device.Allocate((long)half * nHead * headDim * sizeof(float));
        using var bufYHalf        = device.Allocate((long)half * nHead * headDim * sizeof(float));

        // First half: tokens 0..3.
        device.Upload(v.AsSpan(0, half * nHead * headDim).ToArray(),        bufVHalf);
        device.Upload(qRoped.AsSpan(0, half * nHead * dState).ToArray(),    bufQRopedHalf);
        device.Upload(kRoped.AsSpan(0, half * nHead * dState).ToArray(),    bufKRopedHalf);
        device.Upload(qkPreDot.AsSpan(0, half * nHead).ToArray(),           bufQkPreDotHalf);
        device.Upload(scale.AsSpan(0, half * nHead).ToArray(),              bufScaleHalf);
        device.Upload(gamma.AsSpan(0, half * nHead).ToArray(),              bufGammaHalf);
        device.Upload(adt.AsSpan(0, half * nHead).ToArray(),                bufAdtHalf);
        device.Upload(z.AsSpan(0, half * nHead * headDim).ToArray(),        bufZHalf);

        kernel.Launch(bufState, bufVHalf, bufQRopedHalf, bufKRopedHalf, bufQkPreDotHalf,
                      bufScaleHalf, bufGammaHalf, bufAdtHalf, bufD, bufZHalf, bufYHalf,
                      half, nHead, headDim, dState, hasZ);
        float[] yFirstHalf = new float[half * nHead * headDim];
        device.Download(bufYHalf, yFirstHalf);

        // Second half: tokens 4..7.
        device.Upload(v.AsSpan(half * nHead * headDim, half * nHead * headDim).ToArray(),        bufVHalf);
        device.Upload(qRoped.AsSpan(half * nHead * dState, half * nHead * dState).ToArray(),    bufQRopedHalf);
        device.Upload(kRoped.AsSpan(half * nHead * dState, half * nHead * dState).ToArray(),    bufKRopedHalf);
        device.Upload(qkPreDot.AsSpan(half * nHead, half * nHead).ToArray(),                    bufQkPreDotHalf);
        device.Upload(scale.AsSpan(half * nHead, half * nHead).ToArray(),                       bufScaleHalf);
        device.Upload(gamma.AsSpan(half * nHead, half * nHead).ToArray(),                       bufGammaHalf);
        device.Upload(adt.AsSpan(half * nHead, half * nHead).ToArray(),                         bufAdtHalf);
        device.Upload(z.AsSpan(half * nHead * headDim, half * nHead * headDim).ToArray(),        bufZHalf);

        kernel.Launch(bufState, bufVHalf, bufQRopedHalf, bufKRopedHalf, bufQkPreDotHalf,
                      bufScaleHalf, bufGammaHalf, bufAdtHalf, bufD, bufZHalf, bufYHalf,
                      half, nHead, headDim, dState, hasZ);
        float[] ySecondHalf = new float[half * nHead * headDim];
        float[] stateSplit = new float[state0.Length];
        device.Download(bufYHalf, ySecondHalf);
        device.Download(bufState, stateSplit);

        // 3. Compare the concatenated split output to the one-shot output.
        // Bit-identical: the kernel does the exact same sequence of ops in
        // either case (state is the only stateful buffer).
        for (int i = 0; i < half * nHead * headDim; i++)
            Assert.Equal(yOneShot[i], yFirstHalf[i]);
        for (int i = 0; i < half * nHead * headDim; i++)
            Assert.Equal(yOneShot[half * nHead * headDim + i], ySecondHalf[i]);
        for (int i = 0; i < state0.Length; i++)
            Assert.Equal(stateOneShot[i], stateSplit[i]);
    }

    /// <summary>U(-0.1, 0.1) — small magnitudes keep the recurrence numerically tame.</summary>
    private static float[] SmallRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        return arr;
    }

    /// <summary>U(-0.5, -0.05) — adt = _A·DT is negative-clamped, so decay = exp(adt) ∈ (0, 1].</summary>
    private static float[] NegativeRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(-(rng.NextDouble() * 0.45 + 0.05));
        return arr;
    }
}
