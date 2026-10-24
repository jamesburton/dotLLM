using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Mamba-2 selective-scan Vulkan kernel.
/// </summary>
/// <remarks>
/// <para>
/// Reference is <see cref="Mamba2SelectiveScan.Execute"/> (scalar CPU).
/// Tolerance is abs 1e-3 / rel 1e-3 — softplus + exp + the inner k-loop
/// recurrence accumulate F32 rounding faster than the pointwise kernels;
/// the per-thread loop order is identical to the CPU's, but exp/softplus
/// fast-paths and FMA emission can shift the last bits of the running
/// state across iterations.
/// </para>
/// <para>
/// Inputs are generated with small magnitudes (~U(-0.1, 0.1)) so the scan
/// stays numerically tame: A is forced negative (exp(dt*A) decays), and dt,
/// x, B, C are kept tiny so the running state magnitude does not blow up
/// over multi-token sequences.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMamba2SelectiveScanF32KernelTests
{
    private const float AbsTol = 1e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(2, 4, 8, 1, 1)]                  // smallest decode shape
    [InlineData(4, 8, 16, 2, 1)]                 // groups, single token
    [InlineData(4, 8, 16, 2, 4)]                 // groups, multi-token prefill
    [InlineData(10, 80, 128, 10, 1)]             // Nemotron-H-realistic decode
    [InlineData(10, 80, 128, 10, 8)]             // Nemotron-H-realistic prefill
    public void Launch_MatchesCpuReference(
        int nHead, int headDim, int dState, int nGroup, int seqLen)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x4A31 ^ (nHead * 131) ^ (headDim * 71)
                             ^ (dState * 53) ^ (nGroup * 23) ^ seqLen);

        int dInner = nHead * headDim;
        float[] state0 = SmallRandom(rng, nHead * headDim * dState);
        float[] x = SmallRandom(rng, seqLen * dInner);
        float[] dt = SmallRandom(rng, seqLen * nHead);
        float[] a = NegativeRandom(rng, nHead);   // GGUF stores A negative
        float[] b = SmallRandom(rng, seqLen * nGroup * dState);
        float[] c = SmallRandom(rng, seqLen * nGroup * dState);

        // CPU reference — operates on a copy of state0 so we can re-use the
        // same initial state for the GPU run.
        float[] stateCpu = (float[])state0.Clone();
        float[] yCpu = new float[seqLen * dInner];
        Mamba2SelectiveScan.Execute(
            stateCpu, x, dt, a, b, c, yCpu,
            nHead, headDim, dState, nGroup, seqLen);

        // GPU run.
        using var device = VulkanDevice.Create();
        using var kernel = Mamba2SelectiveScanF32Kernel.Create(device, spvDir);

        using var bufState = device.Allocate((long)state0.Length * sizeof(float));
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufDt = device.Allocate((long)dt.Length * sizeof(float));
        using var bufA = device.Allocate((long)a.Length * sizeof(float));
        using var bufB = device.Allocate((long)b.Length * sizeof(float));
        using var bufC = device.Allocate((long)c.Length * sizeof(float));
        using var bufY = device.Allocate((long)yCpu.Length * sizeof(float));

        device.Upload(state0, bufState);
        device.Upload(x, bufX);
        device.Upload(dt, bufDt);
        device.Upload(a, bufA);
        device.Upload(b, bufB);
        device.Upload(c, bufC);

        kernel.Launch(bufState, bufX, bufDt, bufA, bufB, bufC, bufY,
                      nHead, headDim, dState, nGroup, seqLen);

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
        // buffer must produce the same y rows (within tolerance) as a single
        // seqLen=8 call. This is the property the decode loop relies on:
        // each token's call leaves state in the right shape for the next.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int nHead = 4, headDim = 8, dState = 16, nGroup = 2, seqLen = 8;
        int dInner = nHead * headDim;

        var rng = new Random(unchecked((int)0xBEEFCAFE));
        float[] state0 = SmallRandom(rng, nHead * headDim * dState);
        float[] x = SmallRandom(rng, seqLen * dInner);
        float[] dt = SmallRandom(rng, seqLen * nHead);
        float[] a = NegativeRandom(rng, nHead);
        float[] b = SmallRandom(rng, seqLen * nGroup * dState);
        float[] c = SmallRandom(rng, seqLen * nGroup * dState);

        using var device = VulkanDevice.Create();
        using var kernel = Mamba2SelectiveScanF32Kernel.Create(device, spvDir);

        using var bufState = device.Allocate((long)state0.Length * sizeof(float));
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufDt = device.Allocate((long)dt.Length * sizeof(float));
        using var bufA = device.Allocate((long)a.Length * sizeof(float));
        using var bufB = device.Allocate((long)b.Length * sizeof(float));
        using var bufC = device.Allocate((long)c.Length * sizeof(float));
        using var bufY = device.Allocate((long)seqLen * dInner * sizeof(float));

        // 1. One-shot seqLen=8 baseline.
        device.Upload(state0, bufState);
        device.Upload(x, bufX);
        device.Upload(dt, bufDt);
        device.Upload(a, bufA);
        device.Upload(b, bufB);
        device.Upload(c, bufC);
        kernel.Launch(bufState, bufX, bufDt, bufA, bufB, bufC, bufY,
                      nHead, headDim, dState, nGroup, seqLen);
        float[] yOneShot = new float[seqLen * dInner];
        float[] stateOneShot = new float[state0.Length];
        device.Download(bufY, yOneShot);
        device.Download(bufState, stateOneShot);

        // 2. Two seqLen=4 calls on the same state buffer.
        device.Upload(state0, bufState);
        // First half: tokens 0..3 of x, dt, b, c.
        using var bufXHalf = device.Allocate((long)4 * dInner * sizeof(float));
        using var bufDtHalf = device.Allocate((long)4 * nHead * sizeof(float));
        using var bufBHalf = device.Allocate((long)4 * nGroup * dState * sizeof(float));
        using var bufCHalf = device.Allocate((long)4 * nGroup * dState * sizeof(float));
        using var bufYHalf = device.Allocate((long)4 * dInner * sizeof(float));

        device.Upload(x.AsSpan(0, 4 * dInner).ToArray(), bufXHalf);
        device.Upload(dt.AsSpan(0, 4 * nHead).ToArray(), bufDtHalf);
        device.Upload(b.AsSpan(0, 4 * nGroup * dState).ToArray(), bufBHalf);
        device.Upload(c.AsSpan(0, 4 * nGroup * dState).ToArray(), bufCHalf);
        kernel.Launch(bufState, bufXHalf, bufDtHalf, bufA, bufBHalf, bufCHalf, bufYHalf,
                      nHead, headDim, dState, nGroup, 4);
        float[] yFirstHalf = new float[4 * dInner];
        device.Download(bufYHalf, yFirstHalf);

        device.Upload(x.AsSpan(4 * dInner, 4 * dInner).ToArray(), bufXHalf);
        device.Upload(dt.AsSpan(4 * nHead, 4 * nHead).ToArray(), bufDtHalf);
        device.Upload(b.AsSpan(4 * nGroup * dState, 4 * nGroup * dState).ToArray(), bufBHalf);
        device.Upload(c.AsSpan(4 * nGroup * dState, 4 * nGroup * dState).ToArray(), bufCHalf);
        kernel.Launch(bufState, bufXHalf, bufDtHalf, bufA, bufBHalf, bufCHalf, bufYHalf,
                      nHead, headDim, dState, nGroup, 4);
        float[] ySecondHalf = new float[4 * dInner];
        float[] stateSplit = new float[state0.Length];
        device.Download(bufYHalf, ySecondHalf);
        device.Download(bufState, stateSplit);

        // 3. Compare the concatenated split output to the one-shot output.
        // Should be bit-identical: the kernel does the exact same sequence
        // of ops in either case (state is the only stateful buffer).
        for (int i = 0; i < 4 * dInner; i++)
            Assert.Equal(yOneShot[i], yFirstHalf[i]);
        for (int i = 0; i < 4 * dInner; i++)
            Assert.Equal(yOneShot[4 * dInner + i], ySecondHalf[i]);
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

    /// <summary>U(-0.5, -0.05) — A is stored negative by the GGUF converter so exp(dt*A) decays.</summary>
    private static float[] NegativeRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(-(rng.NextDouble() * 0.45 + 0.05));
        return arr;
    }
}
