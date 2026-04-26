using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity tests for the Mamba-3 data-dependent RoPE Vulkan kernel.
/// </summary>
/// <remarks>
/// <para>
/// Reference is <see cref="Mamba3DataRoPE.ExecuteCanonical"/> (CPU). Tolerance
/// is abs 1e-3 / rel 1e-3 — tanh, mod 2π, sin and cos accumulate F32 noise
/// across t, and the GLSL <c>tanh</c>/<c>floor</c>/<c>cos</c>/<c>sin</c>
/// implementations may differ in the last bits from the CPU intrinsics. The
/// per-head per-lane recurrence shape is identical though, so drift stays
/// bounded.
/// </para>
/// <para>
/// Inputs are bounded: <c>anglesRaw</c> in <c>[-1, 1]</c> (so <c>tanh*π</c> stays
/// inside <c>[-π, π]</c>) and <c>dt</c> in <c>[0, 0.5]</c> (matches the
/// post-softplus distribution and keeps the cumsum from blowing up too fast).
/// b and c start small (~U(-0.5, 0.5)) so the rotation outputs stay in a
/// numerically tame range.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMamba3DataRopeF32KernelTests
{
    private const float AbsTol = 1e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 1, 2,  16,  4, /*halved*/ false)]   // SISO Pairwise — single token, tiny
    [InlineData(4, 1, 4,  32,  8, /*halved*/ false)]   // SISO Pairwise — multi-token prefill
    [InlineData(1, 2, 2,  16,  4, /*halved*/ true)]    // MIMO Halved   — single token, tiny
    [InlineData(4, 2, 4,  32,  8, /*halved*/ true)]    // MIMO Halved   — multi-token prefill
    public void Launch_MatchesCpuReference(
        int seqLen, int nRank, int nHead, int dState, int numRopeAngles, bool halved)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x3A21 ^ (seqLen * 131) ^ (nRank * 71)
                             ^ (nHead * 53) ^ (dState * 23) ^ numRopeAngles ^ (halved ? 1 : 0));

        int bcLen  = seqLen * nRank * nHead * dState;
        int angLen = seqLen * numRopeAngles;
        int dtLen  = seqLen * nHead;
        int cumLen = nHead * numRopeAngles;
        var mode = halved ? Mamba3RoPEMode.Halved : Mamba3RoPEMode.Pairwise;
        var modeVk = halved ? Mamba3RopeMode.Halved : Mamba3RopeMode.Pairwise;

        float[] b0 = MidRandom(rng, bcLen);
        float[] c0 = MidRandom(rng, bcLen);
        float[] anglesRaw = SmallRandom(rng, angLen);   // [-1, 1] before tanh
        float[] dt = PositiveSmallRandom(rng, dtLen);   // [0, 0.5] post-softplus
        float[] cumPrev = SmallRandom(rng, cumLen);     // arbitrary seed in [-0.1, 0.1]

        // CPU reference — runs on a copy of (b, c) so the same starting buffers
        // can be uploaded to the GPU.
        float[] bCpu = (float[])b0.Clone();
        float[] cCpu = (float[])c0.Clone();
        float[] cumOutCpu = new float[cumLen];
        Mamba3DataRoPE.ExecuteCanonical(
            bCpu, cCpu, anglesRaw, dt, cumPrev, cumOutCpu,
            seqLen, nRank, nHead, dState, numRopeAngles, mode);

        // GPU run.
        using var device = VulkanDevice.Create();
        using var kernel = Mamba3DataRopeF32Kernel.Create(device, spvDir);

        using var bufB = device.Allocate((long)b0.Length * sizeof(float));
        using var bufC = device.Allocate((long)c0.Length * sizeof(float));
        using var bufAng = device.Allocate((long)anglesRaw.Length * sizeof(float));
        using var bufDt = device.Allocate((long)dt.Length * sizeof(float));
        using var bufCumPrev = device.Allocate((long)cumLen * sizeof(float));
        using var bufCumOut = device.Allocate((long)cumLen * sizeof(float));

        device.Upload(b0, bufB);
        device.Upload(c0, bufC);
        device.Upload(anglesRaw, bufAng);
        device.Upload(dt, bufDt);
        device.Upload(cumPrev, bufCumPrev);
        // bufCumOut: leave whatever — kernel will overwrite.

        kernel.Launch(bufB, bufC, bufAng, bufDt, bufCumPrev, bufCumOut,
                      seqLen, nRank, nHead, dState, numRopeAngles, modeVk,
                      hasCumPrev: true, writeCumOut: true);

        float[] bGpu = new float[bcLen];
        float[] cGpu = new float[bcLen];
        float[] cumOutGpu = new float[cumLen];
        device.Download(bufB, bGpu);
        device.Download(bufC, cGpu);
        device.Download(bufCumOut, cumOutGpu);

        // b parity.
        for (int i = 0; i < bcLen; i++)
        {
            float diff = MathF.Abs(bCpu[i] - bGpu[i]);
            float bar = AbsTol + RelTol * MathF.Abs(bCpu[i]);
            Assert.True(diff <= bar,
                $"b[{i}]: cpu={bCpu[i]:F6} vs vulkan={bGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        // c parity.
        for (int i = 0; i < bcLen; i++)
        {
            float diff = MathF.Abs(cCpu[i] - cGpu[i]);
            float bar = AbsTol + RelTol * MathF.Abs(cCpu[i]);
            Assert.True(diff <= bar,
                $"c[{i}]: cpu={cCpu[i]:F6} vs vulkan={cGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        // cumOut parity.
        for (int i = 0; i < cumLen; i++)
        {
            float diff = MathF.Abs(cumOutCpu[i] - cumOutGpu[i]);
            float bar = AbsTol + RelTol * MathF.Abs(cumOutCpu[i]);
            Assert.True(diff <= bar,
                $"cumOut[{i}]: cpu={cumOutCpu[i]:F6} vs vulkan={cumOutGpu[i]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void Launch_SeedFromZero_WhenHasCumPrevFalse()
    {
        // hasCumPrev=false must seed cum to 0 regardless of cumPrev contents.
        // Compare against a CPU run with empty cumAnglePrev.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqLen = 4, nRank = 1, nHead = 2, dState = 16, numRopeAngles = 4;
        int bcLen = seqLen * nRank * nHead * dState;
        int cumLen = nHead * numRopeAngles;

        var rng = new Random(unchecked((int)0xDEADC0DE));
        float[] b0 = MidRandom(rng, bcLen);
        float[] c0 = MidRandom(rng, bcLen);
        float[] anglesRaw = SmallRandom(rng, seqLen * numRopeAngles);
        float[] dt = PositiveSmallRandom(rng, seqLen * nHead);
        // Stuff cumPrev with junk to prove the shader ignores it when the flag is 0.
        float[] cumPrev = new float[cumLen];
        for (int i = 0; i < cumLen; i++) cumPrev[i] = 999f;

        float[] bCpu = (float[])b0.Clone();
        float[] cCpu = (float[])c0.Clone();
        Mamba3DataRoPE.ExecuteCanonical(
            bCpu, cCpu, anglesRaw, dt,
            cumAnglePrev: ReadOnlySpan<float>.Empty,
            cumAngleOut:  Span<float>.Empty,
            seqLen, nRank, nHead, dState, numRopeAngles, Mamba3RoPEMode.Pairwise);

        using var device = VulkanDevice.Create();
        using var kernel = Mamba3DataRopeF32Kernel.Create(device, spvDir);

        using var bufB = device.Allocate((long)bcLen * sizeof(float));
        using var bufC = device.Allocate((long)bcLen * sizeof(float));
        using var bufAng = device.Allocate((long)anglesRaw.Length * sizeof(float));
        using var bufDt = device.Allocate((long)dt.Length * sizeof(float));
        using var bufCumPrev = device.Allocate((long)cumLen * sizeof(float));
        using var bufCumOut = device.Allocate((long)cumLen * sizeof(float));

        device.Upload(b0, bufB);
        device.Upload(c0, bufC);
        device.Upload(anglesRaw, bufAng);
        device.Upload(dt, bufDt);
        device.Upload(cumPrev, bufCumPrev);

        kernel.Launch(bufB, bufC, bufAng, bufDt, bufCumPrev, bufCumOut,
                      seqLen, nRank, nHead, dState, numRopeAngles, Mamba3RopeMode.Pairwise,
                      hasCumPrev: false, writeCumOut: false);

        float[] bGpu = new float[bcLen];
        float[] cGpu = new float[bcLen];
        device.Download(bufB, bGpu);
        device.Download(bufC, cGpu);

        for (int i = 0; i < bcLen; i++)
        {
            float diff = MathF.Abs(bCpu[i] - bGpu[i]);
            float bar = AbsTol + RelTol * MathF.Abs(bCpu[i]);
            Assert.True(diff <= bar, $"b[{i}]: cpu={bCpu[i]:F6} vs vulkan={bGpu[i]:F6}");
            diff = MathF.Abs(cCpu[i] - cGpu[i]);
            bar = AbsTol + RelTol * MathF.Abs(cCpu[i]);
            Assert.True(diff <= bar, $"c[{i}]: cpu={cCpu[i]:F6} vs vulkan={cGpu[i]:F6}");
        }
    }

    [SkippableFact]
    public void Launch_StatePersistsAcrossCalls()
    {
        // Splitting a seqLen=8 call into two seqLen=4 calls (with cumOut[1]
        // threaded into cumPrev[2]) must produce bit-identical b/c outputs to
        // a single seqLen=8 call. Each split run does the same arithmetic in
        // the same order on the same lanes, so the result is exact (no
        // tolerance — Assert.Equal).
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqLen = 8, nRank = 2, nHead = 4, dState = 32, numRopeAngles = 8;
        int bcLen = seqLen * nRank * nHead * dState;
        int halfLen = (seqLen / 2) * nRank * nHead * dState;
        int angLen = seqLen * numRopeAngles;
        int angHalf = (seqLen / 2) * numRopeAngles;
        int dtLen = seqLen * nHead;
        int dtHalf = (seqLen / 2) * nHead;
        int cumLen = nHead * numRopeAngles;
        var mode = Mamba3RopeMode.Halved;

        var rng = new Random(unchecked((int)0xCAFE1234));
        float[] b0 = MidRandom(rng, bcLen);
        float[] c0 = MidRandom(rng, bcLen);
        float[] anglesRaw = SmallRandom(rng, angLen);
        float[] dt = PositiveSmallRandom(rng, dtLen);

        using var device = VulkanDevice.Create();
        using var kernel = Mamba3DataRopeF32Kernel.Create(device, spvDir);

        using var bufB = device.Allocate((long)bcLen * sizeof(float));
        using var bufC = device.Allocate((long)bcLen * sizeof(float));
        using var bufAng = device.Allocate((long)angLen * sizeof(float));
        using var bufDt = device.Allocate((long)dtLen * sizeof(float));
        using var bufCumPrev = device.Allocate((long)cumLen * sizeof(float));
        using var bufCumOut = device.Allocate((long)cumLen * sizeof(float));

        // 1. One-shot seqLen=8 baseline. Seed cum from 0 (hasCumPrev=false).
        device.Upload(b0, bufB);
        device.Upload(c0, bufC);
        device.Upload(anglesRaw, bufAng);
        device.Upload(dt, bufDt);
        kernel.Launch(bufB, bufC, bufAng, bufDt, bufCumPrev, bufCumOut,
                      seqLen, nRank, nHead, dState, numRopeAngles, mode,
                      hasCumPrev: false, writeCumOut: true);
        float[] bOneShot = new float[bcLen];
        float[] cOneShot = new float[bcLen];
        float[] cumOneShot = new float[cumLen];
        device.Download(bufB, bOneShot);
        device.Download(bufC, cOneShot);
        device.Download(bufCumOut, cumOneShot);

        // 2. Two seqLen=4 calls. We use half-sized buffers for the t-sliced
        //    inputs, full-sized for b and c (the kernel only touches the
        //    relevant token slabs anyway). For b/c we re-upload the whole
        //    b0/c0 so the second-half indices are valid, but only the first 4
        //    tokens get rotated in the first call, then the second 4 tokens
        //    in the second call.
        using var bufBHalf = device.Allocate((long)halfLen * sizeof(float));
        using var bufCHalf = device.Allocate((long)halfLen * sizeof(float));
        using var bufAngHalf = device.Allocate((long)angHalf * sizeof(float));
        using var bufDtHalf = device.Allocate((long)dtHalf * sizeof(float));

        // First half: tokens 0..3.
        device.Upload(b0.AsSpan(0, halfLen).ToArray(), bufBHalf);
        device.Upload(c0.AsSpan(0, halfLen).ToArray(), bufCHalf);
        device.Upload(anglesRaw.AsSpan(0, angHalf).ToArray(), bufAngHalf);
        device.Upload(dt.AsSpan(0, dtHalf).ToArray(), bufDtHalf);
        kernel.Launch(bufBHalf, bufCHalf, bufAngHalf, bufDtHalf, bufCumPrev, bufCumOut,
                      seqLen / 2, nRank, nHead, dState, numRopeAngles, mode,
                      hasCumPrev: false, writeCumOut: true);
        float[] bFirstHalf = new float[halfLen];
        float[] cFirstHalf = new float[halfLen];
        float[] cumAfterHalf1 = new float[cumLen];
        device.Download(bufBHalf, bFirstHalf);
        device.Download(bufCHalf, cFirstHalf);
        device.Download(bufCumOut, cumAfterHalf1);

        // Second half: tokens 4..7. Seed cumPrev with cumAfterHalf1 so the
        // recurrence resumes where it left off.
        device.Upload(cumAfterHalf1, bufCumPrev);
        device.Upload(b0.AsSpan(halfLen, halfLen).ToArray(), bufBHalf);
        device.Upload(c0.AsSpan(halfLen, halfLen).ToArray(), bufCHalf);
        device.Upload(anglesRaw.AsSpan(angHalf, angHalf).ToArray(), bufAngHalf);
        device.Upload(dt.AsSpan(dtHalf, dtHalf).ToArray(), bufDtHalf);
        kernel.Launch(bufBHalf, bufCHalf, bufAngHalf, bufDtHalf, bufCumPrev, bufCumOut,
                      seqLen / 2, nRank, nHead, dState, numRopeAngles, mode,
                      hasCumPrev: true, writeCumOut: true);
        float[] bSecondHalf = new float[halfLen];
        float[] cSecondHalf = new float[halfLen];
        float[] cumSplit = new float[cumLen];
        device.Download(bufBHalf, bSecondHalf);
        device.Download(bufCHalf, cSecondHalf);
        device.Download(bufCumOut, cumSplit);

        // 3. Compare. Bit-identical: same arithmetic in the same order.
        for (int i = 0; i < halfLen; i++)
            Assert.Equal(bOneShot[i], bFirstHalf[i]);
        for (int i = 0; i < halfLen; i++)
            Assert.Equal(bOneShot[halfLen + i], bSecondHalf[i]);
        for (int i = 0; i < halfLen; i++)
            Assert.Equal(cOneShot[i], cFirstHalf[i]);
        for (int i = 0; i < halfLen; i++)
            Assert.Equal(cOneShot[halfLen + i], cSecondHalf[i]);
        for (int i = 0; i < cumLen; i++)
            Assert.Equal(cumOneShot[i], cumSplit[i]);
    }

    /// <summary>U(-1, 1) — anglesRaw stays inside tanh's near-linear region.</summary>
    private static float[] SmallRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }

    /// <summary>U(0, 0.5) — post-softplus dt magnitudes; keeps cumsum from blowing up over the test sequence lengths.</summary>
    private static float[] PositiveSmallRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 0.5);
        return arr;
    }

    /// <summary>U(-0.5, 0.5) — modest magnitudes for b and c.</summary>
    private static float[] MidRandom(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() - 0.5);
        return arr;
    }
}
