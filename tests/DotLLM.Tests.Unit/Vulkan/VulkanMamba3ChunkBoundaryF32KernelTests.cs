using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Mamba-3 streaming-chunk boundary
/// adjustment kernel. The CPU reference is the inline rank-summed boundary
/// math used by both <c>Mamba3Block.ApplyChunkBoundaryAdjustment</c> (SISO,
/// <c>nRank == 1</c>) and <c>Mamba3CanonicalSsd.ExecuteMimoStreaming</c>
/// (MIMO, <c>nRank ≥ 2</c>):
/// <c>state[h, p, n] += vState[h, p] · (Σ_r kState[r, h, n]) · coef[h]</c>.
/// </summary>
/// <remarks>
/// <para>
/// The kernel runs the rank summation as a sequential float accumulator
/// matching the CPU oracle's <c>kSum += kState[...]</c> loop, so the only
/// numerical drift sources are reduction-order noise from the per-element
/// multiply-add. Tolerance abs 1e-5 / rel 1e-4 — tighter than the QkNorm
/// kernel since this kernel has no GPU tree reduction.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMamba3ChunkBoundaryF32KernelTests
{
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    [SkippableTheory]
    // SISO shapes (nRank == 1).
    [InlineData(1, 1, 1, 1)]                          // smallest — single (h, p, n).
    [InlineData(4, 4, 8, 1)]                          // SISO mini.
    [InlineData(2, 16, 16, 1)]                        // SISO multi-(p, n) inside one wg-tile.
    [InlineData(4, 32, 64, 1)]                        // SISO realistic Mamba-3 dims.
    [InlineData(3, 17, 19, 1)]                        // SISO non-WG-aligned (p, n) — exercises bounds.
    // MIMO shapes (nRank ≥ 2).
    [InlineData(4, 4, 8, 2)]                          // MIMO mini, R=2.
    [InlineData(2, 8, 16, 4)]                         // MIMO R=4.
    [InlineData(4, 32, 64, 2)]                        // MIMO realistic with rank.
    public void Launch_MatchesCpuReference(int nHead, int headDim, int dState, int nRank)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0x80B0 + nHead * 31 + headDim * 17 + dState * 7 + nRank);
        int stateLen = nHead * headDim * dState;
        int vLen = nHead * headDim;
        int kLen = nRank * nHead * dState;

        // Random initial state, V, K, coef. CoEf is a per-head scalar — pick
        // values away from zero so the early-out `c == 0.0` branch in the kernel
        // is exercised for one head and the non-zero path is exercised for the
        // others.
        float[] state0 = RandomFloats(rng, stateLen, 0.5f);
        float[] vState = RandomFloats(rng, vLen, 0.5f);
        float[] kState = RandomFloats(rng, kLen, 0.5f);
        float[] coef = RandomFloats(rng, nHead, 0.5f);
        if (nHead >= 2) coef[1] = 0.0f;               // exercise the early-out branch.

        // CPU oracle — same equation as Mamba3Block.ApplyChunkBoundaryAdjustment
        // and Mamba3CanonicalSsd.ExecuteMimoStreaming, generalised to any nRank.
        float[] expected = (float[])state0.Clone();
        ApplyBoundaryReference(expected, vState, kState, coef, nHead, headDim, dState, nRank);

        using var device = VulkanDevice.Create();
        using var kernel = Mamba3ChunkBoundaryF32Kernel.Create(device, spvDir);

        using var bufState = device.Allocate((long)stateLen * sizeof(float));
        using var bufV = device.Allocate((long)vLen * sizeof(float));
        using var bufK = device.Allocate((long)kLen * sizeof(float));
        using var bufCoef = device.Allocate((long)nHead * sizeof(float));

        device.Upload(state0, bufState);
        device.Upload(vState, bufV);
        device.Upload(kState, bufK);
        device.Upload(coef, bufCoef);

        kernel.Launch(bufState, bufV, bufK, bufCoef, nHead, headDim, dState, nRank);

        float[] actual = new float[stateLen];
        device.Download(bufState, actual);

        AssertClose(expected, actual, nHead, headDim, dState, nRank);
    }

    [SkippableFact]
    public void Launch_ZeroCoef_LeavesStateUnchanged()
    {
        // Coef == 0 for every head ⇔ first chunk of a sequence (no boundary
        // adjustment). The kernel must early-out and leave state untouched
        // (matching CPU oracle's `if (coef == 0f) continue;` branch).
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int nHead = 4, headDim = 8, dState = 16, nRank = 2;
        var rng = new Random(0xC0EF);
        int stateLen = nHead * headDim * dState;

        float[] state0 = RandomFloats(rng, stateLen, 1.0f);
        float[] vState = RandomFloats(rng, nHead * headDim, 1.0f);
        float[] kState = RandomFloats(rng, nRank * nHead * dState, 1.0f);
        float[] coef = new float[nHead];               // all zero.

        using var device = VulkanDevice.Create();
        using var kernel = Mamba3ChunkBoundaryF32Kernel.Create(device, spvDir);

        using var bufState = device.Allocate((long)stateLen * sizeof(float));
        using var bufV = device.Allocate((long)nHead * headDim * sizeof(float));
        using var bufK = device.Allocate((long)nRank * nHead * dState * sizeof(float));
        using var bufCoef = device.Allocate((long)nHead * sizeof(float));

        device.Upload(state0, bufState);
        device.Upload(vState, bufV);
        device.Upload(kState, bufK);
        device.Upload(coef, bufCoef);

        kernel.Launch(bufState, bufV, bufK, bufCoef, nHead, headDim, dState, nRank);

        float[] actual = new float[stateLen];
        device.Download(bufState, actual);

        for (int i = 0; i < stateLen; i++)
            Assert.Equal(state0[i], actual[i]);
    }

    private static void ApplyBoundaryReference(
        float[] state, float[] vState, float[] kState, float[] coef,
        int nHead, int headDim, int dState, int nRank)
    {
        // kState layout: [R, H, N] row-major.
        int kRankStride = nHead * dState;
        for (int h = 0; h < nHead; h++)
        {
            float c = coef[h];
            if (c == 0.0f) continue;
            int vBase = h * headDim;
            int stateBase = h * headDim * dState;
            for (int p = 0; p < headDim; p++)
            {
                float vp = vState[vBase + p];
                int row = stateBase + p * dState;
                for (int n = 0; n < dState; n++)
                {
                    float kSum = 0f;
                    for (int r = 0; r < nRank; r++)
                    {
                        kSum += kState[r * kRankStride + h * dState + n];
                    }
                    state[row + n] += vp * kSum * c;
                }
            }
        }
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static void AssertClose(float[] expected, float[] actual,
                                     int nHead, int headDim, int dState, int nRank)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        int firstBadIdx = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol)
            {
                if (firstBadIdx < 0) firstBadIdx = i;
                errors++;
            }
        }
        if (errors != 0)
        {
            int h = firstBadIdx / (headDim * dState);
            int rem = firstBadIdx % (headDim * dState);
            int p = rem / dState;
            int n = rem % dState;
            Assert.Fail(
                $"Numerical drift exceeded tolerance " +
                $"(nHead={nHead}, headDim={headDim}, dState={dState}, nRank={nRank}): " +
                $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, " +
                $"first @ h={h} p={p} n={n}: cpu={expected[firstBadIdx]:G9} vs vulkan={actual[firstBadIdx]:G9}");
        }
    }
}
