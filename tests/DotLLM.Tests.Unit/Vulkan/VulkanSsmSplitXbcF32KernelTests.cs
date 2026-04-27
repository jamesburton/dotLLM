using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the SSM split-xBC Vulkan kernel — replaces a
/// per-token loop of THREE <c>vkCmdCopyBuffer</c> regions per token (one
/// each for x, B, C) in the NemotronH SSM forward path with a single fused
/// compute dispatch.
/// </summary>
/// <remarks>
/// The kernel is a strided F32 load/store with no FP arithmetic, so the
/// output is bit-exact wrt the per-token-copy loop it replaces — the
/// tolerance below is therefore zero (exact equality).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSsmSplitXbcF32KernelTests
{
    [SkippableTheory]
    // (seqLen, dInner, bcDim) — covers the canonical NemotronH-style
    // split shapes plus a few pathological ones (dInner < bcDim, equal).
    [InlineData(1, 16, 8)]      // smallest: single token, dInner > bcDim
    [InlineData(3, 32, 16)]     // small prefill, dInner > bcDim
    [InlineData(8, 64, 64)]     // dInner == bcDim
    [InlineData(4, 8, 32)]      // dInner < bcDim (kernel must not read past x)
    [InlineData(16, 128, 32)]   // larger prefill
    public void Launch_MatchesCpuReference(int seqLen, int dInner, int bcDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int convDim = dInner + 2 * bcDim;
        var rng = new Random(unchecked((int)0x55117BC0) + seqLen * 13 + dInner * 7 + bcDim);
        float[] xbc = RandomFloats(rng, seqLen * convDim);

        (float[] expectedX, float[] expectedB, float[] expectedC) =
            CpuSplit(xbc, seqLen, dInner, bcDim);

        using var device = VulkanDevice.Create();
        using var kernel = SsmSplitXbcF32Kernel.Create(device, spvDir);

        using var bufXbc = device.Allocate((long)xbc.Length * sizeof(float));
        using var bufX = device.Allocate((long)expectedX.Length * sizeof(float));
        using var bufB = device.Allocate((long)expectedB.Length * sizeof(float));
        using var bufC = device.Allocate((long)expectedC.Length * sizeof(float));

        device.Upload(xbc, bufXbc);

        kernel.Launch(bufXbc, bufX, bufB, bufC, seqLen, dInner, bcDim);

        float[] actualX = new float[expectedX.Length];
        float[] actualB = new float[expectedB.Length];
        float[] actualC = new float[expectedC.Length];
        device.Download(bufX, actualX);
        device.Download(bufB, actualB);
        device.Download(bufC, actualC);

        // Bit-exact: pure copy, no FP arithmetic anywhere.
        for (int i = 0; i < expectedX.Length; i++)
            Assert.True(expectedX[i] == actualX[i],
                $"X mismatch at i={i} (t={i / dInner}, col={i % dInner}): cpu={expectedX[i]:F6} vs vulkan={actualX[i]:F6}");
        for (int i = 0; i < expectedB.Length; i++)
            Assert.True(expectedB[i] == actualB[i],
                $"B mismatch at i={i} (t={i / bcDim}, col={i % bcDim}): cpu={expectedB[i]:F6} vs vulkan={actualB[i]:F6}");
        for (int i = 0; i < expectedC.Length; i++)
            Assert.True(expectedC[i] == actualC[i],
                $"C mismatch at i={i} (t={i / bcDim}, col={i % bcDim}): cpu={expectedC[i]:F6} vs vulkan={actualC[i]:F6}");
    }

    [SkippableFact]
    public void Launch_RowOffsetContractIsCorrect()
    {
        // Stronger structural check: each destination's row at index t must
        // equal the t-th source row sliced at the correct offset. Catches
        // bugs where x/B/C get swapped or where the kernel reads from the
        // wrong slice offset (e.g. C vs B mixed up).
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqLen = 5, dInner = 24, bcDim = 12;
        int convDim = dInner + 2 * bcDim;
        var rng = new Random(unchecked((int)0xDEADBEEF));
        float[] xbc = RandomFloats(rng, seqLen * convDim);

        using var device = VulkanDevice.Create();
        using var kernel = SsmSplitXbcF32Kernel.Create(device, spvDir);
        using var bufXbc = device.Allocate((long)xbc.Length * sizeof(float));
        using var bufX = device.Allocate((long)seqLen * dInner * sizeof(float));
        using var bufB = device.Allocate((long)seqLen * bcDim * sizeof(float));
        using var bufC = device.Allocate((long)seqLen * bcDim * sizeof(float));

        device.Upload(xbc, bufXbc);
        kernel.Launch(bufXbc, bufX, bufB, bufC, seqLen, dInner, bcDim);

        float[] actualX = new float[seqLen * dInner];
        float[] actualB = new float[seqLen * bcDim];
        float[] actualC = new float[seqLen * bcDim];
        device.Download(bufX, actualX);
        device.Download(bufB, actualB);
        device.Download(bufC, actualC);

        for (int t = 0; t < seqLen; t++)
        {
            int srcBase = t * convDim;
            for (int col = 0; col < dInner; col++)
            {
                float expected = xbc[srcBase + col];
                float got = actualX[t * dInner + col];
                Assert.True(expected == got, $"X t={t} col={col}: expected={expected:F6} vs got={got:F6}");
            }
            for (int col = 0; col < bcDim; col++)
            {
                float expectedB = xbc[srcBase + dInner + col];
                float gotB = actualB[t * bcDim + col];
                Assert.True(expectedB == gotB, $"B t={t} col={col}: expected={expectedB:F6} vs got={gotB:F6}");

                float expectedC = xbc[srcBase + dInner + bcDim + col];
                float gotC = actualC[t * bcDim + col];
                Assert.True(expectedC == gotC, $"C t={t} col={col}: expected={expectedC:F6} vs got={gotC:F6}");
            }
        }
    }

    private static (float[] x, float[] b, float[] c) CpuSplit(
        float[] xbc, int seqLen, int dInner, int bcDim)
    {
        int convDim = dInner + 2 * bcDim;
        var x = new float[seqLen * dInner];
        var b = new float[seqLen * bcDim];
        var c = new float[seqLen * bcDim];
        for (int t = 0; t < seqLen; t++)
        {
            int srcBase = t * convDim;
            for (int col = 0; col < dInner; col++)
                x[t * dInner + col] = xbc[srcBase + col];
            for (int col = 0; col < bcDim; col++)
            {
                b[t * bcDim + col] = xbc[srcBase + dInner + col];
                c[t * bcDim + col] = xbc[srcBase + dInner + bcDim + col];
            }
        }
        return (x, b, c);
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
