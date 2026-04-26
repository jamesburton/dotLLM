using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Mamba-2 / Mamba-3 per-head
/// scalar skip-add kernel (post-scan step #9 of the SSM body):
/// <c>y[t, h*headDim+i] += x[t, h*headDim+i] * D[h]</c>.
/// </summary>
/// <remarks>
/// Reference is a scalar CPU loop matching
/// <see cref="DotLLM.Models.Architectures.NemotronHTransformerModel"/>'s
/// <c>ForwardSsmBody</c>. Tolerance is abs 1e-5 / rel 1e-4 — a single
/// FMA per output element, no reduction, so the kernel should be
/// effectively bit-identical to the reference (the loose tolerance
/// covers IEEE-754 rounding differences between scalar and GPU FP32).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanSsmDSkipF32KernelTests
{
    private const float AbsTol = 1e-5f;
    private const float RelTol = 1e-4f;

    [SkippableTheory]
    [InlineData(1, 2, 4)]      // smallest
    [InlineData(4, 10, 80)]    // NemotronH-realistic
    [InlineData(8, 4, 16)]     // mid-size
    [InlineData(1, 10, 80)]    // decode shape
    [InlineData(3, 3, 17)]     // odd headDim — exercises stride past WG boundary
    public void Launch_MatchesCpuReference(int seqLen, int nHead, int headDim)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xD5C19 + seqLen * 31 + nHead * 17 + headDim * 11);
        int dInner = nHead * headDim;
        float[] y = RandomFloats(rng, seqLen * dInner);
        float[] x = RandomFloats(rng, seqLen * dInner);
        float[] d = RandomFloats(rng, nHead);

        // Reference: y[t, h*headDim+i] += x[t, h*headDim+i] * D[h].
        float[] expected = (float[])y.Clone();
        for (int t = 0; t < seqLen; t++)
        {
            for (int h = 0; h < nHead; h++)
            {
                float dh = d[h];
                for (int i = 0; i < headDim; i++)
                {
                    int idx = t * dInner + h * headDim + i;
                    expected[idx] += x[idx] * dh;
                }
            }
        }

        using var device = VulkanDevice.Create();
        using var kernel = SsmDSkipF32Kernel.Create(device, spvDir);

        using var bufY = device.Allocate((long)y.Length * sizeof(float));
        using var bufX = device.Allocate((long)x.Length * sizeof(float));
        using var bufD = device.Allocate((long)d.Length * sizeof(float));

        device.Upload(y, bufY);
        device.Upload(x, bufX);
        device.Upload(d, bufD);

        kernel.Launch(bufY, bufX, bufD, seqLen, nHead, headDim);

        float[] actual = new float[y.Length];
        device.Download(bufY, actual);

        for (int idx = 0; idx < expected.Length; idx++)
        {
            int t = idx / dInner;
            int rem = idx - t * dInner;
            int h = rem / headDim;
            int i = rem - h * headDim;
            float diff = MathF.Abs(expected[idx] - actual[idx]);
            float bar = AbsTol + RelTol * MathF.Abs(expected[idx]);
            Assert.True(diff <= bar,
                $"t={t}, h={h}, i={i}: cpu={expected[idx]:F6} vs vulkan={actual[idx]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
