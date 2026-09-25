using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the MoE broadcast Vulkan kernel — replaces a
/// seqLen × topK loop of <c>vkCmdCopyBuffer</c> regions in the MoE forward
/// path with a single fused compute dispatch.
/// </summary>
/// <remarks>
/// The kernel is a strided F32 load/store with no FP arithmetic, so the
/// output is bit-exact wrt the CPU reference loop it replaces — the
/// tolerance below is therefore zero (exact equality).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMoeBroadcastF32KernelTests
{
    [SkippableTheory]
    [InlineData(1, 2, 16)]      // smallest: 1 token, k=2 (Mixtral 8/2)
    [InlineData(3, 2, 16)]      // small prefill, k=2
    [InlineData(8, 4, 32)]      // 8-token prefill, k=4
    [InlineData(1, 8, 128)]     // k=8 (DeepSeek-V2 / Qwen3-MoE)
    public void Launch_MatchesCpuReference(int seqLen, int topK, int hidden)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(unchecked((int)0xB10ADCA5) + seqLen * 13 + topK * 7 + hidden);
        float[] input = RandomFloats(rng, seqLen * hidden);

        float[] expected = CpuBroadcast(input, seqLen, topK, hidden);

        using var device = VulkanDevice.Create();
        using var kernel = MoeBroadcastF32Kernel.Create(device, spvDir);

        using var bufIn = device.Allocate((long)input.Length * sizeof(float));
        using var bufOut = device.Allocate((long)expected.Length * sizeof(float));

        device.Upload(input, bufIn);

        kernel.Launch(bufIn, bufOut, seqLen, topK, hidden);

        float[] actual = new float[expected.Length];
        device.Download(bufOut, actual);

        // Bit-exact: pure copy, no FP arithmetic anywhere.
        for (int i = 0; i < expected.Length; i++)
        {
            int n = i / hidden;
            int h = i % hidden;
            Assert.True(expected[i] == actual[i],
                $"n={n} (t={n / topK}, slot={n % topK}), h={h}: cpu={expected[i]:F6} vs vulkan={actual[i]:F6}");
        }
    }

    [SkippableFact]
    public void Launch_TopKOne_PassesRowThrough()
    {
        // With topK=1, output should be byte-identical to the input — sanity
        // check the shape contract (no accidental cross-row mixing).
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqLen = 5, topK = 1, hidden = 8;
        var rng = new Random(0xFEED);
        float[] input = RandomFloats(rng, seqLen * hidden);

        using var device = VulkanDevice.Create();
        using var kernel = MoeBroadcastF32Kernel.Create(device, spvDir);
        using var bufIn = device.Allocate((long)input.Length * sizeof(float));
        using var bufOut = device.Allocate((long)input.Length * sizeof(float));

        device.Upload(input, bufIn);
        kernel.Launch(bufIn, bufOut, seqLen, topK, hidden);

        float[] actual = new float[input.Length];
        device.Download(bufOut, actual);

        for (int i = 0; i < input.Length; i++)
            Assert.True(input[i] == actual[i],
                $"i={i}: input={input[i]:F6} vs output={actual[i]:F6}");
    }

    [SkippableFact]
    public void Launch_ReplicatesEachRowExactlyTopKTimes()
    {
        // Stronger structural check: every (t, slot) row must equal the t-th
        // input row. Catches bugs where the kernel writes the wrong source
        // row (e.g. a rotation by topK rather than a divide-by-topK).
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        const int seqLen = 4, topK = 3, hidden = 16;
        var rng = new Random(unchecked((int)0xCAFEBABE));
        float[] input = RandomFloats(rng, seqLen * hidden);

        using var device = VulkanDevice.Create();
        using var kernel = MoeBroadcastF32Kernel.Create(device, spvDir);
        using var bufIn = device.Allocate((long)input.Length * sizeof(float));
        using var bufOut = device.Allocate((long)seqLen * topK * hidden * sizeof(float));

        device.Upload(input, bufIn);
        kernel.Launch(bufIn, bufOut, seqLen, topK, hidden);

        float[] actual = new float[seqLen * topK * hidden];
        device.Download(bufOut, actual);

        for (int t = 0; t < seqLen; t++)
        {
            for (int slot = 0; slot < topK; slot++)
            {
                int outRow = t * topK + slot;
                for (int h = 0; h < hidden; h++)
                {
                    float expected = input[t * hidden + h];
                    float got = actual[outRow * hidden + h];
                    Assert.True(expected == got,
                        $"t={t} slot={slot} h={h}: expected={expected:F6} vs got={got:F6}");
                }
            }
        }
    }

    private static float[] CpuBroadcast(float[] input, int seqLen, int topK, int hidden)
    {
        var output = new float[seqLen * topK * hidden];
        for (int t = 0; t < seqLen; t++)
        {
            for (int slot = 0; slot < topK; slot++)
            {
                int outRow = t * topK + slot;
                for (int h = 0; h < hidden; h++)
                    output[outRow * hidden + h] = input[t * hidden + h];
            }
        }
        return output;
    }

    private static float[] RandomFloats(Random rng, int count)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        return arr;
    }
}
