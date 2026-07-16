using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan Q4_K → F32 dequant kernel against the CPU
/// oracle <c>DequantizeKQuants.DequantizeQ4_K</c>. Tolerance is 0 ULP — the
/// shader mirrors the oracle's op order exactly and is <c>precise</c>-qualified
/// (no FMA contraction), which is what lets the issue-#147 GPU-side token-embed
/// dequant keep first-forward logits bit-identical to the CPU load path.
/// The 40000-block case spans the kernel's 32768-workgroup dispatch-chunk
/// boundary (discriminates a broken <c>firstBlock</c> offset).
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ4KDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(1024)]
    [InlineData(40000)] // > MaxBlocksPerDispatch — exercises chunked dispatch
    public unsafe void Launch_MatchesCpuOracle(int totalBlocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        int elements = totalBlocks * Q4KFixture.Q4KGroupSize;
        var rng = new Random(0x4C_AF_E0 ^ (totalBlocks * 7));
        float[] srcF32 = Q4KFixture.RandomFloats(rng, elements, range: 0.1f);
        byte[] q4kBytes = Q4KFixture.QuantizeRows(srcF32, m: totalBlocks, k: Q4KFixture.Q4KGroupSize);

        float[] expected = new float[elements];
        fixed (byte* p = q4kBytes)
        {
            Dequantize.ToFloat32((nint)p, elements, QuantizationType.Q4_K, expected);
        }

        using var device = VulkanDevice.Create();
        using var kernel = Q4KDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)q4kBytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(q4kBytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalBlocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        for (int i = 0; i < elements; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"Q4_K dequant mismatch at element {i} (block {i / 256}, t {i % 256}): " +
                            $"cpu={expected[i]:G9} gpu={actual[i]:G9}");
        }
    }
}
