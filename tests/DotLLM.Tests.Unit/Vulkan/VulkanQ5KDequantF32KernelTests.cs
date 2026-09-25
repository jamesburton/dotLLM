using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-parity test for the Vulkan Q5_K → F32 dequant kernel against the CPU
/// oracle <c>DequantizeKQuants.DequantizeQ5_K</c>. Fills the asymmetric gap
/// reported in issue #309: Q2_K, Q3_K, Q4_K and Q6_K each had a standalone
/// dequant parity test and Q5_K — shipped by default — did not.
/// </summary>
/// <remarks>
/// Tolerance is 0 ULP, exactly as for the Q4_K/Q6_K siblings: the shader mirrors
/// the oracle's op order and is <c>precise</c>-qualified (no FMA contraction),
/// so any difference at all is a real divergence. Q5_K shares Q4_K's 12-byte
/// 6-bit (scale, min) packing and adds a separate 32-byte <c>qh</c> plane whose
/// bit <c>j</c> of byte <c>i</c> is the 5th bit of element <c>j*32+i</c> — a
/// transposed read of that plane is the format-specific bug this pins down. The
/// 40000-block case spans the kernel's 32768-workgroup dispatch-chunk boundary
/// (discriminates a broken <c>firstBlock</c> offset).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ5KDequantF32KernelTests
{
    [SkippableTheory]
    [InlineData(1)]
    [InlineData(8)]
    [InlineData(1024)]
    [InlineData(40000)] // > MaxBlocksPerDispatch — exercises chunked dispatch
    public unsafe void Launch_MatchesCpuOracle(int totalBlocks)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "q5_k_dequant_f32.spv")),
            "q5_k_dequant_f32.spv not compiled (glslc / Vulkan SDK required).");

        int elements = totalBlocks * Q5KFixture.Q5KGroupSize;
        var rng = new Random(0x5C_AF_E0 ^ (totalBlocks * 7));
        float[] srcF32 = Q5KFixture.RandomFloats(rng, elements, range: 0.1f);
        byte[] q5kBytes = Q5KFixture.QuantizeRows(srcF32, m: totalBlocks, k: Q5KFixture.Q5KGroupSize);

        float[] expected = new float[elements];
        fixed (byte* p = q5kBytes)
        {
            Dequantize.ToFloat32((nint)p, elements, QuantizationType.Q5_K, expected);
        }

        using var device = VulkanDevice.Create();
        using var kernel = Q5KDequantF32Kernel.Create(device, spvDir);

        long srcBytes = ((long)q5kBytes.Length + 3) & ~3L;
        using var bufSrc = device.Allocate(srcBytes);
        using var bufDst = device.Allocate((long)elements * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(q5kBytes), bufSrc);
        kernel.Launch(bufSrc, bufDst, totalBlocks);

        float[] actual = new float[elements];
        device.Download(bufDst, actual);

        for (int i = 0; i < elements; i++)
        {
            if (BitConverter.SingleToInt32Bits(expected[i]) != BitConverter.SingleToInt32Bits(actual[i]))
                Assert.Fail($"Q5_K dequant mismatch at element {i} (block {i / 256}, t {i % 256}): " +
                            $"cpu={expected[i]:G9} gpu={actual[i]:G9}");
        }
    }
}
