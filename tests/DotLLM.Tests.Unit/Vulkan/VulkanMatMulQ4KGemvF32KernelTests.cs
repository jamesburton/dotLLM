using System.Runtime.CompilerServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q4_K GEMV kernel.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy: generate random FP32 weights, quantise them to Q4_K
/// via the per-fixture <see cref="Q4KFixture.QuantizeRows"/> helper (which
/// mirrors llama.cpp's block_q4_K layout byte-for-byte and is verified
/// against the CPU oracle <c>DequantizeQ4_KScalar</c> in a separate
/// round-trip assertion at the start of each test). Reference result is from
/// a scalar CPU GEMV that reads the same Q4_K bytes the GPU shader sees and
/// dequantises on the fly.
/// </para>
/// <para>
/// Comparing Q4_K-GPU against a Q4_K-byte-identical CPU reference (rather
/// than against the original FP32 weights) catches bugs in the shader's
/// 6-bit scale unpack, nibble selection, and fp16 d/dmin reads — bugs that a
/// quantise-then-compare-to-FP32 reference would mask.
/// </para>
/// <para>
/// Tolerance: absolute 5e-3 / relative 1e-3 — Q4_K's ~4.5 bits/element makes
/// drift inherently larger than Q8_0 at the same K. The CPU oracle and the
/// GPU shader use the same dequant formula but different reduction orders
/// (scalar block-sequential CPU vs. workgroup tree-reduce GPU) which adds a
/// small additional drift on top of the Q4_K rounding noise.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ4KGemvF32KernelTests
{
    private const int Q4KGroupSize = 256;
    private const int Q4KBlockBytes = 144;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 256)]                  // minimum: 1 super-block per row
    [InlineData(8, 256)]                  // 8 rows, 1 super-block — sanity
    [InlineData(4, 512)]                  // 2 super-blocks per row
    [InlineData(16, 768)]                 // 3 super-blocks per row, non-power-of-2
    [InlineData(2048, 768)]               // larger M, exercises workgroup-per-row dispatch
    [InlineData(576, 1024)]               // 4 super-blocks per row
    [InlineData(1024, 1024)]              // square 4 super-blocks
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xCAFE + m * 7 + k * 11);
        float[] weightsF32 = Q4KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q4KFixture.RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q4KGroupSize;
        int rowBytes = blocksPerRow * Q4KBlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ4K = Q4KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ4K.Length);

        // Sanity-check the fixture quantiser against the CPU oracle: a single
        // dequant via DequantizeQ4_KScalar on the produced bytes must match the
        // reduction inside our scalar GEMV reference. This is a structural
        // safeguard — if the fixture mis-packs the 6-bit scales, the kernel
        // tests would be testing nothing.
        Q4KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ4K, m, k);

        float[] expected = Q4KFixture.CpuGemvQ4K(weightsQ4K, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ4KGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ4K), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Q4KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
