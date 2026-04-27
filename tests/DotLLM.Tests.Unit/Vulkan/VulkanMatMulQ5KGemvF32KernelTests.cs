using System.Runtime.CompilerServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q5_K GEMV kernel.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy: generate random FP32 weights, quantise them to Q5_K
/// via the per-fixture <see cref="Q5KFixture.QuantizeRows"/> helper (which
/// mirrors llama.cpp's block_q5_K layout byte-for-byte and is verified
/// against the CPU oracle <c>DequantizeQ5_KScalar</c> in a separate
/// round-trip assertion at the start of each test). Reference result is from
/// a scalar CPU GEMV that reads the same Q5_K bytes the GPU shader sees and
/// dequantises on the fly.
/// </para>
/// <para>
/// Comparing Q5_K-GPU against a Q5_K-byte-identical CPU reference (rather
/// than against the original FP32 weights) catches bugs in the shader's
/// 6-bit scale unpack, qh[] high-bit indexing, low-nibble selection, and
/// fp16 d/dmin reads — bugs that a quantise-then-compare-to-FP32 reference
/// would mask.
/// </para>
/// <para>
/// Tolerance: absolute 5e-3 / relative 1e-3 — same as Q4_K despite Q5_K's
/// finer 5-bit resolution; the dominant drift is reduction-order (scalar
/// CPU vs. workgroup tree-reduce GPU) plus 6-bit scale rounding noise, both
/// of which are independent of the per-element bit width.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ5KGemvF32KernelTests
{
    private const int Q5KGroupSize = 256;
    private const int Q5KBlockBytes = 176;
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
        float[] weightsF32 = Q5KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q5KFixture.RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q5KGroupSize;
        int rowBytes = blocksPerRow * Q5KBlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ5K = Q5KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ5K.Length);

        // Sanity-check the fixture quantiser against the CPU oracle: a single
        // dequant via DequantizeQ5_KScalar on the produced bytes must round-trip
        // to within ~10% relative L2. This is a structural safeguard — if the
        // fixture mis-packs the qh[] high bits or the 6-bit scales, the kernel
        // tests would be testing nothing.
        Q5KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ5K, m, k);

        float[] expected = Q5KFixture.CpuGemvQ5K(weightsQ5K, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ5KGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ5K), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Q5KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
