using System.Runtime.CompilerServices;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q6_K GEMV kernel.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy: generate random FP32 weights, quantise them to Q6_K
/// via the per-fixture <see cref="Q6KFixture.QuantizeRows"/> helper (which
/// mirrors llama.cpp's block_q6_K layout byte-for-byte and is verified
/// against the CPU oracle <c>DequantizeQ6_KScalar</c> in a separate
/// round-trip assertion at the start of each test). Reference result is from
/// a scalar CPU GEMV that reads the same Q6_K bytes the GPU shader sees and
/// dequantises on the fly.
/// </para>
/// <para>
/// Comparing Q6_K-GPU against a Q6_K-byte-identical CPU reference (rather
/// than against the original FP32 weights) catches bugs in the shader's
/// (ql, qh) bit-extraction, the int8 scale sign-extension, the fp16 d read,
/// and block-stride arithmetic — bugs that a quantise-then-compare-to-FP32
/// reference would mask. Q6_K's per-row stride is 210 bytes per super-block,
/// which is NOT 4-byte aligned, so straddle-handling correctness in the
/// per-byte / per-fp16 readers is the most likely failure mode and is
/// directly exercised by the multi-super-block test cases.
/// </para>
/// <para>
/// Tolerance: absolute 5e-3 / relative 1e-3 — same as Q4_K / Q5_K. The
/// dominant drift is reduction-order (scalar CPU vs. workgroup tree-reduce
/// GPU); 6-bit signed quants with 16 int8 scales per super-block give finer
/// resolution than Q4_K / Q5_K, so the per-element drift is actually lower
/// than the K=4/5 siblings, but reduction-order noise dominates either way.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ6KGemvF32KernelTests
{
    private const int Q6KGroupSize = 256;
    private const int Q6KBlockBytes = 210;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(1, 256)]                  // minimum: 1 super-block per row
    [InlineData(8, 256)]                  // 8 rows, 1 super-block — sanity
    [InlineData(4, 512)]                  // 2 super-blocks per row (row stride 420 — 4-aligned)
    [InlineData(16, 768)]                 // 3 super-blocks per row, row stride 630 — NOT 4-aligned
    [InlineData(2048, 768)]               // larger M, exercises workgroup-per-row dispatch with 4-misaligned stride
    [InlineData(576, 1024)]               // 4 super-blocks per row (row stride 840 — 4-aligned)
    [InlineData(1024, 1024)]              // square 4 super-blocks
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xF00D + m * 7 + k * 11);
        float[] weightsF32 = Q6KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = Q6KFixture.RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q6KGroupSize;
        int rowBytes = blocksPerRow * Q6KBlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ6K = Q6KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ6K.Length);

        // Sanity-check the fixture quantiser against the CPU oracle: a single
        // dequant via DequantizeQ6_KScalar on the produced bytes must round-trip
        // to within ~10% relative L2. This is a structural safeguard — if the
        // fixture mis-packs the (ql, qh) groups or the int8 scales, the kernel
        // tests would be testing nothing.
        Q6KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ6K, m, k);

        float[] expected = Q6KFixture.CpuGemvQ6K(weightsQ6K, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ6KGemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ6K), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        Q6KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
