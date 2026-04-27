using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q6_K batched GEMM (prefill path).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <see cref="VulkanMatMulQ6KGemvF32KernelTests"/>: byte-identical
/// Q6_K weights are produced by <see cref="Q6KFixture.QuantizeRows"/> (with a
/// fixture round-trip sanity check), then both the Vulkan kernel and the
/// scalar CPU reference dequantise from the same bytes. This is a stricter
/// test than "quantise + compare to FP32" — it catches block-stride / (ql,
/// qh) bit-extraction / int8 scale sign-extension / fp16 d read bugs that a
/// FP32 reference would mask.
/// </para>
/// <para>
/// Tolerance: absolute 5e-3 / relative 1e-3 — same as Q4_K / Q5_K. The
/// dominant drift is reduction-order (16x16 tile per workgroup with
/// per-K-chunk barrier vs. fully-sequential CPU); Q6_K's 6-bit signed quant
/// resolution is finer than Q4_K / Q5_K so the per-element drift is actually
/// lower, but reduction-order noise dominates either way.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanMatMulQ6KGemmF32KernelTests
{
    private const int Q6KGroupSize = 256;
    private const int Q6KBlockBytes = 210;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ6KGemmF32KernelTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableTheory]
    [InlineData(2, 4, 256)]              // tiny sanity: one super-block per row
    [InlineData(1, 1, 256)]              // single-cell output (bounds check)
    [InlineData(17, 33, 512)]            // non-multiple-of-tile sizes, edge masks
    [InlineData(64, 1024, 1024)]         // square-ish prefill batch
    [InlineData(32, 1536, 768)]          // gate/up shape (k=768=3 super-blocks, row stride 630 — NOT 4-aligned)
    [InlineData(16, 4096, 4096)]         // Llama-3-8B projection (prefill batch — slow but representative)
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xCAB1E + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = Q6KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Q6KFixture.RandomFloats(rng, n * k, range: 1.0f);

        int blocksPerRow = k / Q6KGroupSize;
        int rowBytes = blocksPerRow * Q6KBlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ6K = Q6KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ6K.Length);

        Q6KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ6K, m, k);

        float[] expected = Q6KFixture.CpuGemmQ6K(weightsQ6K, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ6KGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ6K), bufW);
        device.Upload(inputB, bufB);

        var sw = Stopwatch.StartNew();
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        sw.Stop();

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        _output.WriteLine($"Q6_K GEMM dispatch (n={n}, m={m}, k={k}): {sw.Elapsed.TotalMilliseconds:F2} ms");
        Q6KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
