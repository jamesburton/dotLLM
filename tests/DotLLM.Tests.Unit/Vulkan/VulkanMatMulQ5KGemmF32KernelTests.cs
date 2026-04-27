using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q5_K batched GEMM (prefill path).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <see cref="VulkanMatMulQ5KGemvF32KernelTests"/>: byte-identical
/// Q5_K weights are produced by <see cref="Q5KFixture.QuantizeRows"/> (with a
/// fixture round-trip sanity check), then both the Vulkan kernel and the
/// scalar CPU reference dequantise from the same bytes. This is a stricter
/// test than "quantise + compare to FP32" — it catches block-stride / 6-bit
/// scale-unpack / fp16 d-dmin / nibble-half / qh-bit-index bugs that a FP32
/// reference would mask.
/// </para>
/// <para>
/// Tolerance: absolute 5e-3 / relative 1e-3 — same as Q4_K. The dominant
/// drift is reduction-order (16x16 tile per workgroup with per-K-chunk
/// barrier vs. fully-sequential CPU) plus 6-bit scale rounding noise; both
/// independent of per-element bit width.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanMatMulQ5KGemmF32KernelTests
{
    private const int Q5KGroupSize = 256;
    private const int Q5KBlockBytes = 176;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ5KGemmF32KernelTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableTheory]
    [InlineData(2, 4, 256)]              // tiny sanity: one super-block per row
    [InlineData(1, 1, 256)]              // single-cell output (bounds check)
    [InlineData(17, 33, 512)]            // non-multiple-of-tile sizes, edge masks
    [InlineData(64, 1024, 1024)]         // square-ish prefill batch
    [InlineData(32, 1536, 768)]          // SmolLM-ish gate/up shape (k=768=3 super-blocks)
    [InlineData(16, 4096, 4096)]         // Llama-3-8B projection (prefill batch — slow but representative)
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEAD + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = Q5KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Q5KFixture.RandomFloats(rng, n * k, range: 1.0f);

        int blocksPerRow = k / Q5KGroupSize;
        int rowBytes = blocksPerRow * Q5KBlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ5K = Q5KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ5K.Length);

        Q5KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ5K, m, k);

        float[] expected = Q5KFixture.CpuGemmQ5K(weightsQ5K, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ5KGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ5K), bufW);
        device.Upload(inputB, bufB);

        var sw = Stopwatch.StartNew();
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        sw.Stop();

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        _output.WriteLine($"Q5_K GEMM dispatch (n={n}, m={m}, k={k}): {sw.Elapsed.TotalMilliseconds:F2} ms");
        Q5KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
