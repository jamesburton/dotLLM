using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q4_K batched GEMM (prefill path).
/// </summary>
/// <remarks>
/// <para>
/// Mirrors <see cref="VulkanMatMulQ4KGemvF32KernelTests"/>: byte-identical
/// Q4_K weights are produced by <see cref="Q4KFixture.QuantizeRows"/> (with a
/// fixture round-trip sanity check), then both the Vulkan kernel and the
/// scalar CPU reference dequantise from the same bytes. This is a stricter
/// test than "quantise + compare to FP32" — it catches block-stride / 6-bit
/// scale-unpack / fp16 d-dmin / nibble-half bugs that a FP32 reference would
/// mask.
/// </para>
/// <para>
/// Tolerance: absolute 5e-3 / relative 1e-3 — Q4_K's 4.5 bits/element makes
/// drift inherently larger than Q8_0 at the same K. The shader's
/// reduction-order is 16x16 tile per workgroup with a per-K-chunk barrier;
/// the CPU reference is fully sequential. The drift between the two is
/// dominated by Q4_K rounding, well below the tolerances above.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanMatMulQ4KGemmF32KernelTests
{
    private const int Q4KGroupSize = 256;
    private const int Q4KBlockBytes = 144;
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ4KGemmF32KernelTests(ITestOutputHelper output)
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
        float[] weightsF32 = Q4KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);

        int blocksPerRow = k / Q4KGroupSize;
        int rowBytes = blocksPerRow * Q4KBlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ4K = Q4KFixture.QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ4K.Length);

        Q4KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ4K, m, k);

        float[] expected = Q4KFixture.CpuGemmQ4K(weightsQ4K, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ4KGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ4K), bufW);
        device.Upload(inputB, bufB);

        var sw = Stopwatch.StartNew();
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        sw.Stop();

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        _output.WriteLine($"Q4_K GEMM dispatch (n={n}, m={m}, k={k}): {sw.Elapsed.TotalMilliseconds:F2} ms");
        Q4KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
