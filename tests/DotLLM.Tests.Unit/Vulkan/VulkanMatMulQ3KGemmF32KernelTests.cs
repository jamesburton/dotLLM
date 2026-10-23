using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q3_K batched GEMM (prefill path).
/// Mirrors <see cref="VulkanMatMulQ3KGemvF32KernelTests"/>: byte-identical Q3_K
/// weights, GPU vs scalar CPU reference both reading the same bytes.
/// </summary>
[Trait("Category", "GPU")]
public class VulkanMatMulQ3KGemmF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ3KGemmF32KernelTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(2, 4, 256)]
    [InlineData(1, 1, 256)]
    [InlineData(17, 33, 512)]
    [InlineData(64, 1024, 1024)]
    [InlineData(32, 1536, 768)]
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEAD_3C + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = Q3KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Q3KFixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsQ3K = Q3KFixture.QuantizeRows(weightsF32, m, k);
        Q3KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ3K, m, k);

        float[] expected = Q3KFixture.CpuGemmQ3K(weightsQ3K, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ3KGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsQ3K.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ3K), bufW);
        device.Upload(inputB, bufB);

        var sw = Stopwatch.StartNew();
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        sw.Stop();

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        _output.WriteLine($"Q3_K GEMM dispatch (n={n}, m={m}, k={k}): {sw.Elapsed.TotalMilliseconds:F2} ms");
        Q3KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
