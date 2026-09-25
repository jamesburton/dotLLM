using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan Q2_K batched GEMM (prefill path).
/// Mirrors <see cref="VulkanMatMulQ2KGemvF32KernelTests"/>: byte-identical Q2_K
/// weights from <see cref="Q2KFixture.QuantizeRows"/>, GPU vs scalar CPU
/// reference both reading the same bytes. Tolerances match the higher
/// K-quants (5e-3 abs / 1e-3 rel) — the test compares Q2_K-byte-equal paths,
/// so the only source of drift is reduction order (16x16 tile + per-K-chunk
/// barrier on GPU vs sequential on CPU).
/// </summary>
[Trait("Category", "GPU")]
public class VulkanMatMulQ2KGemmF32KernelTests
{
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ2KGemmF32KernelTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(2, 4, 256)]
    [InlineData(1, 1, 256)]
    [InlineData(17, 33, 512)]
    [InlineData(64, 1024, 1024)]
    [InlineData(32, 1536, 768)]
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBEAD_2C + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = Q2KFixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = Q2KFixture.RandomFloats(rng, n * k, range: 1.0f);

        byte[] weightsQ2K = Q2KFixture.QuantizeRows(weightsF32, m, k);
        Q2KFixture.AssertFixtureRoundtrip(weightsF32, weightsQ2K, m, k);

        float[] expected = Q2KFixture.CpuGemmQ2K(weightsQ2K, inputB, m, k, n);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulQ2KGemmF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsQ2K.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ2K), bufW);
        device.Upload(inputB, bufB);

        var sw = Stopwatch.StartNew();
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        sw.Stop();

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        _output.WriteLine($"Q2_K GEMM dispatch (n={n}, m={m}, k={k}): {sw.Elapsed.TotalMilliseconds:F2} ms");
        Q2KFixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
