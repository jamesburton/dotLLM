using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the Vulkan BF16 native GEMV kernel.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy: random FP32 weights → BF16 (round-to-nearest-even
/// via the standard "add 0x7FFF + parity-bit, take top 16" trick — matches
/// PyTorch's <c>tensor.to(bfloat16)</c> bit-for-bit). Reference: a scalar
/// CPU GEMV that reads the same BF16 bytes the GPU shader sees and decodes
/// via shift-left-16 + reinterpret-as-F32 — the exact inverse the shader
/// uses (<c>uintBitsToFloat(bf16_bits &lt;&lt; 16)</c>).
/// </para>
/// <para>
/// Tolerance: <b>abs 1e-2 / rel 5e-3</b> — looser than F16's abs 5e-3 / rel
/// 1e-3 because BF16 has only ~7 mantissa bits (vs F16's 10). At unit-scale
/// outputs and K ≤ 1024 the worst-case round-trip drift is ~5e-3; we set
/// 1e-2 absolute to leave headroom for the workgroup tree-reduce vs. CPU
/// block-sequential reduction order delta. The shader's BF16 decode itself
/// is bit-exact (no rounding on the read side), so the only source of drift
/// versus the CPU reference is reduction order.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulBf16GemvF32KernelTests
{
    private const float AbsTol = 1e-2f;
    private const float RelTol = 5e-3f;

    [SkippableTheory]
    [InlineData(1, 32)]
    [InlineData(8, 64)]
    [InlineData(4, 128)]
    [InlineData(16, 256)]
    [InlineData(2048, 512)]
    [InlineData(576, 768)]
    [InlineData(1024, 1024)]
    [InlineData(4096, 4096)]
    public void Launch_MatchesCpuReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xBF16 + m * 7 + k * 11);
        float[] weightsF32 = F16Bf16Fixture.RandomFloats(rng, m * k, range: 0.1f);
        float[] x = F16Bf16Fixture.RandomFloats(rng, k, range: 1.0f);

        byte[] weightsBf16 = F16Bf16Fixture.QuantizeRowsBf16(weightsF32, m, k);
        Assert.Equal((long)m * k * 2, weightsBf16.Length);

        float[] expected = F16Bf16Fixture.CpuGemvBf16(weightsBf16, x, m, k);

        using var device = VulkanDevice.Create();
        using var kernel = MatMulBf16GemvF32Kernel.Create(device, spvDir);

        long weightsBufBytes = ((long)weightsBf16.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsBf16), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        F16Bf16Fixture.AssertClose(expected, actual, m, k, AbsTol, RelTol);
    }
}
