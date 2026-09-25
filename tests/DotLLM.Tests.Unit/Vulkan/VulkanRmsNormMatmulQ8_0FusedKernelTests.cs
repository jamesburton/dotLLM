using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the fused (RMSNorm + Q8_0 GEMV) kernel.
/// </summary>
/// <remarks>
/// <para>
/// Reference is a CPU computation that mirrors the shader's algorithm:
/// rmsnorm of the FP32 hidden state, then Q8_0 GEMV against the
/// quantised weights. We avoid using the standalone GPU
/// <c>MatMulQ8_0Kernel</c> as reference because the GEMV shader has a
/// latent stride bug at K=32 with M&gt;1 (it uses <c>rowUints*4</c>
/// instead of <c>blocksPerRow*34</c>; tracked as issue #1). The fused
/// shader uses the exact <c>blocksPerRow*34</c> stride matching the GEMM
/// shader and the CPU reference, so a CPU-side comparison is the
/// strictest available.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanRmsNormMatmulQ8_0FusedKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const float AbsTol = 1e-4f;
    private const float RelTol = 1e-3f;
    private const float Eps = 1e-5f;

    [SkippableTheory]
    [InlineData(8, 32)]                   // minimum: 1 block, leader covers entire K via 1 WG
    [InlineData(64, 128)]                 // 4 blocks per row
    [InlineData(576, 576)]                // SmolLM-135M Q proj — square, M = K
    [InlineData(192, 576)]                // SmolLM-135M K/V proj — fewer rows than K
    [InlineData(1536, 576)]               // SmolLM-135M Gate / Up — more rows than K
    [InlineData(49152, 576)]              // SmolLM-135M lm_head shape
    public void Launch_MatchesStandalonePair(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        var rng = new Random(0xF00D + m * 7 + k);
        float[] hidden = RandomFloats(rng, k, range: 0.5f);
        float[] normWeight = RandomFloats(rng, k, range: 0.1f);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);

        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);

        // ─── CPU reference: rmsnorm + Q8_0 GEMV on the same byte blob. ───
        float[] expectedNorm = CpuRmsNorm(hidden, normWeight, Eps);
        float[] expectedY = CpuGemvQ8_0(weightsQ8, expectedNorm, m, k);

        using var device = VulkanDevice.Create();
        var fused = RmsNormMatmulQ8_0FusedKernel.TryCreate(device, spvDir);
        Skip.If(fused is null, "Fused rmsnorm+matmul SPV not present.");

        long weightsBufBytes = ((long)weightsQ8.Length + 3) & ~3L;
        using var bufHidden = device.Allocate((long)k * sizeof(float));
        using var bufNormW = device.Allocate((long)k * sizeof(float));
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufNormOut = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(hidden, bufHidden);
        device.Upload(normWeight, bufNormW);
        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);

        fused!.Launch(bufHidden, bufNormW, bufW, bufNormOut, bufY, m, k, Eps);

        float[] actualNorm = new float[k];
        float[] actualY = new float[m];
        device.Download(bufNormOut, actualNorm);
        device.Download(bufY, actualY);

        AssertClose(expectedNorm, actualNorm, m, k, "normalised hidden state");
        AssertClose(expectedY, actualY, m, k, "matmul output");

        fused.Dispose();
    }

    private static float[] CpuRmsNorm(float[] x, float[] weight, float eps)
    {
        int n = x.Length;
        double sumSq = 0.0;
        for (int i = 0; i < n; i++) sumSq += (double)x[i] * x[i];
        double rinv = 1.0 / Math.Sqrt(sumSq / n + eps);
        var result = new float[n];
        for (int i = 0; i < n; i++)
            result[i] = (float)(x[i] * rinv * weight[i]);
        return result;
    }

    private static unsafe float[] CpuGemvQ8_0(byte[] weightsQ8, float[] x, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var result = new float[m];
        fixed (byte* wPtr = weightsQ8)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Q8_0BlockBytes;
                    float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block);
                    sbyte* qs = (sbyte*)(block + 2);
                    float blockSum = 0;
                    for (int j = 0; j < Q8_0GroupSize; j++)
                        blockSum += (float)qs[j] * x[b * Q8_0GroupSize + j];
                    sum += d * blockSum;
                }
                result[row] = sum;
            }
        }
        return result;
    }

    [SkippableFact]
    public void Create_ReturnsNull_WhenSpvMissing()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        var nonexistent = Path.Combine(Path.GetTempPath(), "dotllm-no-spv-here-" + Guid.NewGuid());
        var kernel = RmsNormMatmulQ8_0FusedKernel.TryCreate(device, nonexistent);
        Assert.Null(kernel);
    }

    [Fact]
    public void SupportsHiddenSize_RespectsCap()
    {
        Assert.True(RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(32));
        Assert.True(RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(576));
        Assert.True(RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(RmsNormMatmulQ8_0FusedKernel.MaxHiddenSize));
        Assert.False(RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(0));
        Assert.False(RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(-1));
        Assert.False(RmsNormMatmulQ8_0FusedKernel.SupportsHiddenSize(RmsNormMatmulQ8_0FusedKernel.MaxHiddenSize + 1));
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    private static unsafe byte[] QuantizeRows(float[] src, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var dst = new byte[m * rowBytes];
        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
            {
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, dstPtr + (long)row * rowBytes, k);
            }
        }
        return dst;
    }

    private static void AssertClose(float[] expected, float[] actual, int m, int k, string label)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"Numerical drift exceeded tolerance ({label}, m={m},k={k}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
