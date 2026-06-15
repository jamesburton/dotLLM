using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the DP4a (integer dot product) Q8_0 GEMV kernel
/// (<see cref="MatMulQ8_0Dp4aKernel"/>).
/// </summary>
/// <remarks>
/// <para>
/// The DP4a kernel quantizes the FP32 activation slice to INT8 per 32-element
/// block before the dot product, so a straight comparison against the
/// FP32-activation scalar path would conflate the (expected) ≈0.8%/element
/// activation-quant error with any actual kernel bug. The primary assertion
/// therefore compares against a CPU reference that quantizes <c>x</c>
/// <b>identically</b> (same per-block scale, same INT8 rounding/clamp, same
/// integer dot) — a tight bound that isolates bit-unpack / packing / scale-combine
/// bugs. The looser FP32-activation drift is reported for context only.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ8_0Dp4aKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // Matched-quant reference: exact except cross-block FP summation order and
    // fp16 scale rounding (identical on both sides). 5e-3 rel / 1e-3 abs is well
    // above that floor and below the ≈0.8% activation-quant error vs FP32.
    private const float AbsTol = 1e-3f;
    private const float RelTol = 5e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ8_0Dp4aKernelTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(1, 32)]
    [InlineData(8, 64)]
    [InlineData(4, 128)]
    [InlineData(49152, 576)]   // SmolLM lm_head / vocab-size output
    [InlineData(576, 576)]     // SmolLM q/k/v projection
    [InlineData(1536, 576)]    // SmolLM gate/up projection
    [InlineData(576, 1536)]    // SmolLM down projection
    [InlineData(8, 32)]        // K=32, M>1 — issue #1 stride family
    [InlineData(4, 96)]        // K=96, blocksPerRow=3 (odd)
    [InlineData(2, 160)]       // K=160, blocksPerRow=5 (odd)
    public void Launch_MatchesMatchedQuantReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' does not support VK_KHR_shader_integer_dot_product.");

        var rng = new Random(0xD9A4 + m * 7 + k);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] x = RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ8.Length);

        // Primary reference: quantize x to INT8 per block, integer dot, two-scale.
        float[] expected = CpuGemvQ8_0Dp4a(weightsQ8, x, m, k);
        // Context reference: FP32 activation (no x-quant) — shows the activation-quant error.
        float[] fp32Ref = CpuGemvQ8_0Fp32X(weightsQ8, x, m, k);

        using var kernel = MatMulQ8_0Dp4aKernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        ReportFp32Drift(actual, fp32Ref, m, k);
        AssertClose(expected, actual, m, k);
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
                MatMul.QuantizeF32ToQ8_0(srcPtr + (long)row * k, dstPtr + (long)row * rowBytes, k);
        }
        return dst;
    }

    /// <summary>
    /// CPU mirror of the DP4a shader: per-block INT8 activation quant
    /// (<c>dx = max|x|/127</c>, <c>xq = round(x/dx)</c> clamped), integer dot
    /// against the int8 weights, accumulated as <c>d * dx * dot</c>.
    /// </summary>
    private static unsafe float[] CpuGemvQ8_0Dp4a(byte[] weightsQ8, float[] x, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var result = new float[m];

        fixed (byte* wPtr = weightsQ8)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0f;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Q8_0BlockBytes;
                    float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block);
                    sbyte* qs = (sbyte*)(block + 2);

                    int xBase = b * Q8_0GroupSize;
                    float maxabs = 0f;
                    for (int j = 0; j < Q8_0GroupSize; j++)
                        maxabs = MathF.Max(maxabs, MathF.Abs(x[xBase + j]));
                    float inv = maxabs > 0f ? 127f / maxabs : 0f;
                    float dx = maxabs / 127f;

                    int dot = 0;
                    for (int j = 0; j < Q8_0GroupSize; j++)
                    {
                        int xq = (int)MathF.Round(x[xBase + j] * inv);
                        xq = Math.Clamp(xq, -127, 127);
                        dot += qs[j] * xq;
                    }
                    sum += d * dx * dot;
                }
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>FP32-activation reference (no x-quant) for context drift reporting.</summary>
    private static unsafe float[] CpuGemvQ8_0Fp32X(byte[] weightsQ8, float[] x, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var result = new float[m];

        fixed (byte* wPtr = weightsQ8)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0f;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Q8_0BlockBytes;
                    float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block);
                    sbyte* qs = (sbyte*)(block + 2);
                    float blockSum = 0f;
                    for (int j = 0; j < Q8_0GroupSize; j++)
                        blockSum += qs[j] * x[b * Q8_0GroupSize + j];
                    sum += d * blockSum;
                }
                result[row] = sum;
            }
        }
        return result;
    }

    private void ReportFp32Drift(float[] actual, float[] fp32Ref, int m, int k)
    {
        float maxRel = 0f;
        for (int i = 0; i < actual.Length; i++)
        {
            float rel = MathF.Abs(actual[i] - fp32Ref[i]) / MathF.Max(MathF.Abs(fp32Ref[i]), 1e-7f);
            if (rel > maxRel) maxRel = rel;
        }
        _output.WriteLine($"DP4a (m={m},k={k}): vs FP32-activation ref maxRel={maxRel:P2} (expected ≈ activation-quant error)");
    }

    private void AssertClose(float[] expected, float[] actual, int m, int k)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        _output.WriteLine($"DP4a (m={m},k={k}): vs matched-quant ref maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, errors={errors}/{expected.Length}");
        Assert.True(errors == 0,
            $"Numerical drift exceeded tolerance (m={m},k={k}): errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
