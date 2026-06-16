using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity test for the pre-quantized DP4a Q8_0 GEMV
/// (<see cref="MatMulQ8_0Dp4aPqKernel"/>). The shared <c>quantize_q8_act</c> pass
/// uses the same per-32-block activation scale as the inline DP4a kernel, so the
/// result must match the same matched-quant CPU reference exactly.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ8_0Dp4aPqKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const float AbsTol = 1e-3f;
    private const float RelTol = 5e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ8_0Dp4aPqKernelTests(ITestOutputHelper output) => _output = output;

    [SkippableTheory]
    [InlineData(1, 32)]
    [InlineData(8, 64)]
    [InlineData(4, 128)]
    [InlineData(49152, 576)]   // lm_head — the high-M shape the inline variant regressed on
    [InlineData(576, 576)]
    [InlineData(1536, 576)]
    [InlineData(576, 1536)]
    [InlineData(8, 32)]
    [InlineData(4, 96)]
    [InlineData(2, 160)]
    public void Launch_MatchesMatchedQuantReference(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' does not support VK_KHR_shader_integer_dot_product.");

        var rng = new Random(0xC0DE + m * 7 + k);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] x = RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q8_0GroupSize;
        int totalBytes = m * blocksPerRow * Q8_0BlockBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);

        float[] expected = CpuGemvQ8_0Dp4a(weightsQ8, x, m, k);

        using var kernel = MatMulQ8_0Dp4aPqKernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(MatMulQ8_0Dp4aPqKernel.XqScratchBytes(k));
        using var bufDx = device.Allocate(MatMulQ8_0Dp4aPqKernel.DxScratchBytes(k));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(x, bufX);

        kernel.Launch(bufW, bufX, bufXq, bufDx, bufY, m, k);

        float[] actual = new float[m];
        device.Download(bufY, actual);

        AssertClose(expected, actual, m, k);
    }

    /// <summary>
    /// Validates the split API used by the forward-pass shared-activation path:
    /// one <see cref="MatMulQ8_0Dp4aPqKernel.RecordQuantizeActivation"/> feeding
    /// two <see cref="MatMulQ8_0Dp4aPqKernel.RecordGemvPrequant"/> calls (K/V-shaped:
    /// same input, different output dims) must produce <b>bit-identical</b> results
    /// to two independent full <see cref="MatMulQ8_0Dp4aPqKernel.Launch"/> calls.
    /// The quantize pass is deterministic, so sharing it cannot change the output —
    /// this guards the refactor that split <c>Record</c> into the two passes.
    /// </summary>
    [SkippableFact]
    public unsafe void SharedQuant_TwoGemvs_MatchPerMatmulLaunch()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' does not support VK_KHR_shader_integer_dot_product.");

        const int k = 576;          // shared contraction dim (K/V both contract over hidden)
        const int mA = 1536;        // K-like output dim
        const int mB = 512;         // V-like output dim (distinct, to catch dim mix-ups)

        var rng = new Random(0x5A5A + k);
        byte[] wqA = QuantizeRows(RandomFloats(rng, mA * k, range: 0.1f), mA, k);
        byte[] wqB = QuantizeRows(RandomFloats(rng, mB * k, range: 0.1f), mB, k);
        float[] x = RandomFloats(rng, k, range: 1.0f);

        using var kernel = MatMulQ8_0Dp4aPqKernel.Create(device, spvDir);

        using var bufWA = device.Allocate((wqA.LongLength + 3) & ~3L);
        using var bufWB = device.Allocate((wqB.LongLength + 3) & ~3L);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(MatMulQ8_0Dp4aPqKernel.XqScratchBytes(k));
        using var bufDx = device.Allocate(MatMulQ8_0Dp4aPqKernel.DxScratchBytes(k));
        using var bufYa = device.Allocate((long)mA * sizeof(float));
        using var bufYb = device.Allocate((long)mB * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(wqA), bufWA);
        device.Upload(new ReadOnlySpan<byte>(wqB), bufWB);
        device.Upload(x, bufX);

        // Reference: two independent full Launch calls (each re-quantizes x).
        kernel.Launch(bufWA, bufX, bufXq, bufDx, bufYa, mA, k);
        float[] refA = new float[mA];
        device.Download(bufYa, refA);
        kernel.Launch(bufWB, bufX, bufXq, bufDx, bufYb, mB, k);
        float[] refB = new float[mB];
        device.Download(bufYb, refB);

        // Shared: one quantize, a barrier, then both GEMVs read the shared scratch.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            kernel.RecordQuantizeActivation(ctx.CommandBuffer, bufX, bufXq, bufDx, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            kernel.RecordGemvPrequant(ctx.CommandBuffer, bufWA, bufXq, bufDx, bufYa, mA, k);
            kernel.RecordGemvPrequant(ctx.CommandBuffer, bufWB, bufXq, bufDx, bufYb, mB, k);
            ctx.SubmitAndWait();
        }
        float[] sharedA = new float[mA];
        float[] sharedB = new float[mB];
        device.Download(bufYa, sharedA);
        device.Download(bufYb, sharedB);

        for (int i = 0; i < mA; i++)
            Assert.True(refA[i] == sharedA[i],
                $"Shared-quant A row {i} differs: per-matmul={refA[i]:G9} vs shared={sharedA[i]:G9}");
        for (int i = 0; i < mB; i++)
            Assert.True(refB[i] == sharedB[i],
                $"Shared-quant B row {i} differs: per-matmul={refB[i]:G9} vs shared={sharedB[i]:G9}");
    }

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
    /// CPU mirror: per-block INT8 activation quant (<c>dx = max|x|/127</c>,
    /// <c>xq = round(x/dx)</c> clamped), integer dot, <c>d * dx * dot</c>.
    /// Identical to the inline DP4a reference — the prequant pass produces the
    /// same per-block scales.
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
        _output.WriteLine($"DP4a-PQ (m={m},k={k}): maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, errors={errors}/{expected.Length}");
        Assert.True(errors == 0,
            $"Numerical drift exceeded tolerance (m={m},k={k}): errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
