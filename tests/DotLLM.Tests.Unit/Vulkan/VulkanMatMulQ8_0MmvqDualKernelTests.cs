using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for the fused dual-output Q8_0 MMVQ decode GEMV (issue #71):
/// <see cref="MatMulQ8_0MmvqDualKernel"/> computes two independent same-K
/// GEMVs (e.g. FFN gate_proj + up_proj) sharing one pre-quantized Q8_1
/// activation in a single dispatch, replacing two separate
/// <see cref="MatMulQ8_0MmvqKernel"/> dispatches.
/// </summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ8_0MmvqDualKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    /// <summary>
    /// Bit-identical parity: the fused dual dispatch must produce EXACTLY the
    /// same per-matrix output as running <see cref="MatMulQ8_0MmvqKernel"/>
    /// twice against the same shared xq/xds scratch (the pre-#71 shared-quant
    /// behaviour). The per-row shader body is a verbatim copy, so any
    /// mismatch here means the row-selection / buffer-routing logic in the
    /// dual shader is wrong — not an activation-quant tolerance question.
    /// </summary>
    [SkippableTheory]
    [InlineData(576, 1536, 1536)]   // SmolLM-135M gate/up shape (issue #71 profiling target)
    [InlineData(2048, 8192, 8192)]  // Llama-3.2-1B gate/up shape
    [InlineData(64, 32, 96)]        // tiny + asymmetric Ma/Mb
    [InlineData(96, 160, 64)]       // odd blocksPerRow (K=96 -> 3 blocks/row), asymmetric
    public void Dual_MatchesTwoSeparateMmvqDispatches_BitIdentical(int k, int mA, int mB)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq.spv missing or unsupported.");
        using var dual = MatMulQ8_0MmvqDualKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq_dual.spv missing or unsupported.");

        var rng = new Random(0x71 + k * 31 + mA * 7 + mB);
        float[] x = RandomFloats(rng, k, range: 1.0f);
        byte[] wA = QuantizeRows(RandomFloats(rng, mA * k, range: 0.1f), mA, k);
        byte[] wB = QuantizeRows(RandomFloats(rng, mB * k, range: 0.1f), mB, k);

        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufWa = device.Allocate(((long)wA.Length + 3) & ~3L);
        using var bufWb = device.Allocate(((long)wB.Length + 3) & ~3L);
        using var bufYaShared = device.Allocate((long)mA * sizeof(float));
        using var bufYbShared = device.Allocate((long)mB * sizeof(float));
        using var bufYaSeparate = device.Allocate((long)mA * sizeof(float));
        using var bufYbSeparate = device.Allocate((long)mB * sizeof(float));

        device.Upload(x, bufX);
        device.Upload(new ReadOnlySpan<byte>(wA), bufWa);
        device.Upload(new ReadOnlySpan<byte>(wB), bufWb);

        // Fused dual dispatch.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            dual.Record(ctx.CommandBuffer, bufWa, bufWb, bufXq, bufXds,
                bufYaShared, bufYbShared, mA, mB, k);
            ctx.SubmitAndWait();
        }

        // Two separate MMVQ dispatches against the same shared quantized activation
        // (mirrors RecordSharedInputMmvqGroup's pre-#71 loop, no inter-GEMV barrier).
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bufWa, bufXq, bufXds, bufYaSeparate, mA, k);
            mmvq.Record(ctx.CommandBuffer, bufWb, bufXq, bufXds, bufYbSeparate, mB, k);
            ctx.SubmitAndWait();
        }

        float[] yaShared = new float[mA];
        float[] ybShared = new float[mB];
        float[] yaSeparate = new float[mA];
        float[] ybSeparate = new float[mB];
        device.Download(bufYaShared, yaShared);
        device.Download(bufYbShared, ybShared);
        device.Download(bufYaSeparate, yaSeparate);
        device.Download(bufYbSeparate, ybSeparate);

        for (int i = 0; i < mA; i++)
            Assert.True(yaShared[i].Equals(yaSeparate[i]),
                $"A mismatch at idx {i} (k={k},mA={mA},mB={mB}): dual={yaShared[i]:G9}, separate={yaSeparate[i]:G9}.");
        for (int i = 0; i < mB; i++)
            Assert.True(ybShared[i].Equals(ybSeparate[i]),
                $"B mismatch at idx {i} (k={k},mA={mA},mB={mB}): dual={ybShared[i]:G9}, separate={ybSeparate[i]:G9}.");
    }

    /// <summary>
    /// Sanity cross-check against the CPU F32 oracle (argmax + loose tolerance,
    /// same bar as <c>VulkanMatMulQ8_0MmvqKernelTests</c>) — confirms the fused
    /// dual kernel isn't merely self-consistent but actually computes the right
    /// GEMV for BOTH halves.
    /// </summary>
    [SkippableFact]
    public void Dual_MatchesF32Oracle_ArgmaxAndTolerance()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var dual = MatMulQ8_0MmvqDualKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq_dual.spv missing or unsupported.");

        const int k = 576, mA = 1536, mB = 1536; // SmolLM-135M FFN gate/up shape
        var rng = new Random(0xD0A1);
        float[] x = RandomFloats(rng, k, range: 1.0f);
        byte[] wA = QuantizeRows(RandomFloats(rng, mA * k, range: 0.1f), mA, k);
        byte[] wB = QuantizeRows(RandomFloats(rng, mB * k, range: 0.1f), mB, k);

        float[] expectedA = CpuGemvQ8_0F32In(wA, x, mA, k);
        float[] expectedB = CpuGemvQ8_0F32In(wB, x, mB, k);

        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufWa = device.Allocate(((long)wA.Length + 3) & ~3L);
        using var bufWb = device.Allocate(((long)wB.Length + 3) & ~3L);
        using var bufYa = device.Allocate((long)mA * sizeof(float));
        using var bufYb = device.Allocate((long)mB * sizeof(float));

        device.Upload(x, bufX);
        device.Upload(new ReadOnlySpan<byte>(wA), bufWa);
        device.Upload(new ReadOnlySpan<byte>(wB), bufWb);

        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            dual.Record(ctx.CommandBuffer, bufWa, bufWb, bufXq, bufXds, bufYa, bufYb, mA, mB, k);
            ctx.SubmitAndWait();
        }

        float[] actualA = new float[mA];
        float[] actualB = new float[mB];
        device.Download(bufYa, actualA);
        device.Download(bufYb, actualB);

        AssertParity(expectedA, actualA, mA, k, "A");
        AssertParity(expectedB, actualB, mB, k, "B");
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers (mirrors VulkanMatMulQ8_0MmvqKernelTests)
    // ─────────────────────────────────────────────────────────────

    private const float RelTol = 3e-2f;

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

    private static unsafe float[] CpuGemvQ8_0F32In(byte[] weightsQ8, float[] x, int m, int k)
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

    private static void AssertParity(float[] expected, float[] actual, int m, int k, string label)
    {
        Assert.Equal(expected.Length, actual.Length);

        int argE = 0, argA = 0;
        for (int i = 1; i < m; i++)
        {
            if (expected[i] > expected[argE]) argE = i;
            if (actual[i] > actual[argA]) argA = i;
        }
        Assert.True(argE == argA,
            $"[{label}] Argmax mismatch (m={m},k={k}): oracle={argE} ({expected[argE]:G6}), " +
            $"dual={argA} ({actual[argA]:G6}).");

        double ss = 0;
        for (int i = 0; i < m; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, m));
        float absTol = MathF.Max(rms, 1e-6f) * RelTol;

        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < m; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"[{label}] Dual MMVQ drift exceeded tolerance (m={m},k={k}): errors={errors}/{m}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }
}
