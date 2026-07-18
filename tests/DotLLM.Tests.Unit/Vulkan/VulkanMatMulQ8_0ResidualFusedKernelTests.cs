using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Bit-exact parity tests for the residual-fused Q8_0 matmul kernels (issue
/// #379): <see cref="MatMulQ8_0MmvqResidualKernel"/> (decode GEMV) and
/// <see cref="MatMulQ8_0MmqResidualKernel"/> (prefill GEMM).
/// </summary>
/// <remarks>
/// Unlike the tolerance-based <c>VulkanMatMulQ8_0Mmvq/MmqKernelTests</c> (which
/// compare against a CPU F32 oracle across the int8-activation-quant error
/// floor), these tests compare the fused kernel directly against the SAME
/// GPU device's unfused (matmul → barrier → <see cref="AddKernel"/>) pair,
/// using the SAME quantized activation buffers for both runs. Both paths run
/// the identical dp4a reduction; the fused shader only changes the final
/// store from <c>c[idx] = value</c> to <c>c[idx] = value + residual[idx]</c>.
/// This must therefore be EXACTLY bit-identical — no tolerance — per the
/// issue's "exact-token parity maintained" acceptance criterion.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ8_0ResidualFusedKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // ─────────────────────────────────────────────────────────────
    // MMVQ (decode, seqLen == 1) residual fusion
    // ─────────────────────────────────────────────────────────────

    [SkippableTheory]
    [InlineData(32, 64)]     // tiny: 2 blocks per row
    [InlineData(1, 32)]      // single output row, one block
    [InlineData(64, 128)]    // 4 blocks per row, odd row-byte align (4*34=136)
    [InlineData(576, 1536)]  // SmolLM-135M o_proj-ish decode shape (issue #379 motivating case)
    [InlineData(4, 96)]      // K=96, blocksPerRow=3 (odd) — phase/stride family
    public void MmvqResidual_BitExact_VsUnfusedMatmulThenAdd(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq.spv missing or unsupported.");
        using var mmvqResidual = MatMulQ8_0MmvqResidualKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq_residual.spv missing or unsupported.");
        using var add = AddKernel.Create(device, spvDir);

        var rng = new Random(0x379 + m * 7 + k);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] x = RandomFloats(rng, k, range: 1.0f);
        float[] residual = RandomFloats(rng, m, range: 2.0f);

        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);

        long weightsBufBytes = ((long)weightsQ8.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufResidual = device.Allocate((long)m * sizeof(float));
        using var bufYUnfused = device.Allocate((long)m * sizeof(float));
        using var bufYAdded = device.Allocate((long)m * sizeof(float));
        using var bufYFused = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(x, bufX);
        device.Upload(residual, bufResidual);

        // Unfused: quantize once, MMVQ, then a separate AddKernel dispatch —
        // the exact pre-#379 call sequence in VulkanTransformerModel.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufYUnfused, m, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            add.Record(ctx.CommandBuffer, bufResidual, bufYUnfused, bufYAdded, m);
            ctx.SubmitAndWait();
        }

        // Fused: quantize once, single residual-fused MMVQ dispatch.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvqResidual.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufResidual, bufYFused, m, k);
            ctx.SubmitAndWait();
        }

        float[] added = new float[m];
        float[] fused = new float[m];
        device.Download(bufYAdded, added);
        device.Download(bufYFused, fused);

        AssertBitExact(added, fused, m, k);
    }

    // ─────────────────────────────────────────────────────────────
    // MMQ (prefill, seqLen > 1) residual fusion
    // ─────────────────────────────────────────────────────────────

    [SkippableTheory]
    [InlineData(2, 4, 32)]      // tiny: one block per row, partial tile
    [InlineData(1, 1, 32)]      // single-cell output
    [InlineData(65, 65, 64)]    // crosses the 64x64 tile boundary on BOTH axes
    [InlineData(17, 33, 64)]    // non-multiple-of-tile N/M, odd row-byte align (2*34=68)
    [InlineData(8, 2048, 2048)] // Llama-3.2-1B-ish prefill shape
    [InlineData(7, 4, 96)]      // K=96, blocksPerRow=3 (odd) — phase/stride family
    public void MmqResidual_BitExact_VsUnfusedMatmulThenAdd(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing or unsupported.");
        using var mmqResidual = MatMulQ8_0MmqResidualKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq_residual.spv missing or unsupported.");
        using var add = AddKernel.Create(device, spvDir);

        var rng = new Random(0x379 + n * 13 + m * 7 + k * 3);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = RandomFloats(rng, n * k, range: 1.0f);
        float[] residual = RandomFloats(rng, n * m, range: 2.0f);

        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);

        long weightsBufBytes = ((long)weightsQ8.Length + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bufResidual = device.Allocate((long)n * m * sizeof(float));
        using var bufCUnfused = device.Allocate((long)n * m * sizeof(float));
        using var bufCAdded = device.Allocate((long)n * m * sizeof(float));
        using var bufCFused = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(inputB, bufB);
        device.Upload(residual, bufResidual);

        // Unfused: quantize rows once, MMQ, then a separate AddKernel dispatch.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n: n, k: k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufCUnfused, m: m, k: k, n: n);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            add.Record(ctx.CommandBuffer, bufResidual, bufCUnfused, bufCAdded, n * m);
            ctx.SubmitAndWait();
        }

        // Fused: quantize rows once, single residual-fused MMQ dispatch.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n: n, k: k);
            KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmqResidual.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufResidual, bufCFused, m: m, k: k, n: n);
            ctx.SubmitAndWait();
        }

        float[] added = new float[n * m];
        float[] fused = new float[n * m];
        device.Download(bufCAdded, added);
        device.Download(bufCFused, fused);

        AssertBitExact(added, fused, n * m, k);
    }

    // ─────────────────────────────────────────────────────────────
    // Fallback safety: fused kernels degrade to null (not a throw) when
    // their SPV isn't present, mirroring the base (unfused) kernels'
    // TryCreate contract — VulkanTransformerModel's RecordMatmul /
    // TryRecordMatmulWithResidualQ8_0 rely on this to fall back to the
    // unfused matmul + AddKernel pair for any model/device that doesn't
    // qualify for fusion.
    // ─────────────────────────────────────────────────────────────

    [SkippableFact]
    public void MmvqResidual_TryCreate_ReturnsNull_WhenSpvMissing()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product.");

        string bogusDir = Path.Combine(Path.GetTempPath(), "dotllm-no-spv-" + Guid.NewGuid());
        Assert.Null(MatMulQ8_0MmvqResidualKernel.TryCreate(device, bogusDir));
    }

    [SkippableFact]
    public void MmqResidual_TryCreate_ReturnsNull_WhenSpvMissing()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product.");

        string bogusDir = Path.Combine(Path.GetTempPath(), "dotllm-no-spv-" + Guid.NewGuid());
        Assert.Null(MatMulQ8_0MmqResidualKernel.TryCreate(device, bogusDir));
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
    /// Exact float equality (no tolerance): both paths run the identical
    /// dp4a reduction on the same GPU with the same inputs, so the fused
    /// store differs from the unfused-then-add result only in dispatch
    /// count, never in the resulting bit pattern.
    /// </summary>
    private static void AssertBitExact(float[] expected, float[] actual, int count, int k)
    {
        Assert.Equal(expected.Length, actual.Length);
        int mismatches = 0;
        for (int i = 0; i < count; i++)
        {
            if (!BitsEqual(expected[i], actual[i]))
                mismatches++;
        }
        Assert.True(mismatches == 0,
            $"Fused vs unfused-then-add mismatch (count={count}, k={k}): {mismatches}/{count} elements differ. " +
            $"First expected={expected[0]:G9}, actual={actual[0]:G9}.");
    }

    private static bool BitsEqual(float a, float b) =>
        BitConverter.SingleToInt32Bits(a) == BitConverter.SingleToInt32Bits(b);
}
