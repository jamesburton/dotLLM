using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a MMVQ decode GEMV (issue #46):
/// <c>QuantizeQ8_1Kernel</c> (F32 activation → Q8_1) followed by
/// <c>MatMulQ8_0MmvqKernel</c> (integer-dot GEMV against int8 Q8_0 weights).
/// </summary>
/// <remarks>
/// <para>
/// MMVQ is NOT bit-exact vs the F32-in Q8_0 GEMV: the activation vector is
/// quantized to int8 (Q8_1) first, so the exact kernel-parity test used by
/// <c>VulkanMatMulQ8_0KernelTests</c> cannot apply. Instead we compare against
/// the CPU F32 oracle (Q8_0-quantized weights · FP32 x) with:
/// </para>
/// <list type="bullet">
///   <item><b>argmax-exact</b> — the position of the max output element must
///     match the oracle. A broken kernel (wrong unpack / scale / accumulate)
///     shifts the argmax; int8-activation quant does not.</item>
///   <item><b>loose abs/rel tolerance</b> — sized to the int8-activation-quant
///     error floor (per-32-block scale, ~1/127 relative per element, averaged
///     over K). Tight enough to catch a structurally broken kernel, loose
///     enough to admit the expected activation-quant drift.</item>
/// </list>
/// <para>
/// Skipped when the device does not advertise
/// <c>VK_KHR_shader_integer_dot_product</c> (the SPV won't run there) — the
/// kernel's <c>TryCreate</c> returns null on such devices and the model falls
/// back to the F32-in GEMV.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ8_0MmvqKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // Tolerance for the int8-activation-quant error. The relative drift of one
    // Q8_1 element is bounded by ~1/127; summed over K with random signs the
    // RMS relative error of a length-K dot is well under 1% for these shapes.
    // 3e-2 rel / abs scaled to the output magnitude is comfortably above that
    // floor and far below what a broken kernel produces.
    private const float RelTol = 3e-2f;

    [SkippableTheory]
    [InlineData(32, 64)]                  // tiny: 2 blocks per row, phase mix
    [InlineData(64, 128)]                 // 4 blocks per row, odd row-byte align (4*34=136)
    [InlineData(2048, 2048)]              // Llama-3.2-1B q/o projection shape
    [InlineData(8192, 2048)]              // Llama-3.2-1B gate/up projection
    [InlineData(2048, 8192)]              // Llama-3.2-1B down projection
    [InlineData(4, 96)]                   // K=96, blocksPerRow=3 (odd) — phase/stride family
    [InlineData(2, 160)]                  // K=160, blocksPerRow=5 (odd) — phase/stride family
    public void Mmvq_MatchesF32Oracle_ArgmaxAndTolerance(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq.spv missing or unsupported.");

        var rng = new Random(0x46 + m * 7 + k);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] x = RandomFloats(rng, k, range: 1.0f);

        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ8.Length);

        // F32 oracle: Q8_0-quantized weights · FP32 x (the F32-in GEMV result).
        float[] expected = CpuGemvQ8_0F32In(weightsQ8, x, m, k);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        using var bufY = device.Allocate((long)m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(x, bufX);

        // Quantize x → Q8_1, then MMVQ GEMV — two dispatches in one submit so
        // the barrier between them is exercised exactly as in the forward pass.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
            DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmvq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufY, m, k);
            ctx.SubmitAndWait();
        }

        float[] actual = new float[m];
        device.Download(bufY, actual);

        AssertParity(expected, actual, m, k);
    }

    /// <summary>
    /// Bit-identical parity for the shared-activation-quant optimisation
    /// (<c>RecordSharedInputMmvqGroup</c>): a group of same-input projections
    /// (e.g. Q/K/V) must produce EXACTLY the same per-projection outputs whether
    /// the activation is quantized once and shared across the group's GEMVs
    /// (SHARE) or re-quantized per projection (NO_SHARE, the original per-call
    /// form). The quantize is deterministic and the GEMVs are identical, so the
    /// outputs are bit-for-bit equal — this discriminates a sharing
    /// implementation that accidentally clobbers the shared scratch or mis-orders
    /// the barrier (which would corrupt some GEMVs) from a correct one.
    /// </summary>
    [SkippableTheory]
    [InlineData(2048, new[] { 2048, 512, 512 })]   // Q/K/V (GQA) over hidden=2048
    [InlineData(2048, new[] { 8192, 8192 })]       // gate/up over hidden=2048
    [InlineData(96, new[] { 64, 32, 160 })]        // odd blocksPerRow mix
    public void Mmvq_SharedQuant_BitIdenticalToPerProjection(int k, int[] outDims)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMVQ path is unavailable here.");

        using var quant = QuantizeQ8_1Kernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1.spv missing.");
        using var mmvq = MatMulQ8_0MmvqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmvq.spv missing or unsupported.");

        var rng = new Random(0x5A + k);
        float[] x = RandomFloats(rng, k, range: 1.0f);

        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1Kernel.PackedBytes(k));
        using var bufXds = device.Allocate(QuantizeQ8_1Kernel.ScaleBytes(k));
        device.Upload(x, bufX);

        // Per-projection weights + output buffers (distinct, as in production).
        var bufW = new VulkanDevice.Buffer[outDims.Length];
        var sharedOut = new VulkanDevice.Buffer[outDims.Length];
        var perProjOut = new VulkanDevice.Buffer[outDims.Length];
        try
        {
            for (int p = 0; p < outDims.Length; p++)
            {
                int m = outDims[p];
                byte[] wq = QuantizeRows(RandomFloats(rng, m * k, range: 0.1f), m, k);
                long wbytes = ((long)wq.Length + 3) & ~3L;
                bufW[p] = device.Allocate(wbytes);
                sharedOut[p] = device.Allocate((long)m * sizeof(float));
                perProjOut[p] = device.Allocate((long)m * sizeof(float));
                device.Upload(new ReadOnlySpan<byte>(wq), bufW[p]);
            }

            // SHARE: one quantize, then one GEMV per projection (no inter-GEMV barrier).
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
                DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                for (int p = 0; p < outDims.Length; p++)
                    mmvq.Record(ctx.CommandBuffer, bufW[p], bufXq, bufXds, sharedOut[p], outDims[p], k);
                ctx.SubmitAndWait();
            }

            // NO_SHARE: re-quantize before each GEMV (per-call form), barriers between.
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                for (int p = 0; p < outDims.Length; p++)
                {
                    if (p > 0) DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    quant.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, k);
                    DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                    mmvq.Record(ctx.CommandBuffer, bufW[p], bufXq, bufXds, perProjOut[p], outDims[p], k);
                }
                ctx.SubmitAndWait();
            }

            for (int p = 0; p < outDims.Length; p++)
            {
                int m = outDims[p];
                float[] s = new float[m];
                float[] np = new float[m];
                device.Download(sharedOut[p], s);
                device.Download(perProjOut[p], np);
                for (int i = 0; i < m; i++)
                    Assert.True(s[i].Equals(np[i]),
                        $"Shared vs per-projection mismatch at proj {p}, idx {i} (k={k}, m={m}): " +
                        $"shared={s[i]:G9}, perProj={np[i]:G9}.");
            }
        }
        finally
        {
            foreach (var b in bufW) b?.Dispose();
            foreach (var b in sharedOut) b?.Dispose();
            foreach (var b in perProjOut) b?.Dispose();
        }
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
    /// F32-in oracle: Q8_0 weights dequantized on the fly, dotted against the
    /// FULL-precision FP32 <c>x</c> (no activation quantization). This is the
    /// result the MMVQ path approximates.
    /// </summary>
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

    private static void AssertParity(float[] expected, float[] actual, int m, int k)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Argmax-exact: a structurally broken kernel moves the argmax.
        int argE = 0, argA = 0;
        for (int i = 1; i < m; i++)
        {
            if (expected[i] > expected[argE]) argE = i;
            if (actual[i] > actual[argA]) argA = i;
        }
        Assert.True(argE == argA,
            $"Argmax mismatch (m={m},k={k}): oracle={argE} ({expected[argE]:G6}), " +
            $"mmvq={argA} ({actual[argA]:G6}).");

        // Output magnitude scale for the abs tolerance (RMS of the oracle).
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
            // Pass when within EITHER the relative OR the magnitude-scaled abs
            // tolerance — small-magnitude outputs have large relative error
            // from activation quant but tiny absolute error, and vice versa.
            if (diff > absTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"MMVQ drift exceeded tolerance (m={m},k={k}): errors={errors}/{m}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }
}
