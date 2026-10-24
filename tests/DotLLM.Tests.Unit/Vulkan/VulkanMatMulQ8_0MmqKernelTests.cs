using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Tolerance + argmax parity test for the dp4a MMQ prefill GEMM (issue #50):
/// <c>QuantizeQ8_1RowsKernel</c> (F32 B[N,K] → Q8_1 row-wise) followed by
/// <c>MatMulQ8_0MmqKernel</c> (integer-dot GEMM against int8 Q8_0 weights).
/// </summary>
/// <remarks>
/// <para>
/// MMQ is NOT bit-exact vs the F32-in Q8_0 GEMM: each activation row is
/// quantized to int8 (Q8_1) first, so the exact kernel-parity test used by
/// <c>VulkanMatMulQ8_0GemmKernelTests</c> cannot apply. Instead we compare
/// against the CPU F32 oracle (Q8_0-quantized weights · FP32 B) with, per
/// output row:
/// </para>
/// <list type="bullet">
///   <item><b>argmax-exact</b> — the position of the max output element in each
///     token row must match the oracle. A broken kernel (wrong unpack / scale /
///     accumulate / row indexing) shifts the argmax; int8-activation quant does
///     not.</item>
///   <item><b>loose abs/rel tolerance</b> — sized to the int8-activation-quant
///     error floor (per-32-block scale, ~1/127 relative per element, averaged
///     over K). Mirrors the MMVQ test's tolerance.</item>
/// </list>
/// <para>
/// Skipped when the device does not advertise
/// <c>VK_KHR_shader_integer_dot_product</c> — <c>TryCreate</c> returns null and
/// the model falls back to the FP GEMM.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanMatMulQ8_0MmqKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // Same int8-activation-quant tolerance as the MMVQ decode test: one Q8_1
    // element's relative drift is bounded by ~1/127; summed over K with random
    // signs the RMS relative error of a length-K dot is well under 1%. 3e-2 is
    // comfortably above that floor and far below what a broken kernel produces.
    private const float RelTol = 3e-2f;

    [SkippableTheory]
    [InlineData(2, 4, 32)]            // tiny: one block per row, partial tile (bounds)
    [InlineData(1, 1, 32)]           // single-cell output
    [InlineData(17, 33, 64)]         // non-multiple-of-tile N/M, odd row-byte align (2*34=68)
    [InlineData(8, 2048, 2048)]      // Llama-3.2-1B q/o projection (small prefill batch)
    [InlineData(16, 8192, 2048)]     // Llama-3.2-1B gate/up projection
    [InlineData(16, 2048, 8192)]     // Llama-3.2-1B down projection
    [InlineData(7, 4, 96)]           // K=96, blocksPerRow=3 (odd) — phase/stride family
    public void Mmq_MatchesF32Oracle_ArgmaxAndTolerance(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            "Device does not advertise VK_KHR_shader_integer_dot_product — MMQ path is unavailable here.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("matmul_q8_0_mmq.spv missing or unsupported.");

        var rng = new Random(0x50 + n * 13 + m * 7 + k * 3);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = RandomFloats(rng, n * k, range: 1.0f);

        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ8.Length);

        // F32 oracle: Q8_0-quantized weights · FP32 B (the F32-in GEMM result).
        float[] expected = CpuGemmQ8_0F32In(weightsQ8, inputB, m, k, n);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(inputB, bufB);

        // Quantize B → Q8_1 row-wise, then MMQ GEMM — two dispatches in one
        // submit so the barrier between them is exercised as in the forward pass.
        using (var ctx = device.CreateSubmitContext())
        {
            ctx.Begin();
            quant.Record(ctx.CommandBuffer, bufB, bufXq, bufXds, n, k);
            DotLLM.Vulkan.Kernels.KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
            mmq.Record(ctx.CommandBuffer, bufW, bufXq, bufXds, bufC, m, k, n);
            ctx.SubmitAndWait();
        }

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        AssertParity(expected, actual, m, k, n);
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
    /// FULL-precision FP32 <c>B</c> (no activation quantization). This is the
    /// result the MMQ path approximates. <c>C[t, row] = dot(W[row], B[t])</c>.
    /// </summary>
    private static unsafe float[] CpuGemmQ8_0F32In(byte[] weightsQ8, float[] b, int m, int k, int n)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        var result = new float[n * m];

        fixed (byte* wPtr = weightsQ8)
        fixed (float* bPtr = b)
        {
            for (int t = 0; t < n; t++)
            {
                float* bRow = bPtr + (long)t * k;
                for (int row = 0; row < m; row++)
                {
                    byte* rowBase = wPtr + (long)row * rowBytes;
                    float sum = 0;
                    for (int blk = 0; blk < blocksPerRow; blk++)
                    {
                        byte* block = rowBase + blk * Q8_0BlockBytes;
                        float d = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block);
                        sbyte* qs = (sbyte*)(block + 2);
                        float blockSum = 0;
                        for (int j = 0; j < Q8_0GroupSize; j++)
                            blockSum += (float)qs[j] * bRow[blk * Q8_0GroupSize + j];
                        sum += d * blockSum;
                    }
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
    }

    private static void AssertParity(float[] expected, float[] actual, int m, int k, int n)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Output magnitude scale for the abs tolerance (RMS of the oracle).
        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, expected.Length));
        float absTol = MathF.Max(rms, 1e-6f) * RelTol;

        // Per-token argmax: a structurally broken kernel moves the argmax to an
        // element the oracle ranks far lower. Activation int8-quant can, however,
        // flip the winner between two near-tied maxima (e.g. 10.013 vs 10.021 at
        // K=8192). Treat that as a pass: a mismatch is only meaningful when the
        // oracle's max is materially (> absTol) above the kernel's chosen max in
        // the ORACLE ranking — i.e. the kernel preferred a genuinely smaller
        // element, not a near-equal neighbour within quant noise.
        for (int t = 0; t < n; t++)
        {
            int argE = 0, argA = 0;
            for (int i = 1; i < m; i++)
            {
                if (expected[t * m + i] > expected[t * m + argE]) argE = i;
                if (actual[t * m + i] > actual[t * m + argA]) argA = i;
            }
            float oracleMax = expected[t * m + argE];
            float oracleAtMmqArg = expected[t * m + argA];
            Assert.True(argE == argA || (oracleMax - oracleAtMmqArg) <= absTol,
                $"Argmax mismatch (n={n},m={m},k={k}) row {t}: oracle={argE} " +
                $"({oracleMax:G6}), mmq={argA} (oracle@{argA}={oracleAtMmqArg:G6}, " +
                $"gap={oracleMax - oracleAtMmqArg:G6} > absTol={absTol:G6}).");
        }

        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            // Pass when within EITHER the relative OR the magnitude-scaled abs
            // tolerance — small-magnitude outputs have large relative error from
            // activation quant but tiny absolute error, and vice versa.
            if (diff > absTol && rel > RelTol) errors++;
        }
        Assert.True(errors == 0,
            $"MMQ drift exceeded tolerance (n={n},m={m},k={k}): errors={errors}/{expected.Length}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }
}
