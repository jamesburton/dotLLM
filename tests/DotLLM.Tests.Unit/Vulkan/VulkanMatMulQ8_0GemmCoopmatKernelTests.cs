using System.Diagnostics;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Numerical-parity + perf test for the coopmat Q8_0 batched GEMM
/// (<see cref="MatMulQ8_0GemmCoopmatKernel"/>). Mirrors
/// <see cref="VulkanMatMulQ8_0GemmKernelTests"/> one-for-one on shapes and
/// tolerance so regressions are caught at the same signal level as the
/// scalar path.
/// </summary>
/// <remarks>
/// <para>
/// Validation strategy is byte-identical to the scalar test: quantize random
/// F32 weights to Q8_0 via <c>DotLLM.Cpu.Kernels.MatMul.QuantizeF32ToQ8_0</c>
/// so both CPU reference and Vulkan kernel see the <em>same</em> Q8_0 bytes,
/// then compare against a scalar dequant-and-multiply reference to catch
/// tile-layout / staging / sign-extension bugs.
/// </para>
/// <para>
/// Shapes (from the scalar GEMM's test set): 2×4×32 sanity, 1×1×32 degenerate,
/// 17×33×64 non-tile-multiple, SmolLM-135M QKV/O (64×576×576), SmolLM-135M
/// Gate/Up (64×1536×576), Llama-3-8B proj (64×4096×4096 — the perf smoke
/// shape, where we record GFLOPS).
/// Tolerance: abs 5e-3 / rel 5e-3 (coopmat-appropriate — see <c>AbsTol</c>
/// field remark for the F16-operand precision analysis).
/// </para>
/// <para>
/// Skip semantics: the whole test class skips when
/// <see cref="VulkanDevice.HasCooperativeMatrix"/> is <c>false</c> on the
/// host. This lets CI on coopmat-less hardware (software Vulkan, older
/// Intel iGPU) pass without false-failing; correctness is exercised via
/// the scalar GEMM test on every host regardless.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public class VulkanMatMulQ8_0GemmCoopmatKernelTests
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;

    // Coopmat-specific tolerances. The scalar Q8_0 GEMM's mandate
    // (abs 1e-4 / rel 1e-3) is physically unreachable for a KHR_coopmat path
    // that uses F16 operands: the F32→F16 cast of B costs ~2^-11 × |B|
    // absolute precision per element, which propagates as a √K random walk
    // to ≈√K × |A_q| × d × ε_F16 absolute error at the output. For
    // Llama-3-8B-shaped K=4096 this sits around 1–2 × 10^-3 absolute on a
    // unit-scale input — well above the scalar bar but ~3 × 10^-4 in the
    // typical case (meanAbs). Looser bounds here (abs 5e-3, rel 5e-3)
    // preserve real regression-detection power while acknowledging the
    // F16-input precision floor of the VK_KHR_cooperative_matrix F16×F16→F32
    // tile on RDNA3.5 / similar hardware. The scalar GEMM test
    // (VulkanMatMulQ8_0GemmKernelTests) enforces the tighter bar on every
    // shape, so CPU-parity coverage is never loosened — only the coopmat
    // fast path's own parity signal.
    private const float AbsTol = 5e-3f;
    private const float RelTol = 5e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanMatMulQ8_0GemmCoopmatKernelTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [SkippableTheory]
    [InlineData(2, 4, 32)]           // tiny sanity: one block per row
    [InlineData(1, 1, 32)]           // single-cell output (bounds check)
    [InlineData(17, 33, 64)]         // non-multiple-of-tile sizes, odd row alignment
    [InlineData(64, 576, 576)]       // SmolLM-135M QKV/O projection (prefill batch)
    [InlineData(64, 1536, 576)]      // SmolLM-135M Gate/Up projection (prefill batch)
    [InlineData(64, 4096, 4096)]     // Llama-3-8B projection (prefill batch — perf smoke)
    public void Launch_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(
            device.HasCooperativeMatrix,
            $"VK_KHR_cooperative_matrix not supported on {device.DeviceName}.");

        var rng = new Random(0xBEEF + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = RandomFloats(rng, n * k, range: 1.0f);

        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);
        Assert.Equal(totalBytes, weightsQ8.Length);

        // CPU reference over the exact Q8_0 bytes the GPU sees.
        float[] expected = CpuGemmQ8_0(weightsQ8, inputB, m, k, n);

        using var kernel = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(inputB, bufB);

        var sw = Stopwatch.StartNew();
        kernel.Launch(bufW, bufB, bufC, m, k, n);
        sw.Stop();

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        AssertClose(expected, actual, m, k, n, sw.Elapsed);
    }

    /// <summary>
    /// Timed benchmark at the Llama-3-8B projection shape (64×4096×4096).
    /// Records <c>ITERS</c> back-to-back dispatches into a *single* command
    /// buffer with a single fence wait at the end — this is the only way to
    /// get a meaningful kernel-only timing, because <see cref="MatMulQ8_0GemmCoopmatKernel.Launch"/>
    /// allocates a fresh command buffer and waits on <c>vkQueueWaitIdle</c>
    /// per call (~300 µs overhead on AMD that obscures a 1 ms kernel).
    /// Skipped on hosts without coopmat support.
    /// </summary>
    [SkippableFact]
    public void Bench_LlamaProj_64x4096x4096()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(
            device.HasCooperativeMatrix,
            $"VK_KHR_cooperative_matrix not supported on {device.DeviceName}.");

        const int n = 64, m = 4096, k = 4096;
        int blocksPerRow = k / Q8_0GroupSize;
        int rowBytes = blocksPerRow * Q8_0BlockBytes;
        int totalBytes = m * rowBytes;

        var rng = new Random(0xBEEF + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = RandomFloats(rng, m * k, range: 0.1f);
        float[] inputB = RandomFloats(rng, n * k, range: 1.0f);
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);

        using var kernel = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir);

        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.Upload(new ReadOnlySpan<byte>(weightsQ8), bufW);
        device.Upload(inputB, bufB);

        // Warmup — first few dispatches include pipeline compilation.
        for (int w = 0; w < 3; w++)
            kernel.Launch(bufW, bufB, bufC, m, k, n);

        // Batched bench: ITERS dispatches per submit, REPS submits, take the
        // best of the per-submit GFLOPS numbers. The best-of-REPS smooths
        // over background activity and thermal jitter on iGPUs (the AMD
        // 8060S runs this workload in a few ms and readings drift ±30% run
        // to run on a shared-power laptop).
        const int iters = 50;
        const int reps  = 10;
        double bestGflops = 0.0;
        double bestMsPerDispatch = 0.0;
        for (int r = 0; r < reps; r++)
        {
            using var ctx = device.CreateSubmitContext();
            ctx.Begin();
            for (int i = 0; i < iters; i++)
                kernel.Record(ctx.CommandBuffer, bufW, bufB, bufC, m, k, n);
            var sw = Stopwatch.StartNew();
            ctx.SubmitAndWait();
            sw.Stop();

            double msPerDispatch = sw.Elapsed.TotalMilliseconds / iters;
            // 2 flops per multiply-accumulate, M*N*K output cells.
            double flops = 2.0 * (double)m * n * k;
            double gflops = (flops / (msPerDispatch * 1e-3)) / 1e9;
            if (gflops > bestGflops)
            {
                bestGflops = gflops;
                bestMsPerDispatch = msPerDispatch;
            }
        }
        _output.WriteLine(
            $"coopmat Q8_0 GEMM 64x4096x4096: best {bestMsPerDispatch:F3} ms/dispatch, {bestGflops:F1} GFLOPS " +
            $"(best of {reps} × {iters}-dispatch single-cmdbuf batches after 3 warmup)");
    }

    // ─────────────────────────────────────────────────────────────
    // Helpers — duplicated from the scalar test for independence.
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

    private static unsafe float[] CpuGemmQ8_0(byte[] weightsQ8, float[] b, int m, int k, int n)
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

    private void AssertClose(float[] expected, float[] actual, int m, int k, int n, TimeSpan elapsed)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        double sumAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > AbsTol && rel > RelTol) errors++;
        }
        double meanAbs = sumAbs / expected.Length;
        _output.WriteLine(
            $"coopmat Q8_0 GEMM n={n} m={m} k={k}: elapsed={elapsed.TotalMilliseconds:F2} ms, " +
            $"maxAbs={maxAbs:G6}, meanAbs={meanAbs:G6}, maxRel={maxRel:G6}");
        Assert.True(errors == 0,
            $"Numerical drift exceeded tolerance (n={n},m={m},k={k}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
