using System.Diagnostics;
using System.Text;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Go/no-go ceiling probe: DP4a (INT8 integer-dot) Q8_0 prefill GEMM vs the
/// scalar (FP32-accumulate) Q8_0 GEMM, on real prefill shapes. Targets Intel
/// Arc Xe-LPG, which exposes accelerated <c>dotPacked4x8AccSatEXT</c> but NOT
/// <c>VK_KHR_cooperative_matrix</c> — so the production prefill baseline on Arc
/// is the scalar GEMM (the coopmat kernel never instantiates). Enable with
/// <c>DOTLLM_VULKAN_GEMM_DP4A_BENCH=1</c>.
/// </summary>
/// <remarks>
/// <para>
/// Methodology mirrors <see cref="VulkanQ8_0Dp4aMicroBench"/>: batched fence
/// (N iterations per submit, one <c>SubmitAndWait</c>, time / N), a
/// compute→compute barrier between iterations so the timing is true per-GEMM
/// GPU time, the two kernels interleaved within each round, and the reported
/// speedup is the <b>median of per-round paired ratios</b> (thermal/turbo drift
/// cancels). <b>Weights are device-local</b> (<c>AllocateDeviceLocal</c> +
/// <c>UploadToDeviceLocal</c>) so PCIe/host-visible latency does not bury the
/// compute signal.
/// </para>
/// <para>
/// Shapes: SmolLM-135M and Llama-3.2-1B projection shapes at seqLen ∈
/// {128, 256, 512}. M and N are kept multiples of 16 and K a multiple of 32 —
/// the probe shader handles perfect-multiple tiles only (edge handling is
/// deferred to a shipping kernel that is only built if this probe clears the
/// ~1.3× bar).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ8_0GemmDp4aProbe
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    // Iterations per submit is chosen adaptively per shape (BenchShape) to keep
    // each command buffer's serialized GPU time well under the Windows TDR
    // watchdog (~2 s) — a single fat submit at the Llama Gate/Up shape can
    // otherwise trip VK_ERROR_DEVICE_LOST.
    private const int Rounds = 7;

    // Tight bar for GPU vs the INT8-activation reference: both sides perform
    // identical integer arithmetic (same per-token-per-block activation quant,
    // same int8 dot, same d_w*d_x scaling), so they may only differ by
    // float-accumulation ORDER. The block dots stay far from int32 saturation.
    // (Comparing against a FULL-precision-activation reference instead is the
    // wrong model — that error grows √K from the inherent INT8-activation cost,
    // ~0.056 abs at K=8192 — so it is reported informational-only.)
    private const float IntRefAbsTol = 1.5e-3f;
    private const float IntRefRelTol = 1e-3f;

    private readonly ITestOutputHelper _output;

    public VulkanQ8_0GemmDp4aProbe(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Numerical parity of the DP4a GEMM vs a CPU Q8_0 reference over the exact
    /// bytes the GPU sees, plus a bug-injection guard (a deliberately wrong
    /// reference must fail the same tolerance) so the parity assertion has
    /// discriminating power.
    /// </summary>
    [SkippableTheory]
    [InlineData(16, 16, 32)]      // single tile, one K-block
    [InlineData(64, 576, 576)]    // SmolLM-135M QKV/O projection, N=64
    [InlineData(128, 2048, 2048)] // Llama-3.2-1B-shaped projection, N=128
    [InlineData(16, 512, 8192)]   // large-K (Down proj K) — int-dot saturation check
    [InlineData(16, 8192, 2048)]  // large-M (Gate/Up M) — many output-row tiles
    public void Dp4aGemm_MatchesCpuReference(int n, int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' lacks VK_KHR_shader_integer_dot_product.");

        var rng = new Random(0xD94A + n * 31 + m * 17 + k * 3);
        float[] weightsF32 = RandomFloats(rng, m * k, 0.1f);
        float[] inputB = RandomFloats(rng, n * k, 1.0f);
        byte[] weightsQ8 = QuantizeRows(weightsF32, m, k);

        // Two references:
        //   fullRef  — dequant weight × FULL-precision activation (informational:
        //              measures the inherent INT8-activation cost, which grows √K).
        //   int8Ref  — quantizes the activation to INT8 per-token-per-32-block
        //              exactly as the shader does, then integer-dots. The GPU
        //              kernel does identical integer arithmetic, so this is the
        //              DISCRIMINATING parity bar (tight tolerance).
        float[] fullRef = CpuGemmQ8_0(weightsQ8, inputB, m, k, n);
        float[] int8Ref = CpuGemmQ8_0Int8Act(weightsQ8, inputB, m, k, n);

        using var kernel = MatMulQ8_0GemmDp4aKernel.Create(device, spvDir);

        int totalBytes = weightsQ8.Length;
        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.AllocateDeviceLocal(weightsBufBytes);
        using var staging = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));

        device.UploadToDeviceLocal(new ReadOnlySpan<byte>(weightsQ8), staging, bufW);
        device.Upload(inputB, bufB);

        kernel.Launch(bufW, bufB, bufC, m, k, n);

        float[] actual = new float[n * m];
        device.Download(bufC, actual);

        // Discriminating parity: GPU vs INT8 reference (identical integer math)
        // at a TIGHT bar — must match to float-accumulation-order error.
        int errors = CountErrors(int8Ref, actual, IntRefAbsTol, IntRefRelTol, out float maxAbs, out float maxRel, out double meanAbs);
        // Informational: deviation from full-precision activation (grows √K).
        CountErrors(fullRef, actual, IntRefAbsTol, IntRefRelTol, out _, out _, out double meanAbsFull);
        _output.WriteLine(
            $"DP4a GEMM parity n={n} m={m} k={k}: vsINT8ref maxAbs={maxAbs:G4}, meanAbs={meanAbs:G4}, maxRel={maxRel:G4}, errors={errors}/{int8Ref.Length}; vsFullPrec meanAbs={meanAbsFull:G4}");
        Assert.True(errors == 0,
            $"DP4a GEMM parity vs INT8 reference failed (n={n},m={m},k={k}): errors={errors}, maxAbs={maxAbs:G6}, maxRel={maxRel:G6}");

        // Bug-injection guard: shifting many cells of the INT8 reference must NOT
        // pass — proves the tight bar catches a real layout/sign/tile bug.
        float[] corrupt = (float[])int8Ref.Clone();
        for (int i = 0; i < corrupt.Length; i += 7) corrupt[i] += 0.05f;
        int corruptErrors = CountErrors(corrupt, actual, IntRefAbsTol, IntRefRelTol, out _, out _, out _);
        Assert.True(corruptErrors > 0, "Bug-injection guard: corrupted reference unexpectedly passed tolerance.");
    }

    [SkippableFact]
    public void Bench_Dp4aGemmVsScalar()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_GEMM_DP4A_BENCH") == "1",
            "DOTLLM_VULKAN_GEMM_DP4A_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' lacks VK_KHR_shader_integer_dot_product.");

        var sb = new StringBuilder();
        sb.AppendLine("| Proj | Shape (N×M×K) | Scalar ms | DP4a ms | Scalar GFLOPS | DP4a GFLOPS | DP4a× |");
        sb.AppendLine("|---|---|---:|---:|---:|---:|---:|");

        // (label, M, K) projection shapes; N (seqLen) swept below.
        (string label, int m, int k)[] projs =
        {
            ("SmolLM-135M QKV/O", 576, 576),
            ("SmolLM-135M Gate/Up", 1536, 576),
            ("Llama-3.2-1B QKV/O", 2048, 2048),
            ("Llama-3.2-1B Gate/Up", 8192, 2048),
            ("Llama-3.2-1B Down", 2048, 8192),
        };
        int[] seqLens = { 128, 256, 512 };

        foreach (var (label, m, k) in projs)
            foreach (int n in seqLens)
                sb.AppendLine(BenchShape(device, spvDir, label, n, m, k));

        _output.WriteLine("Device: " + device.DeviceName + $"  (VendorId 0x{device.VendorId:X4})");
        _output.WriteLine($"HasCooperativeMatrix={device.HasCooperativeMatrix}, HasIntegerDotProduct={device.HasIntegerDotProduct}");
        _output.WriteLine($"Device-local weights; batched fence; iters/submit adaptive (per-row), rounds={Rounds} (median of paired per-round ratios)");
        _output.WriteLine(string.Empty);
        _output.WriteLine(sb.ToString());
    }

    /// <summary>
    /// Isolated re-measure of the two largest sweep shapes (where the batched
    /// sweep showed a DP4a regression) at iters=1 with a cooldown between rounds
    /// and raw per-round timings printed. Discriminates a thermal/contention
    /// soak artifact (intermittent spikes / upward drift) from a genuine
    /// large-shape cliff (stable DP4a slowness with clean parity).
    /// </summary>
    [SkippableFact]
    public void Bench_LargeShapesIsolated()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_GEMM_DP4A_BENCH") == "1",
            "DOTLLM_VULKAN_GEMM_DP4A_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' lacks VK_KHR_shader_integer_dot_product.");

        (string label, int n, int m, int k)[] shapes =
        {
            ("Llama Gate/Up 512×8192×2048", 512, 8192, 2048),
            ("Llama Down 512×2048×8192", 512, 2048, 8192),
        };

        foreach (var (label, n, m, k) in shapes)
        {
            var rng = new Random(0x6E + m * 7 + k * 3 + n);
            byte[] weightsQ8 = QuantizeRandomRows(rng, m, k);
            float[] x = RandomFloats(rng, n * k, 1.0f);
            int totalBytes = weightsQ8.Length;
            long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
            using var bufW = device.AllocateDeviceLocal(weightsBufBytes);
            using var staging = device.Allocate(weightsBufBytes);
            using var bufB = device.Allocate((long)n * k * sizeof(float));
            using var bufC = device.Allocate((long)n * m * sizeof(float));
            device.UploadToDeviceLocal(new ReadOnlySpan<byte>(weightsQ8), staging, bufW);
            device.Upload(x, bufB);

            using var scalar = MatMulQ8_0GemmKernel.Create(device, spvDir);
            using var dp4a = MatMulQ8_0GemmDp4aKernel.Create(device, spvDir);
            Action<nint> recScalar = cb => { scalar.Record(cb, bufW, bufB, bufC, m: m, k: k, n: n); KernelSupport.ComputeToComputeBarrier(cb); };
            Action<nint> recDp4a = cb => { dp4a.Record(cb, bufW, bufB, bufC, m: m, k: k, n: n); KernelSupport.ComputeToComputeBarrier(cb); };

            // Warm up once each, then cool down before measuring.
            TimeBatched(device, recScalar, 1);
            TimeBatched(device, recDp4a, 1);
            System.Threading.Thread.Sleep(1000);

            const int rounds = 9;
            var sMs = new double[rounds];
            var dMs = new double[rounds];
            for (int r = 0; r < rounds; r++)
            {
                sMs[r] = TimeBatched(device, recScalar, 1);
                dMs[r] = TimeBatched(device, recDp4a, 1);
                System.Threading.Thread.Sleep(500); // cooldown between rounds
            }

            _output.WriteLine($"{label}  (device-local, iters=1, {rounds} rounds, 500ms cooldown)");
            _output.WriteLine("  scalar raw ms: " + string.Join(", ", Array.ConvertAll(sMs, v => v.ToString("F1"))));
            _output.WriteLine("  dp4a   raw ms: " + string.Join(", ", Array.ConvertAll(dMs, v => v.ToString("F1"))));
            double sMin = Min(sMs), dMin = Min(dMs);
            _output.WriteLine($"  min scalar={sMin:F1} ms, min dp4a={dMin:F1} ms, best-case DP4a×={sMin / dMin:F2}x, median DP4a×={Median(sMs) / Median(dMs):F2}x");
        }
    }

    private static double Min(double[] vals)
    {
        double mn = double.MaxValue;
        foreach (var v in vals) if (v < mn) mn = v;
        return mn;
    }

    private string BenchShape(VulkanDevice device, string spvDir, string label, int n, int m, int k)
    {
        var rng = new Random(0x6E + m * 7 + k * 3 + n);
        byte[] weightsQ8 = QuantizeRandomRows(rng, m, k);
        float[] x = RandomFloats(rng, n * k, 1.0f);

        int totalBytes = weightsQ8.Length;
        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        using var bufW = device.AllocateDeviceLocal(weightsBufBytes);
        using var staging = device.Allocate(weightsBufBytes);
        using var bufB = device.Allocate((long)n * k * sizeof(float));
        using var bufC = device.Allocate((long)n * m * sizeof(float));
        device.UploadToDeviceLocal(new ReadOnlySpan<byte>(weightsQ8), staging, bufW);
        device.Upload(x, bufB);

        using var scalar = MatMulQ8_0GemmKernel.Create(device, spvDir);
        using var dp4a = MatMulQ8_0GemmDp4aKernel.Create(device, spvDir);

        Action<nint> recScalar = cb => { scalar.Record(cb, bufW, bufB, bufC, m: m, k: k, n: n); KernelSupport.ComputeToComputeBarrier(cb); };
        Action<nint> recDp4a = cb => { dp4a.Record(cb, bufW, bufB, bufC, m: m, k: k, n: n); KernelSupport.ComputeToComputeBarrier(cb); };

        // Warm up each side (pipeline ISA compile + first-submit cost) and
        // estimate single-dispatch cost from the SLOWER (scalar) side so the
        // batched submit stays under the TDR watchdog.
        double warmScalar = TimeBatched(device, recScalar, 2);
        double warmDp4a = TimeBatched(device, recDp4a, 2);
        double slowestMs = Math.Max(warmScalar, warmDp4a);
        // Target ~250 ms of serialized GPU work per submit; clamp to [4, 40].
        int iters = (int)Math.Clamp(250.0 / Math.Max(slowestMs, 0.01), 4, 40);

        var ratios = new double[Rounds];
        var scalarMs = new double[Rounds];
        var dp4aMs = new double[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            scalarMs[r] = TimeBatched(device, recScalar, iters);
            dp4aMs[r] = TimeBatched(device, recDp4a, iters);
            ratios[r] = scalarMs[r] / dp4aMs[r];
        }
        double sMed = Median(scalarMs);
        double dMed = Median(dp4aMs);
        double flops = 2.0 * (double)m * n * k;
        double sGf = (flops / (sMed * 1e-3)) / 1e9;
        double dGf = (flops / (dMed * 1e-3)) / 1e9;

        return $"| {label} | {n}×{m}×{k} | {sMed:F4} | {dMed:F4} | {sGf:F1} | {dGf:F1} | {Median(ratios):F2}x |  (iters={iters})";
    }

    private static double TimeBatched(VulkanDevice device, Action<nint> recordOne, int n)
    {
        using var ctx = device.CreateSubmitContext();
        ctx.Begin();
        for (int i = 0; i < n; i++) recordOne(ctx.CommandBuffer);
        var sw = Stopwatch.StartNew();
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds / n;
    }

    private static double Median(double[] vals)
    {
        var s = (double[])vals.Clone();
        Array.Sort(s);
        int n = s.Length;
        return n % 2 == 1 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
    }

    private static int CountErrors(float[] expected, float[] actual, float absTol, float relTol, out float maxAbs, out float maxRel, out double meanAbs)
    {
        int errors = 0;
        maxAbs = 0; maxRel = 0;
        double sumAbs = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float diff = MathF.Abs(expected[i] - actual[i]);
            float rel = diff / MathF.Max(MathF.Abs(expected[i]), 1e-7f);
            sumAbs += diff;
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > relTol) errors++;
        }
        meanAbs = sumAbs / expected.Length;
        return errors;
    }

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
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

    private static unsafe byte[] QuantizeRandomRows(Random rng, int m, int k)
    {
        int blocksPerRow = k / Q8_0GroupSize;
        var dst = new byte[m * blocksPerRow * Q8_0BlockBytes];
        float[] rowF = new float[k];
        fixed (byte* d = dst)
        fixed (float* rp = rowF)
        {
            for (int row = 0; row < m; row++)
            {
                for (int i = 0; i < k; i++) rowF[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * 0.1);
                MatMul.QuantizeF32ToQ8_0(rp, d + (long)row * blocksPerRow * Q8_0BlockBytes, k);
            }
        }
        return dst;
    }

    // CPU reference that quantizes the activation to INT8 EXACTLY as the shader
    // does (per-token, per-32-block symmetric amax/127, round-to-nearest, clamp
    // [-127,127]), integer-dots against the Q8_0 weight int8, and scales each
    // block by d_w * d_x. This is the discriminating bar for the GPU kernel.
    private static unsafe float[] CpuGemmQ8_0Int8Act(byte[] weightsQ8, float[] b, int m, int k, int n)
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
                        float dW = (float)System.Runtime.CompilerServices.Unsafe.ReadUnaligned<Half>(block);
                        sbyte* qw = (sbyte*)(block + 2);

                        // Activation block: amax -> scale, quantize, int dot.
                        float* xBlk = bRow + blk * Q8_0GroupSize;
                        float amax = 0;
                        for (int j = 0; j < Q8_0GroupSize; j++) amax = MathF.Max(amax, MathF.Abs(xBlk[j]));
                        float dX = amax / 127.0f;
                        float invX = dX > 0 ? 127.0f / amax : 0.0f;
                        int dot = 0;
                        for (int j = 0; j < Q8_0GroupSize; j++)
                        {
                            int qx = (int)MathF.Round(xBlk[j] * invX);
                            qx = Math.Clamp(qx, -127, 127);
                            dot += (int)qw[j] * qx;
                        }
                        sum += dW * dX * dot;
                    }
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
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
}
