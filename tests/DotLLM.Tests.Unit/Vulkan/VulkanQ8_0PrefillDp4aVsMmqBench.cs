using System.Diagnostics;
using System.Text;
using DotLLM.Cpu.Kernels;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Decisive head-to-head: the issue #69 probe kernel
/// (<see cref="MatMulQ8_0GemmDp4aKernel"/>, in-shader per-K-block activation
/// requant) vs the issue #50 production MMQ prefill path already on <c>dev</c>
/// (<see cref="QuantizeQ8_1RowsKernel"/> + <see cref="MatMulQ8_0MmqKernel"/>),
/// both DP4a INT8 Q8_0 prefill GEMMs.
/// </summary>
/// <remarks>
/// <para>
/// The #69 task brief assumed the Arc Q8_0-prefill baseline was the
/// <em>scalar</em> GEMM, because coopmat is absent on Arc. But #50 (merged into
/// <c>dev</c> the day before the probe was authored) already wires a DP4a MMQ
/// prefill GEMM ahead of the coopmat/scalar fallbacks, gated on
/// <see cref="VulkanDevice.HasIntegerDotProduct"/> — so on Arc the real
/// production prefill baseline is #50's MMQ path, NOT scalar. The only number
/// that decides whether #69 should ship is probe-vs-#50.
/// </para>
/// <para>
/// Fairness: the #50 side is timed as the full per-prefill cost
/// (<c>quantize_q8_1_rows</c> + <c>matmul_q8_0_mmq</c> + the compute→compute
/// barrier between them), exactly as <c>RecordMatmul</c> dispatches it. The
/// probe side is the single in-shader-requant GEMM. Methodology mirrors
/// <see cref="VulkanQ8_0GemmDp4aProbe"/>: device-local weights, batched fence,
/// compute→compute barrier between iterations, interleaved, median of paired
/// per-round ratios. Enable with <c>DOTLLM_VULKAN_GEMM_DP4A_BENCH=1</c>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public class VulkanQ8_0PrefillDp4aVsMmqBench
{
    private const int Q8_0BlockBytes = 34;
    private const int Q8_0GroupSize = 32;
    private const int Rounds = 7;

    private readonly ITestOutputHelper _output;

    public VulkanQ8_0PrefillDp4aVsMmqBench(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void Bench_ProbeDp4aVsMmqPrefillPath()
    {
        Skip.IfNot(
            Environment.GetEnvironmentVariable("DOTLLM_VULKAN_GEMM_DP4A_BENCH") == "1",
            "DOTLLM_VULKAN_GEMM_DP4A_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct,
            $"Device '{device.DeviceName}' lacks VK_KHR_shader_integer_dot_product.");

        var sb = new StringBuilder();
        sb.AppendLine("| Proj | Shape (N×M×K) | MMQ#50 ms | Probe#69 ms | MMQ GFLOPS | Probe GFLOPS | Probe/MMQ× |");
        sb.AppendLine("|---|---|---:|---:|---:|---:|---:|");

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
        _output.WriteLine($"Device-local weights; batched fence; iters/submit adaptive; rounds={Rounds} (median of paired per-round ratios)");
        _output.WriteLine("MMQ#50 side = quantize_q8_1_rows + matmul_q8_0_mmq (full RecordMatmul prefill cost). Probe#69 side = matmul_q8_0_gemm_dp4a (in-shader requant).");
        _output.WriteLine("Probe/MMQ× > 1 means the probe is FASTER than the production #50 path.");
        _output.WriteLine(string.Empty);
        _output.WriteLine(sb.ToString());
    }

    private string BenchShape(VulkanDevice device, string spvDir, string label, int n, int m, int k)
    {
        var rng = new Random(0x6E + m * 7 + k * 3 + n);
        byte[] weightsQ8 = QuantizeRandomRows(rng, m, k);
        float[] x = RandomFloats(rng, n * k, 1.0f);

        int totalBytes = weightsQ8.Length;
        long weightsBufBytes = ((long)totalBytes + 3) & ~3L;
        long bBytes = (long)n * k * sizeof(float);
        // ALL operands device-local — host-visible activation memory would bury
        // the compute signal asymmetrically: the probe re-reads the FP32
        // activation tile once per M-tile (activation-read-bound), while #50
        // reads compact int8 after a single quantize pass, so a host-visible B
        // penalizes the probe far more. Weights + B are uploaded via staging;
        // the Q8_1 scratch and C output are GPU-written-only (never downloaded
        // here), so device-local is both faithful and free.
        using var bufW = device.AllocateDeviceLocal(weightsBufBytes);
        using var stagingW = device.Allocate(weightsBufBytes);
        using var bufB = device.AllocateDeviceLocal(bBytes);
        using var stagingB = device.Allocate(bBytes);
        using var bufC = device.AllocateDeviceLocal((long)n * m * sizeof(float));
        // #50 activation scratch (Q8_1 rows): packed int8 + per-block (scale,sum).
        using var bufXq = device.AllocateDeviceLocal(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
        using var bufXds = device.AllocateDeviceLocal(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
        device.UploadToDeviceLocal(new ReadOnlySpan<byte>(weightsQ8), stagingW, bufW);
        var xBytes = new byte[bBytes];
        Buffer.BlockCopy(x, 0, xBytes, 0, (int)bBytes);
        device.UploadToDeviceLocal(new ReadOnlySpan<byte>(xBytes), stagingB, bufB);

        using var probe = MatMulQ8_0GemmDp4aKernel.Create(device, spvDir);
        using var quantRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new InvalidOperationException("quantize_q8_1_rows kernel unavailable.");
        using var mmq = MatMulQ8_0MmqKernel.TryCreate(device, spvDir)
            ?? throw new InvalidOperationException("matmul_q8_0_mmq kernel unavailable.");

        // #50 prefill path: quantize the activation rows, barrier, then MMQ GEMM.
        Action<nint> recMmq = cb =>
        {
            quantRows.Record(cb, bufB, bufXq, bufXds, n: n, k: k);
            KernelSupport.ComputeToComputeBarrier(cb);
            mmq.Record(cb, bufW, bufXq, bufXds, bufC, m: m, k: k, n: n);
            KernelSupport.ComputeToComputeBarrier(cb);
        };
        // #69 probe path: single in-shader-requant GEMM.
        Action<nint> recProbe = cb =>
        {
            probe.Record(cb, bufW, bufB, bufC, m: m, k: k, n: n);
            KernelSupport.ComputeToComputeBarrier(cb);
        };

        double warmMmq = TimeBatched(device, recMmq, 2);
        double warmProbe = TimeBatched(device, recProbe, 2);
        double slowestMs = Math.Max(warmMmq, warmProbe);
        int iters = (int)Math.Clamp(250.0 / Math.Max(slowestMs, 0.01), 4, 40);

        var ratios = new double[Rounds];
        var mmqMs = new double[Rounds];
        var probeMs = new double[Rounds];
        for (int r = 0; r < Rounds; r++)
        {
            mmqMs[r] = TimeBatched(device, recMmq, iters);
            probeMs[r] = TimeBatched(device, recProbe, iters);
            ratios[r] = mmqMs[r] / probeMs[r]; // >1 => probe faster
        }
        double mMed = Median(mmqMs);
        double pMed = Median(probeMs);
        double flops = 2.0 * (double)m * n * k;
        double mGf = (flops / (mMed * 1e-3)) / 1e9;
        double pGf = (flops / (pMed * 1e-3)) / 1e9;

        return $"| {label} | {n}×{m}×{k} | {mMed:F4} | {pMed:F4} | {mGf:F1} | {pGf:F1} | {Median(ratios):F2}x |  (iters={iters})";
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

    private static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++) arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
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
}
