using System.Diagnostics;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark + A/B for the Vulkan I2_S (BitNet ternary) decode GEMV on the
/// Meteor-Lake Arc iGPU. Times the production kernel (<c>matmul_i2_s_f32_gemv.spv</c>) and, when
/// present, an alternate variant (<c>matmul_i2_s_v0.spv</c>) side by side in one process at BitNet
/// b1.58 2B4T projection shapes (o_proj / gate-up / down).
/// </summary>
/// <remarks>
/// Methodology (learned from the MMVQ bench on this throttling display-Arc):
/// <list type="bullet">
/// <item><b>Compute-bound timed region:</b> each pass submits <c>batch</c> GEMVs behind one fence
/// (default 64; env <c>DOTLLM_I2S_BENCH_BATCH</c>) so the ~ms submit/fence latency amortizes — the
/// reported weight-GB/s should approach the device bandwidth when compute-bound, not the ~1 GB/s a
/// launch-bound bench shows.</item>
/// <item><b>Interleaved paired A/B:</b> per pass the two variants are timed back-to-back (order
/// alternated across passes) and the per-pass RATIO is medianed — cancels the iGPU clock/thermal
/// drift that makes measure-all-A-then-all-B unreliable.</item>
/// <item><b>P-core pinned:</b> the host is pinned to P-cores (mask <c>DOTLLM_BENCH_AFFINITY</c>,
/// default 0x3F = the 155H's 6 P-cores) so submit/sync isn't scheduled on the slower E/LP-E cores.</item>
/// </list>
/// Enable with <c>DOTLLM_I2S_GEMV_BENCH=1</c>; Arc: <c>DOTLLM_VULKAN_DEVICE_VENDOR=0x8086</c>. The
/// Arc's 2s TDR can reset the device under sustained compute — run under a shell retry+cooldown loop.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanI2SGemvBench
{
    private const string VariantSpv = "matmul_i2_s_v0.spv";   // optional A/B comparand
    private const int WarmupPasses = 2;
    private const int Passes = 9;

    private readonly ITestOutputHelper _output;
    public VulkanI2SGemvBench(ITestOutputHelper output) => _output = output;

    private static readonly (string Tag, int M, int K)[] Shapes =
    [
        ("o_proj   (2560x2560)", 2560, 2560),
        ("gate/up  (6912x2560)", 6912, 2560),
        ("down     (2560x6912)", 2560, 6912),
    ];

    [SkippableFact]
    public void Bench_I2SGemv()
    {
        Skip.IfNot(Environment.GetEnvironmentVariable("DOTLLM_I2S_GEMV_BENCH") == "1",
            "DOTLLM_I2S_GEMV_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        PinToPCores();
        int batch = EnvInt("DOTLLM_I2S_BENCH_BATCH", 64);

        using var device = VulkanDevice.Create();
        using var current = MatMulI2SGemvF32Kernel.Create(device, spvDir);

        // Optional A/B comparand — present only when a benchmark places it in the spv dir.
        MatMulI2SGemvF32Kernel? variant = null;
        if (File.Exists(Path.Combine(spvDir, VariantSpv)))
            variant = MatMulI2SGemvF32Kernel.Create(device, spvDir, VariantSpv);

        try
        {
            _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}");
            _output.WriteLine($"Affinity: 0x{Environment.GetEnvironmentVariable("DOTLLM_BENCH_AFFINITY") ?? "3F"}  "
                + $"batch={batch}  schedule: {WarmupPasses} warmup + {Passes} interleaved paired passes (median)");
            _output.WriteLine(variant is null
                ? "| shape | µs/GEMV (current) | weight GB/s |"
                : "| shape | current µs | variant(v0) µs | current/v0 | current GB/s |");
            _output.WriteLine("|---|---:|---:|---:|---:|");

            var rng = new Random(0x12_5C);
            foreach (var (tag, m, k) in Shapes)
            {
                long rowBytes = (long)k / 4;
                long wBytes = m * rowBytes + sizeof(float);
                using var bufW = device.Allocate(wBytes);
                using var bufX = device.Allocate((long)k * sizeof(float));
                using var bufY = device.Allocate((long)m * sizeof(float));

                byte[] w = new byte[wBytes];
                rng.NextBytes(w);                  // random packed codes; timing is data-independent
                float[] x = new float[k];
                for (int i = 0; i < k; i++) x[i] = rng.NextSingle() * 2f - 1f;
                device.Upload(new ReadOnlySpan<byte>(w), bufW);
                device.Upload(x, bufX);

                double curUs, varUs, ratio;
                if (variant is null)
                {
                    curUs = MeasureMedian(device, current, bufW, bufX, bufY, m, k, batch);
                    double gbps = (m * rowBytes) / (curUs * 1e-6) / 1e9;
                    _output.WriteLine($"| {tag} | {curUs:F2} | {gbps:F1} |");
                }
                else
                {
                    (curUs, varUs, ratio) = MeasurePaired(device, current, variant, bufW, bufX, bufY, m, k, batch);
                    double gbps = (m * rowBytes) / (curUs * 1e-6) / 1e9;
                    _output.WriteLine($"| {tag} | {curUs:F2} | {varUs:F2} | {ratio:F2}x | {gbps:F1} |");
                }
            }
        }
        finally
        {
            variant?.Dispose();
        }
    }

    private static (double cur, double var, double ratio) MeasurePaired(
        VulkanDevice device, MatMulI2SGemvF32Kernel cur, MatMulI2SGemvF32Kernel var,
        VulkanDevice.Buffer w, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int m, int k, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, cur, w, x, y, m, k, batch);
            RunPass(device, var, w, x, y, m, k, batch);
        }
        var curUs = new double[Passes];
        var varUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tc, tv;
            if ((p & 1) == 0)   // alternate order so first-vs-second bias cancels
            {
                tc = RunPass(device, cur, w, x, y, m, k, batch);
                tv = RunPass(device, var, w, x, y, m, k, batch);
            }
            else
            {
                tv = RunPass(device, var, w, x, y, m, k, batch);
                tc = RunPass(device, cur, w, x, y, m, k, batch);
            }
            curUs[p] = tc; varUs[p] = tv; ratios[p] = tv > 0 ? tc / tv : 0;
        }
        Array.Sort(curUs); Array.Sort(varUs); Array.Sort(ratios);
        return (curUs[Passes / 2], varUs[Passes / 2], ratios[Passes / 2]);
    }

    private static double MeasureMedian(
        VulkanDevice device, MatMulI2SGemvF32Kernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int m, int k, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++) RunPass(device, kernel, w, x, y, m, k, batch);
        var us = new double[Passes];
        for (int p = 0; p < Passes; p++) us[p] = RunPass(device, kernel, w, x, y, m, k, batch);
        Array.Sort(us);
        return us[Passes / 2];
    }

    private static double RunPass(
        VulkanDevice device, MatMulI2SGemvF32Kernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer x, VulkanDevice.Buffer y, int m, int k, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            kernel.Record(ctx.CommandBuffer, w, x, y, m, k);   // no barriers → dispatches overlap (compute-bound)
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;

    private static void PinToPCores()
    {
        string mask = Environment.GetEnvironmentVariable("DOTLLM_BENCH_AFFINITY") ?? "3F";
        try
        {
            nint affinity = (nint)Convert.ToInt64(mask, 16);
            Process.GetCurrentProcess().ProcessorAffinity = affinity;
        }
        catch { /* affinity is best-effort */ }
    }
}
