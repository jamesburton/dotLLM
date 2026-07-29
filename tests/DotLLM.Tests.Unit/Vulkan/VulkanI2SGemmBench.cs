using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark + A/B for the Vulkan I2_S (BitNet ternary) prefill GEMM on the
/// Meteor-Lake Arc iGPU. Times the baseline scalar-dot kernel
/// (<c>matmul_i2_s_f32_gemm.spv</c>, 16x16 tile / one thread per output cell) against the
/// register-blocked variant (<c>matmul_i2_s_f32_gemm_rb.spv</c>, 32x32 tile / one 2x2
/// micro-tile per thread) side by side in one process at BitNet b1.58 2B4T projection shapes.
/// </summary>
/// <remarks>
/// <para>
/// The hypothesis under test: the baseline does 128 MACs per K-chunk against 256 shared-memory
/// loads (0.5 MAC/load) and is shared-memory-bound rather than ALU-bound. The 2x2 micro-tile
/// reuses each staged value twice for 1.0 MAC/load. Whether that converts to wall-clock on
/// Xe-LPG is exactly what this measures — the <c>uint</c>+vec4 GEMV variant was also
/// theoretically better and measured 1.12-1.34x SLOWER, so nothing ships unmeasured.
/// </para>
/// <para>
/// Methodology is deliberately identical to <see cref="VulkanI2SGemvBench"/>, because on this
/// throttling display-Arc it is the methodology that makes results trustworthy:
/// </para>
/// <list type="bullet">
/// <item><b>Compute-bound timed region:</b> each pass submits <c>batch</c> GEMMs behind one fence
/// (default 8; env <c>DOTLLM_I2S_GEMM_BENCH_BATCH</c>) so submit/fence latency amortizes.</item>
/// <item><b>Interleaved paired A/B:</b> per pass both variants are timed back-to-back with the
/// order alternated, and the per-pass RATIO is medianed — this cancels the clock/thermal drift
/// that makes measure-all-A-then-all-B unreliable here.</item>
/// <item><b>P-core pinned:</b> host pinned via <c>DOTLLM_BENCH_AFFINITY</c> (default 0x3F = the
/// 155H's 6 P-cores) so submit/sync avoids the E/LP-E cores.</item>
/// </list>
/// <para>
/// Enable with <c>DOTLLM_I2S_GEMM_BENCH=1</c>; Arc: <c>DOTLLM_VULKAN_DEVICE_VENDOR=0x8086</c>.
/// Token count is <c>DOTLLM_I2S_GEMM_BENCH_TOKENS</c> (default 64). The Arc's 2s TDR can reset
/// the device under sustained compute — run under a shell retry+cooldown loop.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanI2SGemmBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;

    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the benchmark with the xUnit output sink.</summary>
    /// <param name="output">Sink for the result table.</param>
    public VulkanI2SGemmBench(ITestOutputHelper output) => _output = output;

    private static readonly (string Tag, int M, int K)[] Shapes =
    [
        ("o_proj   (2560x2560)", 2560, 2560),
        ("gate/up  (6912x2560)", 6912, 2560),
        ("down     (2560x6912)", 2560, 6912),
    ];

    /// <summary>Times the baseline and register-blocked I2_S GEMM variants, interleaved and paired.</summary>
    [SkippableFact]
    public void Bench_I2SGemm()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_I2S_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_I2S_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        PinToPCores();
        int batch = EnvInt("DOTLLM_I2S_GEMM_BENCH_BATCH", 8);
        int tokens = EnvInt("DOTLLM_I2S_GEMM_BENCH_TOKENS", 64);

        using var device = VulkanDevice.Create();
        using var baseline = MatMulI2SGemmF32Kernel.Create(device, spvDir, I2SGemmVariant.Scalar);
        using var blocked = MatMulI2SGemmF32Kernel.Create(device, spvDir, I2SGemmVariant.RegisterBlocked);

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine($"Affinity: 0x{Environment.GetEnvironmentVariable("DOTLLM_BENCH_AFFINITY") ?? "3F"}  "
            + $"tokens={tokens}  batch={batch}  schedule: {WarmupPasses} warmup + {Passes} interleaved paired passes (median)");
        _output.WriteLine("| shape | baseline µs | reg-blocked µs | speedup | baseline GFLOP/s | rb GFLOP/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|");

        var rng = new Random(0x12_5C);
        foreach (var (tag, m, k) in Shapes)
        {
            long rowBytes = (long)k / 4;
            long wBytes = m * rowBytes + sizeof(float);
            using var bufW = device.Allocate(wBytes);
            using var bufB = device.Allocate((long)tokens * k * sizeof(float));
            using var bufC = device.Allocate((long)tokens * m * sizeof(float));

            byte[] w = new byte[wBytes];
            rng.NextBytes(w);                  // random packed codes; timing is data-independent
            float[] b = new float[(long)tokens * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
            device.Upload(new ReadOnlySpan<byte>(w), bufW);
            device.Upload(b, bufB);

            (double baseUs, double rbUs, double ratio) =
                MeasurePaired(device, baseline, blocked, bufW, bufB, bufC, m, k, tokens, batch);

            // 2 flops per MAC, M*K*tokens MACs.
            double flop = 2.0 * m * (double)k * tokens;
            double baseGflops = flop / (baseUs * 1e-6) / 1e9;
            double rbGflops = flop / (rbUs * 1e-6) / 1e9;
            _output.WriteLine($"| {tag} | {baseUs:F2} | {rbUs:F2} | {ratio:F2}x | {baseGflops:F1} | {rbGflops:F1} |");
        }
    }

    private static (double baseline, double blocked, double ratio) MeasurePaired(
        VulkanDevice device, MatMulI2SGemmF32Kernel baseline, MatMulI2SGemmF32Kernel blocked,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, baseline, w, b, c, m, k, n, batch);
            RunPass(device, blocked, w, b, c, m, k, n, batch);
        }

        var baseUs = new double[Passes];
        var rbUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tb, tr;
            if ((p & 1) == 0)   // alternate order so first-vs-second bias cancels
            {
                tb = RunPass(device, baseline, w, b, c, m, k, n, batch);
                tr = RunPass(device, blocked, w, b, c, m, k, n, batch);
            }
            else
            {
                tr = RunPass(device, blocked, w, b, c, m, k, n, batch);
                tb = RunPass(device, baseline, w, b, c, m, k, n, batch);
            }

            baseUs[p] = tb;
            rbUs[p] = tr;
            ratios[p] = tr > 0 ? tb / tr : 0;   // >1 means the register-blocked variant is faster
        }

        Array.Sort(baseUs); Array.Sort(rbUs); Array.Sort(ratios);
        return (baseUs[Passes / 2], rbUs[Passes / 2], ratios[Passes / 2]);
    }

    private static double RunPass(
        VulkanDevice device, MatMulI2SGemmF32Kernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            kernel.Record(ctx.CommandBuffer, w, b, c, m, k, n);   // no barriers → dispatches overlap
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;

    private static void PinToPCores()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return; // ProcessorAffinity is only supported on Windows/Linux; affinity is best-effort.

        string mask = Environment.GetEnvironmentVariable("DOTLLM_BENCH_AFFINITY") ?? "3F";
        try
        {
            nint affinity = (nint)Convert.ToInt64(mask, 16);
            Process.GetCurrentProcess().ProcessorAffinity = affinity;
        }
        catch { /* affinity is best-effort */ }
    }
}
