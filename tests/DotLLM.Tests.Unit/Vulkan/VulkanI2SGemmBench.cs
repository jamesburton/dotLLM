using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark + A/B for the Vulkan I2_S (BitNet ternary) prefill GEMM variants,
/// timed side by side in one process at BitNet b1.58 2B4T projection shapes. Runs
/// <c>scalar -> register-blocked</c> always, and <c>register-blocked -> coopmat</c> on devices
/// advertising <c>VK_KHR_cooperative_matrix</c>.
/// </summary>
/// <remarks>
/// <para>
/// The original hypothesis: the scalar kernel does 128 MACs per K-chunk against 256 shared-memory
/// loads (0.5 MAC/load) and is shared-memory-bound rather than ALU-bound. The 2x2 micro-tile
/// reuses each staged value twice for 1.0 MAC/load. That held — 1.63-1.67x on Xe-LPG — so
/// register-blocked is production, and it is therefore the bar every later challenger is paired
/// against. Comparing a challenger against the scalar kernel would flatter it against a baseline
/// no longer shipped. Nothing ships unmeasured: the <c>uint</c>+vec4 GEMV variant was also
/// theoretically better and measured 1.12-1.34x SLOWER.
/// </para>
/// <para>
/// Each comparison is its own properly-paired interleaved A/B; three variants in one pass would
/// break the pairing. Methodology is deliberately identical to <see cref="VulkanI2SGemvBench"/>,
/// because on the throttling display-Arc it is the methodology that makes results trustworthy:
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

    /// <summary>Times each I2_S GEMM variant pair, interleaved and paired.</summary>
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

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  "
            + $"Coopmat: {device.HasCooperativeMatrix}");
        _output.WriteLine($"Affinity: 0x{Environment.GetEnvironmentVariable("DOTLLM_BENCH_AFFINITY") ?? "3F"}  "
            + $"tokens={tokens}  batch={batch}  schedule: {WarmupPasses} warmup + {Passes} interleaved paired passes (median)");

        // Each comparison is its own properly-paired interleaved A/B. Running three
        // variants in one pass would break the pairing, so we do reference-vs-challenger
        // pairs instead. RegisterBlocked is the production kernel and therefore the bar
        // any challenger has to clear — comparing a challenger against Scalar would
        // flatter it against a baseline we no longer ship.
        var comparisons = new List<(string Label, I2SGemmVariant Reference, I2SGemmVariant Challenger)>
        {
            ("scalar -> register-blocked", I2SGemmVariant.Scalar, I2SGemmVariant.RegisterBlocked),
            // Attacks the unpack rather than the multiply — the bottleneck the coopmat
            // attempts identified (#229). Bit-exact vs RegisterBlocked, so this is pure perf.
            ("register-blocked -> wide-load unpack", I2SGemmVariant.RegisterBlocked, I2SGemmVariant.RegisterBlockedWide),
            // Bank-conflict fix: sharedW row stride 128 -> 129 words. Also bit-exact.
            ("register-blocked -> bank-padded sharedW", I2SGemmVariant.RegisterBlocked, I2SGemmVariant.RegisterBlockedPadded),
            // Profile-driven: VTune says 86.9% XVE stalled at 73.3% occupancy, i.e. latency-bound
            // with threads resident. Tests ILP (4x4 micro-tile, 64 threads) at CONSTANT shared memory.
            ("register-blocked -> 4x4 ILP", I2SGemmVariant.RegisterBlocked, I2SGemmVariant.RegisterBlocked4x4),
            // Control: halves shared memory. The profile predicts this is FLAT (occupancy is fine).
            ("register-blocked -> f16 shared (occupancy control)", I2SGemmVariant.RegisterBlocked, I2SGemmVariant.RegisterBlockedF16Shared),
            // Bit-exact: only the weight tile is F16. Halves weight-side SLM traffic (footprint 24 KB).
            ("register-blocked -> f16 weight tile (bit-exact)", I2SGemmVariant.RegisterBlocked, I2SGemmVariant.RegisterBlockedWeightF16),
        };
        if (device.HasCooperativeMatrix)
        {
            comparisons.Add(("register-blocked -> coopmat", I2SGemmVariant.RegisterBlocked, I2SGemmVariant.Coopmat));
            comparisons.Add(("coopmat -> coopmat32 (1 subgroup/wg probe)", I2SGemmVariant.Coopmat, I2SGemmVariant.Coopmat32));
            // Controlled test of the tile-size hypothesis: both are 1 subgroup/wg,
            // only the output tile differs (16x16 -> 32x32).
            comparisons.Add(("coopmat32 -> warptile (tile size only)", I2SGemmVariant.Coopmat32, I2SGemmVariant.CoopmatWarptile));
            // And against the production bar, which is what decides promotion.
            comparisons.Add(("register-blocked -> warptile", I2SGemmVariant.RegisterBlocked, I2SGemmVariant.CoopmatWarptile));
        }
        else
            _output.WriteLine("NOTE: VK_KHR_cooperative_matrix absent — coopmat comparison skipped.");

        // Optional substring filter so a profiler run (VTune gpu-hotspots) can target one
        // comparison instead of the whole sweep — under a profiler the full matrix is far too slow.
        string? only = Environment.GetEnvironmentVariable("DOTLLM_I2S_GEMM_BENCH_FILTER");
        if (!string.IsNullOrEmpty(only))
        {
            comparisons.RemoveAll(cmp => !cmp.Label.Contains(only, StringComparison.OrdinalIgnoreCase));
            _output.WriteLine($"FILTER: '{only}' -> {comparisons.Count} comparison(s)");
        }

        foreach (var (label, refVariant, challVariant) in comparisons)
        {
            using var refKernel = MatMulI2SGemmF32Kernel.Create(device, spvDir, refVariant);
            using var challKernel = MatMulI2SGemmF32Kernel.Create(device, spvDir, challVariant);

            _output.WriteLine("");
            _output.WriteLine($"### {label}");
            _output.WriteLine("| shape | reference µs | challenger µs | speedup | ref GFLOP/s | chall GFLOP/s |");
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
                rng.NextBytes(w);              // random packed codes; timing is data-independent
                float[] b = new float[(long)tokens * k];
                for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
                device.Upload(new ReadOnlySpan<byte>(w), bufW);
                device.Upload(b, bufB);

                (double refUs, double challUs, double ratio) =
                    MeasurePaired(device, refKernel, challKernel, bufW, bufB, bufC, m, k, tokens, batch);

                // 2 flops per MAC, M*K*tokens MACs.
                double flop = 2.0 * m * (double)k * tokens;
                double refGflops = flop / (refUs * 1e-6) / 1e9;
                double challGflops = flop / (challUs * 1e-6) / 1e9;
                _output.WriteLine($"| {tag} | {refUs:F2} | {challUs:F2} | {ratio:F2}x | {refGflops:F1} | {challGflops:F1} |");
            }
        }
    }

    private static (double reference, double challenger, double ratio) MeasurePaired(
        VulkanDevice device, MatMulI2SGemmF32Kernel reference, MatMulI2SGemmF32Kernel challenger,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, reference, w, b, c, m, k, n, batch);
            RunPass(device, challenger, w, b, c, m, k, n, batch);
        }

        var baseUs = new double[Passes];
        var rbUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tb, tr;
            if ((p & 1) == 0)   // alternate order so first-vs-second bias cancels
            {
                tb = RunPass(device, reference, w, b, c, m, k, n, batch);
                tr = RunPass(device, challenger, w, b, c, m, k, n, batch);
            }
            else
            {
                tr = RunPass(device, challenger, w, b, c, m, k, n, batch);
                tb = RunPass(device, reference, w, b, c, m, k, n, batch);
            }

            baseUs[p] = tb;
            rbUs[p] = tr;
            ratios[p] = tr > 0 ? tb / tr : 0;   // >1 means the challenger is faster
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
