using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #443: legacy 16x16 coopmat GEMM versus the shared 128x128x4 blocked template,
/// one arm per quantisation, all in ONE process.
/// </summary>
/// <remarks>
/// <para>
/// Methodology is <see cref="VulkanCoopmatGemmLadderBench"/>'s, unchanged, because that is the
/// harness the PQ2_0 result was established with: batched submissions behind one fence,
/// interleaved passes with the order reversed every pass, and the reported number is the
/// <b>median of per-pass ratios</b> — never a ratio of medians. Absolute tok/s on this UMA box
/// swings ~40% with CPU memory-bandwidth contention; only a back-to-back same-session ratio
/// survives that.
/// </para>
/// <para>
/// <b>The honest row exceeds the 32 MB MALL.</b> gfx1151's last-level cache is 32 MB, so a
/// weight tensor that fits in it measures cache behaviour rather than the memory system a real
/// model sees. Each arm therefore carries one shape whose packed weights exceed 32 MB and one
/// cache-resident shape for contrast, and the first is the one to quote.
/// </para>
/// <para>
/// <b>Small n is checked, not assumed.</b> A 128-wide N tile at n=4 is ~97% empty. On PQ2_0 the
/// blocked tile still won there (p=4 2.19x, p=8 2.12x) — but that is a fact about PQ2_0's
/// unpack cost, not a law, so <c>DOTLLM_BLOCKED_GEMM_TOKENS</c> exists to re-run any arm at
/// small n before a selection flip is trusted.
/// </para>
/// </remarks>
[Trait("Category", "Benchmark")]
public sealed class VulkanBlockedGemmTemplateBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;

    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the benchmark with the xUnit output sink.</summary>
    /// <param name="output">Sink for the result table.</param>
    public VulkanBlockedGemmTemplateBench(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// F16: <see cref="F16GemmCoopmatVariant.Coopmat64"/> (16x16, one subgroup) against
    /// <see cref="F16GemmCoopmatVariant.Blocked128x128x4"/>.
    /// </summary>
    /// <remarks>
    /// F16 has <b>no dequant at all</b>, so this arm is the roofline for what the tile alone
    /// buys: the only term that changes is <c>M*K*N / BN</c> weight staging.
    /// </remarks>
    [SkippableFact]
    public void Bench_F16_LegacyVsBlocked()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_BLOCKED_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_BLOCKED_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "VK_KHR_cooperative_matrix absent.");

        var legacy = F16GemmCoopmatVariant.Coopmat64;
        var blocked = F16GemmCoopmatVariant.Blocked128x128x4;
        Skip.IfNot(blocked.IsSupportedOn(device, spvDir), $"{blocked.SpvFileName} not compiled.");

        int tokens = EnvInt("DOTLLM_BLOCKED_GEMM_TOKENS", 256);
        int batch = EnvInt("DOTLLM_BLOCKED_GEMM_BATCH", 4);

        Header(device, spvDir, "F16 (no dequant — the tile in isolation)", tokens, batch);

        // F16 is 2 B/element: 8192x4096 packs to 64 MB, comfortably past the 32 MB MALL.
        (string Tag, int M, int K)[] shapes =
        [
            ("8192x4096   HONEST: 64 MB packed, exceeds 32 MB MALL", 8192, 4096),
            ("2048x2048   CACHE-RESIDENT contrast: 8 MB",            2048, 2048),
        ];

        using var refKernel = MatMulF16GemmCoopmatKernel.Create(device, spvDir, legacy);
        using var challKernel = MatMulF16GemmCoopmatKernel.Create(device, spvDir, blocked);

        Table();
        var rng = new Random(0x443F16A);
        foreach (var (tag, m, k) in shapes)
        {
            long wBytes = (long)m * k * 2;
            using var bufW = device.Allocate((wBytes + 3) & ~3L);
            using var bufB = device.Allocate((long)tokens * k * sizeof(float));
            using var bufC = device.Allocate((long)tokens * m * sizeof(float));

            // Buffers are re-allocated per shape while the kernels persist, so the handle-keyed
            // descriptor cache can hand back a set bound to freed memory — the failure mode is
            // correct-then-zeros output, which mimics a truncated kernel.
            refKernel.InvalidateDescriptorCache();
            challKernel.InvalidateDescriptorCache();

            byte[] w = new byte[wBytes];
            rng.NextBytes(w);
            // Random bytes include F16 NaN/Inf patterns; timing is data-independent on this
            // hardware but denormal-free data keeps the numbers boring. Mask the exponent down.
            for (long i = 1; i < w.LongLength; i += 2) w[i] &= 0x3F;
            float[] b = new float[(long)tokens * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
            device.Upload(new ReadOnlySpan<byte>(w), bufW);
            device.Upload(b, bufB);

            (double refUs, double challUs, double ratio) = MeasurePaired(
                device,
                (cmd) => refKernel.Record(cmd, bufW, bufB, bufC, m, k, tokens),
                (cmd) => challKernel.Record(cmd, bufW, bufB, bufC, m, k, tokens),
                batch);

            Row(tag, m, k, tokens, refUs, challUs, ratio);
        }
    }

    /// <summary>
    /// Q8_0: <see cref="Q8_0GemmCoopmatVariant.Coopmat64"/> against
    /// <see cref="Q8_0GemmCoopmatVariant.Blocked128x128x4"/>.
    /// </summary>
    /// <remarks>
    /// <b>The coopmat GEMM is not what ships for Q8_0 prefill on this device</b> — the dp4a MMQ
    /// path (<c>MatMulQ8_0MmqKernel</c>, issue #50) is preferred and the coopmat kernel is the
    /// fallback behind it. This arm therefore measures the template against the coopmat kernel
    /// it replaces; whether it also beats MMQ is a separate question, measured end to end on a
    /// real Q8_0 model rather than here.
    /// </remarks>
    [SkippableFact]
    public void Bench_Q8_0_LegacyVsBlocked()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_BLOCKED_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_BLOCKED_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "VK_KHR_cooperative_matrix absent.");

        var legacy = Q8_0GemmCoopmatVariant.Coopmat64;
        var blocked = Q8_0GemmCoopmatVariant.Blocked128x128x4;
        Skip.IfNot(blocked.IsSupportedOn(device, spvDir), $"{blocked.SpvFileName} not compiled.");

        int tokens = EnvInt("DOTLLM_BLOCKED_GEMM_TOKENS", 256);
        int batch = EnvInt("DOTLLM_BLOCKED_GEMM_BATCH", 4);

        Header(device, spvDir, "Q8_0 (34 B per 32-element block)", tokens, batch);

        // Q8_0 packs 34 B per 32 elements ~ 1.06 B/element: 8192x4096 packs to 35.7 MB.
        (string Tag, int M, int K)[] shapes =
        [
            ("14336x4096  HONEST: 62 MB packed, exceeds 32 MB MALL", 14336, 4096),
            ("4096x4096   CACHE-RESIDENT contrast: 17.8 MB",          4096, 4096),
        ];

        using var refKernel = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, legacy);
        using var challKernel = MatMulQ8_0GemmCoopmatKernel.Create(device, spvDir, blocked);

        Table();
        var rng = new Random(0x443008);
        foreach (var (tag, m, k) in shapes)
        {
            long rowBytes = (long)(k / 32) * 34;
            long wBytes = m * rowBytes;
            using var bufW = device.Allocate((wBytes + 3) & ~3L);
            using var bufB = device.Allocate((long)tokens * k * sizeof(float));
            using var bufC = device.Allocate((long)tokens * m * sizeof(float));

            refKernel.InvalidateDescriptorCache();
            challKernel.InvalidateDescriptorCache();

            byte[] w = new byte[wBytes];
            rng.NextBytes(w);
            // Keep the per-block fp16 scales small and finite; the int8 codes can be anything.
            for (long blk = 0; blk + 1 < wBytes; blk += 34) w[blk + 1] &= 0x2F;
            float[] b = new float[(long)tokens * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
            device.Upload(new ReadOnlySpan<byte>(w), bufW);
            device.Upload(b, bufB);

            (double refUs, double challUs, double ratio) = MeasurePaired(
                device,
                (cmd) => refKernel.Record(cmd, bufW, bufB, bufC, m, k, tokens),
                (cmd) => challKernel.Record(cmd, bufW, bufB, bufC, m, k, tokens),
                batch);

            Row(tag, m, k, tokens, refUs, challUs, ratio);
        }
    }

    /// <summary>
    /// I2_S: the production <see cref="I2SGemmVariant.RegisterBlocked"/> F32 kernel against
    /// <see cref="I2SGemmVariant.Blocked128x128x4"/>.
    /// </summary>
    /// <remarks>
    /// <b>The shipping bar on gfx1151 is <see cref="I2SGemmVariant.Coopmat32Wave32"/>, not
    /// RegisterBlocked</b> — <see cref="I2SGemmVariant.SelectFor"/> prefers it on AMD, where it
    /// measured 1.64-1.74x over the register-blocked kernel. Both references are run: the
    /// selected one is the bar a flip has to clear, RegisterBlocked is the F32 context.
    /// </remarks>
    [SkippableFact]
    public void Bench_I2S_ProductionVsBlocked()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_BLOCKED_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_BLOCKED_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "VK_KHR_cooperative_matrix absent.");

        var blocked = I2SGemmVariant.Blocked128x128x4;
        Skip.IfNot(File.Exists(Path.Combine(spvDir, blocked.SpvFileName)), $"{blocked.SpvFileName} not compiled.");

        int tokens = EnvInt("DOTLLM_BLOCKED_GEMM_TOKENS", 256);
        int batch = EnvInt("DOTLLM_BLOCKED_GEMM_BATCH", 4);

        Header(device, spvDir, "I2_S (ternary, 32 B per 128-element block)", tokens, batch);

        // I2_S packs 0.25 B/element: 65536x4096 packs to 64 MB.
        (string Tag, int M, int K)[] shapes =
        [
            ("65536x4096  HONEST: 64 MB packed, exceeds 32 MB MALL", 65536, 4096),
            ("8192x4096   CACHE-RESIDENT contrast: 8 MB",             8192, 4096),
        ];

        foreach (var (label, refVariant) in new[]
        {
            ("SelectFor's Coopmat32Wave32 -> blocked 128x128x4 [THE SHIPPING BAR]", I2SGemmVariant.Coopmat32Wave32),
            ("RegisterBlocked (F32) -> blocked 128x128x4       [F32 context]", I2SGemmVariant.RegisterBlocked),
            ("legacy coopmat 16x16 -> blocked 128x128x4        [tile only]", I2SGemmVariant.Coopmat),
        })
        {
            using var refKernel = MatMulI2SGemmF32Kernel.Create(device, spvDir, refVariant);
            using var challKernel = MatMulI2SGemmF32Kernel.Create(device, spvDir, blocked);

            _output.WriteLine("");
            _output.WriteLine($"### {label}");
            Table();

            var rng = new Random(0x443125);
            foreach (var (tag, m, k) in shapes)
            {
                long wBytes = (long)m * (k / 4) + 4;   // packed codes + the tail f32 scale
                using var bufW = device.Allocate((wBytes + 3) & ~3L);
                using var bufB = device.Allocate((long)tokens * k * sizeof(float));
                using var bufC = device.Allocate((long)tokens * m * sizeof(float));

                refKernel.InvalidateDescriptorCache();
                challKernel.InvalidateDescriptorCache();

                byte[] w = new byte[wBytes];
                rng.NextBytes(w);
                BitConverter.TryWriteBytes(w.AsSpan((int)((long)m * (k / 4))), 0.01f);
                float[] b = new float[(long)tokens * k];
                for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
                device.Upload(new ReadOnlySpan<byte>(w), bufW);
                device.Upload(b, bufB);

                (double refUs, double challUs, double ratio) = MeasurePaired(
                    device,
                    (cmd) => refKernel.Record(cmd, bufW, bufB, bufC, m, k, tokens),
                    (cmd) => challKernel.Record(cmd, bufW, bufB, bufC, m, k, tokens),
                    batch);

                Row(tag, m, k, tokens, refUs, challUs, ratio);
            }
        }
    }

    private void Header(VulkanDevice device, string spvDir, string what, int tokens, int batch)
    {
        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}");
        _output.WriteLine($"spvDir: {spvDir}");
        _output.WriteLine($"### {what}");
        _output.WriteLine($"tokens={tokens}  batch={batch}  {WarmupPasses} warmup + {Passes} interleaved "
            + "order-reversed passes (median of per-pass RATIOS)");
    }

    private void Table()
    {
        _output.WriteLine("| shape | reference us | challenger us | speedup | ref GFLOP/s | chall GFLOP/s |");
        _output.WriteLine("|---|---:|---:|---:|---:|---:|");
    }

    private void Row(string tag, int m, int k, int tokens, double refUs, double challUs, double ratio)
    {
        double flop = 2.0 * m * (double)k * tokens;
        _output.WriteLine($"| {tag} | {refUs:F1} | {challUs:F1} | {ratio:F2}x | "
            + $"{flop / (refUs * 1e-6) / 1e9:F1} | {flop / (challUs * 1e-6) / 1e9:F1} |");
    }

    private static (double reference, double challenger, double ratio) MeasurePaired(
        VulkanDevice device, Action<nint> reference, Action<nint> challenger, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunPass(device, reference, batch);
            RunPass(device, challenger, batch);
        }

        var refUs = new double[Passes];
        var challUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tr, tc;
            if ((p & 1) == 0)   // alternate order so first-vs-second bias cancels
            {
                tr = RunPass(device, reference, batch);
                tc = RunPass(device, challenger, batch);
            }
            else
            {
                tc = RunPass(device, challenger, batch);
                tr = RunPass(device, reference, batch);
            }

            refUs[p] = tr;
            challUs[p] = tc;
            ratios[p] = tc > 0 ? tr / tc : 0;   // >1 means the challenger is faster
        }

        Array.Sort(refUs); Array.Sort(challUs); Array.Sort(ratios);
        return (refUs[Passes / 2], challUs[Passes / 2], ratios[Passes / 2]);
    }

    private static double RunPass(VulkanDevice device, Action<nint> record, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++) record(ctx.CommandBuffer);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;
}
