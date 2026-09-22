using System.Diagnostics;
using System.Globalization;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Opt-in micro-benchmark and A/B for the Vulkan PQ2_0 (PrismML Bonsai ternary) prefill path
/// (issue #233), timed side by side in one process.
/// </summary>
/// <remarks>
/// <para>
/// <b>What the reference is.</b> Before #233 there was no PQ2_0 GEMM at all — the dispatcher
/// threw for <c>seqLen &gt; 1</c> and prefill had to feed tokens one at a time through the GEMV
/// decode kernel. So the honest baseline is exactly that: <c>n</c> GEMV dispatches, one per
/// token, recorded into the same command buffer. The challenger is the new 32x32
/// register-blocked GEMM in one dispatch. This is a like-for-like measurement of the change the
/// PR actually makes to prefill, not a comparison against a strawman.
/// </para>
/// <para>
/// <b>Methodology</b> is deliberately identical to <see cref="VulkanI2SGemmBench"/>, because on
/// this box it is the methodology that makes a number trustworthy:
/// </para>
/// <list type="bullet">
/// <item><b>Compute-bound timed region:</b> each pass submits <c>batch</c> repetitions behind one
/// fence (default 8; env <c>DOTLLM_PQ2_0_GEMM_BENCH_BATCH</c>) so submit/fence latency
/// amortizes.</item>
/// <item><b>Interleaved paired A/B with the order reversed every pass</b>, medianing the per-pass
/// RATIO. A one-directional cold-vs-warm comparison on this box has produced 2-3x PHANTOM deltas
/// purely from GPU clock ramp; alternating the order within a session cancels it.</item>
/// <item><b>Same-session only.</b> On a UMA part absolute throughput swings ~40% with CPU
/// memory-bandwidth contention, so only the back-to-back ratio within one process is reported as
/// evidence. The absolute µs/GFLOP-s columns are context, not claims.</item>
/// </list>
/// <para>
/// <b>Cache-residency caveat — read the per-shape numbers with this in mind.</b> The looped-GEMV
/// reference re-dispatches over the SAME weight tensor <c>n</c> times back to back, so on a part
/// with a large last-level cache (gfx1151 has a 32 MB MALL) any tensor that fits stays resident
/// and the reference never pays DRAM traffic for repeats 2..n. Real one-token-at-a-time prefill
/// does the opposite: between two visits to the same tensor the model walks all 7.2 GB of
/// Bonsai-27B, so every visit is a cold DRAM read. The micro-benchmark therefore <i>understates</i>
/// the fallback's true cost on every tensor under ~32 MB — which is all of the per-layer
/// projections. <c>lm_head</c> (337 MB packed) is included precisely because it cannot fit, and is
/// the shape whose ratio reflects the uncached reality the other shapes would show in situ.
/// </para>
/// <para>
/// Enable with <c>DOTLLM_PQ2_0_GEMM_BENCH=1</c>. Token count is
/// <c>DOTLLM_PQ2_0_GEMM_BENCH_TOKENS</c> (default 64).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanPQ2_0GemmBench
{
    private const int WarmupPasses = 2;
    private const int Passes = 9;
    private const int GroupSize = 128;
    private const int GroupBytes = 34;

    private readonly ITestOutputHelper _output;

    /// <summary>Initializes the benchmark with the xUnit output sink.</summary>
    /// <param name="output">Sink for the result table.</param>
    public VulkanPQ2_0GemmBench(ITestOutputHelper output) => _output = output;

    /// <summary>
    /// Projection shapes, read off the real <c>Ternary-Bonsai-27B-Q2_0.gguf</c> tensor headers
    /// (<c>embedding_length</c> 5120, <c>feed_forward_length</c> 17408; GGUF dims are
    /// <c>[K, M]</c>). Override with <c>DOTLLM_PQ2_0_GEMM_BENCH_SHAPES</c> as a
    /// semicolon-separated <c>tag,M,K</c> list to re-run against another checkpoint without a
    /// rebuild.
    /// </summary>
    private static readonly (string Tag, int M, int K)[] DefaultShapes =
    [
        ("attn_q      (12288x5120)", 12288, 5120),
        ("attn_output  (5120x6144)",  5120, 6144),
        ("ffn_gate/up (17408x5120)", 17408, 5120),
        ("ffn_down    (5120x17408)",  5120, 17408),
        ("lm_head    (248320x5120)", 248320, 5120),
    ];

    /// <summary>
    /// Times every available GEMM variant against every other, interleaved and order-reversed
    /// (issue #236). The pair that decides promotion is
    /// <c>register-blocked -&gt; coopmat</c>: #233's register-blocked kernel is what ships, so it
    /// is the bar, and a challenger measured against the looped-GEMV fallback instead would be
    /// flattered by a baseline no longer in the code.
    /// </summary>
    /// <remarks>
    /// Read <c>lm_head</c> (337 MB packed) as the honest row and the projections as the optimistic
    /// one — see the class remarks on MALL residency. Enable with
    /// <c>DOTLLM_PQ2_0_GEMM_BENCH=1</c>.
    /// </remarks>
    [SkippableFact]
    public void Bench_PQ2_0GemmVariants()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_PQ2_0_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        PinToPCores();
        int batch = EnvInt("DOTLLM_PQ2_0_GEMM_BENCH_BATCH", 8);
        int tokens = EnvInt("DOTLLM_PQ2_0_GEMM_BENCH_TOKENS", 64);
        var shapes = ParseShapes(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH_SHAPES")) ?? DefaultShapes;

        using var device = VulkanDevice.Create();
        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  Coopmat: {device.HasCooperativeMatrix}");
        _output.WriteLine($"tokens={tokens}  batch={batch}  schedule: {WarmupPasses} warmup + {Passes} interleaved order-reversed passes (median of per-pass ratios)");
        _output.WriteLine($"SelectFor(device) => {PQ2_0GemmVariant.SelectFor(device).SpvFileName}");

        var comparisons = new List<(string Label, PQ2_0GemmVariant Reference, PQ2_0GemmVariant Challenger)>();
        if (PQ2_0GemmVariant.Coopmat.IsSupportedOn(device))
            comparisons.Add(("register-blocked -> coopmat (64-thread wg, driver-default wave)",
                PQ2_0GemmVariant.RegisterBlocked, PQ2_0GemmVariant.Coopmat));
        if (PQ2_0GemmVariant.Coopmat32.IsSupportedOn(device))
        {
            comparisons.Add(("coopmat -> coopmat32 (32-thread wg pinned to wave32)",
                PQ2_0GemmVariant.Coopmat, PQ2_0GemmVariant.Coopmat32));
            comparisons.Add(("register-blocked -> coopmat32",
                PQ2_0GemmVariant.RegisterBlocked, PQ2_0GemmVariant.Coopmat32));
        }
        Skip.If(comparisons.Count == 0, "VK_KHR_cooperative_matrix absent — nothing to compare.");

        foreach (var (label, refVariant, challVariant) in comparisons)
        {
            using var refKernel = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, refVariant);
            using var challKernel = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, challVariant);

            _output.WriteLine("");
            _output.WriteLine($"### {label}");
            _output.WriteLine("| shape | reference µs | challenger µs | speedup | ref GFLOP/s | chall GFLOP/s |");
            _output.WriteLine("|---|---:|---:|---:|---:|---:|");

            var rng = new Random(0x2A_53);
            foreach (var (tag, m, k) in shapes)
            {
                long rowBytes = (long)(k / GroupSize) * GroupBytes;
                long wBytes = m * rowBytes;
                using var bufW = device.Allocate((wBytes + 3) & ~3L);
                using var bufB = device.Allocate((long)tokens * k * sizeof(float));
                using var bufC = device.Allocate((long)tokens * m * sizeof(float));

                byte[] w = new byte[wBytes];
                rng.NextBytes(w);              // random packed codes; timing is data-independent
                float[] b = new float[(long)tokens * k];
                for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
                device.Upload(new ReadOnlySpan<byte>(w), bufW);
                device.Upload(b, bufB);

                (double refUs, double challUs, double ratio) =
                    MeasurePairedGemm(device, refKernel, challKernel, bufW, bufB, bufC, m, k, tokens, batch);

                double flop = 2.0 * m * (double)k * tokens;
                _output.WriteLine($"| {tag} | {refUs:F2} | {challUs:F2} | {ratio:F2}x | "
                    + $"{flop / (refUs * 1e-6) / 1e9:F1} | {flop / (challUs * 1e-6) / 1e9:F1} |");
            }
        }
    }

    private static (double reference, double challenger, double ratio) MeasurePairedGemm(
        VulkanDevice device, MatMulPQ2_0GemmF32Kernel reference, MatMulPQ2_0GemmF32Kernel challenger,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunGemmPass(device, reference, w, b, c, m, k, n, batch);
            RunGemmPass(device, challenger, w, b, c, m, k, n, batch);
        }

        var refUs = new double[Passes];
        var challUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tr, tc;
            if ((p & 1) == 0)   // alternate order so first-vs-second bias cancels
            {
                tr = RunGemmPass(device, reference, w, b, c, m, k, n, batch);
                tc = RunGemmPass(device, challenger, w, b, c, m, k, n, batch);
            }
            else
            {
                tc = RunGemmPass(device, challenger, w, b, c, m, k, n, batch);
                tr = RunGemmPass(device, reference, w, b, c, m, k, n, batch);
            }

            refUs[p] = tr;
            challUs[p] = tc;
            ratios[p] = tc > 0 ? tr / tc : 0;   // >1 means the challenger is faster
        }

        Array.Sort(refUs); Array.Sort(challUs); Array.Sort(ratios);
        return (refUs[Passes / 2], challUs[Passes / 2], ratios[Passes / 2]);
    }

    /// <summary>
    /// Times looped-GEMV prefill (the pre-#233 fallback) against each available GEMM variant,
    /// interleaved and paired.
    /// </summary>
    [SkippableFact]
    public void Bench_PQ2_0Gemm_VsLoopedGemvFallback()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_PQ2_0_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        PinToPCores();
        int batch = EnvInt("DOTLLM_PQ2_0_GEMM_BENCH_BATCH", 8);
        int tokens = EnvInt("DOTLLM_PQ2_0_GEMM_BENCH_TOKENS", 64);
        var shapes = ParseShapes(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH_SHAPES")) ?? DefaultShapes;

        using var device = VulkanDevice.Create();
        using var gemv = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  Coopmat: {device.HasCooperativeMatrix}");
        _output.WriteLine($"tokens={tokens}  batch={batch}  schedule: {WarmupPasses} warmup + {Passes} interleaved order-reversed passes (median of per-pass ratios)");

        foreach (var variant in PQ2_0GemmVariant.AvailableOn(device))
        {
            using var gemm = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, variant);

            _output.WriteLine("");
            _output.WriteLine($"### looped-GEMV -> {variant.SpvFileName}");
            _output.WriteLine("| shape | looped-GEMV µs | GEMM µs | speedup | GEMV GFLOP/s | GEMM GFLOP/s |");
            _output.WriteLine("|---|---:|---:|---:|---:|---:|");

            var rng = new Random(0x2A_53);
            foreach (var (tag, m, k) in shapes)
            {
                long rowBytes = (long)(k / GroupSize) * GroupBytes;
                long wBytes = m * rowBytes;
                using var bufW = device.Allocate((wBytes + 3) & ~3L);
                using var bufB = device.Allocate((long)tokens * k * sizeof(float));
                using var bufC = device.Allocate((long)tokens * m * sizeof(float));

                byte[] w = new byte[wBytes];
                rng.NextBytes(w);              // random packed codes; timing is data-independent
                float[] b = new float[(long)tokens * k];
                for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
                device.Upload(new ReadOnlySpan<byte>(w), bufW);
                device.Upload(b, bufB);

                (double gemvUs, double gemmUs, double ratio) =
                    MeasurePaired(device, gemv, gemm, bufW, bufB, bufC, m, k, tokens, batch);

                double flop = 2.0 * m * (double)k * tokens;   // 2 flops per MAC
                _output.WriteLine($"| {tag} | {gemvUs:F2} | {gemmUs:F2} | {ratio:F2}x | "
                    + $"{flop / (gemvUs * 1e-6) / 1e9:F1} | {flop / (gemmUs * 1e-6) / 1e9:F1} |");
            }
        }
    }

    private static (double gemv, double gemm, double ratio) MeasurePaired(
        VulkanDevice device, MatMulPQ2_0GemvF32Kernel gemv, MatMulPQ2_0GemmF32Kernel gemm,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        for (int i = 0; i < WarmupPasses; i++)
        {
            RunGemvPass(device, gemv, w, b, c, m, k, n, batch);
            RunGemmPass(device, gemm, w, b, c, m, k, n, batch);
        }

        var gemvUs = new double[Passes];
        var gemmUs = new double[Passes];
        var ratios = new double[Passes];
        for (int p = 0; p < Passes; p++)
        {
            double tv, tm;
            if ((p & 1) == 0)   // alternate order so first-vs-second bias cancels
            {
                tv = RunGemvPass(device, gemv, w, b, c, m, k, n, batch);
                tm = RunGemmPass(device, gemm, w, b, c, m, k, n, batch);
            }
            else
            {
                tm = RunGemmPass(device, gemm, w, b, c, m, k, n, batch);
                tv = RunGemvPass(device, gemv, w, b, c, m, k, n, batch);
            }

            gemvUs[p] = tv;
            gemmUs[p] = tm;
            ratios[p] = tm > 0 ? tv / tm : 0;   // >1 means the GEMM is faster
        }

        Array.Sort(gemvUs); Array.Sort(gemmUs); Array.Sort(ratios);
        return (gemvUs[Passes / 2], gemmUs[Passes / 2], ratios[Passes / 2]);
    }

    /// <summary>
    /// The pre-#233 prefill fallback: one GEMV dispatch per token. Each token <c>t</c> reads
    /// <c>b[t·K ..]</c> and writes <c>c[t·M ..]</c>, so the offsets are expressed by binding
    /// sub-ranges — but <see cref="VulkanDevice.Buffer"/> has no offset view here, so the loop
    /// re-dispatches over the same buffers. That is timing-equivalent (identical dispatch count,
    /// identical bytes moved per dispatch) and is what is being measured.
    /// </summary>
    private static double RunGemvPass(
        VulkanDevice device, MatMulPQ2_0GemvF32Kernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            for (int t = 0; t < n; t++)
                kernel.Record(ctx.CommandBuffer, w, b, c, m, k);   // no barriers → dispatches overlap
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static double RunGemmPass(
        VulkanDevice device, MatMulPQ2_0GemmF32Kernel kernel,
        VulkanDevice.Buffer w, VulkanDevice.Buffer b, VulkanDevice.Buffer c,
        int m, int k, int n, int batch)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++)
            kernel.Record(ctx.CommandBuffer, w, b, c, m, k, n);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    /// <summary>
    /// Issue #446 — the small-n dispatch crossover. Sweeps <c>n = 1..64</c> with the
    /// <b>current default</b> GEMM variant (and <see cref="PQ2_0GemmVariant.Coopmat32"/> as the
    /// tie-back to #435's numbers) against a looped GEMV, to locate the <c>n</c> at which
    /// <c>RecordMatmul</c>'s <c>seqLen == 1 ? GEMV : GEMM</c> dispatch should actually switch.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>The GEMV arm is a lower bound on a real looped GEMV, so its verdict is one-sided.</b>
    /// <see cref="RunGemvPass"/> re-dispatches over the same buffer offsets with no barriers, so
    /// the dispatches overlap and every repeat after the first hits whatever the previous one
    /// left in cache. A shippable loop would bind per-token sub-ranges; since the three buffer
    /// handles are unchanged that costs only two extra push-constant words (the
    /// <c>DescriptorSetCache</c> is handle-keyed and would still hit), and the disjoint
    /// <c>y[t·M..]</c> writes still need no barriers — so the only real optimism left is cache
    /// residency, which cannot apply to <c>lm_head</c> at 337 MB packed. <b>Therefore: wherever
    /// the GEMM beats this arm, it also beats a real looped GEMV, and no crossover exists
    /// there.</b> The converse is not safe to assert on the sub-32 MB projections.
    /// </para>
    /// <para>
    /// <c>speedup</c> is GEMM-relative: <c>&gt; 1.00x</c> means the GEMM wins and the current
    /// dispatch is right; <c>&lt; 1.00x</c> means a looped GEMV would be faster at that n.
    /// Enable with <c>DOTLLM_PQ2_0_GEMM_BENCH=1</c>; override the ladder with
    /// <c>DOTLLM_PQ2_0_CROSSOVER_N</c> (comma-separated).
    /// </para>
    /// </remarks>
    [SkippableFact]
    public void Bench_PQ2_0SmallNCrossover()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_PQ2_0_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        PinToPCores();
        int baseBatch = EnvInt("DOTLLM_PQ2_0_GEMM_BENCH_BATCH", 8);
        int[] ns = ParseNs(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_CROSSOVER_N"))
            ?? [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
        var shapes = ParseShapes(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH_SHAPES"))
            ?? [DefaultShapes[2], DefaultShapes[4]];   // ffn_gate/up (fits MALL) + lm_head (does not)

        using var device = VulkanDevice.Create();
        using var gemv = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);

        var selected = PQ2_0GemmVariant.SelectFor(device);
        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  Coopmat: {device.HasCooperativeMatrix}");
        _output.WriteLine($"SelectFor(device) => {selected.SpvFileName}   (THE CURRENT DEFAULT)");
        _output.WriteLine($"baseBatch={baseBatch}  schedule: {WarmupPasses} warmup + {Passes} interleaved order-reversed passes (median of per-pass ratios)");
        _output.WriteLine("speedup is GEMM-relative: >1.00x = GEMM wins (current dispatch correct); <1.00x = a looped GEMV would win.");

        var arms = new List<(string Label, PQ2_0GemmVariant Variant)> { ($"DEFAULT {selected.SpvFileName}", selected) };
        if (PQ2_0GemmVariant.Coopmat32.IsSupportedOn(device) && selected != PQ2_0GemmVariant.Coopmat32)
            arms.Add(("#435 tie-back: coopmat32", PQ2_0GemmVariant.Coopmat32));

        foreach (var (label, variant) in arms)
        {
            using var gemm = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, variant);
            foreach (var (tag, m, k) in shapes)
            {
                _output.WriteLine("");
                _output.WriteLine($"### {label} — {tag}");
                _output.WriteLine("| n | batch | looped-GEMV µs | GEMM µs | speedup |");
                _output.WriteLine("|---:|---:|---:|---:|---:|");

                long rowBytes = (long)(k / GroupSize) * GroupBytes;
                long wBytes = m * rowBytes;
                int maxN = ns.Max();
                using var bufW = device.Allocate((wBytes + 3) & ~3L);
                using var bufB = device.Allocate((long)maxN * k * sizeof(float));
                using var bufC = device.Allocate((long)maxN * m * sizeof(float));

                var rng = new Random(0x2A_53);
                byte[] w = new byte[wBytes];
                rng.NextBytes(w);              // random packed codes; timing is data-independent
                float[] b = new float[(long)maxN * k];
                for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
                device.Upload(new ReadOnlySpan<byte>(w), bufW);
                device.Upload(b, bufB);

                foreach (int n in ns)
                {
                    // The GEMV arm costs O(n) dispatches, so hold total work roughly constant as
                    // n grows or lm_head at n=64 takes ~a minute per pass and burns the lock.
                    int batch = Math.Max(1, baseBatch / Math.Max(1, (n + 7) / 8));
                    (double gemvUs, double gemmUs, double ratio) =
                        MeasurePaired(device, gemv, gemm, bufW, bufB, bufC, m, k, n, batch);
                    _output.WriteLine($"| {n} | {batch} | {gemvUs:F2} | {gemmUs:F2} | {ratio:F2}x |");
                }
            }
        }
    }

    /// <summary>
    /// Issue #470 — the multi-column GEMV against the two kernels it sits between: the real
    /// per-token GEMV loop it replaces (per-token offsets, as <see cref="PQ2_0SmallNDispatch"/>
    /// records it) and the default GEMM. Three arms, and the order rotates every pass.
    /// </summary>
    /// <remarks>
    /// Every column is reported against <b>one single-column GEMV</b> (<c>n = 1</c> of the loop
    /// arm), which is the ratio #470's targets are stated in (S=2 at 1.2x or less, S=4 at 1.4x
    /// or less). Read <c>lm_head</c> as the honest row. It is 337 MB packed, so each repetition is
    /// a DRAM read, as in a real forward. The projections fit the 32 MB MALL, so their repetitions
    /// are cache hits and flatter both GEMV arms. Enable with <c>DOTLLM_PQ2_0_GEMM_BENCH=1</c>;
    /// <c>DOTLLM_PQ2_0_CROSSOVER_N</c> overrides the column ladder (max 8).
    /// </remarks>
    [SkippableFact]
    public void Bench_PQ2_0MultiColumnGemv()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_PQ2_0_GEMM_BENCH=1 to enable this benchmark.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        PinToPCores();
        int batch = EnvInt("DOTLLM_PQ2_0_GEMM_BENCH_BATCH", 8);
        int[] ns = (ParseNs(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_CROSSOVER_N")) ?? [1, 2, 3, 4, 5, 6, 7, 8])
            .Where(n => n <= MatMulPQ2_0GemvF32Kernel.MaxColumns).ToArray();
        var shapes = ParseShapes(Environment.GetEnvironmentVariable("DOTLLM_PQ2_0_GEMM_BENCH_SHAPES")) ?? DefaultShapes;

        using var device = VulkanDevice.Create();
        using var gemv = MatMulPQ2_0GemvF32Kernel.Create(device, spvDir);
        var variant = PQ2_0GemmVariant.SelectFor(device);
        using var gemm = MatMulPQ2_0GemmF32Kernel.Create(device, spvDir, variant);

        _output.WriteLine($"Device: {device.DeviceName}  SubgroupSize: {device.SubgroupSize}  GEMM: {variant.SpvFileName}");
        _output.WriteLine($"batch={batch}  schedule: {WarmupPasses} warmup + {Passes} passes, 3 arms, order rotated every pass (medians)");

        foreach (var (tag, m, k) in shapes)
        {
            _output.WriteLine("");
            _output.WriteLine($"### {tag}");
            _output.WriteLine("| n | loop µs | multi-col µs | GEMM µs | loop / 1-col | multi-col / 1-col | GEMM / 1-col |");
            _output.WriteLine("|---:|---:|---:|---:|---:|---:|---:|");

            long wBytes = (long)m * (k / GroupSize) * GroupBytes;
            int maxN = ns.Max();
            using var bufW = device.Allocate((wBytes + 3) & ~3L);
            using var bufB = device.Allocate((long)maxN * k * sizeof(float));
            using var bufC = device.Allocate((long)maxN * m * sizeof(float));

            var rng = new Random(0x2A_70);
            byte[] w = new byte[wBytes];
            rng.NextBytes(w);
            float[] b = new float[(long)maxN * k];
            for (int i = 0; i < b.Length; i++) b[i] = rng.NextSingle() * 2f - 1f;
            device.Upload(new ReadOnlySpan<byte>(w), bufW);
            device.Upload(b, bufB);

            double oneColUs = 0;
            foreach (int n in ns)
            {
                Func<double>[] arms =
                [
                    () => Time(device, batch, cb => { for (int t = 0; t < n; t++) gemv.Record(cb, bufW, bufB, bufC, m, k, t * k, t * m); }),
                    () => Time(device, batch, cb => gemv.RecordColumns(cb, bufW, bufB, bufC, m, k, n)),
                    () => Time(device, batch, cb => gemm.Record(cb, bufW, bufB, bufC, m, k, n)),
                ];
                for (int i = 0; i < WarmupPasses; i++)
                    foreach (var arm in arms) arm();

                var us = new double[arms.Length][];
                for (int a = 0; a < arms.Length; a++) us[a] = new double[Passes];
                for (int p = 0; p < Passes; p++)
                    for (int j = 0; j < arms.Length; j++)
                    {
                        int a = (p + j) % arms.Length;   // rotate which arm goes first
                        us[a][p] = arms[a]();
                    }

                double[] med = us.Select(v => { Array.Sort(v); return v[Passes / 2]; }).ToArray();
                if (n == 1 || oneColUs == 0) oneColUs = med[0];
                _output.WriteLine($"| {n} | {med[0]:F1} | {med[1]:F1} | {med[2]:F1} | "
                    + $"{med[0] / oneColUs:F2}x | {med[1] / oneColUs:F2}x | {med[2] / oneColUs:F2}x |");
            }
        }
    }

    private static double Time(VulkanDevice device, int batch, Action<nint> record)
    {
        using var ctx = device.CreateSubmitContext();
        var sw = Stopwatch.StartNew();
        ctx.Begin();
        for (int i = 0; i < batch; i++) record(ctx.CommandBuffer);
        ctx.SubmitAndWait();
        sw.Stop();
        return sw.Elapsed.TotalMicroseconds / batch;
    }

    private static int[]? ParseNs(string? spec)
    {
        if (string.IsNullOrWhiteSpace(spec)) return null;
        var list = new List<int>();
        foreach (string entry in spec.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
            if (int.TryParse(entry, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0)
                list.Add(v);
        return list.Count > 0 ? [.. list] : null;
    }

    private static (string Tag, int M, int K)[]? ParseShapes(string? spec)
    {
        if (string.IsNullOrWhiteSpace(spec)) return null;
        var list = new List<(string, int, int)>();
        foreach (string entry in spec.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            string[] parts = entry.Split(',');
            if (parts.Length == 3
                && int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int m)
                && int.TryParse(parts[2], NumberStyles.Integer, CultureInfo.InvariantCulture, out int k))
                list.Add((parts[0], m, k));
        }
        return list.Count > 0 ? [.. list] : null;
    }

    private static int EnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;

    private static void PinToPCores()
    {
        if (!OperatingSystem.IsWindows() && !OperatingSystem.IsLinux())
            return; // ProcessorAffinity is only supported on Windows/Linux; affinity is best-effort.

        string? mask = Environment.GetEnvironmentVariable("DOTLLM_BENCH_AFFINITY");
        if (string.IsNullOrWhiteSpace(mask)) return;
        try
        {
            Process.GetCurrentProcess().ProcessorAffinity = (nint)Convert.ToInt64(mask, 16);
        }
        catch { /* affinity is best-effort */ }
    }
}
