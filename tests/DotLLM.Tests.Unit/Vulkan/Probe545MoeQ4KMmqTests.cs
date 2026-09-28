using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #545 for <c>moe_indexed_matmul_q4_k_q8_1.comp</c> — the MoE routed
/// expert GEMM, scored against the same f64 oracle the dense families use.
/// <b>The two-level accumulation was measured here and REJECTED</b>; this probe
/// now guards the family at its current accuracy and records why.
/// </summary>
/// <remarks>
/// <para>
/// The existing <see cref="VulkanMoeIndexedMatmulQ4KMmqKernelTests"/> is a
/// correctness test with an argmax-plus-tolerance oracle. That cannot see the
/// thing #545 is about, which is a few parts in 10^7 of reduction error: it is a
/// <b>magnitude</b> measurement, so it needs a magnitude reference.
/// </para>
/// <para>
/// One expert and every row routed to it, deliberately. The routing is not what
/// is being measured — the per-row reduction is — and with a single expert the
/// arithmetic is exactly the Q4_K x Q8_1 dot that
/// <see cref="KQuantMmqOracle"/> already scores for the dense kernels, so the
/// same validated oracle applies instead of a second bespoke one. (That oracle
/// was itself checked against #549's hand-rolled Q4_K unpack and agreed to
/// 0.000E+000; a family-agnostic oracle that is wrong is wrong for every family
/// at once, which is why that check came first.)
/// </para>
/// <para>
/// The activation is dequantized from the words the <b>GPU's own</b> quantizer
/// wrote, downloaded back — never a CPU re-quantization of the f32 input, which
/// would inject rounding that is not the kernel's and is how an earlier probe in
/// this campaign scored 2.098 RMS against a stale buffer.
/// </para>
/// <h3>Why #545's fix is NOT applied to this shader</h3>
/// <para>
/// Two forms were built and measured against a retained pre-fix baseline, both
/// same-session and order-reversed, 41 passes, twice each:
/// </para>
/// <list type="bullet">
///   <item><b>A</b> — a persistent <c>accPart</c> folded every 4 super-blocks, the form
///   that shipped on the dense families: accuracy 3.753E-07 -&gt; 1.703E-07 (2.20x),
///   cost <b>0.830x / 0.834x</b> at the real 26B gate/up shape (n=128, m=704, k=2816,
///   8 experts) and 1.052-1.083x at a deep-K one.</item>
///   <item><b>B</b> — the partial declared INSIDE the outer loop body and folded per
///   super-block, so its live range spans one iteration instead of the whole row:
///   <i>better</i> accuracy, 1.556E-07 (2.41x), and still <b>0.890x</b> at the 26B
///   shape.</item>
/// </list>
/// <para>
/// The deciding arm was neither: a <b>ballast</b> copy of the PRE-FIX kernel carrying one
/// extra live float and no two-level accumulation at all, seeded from a push constant so
/// the compiler cannot fold it away. It reproduced the whole effect — <b>0.850x</b> at
/// the 26B shape and +6.9% at deep-K. So the cost is <b>the register, not the
/// arithmetic</b>: the driver reports 48 -&gt; 49 VGPRs, this kernel has <b>zero LDS</b>
/// and is therefore purely occupancy-bound, and 48 sits on a wave64 allocation
/// granularity boundary. Every two-level form needs at least one extra live float, so
/// <b>no formulation can avoid it</b> — and by the same token the deep-K "gain" is a
/// register artifact too, not something the fix earned.
/// </para>
/// <para>
/// That is the same call the campaign already made on IQ4_NL (13% for 1.92x, refused),
/// on a worse trade: 11-17% at the shape that actually runs, for 2.2-2.4x on a 1e-7
/// term. The bound below therefore holds this family at its CURRENT accuracy, so it is
/// still guarded against drifting worse.
/// </para>
/// <para>
/// To re-derive any of it: restore <c>moe_indexed_matmul_q4_k_q8_1_pre545.comp</c> (and
/// the A/B/ballast variants) from this branch's history, and select one with
/// <c>DOTLLM_545_MOE_ARM=&lt;spv base name&gt;</c>. The cost test skips itself when the
/// baseline SPIR-V is absent, which it now is.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe545MoeQ4KMmqTests
{
    private readonly ITestOutputHelper _out;
    public Probe545MoeQ4KMmqTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Absolute relative-RMS bound. Absolute rather than a ratio against the MMVQ
    /// sibling: #549 established that which side is weaker is per-quant, so a
    /// ratio gate is satisfied whenever both kernels regress together.
    /// <para>
    /// This holds the family at its CURRENT, deliberately unfixed accuracy
    /// (1.535E-07 at k=2048, 3.753E-07 at k=8192) — see the class remarks for why
    /// the fix was rejected. Sized just above the deep-K figure, so the family is
    /// still guarded against getting worse; had the fix been taken the bound would
    /// be 1.85e-7 (measured post-fix 1.535E-07 / 1.703E-07 for form A,
    /// 1.120E-07 / 1.556E-07 for form B).
    /// </para>
    /// </summary>
    private const double MmqRelBound = 4.00e-7;

    private static bool PreFixArm =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_545_PRE_FIX"), "1", StringComparison.Ordinal);

    private static string Shader =>
        Environment.GetEnvironmentVariable("DOTLLM_545_MOE_ARM") is { Length: > 0 } arm ? arm
        : PreFixArm ? "moe_indexed_matmul_q4_k_q8_1_pre545" : "moe_indexed_matmul_q4_k_q8_1";

    [SkippableFact]
    public void MoeQ4K_Mmq_ReductionDepth()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No VK_KHR_shader_integer_dot_product.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var kernel = MoeIndexedMatmulQ4KMmqKernel.TryCreate(device, spvDir, Shader)
            ?? throw new Xunit.Sdk.XunitException($"{Shader}.spv missing or unsupported.");

        var sb = new StringBuilder();
        sb.AppendLine($"moe_indexed Q4_K MMQ on {device.DeviceName}  [{Shader}]");

        double worst = 0;
        // k=2048 against k=8192: the gain from a depth fix must be LARGEST at the
        // deepest reduction, which is what distinguishes it from a coincidence.
        foreach ((int m, int k) in new[] { (2048, 2048), (2048, 8192) })
        {
            const int n = 1, numExperts = 1;
            var rng = new Random(0x545E + m * 31 + k * 7);
            float[] bankF32 = Q4KFixture.RandomFloats(rng, numExperts * m * k, range: 0.1f);
            float[] x = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);
            byte[] bank = Q4KFixture.QuantizeRows(bankF32, numExperts * m, k);
            int[] indices = new int[n];   // every row -> expert 0

            using var bankBuf = device.Allocate(((long)bank.Length + 3) & ~3L);
            using var xBuf = device.Allocate((long)x.Length * sizeof(float));
            using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
            using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
            using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
            using var yBuf = device.Allocate((long)n * m * sizeof(float));

            device.Upload(new ReadOnlySpan<byte>(bank), bankBuf);
            device.Upload(x, xBuf);
            device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);

            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quant.Record(ctx.CommandBuffer, xBuf, xqBuf, xdsBuf, n, k);
                KernelSupport.ComputeToComputeBarrier(ctx.CommandBuffer);
                kernel.Record(ctx.CommandBuffer, bankBuf, xqBuf, xdsBuf, idxBuf, yBuf, m, k, n, numExperts);
                ctx.SubmitAndWait();
            }

            var got = new float[(long)n * m];
            device.Download(yBuf, got);

            // The oracle consumes the activation the GPU actually produced.
            var xqWords = new float[QuantizeQ8_1RowsKernel.PackedBytes(n, k) / sizeof(float)];
            var xds = new float[QuantizeQ8_1RowsKernel.ScaleBytes(n, k) / sizeof(float)];
            device.Download(xqBuf, xqWords);
            device.Download(xdsBuf, xds);

            double[] xDeq = KQuantMmqOracle.DequantizeQ8_1Activation(xqWords, xds, k);
            float[] wDeq = KQuantMmqOracle.DequantizeWeights(QuantFamily.Q4_K, bank, m, k);
            double[] oracle = KQuantMmqOracle.Dot(wDeq, xDeq, m, k);

            double se = 0, on = 0;
            for (int i = 0; i < m; i++)
            {
                double d = got[i] - oracle[i];
                se += d * d;
                on += oracle[i] * oracle[i];
            }
            double rel = Math.Sqrt(se / m) / Math.Sqrt(on / m);
            worst = Math.Max(worst, rel);
            sb.AppendLine($"   m={m,5} k={k,5}  |oracle| rms={Math.Sqrt(on / m):E3}   MMQ rel={rel:E3}");
        }

        _out.WriteLine(sb.ToString());

        // No assertion when an explicit arm is selected: those shaders are the
        // rejected candidates, measured for the record rather than gated.
        if (PreFixArm || Environment.GetEnvironmentVariable("DOTLLM_545_MOE_ARM") is { Length: > 0 })
        {
            sb.AppendLine($"   [arm {Shader}] bound {MmqRelBound:E3} "
                        + (worst >= MmqRelBound ? "exceeded" : "within"));
            _out.WriteLine(sb.ToString());
            return;
        }

        Assert.True(worst < MmqRelBound,
            $"MoE Q4_K MMQ prefill accuracy has regressed: worst relative rms {worst:E3} "
            + $">= {MmqRelBound:E3}." + Environment.NewLine + sb);
    }

    // ─────────────────────────────────────────────────────────────
    // Prefill cost — same-session, order-reversed against the retained baseline
    // ─────────────────────────────────────────────────────────────

    private const int WarmupPasses = 2;
    private const int Batch = 4;

    private static int Passes =>
        int.TryParse(Environment.GetEnvironmentVariable("DOTLLM_545_BENCH_PASSES"), out int p) && p > 0 ? p : 9;

    /// <summary>
    /// Both arms held open in ONE process and order-reversed per pass: this box has
    /// produced 2-3x phantom deltas from cold-vs-warm process launches alone.
    /// Enable with <c>DOTLLM_545_BENCH=1</c>.
    /// </summary>
    [SkippableFact]
    public void MoeQ4K_Mmq_PrefillCost()
    {
        Skip.IfNot(string.Equals(Environment.GetEnvironmentVariable("DOTLLM_545_BENCH"), "1", StringComparison.Ordinal),
            "DOTLLM_545_BENCH=1 to enable.");
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No VK_KHR_shader_integer_dot_product.");
        Skip.IfNot(File.Exists(Path.Combine(spvDir, "moe_indexed_matmul_q4_k_q8_1_pre545.spv")),
            "moe_indexed_matmul_q4_k_q8_1_pre545.spv absent — baseline retired.");

        using var quant = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");
        using var baseline = MoeIndexedMatmulQ4KMmqKernel.TryCreate(device, spvDir, "moe_indexed_matmul_q4_k_q8_1_pre545")!;
        using var candidate = MoeIndexedMatmulQ4KMmqKernel.TryCreate(device, spvDir, Shader)!;

        var sb = new StringBuilder();
        sb.AppendLine($"moe_indexed Q4_K MMQ on {device.DeviceName}  candidate=[{Shader}]");
        sb.AppendLine($"Passes={Passes} (median, min-max), Batch={Batch} dispatches/pass, order reversed per pass.");
        sb.AppendLine("| shape (n,m,k,experts) | baseline us med [IQR] (min-max) | #545 us med [IQR] (min-max) | speedup (median) |");
        sb.AppendLine("|---|---:|---:|---:|");

        // The real 26B gate/up shape the correctness test uses (K=2816, Ie=704) plus a
        // deep-K one, and a prefill-width row count so this is not a decode shape.
        foreach ((int n, int m, int k, int experts) in new[] { (128, 704, 2816, 8), (128, 2048, 8192, 4) })
        {
            var rng = new Random(0x545F + n + m * 7 + k * 13);
            float[] bankF32 = Q4KFixture.RandomFloats(rng, experts * m * k, range: 0.1f);
            float[] x = Q4KFixture.RandomFloats(rng, n * k, range: 1.0f);
            byte[] bank = Q4KFixture.QuantizeRows(bankF32, experts * m, k);
            var indices = new int[n];
            for (int i = 0; i < n; i++) indices[i] = i % experts;

            using var bankBuf = device.Allocate(((long)bank.Length + 3) & ~3L);
            using var xBuf = device.Allocate((long)x.Length * sizeof(float));
            using var xqBuf = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(n, k));
            using var xdsBuf = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(n, k));
            using var idxBuf = device.Allocate((long)indices.Length * sizeof(int));
            using var yBuf = device.Allocate((long)n * m * sizeof(float));

            device.Upload(new ReadOnlySpan<byte>(bank), bankBuf);
            device.Upload(x, xBuf);
            device.Upload(MemoryMarshal.AsBytes<int>(indices), idxBuf);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quant.Record(ctx.CommandBuffer, xBuf, xqBuf, xdsBuf, n, k);
                ctx.SubmitAndWait();   // common to both arms; must not be timed
            }

            double Time(bool isBaseline)
            {
                var kern = isBaseline ? baseline : candidate;
                var sw = Stopwatch.StartNew();
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                for (int i = 0; i < Batch; i++)
                    kern.Record(ctx.CommandBuffer, bankBuf, xqBuf, xdsBuf, idxBuf, yBuf, m, k, n, experts);
                ctx.SubmitAndWait();
                sw.Stop();
                return sw.Elapsed.TotalMilliseconds * 1000.0 / Batch;
            }

            for (int w = 0; w < WarmupPasses; w++) { Time(true); Time(false); }

            var baseUs = new List<double>();
            var newUs = new List<double>();
            for (int p = 0; p < Passes; p++)
            {
                if ((p & 1) == 0) { baseUs.Add(Time(true)); newUs.Add(Time(false)); }
                else { newUs.Add(Time(false)); baseUs.Add(Time(true)); }
            }
            baseUs.Sort(); newUs.Sort();
            int np = baseUs.Count;
            double bMed = baseUs[np / 2], nMed = newUs[np / 2];
            double bQ1 = baseUs[np / 4], bQ3 = baseUs[np * 3 / 4];
            double nQ1 = newUs[np / 4], nQ3 = newUs[np * 3 / 4];
            sb.AppendLine(string.Create(CultureInfo.InvariantCulture,
                $"| ({n},{m},{k},{experts}) | {bMed:F1} [{bQ1:F1}-{bQ3:F1}] ({baseUs[0]:F1}-{baseUs[^1]:F1}) "
                + $"| {nMed:F1} [{nQ1:F1}-{nQ3:F1}] ({newUs[0]:F1}-{newUs[^1]:F1}) | {bMed / nMed:F3}x |"));
        }

        if (device.HasShaderInfoAmd)
        {
            foreach ((string label, nint pipe) in new[]
                     { ("pre545", baseline.PipelineHandle), ("shipped", candidate.PipelineHandle) })
            {
                var st = device.GetShaderStatisticsAmd(pipe);
                sb.AppendLine($"{label}: VGPRs {st.resourceUsage.numUsedVgprs}/{st.numAvailableVgprs}  "
                            + $"LDS {st.resourceUsage.ldsUsageSizeInBytes}B  scratch {st.resourceUsage.scratchMemUsageInBytes}B");
            }
        }

        _out.WriteLine(sb.ToString());
    }
}
