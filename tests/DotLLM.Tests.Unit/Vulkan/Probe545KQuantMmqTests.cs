using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using System.Text;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Issue #545 — reduction depth in the K-quant / IQ4 prefill GEMMs, scored on
/// the family-agnostic f64 oracle.
/// </summary>
/// <remarks>
/// <para>
/// #544 (Q8_0) and #549 (Q4_K) both found the MMQ prefill GEMM accumulating all
/// K-blocks sequentially while the MMVQ decode GEMV finishes with a
/// <c>subgroupAdd</c> tree, and both found the two-level fix free or faster.
/// This covers the four remaining families that are reachable on a default
/// dispatch and share the pattern.
/// </para>
/// <para>
/// <b>Two things #549 established that this test is shaped around:</b>
/// </para>
/// <list type="number">
///   <item>The MMQ/MMVQ <i>baseline ratio</i> is not a screen. Q4_K's was
///     1.03-1.05 — nothing like Q8_0's 1.65-2.37, and not widening with K — and
///     the fix still bought 1.46-2.14x, because the ratio compares two imperfect
///     kernels and MMVQ has depth of its own. So each family is measured before
///     and after, not screened out on its ratio.</item>
///   <item>Which side is weaker is <i>per-quant</i>: after #549, Q4_K MMQ is
///     better than its MMVQ, the reverse of Q8_0. So the gate is an absolute
///     relative-RMS bound per family, never a ratio, which a ratio gate would let
///     through whenever both kernels regressed together.</item>
/// </list>
/// <para>
/// <b>IQ4_NL is measured here but deliberately left unfixed.</b> The two-level
/// accumulation buys it 1.92x accuracy but costs <b>13%</b> prefill throughput
/// (0.870 / 0.864 at (512,2048,2048) across two 41-pass runs on a quiet box).
/// The cause is occupancy and it is specific to this shader: it is by far the
/// lightest of the four — 68 VGPRs, 5120 B LDS, 32 SGPRs — so 16 extra live
/// floats are proportionally large, and 68 -> 83 VGPRs loses a wave slot, which
/// Q3_K is the largest gain in the family (2.71x at k=8192) and the cheapest per unit
/// of it: it has the deepest sequential run — 2 halves x 8 sub-blocks = 16 accumulates
/// per super-block, against Q6_K's 2 — and its VGPRs go 99 -> 93, so it is not paying
/// occupancy either. ~1.5-3% cost across two 41-pass rounds.
///
/// matches the measured loss almost exactly. Q6_K and IQ4_XS end up using FEWER
/// registers after the change and cost 2-5%, so their residual is the fold work
/// itself rather than occupancy; Q5_K is unchanged at 144 and is at parity.
/// Do not "finish the set" by applying the change to IQ4_NL without re-measuring
/// — the trade is genuinely worse there.
/// </para>
/// <para>
/// The oracle is <see cref="KQuantMmqOracle"/>, anchored to #549's bespoke Q4_K
/// oracle by <see cref="Probe545GenericOracleAgreementTests"/> — they agree
/// exactly, 0.000E+000 relative. Each family additionally validates it against
/// that family's own MMVQ before any conclusion is drawn.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe545KQuantMmqTests
{
    /// <summary>MMVQ must agree with the oracle this closely, or the oracle is wrong for this family.</summary>
    private const double OracleValidationRel = 1e-5;

    /// <summary>
    /// <c>DOTLLM_545_PRE_FIX=1</c> dispatches <c>{family}_mmq_pre545.spv</c> — the
    /// retained pre-two-level-accumulation shader — instead of the shipping module,
    /// so the RED half of this measurement is one command rather than a manual
    /// SPIR-V swap in <c>bin/</c>. The gate assertions are skipped in that mode:
    /// the point of the arm is to show the bound FAILS without the fix.
    /// <para>
    /// Worth having because the alternative is a swap that has to be undone, and a
    /// forgotten one leaves a stale module running the next measurement — which is
    /// exactly how a fixed-GPU-vs-stale-CPU comparison once produced a convincing
    /// fictitious bug in this repo.
    /// </para>
    /// </summary>
    private static bool PreFixArm =>
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_545_PRE_FIX"), "1", StringComparison.Ordinal);

    private static string MmqShader(QuantFamily f)
    {
        string baseName = f switch
        {
            QuantFamily.Q3_K => "matmul_q3_k_mmq",
            QuantFamily.Q5_K => "matmul_q5_k_mmq",
            QuantFamily.Q6_K => "matmul_q6_k_mmq",
            QuantFamily.IQ4_NL => "matmul_iq4_nl_mmq",
            QuantFamily.IQ4_XS => "matmul_iq4_xs_mmq",
            _ => throw new ArgumentOutOfRangeException(nameof(f)),
        };
        return PreFixArm ? baseName + "_pre545" : baseName;
    }

    private readonly ITestOutputHelper _out;
    public Probe545KQuantMmqTests(ITestOutputHelper output) => _out = output;

    public static TheoryData<QuantFamily> Families => new()
    {
        QuantFamily.Q3_K,     // Q3_K_M / Q3_K_S; also the family whose quant error is the
                              // project's known weak spot (2.9x llama.cpp's on #519/#520),
                              // so its prefill GEMM is the last place it should lose more
        QuantFamily.Q6_K,     // on the Q4_K_M shipping path (attn_v / ffn_down / output)
        QuantFamily.Q5_K,     // Q5_K_M
        QuantFamily.IQ4_NL,
        QuantFamily.IQ4_XS,
    };

    /// <summary>
    /// Per-family relative-RMS bound for the prefill GEMM, set just above the
    /// measured post-fix maximum.
    /// <para>
    /// Sized to catch a return to the pre-#545 accumulation at the <b>deep-K</b>
    /// shape, which is where the defect lives and where the gap is unambiguous
    /// (pre-fix 2.88-3.28E-07 against post-fix 1.18-1.59E-07). At k=2048 the
    /// pre-fix figures for some families already sat below any bound that the
    /// post-fix k=8192 figure permits, so a single per-family bound cannot police
    /// both shapes — rather than over-fit a per-shape table, the bound guards the
    /// case that matters and the test prints both rows.
    /// </para>
    /// <para>
    /// NOT a ratio against MMVQ. #549 established that which side is weaker is
    /// per-quant — here Q5_K MMQ is already better than its MMVQ (0.46-0.81x) —
    /// so a ratio gate would be satisfied by both kernels regressing together.
    /// </para>
    /// </summary>
    private static double MmqRelBound(QuantFamily f) => f switch
    {
        // Q3_K is the only family whose bound discriminates at BOTH shapes: pre-fix
        // 2.015E-07 (k=2048) and 4.093E-07 (k=8192) are both above it, where the other
        // families' k=2048 pre-fix figures sit below any bound their k=8192 post-fix
        // number permits. That follows from Q3_K having the deepest sequential run in
        // the family — 2 halves x 8 sub-blocks = 16 accumulates per super-block.
        QuantFamily.Q3_K => 1.70e-7,    // fixed: post 1.493E-07 / 1.512E-07, pre 2.015E-07 / 4.093E-07
        QuantFamily.Q6_K => 1.35e-7,    // fixed: post 1.181E-07, pre 2.881E-07
        QuantFamily.Q5_K => 1.80e-7,    // fixed: post 1.587E-07, pre 3.276E-07
        QuantFamily.IQ4_XS => 1.40e-7,  // fixed: post 1.220E-07, pre 2.948E-07

        // IQ4_NL is deliberately NOT fixed — see the class remarks. This bound
        // holds it at its CURRENT (unfixed) accuracy, 1.452E-07 / 2.956E-07, so
        // the family is still guarded against drifting worse.
        QuantFamily.IQ4_NL => 3.10e-7,
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    private static readonly (int m, int k)[] Shapes =
    [
        (2048, 2048),
        (2048, 8192),   // deepest reduction — where a depth fix must help most
    ];

    [SkippableTheory]
    [MemberData(nameof(Families))]
    public void KQuant_Mmq_ReductionDepth(QuantFamily family)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No VK_KHR_shader_integer_dot_product.");

        using var quantRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        var sb = new StringBuilder();
        sb.AppendLine($"{family} on {device.DeviceName}  [{MmqShader(family)}]");

        foreach ((int m, int k) in Shapes)
        {
            var rng = new Random(0x545 + (int)family * 977 + m * 31 + k);
            byte[] weights = QuantizeRows(family, rng, m, k);
            float[] x = RandomFloats(family, rng, k);

            using var bufW = device.Allocate(((long)weights.Length + 3) & ~3L);
            using var bufX = device.Allocate((long)k * sizeof(float));
            using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, k));
            using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, k));
            using var bufV = device.Allocate((long)m * sizeof(float));
            using var bufQ = device.Allocate((long)m * sizeof(float));

            device.Upload(new ReadOnlySpan<byte>(weights), bufW);
            device.Upload(x, bufX);
            using (var ctx = device.CreateSubmitContext())
            {
                ctx.Begin();
                quantRows.Record(ctx.CommandBuffer, bufX, bufXq, bufXds, 1, k);
                ctx.SubmitAndWait();
            }

            var xqWords = new float[QuantizeQ8_1RowsKernel.PackedBytes(1, k) / sizeof(float)];
            var xds = new float[QuantizeQ8_1RowsKernel.ScaleBytes(1, k) / sizeof(float)];
            device.Download(bufXq, xqWords);
            device.Download(bufXds, xds);

            bool ran = Dispatch(family, device, spvDir, bufW, bufXq, bufXds, bufV, bufQ, m, k);
            Skip.IfNot(ran, $"{family} MMQ/MMVQ spv missing or unsupported on this device.");

            var gotV = new float[m];
            var gotQ = new float[m];
            device.Download(bufV, gotV);
            device.Download(bufQ, gotQ);

            double[] xF = KQuantMmqOracle.DequantizeQ8_1Activation(xqWords, xds, k);
            float[] wF = KQuantMmqOracle.DequantizeWeights(family, weights, m, k);
            double[] oracle = KQuantMmqOracle.Dot(wF, xF, m, k);

            (double relV, double maxV, double scale) = KQuantMmqOracle.Score(oracle, gotV);
            (double relQ, double maxQ, _) = KQuantMmqOracle.Score(oracle, gotQ);

            sb.AppendLine($"  m={m,5} k={k,5}  |oracle| rms={scale:E3}");
            sb.AppendLine($"    MMVQ rel={relV:E3} maxAbs={maxV:E3}");
            sb.AppendLine($"    MMQ  rel={relQ:E3} maxAbs={maxQ:E3}   ratio(MMQ/MMVQ)={relQ / relV:F3}");

            // Oracle validation for THIS family, before anything is concluded from it.
            Assert.True(relV < OracleValidationRel,
                $"the f64 oracle does not agree with {family} MMVQ at m={m} k={k} "
                + $"(relative rms {relV:E3}). The ORACLE is the suspect, not the kernel — check the "
                + "dequant path for this family." + Environment.NewLine + sb);

            // In the pre-fix arm the bound is EXPECTED to be violated — that is the
            // RED this measurement exists to show — so record it instead of failing.
            if (PreFixArm)
            {
                // Shape-aware, because a non-violation at the SHALLOW shape is expected
                // for most families rather than alarming: the per-family bound is sized
                // to the deep-K case (see MmqRelBound's remarks), and several families'
                // k=2048 pre-fix figures sit below any bound their k=8192 post-fix figure
                // permits. Reporting both the same way would cry wolf — and the margins
                // are thin enough to matter: Q5_K's k=2048 pre-fix cleared its bound by
                // 6.8% while its k=8192 pre-fix reading moved 7.6% between sessions on a
                // different seed, so that row can flip without anything being wrong.
                bool deepest = k == Shapes[^1].k;
                string verdict = relQ >= MmqRelBound(family)
                    ? "violated as expected"
                    : deepest
                        ? "NOT violated at the DEEPEST shape — this family's gate does not "
                          + "discriminate the fix it is supposed to hold, which is a finding"
                        : "not violated (shallow shape; this bound guards deep-K only)";
                sb.AppendLine($"     [pre-fix arm] bound {MmqRelBound(family):E3} {verdict}");
            }
            else
            {
                Assert.True(relQ < MmqRelBound(family),
                    $"{family} MMQ (prefill) accuracy has regressed at m={m} k={k}: relative rms "
                    + $"{relQ:E3} >= {MmqRelBound(family):E3}." + Environment.NewLine + sb);
            }
        }

        _out.WriteLine(sb.ToString());
    }

    // ─────────────────────────────────────────────────────────────

    /// <summary>Exposed so the cost bench builds identical inputs.</summary>
    internal static float[] RandomFloatsFor(QuantFamily f, Random rng, int count) => RandomFloats(f, rng, count);

    /// <summary>Exposed so the cost bench builds identical inputs.</summary>
    internal static byte[] QuantizeRowsFor(QuantFamily f, Random rng, int m, int k) => QuantizeRows(f, rng, m, k);

    private static float[] RandomFloats(QuantFamily f, Random rng, int count) => f switch
    {
        QuantFamily.Q3_K => Q3KFixture.RandomFloats(rng, count, 1.0f),
        QuantFamily.Q5_K => Q5KFixture.RandomFloats(rng, count, 1.0f),
        QuantFamily.Q6_K => Q6KFixture.RandomFloats(rng, count, 1.0f),
        QuantFamily.IQ4_NL or QuantFamily.IQ4_XS => Iq4Fixture.RandomFloats(rng, count, 1.0f),
        _ => throw new ArgumentOutOfRangeException(nameof(f)),
    };

    private static byte[] QuantizeRows(QuantFamily f, Random rng, int m, int k)
    {
        float[] src = RandomFloats(f, rng, m * k);
        return f switch
        {
            QuantFamily.Q3_K => Q3KFixture.QuantizeRows(src, m, k),
            QuantFamily.Q5_K => Q5KFixture.QuantizeRows(src, m, k),
            QuantFamily.Q6_K => Q6KFixture.QuantizeRows(src, m, k),
            QuantFamily.IQ4_NL => Iq4Fixture.QuantizeRowsIq4Nl(src, m, k),
            QuantFamily.IQ4_XS => Iq4Fixture.QuantizeRowsIq4Xs(src, m, k),
            _ => throw new ArgumentOutOfRangeException(nameof(f)),
        };
    }

    /// <summary>Records both kernels for the family into one submit; false if either is unavailable.</summary>
    private static bool Dispatch(
        QuantFamily f, VulkanDevice device, string spvDir,
        VulkanDevice.Buffer w, VulkanDevice.Buffer xq, VulkanDevice.Buffer xds,
        VulkanDevice.Buffer outV, VulkanDevice.Buffer outQ, int m, int k)
    {
        switch (f)
        {
            case QuantFamily.Q3_K:
            {
                using var v = MatMulQ3KMmvqKernel.TryCreate(device, spvDir);
                using var q = MatMulQ3KMmqKernel.TryCreate(device, spvDir, MmqShader(f));
                if (v is null || q is null) return false;
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                v.Record(ctx.CommandBuffer, w, xq, xds, outV, m, k);
                q.Record(ctx.CommandBuffer, w, xq, xds, outQ, m, k, 1);
                ctx.SubmitAndWait();
                return true;
            }
            case QuantFamily.Q5_K:
            {
                using var v = MatMulQ5KMmvqKernel.TryCreate(device, spvDir);
                using var q = MatMulQ5KMmqKernel.TryCreate(device, spvDir, MmqShader(f));
                if (v is null || q is null) return false;
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                v.Record(ctx.CommandBuffer, w, xq, xds, outV, m, k);
                q.Record(ctx.CommandBuffer, w, xq, xds, outQ, m, k, 1);
                ctx.SubmitAndWait();
                return true;
            }
            case QuantFamily.Q6_K:
            {
                using var v = MatMulQ6KMmvqKernel.TryCreate(device, spvDir);
                using var q = MatMulQ6KMmqKernel.TryCreate(device, spvDir, MmqShader(f));
                if (v is null || q is null) return false;
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                v.Record(ctx.CommandBuffer, w, xq, xds, outV, m, k);
                q.Record(ctx.CommandBuffer, w, xq, xds, outQ, m, k, 1);
                ctx.SubmitAndWait();
                return true;
            }
            case QuantFamily.IQ4_NL:
            {
                using var v = MatMulIq4NlMmvqKernel.TryCreate(device, spvDir);
                using var q = MatMulIq4NlMmqKernel.TryCreate(device, spvDir, MmqShader(f));
                if (v is null || q is null) return false;
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                v.Record(ctx.CommandBuffer, w, xq, xds, outV, m, k);
                q.Record(ctx.CommandBuffer, w, xq, xds, outQ, m, k, 1);
                ctx.SubmitAndWait();
                return true;
            }
            case QuantFamily.IQ4_XS:
            {
                using var v = MatMulIq4XsMmvqKernel.TryCreate(device, spvDir);
                using var q = MatMulIq4XsMmqKernel.TryCreate(device, spvDir, MmqShader(f));
                if (v is null || q is null) return false;
                using var ctx = device.CreateSubmitContext();
                ctx.Begin();
                v.Record(ctx.CommandBuffer, w, xq, xds, outV, m, k);
                q.Record(ctx.CommandBuffer, w, xq, xds, outQ, m, k, 1);
                ctx.SubmitAndWait();
                return true;
            }
            default: throw new ArgumentOutOfRangeException(nameof(f));
        }
    }
}
