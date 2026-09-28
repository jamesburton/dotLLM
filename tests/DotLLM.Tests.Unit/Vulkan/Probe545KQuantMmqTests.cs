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

    private readonly ITestOutputHelper _out;
    public Probe545KQuantMmqTests(ITestOutputHelper output) => _out = output;

    public static TheoryData<QuantFamily> Families => new()
    {
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
        QuantFamily.Q6_K => 1.35e-7,    // post-fix max 1.181E-07, pre-fix 2.881E-07
        QuantFamily.Q5_K => 1.80e-7,    // post-fix max 1.587E-07, pre-fix 3.276E-07
        QuantFamily.IQ4_NL => 1.75e-7,  // post-fix max 1.542E-07, pre-fix 2.956E-07
        QuantFamily.IQ4_XS => 1.40e-7,  // post-fix max 1.220E-07, pre-fix 2.948E-07
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
        sb.AppendLine($"{family} on {device.DeviceName}");

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

            Assert.True(relQ < MmqRelBound(family),
                $"{family} MMQ (prefill) accuracy has regressed at m={m} k={k}: relative rms "
                + $"{relQ:E3} >= {MmqRelBound(family):E3}." + Environment.NewLine + sb);
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
            case QuantFamily.Q5_K:
            {
                using var v = MatMulQ5KMmvqKernel.TryCreate(device, spvDir);
                using var q = MatMulQ5KMmqKernel.TryCreate(device, spvDir);
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
                using var q = MatMulQ6KMmqKernel.TryCreate(device, spvDir);
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
                using var q = MatMulIq4NlMmqKernel.TryCreate(device, spvDir);
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
                using var q = MatMulIq4XsMmqKernel.TryCreate(device, spvDir);
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
