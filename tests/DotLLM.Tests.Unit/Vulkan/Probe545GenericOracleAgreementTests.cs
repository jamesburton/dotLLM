using DotLLM.Vulkan;
using DotLLM.Vulkan.Kernels;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Pre-flight for <see cref="KQuantMmqOracle"/>: the family-agnostic oracle must
/// reproduce the bespoke Q4_K oracle that #549 already validated against MMVQ.
/// </summary>
/// <remarks>
/// <para>
/// #545 scores seven quant families on one harness. If that harness is wrong it
/// is wrong for <b>every family at once</b>, and each family's own
/// "validate against MMVQ first" step would then be comparing two things that
/// share the same mistake. So the generic oracle is anchored to the one oracle
/// already known-good.
/// </para>
/// <para>
/// The bar is <b>1e-13 relative, not 1e-7</b>. Both are f64 evaluations of the
/// same arithmetic over the same bytes — dequantize-then-dot versus
/// scale-times-integer-dot are algebraically identical rearrangements — so they
/// should agree to near machine epsilon accumulated over K terms. A 1e-7
/// agreement would mean one of them is doing f32 work somewhere, which is
/// exactly the footing error this check exists to catch.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class Probe545GenericOracleAgreementTests
{
    private const double AgreementRelBound = 1e-13;

    private readonly ITestOutputHelper _out;
    public Probe545GenericOracleAgreementTests(ITestOutputHelper output) => _out = output;

    [SkippableTheory]
    [InlineData(512, 2048)]
    [InlineData(256, 8192)]
    public void GenericOracle_ReproducesTheValidatedQ4KOracle(int m, int k)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasIntegerDotProduct, "No VK_KHR_shader_integer_dot_product.");

        using var quantRows = QuantizeQ8_1RowsKernel.TryCreate(device, spvDir)
            ?? throw new Xunit.Sdk.XunitException("quantize_q8_1_rows.spv missing.");

        var rng = new Random(0x545A + m + k);
        byte[] weightsQ4K = Q4KFixture.QuantizeRows(Q4KFixture.RandomFloats(rng, m * k, 0.1f), m, k);
        float[] x = Q4KFixture.RandomFloats(rng, k, 1.0f);

        // The activation must come from the GPU quantizer, as both oracles consume it.
        using var bufX = device.Allocate((long)k * sizeof(float));
        using var bufXq = device.Allocate(QuantizeQ8_1RowsKernel.PackedBytes(1, k));
        using var bufXds = device.Allocate(QuantizeQ8_1RowsKernel.ScaleBytes(1, k));
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

        // A: the bespoke Q4_K oracle validated in #549 (scale x integer-dot, per sub-block).
        double[] bespoke = Probe545Q4KReductionTests.OracleF64ForCrossCheck(weightsQ4K, xqWords, xds, m, k);

        // B: the generic oracle (dequantize weights, dequantize activation, f64 dot).
        double[] xF = KQuantMmqOracle.DequantizeQ8_1Activation(xqWords, xds, k);
        float[] wF = KQuantMmqOracle.DequantizeWeights(QuantFamily.Q4_K, weightsQ4K, m, k);
        double[] generic = KQuantMmqOracle.Dot(wF, xF, m, k);

        double se = 0, os = 0, mx = 0;
        for (int i = 0; i < m; i++)
        {
            double d = Math.Abs(generic[i] - bespoke[i]);
            se += d * d; os += bespoke[i] * bespoke[i];
            mx = Math.Max(mx, d);
        }
        double rel = Math.Sqrt(se / m) / Math.Sqrt(os / m);

        _out.WriteLine($"m={m} k={k}  generic vs bespoke Q4_K oracle: relative rms={rel:E3} maxAbs={mx:E3}");

        Assert.True(rel < AgreementRelBound,
            $"the family-agnostic oracle disagrees with the validated Q4_K oracle at m={m} k={k} "
            + $"(relative rms {rel:E3} >= {AgreementRelBound:E0}). These are two f64 evaluations of "
            + "the same arithmetic, so they must agree to near machine epsilon; a ~1e-7 disagreement "
            + "means one of them does f32 work or has a scale/min-term footing wrong. Do NOT measure "
            + "any family with the generic oracle until this passes — it would be wrong for all of "
            + "them at once.");
    }
}
