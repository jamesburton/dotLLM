using DotLLM.Cpu.Kernels;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Family-agnostic f64 reference for a quantized MMQ / MMVQ matmul, used by
/// issue #545 to score every quant family's prefill GEMM on one harness.
/// </summary>
/// <remarks>
/// <para>
/// #549 scored Q4_K with a bespoke unpack that re-implemented the block layout.
/// That does not scale to seven more families and is seven more chances to get a
/// min-term sign or a scale footing wrong. This computes the same quantity a
/// different way: <b>dequantize the weights, dequantize the activation, dot in
/// double</b>. Both GPU kernels compute exactly <c>sum_i w_i * x_i</c> — their
/// per-block scale/offset arithmetic is that sum rearranged — so a dequantized
/// f64 dot is the correct oracle for any family, and the only thing that varies
/// between it and either kernel is the order and precision of the summation.
/// </para>
/// <para>
/// <b>It is validated against #549's bespoke Q4_K oracle before use</b>
/// (<c>Probe545GenericOracleAgreementTests</c>), to ~1e-15 relative rather than
/// ~1e-7: both are f64 evaluations of the same arithmetic, so near-exact
/// agreement is the bar. A family-agnostic oracle that is wrong is wrong for
/// every family at once, which is why that check exists before any family is
/// measured with it.
/// </para>
/// <para>
/// Dequantization uses the <b>scalar</b> CPU tier where the family offers one.
/// Dequantization is per-element with no reduction, so the tier should not
/// matter — but "should not" is the word that has cost this project weeks
/// (Q3_K's dequant was transposed in every backend for months), so it is pinned
/// rather than assumed.
/// </para>
/// </remarks>
internal static unsafe class KQuantMmqOracle
{
    private const int Q8_1GroupSize = 32;

    /// <summary>
    /// Unpacks the Q8_1 activation the GPU quantizer produced. Takes the packed
    /// words and <c>(d, s)</c> pairs <b>downloaded from the device</b> — never a
    /// CPU re-quantization of the f32 input, which would inject rounding that is
    /// not the kernel's.
    /// </summary>
    public static double[] DequantizeQ8_1Activation(float[] xqWords, float[] xds, int k)
    {
        var xInt = new sbyte[k];
        for (int w = 0; w < xqWords.Length; w++)
        {
            int bits = BitConverter.SingleToInt32Bits(xqWords[w]);
            for (int b = 0; b < 4; b++)
            {
                int idx = w * 4 + b;
                if (idx < k) xInt[idx] = (sbyte)((bits >> (b * 8)) & 0xFF);
            }
        }
        var x = new double[k];
        for (int i = 0; i < k; i++) x[i] = xds[(i / Q8_1GroupSize) * 2] * (double)xInt[i];
        return x;
    }

    /// <summary>
    /// <c>y[row] = sum_i w[row, i] * x[i]</c> accumulated in <see cref="double"/>,
    /// over row-major dequantized weights.
    /// </summary>
    public static double[] Dot(float[] dequantizedWeights, double[] x, int m, int k)
    {
        var y = new double[m];
        for (int row = 0; row < m; row++)
        {
            long baseIdx = (long)row * k;
            double sum = 0.0;
            for (int i = 0; i < k; i++)
                sum += (double)dequantizedWeights[baseIdx + i] * x[i];
            y[row] = sum;
        }
        return y;
    }

    /// <summary>
    /// Dequantizes <paramref name="quantized"/> (row-major, <paramref name="m"/> x
    /// <paramref name="k"/>) to f32 using the named family's scalar CPU path.
    /// </summary>
    public static float[] DequantizeWeights(QuantFamily family, byte[] quantized, int m, int k)
    {
        var dest = new float[(long)m * k];
        fixed (byte* src = quantized)
        {
            nint p = (nint)src;
            long n = (long)m * k;
            switch (family)
            {
                case QuantFamily.Q2_K: Dequantize.DequantizeQ2_K(p, n, dest); break;
                case QuantFamily.Q3_K: Dequantize.DequantizeQ3_KScalar(p, n, dest); break;
                case QuantFamily.Q4_K: Dequantize.DequantizeQ4_KScalar(p, n, dest); break;
                case QuantFamily.Q5_K: Dequantize.DequantizeQ5_KScalar(p, n, dest); break;
                case QuantFamily.Q6_K: Dequantize.DequantizeQ6_KScalar(p, n, dest); break;
                case QuantFamily.IQ4_NL: Dequantize.DequantizeIQ4_NL(p, n, dest); break;
                case QuantFamily.IQ4_XS: Dequantize.DequantizeIQ4_XS(p, n, dest); break;
                default: throw new ArgumentOutOfRangeException(nameof(family), family, null);
            }
        }
        return dest;
    }

    /// <summary>Relative RMS and max absolute deviation of <paramref name="got"/> from the oracle.</summary>
    public static (double relRms, double maxAbs, double scale) Score(double[] oracle, float[] got)
    {
        double se = 0, mx = 0, os = 0;
        for (int i = 0; i < oracle.Length; i++)
        {
            double d = Math.Abs(got[i] - oracle[i]);
            se += d * d;
            os += oracle[i] * oracle[i];
            mx = Math.Max(mx, d);
        }
        double scale = Math.Sqrt(os / oracle.Length);
        return (Math.Sqrt(se / oracle.Length) / scale, mx, scale);
    }
}

/// <summary>Quant families whose MMQ prefill GEMM #545 covers. Public so it can be a xUnit theory parameter.</summary>
public enum QuantFamily
{
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K,
    IQ4_NL,
    IQ4_XS,
}
