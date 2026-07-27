using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Shared parity assertion for the IQ4 MMVQ kernel tests (issue #339): tolerance
/// check first (a structurally broken kernel fails here), then argmax-exact unless
/// the kernel picked a near-tie (activation int8 quant can flip a near-tied winner).
/// </summary>
internal static class Iq4MmvqParity
{
    public static void Assert(float[] expected, float[] actual, int m, int k, float relTol, string label)
    {
        Xunit.Assert.Equal(expected.Length, actual.Length);

        int argE = 0, argA = 0;
        for (int i = 1; i < m; i++)
        {
            if (expected[i] > expected[argE]) argE = i;
            if (actual[i] > actual[argA]) argA = i;
        }

        double ss = 0;
        for (int i = 0; i < m; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)System.Math.Sqrt(ss / System.Math.Max(1, m));
        float absTol = System.MathF.Max(rms, 1e-6f) * relTol;

        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < m; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = System.MathF.Abs(e - a);
            float rel = diff / System.MathF.Max(System.MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > relTol) errors++;
        }
        Xunit.Assert.True(errors == 0,
            $"{label} MMVQ drift exceeded tolerance (m={m},k={k}): errors={errors}/{m}, " +
            $"maxAbs={maxAbs:G9}, maxRel={maxRel:G9}, absTol={absTol:G9}.");

        bool nearTie = System.MathF.Abs(expected[argE] - expected[argA]) <= absTol;
        Xunit.Assert.True(argE == argA || nearTie,
            $"{label} argmax mismatch beyond near-tie (m={m},k={k}): oracle={argE} ({expected[argE]:G6}), " +
            $"mmvq={argA} (oracle there {expected[argA]:G6}), absTol={absTol:G9}.");
    }
}
