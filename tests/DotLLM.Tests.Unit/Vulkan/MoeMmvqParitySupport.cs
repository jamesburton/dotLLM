using System.Runtime.InteropServices;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Shared helpers for the MoE indexed MMVQ (dp4a decode GEMV) parity tests —
/// issue #309, covering <c>moe_indexed_matmul_q4_k_mmvq</c>,
/// <c>moe_indexed_matmul_q5_1_mmvq</c> and <c>moe_indexed_matmul_q8_0_mmvq</c>.
/// </summary>
/// <remarks>
/// <para>
/// The MMVQ kernels consume <b>int8 (Q8_1) activations</b>, so a CPU oracle fed
/// full-precision FP32 <c>x</c> is a DIFFERENT numerical tier and can only be
/// compared with a loose tolerance (the dense-MMVQ sibling tests use 3e-2).
/// That is weak enough to admit a genuinely broken kernel on small shapes.
/// </para>
/// <para>
/// So the primary oracle here is <b>same-tier</b>: after the quantize dispatch we
/// download the exact <c>xq</c>/<c>xds</c> bytes the shader read and feed those to
/// a scalar C# reimplementation of the shader's own integer-dot math. Both sides
/// then compute the same quantity in exact arithmetic; the only admissible
/// difference is fp32 accumulation order (the shader accumulates per-lane and
/// finishes with a <c>subgroupAdd</c>). That justifies a tolerance ~2 orders of
/// magnitude tighter than the cross-tier bound, per the repo's "check the CPU's
/// own tier before widening any parity bound" rule.
/// </para>
/// </remarks>
internal static class MoeMmvqParitySupport
{
    /// <summary>Elements per Q8_1 activation block.</summary>
    public const int Q8_1GroupSize = 32;

    /// <summary>
    /// Downloads the packed Q8_1 activations produced by
    /// <c>QuantizeQ8_1RowsKernel</c> and unpacks them to one signed byte per
    /// element (row-major, <paramref name="n"/> × <paramref name="k"/>).
    /// </summary>
    public static sbyte[] DownloadActivationBytes(VulkanDevice device, VulkanDevice.Buffer xq, int n, int k)
    {
        var words = new uint[(long)n * (k / 4)];
        device.Download(xq, MemoryMarshal.Cast<uint, float>(words.AsSpan()));

        var bytes = new sbyte[(long)n * k];
        for (int w = 0; w < words.Length; w++)
        {
            uint u = words[w];
            bytes[w * 4 + 0] = unchecked((sbyte)(u & 0xFFu));
            bytes[w * 4 + 1] = unchecked((sbyte)((u >> 8) & 0xFFu));
            bytes[w * 4 + 2] = unchecked((sbyte)((u >> 16) & 0xFFu));
            bytes[w * 4 + 3] = unchecked((sbyte)((u >> 24) & 0xFFu));
        }
        return bytes;
    }

    /// <summary>
    /// Downloads the per-Q8_1-block <c>(d_x, s)</c> pairs as a flat float array:
    /// index <c>2*(row*blocksPerRow + b)</c> is <c>d_x</c>, <c>+1</c> is the
    /// block sum <c>s = d_x · Σ xq</c>.
    /// </summary>
    public static float[] DownloadActivationScales(VulkanDevice device, VulkanDevice.Buffer xds, int n, int k)
    {
        var ds = new float[(long)n * (k / Q8_1GroupSize) * 2];
        device.Download(xds, ds);
        return ds;
    }

    /// <summary>Reads a little-endian IEEE-754 binary16 at <paramref name="byteOffset"/>.</summary>
    public static float ReadHalf(byte[] blob, long byteOffset) =>
        (float)BitConverter.UInt16BitsToHalf((ushort)(blob[byteOffset] | (blob[byteOffset + 1] << 8)));

    /// <summary>
    /// Per-row expert indices drawn from a pool of <paramref name="activePool"/>
    /// distinct experts. Callers must pass a pool &gt; 1 and <c>count</c> &gt; 1
    /// for the result to discriminate broadcast-style indexing bugs.
    /// </summary>
    public static int[] RandomIndices(Random rng, int count, int numExperts, int activePool)
    {
        int pool = Math.Min(activePool, numExperts);
        var unique = new HashSet<int>();
        while (unique.Count < pool)
            unique.Add(rng.Next(numExperts));
        var poolArr = unique.ToArray();
        var indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = poolArr[rng.Next(poolArr.Length)];

        // Guarantee the batch is genuinely mixed: a test where every row routes
        // to the same expert proves nothing about the per-row index lookup.
        if (pool > 1 && count > 1)
        {
            indices[0] = poolArr[0];
            indices[count - 1] = poolArr[^1];
        }
        return indices;
    }

    /// <summary>
    /// Tight same-tier parity: exact-arithmetic-identical quantities differing
    /// only by fp32 accumulation order.
    /// </summary>
    /// <param name="relTol">
    /// Relative bound. <c>2e-4</c> is ~1000× the fp32 epsilon-per-add (6e-8) and
    /// covers the worst realistic reordering drift over the longest K used here
    /// (2816 → 88 super-blocks), while still being 150× tighter than the
    /// cross-tier 3e-2 bound the dense MMVQ tests must use.
    /// </param>
    public static void AssertSameTierParity(
        float[] expected, float[] actual, int m, int n, string what, float relTol = 2e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);

        // Magnitude scale for the absolute floor: outputs near zero have huge
        // relative error from benign cancellation but negligible absolute error.
        double ss = 0;
        for (int i = 0; i < expected.Length; i++) ss += (double)expected[i] * expected[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, expected.Length));
        float absTol = MathF.Max(rms, 1e-6f) * relTol;

        int errors = 0, worst = -1;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i], a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) { maxAbs = diff; worst = i; }
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > relTol) errors++;
        }
        Assert.True(errors == 0,
            $"{what}: same-tier drift exceeded tolerance (m={m},n={n}): errors={errors}/{expected.Length}, " +
            $"maxAbs={maxAbs:G9} at [{worst}] (cpu={(worst >= 0 ? expected[worst] : 0):G9}, " +
            $"gpu={(worst >= 0 ? actual[worst] : 0):G9}), maxRel={maxRel:G9}, absTol={absTol:G9}.");
    }

    /// <summary>
    /// Argmax agreement against a full-precision (F32-activation) oracle — a
    /// cheap cross-tier sanity check on top of the tight same-tier bound: a
    /// structurally broken kernel moves the per-row argmax, int8 activation
    /// quantization (essentially) does not.
    /// </summary>
    public static void AssertArgmaxAgreement(float[] f32Oracle, float[] actual, int m, int n, string what)
    {
        double ss = 0;
        for (int i = 0; i < f32Oracle.Length; i++) ss += (double)f32Oracle[i] * f32Oracle[i];
        float rms = (float)Math.Sqrt(ss / Math.Max(1, f32Oracle.Length));
        float tieTol = MathF.Max(rms, 1e-6f) * 3e-2f;

        for (int row = 0; row < n; row++)
        {
            int argE = 0, argA = 0;
            for (int i = 1; i < m; i++)
            {
                if (f32Oracle[row * m + i] > f32Oracle[row * m + argE]) argE = i;
                if (actual[row * m + i] > actual[row * m + argA]) argA = i;
            }
            float oracleMax = f32Oracle[row * m + argE];
            float oracleAtArg = f32Oracle[row * m + argA];
            Assert.True(argE == argA || (oracleMax - oracleAtArg) <= tieTol,
                $"{what}: argmax mismatch (m={m},n={n}) row {row}: oracle={argE} ({oracleMax:G6}), " +
                $"mmvq={argA} (oracle@{argA}={oracleAtArg:G6}, gap={oracleMax - oracleAtArg:G6} > {tieTol:G6}).");
        }
    }
}
