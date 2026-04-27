using System.Runtime.CompilerServices;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only F16 / BF16 fixture helpers shared by the F16 / BF16 GEMV / GEMM
/// kernel tests.
/// </summary>
/// <remarks>
/// <para>
/// Why this exists: production loaders ingest F16 / BF16 SafeTensors weights
/// directly; there is no F32-to-F16 / F32-to-BF16 quantiser in the production
/// codebase since each conversion is a one-line cast (<c>(Half)f</c> /
/// shift-right-16-with-rounding) — but parity tests need to manufacture
/// arbitrary FP32 fixtures and serialise them into the on-device byte layout
/// the new kernels read.
/// </para>
/// <para>
/// All conversions here match the kernel's inverse exactly so the kernel sees
/// the same bit pattern that a real loader would deliver:
/// <list type="bullet">
///   <item><description><b>F16</b>: round-to-nearest-even via <see cref="Half"/>'s
///     <c>(Half)f</c> cast — what every CPU dequant path uses.</description></item>
///   <item><description><b>BF16</b>: round-to-nearest-even via the standard
///     "round_to_bfloat16" trick: add 0x7FFF to the F32 bit pattern (with
///     the round-to-even tiebreak), then take the top 16 bits. This matches
///     <c>torch.tensor(x, dtype=torch.bfloat16)</c> bit-for-bit.</description></item>
/// </list>
/// </para>
/// </remarks>
internal static unsafe class F16Bf16Fixture
{
    /// <summary>Generate a uniform-random FP32 array in [-range, range).</summary>
    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    // ─────────────────────────────────────────────────────────────
    //  F16
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Convert a row-major <c>[m, k]</c> FP32 matrix to F16 bytes (2 bytes /
    /// element). Round-to-nearest-even via <see cref="Half"/>. Output stride:
    /// <c>k * 2</c> bytes per row.
    /// </summary>
    public static byte[] QuantizeRowsF16(float[] src, int m, int k)
    {
        if (src.Length < (long)m * k)
            throw new ArgumentException($"src too short: {src.Length} < {(long)m * k}", nameof(src));
        var dst = new byte[(long)m * k * 2];
        fixed (float* pSrc = src)
        fixed (byte* pDst = dst)
        {
            float* s = pSrc;
            Half* d = (Half*)pDst;
            long n = (long)m * k;
            for (long i = 0; i < n; i++) d[i] = (Half)s[i];
        }
        return dst;
    }

    /// <summary>
    /// Decode F16 bytes back to FP32 — the kernel-equivalent oracle. Matches
    /// what the GPU shader produces when it reads the same bytes through
    /// <c>unpackHalf2x16</c> (IEEE-754 binary16 to binary32 — exact, no
    /// rounding, since F32 is a strict superset of F16).
    /// </summary>
    public static float[] DecodeF16(byte[] f16Bytes, int m, int k)
    {
        var dst = new float[(long)m * k];
        fixed (byte* p = f16Bytes)
        fixed (float* d = dst)
        {
            Half* h = (Half*)p;
            long n = (long)m * k;
            for (long i = 0; i < n; i++) d[i] = (float)h[i];
        }
        return dst;
    }

    // ─────────────────────────────────────────────────────────────
    //  BF16
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Round-to-nearest-even FP32 to BF16 (top 16 bits of the F32 binary
    /// representation, with mantissa rounded). Matches PyTorch's
    /// <c>tensor.to(bfloat16)</c> and the <c>at::native::round_to_nearest_even</c>
    /// helper — the canonical bit-level reference.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ushort RoundFloatToBf16(float f)
    {
        // NaN handling: preserve a quiet-NaN bit pattern (top of mantissa set).
        // This matches PyTorch / TF / JAX. Real-model weights never contain NaN
        // so this is mostly defensive.
        uint bits = (uint)BitConverter.SingleToInt32Bits(f);
        if ((bits & 0x7FFFFFFFu) > 0x7F800000u)
        {
            // NaN: top 16 bits with quiet-bit forced.
            return (ushort)((bits >> 16) | 0x40u);
        }
        // Round to nearest even: add 0x7FFF + (bit 16) for ties-to-even.
        uint rounding = 0x7FFFu + ((bits >> 16) & 1u);
        bits += rounding;
        return (ushort)(bits >> 16);
    }

    /// <summary>
    /// Reconstruct FP32 from a BF16 16-bit pattern: shift left 16, reinterpret
    /// as F32. Lossless reverse of <see cref="RoundFloatToBf16"/> on the
    /// quantised value (the rounding error is intrinsic to BF16, not the
    /// reverse path).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float Bf16ToFloat(ushort bits)
    {
        uint u = ((uint)bits) << 16;
        return BitConverter.Int32BitsToSingle((int)u);
    }

    /// <summary>
    /// Convert a row-major <c>[m, k]</c> FP32 matrix to BF16 bytes
    /// (2 bytes / element, little-endian — top half of the F32 bit pattern).
    /// </summary>
    public static byte[] QuantizeRowsBf16(float[] src, int m, int k)
    {
        if (src.Length < (long)m * k)
            throw new ArgumentException($"src too short: {src.Length} < {(long)m * k}", nameof(src));
        var dst = new byte[(long)m * k * 2];
        fixed (byte* pDst = dst)
        {
            ushort* d = (ushort*)pDst;
            long n = (long)m * k;
            for (long i = 0; i < n; i++) d[i] = RoundFloatToBf16(src[i]);
        }
        return dst;
    }

    /// <summary>
    /// Decode BF16 bytes back to FP32 — the kernel-equivalent oracle. The
    /// shader reconstructs F32 via the identical shift-left-16 path, so the
    /// values are bit-exact between CPU oracle and GPU output (excluding
    /// per-thread reduction order drift).
    /// </summary>
    public static float[] DecodeBf16(byte[] bf16Bytes, int m, int k)
    {
        var dst = new float[(long)m * k];
        fixed (byte* p = bf16Bytes)
        fixed (float* d = dst)
        {
            ushort* h = (ushort*)p;
            long n = (long)m * k;
            for (long i = 0; i < n; i++) d[i] = Bf16ToFloat(h[i]);
        }
        return dst;
    }

    // ─────────────────────────────────────────────────────────────
    //  Reference matmul kernels (CPU oracle)
    // ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Scalar CPU GEMV reference: reads the same F16 bytes the GPU sees,
    /// dequantises on the fly, dots against FP32 <c>x</c>. Block-sequential
    /// reduction order — the GPU shader uses a workgroup tree reduce, which
    /// produces small-but-nonzero drift covered by the test tolerances.
    /// </summary>
    public static float[] CpuGemvF16(byte[] weightsF16, float[] x, int m, int k)
    {
        var result = new float[m];
        fixed (byte* wPtr = weightsF16)
        {
            for (int row = 0; row < m; row++)
            {
                Half* rowSrc = (Half*)wPtr + (long)row * k;
                float sum = 0;
                for (int j = 0; j < k; j++)
                    sum += (float)rowSrc[j] * x[j];
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>
    /// Scalar CPU GEMM reference for F16: <c>C[N, M] = B[N, K] @ W_f16[M, K]^T</c>.
    /// </summary>
    public static float[] CpuGemmF16(byte[] weightsF16, float[] inputB, int m, int k, int n)
    {
        var result = new float[n * m];
        fixed (byte* wPtr = weightsF16)
        {
            for (int t = 0; t < n; t++)
            {
                int bRowBase = t * k;
                for (int row = 0; row < m; row++)
                {
                    Half* rowSrc = (Half*)wPtr + (long)row * k;
                    float sum = 0;
                    for (int j = 0; j < k; j++)
                        sum += (float)rowSrc[j] * inputB[bRowBase + j];
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
    }

    /// <summary>Scalar CPU GEMV for BF16. Same shape as F16 helper.</summary>
    public static float[] CpuGemvBf16(byte[] weightsBf16, float[] x, int m, int k)
    {
        var result = new float[m];
        fixed (byte* wPtr = weightsBf16)
        {
            for (int row = 0; row < m; row++)
            {
                ushort* rowSrc = (ushort*)wPtr + (long)row * k;
                float sum = 0;
                for (int j = 0; j < k; j++)
                    sum += Bf16ToFloat(rowSrc[j]) * x[j];
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>Scalar CPU GEMM for BF16.</summary>
    public static float[] CpuGemmBf16(byte[] weightsBf16, float[] inputB, int m, int k, int n)
    {
        var result = new float[n * m];
        fixed (byte* wPtr = weightsBf16)
        {
            for (int t = 0; t < n; t++)
            {
                int bRowBase = t * k;
                for (int row = 0; row < m; row++)
                {
                    ushort* rowSrc = (ushort*)wPtr + (long)row * k;
                    float sum = 0;
                    for (int j = 0; j < k; j++)
                        sum += Bf16ToFloat(rowSrc[j]) * inputB[bRowBase + j];
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Asserts every cell is within either abs <paramref name="absTol"/> or
    /// rel <paramref name="relTol"/> of the expected value — same shape as
    /// <see cref="Q4KFixture.AssertClose"/>.
    /// </summary>
    public static void AssertClose(float[] expected, float[] actual, int m, int k,
        float absTol, float relTol)
    {
        Assert.Equal(expected.Length, actual.Length);
        int errors = 0;
        float maxAbs = 0, maxRel = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float e = expected[i];
            float a = actual[i];
            float diff = MathF.Abs(e - a);
            float rel = diff / MathF.Max(MathF.Abs(e), 1e-7f);
            if (diff > maxAbs) maxAbs = diff;
            if (rel > maxRel) maxRel = rel;
            if (diff > absTol && rel > relTol) errors++;
        }
        Assert.True(errors == 0,
            $"Numerical drift exceeded tolerance (m={m},k={k}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
