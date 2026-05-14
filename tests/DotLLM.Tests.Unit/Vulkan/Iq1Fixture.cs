using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only IQ1_S fixture helpers shared by the dequant / GEMV / GEMM
/// kernel tests. Mirrors the IQ4 fixture's responsibilities: round-trip
/// arbitrary FP32 weights through IQ1_S bytes (production code only
/// dequantises) and provide CPU-scalar oracles for the matmul shapes.
/// </summary>
/// <remarks>
/// IQ1_S is the most aggressive GGUF quant — 1.5-1.7 bpw. The codebook is a
/// 2048-entry table of 8 packed signed-int8 ternary values ({-1, 0, +1}). The
/// fixture's quantiser does a best-effort fit:
///  - per-32-element sub-block: pick the 3-bit scale and sign-of-delta that
///    minimise reconstruction error against the full codebook search;
///  - per-8-element group: brute-force the 11-bit codebook index against the
///    sub-block's effective scale.
///
/// The round-trip drift is much higher than IQ4 (~25-50% relative) because the
/// codebook can only encode 1.5 bits per element. The fixture simply asserts
/// the drift is bounded; the kernel-under-test then asserts that the GPU and
/// CPU paths produce the same numbers from the same bytes (independent of how
/// good the fixture's quantisation is).
/// </remarks>
internal static unsafe class Iq1Fixture
{
    public const int Iq1SGroupSize = 256;
    public const int Iq1SBlockBytes = 50;
    public const int Iq1SNumSubBlocks = 8;
    public const int Iq1SSubBlockSize = 32;
    public const float Iq1SDelta = 0.125f;

    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// Quantise an [m, k] row-major FP32 matrix to IQ1_S bytes. k must be a
    /// multiple of 256.
    /// </summary>
    public static byte[] QuantizeRowsIq1S(float[] src, int m, int k)
    {
        if ((k % Iq1SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq1SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq1SGroupSize;
        int rowBytes = blocksPerRow * Iq1SBlockBytes;
        var dst = new byte[m * rowBytes];

        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
            {
                float* rowSrc = srcPtr + (long)row * k;
                byte* rowDst = dstPtr + (long)row * rowBytes;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    QuantizeSuperBlockIq1S(
                        rowSrc + b * Iq1SGroupSize,
                        rowDst + b * Iq1SBlockBytes);
                }
            }
        }
        return dst;
    }

    /// <summary>
    /// Quantises one IQ1_S super-block. Strategy:
    ///   1. Pick a single super-block d that scales the largest |sub-amax| / 7
    ///      to fit the dl-factor (2*7+1) = 15.
    ///   2. Per sub-block: pick 3-bit ls in [0..7] and sign-of-delta to minimise
    ///      sum((x_i - dl * (grid[best(i)] + delta))^2).
    ///   3. Per group of 8: brute-force the 2048-entry codebook for the index
    ///      that minimises ||x - dl * (grid + delta)||.
    /// This is intentionally simpler than ggml's reference quantiser — the
    /// fixture only needs the round-trip drift to be bounded, not optimal.
    /// </summary>
    private static void QuantizeSuperBlockIq1S(float* src, byte* dst)
    {
        ReadOnlySpan<ulong> grid = Dequantize.Iq1SGrid;

        // Pass 1: per-sub-block amax to size the super-block d.
        Span<float> subAmax = stackalloc float[Iq1SNumSubBlocks];
        float maxAmax = 0;
        for (int ib = 0; ib < Iq1SNumSubBlocks; ib++)
        {
            float* subSrc = src + ib * Iq1SSubBlockSize;
            float amax = 0;
            for (int i = 0; i < Iq1SSubBlockSize; i++)
            {
                float a = MathF.Abs(subSrc[i]);
                if (a > amax) amax = a;
            }
            subAmax[ib] = amax;
            if (amax > maxAmax) maxAmax = amax;
        }

        // Effective sub-block scale dl = d * (2 * ls + 1), ls in [0..7] -> dl_factor in [1..15].
        // Pick d so the largest |x_i| can be reached at dl_factor=15 with grid value +/-1
        // (dropping the +/-delta contribution since |delta|<1).
        // Headroom factor of 1.0 keeps clip frequency low; codebook quant noise dominates.
        float d = maxAmax > 0 ? maxAmax / 15.0f : 0.0f;

        // Quantise sub-blocks.
        Span<ushort> qhVals = stackalloc ushort[Iq1SNumSubBlocks];
        Span<byte> qsBytes = stackalloc byte[32];

        for (int ib = 0; ib < Iq1SNumSubBlocks; ib++)
        {
            float* subSrc = src + ib * Iq1SSubBlockSize;

            // Per (ls, signOfDelta) candidate, find best codebook indices for each
            // group of 8 and total error. Pick the (ls, sign) with minimum total.
            int bestLs = 0;
            int bestSign = 0;
            uint bestQhLowBits = 0;
            byte bestQs0 = 0, bestQs1 = 0, bestQs2 = 0, bestQs3 = 0;
            float bestTotalErr = float.MaxValue;

            for (int ls = 0; ls < 8; ls++)
            {
                float dl = d * (2 * ls + 1);
                if (dl == 0) dl = 1e-30f;
                for (int signBit = 0; signBit < 2; signBit++)
                {
                    float delta = signBit == 0 ? Iq1SDelta : -Iq1SDelta;
                    float totalErr = 0;
                    uint qhLowBits = 0;
                    Span<byte> groupQs = stackalloc byte[4];

                    for (int l = 0; l < 4; l++)
                    {
                        float* groupSrc = subSrc + l * 8;
                        // Brute-force search over all 2048 grid entries.
                        int bestIdx = 0;
                        float bestErr = float.MaxValue;
                        for (int idx = 0; idx < 2048; idx++)
                        {
                            ulong gridEntry = grid[idx];
                            float err = 0;
                            for (int j = 0; j < 8; j++)
                            {
                                sbyte g = (sbyte)((gridEntry >> (8 * j)) & 0xff);
                                float reconstr = dl * (g + delta);
                                float diff = groupSrc[j] - reconstr;
                                err += diff * diff;
                                if (err >= bestErr) break;
                            }
                            if (err < bestErr) { bestErr = err; bestIdx = idx; }
                        }
                        totalErr += bestErr;
                        groupQs[l] = (byte)(bestIdx & 0xff);
                        qhLowBits |= (uint)((bestIdx >> 8) & 7) << (3 * l);

                        if (totalErr >= bestTotalErr) break;
                    }

                    if (totalErr < bestTotalErr)
                    {
                        bestTotalErr = totalErr;
                        bestLs = ls;
                        bestSign = signBit;
                        bestQhLowBits = qhLowBits;
                        bestQs0 = groupQs[0];
                        bestQs1 = groupQs[1];
                        bestQs2 = groupQs[2];
                        bestQs3 = groupQs[3];
                    }
                }
            }

            qsBytes[ib * 4 + 0] = bestQs0;
            qsBytes[ib * 4 + 1] = bestQs1;
            qsBytes[ib * 4 + 2] = bestQs2;
            qsBytes[ib * 4 + 3] = bestQs3;
            ushort qh = (ushort)(bestQhLowBits & 0x0FFFu);
            qh |= (ushort)((bestLs & 7) << 12);
            if (bestSign == 1) qh |= 0x8000;
            qhVals[ib] = qh;
        }

        // Write block bytes.
        Unsafe.WriteUnaligned(dst, (Half)d);
        for (int i = 0; i < 32; i++) dst[2 + i] = qsBytes[i];
        for (int ib = 0; ib < Iq1SNumSubBlocks; ib++)
        {
            ushort qh = qhVals[ib];
            dst[2 + 32 + ib * 2 + 0] = (byte)(qh & 0xFF);
            dst[2 + 32 + ib * 2 + 1] = (byte)((qh >> 8) & 0xFF);
        }
    }

    /// <summary>
    /// Sanity check: dequantise the fixture bytes via the CPU oracle and
    /// confirm the per-row relative L2 drift versus the source FP32 is within
    /// IQ1_S's expected bound.
    /// </summary>
    public static void AssertFixtureRoundtripIq1S(float[] srcF32, byte[] q1Bytes, int m, int k,
        float maxRelTol = 0.6f)
    {
        var dequant = new float[m * k];
        fixed (byte* p = q1Bytes)
        fixed (float* d = dequant)
        {
            Dequantize.ToFloat32((nint)p, (long)m * k, QuantizationType.IQ1_S,
                new Span<float>(d, m * k));
        }

        float maxRel = 0;
        for (int row = 0; row < m; row++)
        {
            double src2 = 0, diff2 = 0;
            for (int i = 0; i < k; i++)
            {
                float s = srcF32[(long)row * k + i];
                float r = dequant[(long)row * k + i];
                src2 += s * s;
                double e = s - r;
                diff2 += e * e;
            }
            float rel = src2 > 0 ? (float)Math.Sqrt(diff2 / src2) : 0;
            if (rel > maxRel) maxRel = rel;
        }
        Assert.True(maxRel < maxRelTol,
            $"IQ1_S fixture round-trip drift too large: maxRel={maxRel:G4} (m={m}, k={k}, tol={maxRelTol:G4}). " +
            "Fixture quantiser is likely mis-packing or picking a bad d.");
    }

    /// <summary>Scalar CPU dequant reference using the same byte buffer the GPU sees.</summary>
    public static float[] CpuDequantizeIq1S(byte[] bytes, int totalElements)
    {
        var dst = new float[totalElements];
        fixed (byte* p = bytes)
        fixed (float* d = dst)
        {
            Dequantize.ToFloat32((nint)p, totalElements, QuantizationType.IQ1_S,
                new Span<float>(d, totalElements));
        }
        return dst;
    }

    /// <summary>Scalar CPU GEMV reference: <c>y[m] = sum_k W[m, k] * x[k]</c>.</summary>
    public static float[] CpuGemvIq1S(byte[] weightsIq1S, float[] x, int m, int k)
    {
        if ((k % Iq1SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq1SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq1SGroupSize;
        int rowBytes = blocksPerRow * Iq1SBlockBytes;
        var result = new float[m];
        ReadOnlySpan<ulong> grid = Dequantize.Iq1SGrid;

        fixed (byte* wPtr = weightsIq1S)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Iq1SBlockBytes;
                    float d = (float)Unsafe.ReadUnaligned<Half>(block);
                    byte* qs = block + 2;
                    ushort* qh = (ushort*)(block + 2 + 32);
                    int xBase = b * Iq1SGroupSize;

                    for (int ib = 0; ib < Iq1SNumSubBlocks; ib++)
                    {
                        ushort qhVal = qh[ib];
                        float dl = d * (2 * ((qhVal >> 12) & 7) + 1);
                        float delta = (qhVal & 0x8000) != 0 ? -Iq1SDelta : Iq1SDelta;
                        int qsBase = ib * 4;
                        for (int l = 0; l < 4; l++)
                        {
                            int idx = qs[qsBase + l] | (((qhVal >> (3 * l)) & 7) << 8);
                            ulong gridEntry = grid[idx];
                            int xOff = xBase + ib * 32 + l * 8;
                            for (int j = 0; j < 8; j++)
                            {
                                sbyte g = (sbyte)((gridEntry >> (8 * j)) & 0xff);
                                sum += x[xOff + j] * dl * (g + delta);
                            }
                        }
                    }
                }
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>Scalar CPU GEMM reference: <c>C[N, M] = B[N, K] @ W_iq1s[M, K]^T</c>.</summary>
    public static float[] CpuGemmIq1S(byte[] weightsIq1S, float[] inputB, int m, int k, int n)
    {
        if ((k % Iq1SGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq1SGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq1SGroupSize;
        int rowBytes = blocksPerRow * Iq1SBlockBytes;
        var result = new float[n * m];
        ReadOnlySpan<ulong> grid = Dequantize.Iq1SGrid;

        fixed (byte* wPtr = weightsIq1S)
        {
            for (int t = 0; t < n; t++)
            {
                int bRowBase = t * k;
                for (int row = 0; row < m; row++)
                {
                    byte* rowBase = wPtr + (long)row * rowBytes;
                    float sum = 0;
                    for (int b = 0; b < blocksPerRow; b++)
                    {
                        byte* block = rowBase + b * Iq1SBlockBytes;
                        float d = (float)Unsafe.ReadUnaligned<Half>(block);
                        byte* qs = block + 2;
                        ushort* qh = (ushort*)(block + 2 + 32);
                        int xBase = b * Iq1SGroupSize;

                        for (int ib = 0; ib < Iq1SNumSubBlocks; ib++)
                        {
                            ushort qhVal = qh[ib];
                            float dl = d * (2 * ((qhVal >> 12) & 7) + 1);
                            float delta = (qhVal & 0x8000) != 0 ? -Iq1SDelta : Iq1SDelta;
                            int qsBase = ib * 4;
                            for (int l = 0; l < 4; l++)
                            {
                                int idx = qs[qsBase + l] | (((qhVal >> (3 * l)) & 7) << 8);
                                ulong gridEntry = grid[idx];
                                int xOff = xBase + ib * 32 + l * 8;
                                for (int j = 0; j < 8; j++)
                                {
                                    sbyte g = (sbyte)((gridEntry >> (8 * j)) & 0xff);
                                    sum += inputB[bRowBase + xOff + j] * dl * (g + delta);
                                }
                            }
                        }
                    }
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
    }

    public static void AssertClose(float[] expected, float[] actual, string context,
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
            $"Numerical drift exceeded tolerance ({context}): " +
            $"errors={errors}/{expected.Length}, maxAbs={maxAbs:G9}, maxRel={maxRel:G9}");
    }
}
