using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only IQ4_NL / IQ4_XS fixture helpers shared by the dequant / GEMV /
/// GEMM kernel tests.
/// </summary>
/// <remarks>
/// <para>
/// As with the Q4_K fixture, the production CPU side has full IQ4 dequantise
/// kernels but no F32-to-IQ4 quantiser — production loaders only ingest
/// pre-quantised GGUF data. For Vulkan parity we round-trip arbitrary FP32
/// fixtures through IQ4 bytes. Layouts match
/// <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ4_NL/_XS</c> and llama.cpp's
/// <c>block_iq4_nl</c> / <c>block_iq4_xs</c> byte-for-byte.
/// </para>
/// <para>
/// Per-element decode is a 16-entry signed-int8 lookup (<c>kvalues_iq4nl</c>),
/// not a linear int — so the quantiser does a best-effort fit using only the
/// per-block (or per-sub-block) <c>d</c> scale and 4-bit codebook indices.
/// </para>
/// </remarks>
internal static unsafe class Iq4Fixture
{
    public const int Iq4NlGroupSize = 32;
    public const int Iq4NlBlockBytes = 18;

    public const int Iq4XsGroupSize = 256;
    public const int Iq4XsBlockBytes = 136;
    public const int Iq4XsNumSubBlocks = 8;
    public const int Iq4XsSubBlockSize = 32;

    /// <summary>Signed-int8 lookup shared by IQ4_NL and IQ4_XS (ggml's kvalues_iq4nl).</summary>
    public static ReadOnlySpan<sbyte> KvaluesIq4Nl =>
    [
        -127, -104, -83, -65, -49, -35, -22, -10,
        1, 13, 25, 38, 53, 69, 89, 113
    ];

    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    // ── IQ4_NL ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Quantise an [m, k] row-major FP32 matrix to IQ4_NL bytes. k must be a
    /// multiple of 32. Per-block layout:
    ///   bytes [0,1]  = fp16 d
    ///   bytes [2..17] = qs[16] (low nibble = element j, high nibble = element j + 16)
    /// </summary>
    public static byte[] QuantizeRowsIq4Nl(float[] src, int m, int k)
    {
        if ((k % Iq4NlGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4NlGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4NlGroupSize;
        int rowBytes = blocksPerRow * Iq4NlBlockBytes;
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
                    QuantizeBlockIq4Nl(rowSrc + b * Iq4NlGroupSize, rowDst + b * Iq4NlBlockBytes);
                }
            }
        }
        return dst;
    }

    /// <summary>
    /// Picks the per-block <c>d</c> by minimising <c>sum((x_i - d * kv[q_i])^2)</c>
    /// across the 32-element block, where <c>q_i = argmin_q |x_i / d - kv[q]|</c>.
    /// Closed-form approximation: <c>d = amax(x) / 113</c> (largest positive
    /// codebook value) — picks a scale that places the loudest element at the
    /// top of the codebook range. Identical philosophy to ggml's reference
    /// <c>quantize_row_iq4_nl_impl</c> in spirit (rmse iteration there does the
    /// same thing but with a few refinement passes).
    /// </summary>
    private static void QuantizeBlockIq4Nl(float* src, byte* dst)
    {
        float amax = 0;
        for (int i = 0; i < Iq4NlGroupSize; i++)
        {
            float a = MathF.Abs(src[i]);
            if (a > amax) amax = a;
        }

        // d chosen so kvalues_iq4nl[-1..14] covers [-127*d, 113*d]. We split the
        // budget evenly: amax / max(|kv|) where max |kv| = 127.
        float d = amax > 0 ? amax / 127.0f : 0.0f;
        float invD = d > 0 ? 1.0f / d : 0.0f;

        Half hd = (Half)d;
        Unsafe.WriteUnaligned(dst, hd);

        ReadOnlySpan<sbyte> kv = KvaluesIq4Nl;
        byte[] nibbles = new byte[Iq4NlGroupSize];
        for (int i = 0; i < Iq4NlGroupSize; i++)
        {
            float target = src[i] * invD;
            int best = 0;
            float bestErr = MathF.Abs(target - kv[0]);
            for (int q = 1; q < 16; q++)
            {
                float err = MathF.Abs(target - kv[q]);
                if (err < bestErr) { bestErr = err; best = q; }
            }
            nibbles[i] = (byte)best;
        }

        // qs[j]: low nibble = element j, high nibble = element j + 16
        for (int j = 0; j < 16; j++)
        {
            byte lo = nibbles[j];
            byte hi = nibbles[j + 16];
            dst[2 + j] = (byte)((lo & 0xF) | ((hi & 0xF) << 4));
        }
    }

    /// <summary>
    /// Sanity check: dequantise the fixture bytes via the CPU oracle and confirm
    /// the per-row relative L2 drift versus the source FP32 is within expectations
    /// for IQ4_NL (~6% of dynamic range — IQ4_NL is non-linear so per-element
    /// rounding error varies more than linear Q4).
    /// </summary>
    public static void AssertFixtureRoundtripIq4Nl(float[] srcF32, byte[] q4Bytes, int m, int k)
    {
        var dequant = new float[m * k];
        fixed (byte* p = q4Bytes)
        fixed (float* d = dequant)
        {
            Dequantize.ToFloat32((nint)p, (long)m * k, QuantizationType.IQ4_NL,
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
        Assert.True(maxRel < 0.20f,
            $"IQ4_NL fixture round-trip drift too large: maxRel={maxRel:G4} (m={m}, k={k}). " +
            "Fixture quantiser is likely mis-packing or picking a bad d.");
    }

    /// <summary>Scalar CPU dequant reference using the same byte buffer the GPU sees.</summary>
    public static float[] CpuDequantizeIq4Nl(byte[] bytes, int totalElements)
    {
        var dst = new float[totalElements];
        fixed (byte* p = bytes)
        fixed (float* d = dst)
        {
            Dequantize.ToFloat32((nint)p, totalElements, QuantizationType.IQ4_NL,
                new Span<float>(d, totalElements));
        }
        return dst;
    }

    /// <summary>Scalar CPU GEMV reference: reads the same IQ4_NL bytes the GPU sees.</summary>
    public static float[] CpuGemvIq4Nl(byte[] weightsIq4Nl, float[] x, int m, int k)
    {
        if ((k % Iq4NlGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4NlGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4NlGroupSize;
        int rowBytes = blocksPerRow * Iq4NlBlockBytes;
        var result = new float[m];
        ReadOnlySpan<sbyte> kv = KvaluesIq4Nl;

        fixed (byte* wPtr = weightsIq4Nl)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Iq4NlBlockBytes;
                    float d = (float)Unsafe.ReadUnaligned<Half>(block);
                    byte* qs = block + 2;
                    int xBase = b * Iq4NlGroupSize;

                    float subSum = 0;
                    for (int j = 0; j < 16; j++)
                    {
                        byte q = qs[j];
                        int nLo = q & 0xF;
                        int nHi = q >> 4;
                        subSum += x[xBase + j] * kv[nLo]
                                + x[xBase + j + 16] * kv[nHi];
                    }
                    sum += d * subSum;
                }
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>Scalar CPU GEMM reference: <c>C[N, M] = B[N, K] @ W_iq4nl[M, K]^T</c>.</summary>
    public static float[] CpuGemmIq4Nl(byte[] weightsIq4Nl, float[] inputB, int m, int k, int n)
    {
        if ((k % Iq4NlGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4NlGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4NlGroupSize;
        int rowBytes = blocksPerRow * Iq4NlBlockBytes;
        var result = new float[n * m];
        ReadOnlySpan<sbyte> kv = KvaluesIq4Nl;

        fixed (byte* wPtr = weightsIq4Nl)
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
                        byte* block = rowBase + b * Iq4NlBlockBytes;
                        float d = (float)Unsafe.ReadUnaligned<Half>(block);
                        byte* qs = block + 2;
                        int xBase = b * Iq4NlGroupSize;

                        float subSum = 0;
                        for (int j = 0; j < 16; j++)
                        {
                            byte q = qs[j];
                            int nLo = q & 0xF;
                            int nHi = q >> 4;
                            subSum += inputB[bRowBase + xBase + j] * kv[nLo]
                                    + inputB[bRowBase + xBase + j + 16] * kv[nHi];
                        }
                        sum += d * subSum;
                    }
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
    }

    // ── IQ4_XS ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Quantise an [m, k] row-major FP32 matrix to IQ4_XS bytes. k must be a
    /// multiple of 256. Per-super-block layout:
    ///   bytes [0,1]    = fp16 d
    ///   bytes [2,3]    = scales_h (uint16 LE)
    ///   bytes [4..7]   = scales_l[4]
    ///   bytes [8..135] = qs[128]  (8 sub-blocks of 16 bytes each)
    /// </summary>
    public static byte[] QuantizeRowsIq4Xs(float[] src, int m, int k)
    {
        if ((k % Iq4XsGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4XsGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4XsGroupSize;
        int rowBytes = blocksPerRow * Iq4XsBlockBytes;
        var dst = new byte[m * rowBytes];

        Span<float> subAmax = stackalloc float[Iq4XsNumSubBlocks];

        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
            {
                float* rowSrc = srcPtr + (long)row * k;
                byte* rowDst = dstPtr + (long)row * rowBytes;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    QuantizeSuperBlockIq4Xs(
                        rowSrc + b * Iq4XsGroupSize,
                        rowDst + b * Iq4XsBlockBytes,
                        subAmax);
                }
            }
        }
        return dst;
    }

    /// <summary>
    /// Quantises one IQ4_XS super-block. Per-sub-block: pick |amax| / 127 as
    /// the effective scale (= <c>d * (ls - 32)</c>), then deconstruct into
    /// (super-block d, per-sub-block ls - 32). 6-bit ls range = [0..63], so the
    /// signed effective range is [-32..31]. Choose d to fit the largest
    /// |sub-amax| at ls=31 (or ls=-32 if the largest is negative; we just use
    /// abs and pick the sign during nibble quant).
    /// </summary>
    private static void QuantizeSuperBlockIq4Xs(float* src, byte* dst, Span<float> subAmax)
    {
        // Pass 1: per-sub-block amax. Track signed extreme (i.e. the value with
        // greatest absolute magnitude) to decide the sign of the sub-block scale.
        Span<float> subSigned = stackalloc float[Iq4XsNumSubBlocks];
        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
        {
            float* subSrc = src + ib * Iq4XsSubBlockSize;
            float ex = 0;
            float exAbs = 0;
            for (int i = 0; i < Iq4XsSubBlockSize; i++)
            {
                float v = subSrc[i];
                float a = MathF.Abs(v);
                if (a > exAbs) { exAbs = a; ex = v; }
            }
            subSigned[ib] = ex;
            subAmax[ib] = exAbs;
        }

        // Pass 2: pick super-block d such that the largest |sub-extreme| maps to
        // an ls of 31 (= sign-positive extreme of the codebook scaling). Per
        // sub-block: signed_scale = sign(ex) * exAbs / 127. signed_scale = d * (ls - 32).
        // To keep precision we use d = max(signed_scale_abs) / 31 and then set ls per
        // sub-block to round(sub_signed_scale / d) + 32 (clamped to [0, 63]).
        float maxSignedScaleAbs = 0;
        Span<float> subSignedScale = stackalloc float[Iq4XsNumSubBlocks];
        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
        {
            float sgn = subSigned[ib] >= 0 ? 1f : -1f;
            float scale = sgn * (subAmax[ib] / 127.0f);
            subSignedScale[ib] = scale;
            float a = MathF.Abs(scale);
            if (a > maxSignedScaleAbs) maxSignedScaleAbs = a;
        }
        float d = maxSignedScaleAbs > 0 ? maxSignedScaleAbs / 31.0f : 0.0f;
        float invD = d > 0 ? 1.0f / d : 0.0f;

        // Pass 3: per-sub-block ls (6-bit). Quantise effective scale.
        Span<byte> ls6 = stackalloc byte[Iq4XsNumSubBlocks];
        Span<float> dlUsed = stackalloc float[Iq4XsNumSubBlocks];
        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
        {
            int v = d > 0 ? (int)MathF.Round(subSignedScale[ib] * invD) + 32 : 32;
            v = Math.Clamp(v, 0, 63);
            ls6[ib] = (byte)v;
            dlUsed[ib] = d * (v - 32);
        }

        // Pass 4: nibble-quant each element via codebook nearest.
        ReadOnlySpan<sbyte> kv = KvaluesIq4Nl;
        Span<byte> nibbles = stackalloc byte[Iq4XsGroupSize];
        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
        {
            float dl = dlUsed[ib];
            float invDl = dl != 0 ? 1.0f / dl : 0.0f;
            float* subSrc = src + ib * Iq4XsSubBlockSize;
            int outBase = ib * Iq4XsSubBlockSize;
            for (int i = 0; i < Iq4XsSubBlockSize; i++)
            {
                int best = 8; // codebook index 8 = 1, near zero
                if (dl != 0)
                {
                    float target = subSrc[i] * invDl;
                    best = 0;
                    float bestErr = MathF.Abs(target - kv[0]);
                    for (int q = 1; q < 16; q++)
                    {
                        float err = MathF.Abs(target - kv[q]);
                        if (err < bestErr) { bestErr = err; best = q; }
                    }
                }
                nibbles[outBase + i] = (byte)best;
            }
        }

        // Pass 5: write block bytes.
        Unsafe.WriteUnaligned(dst, (Half)d);
        // scales_h (uint16): top 2 bits of each ls6 (bits 4..5).
        ushort scalesH = 0;
        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
            scalesH |= (ushort)(((ls6[ib] >> 4) & 0x3) << (2 * ib));
        dst[2] = (byte)(scalesH & 0xFF);
        dst[3] = (byte)((scalesH >> 8) & 0xFF);
        // scales_l[4]: low 4 bits of each ls6, packed two per byte.
        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
        {
            int half = ib >> 1;
            int shift = 4 * (ib & 1);
            int low = ls6[ib] & 0xF;
            if ((ib & 1) == 0) dst[4 + half] = (byte)low;
            else dst[4 + half] |= (byte)(low << shift);
        }
        // qs[128]: per sub-block, 16 bytes with low nibble = element j, high = j + 16.
        byte* qs = dst + 8;
        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
        {
            int subBase = ib * Iq4XsSubBlockSize;
            int outBase = ib * 16;
            for (int j = 0; j < 16; j++)
            {
                byte lo = nibbles[subBase + j];
                byte hi = nibbles[subBase + j + 16];
                qs[outBase + j] = (byte)((lo & 0xF) | ((hi & 0xF) << 4));
            }
        }
    }

    public static void AssertFixtureRoundtripIq4Xs(float[] srcF32, byte[] q4Bytes, int m, int k)
    {
        var dequant = new float[m * k];
        fixed (byte* p = q4Bytes)
        fixed (float* d = dequant)
        {
            Dequantize.ToFloat32((nint)p, (long)m * k, QuantizationType.IQ4_XS,
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
        Assert.True(maxRel < 0.20f,
            $"IQ4_XS fixture round-trip drift too large: maxRel={maxRel:G4} (m={m}, k={k}).");
    }

    public static float[] CpuDequantizeIq4Xs(byte[] bytes, int totalElements)
    {
        var dst = new float[totalElements];
        fixed (byte* p = bytes)
        fixed (float* d = dst)
        {
            Dequantize.ToFloat32((nint)p, totalElements, QuantizationType.IQ4_XS,
                new Span<float>(d, totalElements));
        }
        return dst;
    }

    public static float[] CpuGemvIq4Xs(byte[] weightsIq4Xs, float[] x, int m, int k)
    {
        if ((k % Iq4XsGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4XsGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4XsGroupSize;
        int rowBytes = blocksPerRow * Iq4XsBlockBytes;
        var result = new float[m];
        ReadOnlySpan<sbyte> kv = KvaluesIq4Nl;

        fixed (byte* wPtr = weightsIq4Xs)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Iq4XsBlockBytes;
                    float d = (float)Unsafe.ReadUnaligned<Half>(block);
                    ushort scalesH = (ushort)(block[2] | (block[3] << 8));
                    byte* scalesL = block + 4;
                    byte* qs = block + 8;
                    int xBase = b * Iq4XsGroupSize;

                    for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
                    {
                        int low = (scalesL[ib >> 1] >> (4 * (ib & 1))) & 0xF;
                        int high = (scalesH >> (2 * ib)) & 0x3;
                        int ls = low | (high << 4);
                        float dl = d * (ls - 32);

                        int subBase = xBase + ib * Iq4XsSubBlockSize;
                        byte* subQs = qs + ib * 16;
                        float subSum = 0;
                        for (int j = 0; j < 16; j++)
                        {
                            byte q = subQs[j];
                            int nLo = q & 0xF;
                            int nHi = q >> 4;
                            subSum += x[subBase + j] * kv[nLo]
                                    + x[subBase + j + 16] * kv[nHi];
                        }
                        sum += dl * subSum;
                    }
                }
                result[row] = sum;
            }
        }
        return result;
    }

    public static float[] CpuGemmIq4Xs(byte[] weightsIq4Xs, float[] inputB, int m, int k, int n)
    {
        if ((k % Iq4XsGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq4XsGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq4XsGroupSize;
        int rowBytes = blocksPerRow * Iq4XsBlockBytes;
        var result = new float[n * m];
        ReadOnlySpan<sbyte> kv = KvaluesIq4Nl;

        fixed (byte* wPtr = weightsIq4Xs)
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
                        byte* block = rowBase + b * Iq4XsBlockBytes;
                        float d = (float)Unsafe.ReadUnaligned<Half>(block);
                        ushort scalesH = (ushort)(block[2] | (block[3] << 8));
                        byte* scalesL = block + 4;
                        byte* qs = block + 8;
                        int xBase = b * Iq4XsGroupSize;

                        for (int ib = 0; ib < Iq4XsNumSubBlocks; ib++)
                        {
                            int low = (scalesL[ib >> 1] >> (4 * (ib & 1))) & 0xF;
                            int high = (scalesH >> (2 * ib)) & 0x3;
                            int ls = low | (high << 4);
                            float dl = d * (ls - 32);

                            int subBase = xBase + ib * Iq4XsSubBlockSize;
                            byte* subQs = qs + ib * 16;
                            float subSum = 0;
                            for (int j = 0; j < 16; j++)
                            {
                                byte q = subQs[j];
                                int nLo = q & 0xF;
                                int nHi = q >> 4;
                                subSum += inputB[bRowBase + subBase + j] * kv[nLo]
                                        + inputB[bRowBase + subBase + j + 16] * kv[nHi];
                            }
                            sum += dl * subSum;
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
