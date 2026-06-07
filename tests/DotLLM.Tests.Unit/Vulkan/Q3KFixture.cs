using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only Q3_K fixture helpers shared by the GEMV / GEMM kernel tests.
/// Sibling of <see cref="Q2KFixture"/> / <see cref="Q4KFixture"/> /
/// <see cref="Q5KFixture"/> / <see cref="Q6KFixture"/>.
/// </summary>
/// <remarks>
/// <para>
/// Q3_K block layout (matches <c>DequantizeQ3_KScalar</c> / llama.cpp):
/// <c>hmask[32] (1 bit/elem) + qs[64] (2 bits/elem) + scales[12] (16 × 6-bit unsigned, biased -32) + d (fp16) = 110 bytes / 256 elements</c>.
/// Decoded value per element <c>t</c> with <c>sub = t / 16</c>:
/// <c>q3 = ((hmask[t/8] &gt;&gt; (t%8)) &amp; 1) &lt;&lt; 2 | ((qs[t/4] &gt;&gt; ((t%4)*2)) &amp; 3) - 4</c>
/// gives a signed 3-bit value in <c>-4..3</c>;
/// <c>signedScale = scales[sub] - 32</c> in <c>-32..31</c>;
/// <c>value = d * signedScale * q3</c>.
/// </para>
/// <para>
/// Quantiser (mirrors llama.cpp's <c>quantize_row_q3_K_ref</c>): per 16-element
/// sub-block, fit a single signed scale via amax against the 3-bit signed
/// codebook <c>-4..3</c>; the global <c>d</c> is chosen so that the
/// per-sub-block scales fit in the unsigned 6-bit table after the +32 bias.
/// Like the other fixtures this is a minimal scalar quantiser — its only job
/// is to round-trip cleanly through <c>DequantizeQ3_KScalar</c> so the kernel
/// tests test the kernel.
/// </para>
/// </remarks>
internal static unsafe class Q3KFixture
{
    public const int Q3KGroupSize = 256;
    public const int Q3KBlockBytes = 110;
    public const int SubBlockSize = 16;
    public const int NumSubBlocks = 16;

    /// <summary>
    /// Generate a uniform-random FP32 array in [-range, range). All 16
    /// sub-blocks per super-block carry random data — the CPU-oracle bug
    /// flagged by Agent 4 (sub_12..15 read from bytes 8..11 high nibble
    /// instead of bytes 4..7) has been fixed in <c>DequantizeQ3_KScalar</c>
    /// and the matching CUDA + Vulkan kernels.
    /// </summary>
    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// Quantise an <c>[m, k]</c> row-major FP32 matrix to Q3_K bytes (one
    /// 110-byte super-block per 256 elements per row). Layout matches
    /// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ3_KScalar</c>
    /// byte-for-byte.
    /// </summary>
    public static byte[] QuantizeRows(float[] src, int m, int k)
    {
        if ((k % Q3KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q3KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q3KGroupSize;
        int rowBytes = blocksPerRow * Q3KBlockBytes;
        var dst = new byte[m * rowBytes];

        Span<float> subScales = stackalloc float[NumSubBlocks];
        Span<int> subScalesI  = stackalloc int[NumSubBlocks]; // signed, range -32..31 (raw)
        Span<int> q3vals      = stackalloc int[Q3KGroupSize]; // signed, -4..3

        fixed (float* srcPtr = src)
        fixed (byte*  dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
            {
                float* rowSrc = srcPtr + (long)row * k;
                byte*  rowDst = dstPtr + (long)row * rowBytes;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    QuantizeSuperBlock(
                        rowSrc + b * Q3KGroupSize,
                        rowDst + b * Q3KBlockBytes,
                        subScales, subScalesI, q3vals);
                }
            }
        }
        return dst;
    }

    private static unsafe void QuantizeSuperBlock(
        float* srcSuper, byte* dstSuper,
        Span<float> subScales, Span<int> subScalesI, Span<int> q3vals)
    {
        // Pass 1: per-sub-block signed scale via amax fit against -4 (most-negative quant).
        // Mirrors the canonical Q3_K choice — pin the most-negative quant to the actual
        // amax of the sub-block.
        float globalAmaxScale = 0;
        for (int j = 0; j < NumSubBlocks; j++)
        {
            float* sub = srcSuper + j * SubBlockSize;
            float amax = 0; float amaxSigned = 0;
            for (int i = 0; i < SubBlockSize; i++)
            {
                float a = MathF.Abs(sub[i]);
                if (a > amax) { amax = a; amaxSigned = sub[i]; }
            }
            float sc = amaxSigned == 0 ? 0 : amaxSigned / -4.0f;
            subScales[j] = sc;
            float a2 = MathF.Abs(sc);
            if (a2 > globalAmaxScale) globalAmaxScale = a2;
        }

        // Pass 2: global d so per-sub-block signed scales fit in -32..31.
        // We aim signed scale = round(sc / d). Choose d = amax / -32 so the largest
        // |sc| maps to ±32 (clamped to 31 after rounding).
        float dF = globalAmaxScale > 0 ? globalAmaxScale / 32.0f : 0;

        for (int j = 0; j < NumSubBlocks; j++)
        {
            int sQ = dF > 0 ? (int)MathF.Round(subScales[j] / dF) : 0;
            sQ = Math.Clamp(sQ, -32, 31);
            subScalesI[j] = sQ;
        }

        // Pass 3: per-element 3-bit signed quant. value ≈ d * sQ * q3 → q3 = round(x / (d*sQ)).
        for (int j = 0; j < NumSubBlocks; j++)
        {
            int sQ = subScalesI[j];
            float scF = dF * sQ;
            float* sub = srcSuper + j * SubBlockSize;
            int outBase = j * SubBlockSize;
            for (int i = 0; i < SubBlockSize; i++)
            {
                int q;
                if (scF != 0)
                {
                    int v = (int)MathF.Round(sub[i] / scF);
                    q = Math.Clamp(v, -4, 3);
                }
                else
                {
                    q = 0;
                }
                q3vals[outBase + i] = q;
            }
        }

        // Pass 4: pack hmask (1 bit per element from (q3+4)>>2), qs (2 low bits of q3+4),
        // scales[12] (16 × 6-bit unsigned, encoded as q3 raw + 32 → 0..63), d (fp16).
        byte* hmaskPtr  = dstSuper;        // 32 bytes
        byte* qsPtr     = dstSuper + 32;   // 64 bytes
        byte* scalesPtr = dstSuper + 96;   // 12 bytes
        for (int i = 0; i < 32; i++) hmaskPtr[i] = 0;
        for (int i = 0; i < 64; i++) qsPtr[i]    = 0;
        for (int i = 0; i < 12; i++) scalesPtr[i] = 0;

        for (int t = 0; t < Q3KGroupSize; t++)
        {
            int q3plus4 = q3vals[t] + 4; // 0..7 (3-bit unsigned)
            int low2 = q3plus4 & 0x3;
            int hi1  = (q3plus4 >> 2) & 0x1;

            qsPtr[t >> 2] |= (byte)(low2 << ((t & 3) * 2));
            hmaskPtr[t >> 3] |= (byte)(hi1 << (t & 7));
        }

        // Pack scales[12]: 6-bit unsigned per sub-block (subScalesI[j] + 32, range 0..63).
        // Bit layout (mirrors DequantizeQ3_KScalar's unpack, post Q3_K fix):
        //   sub  0..7  low nibble = scales12[sub] low nibble
        //   sub  8..15 low nibble = scales12[sub-8] high nibble (NOT sub-4 — that
        //   collides with the high-2-bits packing). Bytes 0..7 carry both halves.
        //   sub  0..15 high 2 bits = scales12[8 + sub/4], shifted by (sub%4)*2.
        for (int sub = 0; sub < 16; sub++)
        {
            int scale6 = (subScalesI[sub] + 32) & 0x3F;
            int low4 = scale6 & 0xF;
            if (sub < 8)
            {
                scalesPtr[sub] = (byte)((scalesPtr[sub] & 0xF0) | low4);
            }
            else
            {
                int srcByte = sub - 8;
                scalesPtr[srcByte] = (byte)((scalesPtr[srcByte] & 0x0F) | (low4 << 4));
            }
        }
        for (int sub = 0; sub < 16; sub++)
        {
            int scale6 = (subScalesI[sub] + 32) & 0x3F;
            int hi2  = (scale6 >> 4) & 0x3;
            int hiByte = 8 + (sub / 4);
            int hiShift = (sub % 4) * 2;
            byte mask = (byte)(0x3 << hiShift);
            scalesPtr[hiByte] = (byte)((scalesPtr[hiByte] & ~mask) | (hi2 << hiShift));
        }

        // d at offset 108.
        Unsafe.WriteUnaligned(dstSuper + 108, (Half)dF);
    }

    /// <summary>Dequantise an entire Q3_K blob to FP32 via the CPU oracle.
    /// Used by the dequant-kernel parity test.</summary>
    public static float[] CpuDequantizeQ3K(byte[] q3kBytes, int totalElements)
    {
        var dst = new float[totalElements];
        fixed (byte* p = q3kBytes)
        {
            Dequantize.ToFloat32((nint)p, totalElements, QuantizationType.Q3_K, dst);
        }
        return dst;
    }

    /// <summary>
    /// Smoke check on the fixture round-trip. The Q3_K dequant bug
    /// (sub_12..15 reading from bytes 8..11 high nibble) has been fixed in
    /// <c>DequantizeQ3_KScalar</c>, so round-trip drift on Q3_K now matches
    /// the K-quant family at large (~few percent depending on data distribution).
    /// </summary>
    public static void AssertFixtureRoundtrip(float[] srcF32, byte[] q3kBytes, int m, int k)
    {
        var dequant = new float[m * k];
        fixed (byte* p = q3kBytes)
        fixed (float* d = dequant)
        {
            Dequantize.ToFloat32((nint)p, (long)m * k, QuantizationType.Q3_K,
                new Span<float>(d, m * k));
        }

        double src2Total = 0, diff2Total = 0;
        long n = (long)m * k;
        for (long i = 0; i < n; i++)
        {
            float s = srcF32[i];
            float r = dequant[i];
            src2Total += s * s;
            double e = s - r;
            diff2Total += e * e;
        }
        float relAgg = src2Total > 0 ? (float)Math.Sqrt(diff2Total / src2Total) : 0;
        Assert.True(relAgg < 0.30f,
            $"Q3_K fixture aggregate round-trip drift too large: rel={relAgg:G4} (m={m}, k={k}). " +
            "Fixture quantiser is likely mis-packing the hmask, qs[], or 6-bit-then-biased scales.");
    }

    /// <summary>
    /// Scalar CPU GEMV reference: reads the same Q3_K bytes the GPU sees,
    /// dequantises on the fly, dots against FP32 <c>x</c>. Block-sequential
    /// reduction order — the GPU shader uses a workgroup tree reduce.
    /// </summary>
    public static float[] CpuGemvQ3K(byte[] weightsQ3K, float[] x, int m, int k)
    {
        if ((k % Q3KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q3KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q3KGroupSize;
        int rowBytes = blocksPerRow * Q3KBlockBytes;
        var result = new float[m];

        fixed (byte* wPtr = weightsQ3K)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block  = rowBase + b * Q3KBlockBytes;
                    byte* hmask  = block;
                    byte* qs     = block + 32;
                    byte* scales = block + 96;
                    float d      = (float)Unsafe.ReadUnaligned<Half>(block + 108);
                    int xBase = b * Q3KGroupSize;

                    for (int sub = 0; sub < 16; sub++)
                    {
                        int scale6 = UnpackQ3Scale(scales, sub);
                        int signedSc = scale6 - 32;
                        float scF = d * signedSc;
                        int outBase = xBase + sub * 16;
                        for (int l = 0; l < 16; l++)
                        {
                            int t = sub * 16 + l;
                            int qBits = (qs[t >> 2] >> ((t & 3) * 2)) & 0x3;
                            int hBit  = (hmask[t >> 3] >> (t & 7)) & 0x1;
                            int signed3 = ((hBit << 2) | qBits) - 4;
                            float w = scF * signed3;
                            sum += w * x[outBase + l];
                        }
                    }
                }
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>
    /// Scalar CPU GEMM reference: <c>C[N, M] = B[N, K] @ W_q3k[M, K]^T</c>.
    /// Reads the same Q3_K bytes the GPU sees.
    /// </summary>
    public static float[] CpuGemmQ3K(byte[] weightsQ3K, float[] inputB, int m, int k, int n)
    {
        if ((k % Q3KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q3KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q3KGroupSize;
        int rowBytes = blocksPerRow * Q3KBlockBytes;
        var result = new float[n * m];

        fixed (byte* wPtr = weightsQ3K)
        {
            for (int t = 0; t < n; t++)
            {
                int bRowBase = t * k;
                for (int row = 0; row < m; row++)
                {
                    byte* rowBaseW = wPtr + (long)row * rowBytes;
                    float sum = 0;
                    for (int bb = 0; bb < blocksPerRow; bb++)
                    {
                        byte* block  = rowBaseW + bb * Q3KBlockBytes;
                        byte* hmask  = block;
                        byte* qs     = block + 32;
                        byte* scales = block + 96;
                        float d      = (float)Unsafe.ReadUnaligned<Half>(block + 108);
                        int xBase = bb * Q3KGroupSize;

                        for (int sub = 0; sub < 16; sub++)
                        {
                            int scale6 = UnpackQ3Scale(scales, sub);
                            int signedSc = scale6 - 32;
                            float scF = d * signedSc;
                            int outBase = xBase + sub * 16;
                            for (int l = 0; l < 16; l++)
                            {
                                int tIdx = sub * 16 + l;
                                int qBits = (qs[tIdx >> 2] >> ((tIdx & 3) * 2)) & 0x3;
                                int hBit  = (hmask[tIdx >> 3] >> (tIdx & 7)) & 0x1;
                                int signed3 = ((hBit << 2) | qBits) - 4;
                                float w = scF * signed3;
                                sum += w * inputB[bRowBase + outBase + l];
                            }
                        }
                    }
                    result[t * m + row] = sum;
                }
            }
        }
        return result;
    }

    private static int UnpackQ3Scale(byte* scales12, int sub)
    {
        int lowNibble;
        if (sub < 8)
            lowNibble = scales12[sub] & 0xF;
        else
            lowNibble = (scales12[sub - 8] >> 4) & 0xF;
        int hiByte  = 8 + (sub / 4);
        int hiShift = (sub % 4) * 2;
        int hiBits  = (scales12[hiByte] >> hiShift) & 0x3;
        return lowNibble | (hiBits << 4);
    }

    /// <summary>Asserts every cell is within either abs <paramref name="absTol"/>
    /// or rel <paramref name="relTol"/> of the expected value.</summary>
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
