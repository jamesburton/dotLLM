using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only Q2_K fixture helpers shared by the GEMV / GEMM kernel tests and
/// the optional model-level Q2_K parity tests. Sibling of <see cref="Q4KFixture"/>
/// / <see cref="Q5KFixture"/> / <see cref="Q6KFixture"/>.
/// </summary>
/// <remarks>
/// <para>
/// The production CPU side has full Q2_K dequantise via
/// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ2_K</c> but no
/// F32-to-Q2_K quantiser — production loaders only ingest pre-quantised GGUF
/// data. For Vulkan parity we need to round-trip arbitrary FP32 fixtures into
/// Q2_K bytes, so this file contains a minimal scalar quantiser that produces
/// the same byte layout as llama.cpp's <c>quantize_row_q2_K_ref</c>: 256-element
/// super-blocks of 84 bytes laid out as
/// <c>[scales[16]][qs[64]][d:fp16][dmin:fp16]</c>.
/// </para>
/// <para>
/// Per-element decode (mirrors the CPU oracle): for sub-block <c>j</c> in
/// <c>[0, 16)</c> and element <c>l</c> in <c>[0, 16)</c>:
/// <c>q2 = (qs[t/4] &gt;&gt; ((t%4)*2)) &amp; 3</c>,
/// <c>scale = scales[j] &amp; 0xF</c>, <c>dmCoef = (scales[j] &gt;&gt; 4) &amp; 0xF</c>,
/// <c>value = d * scale * q2 - dmin * dmCoef</c>.
/// </para>
/// <para>
/// The fixture quantiser is intentionally NOT placed in <c>DotLLM.Cpu</c> —
/// production code does not need to write Q2_K, and shipping a quantiser
/// without proper amax/grid-search optimisation would invite misuse.
/// </para>
/// </remarks>
internal static unsafe class Q2KFixture
{
    public const int Q2KGroupSize = 256;
    public const int Q2KBlockBytes = 84;
    public const int SubBlockSize = 16;
    public const int NumSubBlocks = 16;

    /// <summary>Generate a uniform-random FP32 array in [-range, range).</summary>
    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// Quantise an <c>[m, k]</c> row-major FP32 matrix to Q2_K bytes (one
    /// 84-byte super-block per 256 elements per row). Layout matches
    /// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ2_K</c> byte-for-byte.
    /// </summary>
    /// <remarks>
    /// Per sub-block of 16 elements: pick <c>min_j</c> = sub-block minimum,
    /// <c>max_j</c> = sub-block maximum, then quantise to
    /// <c>round((x - min_j) / step)</c> with <c>step = (max_j - min_j) / 3</c>
    /// (Q2_K nibbles are 0..3). The 16 per-sub-block (scale, dmCoef) pairs are
    /// then 4-bit-quantised against shared <c>d</c> / <c>dmin</c> so the
    /// reconstructed <c>value = d*scale*q2 - dmin*dmCoef</c> approximates the
    /// source. Not production-quality (no rmse search) — only needs to
    /// round-trip cleanly through <c>DequantizeQ2_K</c> for the kernel parity
    /// tests to test the kernel rather than the fixture.
    /// </remarks>
    public static byte[] QuantizeRows(float[] src, int m, int k)
    {
        if ((k % Q2KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q2KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q2KGroupSize;
        int rowBytes = blocksPerRow * Q2KBlockBytes;
        var dst = new byte[m * rowBytes];

        Span<float> subScales = stackalloc float[NumSubBlocks];
        Span<float> subMins   = stackalloc float[NumSubBlocks];
        Span<byte>  q2vals    = stackalloc byte[Q2KGroupSize];

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
                        rowSrc + b * Q2KGroupSize,
                        rowDst + b * Q2KBlockBytes,
                        subScales, subMins, q2vals);
                }
            }
        }
        return dst;
    }

    private static unsafe void QuantizeSuperBlock(
        float* srcSuper, byte* dstSuper,
        Span<float> subScales, Span<float> subMins, Span<byte> q2vals)
    {
        // Pass 1: per-sub-block (scale, min) at full FP32 precision.
        for (int j = 0; j < NumSubBlocks; j++)
        {
            float* sub = srcSuper + j * SubBlockSize;
            float minV = sub[0];
            float maxV = sub[0];
            for (int i = 1; i < SubBlockSize; i++)
            {
                float v = sub[i];
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
            }
            float range = maxV - minV;
            float sc = range > 0 ? range / 3.0f : 0.0f;
            float mn = -minV;
            subScales[j] = sc;
            subMins[j] = mn;
        }

        // Pass 2: choose shared d / dmin so per-sub-block (sc, mn) fit in 4 bits.
        float scAmax = 0, mnAmax = 0;
        for (int j = 0; j < NumSubBlocks; j++)
        {
            float a = MathF.Abs(subScales[j]); if (a > scAmax) scAmax = a;
            float b = MathF.Abs(subMins[j]);   if (b > mnAmax) mnAmax = b;
        }
        float dF    = scAmax > 0 ? scAmax / 15.0f : 0.0f;
        float dminF = mnAmax > 0 ? mnAmax / 15.0f : 0.0f;

        // Pass 3: round each per-sub-block scale and min to 4-bit unsigned.
        Span<byte>  sc4    = stackalloc byte[NumSubBlocks];
        Span<byte>  mn4    = stackalloc byte[NumSubBlocks];
        Span<float> scUsed = stackalloc float[NumSubBlocks];
        Span<float> mnUsed = stackalloc float[NumSubBlocks];
        for (int j = 0; j < NumSubBlocks; j++)
        {
            int sQ = dF > 0 ? (int)MathF.Round(subScales[j] / dF) : 0;
            int mQ = dminF > 0 ? (int)MathF.Round(subMins[j] / dminF) : 0;
            sQ = Math.Clamp(sQ, 0, 15);
            mQ = Math.Clamp(mQ, 0, 15);
            sc4[j]   = (byte)sQ;
            mn4[j]   = (byte)mQ;
            scUsed[j] = dF * sQ;
            mnUsed[j] = dminF * mQ;
        }

        // Pass 4: quantise each element to a 2-bit unsigned q2 using (sc_used, mn_used).
        // Decode formula is x ≈ sc * q2 - mn → q2 ≈ (x + mn) / sc.
        for (int j = 0; j < NumSubBlocks; j++)
        {
            float sc = scUsed[j];
            float mn = mnUsed[j];
            float* sub = srcSuper + j * SubBlockSize;
            int outBase = j * SubBlockSize;
            for (int i = 0; i < SubBlockSize; i++)
            {
                int q;
                if (sc > 0)
                {
                    int v = (int)MathF.Round((sub[i] + mn) / sc);
                    q = Math.Clamp(v, 0, 3);
                }
                else
                {
                    q = 0;
                }
                q2vals[outBase + i] = (byte)q;
            }
        }

        // Pass 5: write block bytes.
        // scales[16] @ 0: low nibble = sc4, high nibble = mn4.
        for (int j = 0; j < NumSubBlocks; j++)
        {
            dstSuper[j] = (byte)((sc4[j] & 0xF) | ((mn4[j] & 0xF) << 4));
        }
        // qs[64] @ 16: 4 elements per byte.
        byte* qsPtr = dstSuper + 16;
        for (int b = 0; b < 64; b++)
        {
            int t0 = b * 4;
            byte packed = (byte)(
                (q2vals[t0 + 0] & 3) |
                ((q2vals[t0 + 1] & 3) << 2) |
                ((q2vals[t0 + 2] & 3) << 4) |
                ((q2vals[t0 + 3] & 3) << 6));
            qsPtr[b] = packed;
        }
        // d / dmin @ 80, 82.
        Unsafe.WriteUnaligned(dstSuper + 80, (Half)dF);
        Unsafe.WriteUnaligned(dstSuper + 82, (Half)dminF);
    }

    /// <summary>Dequantise an entire Q2_K blob to FP32 via the CPU oracle.
    /// Used by the dequant-kernel parity test.</summary>
    public static float[] CpuDequantizeQ2K(byte[] q2kBytes, int totalElements)
    {
        var dst = new float[totalElements];
        fixed (byte* p = q2kBytes)
        {
            Dequantize.ToFloat32((nint)p, totalElements, QuantizationType.Q2_K, dst);
        }
        return dst;
    }

    /// <summary>
    /// Sanity check: dequantise the fixture bytes via the CPU oracle and
    /// confirm the round-trip drift versus the source FP32 is within
    /// expectations for Q2_K (~5-10% of dynamic range — Q2_K's 2-bit
    /// resolution is the lowest of the K-quants).
    /// </summary>
    public static void AssertFixtureRoundtrip(float[] srcF32, byte[] q2kBytes, int m, int k)
    {
        var dequant = new float[m * k];
        fixed (byte* p = q2kBytes)
        fixed (float* d = dequant)
        {
            Dequantize.ToFloat32((nint)p, (long)m * k, QuantizationType.Q2_K,
                new Span<float>(d, m * k));
        }

        // Aggregate over the whole matrix — per-row maxRel is too noisy with
        // 16-element sub-blocks of Q2_K (a sub-block of mostly near-zero values
        // can explode per-row rel L2 even when per-element drift is in the
        // expected ±step/2 budget for 2-bit quants).
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
            $"Q2_K fixture aggregate round-trip drift too large: rel={relAgg:G4} (m={m}, k={k}). " +
            "Fixture quantiser is likely mis-packing the scales nibbles or the qs[] 2-bit fields.");
    }

    /// <summary>
    /// Scalar CPU GEMV reference: reads the same Q2_K bytes the GPU sees,
    /// dequantises on the fly, dots against FP32 <c>x</c>. Block-sequential
    /// reduction order — the GPU shader uses a workgroup tree reduce, which
    /// produces the small-but-nonzero drift covered by the abs/rel tolerances
    /// in the test.
    /// </summary>
    public static float[] CpuGemvQ2K(byte[] weightsQ2K, float[] x, int m, int k)
    {
        if ((k % Q2KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q2KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q2KGroupSize;
        int rowBytes = blocksPerRow * Q2KBlockBytes;
        var result = new float[m];

        fixed (byte* wPtr = weightsQ2K)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block   = rowBase + b * Q2KBlockBytes;
                    byte* scales  = block;
                    byte* qs      = block + 16;
                    float d       = (float)Unsafe.ReadUnaligned<Half>(block + 80);
                    float dmin    = (float)Unsafe.ReadUnaligned<Half>(block + 82);
                    int xBase = b * Q2KGroupSize;

                    for (int sub = 0; sub < 16; sub++)
                    {
                        int scByte = scales[sub];
                        float sc = d * (scByte & 0xF);
                        float mn = dmin * ((scByte >> 4) & 0xF);
                        int outBase = xBase + sub * 16;
                        for (int l = 0; l < 16; l++)
                        {
                            int t = sub * 16 + l;
                            int qByte = qs[t >> 2];
                            int q2 = (qByte >> ((t & 3) * 2)) & 0x3;
                            float w = sc * q2 - mn;
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
    /// Scalar CPU GEMM reference: <c>C[N, M] = B[N, K] @ W_q2k[M, K]^T</c>.
    /// Reads the same Q2_K bytes the GPU sees.
    /// </summary>
    public static float[] CpuGemmQ2K(byte[] weightsQ2K, float[] inputB, int m, int k, int n)
    {
        if ((k % Q2KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q2KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q2KGroupSize;
        int rowBytes = blocksPerRow * Q2KBlockBytes;
        var result = new float[n * m];

        fixed (byte* wPtr = weightsQ2K)
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
                        byte* block  = rowBaseW + bb * Q2KBlockBytes;
                        byte* scales = block;
                        byte* qs     = block + 16;
                        float d      = (float)Unsafe.ReadUnaligned<Half>(block + 80);
                        float dmin   = (float)Unsafe.ReadUnaligned<Half>(block + 82);
                        int xBase = bb * Q2KGroupSize;

                        for (int sub = 0; sub < 16; sub++)
                        {
                            int scByte = scales[sub];
                            float sc = d * (scByte & 0xF);
                            float mn = dmin * ((scByte >> 4) & 0xF);
                            int outBase = xBase + sub * 16;
                            for (int l = 0; l < 16; l++)
                            {
                                int tIdx = sub * 16 + l;
                                int qByte = qs[tIdx >> 2];
                                int q2 = (qByte >> ((tIdx & 3) * 2)) & 0x3;
                                float w = sc * q2 - mn;
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
