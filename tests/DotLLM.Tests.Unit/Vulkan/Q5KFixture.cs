using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only Q5_K fixture helpers shared by the GEMV / GEMM kernel tests and
/// any future model-level Q5_K parity tests. Sibling of <see cref="Q4KFixture"/>.
/// </summary>
/// <remarks>
/// <para>
/// Why this exists: the production CPU side has full Q5_K dequantise + matmul
/// kernels (see <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ5_K</c>) but
/// no F32-to-Q5_K quantiser — production loaders only ingest pre-quantised
/// GGUF data. For Vulkan parity we need to round-trip arbitrary FP32 fixtures
/// into Q5_K bytes, so this file contains a minimal scalar quantiser that
/// produces the same byte layout as llama.cpp's <c>quantize_row_q5_K</c>:
/// 256-element super-blocks of 176 bytes laid out as
/// <c>[d:fp16][dmin:fp16][scales[12]][qh[32]][qs[128]]</c> with the 8x6-bit
/// scales/mins packed via the inverse of <c>UnpackQ4Q5Scales</c> (shared with
/// Q4_K) and 32 high-bit bytes laid out so that bit <c>j</c> of <c>qh[i]</c>
/// is the 5th bit of element <c>j*32 + i</c> (matches the CPU oracle's
/// <c>(qh[i] &gt;&gt; j) &amp; 1</c> read).
/// </para>
/// <para>
/// The fixture quantiser is intentionally NOT placed in <c>DotLLM.Cpu</c> —
/// production code does not need to write Q5_K, and shipping a quantiser
/// without proper amax/grid-search optimisation would invite misuse.
/// </para>
/// </remarks>
internal static unsafe class Q5KFixture
{
    public const int Q5KGroupSize = 256;
    public const int Q5KBlockBytes = 176;
    public const int SubBlockSize = 32;
    public const int NumSubBlocks = 8;

    /// <summary>Generate a uniform-random FP32 array in [-range, range).</summary>
    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// Quantise an <c>[m, k]</c> row-major FP32 matrix to Q5_K bytes (one
    /// 176-byte super-block per 256 elements per row). Layout matches
    /// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ5_KScalar</c>
    /// byte-for-byte.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The quantiser is a straightforward affine fit per sub-block of 32
    /// elements: choose <c>min_j</c> = sub-block minimum, <c>max_j</c> =
    /// sub-block maximum, then quantise to <c>(x - min_j) / step</c> with
    /// step = <c>(max_j - min_j) / 31</c> (Q5_K has 5-bit unsigned values
    /// 0..31 — twice the resolution of Q4_K). The 8 per-sub-block (scale, min)
    /// pairs are then 6-bit-quantised to the super-block's <c>d</c> /
    /// <c>dmin</c> via amax, packed via the inverse of
    /// <see cref="DotLLM.Cpu.Kernels.Dequantize.UnpackQ4Q5Scales"/>.
    /// </para>
    /// <para>
    /// This is NOT a production-quality quantiser — it does no rmse search
    /// across candidate (min, max) windows, no clipping calibration, no L2
    /// minimisation. It produces values close enough to the source FP32 that
    /// the round-trip dequant is well within the abs 5e-3 / rel 1e-3
    /// tolerance the Vulkan parity tests demand, and that's all this fixture
    /// needs to do. The CPU oracle <c>DequantizeQ5_KScalar</c> is the
    /// authoritative reader; this code only needs to write bytes that
    /// round-trip cleanly through it.
    /// </para>
    /// </remarks>
    public static byte[] QuantizeRows(float[] src, int m, int k)
    {
        if ((k % Q5KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5KGroupSize;
        int rowBytes = blocksPerRow * Q5KBlockBytes;
        var dst = new byte[m * rowBytes];

        Span<float> subBlockScales = stackalloc float[NumSubBlocks];
        Span<float> subBlockMins = stackalloc float[NumSubBlocks];
        Span<byte> q5Vals = stackalloc byte[Q5KGroupSize]; // 0..31 per element

        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
            {
                float* rowSrc = srcPtr + (long)row * k;
                byte* rowDst = dstPtr + (long)row * rowBytes;

                for (int b = 0; b < blocksPerRow; b++)
                {
                    float* superSrc = rowSrc + b * Q5KGroupSize;
                    byte* superDst = rowDst + b * Q5KBlockBytes;
                    QuantizeSuperBlock(superSrc, superDst, subBlockScales, subBlockMins, q5Vals);
                }
            }
        }
        return dst;
    }

    /// <summary>
    /// Quantises one Q5_K super-block (256 source floats -> 176 destination bytes).
    /// </summary>
    private static unsafe void QuantizeSuperBlock(
        float* srcSuper, byte* dstSuper,
        Span<float> subScales, Span<float> subMins, Span<byte> q5Vals)
    {
        // ── Pass 1: per-sub-block (scale, min) at full FP32 precision ──
        for (int j = 0; j < NumSubBlocks; j++)
        {
            float* subSrc = srcSuper + j * SubBlockSize;
            float minV = subSrc[0];
            float maxV = subSrc[0];
            for (int i = 1; i < SubBlockSize; i++)
            {
                float v = subSrc[i];
                if (v < minV) minV = v;
                if (v > maxV) maxV = v;
            }
            float range = maxV - minV;
            // Q5_K values are unsigned 0..31. We'll scale (x - min) by step
            // such that step * 31 ≈ range. Storage form is
            //   x ≈ d*scale_j*q5 - dmin*min_j
            // We pre-compute (sc_float, mn_float) such that
            //   x_i ≈ sc_float*q5 - mn_float
            // with q5 = round((x_i - minV) / sc_float), giving
            //   x_i ≈ sc_float*q5 + minV  →  mn_float = -minV.
            float sc = range > 0 ? range / 31.0f : 0.0f;
            float mn = -minV;
            subScales[j] = sc;
            subMins[j] = mn;
        }

        // ── Pass 2: quantise each (scale, min) to 6-bit unsigned via shared d / dmin. ──
        // d is chosen so that scale_j / d fits in 6 bits (0..63) for the largest scale.
        float scAmax = 0.0f;
        float mnAmax = 0.0f;
        for (int j = 0; j < NumSubBlocks; j++)
        {
            float a = MathF.Abs(subScales[j]);
            if (a > scAmax) scAmax = a;
            float b = MathF.Abs(subMins[j]);
            if (b > mnAmax) mnAmax = b;
        }
        float dF = scAmax > 0 ? scAmax / 63.0f : 0.0f;
        float dminF = mnAmax > 0 ? mnAmax / 63.0f : 0.0f;

        // ── Pass 3: round each per-sub-block scale and min to 6-bit, then re-derive
        //   (sc_used, mn_used) from the rounded values for the q5 quantisation pass.
        //   This avoids drift between encoder and decoder when sc/mn cannot be
        //   represented exactly in 6 bits. ──
        Span<byte> sc6 = stackalloc byte[NumSubBlocks];
        Span<byte> mn6 = stackalloc byte[NumSubBlocks];
        Span<float> scUsed = stackalloc float[NumSubBlocks];
        Span<float> mnUsed = stackalloc float[NumSubBlocks];
        for (int j = 0; j < NumSubBlocks; j++)
        {
            int sQ = dF > 0 ? (int)MathF.Round(subScales[j] / dF) : 0;
            int mQ = dminF > 0 ? (int)MathF.Round(subMins[j] / dminF) : 0;
            sQ = Math.Clamp(sQ, 0, 63);
            mQ = Math.Clamp(mQ, 0, 63);
            sc6[j] = (byte)sQ;
            mn6[j] = (byte)mQ;
            scUsed[j] = dF * sQ;
            mnUsed[j] = dminF * mQ;
        }

        // ── Pass 4: quantise each element to a 5-bit unsigned value using
        //   (sc_used, mn_used). q5 target: round((x + mn_used) / sc_used). ──
        for (int j = 0; j < NumSubBlocks; j++)
        {
            float sc = scUsed[j];
            float mn = mnUsed[j];
            float* subSrc = srcSuper + j * SubBlockSize;
            int outBase = j * SubBlockSize;
            for (int i = 0; i < SubBlockSize; i++)
            {
                int q;
                if (sc > 0)
                {
                    int v = (int)MathF.Round((subSrc[i] + mn) / sc);
                    q = Math.Clamp(v, 0, 31);
                }
                else
                {
                    q = 0;
                }
                q5Vals[outBase + i] = (byte)q;
            }
        }

        // ── Pass 5: write block bytes. ──
        Unsafe.WriteUnaligned(dstSuper, (Half)dF);
        Unsafe.WriteUnaligned(dstSuper + 2, (Half)dminF);

        // Pack 8 6-bit scales + 8 6-bit mins into 12 bytes — exactly the inverse of
        // DequantizeKQuants.UnpackQ4Q5Scales / llama.cpp's get_scale_min_k4. Same
        // packing as Q4_K (this is why the unpacker function name covers BOTH formats).
        // Sub-blocks 0..3:
        //   bytes [0..3]: low 6 bits = scale_j   (top 2 bits hold scale_{j+4} top bits)
        //   bytes [4..7]: low 6 bits = min_j     (top 2 bits hold min_{j+4} top bits)
        // Sub-blocks 4..7:
        //   bytes [8..11]: low 4 bits = scale_{j+4} bits [0..3]
        //                  high 4 bits = min_{j+4}    bits [0..3]
        //   top 2 bits of scale_{j+4} -> bits [6..7] of bytes [j]   (j in 0..3)
        //   top 2 bits of min_{j+4}   -> bits [6..7] of bytes [j+4] (j in 0..3)
        byte* scalesPtr = dstSuper + 4;
        for (int j = 0; j < 4; j++)
        {
            byte sLow = (byte)(sc6[j] & 0x3F);
            byte mLow = (byte)(mn6[j] & 0x3F);
            byte sHi = (byte)((sc6[j + 4] >> 4) & 0x3); // bits [4..5] of partner -> [6..7] here
            byte mHi = (byte)((mn6[j + 4] >> 4) & 0x3);
            scalesPtr[j]     = (byte)(sLow | (sHi << 6));
            scalesPtr[j + 4] = (byte)(mLow | (mHi << 6));
        }
        for (int j = 4; j < 8; j++)
        {
            byte sLow4 = (byte)(sc6[j] & 0xF);
            byte mLow4 = (byte)(mn6[j] & 0xF);
            scalesPtr[j + 4] = (byte)(sLow4 | (mLow4 << 4));
        }

        // qh[]: 32 bytes, one byte per element-position-within-sub-block (i in 0..31).
        // Bit `j` of qh[i] is the 5th bit of element `j*32 + i` (i.e. position i in
        // sub-block j). This matches the CPU oracle's
        //   bit5 = (qh[i] >> j) & 1
        // read; it is NOT a flat bitfield.
        byte* qhPtr = dstSuper + 16;
        for (int i = 0; i < 32; i++)
        {
            byte qh = 0;
            for (int j = 0; j < NumSubBlocks; j++)
            {
                int q5 = q5Vals[j * SubBlockSize + i];
                int hi = (q5 >> 4) & 1;
                qh |= (byte)(hi << j);
            }
            qhPtr[i] = qh;
        }

        // qs[]: pack low 4 bits of (sub-block-pair, position) into 128 bytes.
        // Sub-blocks (2p, 2p+1) share 32 qs bytes; element i of sub-block 2p
        // goes into the low nibble of qs[p*32 + i], element i of sub-block
        // 2p+1 goes into the high nibble. Same packing as Q4_K.
        byte* qsPtr = dstSuper + 48;
        for (int p = 0; p < 4; p++)
        {
            int evenBase = (2 * p) * SubBlockSize;
            int oddBase = (2 * p + 1) * SubBlockSize;
            for (int i = 0; i < SubBlockSize; i++)
            {
                byte lo = (byte)(q5Vals[evenBase + i] & 0xF);
                byte hi = (byte)(q5Vals[oddBase + i] & 0xF);
                qsPtr[p * SubBlockSize + i] = (byte)(lo | (hi << 4));
            }
        }
    }

    /// <summary>
    /// Sanity check: dequantise the fixture bytes via the CPU oracle and
    /// confirm the round-trip drift versus the source FP32 is within
    /// expectations for Q5_K (~2.5% of dynamic range — twice the resolution
    /// of Q4_K). This is a structural safeguard so that later kernel-output
    /// asserts are testing the kernel, not a broken fixture quantiser.
    /// </summary>
    public static void AssertFixtureRoundtrip(float[] srcF32, byte[] q5kBytes, int m, int k)
    {
        var dequant = new float[m * k];
        fixed (byte* p = q5kBytes)
        fixed (float* d = dequant)
        {
            // The CPU oracle dequantises a flat element span; all m rows share
            // the same per-row stride so we can call it once on the full blob.
            Dequantize.ToFloat32((nint)p, (long)m * k, QuantizationType.Q5_K,
                new Span<float>(d, m * k));
        }

        // Compute relative L2 drift per row. Q5_K's 5-bit resolution is 2x Q4_K so
        // we keep the same generous bar — it's a structural check, not a tightness
        // test (the kernel-vs-CPU parity asserts handle that).
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
        Assert.True(maxRel < 0.10f,
            $"Q5_K fixture round-trip drift too large: maxRel={maxRel:G4} (m={m}, k={k}). " +
            "Fixture quantiser is likely mis-packing the 6-bit scales, qh[] high bits, or qs[] nibbles.");
    }

    /// <summary>
    /// Scalar CPU GEMV reference: reads the same Q5_K bytes the GPU sees,
    /// dequantises on the fly, dots against FP32 <c>x</c>. Block-sequential
    /// reduction order — the GPU shader uses a workgroup tree reduce, which
    /// produces the small-but-nonzero drift covered by the abs/rel tolerances
    /// in the test.
    /// </summary>
    public static float[] CpuGemvQ5K(byte[] weightsQ5K, float[] x, int m, int k)
    {
        if ((k % Q5KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5KGroupSize;
        int rowBytes = blocksPerRow * Q5KBlockBytes;
        var result = new float[m];

        Span<byte> scBuf = stackalloc byte[8];
        Span<byte> mnBuf = stackalloc byte[8];

        fixed (byte* wPtr = weightsQ5K)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Q5KBlockBytes;
                    float d = (float)Unsafe.ReadUnaligned<Half>(block);
                    float dmin = (float)Unsafe.ReadUnaligned<Half>(block + 2);
                    fixed (byte* sc = scBuf)
                    fixed (byte* mn = mnBuf)
                    {
                        Dequantize.UnpackQ4Q5Scales(block + 4, sc, mn);
                    }
                    byte* qh = block + 16;
                    byte* qs = block + 48;
                    int xBase = b * Q5KGroupSize;

                    for (int j = 0; j < 8; j++)
                    {
                        float scF = d * scBuf[j];
                        float mnF = dmin * mnBuf[j];
                        int pairIdx = j / 2;
                        int nibbleHalf = j % 2;
                        int outBase = xBase + j * SubBlockSize;
                        for (int i = 0; i < SubBlockSize; i++)
                        {
                            int qsByte = pairIdx * SubBlockSize + i;
                            int lo4 = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                            int hi = (qh[i] >> j) & 1;
                            int q5 = lo4 | (hi << 4);
                            float w = scF * q5 - mnF;
                            sum += w * x[outBase + i];
                        }
                    }
                }
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>
    /// Scalar CPU GEMM reference: <c>C[N, M] = B[N, K] @ W_q5k[M, K]^T</c>.
    /// Reads the same Q5_K bytes the GPU sees.
    /// </summary>
    public static float[] CpuGemmQ5K(byte[] weightsQ5K, float[] inputB, int m, int k, int n)
    {
        if ((k % Q5KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q5KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q5KGroupSize;
        int rowBytes = blocksPerRow * Q5KBlockBytes;
        var result = new float[n * m];

        Span<byte> scBuf = stackalloc byte[8];
        Span<byte> mnBuf = stackalloc byte[8];

        fixed (byte* wPtr = weightsQ5K)
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
                        byte* block = rowBaseW + bb * Q5KBlockBytes;
                        float d = (float)Unsafe.ReadUnaligned<Half>(block);
                        float dmin = (float)Unsafe.ReadUnaligned<Half>(block + 2);
                        fixed (byte* sc = scBuf)
                        fixed (byte* mn = mnBuf)
                        {
                            Dequantize.UnpackQ4Q5Scales(block + 4, sc, mn);
                        }
                        byte* qh = block + 16;
                        byte* qs = block + 48;
                        int xBase = bb * Q5KGroupSize;

                        for (int j = 0; j < 8; j++)
                        {
                            float scF = d * scBuf[j];
                            float mnF = dmin * mnBuf[j];
                            int pairIdx = j / 2;
                            int nibbleHalf = j % 2;
                            int outBase = xBase + j * SubBlockSize;
                            for (int i = 0; i < SubBlockSize; i++)
                            {
                                int qsByte = pairIdx * SubBlockSize + i;
                                int lo4 = nibbleHalf == 0 ? (qs[qsByte] & 0xF) : (qs[qsByte] >> 4);
                                int hi = (qh[i] >> j) & 1;
                                int q5 = lo4 | (hi << 4);
                                float w = scF * q5 - mnF;
                                sum += w * inputB[bRowBase + outBase + i];
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
