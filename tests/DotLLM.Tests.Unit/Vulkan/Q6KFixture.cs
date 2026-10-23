using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only Q6_K fixture helpers shared by the GEMV / GEMM kernel tests and
/// any future model-level Q6_K parity tests. Sibling of <see cref="Q4KFixture"/>
/// and <see cref="Q5KFixture"/>.
/// </summary>
/// <remarks>
/// <para>
/// Why this exists: the production CPU side has full Q6_K dequantise + matmul
/// kernels (see <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ6_K</c>) but
/// no F32-to-Q6_K quantiser — production loaders only ingest pre-quantised
/// GGUF data. For Vulkan parity we need to round-trip arbitrary FP32 fixtures
/// into Q6_K bytes, so this file contains a minimal scalar quantiser that
/// produces the same byte layout as llama.cpp's <c>quantize_row_q6_K_ref</c>:
/// 256-element super-blocks of 210 bytes laid out as
/// <c>[ql[128]][qh[64]][scales[16]:int8][d:fp16]</c>.
/// </para>
/// <para>
/// Q6_K is structurally simpler than Q4_K / Q5_K on the metadata side:
/// scale-only reconstruction (no <c>dmin</c> / no min table), 16 signed
/// <c>int8</c> scales (no 6-bit packed scale array), and signed quants in
/// <c>-32..31</c>. The byte-extraction is more intricate, however: each
/// super-block is processed in two 128-element halves, each with 4 groups of
/// 32 elements that share a 32-byte <c>qh</c> slab. See
/// <c>DequantizeQ6_KScalar</c> for the canonical loop and
/// <c>matmul_q6_k_gemv_f32.comp</c> for the shader replica.
/// </para>
/// <para>
/// The fixture quantiser is intentionally NOT placed in <c>DotLLM.Cpu</c> —
/// production code does not need to write Q6_K, and shipping a quantiser
/// without proper amax/grid-search optimisation would invite misuse.
/// </para>
/// </remarks>
internal static unsafe class Q6KFixture
{
    public const int Q6KGroupSize = 256;
    public const int Q6KBlockBytes = 210;
    public const int SubBlockSize = 16;     // one int8 scale per 16 elements
    public const int NumSubBlocks = 16;     // sub-blocks per super-block

    /// <summary>Generate a uniform-random FP32 array in [-range, range).</summary>
    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>
    /// Quantise an <c>[m, k]</c> row-major FP32 matrix to Q6_K bytes (one
    /// 210-byte super-block per 256 elements per row). Layout matches
    /// <c>DotLLM.Cpu.Kernels.DequantizeKQuants.DequantizeQ6_KScalar</c>
    /// byte-for-byte.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Algorithm (mirrors llama.cpp's <c>quantize_row_q6_K_ref</c>):
    /// </para>
    /// <list type="number">
    ///   <item>For each 16-element sub-block, find the value with the largest
    ///   absolute value (preserving sign). Per-sub-block scale (FP32) is
    ///   <c>max_signed / -32</c> — the canonical signed-amax fit for a 6-bit
    ///   signed range of <c>-32..31</c>. (Using <c>-32</c> rather than
    ///   <c>+31</c> as the divisor matches llama.cpp's choice of pinning the
    ///   most-negative quant to the actual max.)</item>
    ///   <item>Find the largest <c>|scale|</c> across the 16 sub-blocks; pick
    ///   global <c>d = max_abs_scale / 127</c> so <c>round(scale_j / d)</c>
    ///   fits in <c>int8</c> with one bit of headroom (the canonical Q6_K
    ///   scaling).</item>
    ///   <item>Quantise each element: <c>q = round(x_i / (d * int_scale[j])) +
    ///   32</c>, clamp to <c>0..63</c>, store as 6 unsigned bits split into
    ///   the low 4 (in <c>ql</c>) and high 2 (in <c>qh</c>) groups dictated by
    ///   the <c>(half, group)</c> layout described in
    ///   <c>matmul_q6_k_gemv_f32.comp</c>.</item>
    /// </list>
    /// <para>
    /// This is NOT a production-quality quantiser — it does no rmse search
    /// across candidate (min, max) windows, no clipping calibration. It
    /// produces values close enough to the source FP32 that the round-trip
    /// dequant is well within the abs 5e-3 / rel 1e-3 tolerance the Vulkan
    /// parity tests demand. The CPU oracle <c>DequantizeQ6_KScalar</c> is the
    /// authoritative reader; this code only needs to write bytes that
    /// round-trip cleanly through it.
    /// </para>
    /// </remarks>
    public static byte[] QuantizeRows(float[] src, int m, int k)
    {
        if ((k % Q6KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q6KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q6KGroupSize;
        int rowBytes = blocksPerRow * Q6KBlockBytes;
        var dst = new byte[m * rowBytes];

        Span<float> subBlockScalesF = stackalloc float[NumSubBlocks];
        Span<sbyte> subBlockScalesI = stackalloc sbyte[NumSubBlocks];
        Span<byte> q6Vals = stackalloc byte[Q6KGroupSize]; // 0..63 per element (signed value + 32)

        fixed (float* srcPtr = src)
        fixed (byte* dstPtr = dst)
        {
            for (int row = 0; row < m; row++)
            {
                float* rowSrc = srcPtr + (long)row * k;
                byte* rowDst = dstPtr + (long)row * rowBytes;

                for (int b = 0; b < blocksPerRow; b++)
                {
                    float* superSrc = rowSrc + b * Q6KGroupSize;
                    byte* superDst = rowDst + b * Q6KBlockBytes;
                    QuantizeSuperBlock(superSrc, superDst, subBlockScalesF, subBlockScalesI, q6Vals);
                }
            }
        }
        return dst;
    }

    /// <summary>
    /// Quantises one Q6_K super-block (256 source floats -> 210 destination bytes).
    /// </summary>
    private static unsafe void QuantizeSuperBlock(
        float* srcSuper, byte* dstSuper,
        Span<float> subScalesF, Span<sbyte> subScalesI, Span<byte> q6Vals)
    {
        // ── Pass 1: per-sub-block signed-amax scale at FP32. ──
        // For each 16-element sub-block, find the signed value with the largest
        // absolute value. Per-sub-block float scale = max_signed / -32, matching
        // llama.cpp's choice of pinning the maximum-magnitude element to q = -32.
        for (int s = 0; s < NumSubBlocks; s++)
        {
            float* subSrc = srcSuper + s * SubBlockSize;
            float maxAbs = 0;
            float maxSigned = 0;
            for (int i = 0; i < SubBlockSize; i++)
            {
                float v = subSrc[i];
                float av = MathF.Abs(v);
                if (av > maxAbs)
                {
                    maxAbs = av;
                    maxSigned = v;
                }
            }
            // scale_f = maxSigned / -32 — sub-block dequant is x = scale_f * q,
            // q in [-32, 31]; we want q = -32 to land on maxSigned.
            subScalesF[s] = maxAbs > 0 ? maxSigned / -32.0f : 0.0f;
        }

        // ── Pass 2: pick global d and 16 int8 scales. ──
        // d = max(|sub_scale|) / 127 so int_scale fits in int8.
        float scAmax = 0;
        for (int s = 0; s < NumSubBlocks; s++)
        {
            float a = MathF.Abs(subScalesF[s]);
            if (a > scAmax) scAmax = a;
        }
        float d = scAmax > 0 ? scAmax / 127.0f : 0.0f;
        float dInv = d > 0 ? 1.0f / d : 0.0f;

        for (int s = 0; s < NumSubBlocks; s++)
        {
            int sQ = (int)MathF.Round(subScalesF[s] * dInv);
            if (sQ > 127) sQ = 127;
            else if (sQ < -128) sQ = -128;
            subScalesI[s] = (sbyte)sQ;
        }

        // ── Pass 3: quantise each element to 6-bit unsigned (signed_value + 32). ──
        for (int s = 0; s < NumSubBlocks; s++)
        {
            float effectiveScale = d * subScalesI[s];
            float* subSrc = srcSuper + s * SubBlockSize;
            int outBase = s * SubBlockSize;
            if (MathF.Abs(effectiveScale) <= 0)
            {
                // Degenerate sub-block — encode all zeros (q = 0 → signed -32, but
                // a zero-effective-scale dequants to zero regardless, so any
                // 6-bit value works; pick 32 (signed 0) for cleanliness).
                for (int i = 0; i < SubBlockSize; i++) q6Vals[outBase + i] = 32;
                continue;
            }
            float invES = 1.0f / effectiveScale;
            for (int i = 0; i < SubBlockSize; i++)
            {
                int qSigned = (int)MathF.Round(subSrc[i] * invES);
                if (qSigned > 31) qSigned = 31;
                else if (qSigned < -32) qSigned = -32;
                q6Vals[outBase + i] = (byte)(qSigned + 32);  // 0..63
            }
        }

        // ── Pass 4: write block bytes. ──
        // Layout (per DequantizeQ6_KScalar):
        //   ql[0..127]  — low 4 bits, two per byte
        //   qh[128..191] — high 2 bits per element, four per byte
        //   scales[192..207] — int8 scales, one per 16-element sub-block
        //   d[208..209] — fp16 global delta
        //
        // The (ql, qh) packing across 4-element groups within a 16-element
        // sub-block follows the byte-exact mapping documented in the GEMV /
        // GEMM shaders. Given a global element index `i` in [0, 256):
        //   half  = i / 128
        //   local = i - half*128         // 0..127
        //   group = local / 32           // 0..3
        //   l     = local % 32           // 0..31
        //   ql_idx = half*64 + (group&1)*32 + l
        //   qh_idx = half*32 + l
        //   nibble = (group >= 2) → high, else low
        //   shift  = group * 2
        // Inverse (write side): for each output position (half, group, l),
        // compute the correct (ql_idx, qh_idx, nibble selector, qh shift) and
        // OR the appropriate bits in.
        byte* qlPtr = dstSuper;
        byte* qhPtr = dstSuper + 128;
        byte* scalesPtr = dstSuper + 192;

        // Zero ql / qh first (we OR bits in below).
        new Span<byte>(qlPtr, 128).Clear();
        new Span<byte>(qhPtr, 64).Clear();

        for (int hf = 0; hf < 2; hf++)
        {
            int qlOff = hf * 64;
            int qhOff = hf * 32;
            int outHalfBase = hf * 128;
            for (int group = 0; group < 4; group++)
            {
                bool useHigh = group >= 2;
                int qlLane = (group & 1) * 32;
                int shift = group * 2;
                for (int l = 0; l < 32; l++)
                {
                    int outIdx = outHalfBase + group * 32 + l;
                    int q6 = q6Vals[outIdx]; // 0..63
                    int lo4 = q6 & 0xF;
                    int hi2 = (q6 >> 4) & 0x3;

                    int qlIdx = qlOff + qlLane + l;
                    int qhIdx = qhOff + l;

                    if (useHigh)
                        qlPtr[qlIdx] |= (byte)(lo4 << 4);
                    else
                        qlPtr[qlIdx] |= (byte)lo4;
                    qhPtr[qhIdx] |= (byte)(hi2 << shift);
                }
            }
        }

        // scales — int8 directly.
        for (int s = 0; s < NumSubBlocks; s++)
        {
            scalesPtr[s] = (byte)subScalesI[s];
        }

        // d — fp16 at offset 208.
        Unsafe.WriteUnaligned(dstSuper + 208, (Half)d);
    }

    /// <summary>
    /// Sanity check: dequantise the fixture bytes via the CPU oracle and
    /// confirm the round-trip drift versus the source FP32 is within
    /// expectations for Q6_K (~1.5% of dynamic range — Q6_K's 6-bit signed
    /// resolution is the highest of the K-quants). This is a structural
    /// safeguard so that later kernel-output asserts are testing the kernel,
    /// not a broken fixture quantiser.
    /// </summary>
    public static void AssertFixtureRoundtrip(float[] srcF32, byte[] q6kBytes, int m, int k)
    {
        var dequant = new float[m * k];
        fixed (byte* p = q6kBytes)
        fixed (float* d = dequant)
        {
            // The CPU oracle dequantises a flat element span; all m rows share
            // the same per-row stride so we can call it once on the full blob.
            Dequantize.ToFloat32((nint)p, (long)m * k, QuantizationType.Q6_K,
                new Span<float>(d, m * k));
        }

        // Compute relative L2 drift per row. Same generous bar as Q4_K / Q5_K
        // — it's a structural check, not a tightness test.
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
            $"Q6_K fixture round-trip drift too large: maxRel={maxRel:G4} (m={m}, k={k}). " +
            "Fixture quantiser is likely mis-packing the (ql, qh) groups, the int8 scales, or the fp16 delta.");
    }

    /// <summary>
    /// Scalar CPU GEMV reference: reads the same Q6_K bytes the GPU sees,
    /// dequantises on the fly, dots against FP32 <c>x</c>. Block-sequential
    /// reduction order — the GPU shader uses a workgroup tree reduce, which
    /// produces the small-but-nonzero drift covered by the abs/rel tolerances
    /// in the test.
    /// </summary>
    public static float[] CpuGemvQ6K(byte[] weightsQ6K, float[] x, int m, int k)
    {
        if ((k % Q6KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q6KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q6KGroupSize;
        int rowBytes = blocksPerRow * Q6KBlockBytes;
        var result = new float[m];

        fixed (byte* wPtr = weightsQ6K)
        {
            for (int row = 0; row < m; row++)
            {
                byte* rowBase = wPtr + (long)row * rowBytes;
                float sum = 0;
                for (int b = 0; b < blocksPerRow; b++)
                {
                    byte* block = rowBase + b * Q6KBlockBytes;
                    byte* ql = block;
                    byte* qh = block + 128;
                    sbyte* scales = (sbyte*)(block + 192);
                    float d = (float)Unsafe.ReadUnaligned<Half>(block + 208);
                    int xBase = b * Q6KGroupSize;

                    // Mirror DequantizeQ6_KScalar's loop structure exactly.
                    for (int hf = 0; hf < 2; hf++)
                    {
                        int qlOff = hf * 64;
                        int qhOff = hf * 32;
                        int scOff = hf * 8;
                        int outHalfBase = hf * 128;
                        for (int l = 0; l < 32; l++)
                        {
                            int isc = l / 16;
                            int q1 = ((ql[qlOff + l]      & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                            int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                            int q3 = ((ql[qlOff + l]      >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                            int q4 = ((ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                            float w1 = d * scales[scOff + isc]     * q1;
                            float w2 = d * scales[scOff + isc + 2] * q2;
                            float w3 = d * scales[scOff + isc + 4] * q3;
                            float w4 = d * scales[scOff + isc + 6] * q4;

                            sum += w1 * x[xBase + outHalfBase + l]
                                 + w2 * x[xBase + outHalfBase + l + 32]
                                 + w3 * x[xBase + outHalfBase + l + 64]
                                 + w4 * x[xBase + outHalfBase + l + 96];
                        }
                    }
                }
                result[row] = sum;
            }
        }
        return result;
    }

    /// <summary>
    /// Scalar CPU GEMM reference: <c>C[N, M] = B[N, K] @ W_q6k[M, K]^T</c>.
    /// Reads the same Q6_K bytes the GPU sees.
    /// </summary>
    public static float[] CpuGemmQ6K(byte[] weightsQ6K, float[] inputB, int m, int k, int n)
    {
        if ((k % Q6KGroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Q6KGroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Q6KGroupSize;
        int rowBytes = blocksPerRow * Q6KBlockBytes;
        var result = new float[n * m];

        fixed (byte* wPtr = weightsQ6K)
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
                        byte* block = rowBaseW + bb * Q6KBlockBytes;
                        byte* ql = block;
                        byte* qh = block + 128;
                        sbyte* scales = (sbyte*)(block + 192);
                        float d = (float)Unsafe.ReadUnaligned<Half>(block + 208);
                        int xBase = bb * Q6KGroupSize;

                        for (int hf = 0; hf < 2; hf++)
                        {
                            int qlOff = hf * 64;
                            int qhOff = hf * 32;
                            int scOff = hf * 8;
                            int outHalfBase = hf * 128;
                            for (int l = 0; l < 32; l++)
                            {
                                int isc = l / 16;
                                int q1 = ((ql[qlOff + l]      & 0xF) | (((qh[qhOff + l] >> 0) & 3) << 4)) - 32;
                                int q2 = ((ql[qlOff + l + 32] & 0xF) | (((qh[qhOff + l] >> 2) & 3) << 4)) - 32;
                                int q3 = ((ql[qlOff + l]      >> 4) | (((qh[qhOff + l] >> 4) & 3) << 4)) - 32;
                                int q4 = ((ql[qlOff + l + 32] >> 4) | (((qh[qhOff + l] >> 6) & 3) << 4)) - 32;

                                float w1 = d * scales[scOff + isc]     * q1;
                                float w2 = d * scales[scOff + isc + 2] * q2;
                                float w3 = d * scales[scOff + isc + 4] * q3;
                                float w4 = d * scales[scOff + isc + 6] * q4;

                                sum += w1 * inputB[bRowBase + xBase + outHalfBase + l]
                                     + w2 * inputB[bRowBase + xBase + outHalfBase + l + 32]
                                     + w3 * inputB[bRowBase + xBase + outHalfBase + l + 64]
                                     + w4 * inputB[bRowBase + xBase + outHalfBase + l + 96];
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
