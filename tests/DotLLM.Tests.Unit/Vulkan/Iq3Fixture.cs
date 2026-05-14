using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only IQ3_XXS / IQ3_S fixture helpers shared by the dequant / GEMV /
/// GEMM kernel tests.
/// </summary>
/// <remarks>
/// <para>
/// The production CPU side has full IQ3 dequantise kernels but no F32-to-IQ3
/// quantiser — production loaders only ingest pre-quantised GGUF data. For
/// Vulkan parity we round-trip arbitrary FP32 fixtures through IQ3 bytes.
/// Layouts match <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ3_*</c> and
/// llama.cpp's <c>block_iq3_*</c> byte-for-byte.
/// </para>
/// <para>
/// Per-pair (8 elements) decode reads from two grid rows (g1 for the low
/// 4, g2 for the high 4) plus an 8-bit sign mask. The quantiser does an
/// exhaustive nearest-pair search: each pair tests <c>(grid_count)^2 × 128
/// sign patterns</c>. For IQ3_XXS (256 entries) that is 256 × 256 × 128 ≈
/// 8.4M comparisons per pair; for IQ3_S (512 entries) 512 × 512 × 128 ≈
/// 33.5M. Slow but trivially correct — the kernel parity assertions then
/// exercise the *kernel*, not the fixture.
/// </para>
/// </remarks>
internal static unsafe class Iq3Fixture
{
    public const int Iq3GroupSize = 256;
    public const int Iq3SubBlockSize = 32;
    public const int Iq3NumSubBlocks = 8;
    public const int Iq3PairSize = 8;
    public const int Iq3PairsPerSubBlock = 4;

    public const int Iq3XxsBlockBytes = 98;
    public const int Iq3SBlockBytes = 110;

    public const int Iq3XxsGridSize = 256;
    public const int Iq3SGridSize = 512;

    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>Per-pair best-fit search for IQ3_XXS. Half A (elements 0..3,
    /// sign bits 0..3) is searched freely; half B (elements 4..7, sign bits
    /// 4..7) is searched under a parity constraint so the combined 8-bit
    /// mask is a valid <c>ksigns</c> output. Specifically: bit 7 of the
    /// final mask must equal the XOR of bits 0..6, which means
    /// <c>parity(sigLo) XOR parity(sigHi &amp; 0b0111) == (sigHi &gt;&gt; 3) &amp; 1</c>.
    /// Half-B's 16 sign patterns reduce to the 8 that satisfy this — we
    /// limit the search to those, which preserves quality vs the unconstrained
    /// optimum (since bit 7 = sign of element 7 only, the constraint shifts
    /// at most one element's sign).</summary>
    private static void FindBestPairIq3Xxs(
        ReadOnlySpan<float> targetScaled, ReadOnlySpan<byte> grid,
        out int g1, out int g2, out int signsIdx)
    {
        FindHalf(targetScaled[..4], grid, Iq3XxsGridSize, out g1, out int sigLo);

        int parLo = Parity4(sigLo);
        // Build the 8 valid half-B masks for this sigLo.
        Span<byte> validHigh = stackalloc byte[8];
        for (int i = 0; i < 8; i++)
        {
            int low3 = i & 0x7;
            int parHi3 = Parity3(low3);
            int bit3 = parLo ^ parHi3;
            validHigh[i] = (byte)(low3 | (bit3 << 3));
        }

        FindHalfConstrained(targetScaled[4..], grid, Iq3XxsGridSize, validHigh, out g2, out int sigHi);

        int rawMask = (sigLo & 0xF) | ((sigHi & 0xF) << 4);
        // By construction the mask has correct ksigns parity, so signsIdx
        // is just the low 7 bits.
        signsIdx = rawMask & 0x7F;
    }

    private static int Parity4(int x)
    {
        int p = 0;
        for (int i = 0; i < 4; i++) p ^= (x >> i) & 1;
        return p;
    }

    private static int Parity3(int x)
    {
        int p = 0;
        for (int i = 0; i < 3; i++) p ^= (x >> i) & 1;
        return p;
    }

    private static void FindHalfConstrained(
        ReadOnlySpan<float> target4, ReadOnlySpan<byte> grid, int gridSize,
        ReadOnlySpan<byte> allowedSigns,
        out int gridIdx, out int signs4)
    {
        gridIdx = 0; signs4 = allowedSigns[0];
        float bestErr = float.MaxValue;
        for (int g = 0; g < gridSize; g++)
        {
            int gOff = g * 4;
            for (int si = 0; si < allowedSigns.Length; si++)
            {
                int s = allowedSigns[si];
                float err = 0;
                for (int j = 0; j < 4; j++)
                {
                    float gv = grid[gOff + j];
                    float sign = ((s >> j) & 1) != 0 ? -1f : 1f;
                    float diff = target4[j] - sign * gv;
                    err += diff * diff;
                }
                if (err < bestErr)
                {
                    bestErr = err;
                    gridIdx = g;
                    signs4 = s;
                }
            }
        }
    }

    private static void FindHalf(
        ReadOnlySpan<float> target4, ReadOnlySpan<byte> grid, int gridSize,
        out int gridIdx, out int signs4)
    {
        gridIdx = 0; signs4 = 0;
        float bestErr = float.MaxValue;
        for (int g = 0; g < gridSize; g++)
        {
            int gOff = g * 4;
            for (int s = 0; s < 16; s++)
            {
                float err = 0;
                for (int j = 0; j < 4; j++)
                {
                    float gv = grid[gOff + j];
                    float sign = ((s >> j) & 1) != 0 ? -1f : 1f;
                    float diff = target4[j] - sign * gv;
                    err += diff * diff;
                }
                if (err < bestErr)
                {
                    bestErr = err;
                    gridIdx = g;
                    signs4 = s;
                }
            }
        }
    }

    /// <summary>Per-pair best-fit search for IQ3_S: decomposed search over
    /// each half (g, low-4-signs) independently. IQ3_S writes the full 8-bit
    /// sign byte directly so any of the 256 possible patterns is valid (no
    /// ksigns parity constraint).</summary>
    private static void FindBestPairIq3S(
        ReadOnlySpan<float> targetScaled, ReadOnlySpan<byte> grid,
        out int g1, out int g2, out byte signsByte)
    {
        FindHalf(targetScaled[..4], grid, Iq3SGridSize, out g1, out int sigLo);
        FindHalf(targetScaled[4..], grid, Iq3SGridSize, out g2, out int sigHi);
        signsByte = (byte)((sigLo & 0xF) | ((sigHi & 0xF) << 4));
    }

    // ── IQ3_XXS ────────────────────────────────────────────────────────────

    public static byte[] QuantizeRowsIq3Xxs(float[] src, int m, int k)
    {
        if ((k % Iq3GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq3GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq3GroupSize;
        int rowBytes = blocksPerRow * Iq3XxsBlockBytes;
        var dst = new byte[m * rowBytes];
        ReadOnlySpan<byte> grid = Dequantize.Iq3XxsGrid;

        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int srcOff = row * k + b * Iq3GroupSize;
                int dstOff = row * rowBytes + b * Iq3XxsBlockBytes;
                QuantizeBlockIq3Xxs(src.AsSpan(srcOff, Iq3GroupSize), dst.AsSpan(dstOff, Iq3XxsBlockBytes), grid);
            }
        }
        return dst;
    }

    private static void QuantizeBlockIq3Xxs(ReadOnlySpan<float> src, Span<byte> dst, ReadOnlySpan<byte> grid)
    {
        // Grid byte values are in {0x04, 0x0c, 0x14, 0x1c, 0x24, 0x2c, 0x34, 0x3e};
        // max = 0x3e = 62. Per-pair effective scale: db = d * (0.5 + s4) * 0.5,
        // s4 in [0..15] -> max factor = 7.75 (s4=15). Hence d ≈ amax / (62 * 7.75).
        float amax = 0;
        for (int i = 0; i < src.Length; i++) amax = MathF.Max(amax, MathF.Abs(src[i]));
        float dGuess = amax > 0 ? amax / (62.0f * 7.75f) : 0;
        float invD = dGuess > 0 ? 1.0f / dGuess : 0;

        Half hd = (Half)dGuess;
        Unsafe.WriteUnaligned(ref dst[0], hd);

        Span<int> g1s = stackalloc int[32];        // 8 sub × 4 pairs
        Span<int> g2s = stackalloc int[32];
        Span<int> signs = stackalloc int[32];
        Span<byte> subScales = stackalloc byte[Iq3NumSubBlocks];

        Span<float> pairScaled = stackalloc float[Iq3PairSize];
        for (int ib32 = 0; ib32 < Iq3NumSubBlocks; ib32++)
        {
            float subAmax = 0;
            for (int i = 0; i < Iq3SubBlockSize; i++)
                subAmax = MathF.Max(subAmax, MathF.Abs(src[ib32 * Iq3SubBlockSize + i]));

            int s4 = 15;
            if (dGuess > 0)
            {
                float subTarget = subAmax * invD;
                float scale = subTarget / 62.0f;
                int n = (int)MathF.Round(scale * 2.0f - 0.5f);
                s4 = Math.Clamp(n, 0, 15);
            }
            subScales[ib32] = (byte)s4;

            float dl = dGuess * (0.5f + s4) * 0.5f;
            float invDl = dl != 0 ? 1.0f / dl : 0;

            for (int l = 0; l < Iq3PairsPerSubBlock; l++)
            {
                int pairBase = ib32 * Iq3SubBlockSize + l * Iq3PairSize;
                for (int j = 0; j < Iq3PairSize; j++)
                    pairScaled[j] = src[pairBase + j] * invDl;
                FindBestPairIq3Xxs(pairScaled, grid, out int g1, out int g2, out int s);
                g1s[ib32 * 4 + l] = g1;
                g2s[ib32 * 4 + l] = g2;
                signs[ib32 * 4 + l] = s;
            }
        }

        // qs[64] — 8 grid indices per sub-block (2 per pair, 4 pairs).
        for (int ib32 = 0; ib32 < Iq3NumSubBlocks; ib32++)
        {
            int qsBase = 2 + 8 * ib32;
            for (int l = 0; l < 4; l++)
            {
                dst[qsBase + 2 * l + 0] = (byte)(g1s[ib32 * 4 + l] & 0xff);
                dst[qsBase + 2 * l + 1] = (byte)(g2s[ib32 * 4 + l] & 0xff);
            }
        }

        // scales_and_signs[8 uint32] @ offset 66.
        for (int ib32 = 0; ib32 < Iq3NumSubBlocks; ib32++)
        {
            uint aux32 = 0;
            for (int l = 0; l < 4; l++)
                aux32 |= (uint)(signs[ib32 * 4 + l] & 0x7f) << (7 * l);
            aux32 |= (uint)subScales[ib32] << 28;
            int off = 66 + 4 * ib32;
            dst[off + 0] = (byte)(aux32 & 0xff);
            dst[off + 1] = (byte)((aux32 >> 8) & 0xff);
            dst[off + 2] = (byte)((aux32 >> 16) & 0xff);
            dst[off + 3] = (byte)((aux32 >> 24) & 0xff);
        }
    }

    public static void AssertFixtureRoundtripIq3Xxs(float[] srcF32, byte[] qBytes, int m, int k)
        => AssertRoundtrip(srcF32, qBytes, m, k, QuantizationType.IQ3_XXS, "IQ3_XXS");

    public static float[] CpuDequantizeIq3Xxs(byte[] bytes, int totalElements)
        => CpuDequantize(bytes, totalElements, QuantizationType.IQ3_XXS);

    public static float[] CpuGemvIq3Xxs(byte[] weightsIq3, float[] x, int m, int k)
        => CpuGemvViaDequant(weightsIq3, x, m, k, Iq3XxsBlockBytes, QuantizationType.IQ3_XXS);

    public static float[] CpuGemmIq3Xxs(byte[] weightsIq3, float[] inputB, int m, int k, int n)
        => CpuGemmViaDequant(weightsIq3, inputB, m, k, n, Iq3XxsBlockBytes, QuantizationType.IQ3_XXS);

    // ── IQ3_S ─────────────────────────────────────────────────────────────

    public static byte[] QuantizeRowsIq3S(float[] src, int m, int k)
    {
        if ((k % Iq3GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq3GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq3GroupSize;
        int rowBytes = blocksPerRow * Iq3SBlockBytes;
        var dst = new byte[m * rowBytes];
        ReadOnlySpan<byte> grid = Dequantize.Iq3SGrid;

        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int srcOff = row * k + b * Iq3GroupSize;
                int dstOff = row * rowBytes + b * Iq3SBlockBytes;
                QuantizeBlockIq3S(src.AsSpan(srcOff, Iq3GroupSize), dst.AsSpan(dstOff, Iq3SBlockBytes), grid);
            }
        }
        return dst;
    }

    private static void QuantizeBlockIq3S(ReadOnlySpan<float> src, Span<byte> dst, ReadOnlySpan<byte> grid)
    {
        // IQ3_S grid byte values are in {0x01, 0x03, 0x05, 0x07, 0x09, 0x0b,
        // 0x0d, 0x0f}; max = 15. Per-pair scale: db = d * (1 + 2*s4),
        // s4 in [0..15] -> max factor = 31. Hence d ≈ amax / (15 * 31).
        float amax = 0;
        for (int i = 0; i < src.Length; i++) amax = MathF.Max(amax, MathF.Abs(src[i]));
        float dGuess = amax > 0 ? amax / (15.0f * 31.0f) : 0;
        float invD = dGuess > 0 ? 1.0f / dGuess : 0;

        Half hd = (Half)dGuess;
        Unsafe.WriteUnaligned(ref dst[0], hd);

        Span<int> g1s = stackalloc int[32];
        Span<int> g2s = stackalloc int[32];
        Span<byte> signBytes = stackalloc byte[32];
        Span<byte> subScales = stackalloc byte[Iq3NumSubBlocks];

        Span<float> pairScaled = stackalloc float[Iq3PairSize];
        for (int ib32 = 0; ib32 < Iq3NumSubBlocks; ib32++)
        {
            float subAmax = 0;
            for (int i = 0; i < Iq3SubBlockSize; i++)
                subAmax = MathF.Max(subAmax, MathF.Abs(src[ib32 * Iq3SubBlockSize + i]));

            int s4 = 15;
            if (dGuess > 0)
            {
                float subTarget = subAmax * invD;
                float scale = (subTarget / 15.0f - 1.0f) / 2.0f;
                int n = (int)MathF.Round(scale);
                s4 = Math.Clamp(n, 0, 15);
            }
            subScales[ib32] = (byte)s4;

            float dl = dGuess * (1.0f + 2.0f * s4);
            float invDl = dl != 0 ? 1.0f / dl : 0;

            for (int l = 0; l < Iq3PairsPerSubBlock; l++)
            {
                int pairBase = ib32 * Iq3SubBlockSize + l * Iq3PairSize;
                for (int j = 0; j < Iq3PairSize; j++)
                    pairScaled[j] = src[pairBase + j] * invDl;
                FindBestPairIq3S(pairScaled, grid, out int g1, out int g2, out byte sByte);
                g1s[ib32 * 4 + l] = g1;
                g2s[ib32 * 4 + l] = g2;
                signBytes[ib32 * 4 + l] = sByte;
            }
        }

        // qs[64] — low 8 bits of each grid index (g1, g2 interleaved per pair).
        for (int ib32 = 0; ib32 < Iq3NumSubBlocks; ib32++)
        {
            int qsBase = 2 + 8 * ib32;
            for (int l = 0; l < 4; l++)
            {
                dst[qsBase + 2 * l + 0] = (byte)(g1s[ib32 * 4 + l] & 0xff);
                dst[qsBase + 2 * l + 1] = (byte)(g2s[ib32 * 4 + l] & 0xff);
            }
        }

        // qh[8] — high 1 bit per grid index, 4 bits per sub-block.
        // Per ggml dequant: g1 hi-bit @ position (2*l), g2 hi-bit @ (2*l + 1).
        for (int ib32 = 0; ib32 < Iq3NumSubBlocks; ib32++)
        {
            byte qh = 0;
            for (int l = 0; l < 4; l++)
            {
                int hi1 = (g1s[ib32 * 4 + l] >> 8) & 1;
                int hi2 = (g2s[ib32 * 4 + l] >> 8) & 1;
                qh |= (byte)(hi1 << (2 * l));
                qh |= (byte)(hi2 << (2 * l + 1));
            }
            dst[2 + 64 + ib32] = qh;
        }

        // signs[32] @ offset 74.
        for (int p = 0; p < 32; p++) dst[2 + 64 + 8 + p] = signBytes[p];

        // scales[4] @ offset 106 — paired sub-block scales: ib32 even -> low nibble.
        for (int ib32 = 0; ib32 < Iq3NumSubBlocks; ib32 += 2)
        {
            byte lo = subScales[ib32 + 0];
            byte hi = subScales[ib32 + 1];
            dst[2 + 64 + 8 + 32 + (ib32 >> 1)] = (byte)((lo & 0xF) | ((hi & 0xF) << 4));
        }
    }

    public static void AssertFixtureRoundtripIq3S(float[] srcF32, byte[] qBytes, int m, int k)
        => AssertRoundtrip(srcF32, qBytes, m, k, QuantizationType.IQ3_S, "IQ3_S");

    public static float[] CpuDequantizeIq3S(byte[] bytes, int totalElements)
        => CpuDequantize(bytes, totalElements, QuantizationType.IQ3_S);

    public static float[] CpuGemvIq3S(byte[] weightsIq3, float[] x, int m, int k)
        => CpuGemvViaDequant(weightsIq3, x, m, k, Iq3SBlockBytes, QuantizationType.IQ3_S);

    public static float[] CpuGemmIq3S(byte[] weightsIq3, float[] inputB, int m, int k, int n)
        => CpuGemmViaDequant(weightsIq3, inputB, m, k, n, Iq3SBlockBytes, QuantizationType.IQ3_S);

    // ── Common reference paths ─────────────────────────────────────────────

    private static void AssertRoundtrip(float[] srcF32, byte[] qBytes, int m, int k,
        QuantizationType qt, string label)
    {
        var dequant = new float[m * k];
        fixed (byte* p = qBytes)
        fixed (float* d = dequant)
        {
            Dequantize.ToFloat32((nint)p, (long)m * k, qt, new Span<float>(d, m * k));
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
        // IQ3 has tighter quality than IQ2 — empirically the fixture round-trip
        // sits around 0.15-0.25 L2 drift (vs IQ2's 0.5). The 0.40 cap is a
        // generous guard against an off-by-one in the fixture (eg miswriting
        // qh) without bumping into the kernel's own ~5e-3 floor.
        Assert.True(maxRel < 0.40f,
            $"{label} fixture round-trip drift too large: maxRel={maxRel:G4} (m={m}, k={k}). " +
            "IQ3 tolerance is ~25-40% — exceeding suggests fixture is broken.");
    }

    private static float[] CpuDequantize(byte[] bytes, int totalElements, QuantizationType qt)
    {
        var dst = new float[totalElements];
        fixed (byte* p = bytes)
        fixed (float* d = dst)
        {
            Dequantize.ToFloat32((nint)p, totalElements, qt, new Span<float>(d, totalElements));
        }
        return dst;
    }

    private static float[] CpuGemvViaDequant(byte[] weightsIq3, float[] x, int m, int k,
        int blockBytes, QuantizationType qt)
    {
        if ((k % Iq3GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq3GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq3GroupSize;
        int rowBytes = blocksPerRow * blockBytes;
        var result = new float[m];
        var rowDequant = new float[k];

        fixed (byte* wPtr = weightsIq3)
        fixed (float* dPtr = rowDequant)
        {
            for (int row = 0; row < m; row++)
            {
                Dequantize.ToFloat32((nint)(wPtr + (long)row * rowBytes), k, qt,
                    new Span<float>(dPtr, k));
                float sum = 0;
                for (int i = 0; i < k; i++) sum += rowDequant[i] * x[i];
                result[row] = sum;
            }
        }
        return result;
    }

    private static float[] CpuGemmViaDequant(byte[] weightsIq3, float[] inputB, int m, int k, int n,
        int blockBytes, QuantizationType qt)
    {
        if ((k % Iq3GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq3GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq3GroupSize;
        int rowBytes = blocksPerRow * blockBytes;
        var result = new float[n * m];
        var rowDequant = new float[k];

        fixed (byte* wPtr = weightsIq3)
        fixed (float* dPtr = rowDequant)
        {
            for (int row = 0; row < m; row++)
            {
                Dequantize.ToFloat32((nint)(wPtr + (long)row * rowBytes), k, qt,
                    new Span<float>(dPtr, k));
                for (int t = 0; t < n; t++)
                {
                    int bRowBase = t * k;
                    float sum = 0;
                    for (int i = 0; i < k; i++) sum += inputB[bRowBase + i] * rowDequant[i];
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
