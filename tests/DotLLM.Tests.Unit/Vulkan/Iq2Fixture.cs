using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Kernels;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Test-only IQ2_XXS / IQ2_XS / IQ2_S fixture helpers shared by the dequant /
/// GEMV / GEMM kernel tests.
/// </summary>
/// <remarks>
/// <para>
/// The production CPU side has full IQ2 dequantise kernels but no F32-to-IQ2
/// quantiser — production loaders only ingest pre-quantised GGUF data. For
/// Vulkan parity we round-trip arbitrary FP32 fixtures through IQ2 bytes.
/// Layouts match <c>DotLLM.Cpu.Kernels.Dequantize.DequantizeIQ2_*</c> and
/// llama.cpp's <c>block_iq2_*</c> byte-for-byte.
/// </para>
/// <para>
/// Per-pair (8 elements) decode is a 256/512/1024-entry grid lookup combined
/// with a 7- or 8-bit sign mask, so the quantiser does an exhaustive
/// nearest-pair search. Each pair tests grid_count × 128 sign patterns and
/// keeps the (gridIdx, signsIdx) pair with minimum L2 error against the
/// scaled targets. Slow but trivially correct — the kernel parity assertions
/// then exercise the *kernel*, not the fixture.
/// </para>
/// </remarks>
internal static unsafe class Iq2Fixture
{
    public const int Iq2GroupSize = 256;
    public const int Iq2SubBlockSize = 32;
    public const int Iq2NumSubBlocks = 8;
    public const int Iq2PairSize = 8;
    public const int Iq2PairsPerSubBlock = 4;

    public const int Iq2XxsBlockBytes = 66;
    public const int Iq2XsBlockBytes = 74;
    public const int Iq2SBlockBytes = 82;

    public const int Iq2XxsGridSize = 256;
    public const int Iq2XsGridSize = 512;
    public const int Iq2SGridSize = 1024;

    public static float[] RandomFloats(Random rng, int count, float range)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = (float)((rng.NextDouble() * 2.0 - 1.0) * range);
        return arr;
    }

    /// <summary>Per-pair best-fit search: returns (gridIdx, signsIdx) minimising
    /// L2 against <paramref name="targetScaled"/>. <paramref name="grid"/> is the
    /// IQ2_* codebook (gridSize × 8 bytes); ksigns is the 128-entry sign-pattern
    /// table.</summary>
    private static void FindBestPair(
        ReadOnlySpan<float> targetScaled,
        ReadOnlySpan<byte> grid, int gridSize,
        out int gridIdx, out int signsIdx)
    {
        ReadOnlySpan<byte> ksigns = Dequantize.KsignsIq2Xs;
        gridIdx = 0;
        signsIdx = 0;
        float bestErr = float.MaxValue;
        for (int g = 0; g < gridSize; g++)
        {
            int gOff = g * 8;
            for (int s = 0; s < 128; s++)
            {
                byte mask = ksigns[s];
                float err = 0;
                for (int j = 0; j < 8; j++)
                {
                    float gv = grid[gOff + j];
                    float sign = (mask & (1 << j)) != 0 ? -1f : 1f;
                    float diff = targetScaled[j] - sign * gv;
                    err += diff * diff;
                }
                if (err < bestErr)
                {
                    bestErr = err;
                    gridIdx = g;
                    signsIdx = s;
                }
            }
        }
    }

    /// <summary>Per-pair best-fit search for IQ2_S: 1024-entry grid + 8-bit
    /// signs (no ksigns indirection — IQ2_S stores the sign mask directly).
    /// Searches all (grid × ksigns-pattern) combinations and emits the
    /// resolved 8-bit sign byte (= ksigns[signsIdx]).</summary>
    private static void FindBestPairIq2S(
        ReadOnlySpan<float> targetScaled,
        ReadOnlySpan<byte> grid,
        out int gridIdx, out byte signsByte)
    {
        ReadOnlySpan<byte> ksigns = Dequantize.KsignsIq2Xs;
        gridIdx = 0;
        signsByte = 0;
        float bestErr = float.MaxValue;
        for (int g = 0; g < Iq2SGridSize; g++)
        {
            int gOff = g * 8;
            for (int s = 0; s < 128; s++)
            {
                byte mask = ksigns[s];
                float err = 0;
                for (int j = 0; j < 8; j++)
                {
                    float gv = grid[gOff + j];
                    float sign = (mask & (1 << j)) != 0 ? -1f : 1f;
                    float diff = targetScaled[j] - sign * gv;
                    err += diff * diff;
                }
                if (err < bestErr)
                {
                    bestErr = err;
                    gridIdx = g;
                    signsByte = mask;
                }
            }
        }
    }

    // ── IQ2_XXS ────────────────────────────────────────────────────────────

    /// <summary>
    /// Quantise an [m, k] row-major FP32 matrix to IQ2_XXS bytes. k must be a
    /// multiple of 256.
    /// </summary>
    public static byte[] QuantizeRowsIq2Xxs(float[] src, int m, int k)
    {
        if ((k % Iq2GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq2GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq2GroupSize;
        int rowBytes = blocksPerRow * Iq2XxsBlockBytes;
        var dst = new byte[m * rowBytes];
        ReadOnlySpan<byte> grid = Dequantize.Iq2XxsGrid;

        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int srcOff = row * k + b * Iq2GroupSize;
                int dstOff = row * rowBytes + b * Iq2XxsBlockBytes;
                QuantizeBlockIq2Xxs(src.AsSpan(srcOff, Iq2GroupSize), dst.AsSpan(dstOff, Iq2XxsBlockBytes), grid);
            }
        }
        return dst;
    }

    private static void QuantizeBlockIq2Xxs(ReadOnlySpan<float> src, Span<byte> dst, ReadOnlySpan<byte> grid)
    {
        // Block scale chosen to put |amax| at the codebook ceiling. The grid
        // entries are bytes in {0x08, 0x19, 0x2b} — max=43. Per-pair effective
        // scale = d * (0.5 + s4) * 0.25, s4 ∈ [0,15] so max factor is 7.75/4
        // = ~1.9375. We pick s4=15 for the loudest pair, giving d ≈ amax / (43 * 1.9375).
        float amax = 0;
        for (int i = 0; i < src.Length; i++) amax = MathF.Max(amax, MathF.Abs(src[i]));
        float dGuess = amax > 0 ? amax / (43.0f * 1.9375f) : 0;

        // Per-sub-block: pick s4 then per-pair search. Brute-force s4 ∈ [0,15]
        // is too slow for every test invocation — instead pick the minimal s4
        // that covers the sub-block's amax with the codebook ceiling, then
        // search per-pair. A tighter quantiser would iterate; the parity tests
        // tolerate ~20% L2 round-trip drift from the fixture.
        Span<byte> nibbles = stackalloc byte[Iq2NumSubBlocks];
        Span<int> gridIdxs = stackalloc int[32];          // 8 sub × 4 pairs
        Span<int> signIdxs = stackalloc int[32];

        Half hd = (Half)dGuess;
        Unsafe.WriteUnaligned(ref dst[0], hd);
        float invD = dGuess > 0 ? 1.0f / dGuess : 0;

        for (int ib32 = 0; ib32 < Iq2NumSubBlocks; ib32++)
        {
            float subAmax = 0;
            for (int i = 0; i < Iq2SubBlockSize; i++)
                subAmax = MathF.Max(subAmax, MathF.Abs(src[ib32 * Iq2SubBlockSize + i]));

            int s4 = 15;
            if (dGuess > 0)
            {
                float subTarget = subAmax * invD;
                float scale = subTarget / 43.0f;
                int n = (int)MathF.Round(scale * 4.0f - 0.5f);
                s4 = Math.Clamp(n, 0, 15);
            }
            nibbles[ib32] = (byte)s4;

            float dl = dGuess * (0.5f + s4) * 0.25f;
            float invDl = dl != 0 ? 1.0f / dl : 0;

            Span<float> pairScaled = stackalloc float[Iq2PairSize];
            for (int l = 0; l < Iq2PairsPerSubBlock; l++)
            {
                int pairBase = ib32 * Iq2SubBlockSize + l * Iq2PairSize;
                for (int j = 0; j < Iq2PairSize; j++)
                    pairScaled[j] = src[pairBase + j] * invDl;
                FindBestPair(pairScaled, grid, Iq2XxsGridSize, out int gIdx, out int sIdx);
                gridIdxs[ib32 * 4 + l] = gIdx;
                signIdxs[ib32 * 4 + l] = sIdx;
            }
        }

        // Pack qs[] : per sub-block 8 bytes — a0 (4 grid bytes) and a1 (4*7-bit
        // signs in low 28 + 4-bit s4 in top nibble).
        for (int ib32 = 0; ib32 < Iq2NumSubBlocks; ib32++)
        {
            uint a0 = 0;
            uint a1 = 0;
            for (int l = 0; l < 4; l++)
            {
                a0 |= (uint)(gridIdxs[ib32 * 4 + l] & 0xff) << (8 * l);
                a1 |= (uint)(signIdxs[ib32 * 4 + l] & 0x7f) << (7 * l);
            }
            a1 |= (uint)nibbles[ib32] << 28;
            int qsBase = 2 + 8 * ib32;
            dst[qsBase + 0] = (byte)(a0 & 0xff);
            dst[qsBase + 1] = (byte)((a0 >> 8) & 0xff);
            dst[qsBase + 2] = (byte)((a0 >> 16) & 0xff);
            dst[qsBase + 3] = (byte)((a0 >> 24) & 0xff);
            dst[qsBase + 4] = (byte)(a1 & 0xff);
            dst[qsBase + 5] = (byte)((a1 >> 8) & 0xff);
            dst[qsBase + 6] = (byte)((a1 >> 16) & 0xff);
            dst[qsBase + 7] = (byte)((a1 >> 24) & 0xff);
        }
    }

    public static void AssertFixtureRoundtripIq2Xxs(float[] srcF32, byte[] q2Bytes, int m, int k)
        => AssertRoundtrip(srcF32, q2Bytes, m, k, QuantizationType.IQ2_XXS, "IQ2_XXS");

    public static float[] CpuDequantizeIq2Xxs(byte[] bytes, int totalElements)
        => CpuDequantize(bytes, totalElements, QuantizationType.IQ2_XXS);

    public static float[] CpuGemvIq2Xxs(byte[] weightsIq2, float[] x, int m, int k)
        => CpuGemvViaDequant(weightsIq2, x, m, k, Iq2XxsBlockBytes, QuantizationType.IQ2_XXS);

    public static float[] CpuGemmIq2Xxs(byte[] weightsIq2, float[] inputB, int m, int k, int n)
        => CpuGemmViaDequant(weightsIq2, inputB, m, k, n, Iq2XxsBlockBytes, QuantizationType.IQ2_XXS);

    // ── IQ2_XS ─────────────────────────────────────────────────────────────

    public static byte[] QuantizeRowsIq2Xs(float[] src, int m, int k)
    {
        if ((k % Iq2GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq2GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq2GroupSize;
        int rowBytes = blocksPerRow * Iq2XsBlockBytes;
        var dst = new byte[m * rowBytes];
        ReadOnlySpan<byte> grid = Dequantize.Iq2XsGrid;

        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int srcOff = row * k + b * Iq2GroupSize;
                int dstOff = row * rowBytes + b * Iq2XsBlockBytes;
                QuantizeBlockIq2Xs(src.AsSpan(srcOff, Iq2GroupSize), dst.AsSpan(dstOff, Iq2XsBlockBytes), grid);
            }
        }
        return dst;
    }

    private static void QuantizeBlockIq2Xs(ReadOnlySpan<float> src, Span<byte> dst, ReadOnlySpan<byte> grid)
    {
        float amax = 0;
        for (int i = 0; i < src.Length; i++) amax = MathF.Max(amax, MathF.Abs(src[i]));
        // IQ2_XS: each pair has its own 4-bit sub-scale (two pairs share one
        // scale in IQ2_XS — l<2 -> low nibble, l>=2 -> high nibble of scales[ib32]).
        // We use the same d-guess as IQ2_XXS.
        float dGuess = amax > 0 ? amax / (43.0f * 1.9375f) : 0;
        float invD = dGuess > 0 ? 1.0f / dGuess : 0;

        Half hd = (Half)dGuess;
        Unsafe.WriteUnaligned(ref dst[0], hd);

        Span<int> gridIdxs = stackalloc int[32];
        Span<int> signIdxs = stackalloc int[32];
        Span<byte> scaleNibbles = stackalloc byte[16]; // 8 sub × 2 nibbles

        for (int ib32 = 0; ib32 < Iq2NumSubBlocks; ib32++)
        {
            // Sub-scale 0 covers pairs 0,1; sub-scale 1 covers pairs 2,3.
            for (int half = 0; half < 2; half++)
            {
                float halfAmax = 0;
                for (int l = 0; l < 2; l++)
                {
                    int pairBase = ib32 * Iq2SubBlockSize + (half * 2 + l) * Iq2PairSize;
                    for (int j = 0; j < Iq2PairSize; j++)
                        halfAmax = MathF.Max(halfAmax, MathF.Abs(src[pairBase + j]));
                }

                int s4 = 15;
                if (dGuess > 0)
                {
                    float subTarget = halfAmax * invD;
                    float scale = subTarget / 43.0f;
                    int n = (int)MathF.Round(scale * 4.0f - 0.5f);
                    s4 = Math.Clamp(n, 0, 15);
                }
                scaleNibbles[ib32 * 2 + half] = (byte)s4;
                float dl = dGuess * (0.5f + s4) * 0.25f;
                float invDl = dl != 0 ? 1.0f / dl : 0;

                Span<float> pairScaled = stackalloc float[Iq2PairSize];
                for (int l = 0; l < 2; l++)
                {
                    int pairIdx = half * 2 + l;
                    int pairBase = ib32 * Iq2SubBlockSize + pairIdx * Iq2PairSize;
                    for (int j = 0; j < Iq2PairSize; j++)
                        pairScaled[j] = src[pairBase + j] * invDl;
                    FindBestPair(pairScaled, grid, Iq2XsGridSize, out int gIdx, out int sIdx);
                    gridIdxs[ib32 * 4 + pairIdx] = gIdx;
                    signIdxs[ib32 * 4 + pairIdx] = sIdx;
                }
            }
        }

        // Write qs[32] uint16 — bits 0..8 = grid (9 bits), bits 9..15 = signs (7 bits).
        for (int p = 0; p < 32; p++)
        {
            ushort q = (ushort)((gridIdxs[p] & 0x1FF) | ((signIdxs[p] & 0x7F) << 9));
            int off = 2 + p * 2;
            dst[off + 0] = (byte)(q & 0xff);
            dst[off + 1] = (byte)((q >> 8) & 0xff);
        }

        // Write scales[8] — low nibble = first 2 pairs scale, high = last 2.
        for (int ib32 = 0; ib32 < Iq2NumSubBlocks; ib32++)
        {
            byte lo = scaleNibbles[ib32 * 2 + 0];
            byte hi = scaleNibbles[ib32 * 2 + 1];
            dst[66 + ib32] = (byte)((lo & 0xF) | ((hi & 0xF) << 4));
        }
    }

    public static void AssertFixtureRoundtripIq2Xs(float[] srcF32, byte[] q2Bytes, int m, int k)
        => AssertRoundtrip(srcF32, q2Bytes, m, k, QuantizationType.IQ2_XS, "IQ2_XS");

    public static float[] CpuDequantizeIq2Xs(byte[] bytes, int totalElements)
        => CpuDequantize(bytes, totalElements, QuantizationType.IQ2_XS);

    public static float[] CpuGemvIq2Xs(byte[] weightsIq2, float[] x, int m, int k)
        => CpuGemvViaDequant(weightsIq2, x, m, k, Iq2XsBlockBytes, QuantizationType.IQ2_XS);

    public static float[] CpuGemmIq2Xs(byte[] weightsIq2, float[] inputB, int m, int k, int n)
        => CpuGemmViaDequant(weightsIq2, inputB, m, k, n, Iq2XsBlockBytes, QuantizationType.IQ2_XS);

    // ── IQ2_S ──────────────────────────────────────────────────────────────

    public static byte[] QuantizeRowsIq2S(float[] src, int m, int k)
    {
        if ((k % Iq2GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq2GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq2GroupSize;
        int rowBytes = blocksPerRow * Iq2SBlockBytes;
        var dst = new byte[m * rowBytes];
        ReadOnlySpan<byte> grid = Dequantize.Iq2SGrid;

        for (int row = 0; row < m; row++)
        {
            for (int b = 0; b < blocksPerRow; b++)
            {
                int srcOff = row * k + b * Iq2GroupSize;
                int dstOff = row * rowBytes + b * Iq2SBlockBytes;
                QuantizeBlockIq2S(src.AsSpan(srcOff, Iq2GroupSize), dst.AsSpan(dstOff, Iq2SBlockBytes), grid);
            }
        }
        return dst;
    }

    private static void QuantizeBlockIq2S(ReadOnlySpan<float> src, Span<byte> dst, ReadOnlySpan<byte> grid)
    {
        float amax = 0;
        for (int i = 0; i < src.Length; i++) amax = MathF.Max(amax, MathF.Abs(src[i]));
        float dGuess = amax > 0 ? amax / (43.0f * 1.9375f) : 0;
        float invD = dGuess > 0 ? 1.0f / dGuess : 0;

        Half hd = (Half)dGuess;
        Unsafe.WriteUnaligned(ref dst[0], hd);

        Span<int> gridIdxs = stackalloc int[32];
        Span<byte> signBytes = stackalloc byte[32];
        Span<byte> scaleNibbles = stackalloc byte[16];

        for (int ib32 = 0; ib32 < Iq2NumSubBlocks; ib32++)
        {
            for (int half = 0; half < 2; half++)
            {
                float halfAmax = 0;
                for (int l = 0; l < 2; l++)
                {
                    int pairBase = ib32 * Iq2SubBlockSize + (half * 2 + l) * Iq2PairSize;
                    for (int j = 0; j < Iq2PairSize; j++)
                        halfAmax = MathF.Max(halfAmax, MathF.Abs(src[pairBase + j]));
                }
                int s4 = 15;
                if (dGuess > 0)
                {
                    float subTarget = halfAmax * invD;
                    float scale = subTarget / 43.0f;
                    int n = (int)MathF.Round(scale * 4.0f - 0.5f);
                    s4 = Math.Clamp(n, 0, 15);
                }
                scaleNibbles[ib32 * 2 + half] = (byte)s4;
                float dl = dGuess * (0.5f + s4) * 0.25f;
                float invDl = dl != 0 ? 1.0f / dl : 0;

                Span<float> pairScaled = stackalloc float[Iq2PairSize];
                for (int l = 0; l < 2; l++)
                {
                    int pairIdx = half * 2 + l;
                    int pairBase = ib32 * Iq2SubBlockSize + pairIdx * Iq2PairSize;
                    for (int j = 0; j < Iq2PairSize; j++)
                        pairScaled[j] = src[pairBase + j] * invDl;
                    FindBestPairIq2S(pairScaled, grid, out int gIdx, out byte sByte);
                    gridIdxs[ib32 * 4 + pairIdx] = gIdx;
                    signBytes[ib32 * 4 + pairIdx] = sByte;
                }
            }
        }

        // Write qs[32] (low 8 bits of grid index per pair).
        for (int p = 0; p < 32; p++) dst[2 + p] = (byte)(gridIdxs[p] & 0xff);
        // Write qs_signs[32].
        for (int p = 0; p < 32; p++) dst[2 + 32 + p] = signBytes[p];
        // Write qh[8] — high 2 bits of grid per pair, 4 pairs per byte.
        for (int ib32 = 0; ib32 < Iq2NumSubBlocks; ib32++)
        {
            byte qh = 0;
            for (int l = 0; l < 4; l++)
            {
                byte hi = (byte)((gridIdxs[ib32 * 4 + l] >> 8) & 0x3);
                qh |= (byte)(hi << (2 * l));
            }
            dst[2 + 64 + ib32] = qh;
        }
        // Write scales[8].
        for (int ib32 = 0; ib32 < Iq2NumSubBlocks; ib32++)
        {
            byte lo = scaleNibbles[ib32 * 2 + 0];
            byte hi = scaleNibbles[ib32 * 2 + 1];
            dst[2 + 64 + 8 + ib32] = (byte)((lo & 0xF) | ((hi & 0xF) << 4));
        }
    }

    public static void AssertFixtureRoundtripIq2S(float[] srcF32, byte[] q2Bytes, int m, int k)
        => AssertRoundtrip(srcF32, q2Bytes, m, k, QuantizationType.IQ2_S, "IQ2_S");

    public static float[] CpuDequantizeIq2S(byte[] bytes, int totalElements)
        => CpuDequantize(bytes, totalElements, QuantizationType.IQ2_S);

    public static float[] CpuGemvIq2S(byte[] weightsIq2, float[] x, int m, int k)
        => CpuGemvViaDequant(weightsIq2, x, m, k, Iq2SBlockBytes, QuantizationType.IQ2_S);

    public static float[] CpuGemmIq2S(byte[] weightsIq2, float[] inputB, int m, int k, int n)
        => CpuGemmViaDequant(weightsIq2, inputB, m, k, n, Iq2SBlockBytes, QuantizationType.IQ2_S);

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
        Assert.True(maxRel < 0.55f,
            $"{label} fixture round-trip drift too large: maxRel={maxRel:G4} (m={m}, k={k}). " +
            "IQ2 has a wide tolerance (~50%) due to 2-bit quantisation; if this trips the fixture is broken.");
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

    /// <summary>Reference GEMV via dequant-then-multiply. Reads the same byte
    /// buffer the kernel sees so any divergence is in the kernel arithmetic,
    /// not in our fixture path.</summary>
    private static float[] CpuGemvViaDequant(byte[] weightsIq2, float[] x, int m, int k,
        int blockBytes, QuantizationType qt)
    {
        if ((k % Iq2GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq2GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq2GroupSize;
        int rowBytes = blocksPerRow * blockBytes;
        var result = new float[m];
        var rowDequant = new float[k];

        fixed (byte* wPtr = weightsIq2)
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

    private static float[] CpuGemmViaDequant(byte[] weightsIq2, float[] inputB, int m, int k, int n,
        int blockBytes, QuantizationType qt)
    {
        if ((k % Iq2GroupSize) != 0)
            throw new ArgumentException($"k must be a multiple of {Iq2GroupSize}, got {k}", nameof(k));

        int blocksPerRow = k / Iq2GroupSize;
        int rowBytes = blocksPerRow * blockBytes;
        var result = new float[n * m];
        var rowDequant = new float[k];

        fixed (byte* wPtr = weightsIq2)
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
