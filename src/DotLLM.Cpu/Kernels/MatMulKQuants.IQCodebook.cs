using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// One decoded 32-value sub-block of a codebook IQ format: 32 unsigned grid bytes (four
/// little-endian ulongs), 32 sign bytes (<c>0x01</c> = +, <c>0xFF</c> = −, <c>0x00</c> = zero), and
/// an integer scale for each 16-value half.
/// </summary>
internal struct IqSubBlock
{
    public ulong G0, G1, G2, G3;
    public ulong S0, S1, S2, S3;
    public int ScLo, ScHi;
    /// <summary>IQ1_S only: ±1 sign of the per-sub-block <c>delta</c> (0 for formats without one).</summary>
    public int Delta;
}

/// <summary>Per-format sub-block decoder for the codebook IQ family (issue #605).</summary>
internal unsafe interface IIqFormat
{
    static abstract int BlockBytes { get; }
    static abstract bool HasDelta { get; }
    /// <summary>Multiplier applied to <c>d · d8</c> on top of the integer sub-block scales.</summary>
    static abstract float Factor { get; }
    static abstract IqSubBlock Decode(byte* qk, int ib);
}

/// <summary>
/// Packed <b>IQ3_XXS / IQ3_S / IQ2_XXS / IQ2_XS / IQ2_S / IQ1_S</b> × Q8_K dots (issue #605).
/// </summary>
/// <remarks>
/// <para>All six formats are "grid lookup × sign × small integer scale" over 32-value sub-blocks,
/// so one generic kernel consumes <see cref="IqSubBlock"/>s: <c>PMADDUBSW(grid, SIGN(q8, sign))</c>
/// (grid bytes are unsigned and ≤ 62, so pair sums ≤ 2·62·127 = 15748, no saturation) then PMADDWD
/// by the half's integer scale. Each format's dequantiser is the test oracle; scalar/SSSE3/AVX2 tiers
/// and a 4-column AVX2 kernel (decode once, dot four activation columns) share the decoders.</para>
/// <para>IQ1_S additionally has <c>dl·(grid + delta)</c>: the grid part goes through the same path
/// (its −1/0/+1 bytes serve directly as the sign operand, <c>grid &amp; 1</c> as magnitude) and the
/// delta part is <c>0.125·scale·±1·Σq8</c> taken from Q8_K's <c>bsums</c>.</para>
/// </remarks>
public static unsafe partial class MatMul
{
    // ──────────────────── static tables (native copies: stable pointers, never freed) ────────────────────

    private static readonly nint IqSign7 = BuildSignTable(useKsigns: true);
    private static readonly nint IqSign8 = BuildSignTable(useKsigns: false);
    private static readonly nint IqGridIq2Xxs = CopyNative(Dequantize.Iq2XxsGrid);
    private static readonly nint IqGridIq2Xs = CopyNative(Dequantize.Iq2XsGrid);
    private static readonly nint IqGridIq2S = CopyNative(Dequantize.Iq2SGrid);
    private static readonly nint IqGridIq3Xxs = CopyNative(Dequantize.Iq3XxsGrid);
    private static readonly nint IqGridIq3S = CopyNative(Dequantize.Iq3SGrid);
    private static readonly nint IqGridIq1S = CopyNative(MemoryMarshal.AsBytes(Dequantize.Iq1SGrid));

    private static nint CopyNative(ReadOnlySpan<byte> src)
    {
        byte* p = (byte*)NativeMemory.AlignedAlloc((nuint)src.Length, 64);
        src.CopyTo(new Span<byte>(p, src.Length));
        return (nint)p;
    }

    /// <summary>Sign-mask byte → 8 sign bytes (0xFF where the bit is set, else 0x01).</summary>
    private static nint BuildSignTable(bool useKsigns)
    {
        int count = useKsigns ? 128 : 256;
        ulong* t = (ulong*)NativeMemory.AlignedAlloc((nuint)(count * 8), 64);
        ReadOnlySpan<byte> ks = Dequantize.KsignsIq2Xs;
        for (int i = 0; i < count; i++)
        {
            int mask = useKsigns ? ks[i] : i;
            ulong v = 0;
            for (int j = 0; j < 8; j++)
                v |= (ulong)(((mask >> j) & 1) != 0 ? 0xFF : 0x01) << (8 * j);
            t[i] = v;
        }
        return (nint)t;
    }

    // ──────────────────── Format decoders ────────────────────

    internal struct Iq2XxsFmt : IIqFormat
    {
        public static int BlockBytes => QuantFormat.IQ2_XXSBlockBytes;
        public static bool HasDelta => false;
        public static float Factor => 0.125f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IqSubBlock Decode(byte* qk, int ib)
        {
            byte* p = qk + 2 + 8 * ib;
            uint a0 = Unsafe.ReadUnaligned<uint>(p), a1 = Unsafe.ReadUnaligned<uint>(p + 4);
            ulong* grid = (ulong*)IqGridIq2Xxs, sg = (ulong*)IqSign7;
            IqSubBlock b = default;
            b.G0 = grid[a0 & 0xFF]; b.G1 = grid[(a0 >> 8) & 0xFF];
            b.G2 = grid[(a0 >> 16) & 0xFF]; b.G3 = grid[a0 >> 24];
            b.S0 = sg[a1 & 0x7F]; b.S1 = sg[(a1 >> 7) & 0x7F];
            b.S2 = sg[(a1 >> 14) & 0x7F]; b.S3 = sg[(a1 >> 21) & 0x7F];
            b.ScLo = b.ScHi = (int)(2 * (a1 >> 28) + 1);
            return b;
        }
    }

    internal struct Iq2XsFmt : IIqFormat
    {
        public static int BlockBytes => QuantFormat.IQ2_XSBlockBytes;
        public static bool HasDelta => false;
        public static float Factor => 0.125f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IqSubBlock Decode(byte* qk, int ib)
        {
            ushort* q = (ushort*)(qk + 2 + ib * 8);
            ulong* grid = (ulong*)IqGridIq2Xs, sg = (ulong*)IqSign7;
            IqSubBlock b = default;
            b.G0 = grid[q[0] & 0x1FF]; b.S0 = sg[q[0] >> 9];
            b.G1 = grid[q[1] & 0x1FF]; b.S1 = sg[q[1] >> 9];
            b.G2 = grid[q[2] & 0x1FF]; b.S2 = sg[q[2] >> 9];
            b.G3 = grid[q[3] & 0x1FF]; b.S3 = sg[q[3] >> 9];
            int sc = qk[2 + 64 + ib];
            b.ScLo = 2 * (sc & 0xF) + 1; b.ScHi = 2 * (sc >> 4) + 1;
            return b;
        }
    }

    internal struct Iq2SFmt : IIqFormat
    {
        public static int BlockBytes => QuantFormat.IQ2_SBlockBytes;
        public static bool HasDelta => false;
        public static float Factor => 0.125f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IqSubBlock Decode(byte* qk, int ib)
        {
            byte* lo = qk + 2 + ib * 4, sgn = qk + 2 + 32 + ib * 4;
            int qh = qk[2 + 64 + ib];
            ulong* grid = (ulong*)IqGridIq2S, sg = (ulong*)IqSign8;
            IqSubBlock b = default;
            b.G0 = grid[lo[0] | ((qh & 3) << 8)]; b.S0 = sg[sgn[0]];
            b.G1 = grid[lo[1] | (((qh >> 2) & 3) << 8)]; b.S1 = sg[sgn[1]];
            b.G2 = grid[lo[2] | (((qh >> 4) & 3) << 8)]; b.S2 = sg[sgn[2]];
            b.G3 = grid[lo[3] | (((qh >> 6) & 3) << 8)]; b.S3 = sg[sgn[3]];
            int sc = qk[2 + 64 + 8 + ib];
            b.ScLo = 2 * (sc & 0xF) + 1; b.ScHi = 2 * (sc >> 4) + 1;
            return b;
        }
    }

    internal struct Iq3XxsFmt : IIqFormat
    {
        public static int BlockBytes => QuantFormat.IQ3_XXSBlockBytes;
        public static bool HasDelta => false;
        public static float Factor => 0.25f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IqSubBlock Decode(byte* qk, int ib)
        {
            byte* qs = qk + 2 + ib * 8;
            uint aux = Unsafe.ReadUnaligned<uint>(qk + 2 + 64 + 4 * ib);
            uint* grid = (uint*)IqGridIq3Xxs;
            ulong* sg = (ulong*)IqSign7;
            IqSubBlock b = default;
            b.G0 = grid[qs[0]] | ((ulong)grid[qs[1]] << 32); b.S0 = sg[aux & 0x7F];
            b.G1 = grid[qs[2]] | ((ulong)grid[qs[3]] << 32); b.S1 = sg[(aux >> 7) & 0x7F];
            b.G2 = grid[qs[4]] | ((ulong)grid[qs[5]] << 32); b.S2 = sg[(aux >> 14) & 0x7F];
            b.G3 = grid[qs[6]] | ((ulong)grid[qs[7]] << 32); b.S3 = sg[(aux >> 21) & 0x7F];
            b.ScLo = b.ScHi = (int)(2 * (aux >> 28) + 1);
            return b;
        }
    }

    internal struct Iq3SFmt : IIqFormat
    {
        public static int BlockBytes => QuantFormat.IQ3_SBlockBytes;
        public static bool HasDelta => false;
        public static float Factor => 1.0f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IqSubBlock Decode(byte* qk, int ib)
        {
            byte* qs = qk + 2 + ib * 8;
            int qh = qk[2 + 64 + ib];
            byte* sgn = qk + 2 + 64 + 8 + ib * 4;
            uint* grid = (uint*)IqGridIq3S;
            ulong* sg = (ulong*)IqSign8;
            IqSubBlock b = default;
            b.G0 = grid[qs[0] | ((qh << 8) & 0x100)] | ((ulong)grid[qs[1] | ((qh << 7) & 0x100)] << 32); b.S0 = sg[sgn[0]];
            b.G1 = grid[qs[2] | ((qh << 6) & 0x100)] | ((ulong)grid[qs[3] | ((qh << 5) & 0x100)] << 32); b.S1 = sg[sgn[1]];
            b.G2 = grid[qs[4] | ((qh << 4) & 0x100)] | ((ulong)grid[qs[5] | ((qh << 3) & 0x100)] << 32); b.S2 = sg[sgn[2]];
            b.G3 = grid[qs[6] | ((qh << 2) & 0x100)] | ((ulong)grid[qs[7] | ((qh << 1) & 0x100)] << 32); b.S3 = sg[sgn[3]];
            int sc = qk[2 + 64 + 8 + 32 + (ib >> 1)];
            b.ScLo = b.ScHi = 1 + 2 * ((ib & 1) == 0 ? (sc & 0xF) : (sc >> 4));
            return b;
        }
    }

    internal struct Iq1SFmt : IIqFormat
    {
        public static int BlockBytes => QuantFormat.IQ1_SBlockBytes;
        public static bool HasDelta => true;
        public static float Factor => 1.0f;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static IqSubBlock Decode(byte* qk, int ib)
        {
            byte* qs = qk + 2 + ib * 4;
            int qh = Unsafe.ReadUnaligned<ushort>(qk + 2 + 32 + 2 * ib);
            ulong* grid = (ulong*)IqGridIq1S;
            const ulong ones = 0x0101010101010101UL;
            ulong e0 = grid[qs[0] | ((qh & 7) << 8)], e1 = grid[qs[1] | (((qh >> 3) & 7) << 8)];
            ulong e2 = grid[qs[2] | (((qh >> 6) & 7) << 8)], e3 = grid[qs[3] | (((qh >> 9) & 7) << 8)];
            IqSubBlock b = default;
            // Grid bytes are -1/0/+1: magnitude = byte & 1, and the byte itself is the SIGN operand.
            b.G0 = e0 & ones; b.S0 = e0; b.G1 = e1 & ones; b.S1 = e1;
            b.G2 = e2 & ones; b.S2 = e2; b.G3 = e3 & ones; b.S3 = e3;
            b.ScLo = b.ScHi = 2 * ((qh >> 12) & 7) + 1;
            b.Delta = (qh & 0x8000) != 0 ? -1 : 1;
            return b;
        }
    }

    // ──────────────────── Generic dots ────────────────────

    /// <summary>Scalar tier (no SSSE3): sums <c>grid · sign · q8</c> per half.</summary>
    [SkipLocalsInit]
    internal static float VecDotIq_Q8_KScalar<T>(byte* qk, byte* q8k, int superBlockCount) where T : struct, IIqFormat
    {
        float sumf = 0;
        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float f = (float)Unsafe.ReadUnaligned<Half>(qk) * Unsafe.ReadUnaligned<float>(q8k) * T.Factor;
            sbyte* q8 = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);
            int sumi = 0, deltaAcc = 0;
            for (int ib = 0; ib < 8; ib++)
            {
                IqSubBlock s = T.Decode(qk, ib);
                ulong* g = &s.G0, sg = &s.S0;
                int lo = 0, hi = 0;
                for (int j = 0; j < 32; j++)
                {
                    int v = (int)((g[j >> 3] >> (8 * (j & 7))) & 0xFF) * (sbyte)(sg[j >> 3] >> (8 * (j & 7))) * q8[ib * 32 + j];
                    if (j < 16) lo += v; else hi += v;
                }
                sumi += s.ScLo * lo + s.ScHi * hi;
                if (T.HasDelta) deltaAcc += s.ScLo * s.Delta * (bsums[2 * ib] + bsums[2 * ib + 1]);
            }
            sumf += f * sumi;
            if (T.HasDelta) sumf += f * 0.125f * deltaAcc;
            qk += T.BlockBytes;
            q8k += Q8_K_BlockBytes;
        }
        return sumf;
    }

    /// <summary>128-bit (SSSE3) tier.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotIq_Q8_KSse<T>(byte* qk, byte* q8k, int superBlockCount) where T : struct, IIqFormat
    {
        Vector128<float> acc = Vector128<float>.Zero;
        float deltaF = 0;
        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float f = (float)Unsafe.ReadUnaligned<Half>(qk) * Unsafe.ReadUnaligned<float>(q8k) * T.Factor;
            sbyte* q8 = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);
            Vector128<int> sumi = Vector128<int>.Zero;
            int deltaAcc = 0;
            for (int ib = 0; ib < 8; ib++)
            {
                IqSubBlock s = T.Decode(qk, ib);
                Vector128<sbyte> q8Lo = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8 + ib * 32);
                Vector128<sbyte> q8Hi = Unsafe.ReadUnaligned<Vector128<sbyte>>(q8 + ib * 32 + 16);
                Vector128<short> pLo = Ssse3.MultiplyAddAdjacent(Vector128.Create(s.G0, s.G1).AsByte(),
                    Ssse3.Sign(q8Lo, Vector128.Create(s.S0, s.S1).AsSByte()));
                Vector128<short> pHi = Ssse3.MultiplyAddAdjacent(Vector128.Create(s.G2, s.G3).AsByte(),
                    Ssse3.Sign(q8Hi, Vector128.Create(s.S2, s.S3).AsSByte()));
                sumi = Sse2.Add(sumi, Sse2.Add(
                    Sse2.MultiplyAddAdjacent(pLo, Vector128.Create((short)s.ScLo)),
                    Sse2.MultiplyAddAdjacent(pHi, Vector128.Create((short)s.ScHi))));
                if (T.HasDelta) deltaAcc += s.ScLo * s.Delta * (bsums[2 * ib] + bsums[2 * ib + 1]);
            }
            acc = Sse.Add(acc, Sse.Multiply(Vector128.Create(f), Sse2.ConvertToVector128Single(sumi)));
            if (T.HasDelta) deltaF += f * 0.125f * deltaAcc;
            qk += T.BlockBytes;
            q8k += Q8_K_BlockBytes;
        }
        return HorizontalSumSse(acc) + deltaF;
    }

    /// <summary>AVX2 tier, one activation column.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static float VecDotIq_Q8_KAvx2<T>(byte* qk, byte* q8k, int superBlockCount) where T : struct, IIqFormat
    {
        Vector256<float> acc = Vector256<float>.Zero;
        float deltaF = 0;
        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float f = (float)Unsafe.ReadUnaligned<Half>(qk) * Unsafe.ReadUnaligned<float>(q8k) * T.Factor;
            sbyte* q8 = (sbyte*)(q8k + 4);
            short* bsums = (short*)(q8k + 260);
            Vector256<int> sumi = Vector256<int>.Zero;
            int deltaAcc = 0;
            for (int ib = 0; ib < 8; ib++)
            {
                IqSubBlock s = T.Decode(qk, ib);
                Vector256<byte> g = Vector256.Create(s.G0, s.G1, s.G2, s.G3).AsByte();
                Vector256<sbyte> sg = Vector256.Create(s.S0, s.S1, s.S2, s.S3).AsSByte();
                Vector256<short> p = Avx2.MultiplyAddAdjacent(g,
                    Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(q8 + ib * 32), sg));
                Vector256<short> scale = Vector256.Create(Vector128.Create((short)s.ScLo), Vector128.Create((short)s.ScHi));
                sumi = Avx2.Add(sumi, Avx2.MultiplyAddAdjacent(p, scale));
                if (T.HasDelta) deltaAcc += s.ScLo * s.Delta * (bsums[2 * ib] + bsums[2 * ib + 1]);
            }
            acc = Avx.Add(acc, Avx.Multiply(Vector256.Create(f), Avx.ConvertToVector256Single(sumi)));
            if (T.HasDelta) deltaF += f * 0.125f * deltaAcc;
            qk += T.BlockBytes;
            q8k += Q8_K_BlockBytes;
        }
        return HorizontalSumAvx2Float(acc) + deltaF;
    }

    /// <summary>AVX2 tier, four activation columns sharing one decode of the weights.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotIq_Q8_KAvx2x4<T>(byte* qk, byte* q8k, int q8RowBytes, int superBlockCount, float* result4)
        where T : struct, IIqFormat
    {
        Vector256<float> a0 = Vector256<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
        float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
        byte* x0 = q8k, x1 = q8k + q8RowBytes, x2 = q8k + 2L * q8RowBytes, x3 = q8k + 3L * q8RowBytes;
        for (int sb = 0; sb < superBlockCount; sb++)
        {
            float d = (float)Unsafe.ReadUnaligned<Half>(qk) * T.Factor;
            Vector256<int> s0 = Vector256<int>.Zero, s1 = s0, s2 = s0, s3 = s0;
            int da0 = 0, da1 = 0, da2 = 0, da3 = 0;
            for (int ib = 0; ib < 8; ib++)
            {
                IqSubBlock s = T.Decode(qk, ib);
                Vector256<byte> g = Vector256.Create(s.G0, s.G1, s.G2, s.G3).AsByte();
                Vector256<sbyte> sg = Vector256.Create(s.S0, s.S1, s.S2, s.S3).AsSByte();
                Vector256<short> scale = Vector256.Create(Vector128.Create((short)s.ScLo), Vector128.Create((short)s.ScHi));
                int off = 4 + ib * 32;
                s0 = Avx2.Add(s0, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(g,
                    Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(x0 + off), sg)), scale));
                s1 = Avx2.Add(s1, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(g,
                    Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(x1 + off), sg)), scale));
                s2 = Avx2.Add(s2, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(g,
                    Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(x2 + off), sg)), scale));
                s3 = Avx2.Add(s3, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(g,
                    Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(x3 + off), sg)), scale));
                if (T.HasDelta)
                {
                    int k0 = s.ScLo * s.Delta;
                    da0 += k0 * (((short*)(x0 + 260))[2 * ib] + ((short*)(x0 + 260))[2 * ib + 1]);
                    da1 += k0 * (((short*)(x1 + 260))[2 * ib] + ((short*)(x1 + 260))[2 * ib + 1]);
                    da2 += k0 * (((short*)(x2 + 260))[2 * ib] + ((short*)(x2 + 260))[2 * ib + 1]);
                    da3 += k0 * (((short*)(x3 + 260))[2 * ib] + ((short*)(x3 + 260))[2 * ib + 1]);
                }
            }
            float f0 = d * Unsafe.ReadUnaligned<float>(x0), f1 = d * Unsafe.ReadUnaligned<float>(x1),
                  f2 = d * Unsafe.ReadUnaligned<float>(x2), f3 = d * Unsafe.ReadUnaligned<float>(x3);
            a0 = Avx.Add(a0, Avx.Multiply(Vector256.Create(f0), Avx.ConvertToVector256Single(s0)));
            a1 = Avx.Add(a1, Avx.Multiply(Vector256.Create(f1), Avx.ConvertToVector256Single(s1)));
            a2 = Avx.Add(a2, Avx.Multiply(Vector256.Create(f2), Avx.ConvertToVector256Single(s2)));
            a3 = Avx.Add(a3, Avx.Multiply(Vector256.Create(f3), Avx.ConvertToVector256Single(s3)));
            if (T.HasDelta)
            {
                d0 += f0 * 0.125f * da0; d1 += f1 * 0.125f * da1; d2 += f2 * 0.125f * da2; d3 += f3 * 0.125f * da3;
            }
            qk += T.BlockBytes;
            x0 += Q8_K_BlockBytes; x1 += Q8_K_BlockBytes; x2 += Q8_K_BlockBytes; x3 += Q8_K_BlockBytes;
        }
        result4[0] = HorizontalSumAvx2Float(a0) + d0;
        result4[1] = HorizontalSumAvx2Float(a1) + d1;
        result4[2] = HorizontalSumAvx2Float(a2) + d2;
        result4[3] = HorizontalSumAvx2Float(a3) + d3;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static float VecDotIq_Q8_KPortable<T>(byte* qk, byte* q8k, int superBlockCount) where T : struct, IIqFormat =>
        Ssse3.IsSupported ? VecDotIq_Q8_KSse<T>(qk, q8k, superBlockCount) : VecDotIq_Q8_KScalar<T>(qk, q8k, superBlockCount);

    // ──────────────────── Rows / GEMV / GEMM drivers ────────────────────

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsIq<T>(byte* weights, byte* xQ8K, float* result, int m, int superBlockCount)
        where T : struct, IIqFormat
    {
        int rowBytes = superBlockCount * T.BlockBytes;
        if (Avx2.IsSupported)
            for (int row = 0; row < m; row++)
                result[row] = VecDotIq_Q8_KAvx2<T>(weights + (long)row * rowBytes, xQ8K, superBlockCount);
        else
            for (int row = 0; row < m; row++)
                result[row] = VecDotIq_Q8_KPortable<T>(weights + (long)row * rowBytes, xQ8K, superBlockCount);
    }

    [SkipLocalsInit]
    internal static void ComputeRowRangeIq<T>(byte* weights, byte* inputQ8, int q8RowBytes, float* c,
                                              int m, int n, int superBlockCount, int rowStart, int rowCount)
        where T : struct, IIqFormat
    {
        int rowBytes = superBlockCount * T.BlockBytes;
        float* tmp = stackalloc float[4];
        for (int row = rowStart; row < rowStart + rowCount; row++)
        {
            byte* w = weights + (long)row * rowBytes;
            int t = 0;
            if (Avx2.IsSupported)
            {
                for (; t + 4 <= n; t += 4)
                {
                    VecDotIq_Q8_KAvx2x4<T>(w, inputQ8 + (long)t * q8RowBytes, q8RowBytes, superBlockCount, tmp);
                    c[(long)t * m + row] = tmp[0];
                    c[(long)(t + 1) * m + row] = tmp[1];
                    c[(long)(t + 2) * m + row] = tmp[2];
                    c[(long)(t + 3) * m + row] = tmp[3];
                }
            }
            for (; t < n; t++)
                c[(long)t * m + row] = Avx2.IsSupported
                    ? VecDotIq_Q8_KAvx2<T>(w, inputQ8 + (long)t * q8RowBytes, superBlockCount)
                    : VecDotIq_Q8_KPortable<T>(w, inputQ8 + (long)t * q8RowBytes, superBlockCount);
        }
    }

    private struct IqGemmCtx
    {
        public byte* Weights;
        public byte* InputQ8;
        public float* C;
        public int M, N, SuperBlockCount, Q8RowBytes;
        public delegate*<byte*, byte*, int, float*, int, int, int, int, int, void> Range;
    }

    private static void IqGemmWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<IqGemmCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        ctx.Range(ctx.Weights, ctx.InputQ8, ctx.Q8RowBytes, ctx.C, ctx.M, ctx.N, ctx.SuperBlockCount, start, count);
    }

    [SkipLocalsInit]
    internal static void GemmIq<T>(byte* weights, float* b, float* c, int m, int k, int n,
                                   ComputeThreadPool? pool, byte* preQuantizedInput) where T : struct, IIqFormat
    {
        if (n == 1)
        {
            if (pool is null)
                GemmKQuant(weights, b, c, m, k, n, T.BlockBytes, &ComputeRowsIq<T>, preQuantizedInput);
            else
                GemmKQuantParallel(weights, b, c, m, k, n, T.BlockBytes, &ComputeRowsIq<T>, pool, preQuantizedInput);
            return;
        }

        if (k % KQuantGroupSize != 0)
            throw new ArgumentException($"k must be a multiple of {KQuantGroupSize}, got {k}", nameof(k));

        int sbc = k / KQuantGroupSize;
        int q8RowBytes = (k / Q8_K_GroupSize) * Q8_K_BlockBytes;
        byte[]? rented = preQuantizedInput is null ? ArrayPool<byte>.Shared.Rent(n * q8RowBytes) : null;
        try
        {
            fixed (byte* rentedPtr = rented)
            {
                byte* inputQ8 = preQuantizedInput;
                if (inputQ8 is null)
                {
                    inputQ8 = rentedPtr;
                    for (int t = 0; t < n; t++)
                        QuantizeF32ToQ8_K(b + (long)t * k, inputQ8 + (long)t * q8RowBytes, k);
                }

                if (pool is null || m < ParallelMinRows)
                {
                    ComputeRowRangeIq<T>(weights, inputQ8, q8RowBytes, c, m, n, sbc, 0, m);
                    return;
                }

                var ctx = new IqGemmCtx
                {
                    Weights = weights, InputQ8 = inputQ8, C = c, M = m, N = n,
                    SuperBlockCount = sbc, Q8RowBytes = q8RowBytes, Range = &ComputeRowRangeIq<T>,
                };
                pool.Dispatch((nint)(&ctx), &IqGemmWorker);
            }
        }
        finally
        {
            if (rented is not null) ArrayPool<byte>.Shared.Return(rented);
        }
    }

    [SkipLocalsInit]
    internal static void GemvIq<T>(byte* weights, float* x, float* result, int m, int k, ComputeThreadPool? pool)
        where T : struct, IIqFormat
    {
        if (pool is null || m < ParallelMinRows)
            GemvLowBitKQuant(weights, x, result, m, k, &ComputeRowsIq<T>);
        else
            GemvKQuantParallel(weights, x, result, m, k, T.BlockBytes, &ComputeRowsIq<T>, pool);
    }

    // ──────────────────── Type-keyed entry points ────────────────────

    /// <summary>True for the codebook IQ formats served by the generic packed × Q8_K kernel.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool IsCodebookIq(QuantizationType qt) =>
        qt is QuantizationType.IQ2_XXS or QuantizationType.IQ2_XS or QuantizationType.IQ2_S
            or QuantizationType.IQ3_XXS or QuantizationType.IQ3_S or QuantizationType.IQ1_S;

    /// <summary>Block bytes for a codebook IQ format.</summary>
    internal static int IqCodebookBlockBytes(QuantizationType qt) => QuantFormat.TryGetInfo(qt)!.Value.BlockBytes;

    /// <summary>Row-major ComputeRows pointer for a codebook IQ format (null if <paramref name="qt"/> is not one).</summary>
    internal static delegate*<byte*, byte*, float*, int, int, void> GetComputeRowsIq(QuantizationType qt) => qt switch
    {
        QuantizationType.IQ2_XXS => &ComputeRowsIq<Iq2XxsFmt>,
        QuantizationType.IQ2_XS => &ComputeRowsIq<Iq2XsFmt>,
        QuantizationType.IQ2_S => &ComputeRowsIq<Iq2SFmt>,
        QuantizationType.IQ3_XXS => &ComputeRowsIq<Iq3XxsFmt>,
        QuantizationType.IQ3_S => &ComputeRowsIq<Iq3SFmt>,
        QuantizationType.IQ1_S => &ComputeRowsIq<Iq1SFmt>,
        _ => null,
    };

    /// <summary>GEMV for any codebook IQ format.</summary>
    public static void GemvIQCodebook(QuantizationType qt, byte* weights, float* x, float* result, int m, int k,
                                      ComputeThreadPool? pool)
    {
        switch (qt)
        {
            case QuantizationType.IQ2_XXS: GemvIq<Iq2XxsFmt>(weights, x, result, m, k, pool); break;
            case QuantizationType.IQ2_XS: GemvIq<Iq2XsFmt>(weights, x, result, m, k, pool); break;
            case QuantizationType.IQ2_S: GemvIq<Iq2SFmt>(weights, x, result, m, k, pool); break;
            case QuantizationType.IQ3_XXS: GemvIq<Iq3XxsFmt>(weights, x, result, m, k, pool); break;
            case QuantizationType.IQ3_S: GemvIq<Iq3SFmt>(weights, x, result, m, k, pool); break;
            case QuantizationType.IQ1_S: GemvIq<Iq1SFmt>(weights, x, result, m, k, pool); break;
            default: throw new NotSupportedException($"{qt} is not a codebook IQ format.");
        }
    }

    /// <summary>GEMM for any codebook IQ format (row-partitioned, 4-column AVX2 blocking).</summary>
    public static void GemmIQCodebook(QuantizationType qt, byte* weights, float* b, float* c, int m, int k, int n,
                                      ComputeThreadPool? pool, byte* preQuantizedInput = null)
    {
        switch (qt)
        {
            case QuantizationType.IQ2_XXS: GemmIq<Iq2XxsFmt>(weights, b, c, m, k, n, pool, preQuantizedInput); break;
            case QuantizationType.IQ2_XS: GemmIq<Iq2XsFmt>(weights, b, c, m, k, n, pool, preQuantizedInput); break;
            case QuantizationType.IQ2_S: GemmIq<Iq2SFmt>(weights, b, c, m, k, n, pool, preQuantizedInput); break;
            case QuantizationType.IQ3_XXS: GemmIq<Iq3XxsFmt>(weights, b, c, m, k, n, pool, preQuantizedInput); break;
            case QuantizationType.IQ3_S: GemmIq<Iq3SFmt>(weights, b, c, m, k, n, pool, preQuantizedInput); break;
            case QuantizationType.IQ1_S: GemmIq<Iq1SFmt>(weights, b, c, m, k, n, pool, preQuantizedInput); break;
            default: throw new NotSupportedException($"{qt} is not a codebook IQ format.");
        }
    }
}
