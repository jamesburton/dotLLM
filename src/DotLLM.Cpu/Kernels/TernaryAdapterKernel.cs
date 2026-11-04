using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace DotLLM.Cpu.Kernels;

/// <summary>ISA selection for <see cref="TernaryAdapterKernel"/> (tests force a path; production uses <see cref="Auto"/>).</summary>
public enum TernaryKernelIsa
{
    /// <summary>Widest hardware-accelerated path available.</summary>
    Auto = 0,
    /// <summary>Portable scalar reference (the canonical order, executed lane by lane).</summary>
    Scalar = 1,
    /// <summary>Two <see cref="Vector256{T}"/> halves per 16-lane chunk.</summary>
    Vector256 = 2,
    /// <summary>One <see cref="Vector512{T}"/> per 16-lane chunk.</summary>
    Vector512 = 3,
}

/// <summary>
/// Multiplication-free ternary residual adapter kernel: <c>out = x + W_up * sigma(alpha_j * (W_down * x)_j)</c>
/// with <c>W_down, W_up</c> in {-1, 0, +1}, stored as two 64-column bitmasks (plus, minus) per row block
/// (2 bits/weight) and read in place. Weights are never expanded; accumulation is F32 in registers;
/// the only scratch is the rank-r intermediate supplied by the caller.
/// </summary>
/// <remarks>
/// <para><b>Packed layout.</b> A matrix of <c>rows x cols</c> is stored row-major; each row is
/// <c>ceil(cols/64)</c> blocks of two <c>ulong</c> words <c>[plus, minus]</c>. Bit <c>b</c> of block <c>k</c>
/// refers to column <c>64k + b</c>. Bits at columns &gt;= cols MUST be zero. A weight with both bits set is invalid
/// (the loader rejects it; kernels treat it as +x - x = 0 contribution order-dependent, so never produce it).</para>
/// <para><b>Canonical summation order</b> (identical for all ISAs, hence bit-exact between them). Columns are
/// grouped in chunks of 16 (lane l = column 16c + l; tail lanes read as +0). <i>Down:</i> per row j keep lane
/// accumulators <c>P[16], M[16]</c> = +0; for chunks c ascending <c>P[l] += plus ? x : +0</c>,
/// <c>M[l] += minus ? x : +0</c>; then <c>s = fold(P) - fold(M)</c> with
/// <c>fold</c> = <c>a[i]+=a[i+8] (i&lt;8); a[i]+=a[i+4] (i&lt;4); a[i]+=a[i+2] (i&lt;2); a[0]+a[1]</c>.
/// Then <c>h_j = alpha_j * sigma(s)</c> with ReLU <c>sigma(s) = s &gt; 0 ? s : 0</c> (NaN -&gt; 0) or identity.
/// <i>Up:</i> per output chunk c: <c>acc = +0</c>; for j ascending <c>acc += plus_j ? h_j : +0</c>, then
/// <c>acc -= minus_j ? h_j : +0</c>; <c>out = x + acc</c> (scalar adds, no FMA).</para>
/// </remarks>
public static unsafe class TernaryAdapterKernel
{
    /// <summary>Number of <c>ulong</c> words per packed row (plus/minus interleaved per 64-column block).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int WordsPerRow(int cols) => (cols + 63) / 64 * 2;

    private static readonly Vector256<int> s_bits256Lo = Vector256.Create(1, 2, 4, 8, 16, 32, 64, 128);
    private static readonly Vector256<int> s_bits256Hi = Vector256.Create(256, 512, 1024, 2048, 4096, 8192, 16384, 32768);
    private static readonly Vector512<int> s_bits512 =
        Vector512.Create(1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768);

    /// <summary>
    /// Applies the adapter to one residual row in place.
    /// </summary>
    /// <param name="row">Residual row of <paramref name="d"/> floats, updated in place.</param>
    /// <param name="d">Model (hidden) width.</param>
    /// <param name="rank">Total rank r.</param>
    /// <param name="down">Packed W_down, <c>rank</c> rows of <see cref="WordsPerRow"/> words.</param>
    /// <param name="up">Packed W_up transposed (<c>rank</c> rows over d output columns), same layout.</param>
    /// <param name="rankAlpha">Per-rank-row scale (adapter alpha), length <paramref name="rank"/>.</param>
    /// <param name="relu">ReLU (true) or identity (false) sigma.</param>
    /// <param name="h">Caller-owned scratch of at least <paramref name="rank"/> floats.</param>
    /// <param name="isa">ISA override (tests).</param>
    public static void ApplyRow(float* row, int d, int rank, ulong* down, ulong* up, float* rankAlpha, bool relu,
        float* h, TernaryKernelIsa isa = TernaryKernelIsa.Auto)
    {
        if (isa == TernaryKernelIsa.Auto)
            isa = Vector512.IsHardwareAccelerated ? TernaryKernelIsa.Vector512
                : Vector256.IsHardwareAccelerated ? TernaryKernelIsa.Vector256
                : TernaryKernelIsa.Scalar;
        int wpr = WordsPerRow(d);
        int fullChunks = d >> 4;
        int tail = d & 15;
        // Zero-padded copy of the tail chunk (16 floats on the stack; no heap).
        float* tailBuf = stackalloc float[16];
        for (int i = 0; i < 16; i++) tailBuf[i] = i < tail ? row[(fullChunks << 4) + i] : 0f;

        for (int j = 0; j < rank; j++)
        {
            ulong* w = down + (long)j * wpr;
            float s = isa switch
            {
                TernaryKernelIsa.Vector512 => DownRow512(row, tailBuf, tail != 0, fullChunks, w),
                TernaryKernelIsa.Vector256 => DownRow256(row, tailBuf, tail != 0, fullChunks, w),
                _ => DownRowScalar(row, tailBuf, tail != 0, fullChunks, w),
            };
            float a = relu ? (s > 0f ? s : 0f) : s;
            h[j] = rankAlpha[j] * a;
        }

        int totalChunks = fullChunks + (tail != 0 ? 1 : 0);
        for (int c = 0; c < totalChunks; c++)
        {
            bool isTail = c == fullChunks;
            switch (isa)
            {
                case TernaryKernelIsa.Vector512: UpChunk512(row, tailBuf, isTail, c, tail, rank, up, wpr, h); break;
                case TernaryKernelIsa.Vector256: UpChunk256(row, tailBuf, isTail, c, tail, rank, up, wpr, h); break;
                default: UpChunkScalar(row, tailBuf, isTail, c, tail, rank, up, wpr, h); break;
            }
        }
    }

    // 16 mask bits for chunk c of a row: block c/4 word (plus at +0, minus at +1), shift 16*(c%4).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Masks(ulong* rowWords, int c, out uint plus, out uint minus)
    {
        ulong* blk = rowWords + (c >> 2) * 2;
        int sh = (c & 3) << 4;
        plus = (uint)((blk[0] >> sh) & 0xFFFF);
        minus = (uint)((blk[1] >> sh) & 0xFFFF);
    }

    private static float Fold(float* a)
    {
        for (int i = 0; i < 8; i++) a[i] += a[i + 8];
        for (int i = 0; i < 4; i++) a[i] += a[i + 4];
        for (int i = 0; i < 2; i++) a[i] += a[i + 2];
        return a[0] + a[1];
    }

    // ---------------------------------------------------------------- scalar (canonical) ----

    private static float DownRowScalar(float* row, float* tailBuf, bool hasTail, int fullChunks, ulong* w)
    {
        float* p = stackalloc float[16];
        float* m = stackalloc float[16];
        for (int i = 0; i < 16; i++) { p[i] = 0f; m[i] = 0f; }
        int chunks = fullChunks + (hasTail ? 1 : 0);
        for (int c = 0; c < chunks; c++)
        {
            float* x = c == fullChunks ? tailBuf : row + (c << 4);
            Masks(w, c, out uint pm, out uint mm);
            for (int l = 0; l < 16; l++)
            {
                p[l] += ((pm >> l) & 1) != 0 ? x[l] : 0f;
                m[l] += ((mm >> l) & 1) != 0 ? x[l] : 0f;
            }
        }
        return Fold(p) - Fold(m);
    }

    private static void UpChunkScalar(float* row, float* tailBuf, bool isTail, int c, int tail, int rank, ulong* up, int wpr, float* h)
    {
        float* acc = stackalloc float[16];
        for (int l = 0; l < 16; l++) acc[l] = 0f;
        for (int j = 0; j < rank; j++)
        {
            Masks(up + (long)j * wpr, c, out uint pm, out uint mm);
            float hj = h[j];
            for (int l = 0; l < 16; l++)
            {
                acc[l] += ((pm >> l) & 1) != 0 ? hj : 0f;
                acc[l] -= ((mm >> l) & 1) != 0 ? hj : 0f;
            }
        }
        StoreChunk(row, tailBuf, isTail, c, tail, acc);
    }

    // ---------------------------------------------------------------- Vector256 ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Sel256(Vector256<float> x, Vector256<int> bits, uint mask16Half)
    {
        Vector256<int> m = Vector256.Equals(Vector256.BitwiseAnd(Vector256.Create((int)mask16Half), bits), bits);
        return Vector256.BitwiseAnd(x.AsInt32(), m).AsSingle();
    }

    private static float DownRow256(float* row, float* tailBuf, bool hasTail, int fullChunks, ulong* w)
    {
        Vector256<float> pLo = Vector256<float>.Zero, pHi = pLo, mLo = pLo, mHi = pLo;
        for (int c = 0; c < fullChunks; c++)
        {
            Masks(w, c, out uint pm, out uint mm);
            Vector256<float> xl = Vector256.Load(row + (c << 4));
            Vector256<float> xh = Vector256.Load(row + (c << 4) + 8);
            pLo += Sel256(xl, s_bits256Lo, pm); pHi += Sel256(xh, s_bits256Lo, pm >> 8);
            mLo += Sel256(xl, s_bits256Lo, mm); mHi += Sel256(xh, s_bits256Lo, mm >> 8);
        }
        if (hasTail)
        {
            Masks(w, fullChunks, out uint pm, out uint mm);
            Vector256<float> xl = Vector256.Load(tailBuf);
            Vector256<float> xh = Vector256.Load(tailBuf + 8);
            pLo += Sel256(xl, s_bits256Lo, pm); pHi += Sel256(xh, s_bits256Lo, pm >> 8);
            mLo += Sel256(xl, s_bits256Lo, mm); mHi += Sel256(xh, s_bits256Lo, mm >> 8);
        }
        float* p = stackalloc float[16];
        float* m = stackalloc float[16];
        pLo.Store(p); pHi.Store(p + 8); mLo.Store(m); mHi.Store(m + 8);
        return Fold(p) - Fold(m);
    }

    private static void UpChunk256(float* row, float* tailBuf, bool isTail, int c, int tail, int rank, ulong* up, int wpr, float* h)
    {
        Vector256<float> aLo = Vector256<float>.Zero, aHi = aLo;
        for (int j = 0; j < rank; j++)
        {
            Masks(up + (long)j * wpr, c, out uint pm, out uint mm);
            Vector256<float> hv = Vector256.Create(h[j]);
            aLo += Sel256(hv, s_bits256Lo, pm); aHi += Sel256(hv, s_bits256Lo, pm >> 8);
            aLo -= Sel256(hv, s_bits256Lo, mm); aHi -= Sel256(hv, s_bits256Lo, mm >> 8);
        }
        float* acc = stackalloc float[16];
        aLo.Store(acc); aHi.Store(acc + 8);
        StoreChunk(row, tailBuf, isTail, c, tail, acc);
    }

    // ---------------------------------------------------------------- Vector512 ----

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<float> Sel512(Vector512<float> x, uint mask16)
    {
        Vector512<int> m = Vector512.Equals(Vector512.BitwiseAnd(Vector512.Create((int)mask16), s_bits512), s_bits512);
        return Vector512.BitwiseAnd(x.AsInt32(), m).AsSingle();
    }

    private static float DownRow512(float* row, float* tailBuf, bool hasTail, int fullChunks, ulong* w)
    {
        Vector512<float> p = Vector512<float>.Zero, m = p;
        for (int c = 0; c < fullChunks; c++)
        {
            Masks(w, c, out uint pm, out uint mm);
            Vector512<float> x = Vector512.Load(row + (c << 4));
            p += Sel512(x, pm);
            m += Sel512(x, mm);
        }
        if (hasTail)
        {
            Masks(w, fullChunks, out uint pm, out uint mm);
            Vector512<float> x = Vector512.Load(tailBuf);
            p += Sel512(x, pm);
            m += Sel512(x, mm);
        }
        float* pb = stackalloc float[16];
        float* mb = stackalloc float[16];
        p.Store(pb); m.Store(mb);
        return Fold(pb) - Fold(mb);
    }

    private static void UpChunk512(float* row, float* tailBuf, bool isTail, int c, int tail, int rank, ulong* up, int wpr, float* h)
    {
        Vector512<float> a = Vector512<float>.Zero;
        for (int j = 0; j < rank; j++)
        {
            Masks(up + (long)j * wpr, c, out uint pm, out uint mm);
            Vector512<float> hv = Vector512.Create(h[j]);
            a += Sel512(hv, pm);
            a -= Sel512(hv, mm);
        }
        float* acc = stackalloc float[16];
        a.Store(acc);
        StoreChunk(row, tailBuf, isTail, c, tail, acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void StoreChunk(float* row, float* tailBuf, bool isTail, int c, int tail, float* acc)
    {
        float* x = isTail ? tailBuf : row + (c << 4);
        int n = isTail ? tail : 16;
        float* dst = row + (c << 4);
        for (int l = 0; l < n; l++) dst[l] = x[l] + acc[l];
    }
}
