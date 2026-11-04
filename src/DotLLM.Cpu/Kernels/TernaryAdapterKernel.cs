using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace DotLLM.Cpu.Kernels;

/// <summary>ISA selection for <see cref="TernaryAdapterKernel"/> (tests force a path; production uses <see cref="Auto"/>).</summary>
public enum TernaryKernelIsa
{
    /// <summary>Widest available path (AVX2 when supported, else scalar).</summary>
    Auto = 0,
    /// <summary>Portable scalar path (identical results; integer accumulation is order-independent).</summary>
    Scalar = 1,
    /// <summary>AVX2 masked add/sub on int8 activations.</summary>
    Avx2 = 2,
}

/// <summary>
/// One ternary residual circuit at one layer, in the kernel's in-place layout. Pointers are owned by the caller
/// (the hook). Both matrices are stored as <c>rank</c> rows over <c>d</c> columns of plus/minus bitmasks
/// (<see cref="TernaryAdapterKernel.WordsPerRow"/> words per row): <see cref="Down"/> is the package's down tensor as-is
/// (rank x d), <see cref="UpT"/> is the package's up tensor (d x rank) transposed at load.
/// </summary>
public unsafe struct TernaryCircuit
{
    /// <summary>Rank K of this circuit.</summary>
    public int Rank;
    /// <summary>Packed W_down masks, <c>Rank</c> rows.</summary>
    public ulong* Down;
    /// <summary>Packed W_up^T masks, <c>Rank</c> rows (row j = column j of W_up).</summary>
    public ulong* UpT;
    /// <summary>Per-row F32 scale of W_down (length Rank) - metadata from the package.</summary>
    public float* DownScale;
    /// <summary>Per-row F32 scale of W_up (length d, padded to a multiple of 8 with zeros) - metadata from the package.</summary>
    public float* UpScale;
    /// <summary>Manifest <c>Alpha</c>: multiplier on the residual delta; <c>Alpha &lt;= 0</c> disables the circuit (reference gate semantics).</summary>
    public float Alpha;
    /// <summary>Manifest <c>ActivationThreshold</c>: the circuit contributes only when the largest ReLU'd down-projection value exceeds it.</summary>
    public float Threshold;
}

/// <summary>
/// Multiplication-free ternary residual adapter kernel with the exact numerics of the NeuroSymbolicAI reference
/// (<c>Base.Host/SimulatedTransformerSpine</c> + <c>Expert.Model/TernaryExpert.Forward</c> over <c>Ternary.Core</c>):
/// <code>
/// x8  = absmax_int8(h)                         (scale 127/max|h|, round-half-even, clamp [-128,127]; the scale is NOT re-applied)
/// d_j = (float)(sum_c Wdown[j,c]*x8[c]) * downScale[j]
/// r_j = max(0, d_j);  m = max_j r_j;  open = m &gt; max(threshold, 1e-6)
/// q_j = clamp(round(r_j * (127/m)))             (int8)
/// u_d = (float)(sum_j Wup[d,j]*q_j) * upScale[d]
/// delta_d = u_d * ((m/127) * alpha)
/// h'_d = h_d + (0 + delta1_d + delta2_d + ...)  (circuits in registration order)
/// </code>
/// All weight arithmetic is integer masked add/subtract (no multiplies on weights), so every ISA path is bit-identical
/// to the reference; float operations are single scalar-equivalent IEEE ops in the reference's order (no FMA).
/// </summary>
public static unsafe class TernaryAdapterKernel
{
    /// <summary>Words (<c>ulong</c>) per packed row: <c>[plus, minus]</c> per 64-column block.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int WordsPerRow(int cols) => (cols + 63) / 64 * 2;

    /// <summary>Activation scratch length (int8) for width <paramref name="d"/>: padded to a multiple of 64.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int PaddedCols(int d) => (d + 63) & ~63;

    private static readonly Vector256<byte> s_bitSel = Vector256.Create(
        (byte)1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
    private static readonly Vector256<byte> s_shufIdx = Vector256.Create(
        (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    private static readonly Vector256<int> s_lane8 = Vector256.Create(1, 2, 4, 8, 16, 32, 64, 128);

    /// <summary>Reference absmax int8 quantization of one residual row; zero-fills <c>x8[d..PaddedCols(d))</c>.</summary>
    public static void QuantizeAbsMaxInt8(float* h, int d, sbyte* x8)
    {
        float maxAbs = 0f;
        for (int i = 0; i < d; i++)
        {
            float a = Math.Abs(h[i]);
            if (a > maxAbs) maxAbs = a;
        }
        float qs = maxAbs > 1e-6f ? 127.0f / maxAbs : 0.0f;
        for (int i = 0; i < d; i++)
        {
            int q = (int)Math.Round(h[i] * qs);
            x8[i] = (sbyte)Math.Clamp(q, -128, 127);
        }
        for (int i = d; i < PaddedCols(d); i++) x8[i] = 0;
    }

    /// <summary>
    /// Applies all <paramref name="count"/> circuits of one layer to one residual row in place (see class summary).
    /// </summary>
    /// <param name="row">Residual row (d floats), updated in place.</param>
    /// <param name="d">Model width.</param>
    /// <param name="circuits">Circuits in registration order.</param>
    /// <param name="count">Number of circuits.</param>
    /// <param name="x8">Scratch int8 activations, <see cref="PaddedCols"/> long.</param>
    /// <param name="q8">Scratch int8, at least the summed rank of the circuits.</param>
    /// <param name="dqf">Scratch float, at least <c>count + totalRank</c> (per-circuit dequant factor, NaN = closed; then the ReLU strips).</param>
    /// <param name="isa">ISA override (tests).</param>
    public static void ApplyRow(float* row, int d, TernaryCircuit* circuits, int count, sbyte* x8, sbyte* q8, float* dqf,
        TernaryKernelIsa isa = TernaryKernelIsa.Auto)
    {
        if (isa == TernaryKernelIsa.Auto) isa = Avx2.IsSupported ? TernaryKernelIsa.Avx2 : TernaryKernelIsa.Scalar;
        QuantizeAbsMaxInt8(row, d, x8);
        int wpr = WordsPerRow(d);
        int padded = PaddedCols(d);
        int off = 0;
        bool any = false;
        for (int c = 0; c < count; c++)
        {
            ref TernaryCircuit ci = ref circuits[c];
            dqf[c] = float.NaN;
            if (!(ci.Alpha > 0f)) { off += ci.Rank; continue; }   // Gate <= 0 => exact identity in the reference
            // Down-projection into q8 as a temporary float strip: values are quantized after max is known, so keep the
            // ReLU'd floats in a small stack buffer (rank is bounded by the caller's cap, 256).
            float* r = dqf + count + off;
            float maxAbs = 0f;
            for (int j = 0; j < ci.Rank; j++)
            {
                ulong* w = ci.Down + (long)j * wpr;
                int sum = isa == TernaryKernelIsa.Avx2 ? DownRowAvx2(x8, padded, w) : DownRowScalar(x8, d, w);
                float val = (float)sum * ci.DownScale[j];
                val = Math.Max(0.0f, val);
                r[j] = val;
                if (val > maxAbs) maxAbs = val;
            }
            if (maxAbs > Math.Max(ci.Threshold, 1e-6f))
            {
                float quantScale = 127.0f / maxAbs;
                for (int j = 0; j < ci.Rank; j++)
                {
                    int q = (int)Math.Round(r[j] * quantScale);
                    q8[off + j] = (sbyte)Math.Clamp(q, -128, 127);
                }
                dqf[c] = (maxAbs / 127.0f) * ci.Alpha;
                any = true;
            }
            off += ci.Rank;
        }
        if (!any) return;   // every circuit closed: delta == 0, row unchanged

        int chunks = (d + 7) >> 3;
        float* tmp = stackalloc float[8];
        for (int ch = 0; ch < chunks; ch++)
        {
            int c0 = ch << 3;
            int valid = Math.Min(8, d - c0);
            if (isa == TernaryKernelIsa.Avx2)
            {
                Vector256<float> total = Vector256<float>.Zero;
                int o = 0;
                for (int c = 0; c < count; c++)
                {
                    ref TernaryCircuit ci = ref circuits[c];
                    int rank = ci.Rank;
                    if (float.IsNaN(dqf[c])) { o += rank; continue; }
                    Vector256<int> acc = Vector256<int>.Zero;
                    int byteInWord = ch & 7;
                    long blk = (ch >> 3) * 2;
                    for (int j = 0; j < rank; j++)
                    {
                        byte* rowp = (byte*)(ci.UpT + (long)j * wpr + blk);
                        byte pb = rowp[byteInWord];
                        byte mb = rowp[8 + byteInWord];
                        if ((pb | mb) == 0) continue;
                        Vector256<int> hv = Vector256.Create((int)q8[o + j]);
                        Vector256<int> pm = Avx2.CompareEqual(Avx2.And(Vector256.Create((int)pb), s_lane8), s_lane8);
                        Vector256<int> mm = Avx2.CompareEqual(Avx2.And(Vector256.Create((int)mb), s_lane8), s_lane8);
                        acc = Avx2.Add(acc, Avx2.And(hv, pm));
                        acc = Avx2.Subtract(acc, Avx2.And(hv, mm));
                    }
                    Vector256<float> u = Avx.Multiply(Avx.ConvertToVector256Single(acc), Avx.LoadVector256(ci.UpScale + c0));
                    Vector256<float> delta = Avx.Multiply(u, Vector256.Create(dqf[c]));
                    total = Avx.Add(total, delta);
                    o += rank;
                }
                total.Store(tmp);
            }
            else
            {
                for (int l = 0; l < 8; l++) tmp[l] = 0f;
                int o = 0;
                for (int c = 0; c < count; c++)
                {
                    ref TernaryCircuit ci = ref circuits[c];
                    int rank = ci.Rank;
                    if (float.IsNaN(dqf[c])) { o += rank; continue; }
                    for (int l = 0; l < valid; l++)
                    {
                        int col = c0 + l;
                        int acc = 0;
                        for (int j = 0; j < rank; j++)
                        {
                            ulong* rowp = ci.UpT + (long)j * wpr + (col >> 6) * 2;
                            if (((rowp[0] >> (col & 63)) & 1UL) != 0) acc += q8[o + j];
                            if (((rowp[1] >> (col & 63)) & 1UL) != 0) acc -= q8[o + j];
                        }
                        float u = (float)acc * ci.UpScale[col];
                        tmp[l] = tmp[l] + u * dqf[c];
                    }
                    o += rank;
                }
            }
            for (int l = 0; l < valid; l++) row[c0 + l] = row[c0 + l] + tmp[l];
        }
    }

    /// <summary>Σ(plus·x) − Σ(minus·x) over one packed row, AVX2 (maddubs against ones; exact int32).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static int DownRowAvx2(sbyte* x8, int padded, ulong* w)
    {
        Vector256<byte> ones8 = Vector256.Create((byte)1);
        Vector256<short> ones16 = Vector256.Create((short)1);
        Vector256<int> acc = Vector256<int>.Zero;
        uint* wu = (uint*)w;
        int chunks = padded >> 5;
        for (int c = 0; c < chunks; c++)
        {
            int wi = (c >> 1) * 4 + (c & 1);
            uint p = wu[wi], m = wu[wi + 2];
            Vector256<sbyte> x = Avx.LoadVector256(x8 + (c << 5));
            Vector256<byte> px = Avx2.And(x.AsByte(), Expand32(p));
            Vector256<byte> mx = Avx2.And(x.AsByte(), Expand32(m));
            Vector256<short> d16 = Avx2.Subtract(Avx2.MultiplyAddAdjacent(ones8, px.AsSByte()), Avx2.MultiplyAddAdjacent(ones8, mx.AsSByte()));
            acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(d16, ones16));
        }
        return Vector256.Sum(acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Expand32(uint m)
    {
        Vector256<byte> b = Vector256.Create(m).AsByte();
        Vector256<byte> sh = Avx2.Shuffle(b, s_shufIdx);
        return Avx2.CompareEqual(Avx2.And(sh, s_bitSel), s_bitSel);
    }

    private static int DownRowScalar(sbyte* x8, int d, ulong* w)
    {
        int sum = 0;
        int blocks = (d + 63) / 64;
        for (int b = 0; b < blocks; b++)
        {
            ulong p = w[b * 2], m = w[b * 2 + 1];
            while (p != 0) { sum += x8[b * 64 + BitOperations.TrailingZeroCount(p)]; p &= p - 1; }
            while (m != 0) { sum -= x8[b * 64 + BitOperations.TrailingZeroCount(m)]; m &= m - 1; }
        }
        return sum;
    }
}
