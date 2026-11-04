using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Tests.Unit.Cpu.Kernels;

/// <summary>
/// EXPERIMENTAL probe kernels for #1616 part 2 (zero-multiplication kernels for ternary MODEL weights). Test-project only:
/// nothing here is wired into the engine. Both candidates read Bonsai-2 PQ2_0 weights (34-byte groups of 128 elements:
/// Half scale + 32 bytes, 4 consecutive 2-bit codes per byte, ascending bit offsets, code-1 = ternary) and use int8
/// activations with integer accumulation (CLAUDE.md precision rules: no F32 weight expansion).
/// <list type="bullet">
/// <item><b>K2 "masked add/sub"</b> (the issue's proposal): weights repacked to 2 bitmasks (plus/minus) per 32-element block,
/// same 34 bytes/group; per block the masks are expanded to byte masks (broadcast+pshufb+and+cmpeq, or vpermb on AVX-512
/// VBMI), ANDed with the int8 activations and summed (<c>sum(x&amp;P) - sum(x&amp;M)</c>) with VNNI/maddubs against ones.
/// No multiplies on the weights.</item>
/// <item><b>K3 "planar offset"</b> (a different idea): the PQ2_0 bytes are used IN PLACE with no unpack/interleave at all.
/// Activations are re-laid out per 128-group into 4 planes (plane i = elements 4b+i) and quantized per plane; then
/// <c>t_i = (byte &gt;&gt; 2i) &amp; 3</c> is unsigned {0,1,2} and <c>sum((t-1)*q) = dpbusd(t, q) - sum(q)</c>, with
/// <c>sum(q)</c> folded into a per-group scalar precomputed from the activations.</item>
/// </list>
/// </summary>
internal static unsafe class TernaryMaskedProbeKernels
{
    private const int GroupElems = 128;
    private const int GroupBytes = 34;

    // ------------------------------------------------------------------ K2 repack ----

    /// <summary>
    /// Repacks PQ2_0 rows to the bitmask34 layout per group: <c>[Half scale][P0..P3 : u32 x4][M0..M3 : u32 x4]</c>,
    /// block j covering elements 32j..32j+31, bit e = element 32j+e. Code 3 (a +2 weight, never produced by real PQ2_0
    /// tensors; asserted by the caller) is not representable and maps to plus.
    /// </summary>
    public static void RepackToBitmask34(byte* src, byte* dst, long rows, int k, out long codeThreeCount)
    {
        int groups = k / GroupElems;
        long threes = 0;
        for (long r = 0; r < rows; r++)
            for (int g = 0; g < groups; g++)
            {
                byte* s = src + (r * groups + g) * GroupBytes;
                byte* d = dst + (r * groups + g) * GroupBytes;
                d[0] = s[0]; d[1] = s[1];
                uint* p = (uint*)(d + 2);
                uint* m = (uint*)(d + 18);
                for (int j = 0; j < 4; j++)
                {
                    uint pm = 0, mm = 0;
                    for (int e = 0; e < 32; e++)
                    {
                        int el = 32 * j + e;
                        int code = (s[2 + el / 4] >> (2 * (el & 3))) & 3;
                        if (code == 3) threes++;
                        if (code >= 2) pm |= 1u << e;
                        else if (code == 0) mm |= 1u << e;
                    }
                    Unsafe.WriteUnaligned(p + j, pm);
                    Unsafe.WriteUnaligned(m + j, mm);
                }
            }
        codeThreeCount = threes;
    }

    // ------------------------------------------------------------------ activation prep ----

    /// <summary>Contiguous int8 activations + float block scales (de-blocked Q8_0): <c>q[n*k]</c>, <c>s[n*k/32]</c>.</summary>
    public static void PrepQ8Contiguous(float* x, int n, int k, sbyte* q, float* s)
    {
        int blocks = k / 32;
        byte* tmp = (byte*)NativeMemory.Alloc((nuint)(blocks * 34));
        try
        {
            for (int t = 0; t < n; t++)
            {
                MatMul.QuantizeF32ToQ8_0(x + (long)t * k, tmp, k);
                for (int b = 0; b < blocks; b++)
                {
                    s[(long)t * blocks + b] = MatMul.HalfToSingle(tmp + b * 34);
                    Unsafe.CopyBlockUnaligned(q + (long)t * k + b * 32, tmp + b * 34 + 2, 32);
                }
            }
        }
        finally { NativeMemory.Free(tmp); }
    }

    /// <summary>
    /// Planar quantization for K3. Per token and 128-group: 4 planes of 32 elements (plane i = x[4b+i]), each with its own
    /// absmax/127 scale. Outputs <c>q[n*k]</c> (plane-major within a group), <c>s[n*groups*4]</c>, and the per-group
    /// offset correction <c>c[n*groups] = sum_i s_i * sum(q_i)</c>.
    /// </summary>
    public static void PrepPlanar(float* x, int n, int k, sbyte* q, float* s, float* c)
    {
        int groups = k / GroupElems;
        for (int t = 0; t < n; t++)
            for (int g = 0; g < groups; g++)
            {
                float* xg = x + (long)t * k + g * GroupElems;
                sbyte* qg = q + (long)t * k + g * GroupElems;
                float corr = 0f;
                for (int i = 0; i < 4; i++)
                {
                    float amax = 0f;
                    for (int b = 0; b < 32; b++) amax = MathF.Max(amax, MathF.Abs(xg[4 * b + i]));
                    float sc = amax / 127f;
                    float inv = sc != 0f ? 1f / sc : 0f;
                    int qsum = 0;
                    for (int b = 0; b < 32; b++)
                    {
                        int v = (int)MathF.Round(xg[4 * b + i] * inv);
                        qg[i * 32 + b] = (sbyte)v;
                        qsum += v;
                    }
                    s[((long)t * groups + g) * 4 + i] = sc;
                    corr += sc * qsum;
                }
                c[(long)t * groups + g] = corr;
            }
    }

    // ------------------------------------------------------------------ K2 kernels ----

    private static readonly Vector256<byte> s_shufIdx = Vector256.Create(
        (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
    private static readonly Vector256<byte> s_bitSel = Vector256.Create(
        (byte)1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
        1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Expand256(uint m)
    {
        // pshufb is lane-local; the dword is broadcast to both lanes so bytes 2,3 are available in the high lane.
        Vector256<byte> b = Vector256.Create(m).AsByte();
        Vector256<byte> sh = Avx2.Shuffle(b, s_shufIdx);
        return Avx2.CompareEqual(Avx2.And(sh, s_bitSel), s_bitSel);
    }

    /// <summary>K2, 256-bit: rows [start, start+count) over n tokens. <paramref name="useVnni"/> selects vpdpbusd vs maddubs+maddwd.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void K2Rows256(byte* w, sbyte* xq, float* xs, float* c, int m, int start, int count, int k, int n, bool useVnni)
    {
        int groups = k / GroupElems;
        int blocks = k / 32;
        Vector256<byte> ones8 = Vector256.Create((byte)1);
        Vector256<short> ones16 = Vector256.Create((short)1);
        for (int r = start; r < start + count; r++)
        {
            byte* row = w + (long)r * groups * GroupBytes;
            for (int t = 0; t < n; t++)
            {
                sbyte* xt = xq + (long)t * k;
                float* st = xs + (long)t * blocks;
                Vector256<float> acc = Vector256<float>.Zero;
                for (int g = 0; g < groups; g++)
                {
                    byte* gp = row + g * GroupBytes;
                    float gs = MatMul.HalfToSingle(gp);
                    for (int j = 0; j < 4; j++)
                    {
                        Vector256<byte> pm = Expand256(Unsafe.ReadUnaligned<uint>(gp + 2 + 4 * j));
                        Vector256<byte> mm = Expand256(Unsafe.ReadUnaligned<uint>(gp + 18 + 4 * j));
                        Vector256<sbyte> x = Unsafe.ReadUnaligned<Vector256<sbyte>>(xt + g * 128 + j * 32);
                        Vector256<byte> px = Avx2.And(x.AsByte(), pm);
                        Vector256<byte> mx = Avx2.And(x.AsByte(), mm);
                        Vector256<int> isum;
                        if (useVnni)
                        {
                            Vector256<int> ip = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, ones8, px.AsSByte());
                            Vector256<int> im = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, ones8, mx.AsSByte());
                            isum = Avx2.Subtract(ip, im);
                        }
                        else
                        {
                            Vector256<short> sp = Avx2.MultiplyAddAdjacent(ones8, px.AsSByte());
                            Vector256<short> sm = Avx2.MultiplyAddAdjacent(ones8, mx.AsSByte());
                            isum = Avx2.MultiplyAddAdjacent(Avx2.Subtract(sp, sm), ones16);
                        }
                        float scale = st[g * 4 + j] * gs;
                        acc = Fma.MultiplyAdd(Vector256.Create(scale), Avx.ConvertToVector256Single(isum), acc);
                    }
                }
                c[(long)t * m + r] = HSum(acc);
            }
        }
    }

    /// <summary>
    /// K2, 512-bit (AVX-512BW + VBMI): processes two 32-element blocks (64 activations) per step; the 64-bit plus/minus mask is
    /// expanded with one vpermb. On Zen 5 the 512-bit ops issue as two 256-bit halves, so this is expected to tie at best.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void K2Rows512(byte* w, sbyte* xq, float* xs, float* c, int m, int start, int count, int k, int n)
    {
        int groups = k / GroupElems;
        int blocks = k / 32;
        Vector512<byte> permIdx = Vector512.Create(
            (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
            4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7);
        Vector512<byte> bitSel = Vector512.Create(s_bitSel, s_bitSel);
        Vector512<byte> ones8 = Vector512.Create((byte)1);
        Vector512<short> ones16 = Vector512.Create((short)1);
        for (int r = start; r < start + count; r++)
        {
            byte* row = w + (long)r * groups * GroupBytes;
            for (int t = 0; t < n; t++)
            {
                sbyte* xt = xq + (long)t * k;
                float* st = xs + (long)t * blocks;
                Vector512<float> acc = Vector512<float>.Zero;
                for (int g = 0; g < groups; g++)
                {
                    byte* gp = row + g * GroupBytes;
                    float gs = MatMul.HalfToSingle(gp);
                    for (int j = 0; j < 4; j += 2)
                    {
                        ulong pbits = Unsafe.ReadUnaligned<ulong>(gp + 2 + 4 * j);
                        ulong mbits = Unsafe.ReadUnaligned<ulong>(gp + 18 + 4 * j);
                        Vector512<byte> pm = Avx512BW.CompareEqual(Avx512F.And(Avx512Vbmi.PermuteVar64x8(Vector512.Create(pbits).AsByte(), permIdx), bitSel), bitSel);
                        Vector512<byte> mm = Avx512BW.CompareEqual(Avx512F.And(Avx512Vbmi.PermuteVar64x8(Vector512.Create(mbits).AsByte(), permIdx), bitSel), bitSel);
                        Vector512<byte> x = Unsafe.ReadUnaligned<Vector512<byte>>(xt + g * 128 + j * 32);
                        Vector512<short> sp = Avx512BW.MultiplyAddAdjacent(ones8, Avx512F.And(x, pm).AsSByte());
                        Vector512<short> sm = Avx512BW.MultiplyAddAdjacent(ones8, Avx512F.And(x, mm).AsSByte());
                        Vector512<int> isum = Avx512BW.MultiplyAddAdjacent(Avx512BW.Subtract(sp, sm), ones16);
                        Vector512<float> sc = Vector512.Create(
                            Vector256.Create(st[g * 4 + j] * gs), Vector256.Create(st[g * 4 + j + 1] * gs));
                        acc = Avx512F.FusedMultiplyAdd(sc, Avx512F.ConvertToVector512Single(isum), acc);
                    }
                }
                c[(long)t * m + r] = HSum(Avx.Add(acc.GetLower(), acc.GetUpper()));
            }
        }
    }

    // ------------------------------------------------------------------ K3 kernel ----

    /// <summary>K3: native PQ2_0 rows, planar activations. See class summary.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void K3Rows(byte* w, sbyte* xq, float* xs, float* xc, float* c, int m, int start, int count, int k, int n, bool useVnni)
    {
        int groups = k / GroupElems;
        Vector256<byte> mask3 = Vector256.Create((byte)3);
        Vector256<short> ones16 = Vector256.Create((short)1);
        Vector256<byte> ones8 = Vector256.Create((byte)1);
        for (int r = start; r < start + count; r++)
        {
            byte* row = w + (long)r * groups * GroupBytes;
            // Token-blocked inner loop: the shift/and decode of a group's four planes is shared by all n tokens only when
            // n is small enough to keep accumulators in registers; the simple form below re-decodes per token and
            // relies on the L1-resident group (34 B) - decode cost is 3 shifts + 4 ands per group per token.
            for (int t = 0; t < n; t++)
            {
                sbyte* xt = xq + (long)t * k;
                float* st = xs + (long)t * groups * 4;
                float* ct = xc + (long)t * groups;
                Vector256<float> acc = Vector256<float>.Zero;
                float corr = 0f;
                for (int g = 0; g < groups; g++)
                {
                    byte* gp = row + g * GroupBytes;
                    float gs = MatMul.HalfToSingle(gp);
                    Vector256<byte> v = Unsafe.ReadUnaligned<Vector256<byte>>(gp + 2);
                    Vector256<byte> t0 = Avx2.And(v, mask3);
                    Vector256<byte> t1 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 2).AsByte(), mask3);
                    Vector256<byte> t2 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte(), mask3);
                    Vector256<byte> t3 = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 6).AsByte(), mask3);
                    sbyte* xg = xt + g * 128;
                    Vector256<int> i0, i1, i2, i3;
                    if (useVnni)
                    {
                        i0 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t0, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg));
                        i1 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t1, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 32));
                        i2 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t2, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 64));
                        i3 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, t3, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 96));
                    }
                    else
                    {
                        i0 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t0, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg)), ones16);
                        i1 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t1, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 32)), ones16);
                        i2 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t2, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 64)), ones16);
                        i3 = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(t3, Unsafe.ReadUnaligned<Vector256<sbyte>>(xg + 96)), ones16);
                    }
                    // sum_i s_i * dpbusd(t_i, q_i) for the group, then * g
                    Vector256<float> gacc = Avx.Multiply(Avx.ConvertToVector256Single(i0), Vector256.Create(st[g * 4]));
                    gacc = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i1), Vector256.Create(st[g * 4 + 1]), gacc);
                    gacc = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i2), Vector256.Create(st[g * 4 + 2]), gacc);
                    gacc = Fma.MultiplyAdd(Avx.ConvertToVector256Single(i3), Vector256.Create(st[g * 4 + 3]), gacc);
                    acc = Fma.MultiplyAdd(gacc, Vector256.Create(gs), acc);
                    corr = MathF.FusedMultiplyAdd(gs, ct[g], corr);
                }
                c[(long)t * m + r] = HSum(acc) - corr;
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float HSum(Vector256<float> v)
    {
        Vector128<float> s = Sse.Add(v.GetLower(), v.GetUpper());
        s = Sse.Add(s, Sse.MoveHighToLow(s, s));
        s = Sse.AddScalar(s, Sse.Shuffle(s, s, 0x55));
        return s.ToScalar();
    }

    /// <summary>Read-only streaming over a row range (roofline probe): XOR of all 32-byte words.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static byte StreamBytes(byte* p, long bytes)
    {
        Vector256<byte> a0 = default, a1 = default, a2 = default, a3 = default;
        long i = 0;
        for (; i + 128 <= bytes; i += 128)
        {
            a0 ^= Unsafe.ReadUnaligned<Vector256<byte>>(p + i);
            a1 ^= Unsafe.ReadUnaligned<Vector256<byte>>(p + i + 32);
            a2 ^= Unsafe.ReadUnaligned<Vector256<byte>>(p + i + 64);
            a3 ^= Unsafe.ReadUnaligned<Vector256<byte>>(p + i + 96);
        }
        Vector256<byte> a = (a0 ^ a1) ^ (a2 ^ a3);
        byte r = 0;
        for (int l = 0; l < 32; l++) r ^= a.GetElement(l);
        for (; i < bytes; i++) r ^= p[i];
        return r;
    }
}
