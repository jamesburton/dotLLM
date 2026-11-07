using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Eight-row transposed int8 Q8_0 x Q8_0 kernels on AVX-VNNI (#1119): the Q4_0/Q4_1 eight-row pattern of #1159 applied to Q8_0.
/// </summary>
/// <remarks>
/// <para><b>Why.</b> The production Q8_0 prefill kernel (<see cref="Q8_0Rows4Tokens4Avx2"/>) spends about six vector ops per row per
/// 32-element block (load, sign, <c>vpmaddubsw</c>, <c>vpmaddwd</c>, convert, FMA) and measured no faster than the F32 GEMM on Zen 5
/// (#1109: 297-313 vs 504 GFLOP/s). Here a block of eight rows is loaded once per 32 tokens and transposed so that lane <c>r</c> of
/// vector <c>V_k</c> holds row <c>r</c>'s elements <c>4k..4k+3</c>; a token's eight block sums are then eight <c>vpdpbusd</c> against
/// broadcast four-byte chunks of its activation block, with no horizontal adds and one convert + multiply + add per token per block
/// for all eight rows: about 26 ops per 256 MACs against about 48 before.</para>
///
/// <para><b>Sign handling.</b> <c>vpdpbusd</c> multiplies unsigned by signed bytes. The GEMM biases the <em>weights</em> (<c>w + 128</c>,
/// i.e. <c>w XOR 0x80</c>, computed once per block per eight rows) and keeps the activations signed, so
/// <c>Σ (w+128)·x = Σ w·x + 128·Σ x</c> and the correction <c>128·Σ x</c> is one integer per (token, block), shared by every row. It is
/// computed with the activation scales in a preprocessing pass. All integer arithmetic is exact (a lane holds at most
/// 8·4·255·127 &lt; 2^31). The GEMV uses the sign trick (<c>|x|</c> times <c>sign(w, x)</c>), which needs no correction.</para>
///
/// <para><b>Precision contract.</b> Every output is bit-exact with <see cref="VecDotQ8_0Scalar"/>: the per-block integer sum is exact and
/// the float recurrence is the scalar one, <c>sumf += (dw·dx)·sumi</c> in block order (multiply then add, no FMA, one accumulator per
/// row). Because decode and prefill both run this recurrence on the AVX-VNNI tier, a token's output does not depend on its batch
/// (#530), and both equal the scalar reference, which is what makes the tests bit-exact rather than tolerance-bounded.</para>
///
/// <para><b>Tiers.</b> 256-bit AVX-VNNI only. .NET 10 exposes 512-bit <c>vpdpbusd</c> solely through <c>Avx10v1.V512</c>, which reports
/// unsupported on Zen 5 (Strix Halo), and there is no <c>Avx512Vnni</c> class, so no 512-bit variant exists on the mainstream SDK. An
/// AVX2-only (no VNNI) transposed variant is not written: with <c>vpmaddubsw</c> the <c>w+128</c> operand saturates the int16 pair sum
/// (255·127·2 &gt; 32767), and the sign-trick form needs a per-row <c>vpsignb</c> per token, which brings the op count back to the
/// existing kernels'. Hosts without AVX-VNNI keep the existing Q8_0 kernels unchanged. <c>DOTLLM_CPU_Q8_ROWS8=0</c> restores the
/// previous kernels on VNNI hosts (A/B switch).</para>
/// </remarks>
public static unsafe partial class MatMul
{
    private static readonly bool s_q8Rows8EnvOff = Environment.GetEnvironmentVariable("DOTLLM_CPU_Q8_ROWS8") == "0";

    /// <summary>Runtime switch for the eight-row Q8_0 kernels (default on; <c>DOTLLM_CPU_Q8_ROWS8=0</c> turns them off). Settable for in-process A/B.</summary>
    public static bool Q8_0Rows8Enabled { get; set; } = !s_q8Rows8EnvOff;

    /// <summary>True when the eight-row transposed Q8_0 kernels serve Q8_0 GEMV and GEMM (AVX-VNNI present, switch on).</summary>
    internal static bool Q8_0Rows8Tier
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        get => CpuFeatures.AvxVnni && Q8_0Rows8Enabled;
    }

    /// <summary>Tokens whose accumulators <see cref="ComputeRowsQ8_0MultiVnni_8Rows"/> keeps on the stack per pass over the blocks.</summary>
    private const int Q8MultiTokenChunk = 32;

    /// <summary>Rows per dynamic claim in the pooled eight-row GEMM: 8-aligned, large enough to amortise the claim.</summary>
    private const int Q8Rows8ClaimRows = 64;

    /// <summary>Fills eight byte offsets for rows <c>first..first+7</c>, clamping rows past <paramref name="m"/> to the last valid row (their lanes are discarded).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Q8RowOffsets(int* rowOff, int first, int m, int rowBytes)
    {
        for (int r = 0; r < 8; r++)
            rowOff[r] = Math.Min(first + r, m - 1) * rowBytes;
    }

    /// <summary>
    /// Eight Q8_0 rows against one Q8_0 activation, one row per 32-bit lane, bit-exact per row with <see cref="VecDotQ8_0Scalar"/>.
    /// Row <c>r</c>'s first block is at <c>w + rowOff[r]</c>; offsets may repeat (clamped tails).
    /// </summary>
    /// <param name="w">Base of the weight matrix.</param>
    /// <param name="rowOff">Eight byte offsets of the rows' first blocks.</param>
    /// <param name="xQ8">Q8_0 activation row.</param>
    /// <param name="blockCount">Blocks per row.</param>
    /// <param name="results">Eight outputs.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void VecDotQ8_0Vnni_8Rows(byte* w, int* rowOff, byte* xQ8, int blockCount, float* results)
    {
        Vector256<int> rowIdx = Unsafe.ReadUnaligned<Vector256<int>>(rowOff);
        byte* w0 = w + rowOff[0], w1 = w + rowOff[1], w2 = w + rowOff[2], w3 = w + rowOff[3];
        byte* w4 = w + rowOff[4], w5 = w + rowOff[5], w6 = w + rowOff[6], w7 = w + rowOff[7];
        Vector256<float> acc = Vector256<float>.Zero;
        Vector256<int> zero = Vector256<int>.Zero;

        for (int block = 0; block < blockCount; block++)
        {
            nint bo = (nint)block * Q8_0BlockBytes;
            byte* xb = xQ8 + bo;
            Vector256<sbyte> vx = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2);
            Vector256<byte> absX = Avx2.Sign(vx, vx).AsByte();

            Vector256<int> p0 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w0 + bo + 2), vx));
            Vector256<int> p1 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w1 + bo + 2), vx));
            Vector256<int> p2 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w2 + bo + 2), vx));
            Vector256<int> p3 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w3 + bo + 2), vx));
            Vector256<int> p4 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w4 + bo + 2), vx));
            Vector256<int> p5 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w5 + bo + 2), vx));
            Vector256<int> p6 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w6 + bo + 2), vx));
            Vector256<int> p7 = AvxVnni.MultiplyWideningAndAdd(zero, absX, Avx2.Sign(Unsafe.ReadUnaligned<Vector256<sbyte>>(w7 + bo + 2), vx));

            // [p0..p3 low-lane partials | p0..p3 high-lane partials], likewise p4..p7; folding the lanes gives the eight exact sums.
            Vector256<int> s0123 = Avx2.HorizontalAdd(Avx2.HorizontalAdd(p0, p1), Avx2.HorizontalAdd(p2, p3));
            Vector256<int> s4567 = Avx2.HorizontalAdd(Avx2.HorizontalAdd(p4, p5), Avx2.HorizontalAdd(p6, p7));
            Vector256<int> sumi = Avx2.Add(
                Avx2.Permute2x128(s0123, s4567, 0x20),
                Avx2.Permute2x128(s0123, s4567, 0x31));

            // Each row's Half scale sits at byte 0 of its block: one dword gather (the upper 16 bits are masked off by the convert).
            Vector256<float> dw = HalfToSingleAvx2(Avx2.GatherVector256((int*)(w + bo), rowIdx, 1));
            Vector256<float> scale = Avx.Multiply(dw, Vector256.Create(HalfToSingle(xb)));
            acc = Avx.Add(acc, Avx.Multiply(scale, Avx.ConvertToVector256Single(sumi)));
        }

        Unsafe.WriteUnaligned(results, acc);
    }

    /// <summary>
    /// Row-major Q8_0 GEMV over <paramref name="m"/> rows with the eight-row kernel; the last group of fewer than eight rows runs the
    /// same kernel with its row offsets clamped (duplicate lanes are dropped), so every row is bit-exact with the scalar dot.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ8_0Vnni_8Rows(byte* weights, byte* xQ8, float* result, int m, int blockCount)
    {
        if (m <= 0) return;
        int rowBytes = blockCount * Q8_0BlockBytes;
        int* rowOff = stackalloc int[8];
        float* tmp = stackalloc float[8];
        int row = 0;
        for (; row + 8 <= m; row += 8)
        {
            Q8RowOffsets(rowOff, row, m, rowBytes);
            VecDotQ8_0Vnni_8Rows(weights, rowOff, xQ8, blockCount, result + row);
        }
        if (row < m)
        {
            Q8RowOffsets(rowOff, row, m, rowBytes);
            VecDotQ8_0Vnni_8Rows(weights, rowOff, xQ8, blockCount, tmp);
            for (int r = 0; row + r < m; r++) result[row + r] = tmp[r];
        }
    }

    /// <summary>
    /// Per token and block, the activation scale as a float and <c>128·Σ x</c> as an int (the weight-bias correction of the GEMM):
    /// <c>dx[t·blockCount + b]</c>, <c>corr[t·blockCount + b]</c>.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void Q8_0ActivationTablesVnni(byte* xQ8, int q8RowBytes, int n, int blockCount, float* dx, int* corr)
    {
        Vector256<byte> ones = Vector256.Create((byte)1);
        for (int t = 0; t < n; t++)
        {
            byte* xrow = xQ8 + (long)t * q8RowBytes;
            for (int b = 0; b < blockCount; b++)
            {
                byte* xb = xrow + b * Q8_0BlockBytes;
                dx[t * blockCount + b] = HalfToSingle(xb);
                Vector256<int> s = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, ones, Unsafe.ReadUnaligned<Vector256<sbyte>>(xb + 2));
                corr[t * blockCount + b] = HorizontalSumAvx2(s) << 7;
            }
        }
    }

    /// <summary>Per 128-bit lane, transposes eight rows' eight dwords: output <c>k</c> lane <c>r</c> = input row <c>r</c> dword <c>k</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Transpose8x8(
        Vector256<int> r0, Vector256<int> r1, Vector256<int> r2, Vector256<int> r3,
        Vector256<int> r4, Vector256<int> r5, Vector256<int> r6, Vector256<int> r7,
        out Vector256<byte> v0, out Vector256<byte> v1, out Vector256<byte> v2, out Vector256<byte> v3,
        out Vector256<byte> v4, out Vector256<byte> v5, out Vector256<byte> v6, out Vector256<byte> v7)
    {
        Vector256<long> t0 = Avx2.UnpackLow(r0, r1).AsInt64(), t1 = Avx2.UnpackHigh(r0, r1).AsInt64();
        Vector256<long> t2 = Avx2.UnpackLow(r2, r3).AsInt64(), t3 = Avx2.UnpackHigh(r2, r3).AsInt64();
        Vector256<long> t4 = Avx2.UnpackLow(r4, r5).AsInt64(), t5 = Avx2.UnpackHigh(r4, r5).AsInt64();
        Vector256<long> t6 = Avx2.UnpackLow(r6, r7).AsInt64(), t7 = Avx2.UnpackHigh(r6, r7).AsInt64();
        // u_k: rows 0-3 dword k in the low lane, dword k+4 in the high lane; w_k: rows 4-7 likewise.
        Vector256<long> u0 = Avx2.UnpackLow(t0, t2), u1 = Avx2.UnpackHigh(t0, t2), u2 = Avx2.UnpackLow(t1, t3), u3 = Avx2.UnpackHigh(t1, t3);
        Vector256<long> w0 = Avx2.UnpackLow(t4, t6), w1 = Avx2.UnpackHigh(t4, t6), w2 = Avx2.UnpackLow(t5, t7), w3 = Avx2.UnpackHigh(t5, t7);
        v0 = Avx2.Permute2x128(u0, w0, 0x20).AsByte(); v4 = Avx2.Permute2x128(u0, w0, 0x31).AsByte();
        v1 = Avx2.Permute2x128(u1, w1, 0x20).AsByte(); v5 = Avx2.Permute2x128(u1, w1, 0x31).AsByte();
        v2 = Avx2.Permute2x128(u2, w2, 0x20).AsByte(); v6 = Avx2.Permute2x128(u2, w2, 0x31).AsByte();
        v3 = Avx2.Permute2x128(u3, w3, 0x20).AsByte(); v7 = Avx2.Permute2x128(u3, w3, 0x31).AsByte();
    }

    /// <summary>
    /// Multi-column eight-row Q8_0 GEMM: <c>c[t·ldc + r] = dot(row r, xQ8[t])</c> for <paramref name="rows"/> (1..8) rows and every
    /// <c>t &lt; n</c>, bit-exact per element with <see cref="VecDotQ8_0Scalar"/> (and so with the n = 1 kernel).
    /// Each block's eight rows are biased to unsigned (<c>XOR 0x80</c>) and transposed once per <see cref="Q8MultiTokenChunk"/> tokens;
    /// a token's block sum is eight <c>vpdpbusd</c> against broadcast dwords of its activation block minus the shared
    /// <c>128·Σ x</c> correction from <paramref name="corr"/>.
    /// </summary>
    /// <param name="w">Base of the weight matrix.</param>
    /// <param name="rowOff">Eight byte offsets of the rows' first blocks (clamped for tails).</param>
    /// <param name="rows">Valid rows among the eight (1..8).</param>
    /// <param name="xQ8">Q8_0 activations, one row per token.</param>
    /// <param name="q8RowBytes">Activation row stride in bytes.</param>
    /// <param name="dx">Per token and block, the activation scale (<see cref="Q8_0ActivationTablesVnni"/>).</param>
    /// <param name="corr">Per token and block, <c>128·Σ x</c>.</param>
    /// <param name="n">Token count.</param>
    /// <param name="blockCount">Blocks per row.</param>
    /// <param name="c">Output for row 0 of token 0; token <c>t</c> starts at <c>c + t·ldc</c>.</param>
    /// <param name="ldc">Output stride between tokens.</param>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void ComputeRowsQ8_0MultiVnni_8Rows(byte* w, int* rowOff, int rows, byte* xQ8, int q8RowBytes,
        float* dx, int* corr, int n, int blockCount, float* c, int ldc)
    {
        Vector256<int> rowIdx = Unsafe.ReadUnaligned<Vector256<int>>(rowOff);
        byte* w0 = w + rowOff[0], w1 = w + rowOff[1], w2 = w + rowOff[2], w3 = w + rowOff[3];
        byte* w4 = w + rowOff[4], w5 = w + rowOff[5], w6 = w + rowOff[6], w7 = w + rowOff[7];
        Vector256<int> bias = Vector256.Create(unchecked((int)0x80808080));
        Vector256<int> zero = Vector256<int>.Zero;
        Vector256<float>* acc = stackalloc Vector256<float>[Q8MultiTokenChunk];
        float* tmp = stackalloc float[8];

        for (int t0 = 0; t0 < n; t0 += Q8MultiTokenChunk)
        {
            int nt = Math.Min(Q8MultiTokenChunk, n - t0);
            for (int j = 0; j < nt; j++) acc[j] = Vector256<float>.Zero;
            byte* x0 = xQ8 + (long)t0 * q8RowBytes + 2;
            float* dx0 = dx + (long)t0 * blockCount;
            int* corr0 = corr + (long)t0 * blockCount;

            for (int block = 0; block < blockCount; block++)
            {
                nint bo = (nint)block * Q8_0BlockBytes + 2;
                Transpose8x8(
                    Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w0 + bo), bias), Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w1 + bo), bias),
                    Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w2 + bo), bias), Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w3 + bo), bias),
                    Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w4 + bo), bias), Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w5 + bo), bias),
                    Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w6 + bo), bias), Avx2.Xor(Unsafe.ReadUnaligned<Vector256<int>>(w7 + bo), bias),
                    out Vector256<byte> v0, out Vector256<byte> v1, out Vector256<byte> v2, out Vector256<byte> v3,
                    out Vector256<byte> v4, out Vector256<byte> v5, out Vector256<byte> v6, out Vector256<byte> v7);

                Vector256<float> dw = HalfToSingleAvx2(Avx2.GatherVector256((int*)(w + bo - 2), rowIdx, 1));

                for (int j = 0; j < nt; j++)
                {
                    int* xq = (int*)(x0 + (long)j * q8RowBytes + bo - 2);
                    Vector256<int> s0 = AvxVnni.MultiplyWideningAndAdd(zero, v0, Avx2.BroadcastScalarToVector256(xq).AsSByte());
                    Vector256<int> s1 = AvxVnni.MultiplyWideningAndAdd(zero, v1, Avx2.BroadcastScalarToVector256(xq + 1).AsSByte());
                    s0 = AvxVnni.MultiplyWideningAndAdd(s0, v2, Avx2.BroadcastScalarToVector256(xq + 2).AsSByte());
                    s1 = AvxVnni.MultiplyWideningAndAdd(s1, v3, Avx2.BroadcastScalarToVector256(xq + 3).AsSByte());
                    s0 = AvxVnni.MultiplyWideningAndAdd(s0, v4, Avx2.BroadcastScalarToVector256(xq + 4).AsSByte());
                    s1 = AvxVnni.MultiplyWideningAndAdd(s1, v5, Avx2.BroadcastScalarToVector256(xq + 5).AsSByte());
                    s0 = AvxVnni.MultiplyWideningAndAdd(s0, v6, Avx2.BroadcastScalarToVector256(xq + 6).AsSByte());
                    s1 = AvxVnni.MultiplyWideningAndAdd(s1, v7, Avx2.BroadcastScalarToVector256(xq + 7).AsSByte());
                    Vector256<int> sumi = Avx2.Subtract(Avx2.Add(s0, s1), Avx2.BroadcastScalarToVector256(corr0 + (long)j * blockCount + block));

                    Vector256<float> scale = Avx.Multiply(dw, Avx2.BroadcastScalarToVector256(dx0 + (long)j * blockCount + block));
                    acc[j] = Avx.Add(acc[j], Avx.Multiply(scale, Avx.ConvertToVector256Single(sumi)));
                }
            }

            if (rows == 8)
            {
                for (int j = 0; j < nt; j++)
                    Unsafe.WriteUnaligned(c + (long)(t0 + j) * ldc, acc[j]);
            }
            else
            {
                for (int j = 0; j < nt; j++)
                {
                    Unsafe.WriteUnaligned(tmp, acc[j]);
                    float* o = c + (long)(t0 + j) * ldc;
                    for (int r = 0; r < rows; r++) o[r] = tmp[r];
                }
            }
        }
    }

    /// <summary>Rows <c>[rowStart, rowStart + rowCount)</c> of the eight-row GEMM, every token.</summary>
    [SkipLocalsInit]
    private static void GemmQ8_0Rows8Range(byte* weights, int rowStart, int rowCount, int m, byte* xQ8, int q8RowBytes,
        float* dx, int* corr, int n, int blockCount, float* c)
    {
        int rowBytes = blockCount * Q8_0BlockBytes;
        int* rowOff = stackalloc int[8];
        int end = rowStart + rowCount;
        for (int r0 = rowStart; r0 < end; r0 += 8)
        {
            int rows = Math.Min(8, end - r0);
            Q8RowOffsets(rowOff, r0, m, rowBytes);
            ComputeRowsQ8_0MultiVnni_8Rows(weights, rowOff, rows, xQ8, q8RowBytes, dx, corr, n, blockCount, c + r0, m);
        }
    }

    private struct GemmQ8Rows8Ctx
    {
        public byte* Weights;
        public byte* XQ8;
        public float* C;
        public float* Dx;
        public int* Corr;
        public int M;
        public int N;
        public int BlockCount;
        public int Q8RowBytes;
        public long Next;             // next unclaimed row (dynamic scheduling)
    }

    private static void GemmQ8_0Rows8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmQ8Rows8Ctx>((void*)ctxPtr);
        while (true)
        {
            long s = Interlocked.Add(ref ctx.Next, Q8Rows8ClaimRows) - Q8Rows8ClaimRows;
            if (s >= ctx.M) break;
            int start = (int)s, count = Math.Min(ctx.M - start, Q8Rows8ClaimRows);
            GemmQ8_0Rows8Range(ctx.Weights, start, count, ctx.M, ctx.XQ8, ctx.Q8RowBytes, ctx.Dx, ctx.Corr, ctx.N, ctx.BlockCount, ctx.C);
        }
    }

    /// <summary>
    /// Row-major Q8_0 GEMM over pre-quantized Q8_0 activations on the AVX-VNNI eight-row tier: <c>c[t·m + r] = dot(row r, xQ8[t])</c>,
    /// bit-exact with <see cref="VecDotQ8_0Scalar"/> per element. Rows are claimed dynamically in <see cref="Q8Rows8ClaimRows"/>-row
    /// chunks when a pool is given. Caller must have checked <see cref="Q8_0Rows8Tier"/>.
    /// </summary>
    /// <param name="weights">Q8_0 weights, row-major.</param>
    /// <param name="xQ8">Q8_0 activations, <paramref name="n"/> rows of <c>blockCount·34</c> bytes.</param>
    /// <param name="c">Output [n x m].</param>
    /// <param name="m">Weight rows.</param>
    /// <param name="n">Tokens.</param>
    /// <param name="blockCount">Blocks per row.</param>
    /// <param name="pool">Thread pool, or null.</param>
    [SkipLocalsInit]
    internal static void GemmQ8_0Rows8Vnni(byte* weights, byte* xQ8, float* c, int m, int n, int blockCount, ComputeThreadPool? pool)
    {
        if (m <= 0 || n <= 0) return;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int tables = n * blockCount;
        float[] dxBuf = ArrayPool<float>.Shared.Rent(tables);
        int[] corrBuf = ArrayPool<int>.Shared.Rent(tables);
        try
        {
            fixed (float* dx = dxBuf)
            fixed (int* corr = corrBuf)
            {
                Q8_0ActivationTablesVnni(xQ8, q8RowBytes, n, blockCount, dx, corr);
                if (pool is null || m < ParallelMinRows)
                {
                    GemmQ8_0Rows8Range(weights, 0, m, m, xQ8, q8RowBytes, dx, corr, n, blockCount, c);
                    return;
                }
                var ctx = new GemmQ8Rows8Ctx
                {
                    Weights = weights, XQ8 = xQ8, C = c, Dx = dx, Corr = corr,
                    M = m, N = n, BlockCount = blockCount, Q8RowBytes = q8RowBytes,
                };
                pool.Dispatch((nint)(&ctx), &GemmQ8_0Rows8Worker);
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(dxBuf);
            ArrayPool<int>.Shared.Return(corrBuf);
        }
    }
}
