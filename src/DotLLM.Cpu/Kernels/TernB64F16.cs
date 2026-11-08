using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// <see cref="QuantizationType.TERN_B64_F16"/> (issue #1706) format helpers and CPU kernels. Layout: row-major <c>[out, in]</c>, block = 64
/// consecutive input columns of one row = <c>fp16 scale (2 B, LE)</c> + <c>16 B of 2-bit trits</c> (18 B / 64 weights). Trit code
/// <c>00=0, 01=+1, 10=-1, 11=INVALID</c>; byte <c>k</c> of a block holds columns <c>4k..4k+3</c>, column <c>4k+i</c> in bits <c>2i..2i+1</c>.
/// <c>w[row, 64*b + j] = trit * scale[row, b]</c>.
/// </summary>
/// <remarks>
/// <para>The weights are never expanded into a stored wider copy. Two GEMM kernels read the packed blocks directly:
/// <see cref="GemmQ8"/> (production: int8 activations via the house Q8_0 quantizer, one <c>int32</c> dot per 32 activations folded with
/// the 64-column scale) and <see cref="GemmF32Act"/> (scalar F32-activation reference, the oracle for KL work;
/// <c>DOTLLM_TERN_F32ACT=1</c> selects it for a whole run). A row is unpacked into a small per-thread int8 scratch (one row, not a tensor).</para>
/// <para>Invalid code <c>0b11</c> is a load-time error (<see cref="Validate"/>); the kernels decode it as 0 and must only ever see validated data.</para>
/// </remarks>
public static unsafe class TernB64F16
{
    /// <summary>Weights per block.</summary>
    public const int GroupSize = QuantFormat.TernB64GroupSize;

    /// <summary>Bytes per block.</summary>
    public const int BlockBytes = QuantFormat.TernB64F16BlockBytes;

    private const int Q8Block = QuantFormat.LegacyGroupSize;       // 32
    private const int Q8BlockBytes = QuantFormat.Q8_0BlockBytes;   // 34
    private const int ParallelMinRows = 64;

    /// <summary>True when <c>DOTLLM_TERN_F32ACT=1</c>: dense/expert ternary projections run the F32-activation scalar reference kernel (oracle mode).</summary>
    public static bool UseF32ActivationReference { get; internal set; } = Environment.GetEnvironmentVariable("DOTLLM_TERN_F32ACT") == "1";

    /// <summary>byte -> four trits as packed sbytes (code 3 decodes to 0); LUT for the row unpack.</summary>
    private static readonly uint[] s_unpack = BuildUnpackTable();

    private static uint[] BuildUnpackTable()
    {
        var t = new uint[256];
        for (int b = 0; b < 256; b++)
        {
            uint v = 0;
            for (int i = 0; i < 4; i++)
            {
                int code = (b >> (2 * i)) & 3;
                int trit = code == 1 ? 1 : code == 2 ? -1 : 0;
                v |= (uint)(byte)(sbyte)trit << (8 * i);
            }
            t[b] = v;
        }
        return t;
    }

    /// <summary>Bytes of one row of <paramref name="k"/> input columns.</summary>
    public static long RowBytes(int k) => (long)(k / GroupSize) * BlockBytes;

    /// <summary>Total bytes of an <c>[m, k]</c> tensor.</summary>
    public static long TensorBytes(int m, int k) => (long)m * RowBytes(k);

    /// <summary>
    /// Scans packed blocks for the invalid trit code <c>0b11</c>. A byte holds one iff <c>b &amp; (b &gt;&gt; 1) &amp; 0x55 != 0</c>; scale bytes are skipped.
    /// </summary>
    /// <param name="data">Packed tensor bytes (<c>blockCount * 18</c>).</param>
    /// <returns>Index of the first block holding an invalid code, or -1 when the data is clean.</returns>
    public static long FindInvalidBlock(ReadOnlySpan<byte> data)
    {
        long blocks = data.Length / BlockBytes;
        for (long b = 0; b < blocks; b++)
        {
            var codes = data.Slice((int)(b * BlockBytes) + 2, 16);
            for (int i = 0; i < 16; i++)
                if ((codes[i] & (codes[i] >> 1) & 0x55) != 0) return b;
        }
        return -1;
    }

    /// <summary>Pointer overload of <see cref="FindInvalidBlock(ReadOnlySpan{byte})"/> (word-at-a-time; any length).</summary>
    public static long FindInvalidBlock(byte* data, long byteCount) => FindInvalidBlockLong(data, byteCount);

    private static long FindInvalidBlockLong(byte* data, long byteCount)
    {
        long blocks = byteCount / BlockBytes;
        for (long b = 0; b < blocks; b++)
        {
            byte* c = data + b * BlockBytes + 2;
            ulong lo = Unsafe.ReadUnaligned<ulong>(c), hi = Unsafe.ReadUnaligned<ulong>(c + 8);
            const ulong m55 = 0x5555555555555555UL;
            if (((lo & (lo >> 1) & m55) | (hi & (hi >> 1) & m55)) != 0) return b;
        }
        return -1;
    }

    /// <summary>Load-time check: throws <see cref="InvalidDataException"/> naming <paramref name="name"/> and the row/block of the first invalid trit code.</summary>
    public static void Validate(string name, byte* data, int m, int k)
    {
        if (k % GroupSize != 0)
            throw new InvalidDataException($"TERN_B64_F16 tensor '{name}': in-dimension {k} is not a multiple of {GroupSize}.");
        long bad = FindInvalidBlockLong(data, TensorBytes(m, k));
        if (bad >= 0)
        {
            long blocksPerRow = k / GroupSize;
            throw new InvalidDataException(
                $"TERN_B64_F16 tensor '{name}': invalid trit code 0b11 in row {bad / blocksPerRow}, block {bad % blocksPerRow} (codes are 00=0, 01=+1, 10=-1).");
        }
    }

    /// <summary>
    /// Reference dequantization (TEST ORACLE / diagnostics only; never a stored expansion). Throws <see cref="InvalidDataException"/> on code <c>0b11</c>.
    /// </summary>
    public static void Dequantize(byte* src, long elementCount, Span<float> dest)
    {
        if (elementCount % GroupSize != 0)
            throw new ArgumentException($"TERN_B64_F16 element count must be a multiple of {GroupSize}, got {elementCount}", nameof(elementCount));
        long blocks = elementCount / GroupSize;
        for (long b = 0; b < blocks; b++)
        {
            byte* blk = src + b * BlockBytes;
            float scale = (float)Unsafe.ReadUnaligned<Half>(blk);
            for (int j = 0; j < GroupSize; j++)
            {
                int code = (blk[2 + (j >> 2)] >> (2 * (j & 3))) & 3;
                if (code == 3)
                    throw new InvalidDataException($"TERN_B64_F16: invalid trit code 0b11 at block {b}, column {j}.");
                dest[(int)(b * GroupSize + j)] = code == 0 ? 0f : code == 1 ? scale : -scale;
            }
        }
    }

    /// <summary>
    /// Packs trits and per-block scales into the on-disk layout (used by test writers and exporters). Scales are rounded to FP16.
    /// </summary>
    /// <param name="trits">Row-major <c>[m, k]</c> trits in {-1,0,1} (<c>m*k</c> entries, <c>k % 64 == 0</c>).</param>
    /// <param name="scales">Per-block scales, <c>m * (k/64)</c> entries (row-major).</param>
    /// <returns>The packed bytes (<c>m * k/64 * 18</c>).</returns>
    public static byte[] Pack(ReadOnlySpan<sbyte> trits, ReadOnlySpan<float> scales)
    {
        if (trits.Length % GroupSize != 0) throw new ArgumentException("trit count must be a multiple of 64", nameof(trits));
        int blocks = trits.Length / GroupSize;
        if (scales.Length != blocks) throw new ArgumentException($"expected {blocks} scales, got {scales.Length}", nameof(scales));
        var dst = new byte[blocks * BlockBytes];
        for (int b = 0; b < blocks; b++)
        {
            Unsafe.WriteUnaligned(ref dst[b * BlockBytes], (Half)scales[b]);
            for (int j = 0; j < GroupSize; j++)
            {
                sbyte t = trits[b * GroupSize + j];
                int code = t switch { 0 => 0, 1 => 1, -1 => 2, _ => throw new ArgumentException($"trit {t} out of range", nameof(trits)) };
                dst[b * BlockBytes + 2 + (j >> 2)] |= (byte)(code << (2 * (j & 3)));
            }
        }
        return dst;
    }

    // ────────────────────────── F32-activation scalar reference ──────────────────────────

    /// <summary>
    /// Reference GEMM with F32 activations: <c>c[t*m + r] = sum_j W[r,j] * b[t*k + j]</c>. Scalar; per-block float partial sums folded into a double.
    /// Row-parallel over <paramref name="pool"/>.
    /// </summary>
    public static void GemmF32Act(byte* w, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        CheckK(k);
        var ctx = new F32Ctx { W = w, B = b, C = c, M = m, K = k, N = n };
        RunRows(pool, m, (nint)(&ctx), &F32Worker);
    }

    private struct F32Ctx { public byte* W; public float* B; public float* C; public int M, K, N; }

    private static void F32Worker(nint p, int ti, int tc)
    {
        ref var ctx = ref Unsafe.AsRef<F32Ctx>((void*)p);
        Partition(ctx.M, ti, tc, out int start, out int count);
        long rowBytes = RowBytes(ctx.K);
        int blocks = ctx.K / GroupSize;
        for (int r = start; r < start + count; r++)
        {
            byte* row = ctx.W + r * rowBytes;
            for (int t = 0; t < ctx.N; t++)
            {
                float* x = ctx.B + (long)t * ctx.K;
                double acc = 0;
                for (int bi = 0; bi < blocks; bi++)
                {
                    byte* blk = row + bi * BlockBytes;
                    float scale = (float)Unsafe.ReadUnaligned<Half>(blk);
                    float part = 0f;
                    for (int j = 0; j < GroupSize; j++)
                    {
                        int code = (blk[2 + (j >> 2)] >> (2 * (j & 3))) & 3;
                        float xv = x[bi * GroupSize + j];
                        if (code == 1) part += xv; else if (code == 2) part -= xv;
                    }
                    acc += (double)scale * part;
                }
                ctx.C[(long)t * ctx.M + r] = (float)acc;
            }
        }
    }

    // ────────────────────────── int8-activation (Q8_0) kernel ──────────────────────────

    /// <summary>
    /// Production GEMM: quantizes each of the <paramref name="n"/> activation rows with the house Q8_0 quantizer (per-32 absmax int8, fp16 scale),
    /// then each weight row is unpacked once to int8 trits and dotted against every token: per 64-column block
    /// <c>acc += scale * (dot32(t[0..32], q0) * s0 + dot32(t[32..64], q1) * s1)</c>. <c>c[t*m + r]</c>.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmQ8(byte* w, float* b, float* c, int m, int k, int n, ComputeThreadPool? pool)
    {
        CheckK(k);
        if (UseF32ActivationReference) { GemmF32Act(w, b, c, m, k, n, pool); return; }
        int q8Row = k / Q8Block * Q8BlockBytes;
        byte[] q = ArrayPool<byte>.Shared.Rent(checked(n * q8Row));
        float[] sc = ArrayPool<float>.Shared.Rent(checked(n * (k / Q8Block)));
        try
        {
            fixed (byte* qp = q)
            fixed (float* sp = sc)
            {
                for (int t = 0; t < n; t++)
                {
                    MatMul.QuantizeF32ToQ8_0(b + (long)t * k, qp + (long)t * q8Row, k);
                    byte* qb = qp + (long)t * q8Row;
                    float* ss = sp + (long)t * (k / Q8Block);
                    for (int i = 0; i < k / Q8Block; i++) ss[i] = (float)Unsafe.ReadUnaligned<Half>(qb + i * Q8BlockBytes);
                }
                var ctx = new Q8Ctx { W = w, Q = qp, S = sp, C = c, M = m, K = k, N = n, Q8Row = q8Row };
                RunRows(pool, m, (nint)(&ctx), &Q8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(q);
            ArrayPool<float>.Shared.Return(sc);
        }
    }

    /// <summary>GEMV (<c>n == 1</c>) convenience over <see cref="GemmQ8"/>.</summary>
    public static void GemvQ8(byte* w, float* x, float* y, int m, int k, ComputeThreadPool? pool) => GemmQ8(w, x, y, m, k, 1, pool);

    private struct Q8Ctx { public byte* W; public byte* Q; public float* S; public float* C; public int M, K, N, Q8Row; }

    [SkipLocalsInit]
    private static void Q8Worker(nint p, int ti, int tc)
    {
        ref var ctx = ref Unsafe.AsRef<Q8Ctx>((void*)p);
        Partition(ctx.M, ti, tc, out int start, out int count);
        if (count == 0) return;
        int k = ctx.K, blocks = k / GroupSize;
        long rowBytes = RowBytes(k);
        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        float[] scaleBuf = ArrayPool<float>.Shared.Rent(blocks);
        try
        {
            fixed (sbyte* wi = rowBuf)
            fixed (float* ws = scaleBuf)
            {
                for (int r = start; r < start + count; r++)
                {
                    UnpackRow(ctx.W + r * rowBytes, wi, ws, blocks);
                    for (int t = 0; t < ctx.N; t++)
                        ctx.C[(long)t * ctx.M + r] = DotRow(wi, ws, ctx.Q + (long)t * ctx.Q8Row, ctx.S + (long)t * (k / Q8Block), blocks);
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
            ArrayPool<float>.Shared.Return(scaleBuf);
        }
    }

    /// <summary>Unpacks one packed row into int8 trits (<c>blocks*64</c>) and FP32 block scales.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void UnpackRow(byte* row, sbyte* trits, float* scales, int blocks)
    {
        fixed (uint* lut = s_unpack)
        {
            for (int b = 0; b < blocks; b++)
            {
                byte* blk = row + b * BlockBytes;
                scales[b] = (float)Unsafe.ReadUnaligned<Half>(blk);
                uint* o = (uint*)(trits + b * GroupSize);
                for (int i = 0; i < 16; i++) Unsafe.WriteUnaligned(o + i, lut[blk[2 + i]]);
            }
        }
    }

    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float DotRow(sbyte* wi, float* ws, byte* xq, float* xs, int blocks)
    {
        float acc = 0f;
        for (int b = 0; b < blocks; b++)
        {
            byte* q0 = xq + (2 * b) * Q8BlockBytes + 2;
            byte* q1 = xq + (2 * b + 1) * Q8BlockBytes + 2;
            int s0 = Dot32(wi + b * GroupSize, (sbyte*)q0);
            int s1 = Dot32(wi + b * GroupSize + 32, (sbyte*)q1);
            acc += ws[b] * (s0 * xs[2 * b] + s1 * xs[2 * b + 1]);
        }
        return acc;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int Dot32(sbyte* a, sbyte* b)
    {
        if (CpuFeatures.Avx2)
        {
            var a0 = Avx2.ConvertToVector256Int16(a);
            var a1 = Avx2.ConvertToVector256Int16(a + 16);
            var b0 = Avx2.ConvertToVector256Int16(b);
            var b1 = Avx2.ConvertToVector256Int16(b + 16);
            var s = Avx2.Add(Avx2.MultiplyAddAdjacent(a0, b0), Avx2.MultiplyAddAdjacent(a1, b1));
            return Vector256.Sum(s);
        }
        int acc = 0;
        for (int i = 0; i < 32; i++) acc += a[i] * b[i];
        return acc;
    }

    // ────────────────────────── helpers ──────────────────────────

    private static void CheckK(int k)
    {
        if (k % GroupSize != 0)
            throw new ArgumentException($"TERN_B64_F16 needs in % {GroupSize} == 0, got {k}", nameof(k));
    }

    private static void Partition(int m, int ti, int tc, out int start, out int count)
    {
        int per = (m + tc - 1) / tc;
        start = Math.Min(m, ti * per);
        count = Math.Max(0, Math.Min(per, m - start));
    }

    private static void RunRows(ComputeThreadPool? pool, int m, nint ctx, delegate*<nint, int, int, void> fn)
    {
        if (pool is null || m < ParallelMinRows) fn(ctx, 0, 1);
        else pool.Dispatch(ctx, fn);
    }
}
