using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Row-lane Q4_K GEMM (#1927 E3d; default on, James 2026-11-22; <see cref="Q4KRowLaneEnabled"/>, n >= 16, m % 8 == 0): Q4_K x Q8_K GEMM with SIMD lanes over 8 OUTPUT ROWS,
/// read in place from the native row-major (mmap'd) Q4_K layout. Each 8-row block's super-blocks are transposed once into a
/// per-worker tile of sbCount x 2,752 B (44 KiB at k = 4096, 104 KiB at k = 9728, 150 KiB at k = 14336; about 2.4x the 8-row slice
/// it transposes; lane r = row r's 4 consecutive nibbles), then reused by every token: activations are broadcast
/// 4 bytes at a time, <c>vpmaddubsw</c> results of the 8 four-position chunks of one sub-block are summed in int16 before ONE
/// scale multiply (<c>vpdpwssd</c>/<c>vpmaddwd</c>) per sub-block, as llama.cpp's q4_Kx8 kernel does. No weight copy is stored.
/// <para>
/// NOT bit-identical to <see cref="MatMul.GemmR4TiledQ4_K"/>: the integer super-block sum is formed exactly and converted to float
/// once per (row, super-block), where the current kernel keeps eight K-direction int lanes and converts each. int16 headroom: a
/// <c>vpmaddubsw</c> lane is at most 2 x 15 x 128 = 3,840, eight of them 30,720 &lt; 32,767.
/// </para>
/// </summary>
public static unsafe partial class MatMul
{
    /// <summary>True when the row-lane kernel can run (AVX2 + SSE4.1 + FMA; AVX-VNNI used when present).</summary>
    internal static bool Q4KRowLaneIsa => CpuFeatures.Avx2 && CpuFeatures.Sse41 && CpuFeatures.Fma;

    /// <summary>
    /// Runtime switch for the Q4_K row-lane GEMM: default ON (James 2026-11-22), <c>DOTLLM_CPU_Q4K_ROWLANE=0</c> off. Settable for
    /// in-process A/B and quality gates. Not bit-identical to the R4 kernels (closer to float64); see the class remarks.
    /// </summary>
    public static bool Q4KRowLaneEnabled { get; set; } = Environment.GetEnvironmentVariable("DOTLLM_CPU_Q4K_ROWLANE") != "0";

    /// <summary>Fewest tokens for which a Q4_K GEMM takes the row-lane kernel (a speed choice only: below it the tile transpose does not
    /// pay). Shorter batches run the R4 kernels in the SAME row-lane arithmetic (<see cref="Q4KRlaActive"/>), so a row's bits do not
    /// depend on the batch size or composition (#530, #1927).</summary>
    internal const int Q4KRowLaneMinTokens = 16;

    /// <summary>Row-lane GEMMs issued from the CURRENT thread (tests: immune to concurrent GEMMs on other threads).</summary>
    [ThreadStatic] internal static long t_q4kRowLaneCalls;

    /// <summary>Number of Q4_K GEMMs served by the row-lane kernel (test and profiling hook).</summary>
    public static long Q4KRowLaneCalls;

    internal static bool Q4KRowLaneApplies(int m, int tailRows, int n)
        => Q4KRlaActive && n >= Q4KRowLaneMinTokens && tailRows == 0 && m % 8 == 0;

    /// <summary>Production entry: quantizes the F32 input when no Q8_K input is given, then runs <see cref="GemmQ4KRowLane8"/>.</summary>
    internal static void GemmQ4KRowLaneEntry(byte* w, float* b, byte* inputQ, float* c, int sbCount, int m, int k, int n, ComputeThreadPool? pool)
    {
        Interlocked.Increment(ref Q4KRowLaneCalls);
        t_q4kRowLaneCalls++;
        if (inputQ != null) { GemmQ4KRowLane8(w, inputQ, c, m, sbCount, n, pool); return; }
        int rowBytes = sbCount * Q8_K_BlockBytes;
        byte[] rented = System.Buffers.ArrayPool<byte>.Shared.Rent(n * rowBytes);
        try
        {
            fixed (byte* q = rented)
            {
                for (int t = 0; t < n; t++) QuantizeF32ToQ8_K(b + (long)t * k, q + (long)t * rowBytes, k);
                GemmQ4KRowLane8(w, q, c, m, sbCount, n, pool);
            }
        }
        finally { System.Buffers.ArrayPool<byte>.Shared.Return(rented); }
    }

    // Per 8-row block and super-block: lo/hi nibble chunks [sub 0..7][L 0..7] (64 vectors), scale pairs [8], min pairs [4], d, dmin, scales [8].
    private const int RowLaneSbVectors = 64 + 8 + 4 + 2 + 8;   // + plain int32 scales [8] for the VNNI vpdpbusd path

    private struct RowLaneCtx
    {
        public byte* W;
        public byte* X;
        public float* C;
        public int M, N, SbCount, XRowBytes, Next;
    }

    /// <summary>C[t * m + row] = W[row] . X[t] for rows [0, m) (m % 8 == 0), tokens [0, n). X is Q8_K rows of <paramref name="sbCount"/> blocks.</summary>
    internal static void GemmQ4KRowLane8(byte* w, byte* x, float* c, int m, int sbCount, int n, ComputeThreadPool? pool)
    {
        if (m % 8 != 0) throw new ArgumentException("the row-lane kernel needs m % 8 == 0 (callers fall back to the R4 kernels in the same arithmetic)", nameof(m));
        var ctx = new RowLaneCtx { W = w, X = x, C = c, M = m, N = n, SbCount = sbCount, XRowBytes = sbCount * Q8_K_BlockBytes, Next = 0 };
        if (pool is null) RowLaneWorker((nint)(&ctx), 0, 1);
        else pool.Dispatch((nint)(&ctx), &RowLaneWorker);
    }

    /// <summary>
    /// Per-thread transposed row-lane tile (native, 64-byte aligned, grown on demand to the largest k seen). Finalizable owner, so the
    /// memory is released when the thread exits; <see cref="CpuKernelScratch.Trim(ComputeThreadPool?)"/> frees it deterministically
    /// (model dispose and pool teardown call it). Every thread that drives a Q4_K GEMM of 16+ tokens gets one: pool workers and the
    /// dispatching thread (worker 0).
    /// </summary>
    private sealed class RowLaneTile
    {
        public nint Ptr;
        public int SbCount;

        public void Ensure(int sbCount)
        {
            if (SbCount >= sbCount) return;
            Free();
            Ptr = (nint)NativeMemory.AlignedAlloc((nuint)(sbCount * RowLaneSbVectors * 32), 64);
            SbCount = sbCount;
        }

        public void Free()
        {
            if (Ptr != 0) NativeMemory.AlignedFree((void*)Ptr);
            Ptr = 0;
            SbCount = 0;
        }

        ~RowLaneTile() => Free();
    }

    [ThreadStatic] private static RowLaneTile? t_rowLaneTile;

    /// <summary>Frees the calling thread's row-lane tile (idempotent).</summary>
    internal static void FreeRowLaneTile() => t_rowLaneTile?.Free();

    /// <summary>Bytes of row-lane tile the calling thread holds (test hook).</summary>
    internal static long RowLaneTileBytesOnThisThread => t_rowLaneTile is { Ptr: not 0 } t ? (long)t.SbCount * RowLaneSbVectors * 32 : 0;

    private static void RowLaneWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<RowLaneCtx>((void*)ctxPtr);
        RowLaneTile holder = t_rowLaneTile ??= new RowLaneTile();
        holder.Ensure(ctx.SbCount);
        var tile = (Vector256<int>*)holder.Ptr;
        int blocks = ctx.M / 8;
        while (true)
        {
            int b = Interlocked.Increment(ref ctx.Next) - 1;
            if (b >= blocks) break;
            byte* rows = ctx.W + (long)b * 8 * ctx.SbCount * Q4_K_BlockBytes;
            BuildRowLaneTiles(rows, ctx.SbCount, tile);
            int t = 0;
            if (RowLaneTwoTokens)
                for (; t + 2 <= ctx.N; t += 2)
                    RowLaneToken2(tile, ctx.SbCount, ctx.X + (long)t * ctx.XRowBytes, ctx.XRowBytes, ctx.C + (long)t * ctx.M + b * 8, ctx.M);
            for (; t < ctx.N; t++)
                RowLaneToken(tile, ctx.SbCount, ctx.X + (long)t * ctx.XRowBytes, ctx.C + (long)t * ctx.M + b * 8);
        }
    }

    [SkipLocalsInit]
    private static void BuildRowLaneTiles(byte* rows, int sbCount, Vector256<int>* tile)
    {
        nint rowStride = (nint)sbCount * Q4_K_BlockBytes;
        Vector256<int> m0F = Vector256.Create(0x0F0F0F0F);
        short* sc = stackalloc short[8 * 8];
        short* mn = stackalloc short[8 * 8];
        float* d = stackalloc float[8];
        float* dm = stackalloc float[8];
        int* tmp = stackalloc int[8];
        for (int sb = 0; sb < sbCount; sb++)
        {
            Vector256<int>* tl = tile + sb * RowLaneSbVectors;
            for (int r = 0; r < 8; r++)
            {
                byte* blk = rows + r * rowStride + (long)sb * Q4_K_BlockBytes;
                d[r] = HalfToSingle(blk);
                dm[r] = HalfToSingle(blk + 2);
                UnpackQ4KScalesFast(blk + 4, out Vector128<short> sc16, out Vector256<short> mnDup);
                Sse2.Store(sc + r * 8, sc16);
                // mnDup = [m0 m0 m1 m1 ... m7 m7]: take every other element.
                for (int s = 0; s < 8; s++) mn[r * 8 + s] = mnDup.GetElement(2 * s);
            }
            for (int j = 0; j < 4; j++)
                for (int l = 0; l < 8; l++)
                {
                    for (int r = 0; r < 8; r++) tmp[r] = Unsafe.ReadUnaligned<int>(rows + r * rowStride + (long)sb * Q4_K_BlockBytes + 16 + j * 32 + 4 * l);
                    Vector256<int> raw = Avx.LoadVector256(tmp);
                    tl[(2 * j) * 8 + l] = Avx2.And(raw, m0F);                                   // sub 2j: low nibbles
                    tl[(2 * j + 1) * 8 + l] = Avx2.And(Avx2.ShiftRightLogical(raw, 4), m0F);    // sub 2j+1: high nibbles
                }
            for (int s = 0; s < 8; s++)
            {
                for (int r = 0; r < 8; r++) tmp[r] = (ushort)sc[r * 8 + s] | (sc[r * 8 + s] << 16);
                tl[64 + s] = Avx.LoadVector256(tmp);
                for (int r = 0; r < 8; r++) tmp[r] = sc[r * 8 + s];
                tl[78 + s] = Avx.LoadVector256(tmp);
            }
            for (int k = 0; k < 4; k++)
            {
                for (int r = 0; r < 8; r++) tmp[r] = (ushort)mn[r * 8 + 2 * k] | (mn[r * 8 + 2 * k + 1] << 16);
                tl[72 + k] = Avx.LoadVector256(tmp);
            }
            tl[76] = Avx.LoadVector256(d).AsInt32();
            tl[77] = Avx.LoadVector256(dm).AsInt32();
        }
    }

    /// <summary>Two tokens per tile pass (shared tile loads, two independent chains). Same per-token arithmetic.</summary>
    internal static readonly bool RowLaneTwoTokens = true;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> RowLaneSub(Vector256<int>* wl, byte* q, Vector256<int> isum, Vector256<int> scPlain)
    {
        Vector256<int> a32 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, wl[0].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q)).AsSByte());
        a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[1].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 4)).AsSByte());
        a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[2].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 8)).AsSByte());
        a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[3].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 12)).AsSByte());
        a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[4].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 16)).AsSByte());
        a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[5].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 20)).AsSByte());
        a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[6].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 24)).AsSByte());
        a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[7].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 28)).AsSByte());
        return Avx2.Add(isum, Avx2.MultiplyLow(a32, scPlain));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> RowLaneMins(Vector256<int>* tl, byte* q8k)
    {
        short* bs = (short*)(q8k + 260);
        Vector256<int> mc = Avx2.MultiplyAddAdjacent(tl[72].AsInt16(), Vector256.Create((ushort)(short)(bs[0] + bs[1]) | ((bs[2] + bs[3]) << 16)).AsInt16());
        mc = Avx2.Add(mc, Avx2.MultiplyAddAdjacent(tl[73].AsInt16(), Vector256.Create((ushort)(short)(bs[4] + bs[5]) | ((bs[6] + bs[7]) << 16)).AsInt16()));
        mc = Avx2.Add(mc, Avx2.MultiplyAddAdjacent(tl[74].AsInt16(), Vector256.Create((ushort)(short)(bs[8] + bs[9]) | ((bs[10] + bs[11]) << 16)).AsInt16()));
        return Avx2.Add(mc, Avx2.MultiplyAddAdjacent(tl[75].AsInt16(), Vector256.Create((ushort)(short)(bs[12] + bs[13]) | ((bs[14] + bs[15]) << 16)).AsInt16()));
    }

    private static void RowLaneToken2(Vector256<int>* tile, int sbCount, byte* q8a, int xRowBytes, float* outRow, int ldc)
    {
        if (!CpuFeatures.AvxVnni)
        {
            RowLaneToken(tile, sbCount, q8a, outRow);
            RowLaneToken(tile, sbCount, q8a + xRowBytes, outRow + ldc);
            return;
        }
        byte* q8b = q8a + xRowBytes;
        Vector256<float> accA = Vector256<float>.Zero, smA = accA, accB = accA, smB = accA;
        for (int sb = 0; sb < sbCount; sb++, q8a += Q8_K_BlockBytes, q8b += Q8_K_BlockBytes)
        {
            Vector256<int>* tl = tile + sb * RowLaneSbVectors;
            Vector256<int> iA = Vector256<int>.Zero, iB = Vector256<int>.Zero;
            for (int s = 0; s < 8; s++)
            {
                int off = 4 + (s >> 1) * 64 + (s & 1) * 32;
                Vector256<int>* wl = tl + s * 8;
                Vector256<int> sc = tl[78 + s];
                iA = RowLaneSub(wl, q8a + off, iA, sc);
                iB = RowLaneSub(wl, q8b + off, iB, sc);
            }
            Vector256<float> d4 = tl[76].AsSingle(), dm = tl[77].AsSingle();
            Vector256<float> vdA = Vector256.Create(Unsafe.ReadUnaligned<float>(q8a)), vdB = Vector256.Create(Unsafe.ReadUnaligned<float>(q8b));
            accA = Fma.MultiplyAdd(Avx.Multiply(d4, vdA), Avx.ConvertToVector256Single(iA), accA);
            accB = Fma.MultiplyAdd(Avx.Multiply(d4, vdB), Avx.ConvertToVector256Single(iB), accB);
            smA = Avx.Add(smA, Avx.Multiply(Avx.Multiply(dm, vdA), Avx.ConvertToVector256Single(RowLaneMins(tl, q8a))));
            smB = Avx.Add(smB, Avx.Multiply(Avx.Multiply(dm, vdB), Avx.ConvertToVector256Single(RowLaneMins(tl, q8b))));
        }
        Avx.Store(outRow, Avx.Subtract(accA, smA));
        Avx.Store(outRow + ldc, Avx.Subtract(accB, smB));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void RowLaneToken(Vector256<int>* tile, int sbCount, byte* q8k, float* outRow)
    {
        Vector256<float> acc = Vector256<float>.Zero, sm = Vector256<float>.Zero;
        for (int sb = 0; sb < sbCount; sb++, q8k += Q8_K_BlockBytes)
        {
            Vector256<int>* tl = tile + sb * RowLaneSbVectors;
            float d8 = Unsafe.ReadUnaligned<float>(q8k);
            byte* qs = q8k + 4;
            Vector256<int> isum = Vector256<int>.Zero;
            for (int s = 0; s < 8; s++)
            {
                byte* q = qs + (s >> 1) * 64 + (s & 1) * 32;
                Vector256<int>* wl = tl + s * 8;
                if (CpuFeatures.AvxVnni)
                {
                    // vpdpbusd: 4 u8 x s8 products straight into the row's int32 lane; one exact int32 scale multiply per sub-block.
                    Vector256<int> a32 = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, wl[0].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q)).AsSByte());
                    a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[1].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 4)).AsSByte());
                    a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[2].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 8)).AsSByte());
                    a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[3].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 12)).AsSByte());
                    a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[4].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 16)).AsSByte());
                    a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[5].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 20)).AsSByte());
                    a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[6].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 24)).AsSByte());
                    a32 = AvxVnni.MultiplyWideningAndAdd(a32, wl[7].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 28)).AsSByte());
                    isum = Avx2.Add(isum, Avx2.MultiplyLow(a32, tl[78 + s]));
                    continue;
                }
                Vector256<short> a16 = Avx2.MultiplyAddAdjacent(wl[0].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q)).AsSByte());
                a16 = Avx2.Add(a16, Avx2.MultiplyAddAdjacent(wl[1].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 4)).AsSByte()));
                a16 = Avx2.Add(a16, Avx2.MultiplyAddAdjacent(wl[2].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 8)).AsSByte()));
                a16 = Avx2.Add(a16, Avx2.MultiplyAddAdjacent(wl[3].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 12)).AsSByte()));
                a16 = Avx2.Add(a16, Avx2.MultiplyAddAdjacent(wl[4].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 16)).AsSByte()));
                a16 = Avx2.Add(a16, Avx2.MultiplyAddAdjacent(wl[5].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 20)).AsSByte()));
                a16 = Avx2.Add(a16, Avx2.MultiplyAddAdjacent(wl[6].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 24)).AsSByte()));
                a16 = Avx2.Add(a16, Avx2.MultiplyAddAdjacent(wl[7].AsByte(), Vector256.Create(Unsafe.ReadUnaligned<int>(q + 28)).AsSByte()));
                isum = Avx2.Add(isum, Avx2.MultiplyAddAdjacent(a16, tl[64 + s].AsInt16()));
            }
            Vector256<float> vd8 = Vector256.Create(d8);
            acc = Fma.MultiplyAdd(Avx.Multiply(tl[76].AsSingle(), vd8), Avx.ConvertToVector256Single(isum), acc);

            // Min correction: mc[row] = sum_s min_s[row] * (bsums[2s] + bsums[2s+1]), exact in int32.
            short* bs = (short*)(q8k + 260);
            Vector256<int> mc = Avx2.MultiplyAddAdjacent(tl[72].AsInt16(), Vector256.Create((ushort)(short)(bs[0] + bs[1]) | ((bs[2] + bs[3]) << 16)).AsInt16());
            mc = Avx2.Add(mc, Avx2.MultiplyAddAdjacent(tl[73].AsInt16(), Vector256.Create((ushort)(short)(bs[4] + bs[5]) | ((bs[6] + bs[7]) << 16)).AsInt16()));
            mc = Avx2.Add(mc, Avx2.MultiplyAddAdjacent(tl[74].AsInt16(), Vector256.Create((ushort)(short)(bs[8] + bs[9]) | ((bs[10] + bs[11]) << 16)).AsInt16()));
            mc = Avx2.Add(mc, Avx2.MultiplyAddAdjacent(tl[75].AsInt16(), Vector256.Create((ushort)(short)(bs[12] + bs[13]) | ((bs[14] + bs[15]) << 16)).AsInt16()));
            sm = Avx.Add(sm, Avx.Multiply(Avx.Multiply(tl[77].AsSingle(), vd8), Avx.ConvertToVector256Single(mc)));
        }
        Avx.Store(outRow, Avx.Subtract(acc, sm));
    }
}
