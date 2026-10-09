using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels.Experimental;

// EXPERIMENTAL (issue #832): LittleBit-style factorized linear layer, implemented from the paper's
// equations only (arXiv 2506.13771). NOT wired into model loading. No third-party code is used.
//
//   W ~= diag(h) . Us . diag(l) . Vs^T . diag(g)         (per path; primary + optional residual path)
//   y  = sum_paths  h .* ( Us ( l .* ( Vs^T ( g .* x ) ) ) )
//   Us in {+-1}^{d_out x r}, Vs in {+-1}^{d_in x r}; h (d_out), g (d_in), l (r) are stored F32 (exact widening of fp16/bf16 sources; #864).
//   bits per path = 2r(d_out + d_in + 1) + 16(d_out + d_in + r).
//
// Bit packing (all sign factors): one bit per weight, LSB-first within each byte (element i of a row
// lives in byte i/8, bit i%8), bit value 1 == -1, bit value 0 == +1. Chosen so the bit can be
// expanded straight into a float sign-bit mask.
//   * Us is stored row-major per OUTPUT row: row o holds the r signs Us[o, 0..r), row stride RPad/8
//     bytes, RPad = r rounded up to a multiple of 32.
//   * Vs is stored TRANSPOSED, one bit-row per LATENT unit: row j holds Vs[0..d_in, j], row stride
//     DInPad/8 bytes, DInPad = d_in rounded up to a multiple of 32. Stage 1 is then a contiguous
//     signed-sum of the (scaled) activation vector against a bit-row.
//   * Padding bits are 0; the matching activation / latent entries are 0, so they never contribute.

/// <summary>Which signed-sum kernel <see cref="LittleBitLayer.Gemv"/> runs.</summary>
public enum LittleBitKernel
{
    /// <summary>AVX2: FP32 activations, bits -> sign-bit mask via a 256-entry table, xor, add.</summary>
    Avx2Float,
    /// <summary>AVX-VNNI: int8-quantized activations (extra activation-quantization error), bits -> 0/0xFF bytes, vpdpbusd.</summary>
    VnniInt8,
}

/// <summary>One binary-factor path (primary or residual) of a LittleBit layer. Owns native memory.</summary>
public sealed unsafe class LittleBitPath : IDisposable
{
    /// <summary>Output features.</summary>
    public int DOut { get; }
    /// <summary>Input features.</summary>
    public int DIn { get; }
    /// <summary>Latent rank.</summary>
    public int R { get; }
    /// <summary><see cref="DIn"/> rounded up to a multiple of 32.</summary>
    public int DInPad { get; }
    /// <summary><see cref="R"/> rounded up to a multiple of 32.</summary>
    public int RPad { get; }

    internal byte* UBits;   // [DOut][RPad/8]
    internal byte* VBits;   // [R][DInPad/8]
    internal float* H;      // [DOut]  (F32: holds fp16 or bf16 source scales exactly)
    internal float* G;      // [DIn]
    internal float* L;      // [R]

    /// <summary>Bytes the kernel reads per GEMV for this path (packed bits + F32 scales).</summary>
    public long WeightBytes => (long)DOut * (RPad / 8) + (long)R * (DInPad / 8) + 4L * (DOut + DIn + R);

    /// <summary>Paper's bit count for this path: 2r(d_out+d_in+1) + 16(d_out+d_in+r).</summary>
    public long PaperBits => 2L * R * (DOut + DIn + 1) + 16L * (DOut + DIn + R);

    private LittleBitPath(int dOut, int dIn, int r)
    {
        if (dOut <= 0 || dIn <= 0 || r <= 0) throw new ArgumentOutOfRangeException();
        // The int8 kernel accumulates 255*127*n in an int32.
        if (dIn > 60000 || r > 60000) throw new ArgumentOutOfRangeException(nameof(dIn), "int8 path bound");
        DOut = dOut; DIn = dIn; R = r;
        DInPad = (dIn + 31) / 32 * 32;
        RPad = (r + 31) / 32 * 32;
        UBits = Alloc((long)dOut * (RPad / 8));
        VBits = Alloc((long)r * (DInPad / 8));
        H = (float*)Alloc(4L * dOut);
        G = (float*)Alloc(4L * dIn);
        L = (float*)Alloc(4L * r);
    }

    private static byte* Alloc(long bytes)
    {
        byte* p = (byte*)NativeMemory.AlignedAlloc((nuint)Math.Max(bytes, 64), 64);
        NativeMemory.Clear(p, (nuint)Math.Max(bytes, 64));
        return p;
    }

    /// <summary>
    /// Builds a path from +-1 sign matrices. <paramref name="us"/> is [dOut x r] row-major,
    /// <paramref name="vs"/> is [dIn x r] row-major (paper orientation: Vs in {+-1}^{d_in x r}).
    /// Any value &gt; 0 packs as +1 (bit 0), any value &lt;= 0 as -1 (bit 1).
    /// </summary>
    public static LittleBitPath FromSigns(int dOut, int dIn, int r,
        ReadOnlySpan<sbyte> us, ReadOnlySpan<sbyte> vs,
        ReadOnlySpan<Half> h, ReadOnlySpan<Half> g, ReadOnlySpan<Half> l)
    {
        var p = FromSigns(dOut, dIn, r, us, vs, new float[dOut], new float[dIn], new float[r]);
        TensorPrimitives.ConvertToSingle(h, new Span<float>(p.H, dOut));
        TensorPrimitives.ConvertToSingle(g, new Span<float>(p.G, dIn));
        TensorPrimitives.ConvertToSingle(l, new Span<float>(p.L, r));
        return p;
    }

    /// <summary>As the <see cref="Half"/> overload with F32 scales (exact for fp16 and bf16 sources).</summary>
    public static LittleBitPath FromSigns(int dOut, int dIn, int r,
        ReadOnlySpan<sbyte> us, ReadOnlySpan<sbyte> vs,
        ReadOnlySpan<float> h, ReadOnlySpan<float> g, ReadOnlySpan<float> l)
    {
        if (us.Length != dOut * r || vs.Length != dIn * r || h.Length != dOut || g.Length != dIn || l.Length != r)
            throw new ArgumentException("factor shape mismatch");
        var p = new LittleBitPath(dOut, dIn, r);
        int uStride = p.RPad / 8, vStride = p.DInPad / 8;
        for (int o = 0; o < dOut; o++)
            for (int j = 0; j < r; j++)
                if (us[o * r + j] <= 0) p.UBits[o * uStride + (j >> 3)] |= (byte)(1 << (j & 7));
        for (int i = 0; i < dIn; i++)
            for (int j = 0; j < r; j++)
                if (vs[i * r + j] <= 0) p.VBits[(long)j * vStride + (i >> 3)] |= (byte)(1 << (i & 7));
        h.CopyTo(new Span<float>(p.H, dOut));
        g.CopyTo(new Span<float>(p.G, dIn));
        l.CopyTo(new Span<float>(p.L, r));
        return p;
    }

    /// <summary>
    /// Builds a path directly from the checkpoint's packed sign words (int32, LSB-first, bit 1 = -1, padded with +1):
    /// <paramref name="uWords"/> is [dOut x ceil(r/32)] and <paramref name="vWords"/> is [r x ceil(dIn/32)], exactly the
    /// layout of this class (little-endian words == LSB-first bytes), so this is a row-wise copy. Padding bits are
    /// cleared (they are +1 by the format). Scales are F32; the caller decodes bf16 and pre-multiplies l = v1*u2.
    /// </summary>
    public static LittleBitPath FromPackedWords(int dOut, int dIn, int r,
        ReadOnlySpan<int> uWords, ReadOnlySpan<int> vWords,
        ReadOnlySpan<float> h, ReadOnlySpan<float> g, ReadOnlySpan<float> l)
    {
        var p = new LittleBitPath(dOut, dIn, r);
        int uw = p.RPad / 32, vw = p.DInPad / 32;
        if (uWords.Length != dOut * uw || vWords.Length != r * vw || h.Length != dOut || g.Length != dIn || l.Length != r)
            throw new ArgumentException("packed factor shape mismatch");
        MemoryMarshal.AsBytes(uWords).CopyTo(new Span<byte>(p.UBits, uWords.Length * 4));
        MemoryMarshal.AsBytes(vWords).CopyTo(new Span<byte>(p.VBits, vWords.Length * 4));
        for (int o = 0; o < dOut; o++) ClearBitsFrom(p.UBits + (long)o * (p.RPad / 8), r, p.RPad);
        for (int j = 0; j < r; j++) ClearBitsFrom(p.VBits + (long)j * (p.DInPad / 8), dIn, p.DInPad);
        h.CopyTo(new Span<float>(p.H, dOut));
        g.CopyTo(new Span<float>(p.G, dIn));
        l.CopyTo(new Span<float>(p.L, r));
        return p;
    }

    /// <summary>Random factors (benchmark / test). Scales are positive and small so activations stay O(1).</summary>
    public static LittleBitPath Random(int dOut, int dIn, int r, Random rng)
    {
        var p = new LittleBitPath(dOut, dIn, r);
        FillRandomBytes(p.UBits, (long)dOut * (p.RPad / 8), rng);
        FillRandomBytes(p.VBits, (long)r * (p.DInPad / 8), rng);
        // keep padding bits 0 so the logical content is well defined
        for (int o = 0; o < dOut; o++)
            ClearBitsFrom(p.UBits + (long)o * (p.RPad / 8), r, p.RPad);
        for (int j = 0; j < r; j++)
            ClearBitsFrom(p.VBits + (long)j * (p.DInPad / 8), dIn, p.DInPad);
        for (int o = 0; o < dOut; o++) p.H[o] = 0.5f + rng.NextSingle();
        for (int i = 0; i < dIn; i++) p.G[i] = 0.5f + rng.NextSingle();
        for (int j = 0; j < r; j++) p.L[j] = 0.5f + rng.NextSingle();
        return p;
    }

    private static void FillRandomBytes(byte* p, long n, Random rng)
        => rng.NextBytes(new Span<byte>(p, checked((int)n)));

    private static void ClearBitsFrom(byte* row, int firstBit, int totalBits)
    {
        for (int b = firstBit; b < totalBits; b++) row[b >> 3] &= (byte)~(1 << (b & 7));
    }

    /// <summary>Logical sign of Us[o, j] (+1 / -1), read from the packed bits.</summary>
    public int USign(int o, int j) => ((UBits[(long)o * (RPad / 8) + (j >> 3)] >> (j & 7)) & 1) == 0 ? 1 : -1;
    /// <summary>Logical sign of Vs[i, j] (+1 / -1), read from the packed (transposed) bits.</summary>
    public int VSign(int i, int j) => ((VBits[(long)j * (DInPad / 8) + (i >> 3)] >> (i & 7)) & 1) == 0 ? 1 : -1;

    /// <summary>Creates a deep copy (distinct memory; used by benchmarks to defeat cache residency).</summary>
    public LittleBitPath Clone()
    {
        var c = new LittleBitPath(DOut, DIn, R);
        Buffer.MemoryCopy(UBits, c.UBits, (long)DOut * (RPad / 8), (long)DOut * (RPad / 8));
        Buffer.MemoryCopy(VBits, c.VBits, (long)R * (DInPad / 8), (long)R * (DInPad / 8));
        Buffer.MemoryCopy(H, c.H, 4L * DOut, 4L * DOut);
        Buffer.MemoryCopy(G, c.G, 4L * DIn, 4L * DIn);
        Buffer.MemoryCopy(L, c.L, 4L * R, 4L * R);
        return c;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (UBits == null) return;
        NativeMemory.AlignedFree(UBits); NativeMemory.AlignedFree(VBits);
        NativeMemory.AlignedFree(H); NativeMemory.AlignedFree(G); NativeMemory.AlignedFree(L);
        UBits = null; VBits = null; H = null; G = null; L = null;
    }
}

/// <summary>Caller-owned scratch for <see cref="LittleBitLayer.Gemv"/> (so layers do not each own hot scratch).</summary>
public sealed unsafe class LittleBitScratch : IDisposable
{
    internal readonly int MaxDInPad, MaxRPad;
    internal float* Xs;     // [2][MaxDInPad]
    internal float* T;      // [2][MaxRPad]
    internal sbyte* Xq;     // [2][MaxDInPad]
    internal sbyte* Tq;     // [2][MaxRPad]
    private readonly nint _mem;

    /// <summary>Allocates scratch large enough for layers up to the given padded sizes.</summary>
    public LittleBitScratch(int maxDIn, int maxR)
    {
        MaxDInPad = (maxDIn + 31) / 32 * 32;
        MaxRPad = (maxR + 31) / 32 * 32;
        long bytes = 2L * MaxDInPad * 5 + 2L * MaxRPad * 5 + 256;
        _mem = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        byte* p = (byte*)_mem;
        Xs = (float*)p; p += 2L * MaxDInPad * 4;
        T = (float*)p; p += 2L * MaxRPad * 4;
        Xq = (sbyte*)p; p += 2L * MaxDInPad;
        Tq = (sbyte*)p;
    }

    /// <inheritdoc/>
    public void Dispose() => NativeMemory.AlignedFree((void*)_mem);
}

/// <summary>A LittleBit linear layer: primary path plus optional residual path. GEMV only (decode).</summary>
public sealed unsafe class LittleBitLayer : IDisposable
{
    /// <summary>The 1 or 2 paths (index 0 primary, 1 residual).</summary>
    public LittleBitPath[] Paths { get; }
    /// <summary>Output features.</summary>
    public int DOut => Paths[0].DOut;
    /// <summary>Input features.</summary>
    public int DIn => Paths[0].DIn;

    /// <summary>Creates a layer from 1 or 2 paths (all must share d_out and d_in). Takes ownership.</summary>
    public LittleBitLayer(params LittleBitPath[] paths)
    {
        if (paths.Length is < 1 or > 2) throw new ArgumentException("1 or 2 paths");
        foreach (var p in paths)
            if (p.DOut != paths[0].DOut || p.DIn != paths[0].DIn) throw new ArgumentException("path shape mismatch");
        Paths = paths;
        _maxR = paths.Max(p => p.R);
    }

    private readonly int _maxR;

    /// <summary>Bytes streamed per GEMV (packed bits + FP16 scales, all paths).</summary>
    public long WeightBytes => Paths.Sum(p => p.WeightBytes);
    /// <summary>Paper bits per layer (all paths).</summary>
    public long PaperBits => Paths.Sum(p => p.PaperBits);
    /// <summary>Paper bits per weight element of the dense matrix this layer replaces.</summary>
    public double BitsPerWeight => (double)PaperBits / ((double)DOut * DIn);

    /// <summary>Largest padded d_in / r over paths (for sizing scratch).</summary>
    public (int dIn, int r) ScratchDims => (Paths[0].DIn, Paths.Max(p => p.R));

    /// <summary>Deep copy with distinct memory.</summary>
    public LittleBitLayer Clone() => new(Paths.Select(p => p.Clone()).ToArray());

    /// <inheritdoc/>
    public void Dispose() { foreach (var p in Paths) p.Dispose(); }

    /// <summary>Bit-rows for the bits-to-float sign mask table (256 entries x 8 lanes).</summary>
    private static readonly float* SignTable = BuildSignTable();

    private static float* BuildSignTable()
    {
        uint* t = (uint*)NativeMemory.AlignedAlloc(256 * 8 * 4, 64);
        for (int b = 0; b < 256; b++)
            for (int k = 0; k < 8; k++)
                t[b * 8 + k] = ((b >> k) & 1) != 0 ? 0x80000000u : 0u;
        return (float*)t;
    }

    /// <summary>True when <see cref="LittleBitKernel.VnniInt8"/> can run on this CPU.</summary>
    public static bool VnniSupported => AvxVnni.IsSupported && Avx2.IsSupported;

    // ───────────────────────────── dispatch ─────────────────────────────

    private struct PathCtx
    {
        public byte* U, V;
        public float* H, L;
        public float* Xs, T;
        public sbyte* Xq, Tq;
        public int DOut, DInPad, RPad, R;
        public float XScale, TScale;
        public int XSum, TSum;
    }

    private struct Ctx
    {
        public PathCtx P0, P1;
        public int PathCount;
        public float* Y;
        public int DOut;
        public int Stage1Groups;   // per path
        public LittleBitKernel Kernel;
    }

    /// <summary>
    /// y[0..DOut) = layer(x[0..DIn)). Two pool dispatches (stage 1 over latent-row groups of all paths,
    /// stage 2 over output-row groups). With a null pool everything runs on the calling thread.
    /// </summary>
    [SkipLocalsInit]
    public void Gemv(float* x, float* y, LittleBitScratch scratch, ComputeThreadPool? pool,
                     LittleBitKernel kernel = LittleBitKernel.Avx2Float)
    {
        if (!Avx2.IsSupported) throw new PlatformNotSupportedException("AVX2 required");
        if (kernel == LittleBitKernel.VnniInt8 && !VnniSupported) throw new PlatformNotSupportedException("AVX-VNNI required");

        Ctx ctx = default;
        ctx.PathCount = Paths.Length;
        ctx.Y = y;
        ctx.DOut = DOut;
        ctx.Kernel = kernel;
        int g1 = (_maxR + 7) / 8;
        ctx.Stage1Groups = g1;

        for (int pi = 0; pi < Paths.Length; pi++)
        {
            var p = Paths[pi];
            if (p.DInPad > scratch.MaxDInPad || p.RPad > scratch.MaxRPad) throw new ArgumentException("scratch too small");
            PathCtx pc = default;
            pc.U = p.UBits; pc.V = p.VBits; pc.H = p.H; pc.L = p.L;
            pc.DOut = p.DOut; pc.DInPad = p.DInPad; pc.RPad = p.RPad; pc.R = p.R;
            pc.Xs = scratch.Xs + (long)pi * scratch.MaxDInPad;
            pc.T = scratch.T + (long)pi * scratch.MaxRPad;
            pc.Xq = scratch.Xq + (long)pi * scratch.MaxDInPad;
            pc.Tq = scratch.Tq + (long)pi * scratch.MaxRPad;

            // xs = g .* x (zero padded)
            ScaleByHalf(x, p.G, pc.Xs, p.DIn);
            new Span<float>(pc.Xs + p.DIn, p.DInPad - p.DIn).Clear();
            new Span<float>(pc.T + p.R, p.RPad - p.R).Clear();
            if (kernel == LittleBitKernel.VnniInt8)
                Quantize(pc.Xs, pc.Xq, p.DInPad, out pc.XScale, out pc.XSum);
            if (pi == 0) ctx.P0 = pc; else ctx.P1 = pc;
        }

        int totalS1 = g1 * Paths.Length;
        int totalS2 = (DOut + 7) / 8;
        if (pool is null)
        {
            Stage1Range(&ctx, 0, totalS1);
            if (kernel == LittleBitKernel.VnniInt8) QuantizeT(&ctx);
            Stage2Range(&ctx, 0, totalS2);
        }
        else
        {
            pool.Dispatch((nint)(&ctx), &Stage1Worker);
            if (kernel == LittleBitKernel.VnniInt8) QuantizeT(&ctx);
            pool.Dispatch((nint)(&ctx), &Stage2Worker);
        }
    }

    private static void Stage1Worker(nint c, int tid, int tc)
    {
        Ctx* ctx = (Ctx*)c;
        ComputeThreadPool.PartitionRange(ctx->Stage1Groups * ctx->PathCount, tid, tc, out int s, out int e);
        Stage1Range(ctx, s, e);
    }

    private static void Stage2Worker(nint c, int tid, int tc)
    {
        Ctx* ctx = (Ctx*)c;
        ComputeThreadPool.PartitionRange((ctx->DOut + 7) / 8, tid, tc, out int s, out int e);
        Stage2Range(ctx, s, e);
    }

    /// <summary>t[j] = l[j] * sum_i Vs[i,j] * xs[i]  for latent-row groups [s, e) over (path, group).</summary>
    [SkipLocalsInit]
    private static void Stage1Range(Ctx* ctx, int s, int e)
    {
        int* iout = stackalloc int[8];
        float* fout = stackalloc float[8];
        for (int g = s; g < e; g++)
        {
            int pi = g / ctx->Stage1Groups, grp = g % ctx->Stage1Groups;
            PathCtx* p = pi == 0 ? &ctx->P0 : &ctx->P1;
            int j0 = grp * 8;
            if (j0 >= p->R) continue;
            int valid = Math.Min(8, p->R - j0);
            byte* rows = p->V + (long)j0 * (p->DInPad / 8);
            if (ctx->Kernel == LittleBitKernel.Avx2Float)
            {
                SignedDot8F(rows, p->DInPad / 8, valid, p->Xs, p->DInPad, fout);
                for (int k = 0; k < valid; k++) p->T[j0 + k] = fout[k] * p->L[j0 + k];
            }
            else
            {
                SignedDot8I(rows, p->DInPad / 8, valid, p->Xq, p->DInPad, iout);
                float sc = p->XScale;
                for (int k = 0; k < valid; k++)
                    p->T[j0 + k] = (float)(p->XSum - 2 * (iout[k] / 255)) * sc * p->L[j0 + k];
            }
        }
    }

    /// <summary>y[o] = sum_paths h[o] * sum_j Us[o,j] * t[j]  for output-row groups [s, e).</summary>
    [SkipLocalsInit]
    private static void Stage2Range(Ctx* ctx, int s, int e)
    {
        int* iout = stackalloc int[8];
        float* fout = stackalloc float[8];
        float* acc = stackalloc float[8];
        for (int g = s; g < e; g++)
        {
            int o0 = g * 8;
            int valid = Math.Min(8, ctx->DOut - o0);
            for (int k = 0; k < 8; k++) acc[k] = 0;
            for (int pi = 0; pi < ctx->PathCount; pi++)
            {
                PathCtx* p = pi == 0 ? &ctx->P0 : &ctx->P1;
                byte* rows = p->U + (long)o0 * (p->RPad / 8);
                if (ctx->Kernel == LittleBitKernel.Avx2Float)
                {
                    SignedDot8F(rows, p->RPad / 8, valid, p->T, p->RPad, fout);
                    for (int k = 0; k < valid; k++) acc[k] += fout[k] * p->H[o0 + k];
                }
                else
                {
                    SignedDot8I(rows, p->RPad / 8, valid, p->Tq, p->RPad, iout);
                    float sc = p->TScale;
                    for (int k = 0; k < valid; k++)
                        acc[k] += (float)(p->TSum - 2 * (iout[k] / 255)) * sc * p->H[o0 + k];
                }
            }
            for (int k = 0; k < valid; k++) ctx->Y[o0 + k] = acc[k];
        }
    }

    private static void QuantizeT(Ctx* ctx)
    {
        for (int pi = 0; pi < ctx->PathCount; pi++)
        {
            PathCtx* p = pi == 0 ? &ctx->P0 : &ctx->P1;
            Quantize(p->T, p->Tq, p->RPad, out p->TScale, out p->TSum);
        }
    }

    // ───────────────────────────── helpers ─────────────────────────────

    private static void ScaleByHalf(float* x, float* g, float* dst, int n)
    {
        TensorPrimitives.Multiply(new ReadOnlySpan<float>(g, n), new ReadOnlySpan<float>(x, n), new Span<float>(dst, n));
    }

    private static void Quantize(float* src, sbyte* dst, int n, out float scale, out int sum)
    {
        float amax = TensorPrimitives.MaxMagnitude(new ReadOnlySpan<float>(src, n));
        amax = MathF.Abs(amax);
        scale = amax > 0 ? amax / 127f : 1f;
        float inv = 1f / scale;
        int s = 0;
        for (int i = 0; i < n; i++)
        {
            int q = (int)MathF.Round(src[i] * inv);
            dst[i] = (sbyte)q;
            s += q;
        }
        sum = s;
    }

    /// <summary>
    /// out[k] = sum_{i&lt;n} sign(bit(row k, i)) * v[i] for up to 8 bit-rows (rows beyond <paramref name="valid"/>
    /// alias the last valid row and are ignored). n must be a multiple of 32.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void SignedDot8F(byte* bits, int stride, int valid, float* v, int n, float* out8)
    {
        byte* b0 = bits;
        byte* b1 = bits + (long)Math.Min(1, valid - 1) * stride;
        byte* b2 = bits + (long)Math.Min(2, valid - 1) * stride;
        byte* b3 = bits + (long)Math.Min(3, valid - 1) * stride;
        byte* b4 = bits + (long)Math.Min(4, valid - 1) * stride;
        byte* b5 = bits + (long)Math.Min(5, valid - 1) * stride;
        byte* b6 = bits + (long)Math.Min(6, valid - 1) * stride;
        byte* b7 = bits + (long)Math.Min(7, valid - 1) * stride;
        float* tab = SignTable;
        Vector256<float> a0 = Vector256<float>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        int chunks = n >> 3;
        for (int c = 0; c < chunks; c++)
        {
            var xv = Avx.LoadVector256(v + (c << 3));
            a0 = Avx.Add(a0, Avx.Xor(xv, Avx.LoadVector256(tab + (b0[c] << 3))));
            a1 = Avx.Add(a1, Avx.Xor(xv, Avx.LoadVector256(tab + (b1[c] << 3))));
            a2 = Avx.Add(a2, Avx.Xor(xv, Avx.LoadVector256(tab + (b2[c] << 3))));
            a3 = Avx.Add(a3, Avx.Xor(xv, Avx.LoadVector256(tab + (b3[c] << 3))));
            a4 = Avx.Add(a4, Avx.Xor(xv, Avx.LoadVector256(tab + (b4[c] << 3))));
            a5 = Avx.Add(a5, Avx.Xor(xv, Avx.LoadVector256(tab + (b5[c] << 3))));
            a6 = Avx.Add(a6, Avx.Xor(xv, Avx.LoadVector256(tab + (b6[c] << 3))));
            a7 = Avx.Add(a7, Avx.Xor(xv, Avx.LoadVector256(tab + (b7[c] << 3))));
        }
        var h01 = Avx.HorizontalAdd(a0, a1);
        var h23 = Avx.HorizontalAdd(a2, a3);
        var h45 = Avx.HorizontalAdd(a4, a5);
        var h67 = Avx.HorizontalAdd(a6, a7);
        var q0 = Avx.HorizontalAdd(h01, h23);
        var q1 = Avx.HorizontalAdd(h45, h67);
        Sse.Store(out8, Sse.Add(q0.GetLower(), q0.GetUpper()));
        Sse.Store(out8 + 4, Sse.Add(q1.GetLower(), q1.GetUpper()));
    }

    /// <summary>
    /// out[k] = 255 * sum_{i&lt;n, bit(row k, i)=1} a[i] (int32) for up to 8 bit-rows: the 0xFF byte mask times the int8
    /// activation through vpdpbusd. The caller derives the signed sum as sumA - 2*(out/255). n multiple of 32.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    internal static void SignedDot8I(byte* bits, int stride, int valid, sbyte* a, int n, int* out8)
    {
        byte* b0 = bits;
        byte* b1 = bits + (long)Math.Min(1, valid - 1) * stride;
        byte* b2 = bits + (long)Math.Min(2, valid - 1) * stride;
        byte* b3 = bits + (long)Math.Min(3, valid - 1) * stride;
        byte* b4 = bits + (long)Math.Min(4, valid - 1) * stride;
        byte* b5 = bits + (long)Math.Min(5, valid - 1) * stride;
        byte* b6 = bits + (long)Math.Min(6, valid - 1) * stride;
        byte* b7 = bits + (long)Math.Min(7, valid - 1) * stride;
        // lane0: bytes 0,1 replicated x8; lane1: bytes 2,3 replicated x8
        var shuf = Vector256.Create(
            (byte)0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
            2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        var sel = Vector256.Create((byte)1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128,
            1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128);
        Vector256<int> a0 = Vector256<int>.Zero, a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
        int chunks = n >> 5;
        for (int c = 0; c < chunks; c++)
        {
            var av = Avx.LoadVector256(a + (c << 5));
            int o = c << 2;
            a0 = AvxVnni.MultiplyWideningAndAdd(a0, Expand(b0 + o, shuf, sel), av);
            a1 = AvxVnni.MultiplyWideningAndAdd(a1, Expand(b1 + o, shuf, sel), av);
            a2 = AvxVnni.MultiplyWideningAndAdd(a2, Expand(b2 + o, shuf, sel), av);
            a3 = AvxVnni.MultiplyWideningAndAdd(a3, Expand(b3 + o, shuf, sel), av);
            a4 = AvxVnni.MultiplyWideningAndAdd(a4, Expand(b4 + o, shuf, sel), av);
            a5 = AvxVnni.MultiplyWideningAndAdd(a5, Expand(b5 + o, shuf, sel), av);
            a6 = AvxVnni.MultiplyWideningAndAdd(a6, Expand(b6 + o, shuf, sel), av);
            a7 = AvxVnni.MultiplyWideningAndAdd(a7, Expand(b7 + o, shuf, sel), av);
        }
        var h01 = Avx2.HorizontalAdd(a0, a1);
        var h23 = Avx2.HorizontalAdd(a2, a3);
        var h45 = Avx2.HorizontalAdd(a4, a5);
        var h67 = Avx2.HorizontalAdd(a6, a7);
        var q0 = Avx2.HorizontalAdd(h01, h23);
        var q1 = Avx2.HorizontalAdd(h45, h67);
        Sse2.Store(out8, Sse2.Add(q0.GetLower(), q0.GetUpper()));
        Sse2.Store(out8 + 4, Sse2.Add(q1.GetLower(), q1.GetUpper()));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Expand(byte* p, Vector256<byte> shuf, Vector256<byte> sel)
    {
        var bc = Avx2.BroadcastScalarToVector256((uint*)p).AsByte();
        var m = Avx2.And(Avx2.Shuffle(bc, shuf), sel);
        return Avx2.CompareEqual(m, sel);   // 0xFF where the bit is set (== -1), else 0
    }
}

/// <summary>Scalar reference and dense-decode control for <see cref="LittleBitLayer"/>. Double accumulation.</summary>
public static class LittleBitReference
{
    /// <summary>Straight transcription of y = sum_paths h .* (Us (l .* (Vs^T (g .* x)))) reading the packed bits.</summary>
    public static double[] Gemv(LittleBitLayer layer, ReadOnlySpan<float> x)
    {
        var y = new double[layer.DOut];
        foreach (var p in layer.Paths)
        {
            var t = new double[p.R];
            unsafe
            {
                for (int j = 0; j < p.R; j++)
                {
                    double s = 0;
                    for (int i = 0; i < p.DIn; i++)
                        s += p.VSign(i, j) * ((double)p.G[i] * x[i]);
                    t[j] = (double)p.L[j] * s;
                }
                for (int o = 0; o < p.DOut; o++)
                {
                    double s = 0;
                    for (int j = 0; j < p.R; j++) s += p.USign(o, j) * t[j];
                    y[o] += (double)p.H[o] * s;
                }
            }
        }
        return y;
    }

    /// <summary>
    /// Decodes the factors into a dense F32 matrix W [dOut x dIn] (row-major), summing the paths:
    /// W[o,i] = sum_p h[o] * sum_j Us[o,j] l[j] Vs[i,j] * g[i]. The "F32-decoded control".
    /// </summary>
    public static float[] DecodeDense(LittleBitLayer layer)
    {
        int dOut = layer.DOut, dIn = layer.DIn;
        var w = new float[(long)dOut * dIn];
        foreach (var p in layer.Paths)
        {
            unsafe
            {
                for (int o = 0; o < dOut; o++)
                    for (int i = 0; i < dIn; i++)
                    {
                        float s = 0;
                        for (int j = 0; j < p.R; j++) s += p.USign(o, j) * p.L[j] * p.VSign(i, j);
                        w[(long)o * dIn + i] += p.H[o] * s * p.G[i];
                    }
            }
        }
        return w;
    }

    /// <summary>Dense F32 GEMV y = W x using TensorPrimitives.Dot.</summary>
    public static float[] DenseGemv(float[] w, int dOut, int dIn, ReadOnlySpan<float> x)
    {
        var y = new float[dOut];
        for (int o = 0; o < dOut; o++)
            y[o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(w, o * dIn, dIn), x);
        return y;
    }
}
