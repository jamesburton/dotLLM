using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// BitNet b1.58 ternary (I2_S) matrix multiplication kernels.
/// Weights are ternary {-1, 0, +1} packed 4 codes per byte (block size 128), with a single
/// per-tensor float32 scale stored at the tensor tail (byte offset <c>m·k/4</c>).
///
/// Two compute paths exist, selected at runtime by available ISA:
/// <list type="bullet">
/// <item><b>W2A8 (int8 activations)</b> on AVX2/AVX-VNNI hardware. Activations are quantized
/// once per token to Q8_0 (per-32-block absmax int8); each weight row is unpacked once to
/// int8 ternary {-1,0,+1} and dotted against the int8 activation blocks. For GEMM the row is
/// unpacked a single time and reused across all N tokens.</item>
/// <item><b>Float fallback</b> on hardware without AVX2 (e.g. Westmere). Each weight row is
/// unpacked to float {-1,0,+1} and dotted via <see cref="TensorPrimitives.Dot"/>.</item>
/// </list>
///
/// <para><b>ISA tiers (highest available first):</b></para>
/// <list type="number">
/// <item><see cref="AvxVnni"/> → <c>AvxVnni.MultiplyWideningAndAdd</c> (VPDPBUSD, 256-bit):
/// the ideal int8 dot. One instruction does unsigned×signed byte multiply with int32 accumulate
/// over 4-element groups.</item>
/// <item><see cref="Avx2"/> → <c>Avx2.MultiplyAddAdjacent</c>
/// (VPMADDUBSW, maddubs) followed by a widening MAD to int32. Uses the same sign trick as
/// <c>VecDotQ8_0Avx2</c>.</item>
/// <item>Float fallback via <see cref="TensorPrimitives.Dot"/> (kept for non-AVX2 boxes).</item>
/// </list>
///
/// <para><b>Sign trick (shared by both SIMD tiers).</b> VPDPBUSD and VPMADDUBSW both require an
/// unsigned left operand and a signed right operand. The weight ternary <c>w∈{-1,0,+1}</c> and the
/// activation int8 <c>q</c> are both signed, so we transform: <c>absW = |w| = Sign(w,w) ∈ {0,1}</c>
/// (unsigned) and <c>adjQ = sign(w)·q = Sign(q,w)</c> (signed). Then
/// <c>Σ absW·adjQ = Σ |w|·sign(w)·q = Σ w·q</c>. This mirrors the proven <c>VecDotQ8_0Avx2</c>
/// kernel exactly. (The algebraic alternative — the offset trick <c>Σw·q = Σcode·q − Σq</c> with
/// the unsigned code <c>w+1∈{0,1,2}</c> — is equivalent; the sign trick is used here because it
/// reuses one unpacked representation and matches the existing Q8_0 dot.)</para>
///
/// <para>Per-block dot scaling: for a Q8_0 activation block <c>b</c> with float scale <c>d_b</c>,
/// the int32 accumulator <c>Σ w·q</c> is multiplied by <c>d_b</c> and float-accumulated across
/// blocks; the final sum is multiplied by the per-tensor weight scale.</para>
///
/// <para><b>4x4 register-blocked GEMM tile (issue #232).</b> The W2A8 <i>GEMM</i> (prefill) path
/// evaluates 4 weight rows against 4 tokens in one pass, holding all 16 cell accumulators live —
/// see <see cref="GemmI2_SW2A8RowsTiled"/> / <see cref="VecDotI2SQ8Tile4x4"/>. It is bit-exact with
/// the per-cell form (identical operations in identical order per cell; only the interleaving across
/// cells differs) and is the default whenever <see cref="Avx2"/> is available
/// (<c>DOTLLM_I2S_TILE=0</c> restores the per-cell form). Ternary amortizes more here than Q8_0
/// does: because the weights carry a single per-tensor scale, the only per-block scalar is the Q8_0
/// activation scale <c>d_b</c>, which is <b>row-invariant</b> — its load, <c>Half</c>→float convert
/// and broadcast are paid once per (block, token) rather than once per cell, alongside the weight
/// load and its <c>Sign</c>. Measured 2.1-3.4x single-threaded / 1.7-2.1x pooled at BitNet-2B-4T
/// prefill shapes; unlike Q8_0 (upstream #416) the win does <b>not</b> depend on the AVX-512VL
/// register file.</para>
///
/// <para><b>Ragged K (issue #206).</b> Most published I2_S checkpoints have <c>k</c> an exact
/// multiple of 128 (e.g. BitNet-2B-4T: 2560/6912), which is what the SIMD fast paths above
/// assume. At least one real checkpoint family (1bitLLM-style <c>bitnet_b1_58-large</c>/<c>-xl</c>:
/// hidden=2048, intermediate=5460, and <c>5460 % 128 == 84</c>) genuinely has a non-128-aligned
/// row length on <c>ffn_down</c>. Empirically verified against the real GGUF (tensor byte
/// extents match <c>m·k/4 + 4</c>, rounded up to the file's 32-byte tensor alignment, with zero
/// extra per-row padding) and cross-checked against the upstream bitnet.cpp writer
/// (<c>ggml-bitnet-mad.cpp</c>'s <c>quantize_i2_s</c>): the 128-element block interleave is
/// computed over the <b>flattened <c>m·k</c> element stream</b>, not reset at each row boundary.
/// Concretely, block index and in-block bit position for flattened element <c>i</c> are
/// <c>i/128</c> and <c>i%128</c> — so whenever <c>k % 128 != 0</c>, a row's first element does
/// NOT generally start on a fresh block boundary (only every <c>128/gcd(k,128)</c>-th row does).
/// The 128-aligned fast paths above are unaffected by this — when <c>k % 128 == 0</c> every row
/// boundary coincides with a block boundary, so per-row block resets and the flattened-stream
/// addressing are bit-identical (which is why the fast paths have worked for every 128-aligned
/// model to date). Ragged rows are decoded via <see cref="I2SRaggedCode"/> /
/// <see cref="UnpackRowRagged"/>, a scalar correctness-first path that computes the flattened
/// block/bit address per element directly; it is reached only when <c>k % 128 != 0</c> and never
/// perturbs the aligned SIMD paths.</para>
/// </summary>
public static unsafe partial class MatMul
{
    private const int I2SBlockSize = 128;

    // ── Context structs ──

    private struct GemvI2SCtx
    {
        public byte* Weights;
        public float* X;
        public float* Result;
        public int M;
        public int K;
        public float Scale;
    }

    private struct GemmI2SCtx
    {
        public byte* Weights;
        public float* B;
        public float* C;
        public int M;
        public int K;
        public int N;
        public float Scale;
    }

    private struct GemvI2SQ8Ctx
    {
        public byte* Weights;
        public byte* XQ8;   // quantized activations (Q8_0), one token
        public float* Result;
        public int M;
        public int K;
        public float Scale;
    }

    private struct GemmI2SQ8Ctx
    {
        public byte* Weights;
        public byte* BQ8;   // quantized activations (Q8_0), N tokens contiguous
        public float* C;
        public int M;
        public int K;
        public int N;
        public float Scale;
        public bool Tiled;  // issue #232: use the 4x4 register-blocked inner kernel
    }

    /// <summary>True when a SIMD W2A8 (int8-activation) path is available.</summary>
    private static bool I2SUseW2A8 => Avx2.IsSupported;

    // ─────────────────────────── 4x4 register-blocked GEMM tile (issue #232) ───────────────────────────

    /// <summary>Weight rows per register-blocked GEMM tile (issue #232).</summary>
    private const int I2STileRows = 4;

    /// <summary>Tokens per register-blocked GEMM tile (issue #232).</summary>
    private const int I2STileTokens = 4;

    /// <summary>
    /// Runtime gate for the 4x4 register-blocked I2_S W2A8 GEMM inner kernel (issue #232).
    /// The vector <i>width</i> stays 256-bit — 512-bit dot widening was measured 1.8-1.9x slower on
    /// Zen5 (issues #196/#202) and #416 reached the same conclusion independently.
    ///
    /// <para><b>Measured (Zen5 / Strix Halo, issue #232): the AVX-512VL register file is NOT the
    /// enabler for the ternary tile.</b> The tile keeps 16 <see cref="Vector256{T}"/> float
    /// accumulators live, so #416's Q8_0 result predicted it would need ymm16-31 to avoid spilling.
    /// Re-running the identical A/B with <c>DOTNET_EnableAVX512F=0</c> (RyuJIT restricted to
    /// ymm0-15) reproduced the speedup to within noise at every N — the tile's win comes from
    /// collapsing per-cell work that is shared across a tile row (the row-invariant activation
    /// scale, the weight load and its <c>Sign</c>), which dominates any spill traffic. The gate is
    /// therefore <see cref="Avx2"/>, not AVX-512.</para>
    ///
    /// <para>Overridable via <c>DOTLLM_I2S_TILE</c> (<c>0</c>/<c>1</c>) for A/B runs.</para>
    /// </summary>
    private static readonly bool I2SGemmTile = ResolveI2SGemmTile();

    private static bool ResolveI2SGemmTile()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_I2S_TILE");
        if (env is "0" or "false" or "off") return false;
        return Avx2.IsSupported;
    }

    /// <summary>
    /// I2_S ternary GEMV: <c>result[r] = scale · dot(ternary(A[r,:]), x)</c>.
    /// A is [M,K] packed I2_S (row-major, K a multiple of 128); x is f32 [K]; result is f32 [M].
    /// The per-tensor scale is read from the tail of <paramref name="weights"/>. On AVX2/AVX-VNNI
    /// hardware the activation is quantized to int8 (Q8_0) once and the W2A8 SIMD path runs; older
    /// hardware falls back to the float path. Output rows are partitioned across
    /// <paramref name="threadPool"/> workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvI2_S(byte* weights, float* x, float* result, int m, int k,
                                ComputeThreadPool? threadPool)
    {
        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);

        if (k % I2SBlockSize != 0)
        {
            GemvI2_SRaggedCore(weights, x, result, m, k, scale, threadPool);
            return;
        }

        GemvI2_SCore(weights, x, result, m, k, scale, threadPool);
    }

    /// <summary>
    /// I2_S ternary GEMV with an <b>explicitly supplied</b> per-tensor scale (rather than
    /// reading it from the weight-tensor tail). Used by the indexed-MoE path, where the
    /// per-expert α comes from a scale-per-expert vector and the expert weight banks store
    /// packed trits only (no inline tail scale). <paramref name="weights"/> points at the
    /// packed payload (<c>m·k/4</c> bytes); the inner compute is identical to
    /// <see cref="GemvI2_S(byte*, float*, float*, int, int, ComputeThreadPool?)"/>.
    /// </summary>
    [SkipLocalsInit]
    public static void GemvI2_S(byte* weights, float* x, float* result, int m, int k,
                                float scale, ComputeThreadPool? threadPool)
    {
        if (k % I2SBlockSize != 0)
        {
            GemvI2_SRaggedCore(weights, x, result, m, k, scale, threadPool);
            return;
        }
        GemvI2_SCore(weights, x, result, m, k, scale, threadPool);
    }

    [SkipLocalsInit]
    private static void GemvI2_SCore(byte* weights, float* x, float* result, int m, int k,
                                     float scale, ComputeThreadPool? threadPool)
    {
        if (I2SUseW2A8)
        {
            GemvI2_SW2A8(weights, x, result, m, k, scale, threadPool);
            return;
        }

        if (threadPool is null || m < ParallelMinRows)
        {
            GemvI2_SRows(weights, x, result, 0, m, k, scale);
            return;
        }

        var ctx = new GemvI2SCtx { Weights = weights, X = x, Result = result, M = m, K = k, Scale = scale };
        threadPool.Dispatch((nint)(&ctx), &GemvI2_SWorker);
    }

    // ─────────────────────────── Float fallback (no AVX2) ───────────────────────────

    [SkipLocalsInit]
    private static void GemvI2_SWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvI2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvI2_SRows(ctx.Weights, ctx.X, ctx.Result, start, count, ctx.K, ctx.Scale);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows starting at <paramref name="startRow"/> (float path).</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvI2_SRows(byte* weights, float* x, float* result,
                                     int startRow, int rowCount, int k, float scale)
    {
        int rowBytes = k / 4;
        var xSpan = new ReadOnlySpan<float>(x, k);

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackRow(weights + (long)r * rowBytes, rowSpan, k);
                result[r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan) * scale;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// I2_S ternary GEMM: <c>C[N,M] = (B[N,K] × ternary(A[M,K])^T) · scale</c>.
    /// Each weight row is unpacked once and dotted against all N input rows. On AVX2/AVX-VNNI
    /// hardware all N tokens are quantized to int8 (Q8_0) once and the W2A8 SIMD path runs (each
    /// weight row unpacked to int8 exactly once, then dotted against every token); older hardware
    /// falls back to the float path. Output rows are partitioned across <paramref name="threadPool"/>
    /// workers when present.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmI2_S(byte* weights, float* b, float* c, int m, int k, int n,
                                ComputeThreadPool? threadPool)
    {
        if (n == 1)
        {
            GemvI2_S(weights, b, c, m, k, threadPool);
            return;
        }

        float scale = Unsafe.ReadUnaligned<float>(weights + (long)m * k / 4);

        if (k % I2SBlockSize != 0)
        {
            GemmI2_SRaggedCore(weights, b, c, m, k, n, scale, threadPool);
            return;
        }

        GemmI2_SCore(weights, b, c, m, k, n, scale, threadPool);
    }

    /// <summary>
    /// I2_S ternary GEMM with an <b>explicitly supplied</b> per-tensor scale (rather than
    /// reading it from the weight-tensor tail). Used by the indexed-MoE path
    /// (<see cref="MoeIndexedMatmulI2_S"/>): the per-expert α comes from a scale-per-expert
    /// vector and the expert weight banks store packed trits only (no inline tail scale).
    /// <paramref name="weights"/> points at the packed payload (<c>m·k/4</c> bytes); the inner
    /// compute is identical to
    /// <see cref="GemmI2_S(byte*, float*, float*, int, int, int, ComputeThreadPool?)"/>.
    /// </summary>
    [SkipLocalsInit]
    public static void GemmI2_S(byte* weights, float* b, float* c, int m, int k, int n,
                                float scale, ComputeThreadPool? threadPool)
    {
        if (n == 1)
        {
            GemvI2_S(weights, b, c, m, k, scale, threadPool);
            return;
        }

        if (k % I2SBlockSize != 0)
        {
            GemmI2_SRaggedCore(weights, b, c, m, k, n, scale, threadPool);
            return;
        }

        GemmI2_SCore(weights, b, c, m, k, n, scale, threadPool);
    }

    [SkipLocalsInit]
    private static void GemmI2_SCore(byte* weights, float* b, float* c, int m, int k, int n,
                                     float scale, ComputeThreadPool? threadPool)
    {
        if (I2SUseW2A8)
        {
            GemmI2_SW2A8(weights, b, c, m, k, n, scale, threadPool);
            return;
        }

        if (threadPool is null || m < ParallelMinRows)
        {
            GemmI2_SRows(weights, b, c, m, 0, m, k, n, scale);
            return;
        }

        var ctx = new GemmI2SCtx { Weights = weights, B = b, C = c, M = m, K = k, N = n, Scale = scale };
        threadPool.Dispatch((nint)(&ctx), &GemmI2_SWorker);
    }

    [SkipLocalsInit]
    private static void GemmI2_SWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmI2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmI2_SRows(ctx.Weights, ctx.B, ctx.C, ctx.M, start, count, ctx.K, ctx.N, ctx.Scale);
    }

    /// <summary>Computes <paramref name="rowCount"/> weight rows (over all N tokens) starting at <paramref name="startRow"/> (float path).</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmI2_SRows(byte* weights, float* b, float* c, int m,
                                     int startRow, int rowCount, int k, int n, float scale)
    {
        int rowBytes = k / 4;

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackRow(weights + (long)r * rowBytes, rowSpan, k);
                for (int t = 0; t < n; t++)
                {
                    var xSpan = new ReadOnlySpan<float>(b + (long)t * k, k);
                    c[(long)t * m + r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan) * scale;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    // ─────────────────────────── Ragged K (k % 128 != 0) — issue #206 ───────────────────────────
    //
    // Correctness-first scalar path. Reached ONLY when k % I2SBlockSize != 0 (see the class-level
    // remarks for why the 128-aligned fast paths above cannot simply be reused with a "tail
    // cleanup" — the on-disk block interleave is computed over the flattened m*k stream, so most
    // ragged rows don't even start on a block boundary). Never invoked for aligned tensors, so it
    // cannot regress the benchmarked common-case performance.

    /// <summary>
    /// Decodes the raw 2-bit ternary code (0, 1, 2 — caller subtracts 1 for {-1,0,+1}) at
    /// flattened element index <paramref name="flatIndex"/> of a ragged I2_S tensor, using the
    /// tensor-global (not per-row) 128-element block interleave. Block <c>b = flatIndex/128</c>
    /// occupies packed bytes <c>[b·32, b·32+32)</c>; within a block, in-block position
    /// <c>p = flatIndex%128</c> maps to byte <c>b·32 + p%32</c> and 2-bit field
    /// <c>(p/32)</c> at bit offset <c>6 − 2·(p/32)</c> (same bit convention as the aligned
    /// <see cref="UnpackRow"/>/<see cref="UnpackRowI8"/>, just addressed relative to the whole
    /// tensor rather than one row).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int I2SRaggedCode(byte* tensorBase, long flatIndex)
    {
        long block = flatIndex >> 7;           // flatIndex / 128
        int inBlock = (int)(flatIndex & 127);   // flatIndex % 128
        int groupPos = inBlock & 31;            // byte within the block's 32 bytes
        int groupIdx = inBlock >> 5;            // which of the 4 interleaved slots (0..3)
        byte packed = tensorBase[block * 32 + groupPos];
        int shift = 6 - 2 * groupIdx;
        return (packed >> shift) & 0x3;
    }

    /// <summary>
    /// Unpacks logical row <paramref name="row"/> (arbitrary <paramref name="k"/>, need not be
    /// 128-aligned) of a ragged I2_S tensor into float {-1,0,+1} via <see cref="I2SRaggedCode"/>.
    /// <paramref name="tensorBase"/> is the tensor's packed-payload base (element 0 of row 0) —
    /// the SAME pointer the aligned paths call "weights".
    /// </summary>
    [SkipLocalsInit]
    private static void UnpackRowRagged(byte* tensorBase, long row, int k, Span<float> dest)
    {
        long rowStart = row * (long)k;
        for (int col = 0; col < k; col++)
            dest[col] = I2SRaggedCode(tensorBase, rowStart + col) - 1;
    }

    [SkipLocalsInit]
    private static void GemvI2_SRaggedCore(byte* weights, float* x, float* result, int m, int k,
                                           float scale, ComputeThreadPool? threadPool)
    {
        if (threadPool is null || m < ParallelMinRows)
        {
            GemvI2_SRaggedRows(weights, x, result, 0, m, k, scale);
            return;
        }

        var ctx = new GemvI2SCtx { Weights = weights, X = x, Result = result, M = m, K = k, Scale = scale };
        threadPool.Dispatch((nint)(&ctx), &GemvI2_SRaggedWorker);
    }

    [SkipLocalsInit]
    private static void GemvI2_SRaggedWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvI2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvI2_SRaggedRows(ctx.Weights, ctx.X, ctx.Result, start, count, ctx.K, ctx.Scale);
    }

    /// <summary>Ragged twin of <see cref="GemvI2_SRows"/> — unpacks each row via the
    /// tensor-global addressing (<see cref="UnpackRowRagged"/>) instead of the per-row-reset
    /// fast unpack, since row boundaries don't generally align with block boundaries here.</summary>
    [SkipLocalsInit]
    private static void GemvI2_SRaggedRows(byte* weights, float* x, float* result,
                                           int startRow, int rowCount, int k, float scale)
    {
        var xSpan = new ReadOnlySpan<float>(x, k);

        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackRowRagged(weights, r, k, rowSpan);
                result[r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan) * scale;
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    [SkipLocalsInit]
    private static void GemmI2_SRaggedCore(byte* weights, float* b, float* c, int m, int k, int n,
                                           float scale, ComputeThreadPool? threadPool)
    {
        if (threadPool is null || m < ParallelMinRows)
        {
            GemmI2_SRaggedRows(weights, b, c, m, 0, m, k, n, scale);
            return;
        }

        var ctx = new GemmI2SCtx { Weights = weights, B = b, C = c, M = m, K = k, N = n, Scale = scale };
        threadPool.Dispatch((nint)(&ctx), &GemmI2_SRaggedWorker);
    }

    [SkipLocalsInit]
    private static void GemmI2_SRaggedWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmI2SCtx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmI2_SRaggedRows(ctx.Weights, ctx.B, ctx.C, ctx.M, start, count, ctx.K, ctx.N, ctx.Scale);
    }

    /// <summary>Ragged twin of <see cref="GemmI2_SRows"/>.</summary>
    [SkipLocalsInit]
    private static void GemmI2_SRaggedRows(byte* weights, float* b, float* c, int m,
                                           int startRow, int rowCount, int k, int n, float scale)
    {
        float[] rowBuf = ArrayPool<float>.Shared.Rent(k);
        try
        {
            var rowSpan = rowBuf.AsSpan(0, k);
            for (int r = startRow; r < startRow + rowCount; r++)
            {
                UnpackRowRagged(weights, r, k, rowSpan);
                for (int t = 0; t < n; t++)
                {
                    var xSpan = new ReadOnlySpan<float>(b + (long)t * k, k);
                    c[(long)t * m + r] = TensorPrimitives.Dot((ReadOnlySpan<float>)rowSpan, xSpan) * scale;
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    // ─────────────────────────── W2A8 (int8-activation) SIMD path ───────────────────────────

    /// <summary>
    /// W2A8 GEMV: quantizes the activation to Q8_0 once, then partitions weight rows over the pool.
    /// </summary>
    [SkipLocalsInit]
    private static void GemvI2_SW2A8(byte* weights, float* x, float* result, int m, int k,
                                     float scale, ComputeThreadPool? threadPool)
    {
        int blockCount = k / Q8_0GroupSize;
        int xQ8Bytes = blockCount * Q8_0BlockBytes;

        byte[] xQ8Buf = ArrayPool<byte>.Shared.Rent(xQ8Bytes);
        try
        {
            fixed (byte* xQ8 = xQ8Buf)
            {
                QuantizeF32ToQ8_0(x, xQ8, k);

                if (threadPool is null || m < ParallelMinRows)
                {
                    GemvI2_SW2A8Rows(weights, xQ8, result, 0, m, k, scale);
                    return;
                }

                var ctx = new GemvI2SQ8Ctx { Weights = weights, XQ8 = xQ8, Result = result, M = m, K = k, Scale = scale };
                threadPool.Dispatch((nint)(&ctx), &GemvI2_SW2A8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(xQ8Buf);
        }
    }

    [SkipLocalsInit]
    private static void GemvI2_SW2A8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemvI2SQ8Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemvI2_SW2A8Rows(ctx.Weights, ctx.XQ8, ctx.Result, start, count, ctx.K, ctx.Scale);
    }

    /// <summary>Computes <paramref name="rowCount"/> output rows (W2A8) starting at <paramref name="startRow"/>.</summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemvI2_SW2A8Rows(byte* weights, byte* xQ8, float* result,
                                         int startRow, int rowCount, int k, float scale)
    {
        int rowBytes = k / 4;
        int blockCount = k / Q8_0GroupSize;

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            {
                for (int r = startRow; r < startRow + rowCount; r++)
                {
                    UnpackRowI8(weights + (long)r * rowBytes, wI8, k);
                    result[r] = VecDotI2SQ8(wI8, xQ8, blockCount) * scale;
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// W2A8 GEMM: quantizes all N tokens to Q8_0 once, then partitions weight rows over the pool.
    /// Each weight row is unpacked to int8 exactly once and dotted against every token (amortized).
    /// </summary>
    [SkipLocalsInit]
    private static void GemmI2_SW2A8(byte* weights, float* b, float* c, int m, int k, int n,
                                     float scale, ComputeThreadPool? threadPool)
        => GemmI2_SW2A8(weights, b, c, m, k, n, scale, threadPool, I2SGemmTile);

    /// <inheritdoc cref="GemmI2_SW2A8(byte*, float*, float*, int, int, int, float, ComputeThreadPool?)"/>
    /// <remarks><paramref name="tiled"/> selects the 4x4 register-blocked inner kernel (issue #232);
    /// it is threaded through explicitly so a benchmark can A/B both forms <b>within one run</b>
    /// (cross-run ratios are not sound for this question).</remarks>
    [SkipLocalsInit]
    private static void GemmI2_SW2A8(byte* weights, float* b, float* c, int m, int k, int n,
                                     float scale, ComputeThreadPool? threadPool, bool tiled)
    {
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        long bQ8Bytes = (long)n * q8RowBytes;

        byte[] bQ8Buf = ArrayPool<byte>.Shared.Rent(checked((int)bQ8Bytes));
        try
        {
            fixed (byte* bQ8 = bQ8Buf)
            {
                for (int t = 0; t < n; t++)
                    QuantizeF32ToQ8_0(b + (long)t * k, bQ8 + (long)t * q8RowBytes, k);

                if (threadPool is null || m < ParallelMinRows)
                {
                    GemmI2_SW2A8Rows(weights, bQ8, c, m, 0, m, k, n, scale, tiled);
                    return;
                }

                var ctx = new GemmI2SQ8Ctx { Weights = weights, BQ8 = bQ8, C = c, M = m, K = k, N = n, Scale = scale, Tiled = tiled };
                threadPool.Dispatch((nint)(&ctx), &GemmI2_SW2A8Worker);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(bQ8Buf);
        }
    }

    [SkipLocalsInit]
    private static void GemmI2_SW2A8Worker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<GemmI2SQ8Ctx>((void*)ctxPtr);
        PartitionRows(ctx.M, threadIdx, threadCount, out int start, out int count);
        if (count == 0) return;
        GemmI2_SW2A8Rows(ctx.Weights, ctx.BQ8, ctx.C, ctx.M, start, count, ctx.K, ctx.N, ctx.Scale, ctx.Tiled);
    }

    /// <summary>
    /// Computes <paramref name="rowCount"/> weight rows (over all N tokens, W2A8) starting at
    /// <paramref name="startRow"/>. The weight row is unpacked to int8 once then reused for all N.
    /// When <paramref name="tiled"/> is set, dispatches to the 4x4 register-blocked variant
    /// (issue #232), which is bit-exact with this one.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmI2_SW2A8Rows(byte* weights, byte* bQ8, float* c, int m,
                                         int startRow, int rowCount, int k, int n, float scale,
                                         bool tiled)
    {
        if (tiled)
        {
            GemmI2_SW2A8RowsTiled(weights, bQ8, c, m, startRow, rowCount, k, n, scale);
            return;
        }

        int rowBytes = k / 4;
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k);
        try
        {
            fixed (sbyte* wI8 = rowBuf)
            {
                for (int r = startRow; r < startRow + rowCount; r++)
                {
                    UnpackRowI8(weights + (long)r * rowBytes, wI8, k);   // unpack ONCE per row
                    for (int t = 0; t < n; t++)
                    {
                        byte* xQ8 = bQ8 + (long)t * q8RowBytes;
                        c[(long)t * m + r] = VecDotI2SQ8(wI8, xQ8, blockCount) * scale;
                    }
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// <b>Issue #232 — 4x4 register-blocked twin of <see cref="GemmI2_SW2A8Rows"/>.</b>
    /// Unpacks <see cref="I2STileRows"/> weight rows at a time and evaluates them against
    /// <see cref="I2STileTokens"/> tokens in one pass, keeping all 16 cell accumulators live in
    /// registers. Ragged row/token remainders fall back to the per-cell
    /// <see cref="VecDotI2SQ8"/> used by the untiled path.
    ///
    /// <para><b>Bit-exactness.</b> Per cell <c>(r,t)</c> the tile performs the identical operation
    /// sequence in the identical order as <see cref="VecDotI2SQ8"/> — the same per-block
    /// <c>Sign</c>/dot/<c>ConvertToVector256Single</c>/FMA into the same 8-lane accumulator, then the
    /// same <c>HorizontalSumAvx2Float</c> and the same final multiply by the per-tensor scale. Only
    /// the <i>interleaving across cells</i> changes, which cannot perturb a value. This is a
    /// requirement, not an aspiration: the tile must be bit-identical.</para>
    ///
    /// <para><b>Why ternary can amortize more than Q8_0 here.</b> The only per-block scalar is the
    /// Q8_0 <i>activation</i> scale <c>d_b</c> (the weights carry a single per-tensor scale), so
    /// <c>d_b</c> is row-invariant: its load, <c>Half</c>→float conversion and broadcast happen once
    /// per (block, token) and are reused across all <see cref="I2STileRows"/> rows. For Q8_0 the
    /// per-cell scale is <c>dw[r][b]·dx[t][b]</c> and cannot amortize.</para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void GemmI2_SW2A8RowsTiled(byte* weights, byte* bQ8, float* c, int m,
                                              int startRow, int rowCount, int k, int n, float scale)
    {
        int rowBytes = k / 4;
        int blockCount = k / Q8_0GroupSize;
        int q8RowBytes = blockCount * Q8_0BlockBytes;
        int endRow = startRow + rowCount;

        float* tile = stackalloc float[I2STileRows * I2STileTokens];

        sbyte[] rowBuf = ArrayPool<sbyte>.Shared.Rent(k * I2STileRows);
        try
        {
            fixed (sbyte* w = rowBuf)
            {
                int r = startRow;
                for (; r + I2STileRows <= endRow; r += I2STileRows)
                {
                    for (int i = 0; i < I2STileRows; i++)
                        UnpackRowI8(weights + (long)(r + i) * rowBytes, w + (long)i * k, k);

                    int t = 0;
                    for (; t + I2STileTokens <= n; t += I2STileTokens)
                    {
                        VecDotI2SQ8Tile4x4(w, k, bQ8 + (long)t * q8RowBytes, q8RowBytes, blockCount, tile);
                        for (int i = 0; i < I2STileRows; i++)
                            for (int j = 0; j < I2STileTokens; j++)
                                c[(long)(t + j) * m + r + i] = tile[i * I2STileTokens + j] * scale;
                    }

                    // Token remainder (n % 4) — per-cell, same kernel as the untiled path.
                    for (; t < n; t++)
                    {
                        byte* xQ8 = bQ8 + (long)t * q8RowBytes;
                        for (int i = 0; i < I2STileRows; i++)
                            c[(long)t * m + r + i] = VecDotI2SQ8(w + (long)i * k, xQ8, blockCount) * scale;
                    }
                }

                // Row remainder (rowCount % 4).
                for (; r < endRow; r++)
                {
                    UnpackRowI8(weights + (long)r * rowBytes, w, k);
                    for (int t = 0; t < n; t++)
                        c[(long)t * m + r] = VecDotI2SQ8(w, bQ8 + (long)t * q8RowBytes, blockCount) * scale;
                }
            }
        }
        finally
        {
            ArrayPool<sbyte>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// One 4x4 cell of <see cref="VecDotI2SQ8Tile4x4"/>: <c>acc += d_t · (Σ absW·sign(w)·q)</c>,
    /// operation-for-operation identical to the body of <see cref="VecDotI2SQ8"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> I2STileCell(Vector256<sbyte> absW, Vector256<sbyte> vw,
                                                Vector256<sbyte> vq, Vector256<float> bscale,
                                                Vector256<float> acc, bool useVnni,
                                                Vector256<short> ones)
    {
        Vector256<sbyte> adjQ = Avx2.Sign(vq, vw);

        Vector256<int> isum;
        if (useVnni)
            isum = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, absW.AsByte(), adjQ);
        else
            isum = Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(absW.AsByte(), adjQ), ones);

        Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
        return Fma.IsSupported ? Fma.MultiplyAdd(bscale, fsum, acc) : acc + fsum * bscale;
    }

    /// <summary>
    /// Register-blocked 4x4 W2A8 dot (issue #232): 4 unpacked int8 ternary weight rows
    /// (<paramref name="w"/>, stride <paramref name="wStride"/> elements) against 4 Q8_0 activation
    /// rows (<paramref name="x"/>, stride <paramref name="xStride"/> bytes). Writes 16 unscaled dot
    /// products to <paramref name="outTile"/> in row-major <c>[r·4 + t]</c> order.
    /// The activation block scale <c>d_b</c> is fetched, converted and broadcast once per
    /// (block, token) and reused across all 4 weight rows.
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void VecDotI2SQ8Tile4x4(sbyte* w, int wStride, byte* x, int xStride,
                                           int blockCount, float* outTile)
    {
        Vector256<float> a00 = Vector256<float>.Zero, a01 = Vector256<float>.Zero,
                         a02 = Vector256<float>.Zero, a03 = Vector256<float>.Zero;
        Vector256<float> a10 = Vector256<float>.Zero, a11 = Vector256<float>.Zero,
                         a12 = Vector256<float>.Zero, a13 = Vector256<float>.Zero;
        Vector256<float> a20 = Vector256<float>.Zero, a21 = Vector256<float>.Zero,
                         a22 = Vector256<float>.Zero, a23 = Vector256<float>.Zero;
        Vector256<float> a30 = Vector256<float>.Zero, a31 = Vector256<float>.Zero,
                         a32 = Vector256<float>.Zero, a33 = Vector256<float>.Zero;

        bool useVnni = AvxVnni.IsSupported;
        Vector256<short> ones = Vector256.Create((short)1);

        sbyte* w0 = w;
        sbyte* w1 = w + wStride;
        sbyte* w2 = w + 2 * wStride;
        sbyte* w3 = w + 3 * wStride;

        byte* x0 = x;
        byte* x1 = x + xStride;
        byte* x2 = x + 2 * xStride;
        byte* x3 = x + 3 * xStride;

        for (int block = 0; block < blockCount; block++)
        {
            int xoff = block * Q8_0BlockBytes;
            byte* xb0 = x0 + xoff;
            byte* xb1 = x1 + xoff;
            byte* xb2 = x2 + xoff;
            byte* xb3 = x3 + xoff;

            // Row-invariant per-block activation scales: fetched + broadcast once for all 4 rows.
            Vector256<float> s0 = Vector256.Create((float)Unsafe.ReadUnaligned<Half>(xb0));
            Vector256<float> s1 = Vector256.Create((float)Unsafe.ReadUnaligned<Half>(xb1));
            Vector256<float> s2 = Vector256.Create((float)Unsafe.ReadUnaligned<Half>(xb2));
            Vector256<float> s3 = Vector256.Create((float)Unsafe.ReadUnaligned<Half>(xb3));

            Vector256<sbyte> q0 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb0 + 2);
            Vector256<sbyte> q1 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb1 + 2);
            Vector256<sbyte> q2 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb2 + 2);
            Vector256<sbyte> q3 = Unsafe.ReadUnaligned<Vector256<sbyte>>(xb3 + 2);

            int woff = block * Q8_0GroupSize;

            Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(w0 + woff);
            Vector256<sbyte> absW = Avx2.Sign(vw, vw);
            a00 = I2STileCell(absW, vw, q0, s0, a00, useVnni, ones);
            a01 = I2STileCell(absW, vw, q1, s1, a01, useVnni, ones);
            a02 = I2STileCell(absW, vw, q2, s2, a02, useVnni, ones);
            a03 = I2STileCell(absW, vw, q3, s3, a03, useVnni, ones);

            vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(w1 + woff);
            absW = Avx2.Sign(vw, vw);
            a10 = I2STileCell(absW, vw, q0, s0, a10, useVnni, ones);
            a11 = I2STileCell(absW, vw, q1, s1, a11, useVnni, ones);
            a12 = I2STileCell(absW, vw, q2, s2, a12, useVnni, ones);
            a13 = I2STileCell(absW, vw, q3, s3, a13, useVnni, ones);

            vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(w2 + woff);
            absW = Avx2.Sign(vw, vw);
            a20 = I2STileCell(absW, vw, q0, s0, a20, useVnni, ones);
            a21 = I2STileCell(absW, vw, q1, s1, a21, useVnni, ones);
            a22 = I2STileCell(absW, vw, q2, s2, a22, useVnni, ones);
            a23 = I2STileCell(absW, vw, q3, s3, a23, useVnni, ones);

            vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(w3 + woff);
            absW = Avx2.Sign(vw, vw);
            a30 = I2STileCell(absW, vw, q0, s0, a30, useVnni, ones);
            a31 = I2STileCell(absW, vw, q1, s1, a31, useVnni, ones);
            a32 = I2STileCell(absW, vw, q2, s2, a32, useVnni, ones);
            a33 = I2STileCell(absW, vw, q3, s3, a33, useVnni, ones);
        }

        outTile[0] = HorizontalSumAvx2Float(a00);
        outTile[1] = HorizontalSumAvx2Float(a01);
        outTile[2] = HorizontalSumAvx2Float(a02);
        outTile[3] = HorizontalSumAvx2Float(a03);
        outTile[4] = HorizontalSumAvx2Float(a10);
        outTile[5] = HorizontalSumAvx2Float(a11);
        outTile[6] = HorizontalSumAvx2Float(a12);
        outTile[7] = HorizontalSumAvx2Float(a13);
        outTile[8] = HorizontalSumAvx2Float(a20);
        outTile[9] = HorizontalSumAvx2Float(a21);
        outTile[10] = HorizontalSumAvx2Float(a22);
        outTile[11] = HorizontalSumAvx2Float(a23);
        outTile[12] = HorizontalSumAvx2Float(a30);
        outTile[13] = HorizontalSumAvx2Float(a31);
        outTile[14] = HorizontalSumAvx2Float(a32);
        outTile[15] = HorizontalSumAvx2Float(a33);
    }

    /// <summary>
    /// W2A8 dot: <c>Σ_blocks d_b · Σ_{i∈block} w[i]·q[i]</c> for one int8 ternary weight row
    /// (<paramref name="wI8"/>, contiguous int8 {-1,0,+1}, length <c>k</c>) and one Q8_0-quantized
    /// activation row (<paramref name="xQ8"/>). Dispatches to the VNNI tier when available, else the
    /// AVX2 (maddubs) tier. Both tiers use the sign trick (see class summary).
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static float VecDotI2SQ8(sbyte* wI8, byte* xQ8, int blockCount)
    {
        Vector256<float> acc = Vector256<float>.Zero;
        bool useVnni = AvxVnni.IsSupported;
        Vector256<short> ones = Vector256.Create((short)1);

        for (int block = 0; block < blockCount; block++)
        {
            // Activation Q8_0 block: 2-byte Half scale + 32 sbyte values.
            byte* xBlock = xQ8 + block * Q8_0BlockBytes;
            float dx = (float)Unsafe.ReadUnaligned<Half>(xBlock);

            // 32 contiguous int8 weights aligned with this Q8_0 block.
            Vector256<sbyte> vw = Unsafe.ReadUnaligned<Vector256<sbyte>>(wI8 + block * Q8_0GroupSize);
            Vector256<sbyte> vq = Unsafe.ReadUnaligned<Vector256<sbyte>>(xBlock + 2);

            // Sign trick: absW = |w| ∈ {0,1} (unsigned operand), adjQ = sign(w)·q (signed operand).
            Vector256<sbyte> absW = Avx2.Sign(vw, vw);
            Vector256<sbyte> adjQ = Avx2.Sign(vq, vw);

            Vector256<int> isum;
            if (useVnni)
            {
                // VPDPBUSD: int32 += Σ (unsigned byte · signed byte) over 4-element groups.
                isum = AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, absW.AsByte(), adjQ);
            }
            else
            {
                // VPMADDUBSW (maddubs): unsigned×signed → int16 pairs; then widen to int32.
                Vector256<short> prod = Avx2.MultiplyAddAdjacent(absW.AsByte(), adjQ);
                isum = Avx2.MultiplyAddAdjacent(prod, ones);
            }

            // int32 block sum → float, scale by activation block scale, accumulate.
            Vector256<float> fsum = Avx.ConvertToVector256Single(isum);
            Vector256<float> bscale = Vector256.Create(dx);
            if (Fma.IsSupported)
                acc = Fma.MultiplyAdd(bscale, fsum, acc);
            else
                acc += fsum * bscale;
        }

        return HorizontalSumAvx2Float(acc);
    }

    /// <summary>
    /// Unpacks one I2_S-packed weight row (K codes, K a multiple of 128) into float {-1,0,+1}.
    /// Within each 128-element block, byte at <c>gp</c> holds elements {gp, +32, +64, +96}
    /// at bit offsets {6,4,2,0}; code value maps via <c>(code - 1)</c>.
    ///
    /// <para><b>Not AVX2-vectorized (issue #128):</b> this path is only reached when
    /// <see cref="I2SUseW2A8"/> is <c>false</c> (i.e. <see cref="Avx2.IsSupported"/> is already
    /// <c>false</c> on the running box, per <see cref="GemvI2_SCore"/> / <see cref="GemmI2_SCore"/>),
    /// so an <c>Avx2</c>-gated fast path here would be dead code on every machine that reaches it.
    /// The hot GEMV-decode path (<see cref="UnpackRowI8"/>, used by the W2A8 SIMD tier) is the one
    /// profiled and vectorized — see its doc comment for the measured before/after.</para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackRow(byte* rowPtr, Span<float> dest, int k)
    {
        int blocks = k / I2SBlockSize;
        for (int blk = 0; blk < blocks; blk++)
        {
            byte* bp = rowPtr + blk * 32;
            int outBase = blk * I2SBlockSize;
            for (int gp = 0; gp < 32; gp++)
            {
                byte packed = bp[gp];
                dest[outBase + gp] = ((packed >> 6) & 0x3) - 1;
                dest[outBase + gp + 32] = ((packed >> 4) & 0x3) - 1;
                dest[outBase + gp + 64] = ((packed >> 2) & 0x3) - 1;
                dest[outBase + gp + 96] = (packed & 0x3) - 1;
            }
        }
    }

    /// <summary>
    /// Unpacks one I2_S-packed weight row (K codes, K a multiple of 128) into int8 ternary
    /// {-1,0,+1}, laid out contiguously so that each 32-element slice aligns with a Q8_0 block.
    /// Same bit layout as <see cref="UnpackRow"/>: within a 128-element block, byte at <c>gp</c>
    /// holds elements {gp, +32, +64, +96} at bit offsets {6,4,2,0}; ternary = <c>code - 1</c>.
    ///
    /// <para><b>Issue #128 (GEMV-decode unpack vectorization).</b> This is the row-unpack used by
    /// the W2A8 SIMD tier (<see cref="GemvI2_SW2A8Rows"/>/<see cref="GemmI2_SW2A8Rows"/>), which is
    /// only reachable when <see cref="Avx2.IsSupported"/> is <c>true</c> — so the fast path below is
    /// always live whenever this method runs. Profiling on real BitNet-shaped rows
    /// (m=6912,k=2560 and m=2560,k=6912, Strix Halo / Zen 5 AVX2+AVX-VNNI) showed unpack at 80–84%
    /// of total <c>GemvI2_S</c> wall time for the GEMV/decode path (unamortized — a fresh unpack per
    /// call), which is well above the vectorization threshold; the GEMM/prefill path amortizes this
    /// cost across N tokens and was not the target.</para>
    ///
    /// <para>Vectorized via <see cref="Avx2"/>: each 32-byte packed chunk (one 128-element block)
    /// is zero-extended byte→int16 (<c>Avx2.ConvertToVector256Int16</c>, 2× to cover all 32 bytes),
    /// then for each of the 4 bit offsets {6,4,2,0} the 2-bit field is extracted via
    /// <c>Avx2.ShiftRightLogical</c> + AND 0x3, ternary-mapped via subtract-1 (all done in int16 to
    /// avoid byte-lane shift limitations — AVX2 has no per-byte variable/immediate shift), then
    /// narrowed back to int8 via <c>Avx2.PackSignedSaturate</c> + the standard
    /// <c>Avx2.Permute4x64</c> (control <c>0xD8</c>) lane fix-up that undoes AVX2's cross-128-bit-lane
    /// pack ordering. Falls back to the scalar shift/mask loop on non-AVX2 hardware (defensive; in
    /// practice unreachable given the <see cref="I2SUseW2A8"/> gate above, but kept per the project's
    /// "always provide scalar fallback" convention).</para>
    /// </summary>
    [SkipLocalsInit]
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void UnpackRowI8(byte* rowPtr, sbyte* dest, int k)
    {
        int blocks = k / I2SBlockSize;

        if (Avx2.IsSupported)
        {
            for (int blk = 0; blk < blocks; blk++)
            {
                byte* bp = rowPtr + blk * 32;
                sbyte* outp = dest + blk * I2SBlockSize;

                Vector256<byte> packed = Unsafe.ReadUnaligned<Vector256<byte>>(bp);

                // Zero-extend all 32 packed bytes to int16 lanes (two 128-bit halves → two
                // full 256-bit int16 vectors of 16 lanes each: w0 = bytes[0..15], w1 = bytes[16..31]).
                Vector256<short> w0 = Avx2.ConvertToVector256Int16(packed.GetLower());
                Vector256<short> w1 = Avx2.ConvertToVector256Int16(packed.GetUpper());

                // gp, gp+32, gp+64, gp+96 ← bit offsets 6, 4, 2, 0 (literal shift counts — AVX2's
                // ShiftRightLogical immediate form requires a JIT-visible constant per callsite).
                UnpackI2SField6(w0, w1, outp);
                UnpackI2SField4(w0, w1, outp + 32);
                UnpackI2SField2(w0, w1, outp + 64);
                UnpackI2SField0(w0, w1, outp + 96);
            }
            return;
        }

        for (int blk = 0; blk < blocks; blk++)
        {
            byte* bp = rowPtr + blk * 32;
            int outBase = blk * I2SBlockSize;
            for (int gp = 0; gp < 32; gp++)
            {
                byte packed = bp[gp];
                dest[outBase + gp] = (sbyte)(((packed >> 6) & 0x3) - 1);
                dest[outBase + gp + 32] = (sbyte)(((packed >> 4) & 0x3) - 1);
                dest[outBase + gp + 64] = (sbyte)(((packed >> 2) & 0x3) - 1);
                dest[outBase + gp + 96] = (sbyte)((packed & 0x3) - 1);
            }
        }
    }

    /// <summary>Extracts the bit-6 2-bit code field. See <see cref="UnpackRowI8"/> for the shared algorithm.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackI2SField6(Vector256<short> w0, Vector256<short> w1, sbyte* outp)
        => UnpackI2SFieldCore(Avx2.ShiftRightLogical(w0, 6), Avx2.ShiftRightLogical(w1, 6), outp);

    /// <summary>Extracts the bit-4 2-bit code field. See <see cref="UnpackRowI8"/> for the shared algorithm.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackI2SField4(Vector256<short> w0, Vector256<short> w1, sbyte* outp)
        => UnpackI2SFieldCore(Avx2.ShiftRightLogical(w0, 4), Avx2.ShiftRightLogical(w1, 4), outp);

    /// <summary>Extracts the bit-2 2-bit code field. See <see cref="UnpackRowI8"/> for the shared algorithm.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackI2SField2(Vector256<short> w0, Vector256<short> w1, sbyte* outp)
        => UnpackI2SFieldCore(Avx2.ShiftRightLogical(w0, 2), Avx2.ShiftRightLogical(w1, 2), outp);

    /// <summary>Extracts the bit-0 2-bit code field (no shift needed). See <see cref="UnpackRowI8"/> for the shared algorithm.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackI2SField0(Vector256<short> w0, Vector256<short> w1, sbyte* outp)
        => UnpackI2SFieldCore(w0, w1, outp);

    /// <summary>
    /// Common tail of the 2-bit-field extraction: mask to 2 bits, ternary-map (<c>code - 1</c>),
    /// narrow int16→int8, and write the 32 resulting bytes to <paramref name="outp"/> (matching
    /// the original byte order — see <see cref="UnpackRowI8"/> for the pack/permute lane fix-up
    /// rationale). <paramref name="shifted0"/>/<paramref name="shifted1"/> are the already
    /// bit-shifted int16 lanes for bytes[0..15] / bytes[16..31] respectively.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void UnpackI2SFieldCore(Vector256<short> shifted0, Vector256<short> shifted1, sbyte* outp)
    {
        Vector256<short> three = Vector256.Create((short)3);
        Vector256<short> one = Vector256.Create((short)1);

        Vector256<short> c0 = Avx2.Subtract(Avx2.And(shifted0, three), one);
        Vector256<short> c1 = Avx2.Subtract(Avx2.And(shifted1, three), one);

        // Narrow int16 → int8 (values are in [-1,2], well within sbyte range — no saturation).
        // PackSignedSaturate interleaves 128-bit halves across the 256-bit result (AVX2 lane
        // quirk); Permute4x64(0xD8) restores sequential byte order: bytes[0..15] then [16..31].
        Vector256<sbyte> packed = Avx2.PackSignedSaturate(c0, c1);
        Vector256<sbyte> ordered = Avx2.Permute4x64(packed.AsInt64(), 0xD8).AsSByte();

        Unsafe.WriteUnaligned(outp, ordered);
    }

    // ─────────────────────────── Indexed (MoE) I2_S matmul ───────────────────────────

    /// <summary>
    /// Indexed ternary (I2_S) Mixture-of-Experts matmul — <c>moe_indexed_matmul_i2_s</c>.
    /// For each input row <c>t</c> (a token·top-k-slot assignment), computes
    /// <c>C[t, r] = expertScales[e] · dot(ternary(W_e[r, :]), B[t, :])</c> where
    /// <c>e = rowExpertIds[t]</c> selects the expert.
    ///
    /// <para>This fuses two existing CPU kernels: (a) the per-expert weight-base-offset
    /// addressing used by the grouped CPU MoE FFN path (<c>ProcessRoutedExpert</c> →
    /// <c>base + e·rowBytes</c>), and (b) the trit-unpack + per-tensor-α dequant inner loop
    /// of the dense I2_S GEMM (<see cref="GemmI2_S(byte*, float*, float*, int, int, int, float, ComputeThreadPool?)"/>).
    /// The per-expert scale is taken from the <paramref name="expertScales"/> vector (indexed
    /// by expert id), so the expert weight banks store <b>packed trits only</b> — no inline
    /// tail scale.</para>
    ///
    /// <para>Rows are grouped by expert (counting sort, the same dtype-agnostic idiom the MoE
    /// router uses for bucketing); each touched expert then runs one batched I2_S GEMM over its
    /// gathered rows, so each weight row is trit-unpacked exactly once per expert and reused
    /// across that expert's batch. Output is written in original row order.</para>
    /// </summary>
    /// <param name="expertWeights">Base pointer of the packed I2_S expert banks. Expert <c>e</c>'s
    /// packed payload (<c>m·k/4</c> bytes) lives at <c>expertWeights + e·expertRowBytes</c>.</param>
    /// <param name="expertRowBytes">Byte stride between consecutive expert banks (≥ <c>m·k/4</c>).</param>
    /// <param name="expertScales">Per-expert α scale, indexed by expert id. Must cover every id in <paramref name="rowExpertIds"/>.</param>
    /// <param name="b">F32 activation rows [n × k], row-major.</param>
    /// <param name="c">F32 output [n × m], row-major (<c>c[t·m + r]</c>). Fully overwritten for routed rows.</param>
    /// <param name="m">Output features per expert (weight rows).</param>
    /// <param name="k">Input dimension (a multiple of 128).</param>
    /// <param name="n">Number of input rows (token·slot assignments).</param>
    /// <param name="rowExpertIds">Length-<paramref name="n"/> expert id assigned to each input row.</param>
    /// <param name="threadPool">Optional thread pool — forwarded to the per-expert GEMM.</param>
    /// <remarks>
    /// <paramref name="k"/> need not be a multiple of 128 (issue #206) — the per-expert batched
    /// GEMM below (<see cref="GemmI2_S(byte*, float*, float*, int, int, int, float, ComputeThreadPool?)"/>)
    /// already dispatches to the ragged-safe path when <c>k % 128 != 0</c>, so no guard is needed here.
    /// </remarks>
    [SkipLocalsInit]
    public static void MoeIndexedMatmulI2_S(
        byte* expertWeights, long expertRowBytes, ReadOnlySpan<float> expertScales,
        float* b, float* c, int m, int k, int n,
        ReadOnlySpan<int> rowExpertIds, ComputeThreadPool? threadPool)
    {
        if (rowExpertIds.Length < n)
            throw new ArgumentException("rowExpertIds too small", nameof(rowExpertIds));
        if (n == 0) return;

        // ── Group rows by expert (counting sort) ───────────────────────────────
        // Determine the expert-id range so we can size the histogram tightly.
        int maxExpert = 0;
        for (int t = 0; t < n; t++)
        {
            int e = rowExpertIds[t];
            if (e < 0) throw new ArgumentException($"rowExpertIds[{t}] is negative", nameof(rowExpertIds));
            if (e > maxExpert) maxExpert = e;
        }
        int numExperts = maxExpert + 1;
        if (expertScales.Length < numExperts)
            throw new ArgumentException("expertScales too small for the routed expert ids", nameof(expertScales));

        // cursors[e..e+1) delimits expert e's rows inside groupedRows after the scan.
        int[] cursorsBuf = ArrayPool<int>.Shared.Rent(numExperts + 1);
        int[] groupedRowsBuf = ArrayPool<int>.Shared.Rent(n);
        // Per-expert batched activation / output scratch — sized for the full batch (worst
        // case: all rows route to one expert).
        float[] batchInBuf = ArrayPool<float>.Shared.Rent(n * k);
        float[] batchOutBuf = ArrayPool<float>.Shared.Rent(n * m);

        try
        {
            Span<int> cursors = cursorsBuf.AsSpan(0, numExperts + 1);
            cursors.Clear();

            // Histogram.
            for (int t = 0; t < n; t++) cursors[rowExpertIds[t]]++;

            // Exclusive prefix sum → bucket offsets.
            int running = 0;
            for (int e = 0; e <= numExperts; e++)
            {
                int cnt = cursors[e];
                cursors[e] = running;
                running += cnt;
            }

            // Scatter row indices into per-expert contiguous groups using write cursors.
            int[] writeCursorBuf = ArrayPool<int>.Shared.Rent(numExperts);
            try
            {
                Span<int> writeCursor = writeCursorBuf.AsSpan(0, numExperts);
                for (int e = 0; e < numExperts; e++) writeCursor[e] = cursors[e];
                for (int t = 0; t < n; t++)
                {
                    int e = rowExpertIds[t];
                    groupedRowsBuf[writeCursor[e]++] = t;
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(writeCursorBuf);
            }

            // ── Per-expert batched I2_S GEMM (touched experts only) ─────────────
            fixed (float* batchInPtr = batchInBuf)
            fixed (float* batchOutPtr = batchOutBuf)
            {
                for (int e = 0; e < numExperts; e++)
                {
                    int start = cursors[e];
                    int end = cursors[e + 1];
                    int batch = end - start;
                    if (batch == 0) continue;

                    // Gather this expert's rows into a contiguous batch [batch × k].
                    for (int bi = 0; bi < batch; bi++)
                    {
                        int t = groupedRowsBuf[start + bi];
                        Buffer.MemoryCopy(
                            b + (long)t * k,
                            batchInPtr + (long)bi * k,
                            (long)k * sizeof(float),
                            (long)k * sizeof(float));
                    }

                    // One batched ternary GEMM with this expert's bank + α.
                    byte* bank = expertWeights + (nint)(e * expertRowBytes);
                    GemmI2_S(bank, batchInPtr, batchOutPtr, m, k, batch, expertScales[e], threadPool);

                    // Scatter output rows back to original positions [n × m].
                    for (int bi = 0; bi < batch; bi++)
                    {
                        int t = groupedRowsBuf[start + bi];
                        Buffer.MemoryCopy(
                            batchOutPtr + (long)bi * m,
                            c + (long)t * m,
                            (long)m * sizeof(float),
                            (long)m * sizeof(float));
                    }
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(cursorsBuf);
            ArrayPool<int>.Shared.Return(groupedRowsBuf);
            ArrayPool<float>.Shared.Return(batchInBuf);
            ArrayPool<float>.Shared.Return(batchOutBuf);
        }
    }
}
