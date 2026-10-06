using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Cpu.Threading;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Scaled dot-product attention kernel with GQA head broadcast and causal masking.
/// Handles MHA, MQA, and GQA via <c>groupSize = numHeads / numKvHeads</c>.
/// <para>
/// For each query head <c>h</c>, the corresponding KV head is <c>h / groupSize</c>.
/// Attention is computed as: <c>softmax((Q @ K^T) / sqrt(headDim) + causalMask) @ V</c>.
/// </para>
/// </summary>
public static class Attention
{
    /// <summary>Stackalloc threshold in bytes. Above this, use ArrayPool.</summary>
    private const int StackAllocThreshold = 8192;

    /// <summary>Maximum tile size for tiled attention. Used for constant-size stackalloc.</summary>
    private const int MaxTileSize = 256;

    /// <summary>
    /// Largest visible-key count a query row may have and still take the single-shot (non-tiled)
    /// path. Expressed in floats from <see cref="StackAllocThreshold"/> so a decode step over a
    /// context of this length keeps exactly the numerics it had before #525.
    /// </summary>
    private const int OneShotMaxKeys = StackAllocThreshold / sizeof(float);

    /// <summary>
    /// Computes scaled dot-product attention with causal masking and GQA head broadcast.
    /// Convenience overload that computes <c>scale = 1/sqrt(headDim)</c>.
    /// </summary>
    /// <param name="q">Query tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Value tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">Output tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Number of query positions (tokens being generated).</param>
    /// <param name="seqKv">Number of key/value positions (total context length).</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="positionOffset">Position offset for causal mask. For prefill: 0. For decode: number of cached tokens.</param>
    /// <param name="slidingWindowSize">Optional sliding window size. When non-null, limits attention to the most recent positions.</param>
    /// <param name="softCap">Optional Gemma-2/3 style soft-cap on raw scores. When &gt; 0, raw scores
    /// pass through <c>softCap * tanh(s / softCap)</c> before softmax. Default 0 = disabled. Mirrors the
    /// Vulkan <c>attention_flash_f32.comp</c> convention.</param>
    /// <param name="sinks">Optional per-head attention-sink logits [numHeads] (gpt-oss). Empty = no sinks.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset, int? slidingWindowSize = null, float softCap = 0f,
                                ReadOnlySpan<float> sinks = default)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim), default, slidingWindowSize, softCap,
                   AttentionMaskMode.Causal, 0, sinks);

    /// <summary>
    /// Computes scaled dot-product attention with causal masking, GQA head broadcast, and ALiBi.
    /// </summary>
    /// <param name="q">Query tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Value tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">Output tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Number of query positions.</param>
    /// <param name="seqKv">Number of key/value positions.</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="positionOffset">Position offset for causal mask.</param>
    /// <param name="alibiSlopes">Per-query-head slopes. Length must be at least <paramref name="numHeads"/>.</param>
    /// <param name="slidingWindowSize">Optional sliding window size.</param>
    /// <param name="softCap">Optional Gemma-2/3 style soft-cap on raw scores. Default 0 = disabled.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset, ReadOnlySpan<float> alibiSlopes,
                                int? slidingWindowSize = null, float softCap = 0f)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim), alibiSlopes, slidingWindowSize, softCap);

    /// <summary>
    /// Computes scaled dot-product attention with causal masking, GQA head broadcast, and caller-provided scale.
    /// </summary>
    /// <param name="q">Query tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="k">Key tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="v">Value tensor. Layout: <c>[seqKv, numKvHeads * headDim]</c>.</param>
    /// <param name="output">Output tensor. Layout: <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Number of query positions (tokens being generated).</param>
    /// <param name="seqKv">Number of key/value positions (total context length).</param>
    /// <param name="numHeads">Number of query attention heads.</param>
    /// <param name="numKvHeads">Number of key/value heads.</param>
    /// <param name="headDim">Dimension per attention head.</param>
    /// <param name="positionOffset">Position offset for causal mask. For prefill: 0. For decode: number of cached tokens.</param>
    /// <param name="scale">Attention scale factor applied to dot-product scores.</param>
    /// <param name="slidingWindowSize">Optional sliding window size. When non-null, limits attention to the most recent positions.</param>
    /// <param name="softCap">Optional Gemma-2/3 style soft-cap on raw scores. Default 0 = disabled.</param>
    /// <param name="sinks">Optional per-head attention-sink logits [numHeads] (gpt-oss). When
    /// non-empty, each head's softmax denominator additionally includes <c>exp(sink[h] - max)</c>
    /// — a virtual key receiving probability mass but contributing nothing to the output.</param>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset, float scale, int? slidingWindowSize = null, float softCap = 0f,
                                ReadOnlySpan<float> sinks = default)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, scale, default, slidingWindowSize, softCap,
                   AttentionMaskMode.Causal, 0, sinks);

    /// <summary>
    /// Computes scaled dot-product attention with causal masking, GQA head broadcast, caller-provided scale, and ALiBi.
    /// </summary>
    [SkipLocalsInit]
    public static void Execute(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                Span<float> output,
                                int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                int positionOffset, float scale, ReadOnlySpan<float> alibiSlopes,
                                int? slidingWindowSize = null, float softCap = 0f,
                                AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0,
                                ReadOnlySpan<float> sinks = default)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));
        ValidateAlibiSlopes(alibiSlopes, numHeads);
        if (!sinks.IsEmpty && sinks.Length < numHeads)
            throw new ArgumentException(
                $"sinks must have at least numHeads ({numHeads}) entries, got {sinks.Length}", nameof(sinks));

        int groupSize = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;

        int tileSize = ComputeTileSize(headDim);
        Span<float> scratch = stackalloc float[OneShotMaxKeys];

        for (int h = 0; h < numHeads; h++)
        {
            ExecuteHead(q, k, v, output, scratch, seqQ, seqKv, headDim, scale,
                        qStride, kvStride, positionOffset, tileSize, slidingWindowSize ?? 0,
                        h, h / groupSize, GetAlibiSlope(alibiSlopes, h), softCap,
                        maskMode, prefixLen,
                        sinks.IsEmpty ? float.NegativeInfinity : sinks[h]);
        }
    }

    /// <summary>
    /// Computes attention for a single (query head, KV head) pair, one query row at a time.
    /// <para>
    /// Every reduction is confined to the row's visible key range <c>[visibleStart, visibleEnd)</c>,
    /// which depends only on the query's own position, the mask mode and the sliding window — never
    /// on how many keys happen to be resident in the KV cache. That invariance is the point (#525):
    /// the previous implementation reduced each softmax row over the padded <c>seqKv</c>, and although
    /// <c>-inf</c> padding contributes exactly <c>0.0</c> mathematically, changing the span length
    /// changes SIMD lane assignment and the remainder tail, so the *real* elements accumulate in a
    /// different order. The resulting ULP moved with the KV-cache length, which on a quantized model
    /// is digitized by activation quantization and can change the emitted token. Skipping the padding
    /// is also strictly less work.
    /// </para>
    /// <para>
    /// The per-row dispatch between the single-shot and tiled (online-softmax) forms is made on the
    /// row's own <paramref name="seqKv"/>-independent visible length for the same reason: a per-call
    /// <c>seqQ * seqKv</c> decision would make the arithmetic formula itself depend on cache length.
    /// </para>
    /// </summary>
    private static void ExecuteHead(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                    Span<float> output, Span<float> scratch,
                                    int seqQ, int seqKv, int headDim, float scale,
                                    int qStride, int kvStride, int positionOffset, int tileSize,
                                    int slidingWindowSize, int headIdx, int kvHeadIdx, float alibiSlope,
                                    float softCap, AttentionMaskMode maskMode, int prefixLen,
                                    float sinkLogit)
    {
        bool hasSink = !float.IsNegativeInfinity(sinkLogit);

        for (int i = 0; i < seqQ; i++)
        {
            int qPos = positionOffset + i;
            var qRow = q.Slice(i * qStride + headIdx * headDim, headDim);
            var outRow = output.Slice(i * qStride + headIdx * headDim, headDim);
            outRow.Clear();

            int visibleEnd = VisibleEnd(maskMode, qPos, prefixLen, seqKv);
            int visibleStart = VisibleStart(qPos, slidingWindowSize);
            if (visibleStart >= visibleEnd)
                continue;

            int visibleLen = visibleEnd - visibleStart;

            if (visibleLen <= OneShotMaxKeys)
            {
                OneShotRow(k, v, qRow, outRow, scratch.Slice(0, visibleLen),
                           qPos, visibleStart, headDim, scale, kvStride, kvHeadIdx,
                           alibiSlope, softCap, hasSink, sinkLogit);
            }
            else
            {
                TiledRow(k, v, qRow, outRow, scratch.Slice(0, tileSize),
                         qPos, visibleStart, visibleEnd, headDim, scale, kvStride, kvHeadIdx,
                         tileSize, alibiSlope, softCap, hasSink, sinkLogit);
            }
        }
    }

    /// <summary>
    /// Single-shot attention for one query row: materialize the visible score row, mask-free
    /// (out-of-range keys are never scored), softmax it, then accumulate the weighted values.
    /// </summary>
    private static void OneShotRow(ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                   ReadOnlySpan<float> qRow, Span<float> outRow, Span<float> scores,
                                   int qPos, int visibleStart, int headDim, float scale, int kvStride,
                                   int kvHeadIdx, float alibiSlope, float softCap,
                                   bool hasSink, float sinkLogit)
    {
        int n = scores.Length;

        for (int j = 0; j < n; j++)
        {
            var kRow = k.Slice((visibleStart + j) * kvStride + kvHeadIdx * headDim, headDim);
            scores[j] = TensorPrimitives.Dot(qRow, kRow) * scale;
        }

        if (alibiSlope != 0f)
        {
            for (int j = 0; j < n; j++)
                scores[j] -= alibiSlope * (qPos - (visibleStart + j));
        }

        // Optional Gemma 2/3 attention-logit soft-cap, on raw scores, before softmax.
        if (softCap > 0f)
            ApplySoftCap(scores, softCap);

        // Fused shift+exp+store+sum softmax. The exp is precise since #501 (DOTLLM_FAST_EXP=1
        // restores the Schraudolph approximation, opt-in and CPU-only).
        if (hasSink)
            SoftmaxRowWithSink(scores, sinkLogit);
        else
            Softmax.ExecuteFast(scores, scores);

        for (int j = 0; j < n; j++)
        {
            float w = scores[j];
            if (w == 0f) continue;
            var vRow = v.Slice((visibleStart + j) * kvStride + kvHeadIdx * headDim, headDim);
            TensorPrimitives.MultiplyAdd(vRow, w, outRow, outRow);
        }
    }

    /// <summary>
    /// Tiled (online-softmax) attention for one query row. Used when the visible range is too long
    /// to materialize; tiles start at <paramref name="visibleStart"/> so tile boundaries — and hence
    /// every reduction — are a function of the query position alone.
    /// </summary>
    private static void TiledRow(ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                 ReadOnlySpan<float> qRow, Span<float> outRow, Span<float> tileScores,
                                 int qPos, int visibleStart, int visibleEnd, int headDim, float scale,
                                 int kvStride, int kvHeadIdx, int tileSize, float alibiSlope,
                                 float softCap, bool hasSink, float sinkLogit)
    {
        // Attention sink (gpt-oss): seed the online softmax as if a virtual key
        // with logit sinkLogit (and zero value vector) had already been processed.
        float maxSoFar = hasSink ? sinkLogit : float.NegativeInfinity;
        float sumExp = hasSink ? 1f : 0f;

        for (int tileBase = visibleStart; tileBase < visibleEnd; tileBase += tileSize)
        {
            int tileLen = Math.Min(tileSize, visibleEnd - tileBase);
            var scores = tileScores.Slice(0, tileLen);

            for (int j = 0; j < tileLen; j++)
            {
                var kRow = k.Slice((tileBase + j) * kvStride + kvHeadIdx * headDim, headDim);
                int keyPosition = tileBase + j;
                scores[j] = TensorPrimitives.Dot(qRow, kRow) * scale
                    - alibiSlope * (qPos - keyPosition);
            }

            // Optional Gemma 2/3 soft-cap on raw scores, before softmax (mirrors
            // attention_flash_f32.comp convention).
            if (softCap > 0f)
                ApplySoftCap(scores, softCap);

            float tileMax = TensorPrimitives.Max(scores);
            float newMax = MathF.Max(maxSoFar, tileMax);
            float correction = FastMath.FastExp(maxSoFar - newMax);

            if (correction < 1f)
            {
                sumExp *= correction;
                TensorPrimitives.Multiply(outRow, correction, outRow);
            }

            sumExp += FastMath.ExpSumAndStore(scores, scores, -newMax);

            for (int j = 0; j < tileLen; j++)
            {
                float w = scores[j];
                if (w == 0f) continue;
                var vRow = v.Slice((tileBase + j) * kvStride + kvHeadIdx * headDim, headDim);
                TensorPrimitives.MultiplyAdd(vRow, w, outRow, outRow);
            }

            maxSoFar = newMax;
        }

        if (sumExp > 0f)
            TensorPrimitives.Multiply(outRow, 1f / sumExp, outRow);
    }

    /// <summary>
    /// In-place softmax over <paramref name="row"/> whose denominator includes an
    /// extra sink logit: <c>p_j = exp(x_j - m) / (Σ exp(x_i - m) + exp(sink - m))</c>
    /// with <c>m = max(max(x), sink)</c>. Matches llama.cpp's
    /// <c>ggml_soft_max_add_sinks</c> semantics — the sink absorbs probability
    /// mass but contributes no value vector.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void SoftmaxRowWithSink(Span<float> row, float sink)
    {
        float max = TensorPrimitives.Max((ReadOnlySpan<float>)row);
        float m = MathF.Max(max, sink);
        TensorPrimitives.Add((ReadOnlySpan<float>)row, -m, row);
        TensorPrimitives.Exp((ReadOnlySpan<float>)row, row);
        float sum = TensorPrimitives.Sum((ReadOnlySpan<float>)row) + MathF.Exp(sink - m);
        TensorPrimitives.Multiply((ReadOnlySpan<float>)row, 1f / sum, row);
    }

    /// <summary>
    /// Computes KV tile size targeting L2 cache residency.
    /// Each tile loads Tc K vectors + Tc V vectors of headDim floats.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int ComputeTileSize(int headDim)
    {
        const int L2Budget = 256 * 1024; // 256 KB conservative L2 estimate
        int bytesPerKvToken = headDim * sizeof(float) * 2; // K + V
        int tc = L2Budget / bytesPerKvToken;
        return Math.Clamp(tc, 64, MaxTileSize);
    }

    // ──────────────────── Parallel overloads ────────────────────

    /// <summary>
    /// Pointer-based attention with optional head-parallel execution via <paramref name="pool"/>.
    /// When pool is null or numHeads &lt; 2, falls back to the single-threaded span-based path.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, float* k, float* v, float* output,
                                      int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, ComputeThreadPool? pool,
                                      int? slidingWindowSize = null, float softCap = 0f,
                                      float[]? sinks = null)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim), pool, slidingWindowSize, softCap,
                   AttentionMaskMode.Causal, 0, sinks);

    /// <summary>
    /// Pointer-based attention with optional head-parallel execution and ALiBi.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, float* k, float* v, float* output,
                                      int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, float* alibiSlopes, ComputeThreadPool? pool,
                                      int? slidingWindowSize = null, float softCap = 0f)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, 1.0f / MathF.Sqrt(headDim), alibiSlopes, pool, slidingWindowSize, softCap);

    /// <summary>
    /// Pointer-based attention with caller-provided scale and optional head-parallel execution.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, float* k, float* v, float* output,
                                      int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, float scale, ComputeThreadPool? pool,
                                      int? slidingWindowSize = null, float softCap = 0f,
                                      AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0,
                                      float[]? sinks = null)
        => Execute(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                   positionOffset, scale, null, pool, slidingWindowSize, softCap, maskMode, prefixLen, sinks);

    /// <summary>
    /// Pointer-based attention with caller-provided scale, optional head-parallel execution, and ALiBi.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, float* k, float* v, float* output,
                                      int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                      int positionOffset, float scale, float* alibiSlopes,
                                      ComputeThreadPool? pool, int? slidingWindowSize = null,
                                      float softCap = 0f,
                                      AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0,
                                      float[]? sinks = null)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));

        if (pool is null || numHeads < 2)
        {
            // Fall back to span-based single-threaded path
            int qLen = seqQ * numHeads * headDim;
            int kvLen = seqKv * numKvHeads * headDim;
            Execute(
                new ReadOnlySpan<float>(q, qLen),
                new ReadOnlySpan<float>(k, kvLen),
                new ReadOnlySpan<float>(v, kvLen),
                new Span<float>(output, qLen),
                seqQ, seqKv, numHeads, numKvHeads, headDim, positionOffset, scale,
                alibiSlopes is null ? default : new ReadOnlySpan<float>(alibiSlopes, numHeads),
                slidingWindowSize, softCap, maskMode, prefixLen,
                sinks is null ? ReadOnlySpan<float>.Empty : sinks);
            return;
        }

        fixed (float* sinksPtr = sinks)
        {
            var ctx = new AttentionCtx
            {
                Q = q, K = k, V = v, Output = output,
                SeqQ = seqQ, SeqKv = seqKv, NumHeads = numHeads, NumKvHeads = numKvHeads,
                HeadDim = headDim, Scale = scale, PositionOffset = positionOffset,
                GroupSize = numHeads / numKvHeads,
                QStride = numHeads * headDim,
                KvStride = numKvHeads * headDim,
                TileSize = ComputeTileSize(headDim),
                SlidingWindowSize = slidingWindowSize ?? 0,
                AlibiSlopes = alibiSlopes,
                SoftCap = softCap,
                MaskMode = maskMode,
                PrefixLen = prefixLen,
                Sinks = sinksPtr
            };
            pool.Dispatch((nint)(&ctx), &AttentionWorker);
        }
    }

    private unsafe struct AttentionCtx
    {
        public float* Q;
        public float* K;
        public float* V;
        public float* Output;
        public int SeqQ;
        public int SeqKv;
        public int NumHeads;
        public int NumKvHeads;
        public int HeadDim;
        public float Scale;
        public int PositionOffset;
        public int GroupSize;
        public int QStride;
        public int KvStride;
        public int TileSize;
        /// <summary>Sliding window size. 0 means no sliding window (full context).</summary>
        public int SlidingWindowSize;
        public float* AlibiSlopes;
        /// <summary>Gemma 2/3 attention-logit soft-cap. 0 = disabled.</summary>
        public float SoftCap;
        /// <summary>Attention mask mode. Causal (default) preserves the original fast path.</summary>
        public AttentionMaskMode MaskMode;
        /// <summary>Causal-prefix length for <see cref="AttentionMaskMode.Hybrid"/>.</summary>
        public int PrefixLen;
        /// <summary>Optional per-head sink logits [NumHeads] (gpt-oss). Null = no sinks.</summary>
        public float* Sinks;
    }

    [SkipLocalsInit]
    private static unsafe void AttentionWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<AttentionCtx>((void*)ctxPtr);

        // Partition heads across threads
        ComputeThreadPool.PartitionRange(ctx.NumHeads, threadIdx, threadCount, out int startHead, out int endHead);
        if (startHead >= ctx.NumHeads) return;

        var qSpan = new ReadOnlySpan<float>(ctx.Q, ctx.SeqQ * ctx.QStride);
        var kSpan = new ReadOnlySpan<float>(ctx.K, ctx.SeqKv * ctx.KvStride);
        var vSpan = new ReadOnlySpan<float>(ctx.V, ctx.SeqKv * ctx.KvStride);
        var outSpan = new Span<float>(ctx.Output, ctx.SeqQ * ctx.QStride);

        // Each worker gets its own row scratch — the same bound the single-threaded path uses.
        Span<float> scratch = stackalloc float[OneShotMaxKeys];

        for (int h = startHead; h < endHead; h++)
        {
            ExecuteHead(qSpan, kSpan, vSpan, outSpan, scratch,
                        ctx.SeqQ, ctx.SeqKv, ctx.HeadDim, ctx.Scale,
                        ctx.QStride, ctx.KvStride, ctx.PositionOffset, ctx.TileSize,
                        ctx.SlidingWindowSize, h, h / ctx.GroupSize,
                        ctx.AlibiSlopes is null ? 0f : ctx.AlibiSlopes[h],
                        ctx.SoftCap, ctx.MaskMode, ctx.PrefixLen,
                        ctx.Sinks is null ? float.NegativeInfinity : ctx.Sinks[h]);
        }
    }


    /// <summary>
    /// Scalar reference implementation for correctness verification.
    /// Convenience overload that computes <c>scale = 1/sqrt(headDim)</c>.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                        Span<float> output,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, int? slidingWindowSize = null, float softCap = 0f)
        => ExecuteScalar(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                         positionOffset, 1.0f / MathF.Sqrt(headDim), default, slidingWindowSize, softCap);

    /// <summary>
    /// Scalar reference implementation with ALiBi.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                        Span<float> output,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, ReadOnlySpan<float> alibiSlopes,
                                        int? slidingWindowSize = null, float softCap = 0f)
        => ExecuteScalar(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                         positionOffset, 1.0f / MathF.Sqrt(headDim), alibiSlopes, slidingWindowSize, softCap);

    /// <summary>
    /// Scalar reference implementation with caller-provided scale.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                        Span<float> output,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, float scale, int? slidingWindowSize = null,
                                        float softCap = 0f)
        => ExecuteScalar(q, k, v, output, seqQ, seqKv, numHeads, numKvHeads, headDim,
                         positionOffset, scale, default, slidingWindowSize, softCap);

    /// <summary>
    /// Scalar reference implementation with caller-provided scale and ALiBi.
    /// </summary>
    [SkipLocalsInit]
    internal static void ExecuteScalar(ReadOnlySpan<float> q, ReadOnlySpan<float> k, ReadOnlySpan<float> v,
                                        Span<float> output,
                                        int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                        int positionOffset, float scale, ReadOnlySpan<float> alibiSlopes,
                                        int? slidingWindowSize = null, float softCap = 0f,
                                        AttentionMaskMode maskMode = AttentionMaskMode.Causal, int prefixLen = 0,
                                        ReadOnlySpan<float> sinks = default)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));
        ValidateAlibiSlopes(alibiSlopes, numHeads);

        int groupSize = numHeads / numKvHeads;
        int qStride = numHeads * headDim;
        int kvStride = numKvHeads * headDim;

        float[] scores = new float[seqQ * seqKv];

        for (int h = 0; h < numHeads; h++)
        {
            int kvH = h / groupSize;

            // Scores: Q_h @ K_kvH^T
            for (int i = 0; i < seqQ; i++)
            {
                for (int j = 0; j < seqKv; j++)
                {
                    float dot = 0;
                    for (int d = 0; d < headDim; d++)
                        dot += q[i * qStride + h * headDim + d] * k[j * kvStride + kvH * headDim + d];
                    scores[i * seqKv + j] = dot * scale
                        - GetAlibiSlope(alibiSlopes, h) * (positionOffset + i - j);
                }
            }

            // Optional Gemma 2/3 attention-logit soft-cap (pre-mask, pre-softmax).
            if (softCap > 0f)
            {
                for (int i = 0; i < seqQ * seqKv; i++)
                    scores[i] = softCap * MathF.Tanh(scores[i] / softCap);
            }

            // Attention mask. Causal (default) reproduces the original j > positionOffset+i rule;
            // Bidirectional applies no upper bound; Hybrid is causal below prefixLen, open above.
            for (int i = 0; i < seqQ; i++)
            {
                int qPos = positionOffset + i;
                int upperExclusive = VisibleEnd(maskMode, qPos, prefixLen, seqKv);
                for (int j = upperExclusive; j < seqKv; j++)
                    scores[i * seqKv + j] = float.NegativeInfinity;
            }

            // Sliding window mask
            if (slidingWindowSize.HasValue)
            {
                int window = slidingWindowSize.Value;
                for (int i = 0; i < seqQ; i++)
                {
                    int earliestVisible = positionOffset + i - window + 1;
                    for (int j = 0; j < seqKv && j < earliestVisible; j++)
                    {
                        scores[i * seqKv + j] = float.NegativeInfinity;
                    }
                }
            }

            // Softmax per row (with optional gpt-oss sink logit joining the denominator)
            if (sinks.IsEmpty)
            {
                for (int i = 0; i < seqQ; i++)
                    Softmax.ExecuteScalar(scores.AsSpan(i * seqKv, seqKv), scores.AsSpan(i * seqKv, seqKv));
            }
            else
            {
                float sink = sinks[h];
                for (int i = 0; i < seqQ; i++)
                {
                    var row = scores.AsSpan(i * seqKv, seqKv);
                    float m = sink;
                    for (int j = 0; j < seqKv; j++) m = MathF.Max(m, row[j]);
                    float sum = MathF.Exp(sink - m);
                    for (int j = 0; j < seqKv; j++)
                    {
                        row[j] = MathF.Exp(row[j] - m);
                        sum += row[j];
                    }
                    float inv = 1f / sum;
                    for (int j = 0; j < seqKv; j++) row[j] *= inv;
                }
            }

            // Weighted values
            for (int i = 0; i < seqQ; i++)
            {
                for (int d = 0; d < headDim; d++)
                {
                    float sum = 0;
                    for (int j = 0; j < seqKv; j++)
                        sum += scores[i * seqKv + j] * v[j * kvStride + kvH * headDim + d];
                    output[i * qStride + h * headDim + d] = sum;
                }
            }
        }
    }

    /// <summary>
    /// Computes scaled dot-product scores: <c>scores[i,j] = (Q_h[i,:] · K_kvH[j,:]) * scale</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ScaledDotProductScores(ReadOnlySpan<float> q, ReadOnlySpan<float> k,
                                                 Span<float> scores,
                                                 int seqQ, int seqKv, int headDim, float scale,
                                                 int headIdx, int kvHeadIdx,
                                                 int qStride, int kvStride)
    {
        for (int i = 0; i < seqQ; i++)
        {
            var qRow = q.Slice(i * qStride + headIdx * headDim, headDim);
            for (int j = 0; j < seqKv; j++)
            {
                var kRow = k.Slice(j * kvStride + kvHeadIdx * headDim, headDim);
                scores[i * seqKv + j] = TensorPrimitives.Dot(qRow, kRow) * scale;
            }
        }
    }

    /// <summary>
    /// Adds ALiBi score bias in-place: <c>score[i,j] += -slope * (positionOffset + i - j)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ApplyAlibiBias(Span<float> scores, int seqQ, int seqKv, int positionOffset, float slope)
    {
        if (slope == 0f) return;

        for (int i = 0; i < seqQ; i++)
        {
            int queryPosition = positionOffset + i;
            for (int j = 0; j < seqKv; j++)
                scores[i * seqKv + j] -= slope * (queryPosition - j);
        }
    }

    private static void ValidateAlibiSlopes(ReadOnlySpan<float> alibiSlopes, int numHeads)
    {
        if (!alibiSlopes.IsEmpty && alibiSlopes.Length < numHeads)
            throw new ArgumentException(
                $"ALiBi slope count ({alibiSlopes.Length}) must be at least numHeads ({numHeads}).",
                nameof(alibiSlopes));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float GetAlibiSlope(ReadOnlySpan<float> alibiSlopes, int headIdx) =>
        alibiSlopes.IsEmpty ? 0f : alibiSlopes[headIdx];

    /// <summary>
    /// Applies the Gemma 2/3 attention-logit soft-cap in place: <c>s = softCap * tanh(s / softCap)</c>.
    /// Caller must ensure <paramref name="softCap"/> &gt; 0. Mirrors the Vulkan FA shader convention
    /// (<c>attention_flash_f32.comp</c>) — applied to raw scores (post scale and ALiBi, pre causal mask).
    /// Uses <see cref="TensorPrimitives"/> for SIMD-accelerated tanh.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ApplySoftCap(Span<float> scores, float softCap)
    {
        // s' = softCap * tanh(s / softCap).
        // SIMD path: divide-in-place, tanh-in-place, multiply-in-place. The TensorPrimitives
        // overloads accept aliased source/destination spans.
        float inv = 1.0f / softCap;
        TensorPrimitives.Multiply(scores, inv, scores);
        TensorPrimitives.Tanh(scores, scores);
        TensorPrimitives.Multiply(scores, softCap, scores);
    }

    /// <summary>
    /// Applies causal (autoregressive) mask. Sets <c>scores[i,j] = -inf</c> where
    /// <c>j &gt; positionOffset + i</c> (query at position <c>positionOffset + i</c>
    /// cannot attend to keys at later positions).
    /// Optionally applies a sliding window mask that limits attention to the most recent
    /// <paramref name="slidingWindowSize"/> positions.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ApplyCausalMask(Span<float> scores, int seqQ, int seqKv, int positionOffset,
                                         int? slidingWindowSize = null)
    {
        for (int i = 0; i < seqQ; i++)
        {
            for (int j = positionOffset + i + 1; j < seqKv; j++)
            {
                scores[i * seqKv + j] = float.NegativeInfinity;
            }
        }

        ApplySlidingWindowMask(scores, seqQ, seqKv, positionOffset, slidingWindowSize);
    }

    /// <summary>
    /// Applies the optional sliding-window lower bound in place: masks any key earlier than
    /// <c>positionOffset + i - window + 1</c>. Shared by every mask mode (the window limit composes
    /// with causal / bidirectional / hybrid alike). No-op when <paramref name="slidingWindowSize"/>
    /// is null.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ApplySlidingWindowMask(Span<float> scores, int seqQ, int seqKv,
                                               int positionOffset, int? slidingWindowSize)
    {
        if (!slidingWindowSize.HasValue) return;

        int window = slidingWindowSize.Value;
        for (int i = 0; i < seqQ; i++)
        {
            int earliestVisible = positionOffset + i - window + 1;
            for (int j = 0; j < seqKv && j < earliestVisible; j++)
            {
                scores[i * seqKv + j] = float.NegativeInfinity;
            }
        }
    }

    /// <summary>
    /// Applies the attention mask selected by <paramref name="mode"/>:
    /// <list type="bullet">
    /// <item><see cref="AttentionMaskMode.Causal"/> — identical to <see cref="ApplyCausalMask"/>
    /// (the byte-identical fast path).</item>
    /// <item><see cref="AttentionMaskMode.Bidirectional"/> — no causal upper bound; only the optional
    /// sliding window is applied.</item>
    /// <item><see cref="AttentionMaskMode.Hybrid"/> — query positions <c>&lt; prefixLen</c> stay causal;
    /// query positions <c>&gt;= prefixLen</c> attend to the full <c>[0, seqKv)</c> range.</item>
    /// </list>
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void ApplyMask(Span<float> scores, int seqQ, int seqKv, int positionOffset,
                                   AttentionMaskMode mode, int prefixLen, int? slidingWindowSize = null)
    {
        switch (mode)
        {
            case AttentionMaskMode.Causal:
                // Exact original fast path — preserved verbatim for byte-identical causal output.
                ApplyCausalMask(scores, seqQ, seqKv, positionOffset, slidingWindowSize);
                return;

            case AttentionMaskMode.Bidirectional:
                // No causal upper bound; sliding window (if any) still applies.
                ApplySlidingWindowMask(scores, seqQ, seqKv, positionOffset, slidingWindowSize);
                return;

            case AttentionMaskMode.Hybrid:
                // Causal prefix; bidirectional canvas. A prefix query (qPos < prefixLen) masks
                // future keys exactly like the causal path; a canvas query (qPos >= prefixLen)
                // attends to every key in [0, seqKv).
                for (int i = 0; i < seqQ; i++)
                {
                    int qPos = positionOffset + i;
                    if (qPos < prefixLen)
                    {
                        for (int j = qPos + 1; j < seqKv; j++)
                            scores[i * seqKv + j] = float.NegativeInfinity;
                    }
                }
                ApplySlidingWindowMask(scores, seqQ, seqKv, positionOffset, slidingWindowSize);
                return;
        }
    }

    /// <summary>
    /// Computes the per-query exclusive visible upper bound for the tiled / online-softmax paths.
    /// Causal: <c>min(seqKv, qPos+1)</c>. Bidirectional: <c>seqKv</c>. Hybrid: causal for prefix
    /// queries, <c>seqKv</c> for canvas queries.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int VisibleEnd(AttentionMaskMode mode, int qPos, int prefixLen, int seqKv)
    {
        switch (mode)
        {
            case AttentionMaskMode.Bidirectional:
                return seqKv;
            case AttentionMaskMode.Hybrid:
                return qPos < prefixLen ? Math.Min(seqKv, qPos + 1) : seqKv;
            default: // Causal
                return Math.Min(seqKv, qPos + 1);
        }
    }

    /// <summary>
    /// Computes the per-query inclusive visible lower bound. Without a sliding window this is 0;
    /// with one it is <c>max(0, qPos - window + 1)</c>. Independent of <c>seqKv</c> by construction.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int VisibleStart(int qPos, int slidingWindowSize)
        => slidingWindowSize > 0 ? Math.Max(0, qPos - slidingWindowSize + 1) : 0;

    /// <summary>
    /// Computes weighted sum: <c>output_h[i,:] = sum_j(weights[i,j] * V_kvH[j,:])</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void WeightedValues(ReadOnlySpan<float> weights, ReadOnlySpan<float> v,
                                         Span<float> output,
                                         int seqQ, int seqKv, int headDim,
                                         int headIdx, int kvHeadIdx,
                                         int qStride, int kvStride)
    {
        for (int i = 0; i < seqQ; i++)
        {
            var outSlice = output.Slice(i * qStride + headIdx * headDim, headDim);
            outSlice.Clear();

            for (int j = 0; j < seqKv; j++)
            {
                float w = weights[i * seqKv + j];
                // Exact 0f comparison is safe: softmax(exp(-∞ - max)) == 0f exactly in IEEE 754.
                // ApplyCausalMask writes float.NegativeInfinity, which exp() maps to exactly zero.
                if (w == 0f) continue;

                var vRow = v.Slice(j * kvStride + kvHeadIdx * headDim, headDim);
                TensorPrimitives.MultiplyAdd(vRow, w, outSlice, outSlice);
            }
        }
    }

    // ──────────────────── Quantized KV-cache attention ────────────────────

    /// <summary>
    /// Attention with quantized KV-cache. Dequantizes tiles on-the-fly during attention
    /// computation, then processes the full-precision window region directly.
    /// </summary>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, IQuantizedKvCache kvCache, int layerIndex,
                                       float* output,
                                       int seqQ, int seqKv, int numHeads, int numKvHeads, int headDim,
                                       int positionOffset, ComputeThreadPool? pool,
                                       int? slidingWindowSize = null, float softCap = 0f,
                                       float[]? sinks = null)
    {
        // softCap currently unused on the quantized KV-cache path — Gemma checkpoints
        // ship full-precision KV. When/if Gemma 2/3 quantized KV becomes a target the
        // online-softmax loops above need an ApplySoftCap call sandwiched between
        // score computation and the running-max update.
        _ = softCap;
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));

        float scale = 1.0f / MathF.Sqrt(headDim);
        int kvStride = numKvHeads * headDim;
        int qStride = numHeads * headDim;
        int tileSize = ComputeTileSize(headDim);

        int quantLen = kvCache.QuantizedLength;
        int windowLen = kvCache.WindowLength;
        byte* kQuant = (byte*)kvCache.GetQuantizedKeysPtr(layerIndex);
        byte* vQuant = (byte*)kvCache.GetQuantizedValuesPtr(layerIndex);
        float* kWindow = (float*)kvCache.GetWindowKeysPtr(layerIndex);
        float* vWindow = (float*)kvCache.GetWindowValuesPtr(layerIndex);
        // Per-layer quantized row bytes: equals the scalar property for every uniform
        // model; for Gemma-4 the sliding/global layers carry distinct widths. The
        // kvStride local above is already per-layer (derived from the per-call
        // numKvHeads/headDim the caller passes for this layer).
        int kQuantRowBytes = kvCache.KeyQuantizedRowBytesOf(layerIndex);
        int vQuantRowBytes = kvCache.ValueQuantizedRowBytesOf(layerIndex);

        if (pool is null || numHeads < 2)
        {
            Span<float> tileScores = stackalloc float[MaxTileSize];
            for (int h = 0; h < numHeads; h++)
            {
                ExecuteTiledQuantizedHead(
                    q, kQuant, vQuant, kWindow, vWindow, output, tileScores,
                    seqQ, quantLen, windowLen, headDim, scale,
                    qStride, kvStride, kQuantRowBytes, vQuantRowBytes,
                    positionOffset, tileSize, slidingWindowSize ?? 0,
                    kvCache.KeyDType, kvCache.ValueDType,
                    h, h / (numHeads / numKvHeads),
                    kvCache.WindowCapacity,
                    sinks is null ? float.NegativeInfinity : sinks[h]);
            }
        }
        else
        {
            fixed (float* sinksPtr = sinks)
            {
                var ctx = new QuantizedTiledCtx
                {
                    Q = q, KQuant = kQuant, VQuant = vQuant,
                    KWindow = kWindow, VWindow = vWindow, Output = output,
                    SeqQ = seqQ, QuantLen = quantLen, WindowLen = windowLen,
                    NumHeads = numHeads, NumKvHeads = numKvHeads,
                    HeadDim = headDim, Scale = scale,
                    PositionOffset = positionOffset,
                    GroupSize = numHeads / numKvHeads,
                    QStride = qStride, KvStride = kvStride,
                    KQuantRowBytes = kQuantRowBytes, VQuantRowBytes = vQuantRowBytes,
                    TileSize = tileSize,
                    SlidingWindowSize = slidingWindowSize ?? 0,
                    KeyDType = kvCache.KeyDType, ValueDType = kvCache.ValueDType,
                    WindowCapacity = kvCache.WindowCapacity,
                    Sinks = sinksPtr
                };
                pool.Dispatch((nint)(&ctx), &QuantizedTiledAttentionWorker);
            }
        }
    }

    private unsafe struct QuantizedTiledCtx
    {
        public float* Q;
        public byte* KQuant;
        public byte* VQuant;
        public float* KWindow;
        public float* VWindow;
        public float* Output;
        public int SeqQ;
        public int QuantLen;
        public int WindowLen;
        public int NumHeads;
        public int NumKvHeads;
        public int HeadDim;
        public float Scale;
        public int PositionOffset;
        public int GroupSize;
        public int QStride;
        public int KvStride;
        public int KQuantRowBytes;
        public int VQuantRowBytes;
        public int TileSize;
        public int SlidingWindowSize;
        public KvCacheDType KeyDType;
        public KvCacheDType ValueDType;
        public int WindowCapacity;
        /// <summary>Optional per-head sink logits [NumHeads] (gpt-oss). Null = no sinks.</summary>
        public float* Sinks;
    }

    [SkipLocalsInit]
    private static unsafe void QuantizedTiledAttentionWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<QuantizedTiledCtx>((void*)ctxPtr);

        ComputeThreadPool.PartitionRange(ctx.NumHeads, threadIdx, threadCount, out int startHead, out int endHead);
        if (startHead >= ctx.NumHeads) return;

        Span<float> tileScores = stackalloc float[MaxTileSize];

        for (int h = startHead; h < endHead; h++)
        {
            ExecuteTiledQuantizedHead(
                ctx.Q, ctx.KQuant, ctx.VQuant, ctx.KWindow, ctx.VWindow, ctx.Output, tileScores,
                ctx.SeqQ, ctx.QuantLen, ctx.WindowLen, ctx.HeadDim, ctx.Scale,
                ctx.QStride, ctx.KvStride, ctx.KQuantRowBytes, ctx.VQuantRowBytes,
                ctx.PositionOffset, ctx.TileSize, ctx.SlidingWindowSize,
                ctx.KeyDType, ctx.ValueDType,
                h, h / ctx.GroupSize,
                ctx.WindowCapacity,
                ctx.Sinks is null ? float.NegativeInfinity : ctx.Sinks[h]);
        }
    }

    /// <summary>
    /// Processes a single head for quantized KV-cache attention.
    /// Phase 1: iterate tiles over quantized region with per-tile dequant.
    /// Phase 2: iterate tiles over full-precision window region.
    /// Uses online softmax throughout both phases.
    /// </summary>
    [SkipLocalsInit]
    private static unsafe void ExecuteTiledQuantizedHead(
        float* q, byte* kQuant, byte* vQuant, float* kWindow, float* vWindow, float* output,
        Span<float> tileScores,
        int seqQ, int quantLen, int windowLen, int headDim, float scale,
        int qStride, int kvStride, int kQuantRowBytes, int vQuantRowBytes,
        int positionOffset, int tileSize, int slidingWindowSize,
        KvCacheDType keyDType, KvCacheDType valueDType,
        int headIdx, int kvHeadIdx,
        int windowCapacity,
        float sinkLogit = float.NegativeInfinity)
    {
        int seqKv = quantLen + windowLen;
        int window = slidingWindowSize;
        bool hasSink = !float.IsNegativeInfinity(sinkLogit);

        // Per-tile scratch for dequantized K and V rows
        // Budget: tileSize * headDim * sizeof(float) per buffer
        int scratchElems = tileSize * headDim;
        float* kScratch;
        float* vScratch;
        bool scratchAllocated = false;

        int scratchBytes = scratchElems * sizeof(float);
        if (scratchBytes <= StackAllocThreshold)
        {
            float* kBuf = stackalloc float[scratchElems];
            float* vBuf = stackalloc float[scratchElems];
            kScratch = kBuf;
            vScratch = vBuf;
        }
        else
        {
            kScratch = (float*)NativeMemory.AlignedAlloc((nuint)scratchBytes, 64);
            vScratch = (float*)NativeMemory.AlignedAlloc((nuint)scratchBytes, 64);
            scratchAllocated = true;
        }

        try
        {
            for (int i = 0; i < seqQ; i++)
            {
                var qRow = new ReadOnlySpan<float>(q + i * qStride + headIdx * headDim, headDim);
                var outRow = new Span<float>(output + i * qStride + headIdx * headDim, headDim);
                outRow.Clear();

                int visibleEnd = Math.Min(seqKv, positionOffset + i + 1);
                int visibleStart = (window > 0)
                    ? Math.Max(0, positionOffset + i - window + 1)
                    : 0;
                if (visibleStart >= visibleEnd) continue;

                // Attention sink (gpt-oss): seed the online softmax as if a virtual key
                // with logit sinkLogit (and zero value vector) had already been processed.
                float maxSoFar = hasSink ? sinkLogit : float.NegativeInfinity;
                float sumExp = hasSink ? 1f : 0f;

                // ── Phase 1: Quantized region [visibleStart..min(quantLen, visibleEnd)) ──
                int quantEnd = Math.Min(quantLen, visibleEnd);
                int quantStart = Math.Max(visibleStart, 0);

                for (int tileBase = quantStart; tileBase < quantEnd; tileBase += tileSize)
                {
                    int tileLen = Math.Min(tileSize, quantEnd - tileBase);
                    var scores = tileScores.Slice(0, tileLen);

                    // Dequantize tile of K for this head
                    DequantTile(kQuant, tileBase, tileLen, kvStride, kQuantRowBytes,
                                kvHeadIdx, headDim, keyDType, kScratch);

                    // Compute scores: dot(qRow, kScratch[j*headDim .. (j+1)*headDim]) * scale
                    for (int j = 0; j < tileLen; j++)
                    {
                        var kRow = new ReadOnlySpan<float>(kScratch + j * headDim, headDim);
                        scores[j] = TensorPrimitives.Dot(qRow, kRow) * scale;
                    }

                    // Online softmax update
                    float tileMax = TensorPrimitives.Max(scores);
                    float newMax = MathF.Max(maxSoFar, tileMax);
                    float correction = FastMath.FastExp(maxSoFar - newMax);

                    if (correction < 1f)
                    {
                        sumExp *= correction;
                        TensorPrimitives.Multiply(outRow, correction, outRow);
                    }

                    sumExp += FastMath.ExpSumAndStore(scores, scores, -newMax);

                    // Dequantize tile of V for this head and accumulate
                    DequantTile(vQuant, tileBase, tileLen, kvStride, vQuantRowBytes,
                                kvHeadIdx, headDim, valueDType, vScratch);

                    for (int j = 0; j < tileLen; j++)
                    {
                        float w = scores[j];
                        if (w == 0f) continue;
                        var vRow = new ReadOnlySpan<float>(vScratch + j * headDim, headDim);
                        TensorPrimitives.MultiplyAdd(vRow, w, outRow, outRow);
                    }

                    maxSoFar = newMax;
                }

                // ── Phase 2: Window region [max(quantLen, visibleStart)..visibleEnd) ──
                int windowStart = Math.Max(quantLen, visibleStart);
                for (int tileBase = windowStart; tileBase < visibleEnd; tileBase += tileSize)
                {
                    int tileLen = Math.Min(tileSize, visibleEnd - tileBase);
                    var scores = tileScores.Slice(0, tileLen);

                    for (int j = 0; j < tileLen; j++)
                    {
                        // Map logical position to ring buffer index
                        int ringIdx = (tileBase + j) % windowCapacity;
                        var kRow = new ReadOnlySpan<float>(
                            kWindow + ringIdx * kvStride + kvHeadIdx * headDim, headDim);
                        scores[j] = TensorPrimitives.Dot(qRow, kRow) * scale;
                    }

                    float tileMax = TensorPrimitives.Max(scores);
                    float newMax = MathF.Max(maxSoFar, tileMax);
                    float correction = FastMath.FastExp(maxSoFar - newMax);

                    if (correction < 1f)
                    {
                        sumExp *= correction;
                        TensorPrimitives.Multiply(outRow, correction, outRow);
                    }

                    sumExp += FastMath.ExpSumAndStore(scores, scores, -newMax);

                    for (int j = 0; j < tileLen; j++)
                    {
                        float w = scores[j];
                        if (w == 0f) continue;
                        int ringIdx = (tileBase + j) % windowCapacity;
                        var vRow = new ReadOnlySpan<float>(
                            vWindow + ringIdx * kvStride + kvHeadIdx * headDim, headDim);
                        TensorPrimitives.MultiplyAdd(vRow, w, outRow, outRow);
                    }

                    maxSoFar = newMax;
                }

                if (sumExp > 0f)
                    TensorPrimitives.Multiply(outRow, 1f / sumExp, outRow);
            }
        }
        finally
        {
            if (scratchAllocated)
            {
                NativeMemory.AlignedFree(kScratch);
                NativeMemory.AlignedFree(vScratch);
            }
        }
    }

    /// <summary>
    /// Dequantizes a tile of KV data (Q8_0 or Q4_0) for a specific head into the scratch buffer.
    /// Output layout: <c>[tileLen, headDim]</c> contiguous floats.
    /// </summary>
    [SkipLocalsInit]
    private static unsafe void DequantTile(byte* quantData, int tileBase, int tileLen,
                                            int kvStride, int quantRowBytes,
                                            int kvHeadIdx, int headDim,
                                            KvCacheDType dtype, float* scratch)
    {
        int headOffset = kvHeadIdx * headDim;

        if (dtype == KvCacheDType.Q8_0)
        {
            int blockSize = 32;
            int blockBytes = 34;
            // Offset within a quantized row to reach the start of this head's data
            int headBlockStart = headOffset / blockSize;
            int headBlocks = headDim / blockSize;
            int headQuantOffset = headBlockStart * blockBytes;

            for (int j = 0; j < tileLen; j++)
            {
                byte* rowStart = quantData + (long)(tileBase + j) * quantRowBytes + headQuantOffset;
                float* dst = scratch + j * headDim;
                KvQuantize.Q8_0ToF32(rowStart, dst, headDim);
            }
        }
        else // Q4_0
        {
            int blockSize = 32;
            int blockBytes = 18;
            int headBlockStart = headOffset / blockSize;
            int headBlocks = headDim / blockSize;
            int headQuantOffset = headBlockStart * blockBytes;

            for (int j = 0; j < tileLen; j++)
            {
                byte* rowStart = quantData + (long)(tileBase + j) * quantRowBytes + headQuantOffset;
                float* dst = scratch + j * headDim;
                KvQuantize.Q4_0ToF32(rowStart, dst, headDim);
            }
        }
    }
}
