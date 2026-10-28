using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;
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
        PortableMath.Exp(row, row);
        float sum = TensorPrimitives.Sum((ReadOnlySpan<float>)row) + PortableMath.Exp(sink - m);
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
                    float sum = PortableMath.Exp(sink - m);
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

    // ──────────────────── Tiled attention over cached KV rows (quantized, ring, paged) ────────────────────

    /// <summary>Element format of a cached K or V row, decoded one head slice at a time into F32 scratch.</summary>
    private enum KvRowFormat : byte
    {
        F32,
        F16,
        Q8_0,
        Q4_0,
    }

    /// <summary>
    /// Where the cached K (or V) rows of one segment live, and in what format. Logical row <c>r</c> is at
    /// <c>Base + physical(r) * RowBytes</c> where <c>physical</c> is the identity (contiguous), <c>r % RingCapacity</c>
    /// (ring window) or <c>Blocks[r / BlockSize] * BlockSize + r % BlockSize</c> (paged).
    /// </summary>
    private unsafe struct KvRows
    {
        public byte* Base;
        public int RowBytes;
        /// <summary>Bytes of one head's slice within a row.</summary>
        public int HeadBytes;
        public KvRowFormat Format;
        /// <summary>&gt; 0: ring buffer of this many rows.</summary>
        public int RingCapacity;
        /// <summary>Non-null: logical block → physical block table (paged).</summary>
        public int* Blocks;
        public int BlockSize;

        /// <summary>
        /// Address of logical row <paramref name="row"/>; <paramref name="runLen"/> receives how many rows from it on
        /// (at most <paramref name="limit"/>) are physically consecutive — so a paged block is resolved once, not per row.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public readonly byte* Run(int row, int limit, out int runLen)
        {
            long physical;
            if (Blocks != null)
            {
                int block = row / BlockSize;
                int offset = row - block * BlockSize;
                runLen = Math.Min(BlockSize - offset, limit);
                physical = (long)Blocks[block] * BlockSize + offset;
            }
            else if (RingCapacity > 0)
            {
                int index = row % RingCapacity;
                runLen = Math.Min(RingCapacity - index, limit);
                physical = index;
            }
            else
            {
                runLen = limit;
                physical = row;
            }

            return Base + physical * RowBytes;
        }

        /// <summary>Builds a descriptor; <see cref="HeadBytes"/> follows from the format.</summary>
        public static KvRows Create(void* basePtr, int rowBytes, KvRowFormat format, int headDim,
                                    int ringCapacity = 0, int* blocks = null, int blockSize = 0)
            => new()
            {
                Base = (byte*)basePtr, RowBytes = rowBytes, Format = format,
                HeadBytes = format switch
                {
                    KvRowFormat.F32 => headDim * sizeof(float),
                    KvRowFormat.F16 => headDim * 2,
                    KvRowFormat.Q8_0 => headDim / KvQuantize.BlockSize * KvQuantize.Q8_0BlockBytes,
                    _ => headDim / KvQuantize.BlockSize * KvQuantize.Q4_0BlockBytes,
                },
                RingCapacity = ringCapacity, Blocks = blocks, BlockSize = blockSize,
            };
    }

    /// <summary>
    /// Returns one head's slice of a cached row as F32: the row itself for F32, else decoded into <paramref name="scratch"/>
    /// (<paramref name="headDim"/> floats) by the same per-row helpers the quantized cache always used.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float* LoadHeadRow(byte* src, KvRowFormat format, int headDim, float* scratch)
    {
        switch (format)
        {
            case KvRowFormat.F32:
                return (float*)src;
            case KvRowFormat.F16:
                TensorPrimitives.ConvertToSingle(new ReadOnlySpan<Half>(src, headDim), new Span<float>(scratch, headDim));
                return scratch;
            case KvRowFormat.Q8_0:
                KvQuantize.Q8_0ToF32(src, scratch, headDim);
                return scratch;
            default:
                KvQuantize.Q4_0ToF32(src, scratch, headDim);
                return scratch;
        }
    }

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
        ValidateHeads(numHeads, numKvHeads, headDim);

        int kvStride = numKvHeads * headDim;

        int quantLen = kvCache.QuantizedLength;
        int windowLen = kvCache.WindowLength;
        // Compact sliding-window layer (#1169): row 0 holds position rowBase. Masks are relative, so shift both.
        int rowBase = (kvCache as IWindowedKvCache)?.RowBaseOf(layerIndex) ?? 0;
        quantLen -= rowBase;
        positionOffset -= rowBase;

        // Per-layer quantized row bytes: equals the scalar property for every uniform
        // model; for Gemma-4 the sliding/global layers carry distinct widths. The
        // kvStride local above is already per-layer (derived from the per-call
        // numKvHeads/headDim the caller passes for this layer).
        var ctx = new TiledKvCtx
        {
            K0 = KvRows.Create((void*)kvCache.GetQuantizedKeysPtr(layerIndex), kvCache.KeyQuantizedRowBytesOf(layerIndex),
                               QuantizedRowFormat(kvCache.KeyDType), headDim),
            V0 = KvRows.Create((void*)kvCache.GetQuantizedValuesPtr(layerIndex), kvCache.ValueQuantizedRowBytesOf(layerIndex),
                               QuantizedRowFormat(kvCache.ValueDType), headDim),
            // The full-precision recent window is an F32 ring indexed by logical position.
            K1 = KvRows.Create((void*)kvCache.GetWindowKeysPtr(layerIndex), kvStride * sizeof(float), KvRowFormat.F32, headDim,
                               ringCapacity: kvCache.WindowCapacity),
            V1 = KvRows.Create((void*)kvCache.GetWindowValuesPtr(layerIndex), kvStride * sizeof(float), KvRowFormat.F32, headDim,
                               ringCapacity: kvCache.WindowCapacity),
            Split = quantLen,
            SeqKv = quantLen + windowLen,
            Scale = 1.0f / MathF.Sqrt(headDim),
        };
        RunTiled(ref ctx, q, output, seqQ, numHeads, numKvHeads, headDim, positionOffset, pool,
                 slidingWindowSize, softCap, sinks);
    }

    /// <summary>
    /// Attention that reads a paged KV cache's physical blocks in place (#1268): no gather of the history into a
    /// contiguous copy. Rows may be F16, Q8_0 or F32; each head slice is decoded into a per-worker row scratch.
    /// Same tiled online-softmax arithmetic, worker split and tile boundaries as the quantized-cache path.
    /// </summary>
    /// <param name="q">Queries <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="kvCache">Paged cache already holding positions <c>[0, positionOffset + seqQ)</c>.</param>
    /// <param name="layerIndex">Layer whose blocks to read.</param>
    /// <param name="output">Output <c>[seqQ, numHeads * headDim]</c>.</param>
    /// <param name="seqQ">Query rows.</param>
    /// <param name="numHeads">Query heads.</param>
    /// <param name="numKvHeads">KV heads; <c>numKvHeads * headDim</c> must equal the cache stride.</param>
    /// <param name="headDim">Head dimension.</param>
    /// <param name="positionOffset">Position of the first query row.</param>
    /// <param name="scale">Score scale (1/sqrt(headDim), or Gemma <c>query_pre_attn_scalar</c> / Granite multiplier).</param>
    /// <param name="pool">Optional worker pool (heads partitioned as in the contiguous path).</param>
    /// <param name="slidingWindowSize">Optional sliding window.</param>
    /// <param name="softCap">Attention-logit soft-cap; 0 = disabled.</param>
    /// <param name="sinks">Optional per-head sink logits [numHeads].</param>
    /// <exception cref="ArgumentException">Geometry mismatch, unsupported row type or an incomplete block table.</exception>
    [SkipLocalsInit]
    public static unsafe void Execute(float* q, IPagedKvCache kvCache, int layerIndex, float* output,
                                      int seqQ, int numHeads, int numKvHeads, int headDim, int positionOffset,
                                      float scale, ComputeThreadPool? pool,
                                      int? slidingWindowSize = null, float softCap = 0f, float[]? sinks = null)
    {
        ValidateHeads(numHeads, numKvHeads, headDim);
        int kvStride = numKvHeads * headDim;
        if (kvCache.KvStride != kvStride)
            throw new ArgumentException($"numKvHeads * headDim ({kvStride}) must equal the paged cache stride ({kvCache.KvStride}).");
        int seqKv = kvCache.CurrentLength;
        if (seqQ <= 0 || positionOffset < 0 || (long)positionOffset + seqQ > seqKv)
            throw new ArgumentException("Query positions must already be present in the cache.");

        var keys = kvCache.GetKeyBlocksRef(layerIndex);
        var values = kvCache.GetValueBlocksRef(layerIndex);
        if (keys.DeviceId != -1 || values.DeviceId != -1 || keys.DType != values.DType)
            throw new ArgumentException("Paged attention needs host K/V blocks of one element type.");
        KvRowFormat format = keys.DType == DType.Float16 ? KvRowFormat.F16
            : keys.DType == DType.Q8_0 ? KvRowFormat.Q8_0
            : keys.DType == DType.Float32 ? KvRowFormat.F32
            : throw new ArgumentException($"Unsupported paged KV row type {keys.DType}.");
        if (format == KvRowFormat.Q8_0 && headDim % KvQuantize.BlockSize != 0)
            throw new ArgumentException($"Q8_0 paged KV needs headDim % {KvQuantize.BlockSize} == 0.");

        int blockSize = kvCache.BlockSize;
        ReadOnlySpan<int> blocks = kvCache.BlockIds;
        int usedBlocks = (seqKv + blockSize - 1) / blockSize;
        if (blockSize <= 0 || blocks.Length < usedBlocks)
            throw new ArgumentException("Paged block table does not cover the cached length.", nameof(kvCache));
        uint physicalBlocks = (uint)(keys.Dim0 / blockSize);
        for (int i = 0; i < usedBlocks; i++)
        {
            if ((uint)blocks[i] >= physicalBlocks)
                throw new ArgumentException($"Block id {blocks[i]} lies outside the pool.", nameof(kvCache));
        }

        int rowBytes = (int)keys.DType.ComputeByteCount(kvStride);
        fixed (int* table = blocks)
        {
            var ctx = new TiledKvCtx
            {
                K0 = KvRows.Create((void*)keys.DataPointer, rowBytes, format, headDim, blocks: table, blockSize: blockSize),
                V0 = KvRows.Create((void*)values.DataPointer, rowBytes, format, headDim, blocks: table, blockSize: blockSize),
                Split = seqKv,
                SeqKv = seqKv,
                Scale = scale,
                OneShot = true,
            };
            RunTiled(ref ctx, q, output, seqQ, numHeads, numKvHeads, headDim, positionOffset, pool,
                     slidingWindowSize, softCap, sinks);
        }
    }

    private static void ValidateHeads(int numHeads, int numKvHeads, int headDim)
    {
        if (headDim <= 0)
            throw new ArgumentException($"headDim must be positive, got {headDim}", nameof(headDim));
        if (numKvHeads <= 0 || numHeads % numKvHeads != 0)
            throw new ArgumentException(
                $"numHeads ({numHeads}) must be divisible by numKvHeads ({numKvHeads})", nameof(numKvHeads));
    }

    private static KvRowFormat QuantizedRowFormat(KvCacheDType dtype) => dtype switch
    {
        KvCacheDType.Q8_0 => KvRowFormat.Q8_0,
        KvCacheDType.Q4_0 => KvRowFormat.Q4_0,
        _ => throw new ArgumentOutOfRangeException(nameof(dtype), dtype, "Quantized KV region must be Q8_0 or Q4_0."),
    };

    /// <summary>Fills the shared fields of <paramref name="ctx"/> and runs every head, on <paramref name="pool"/> when given.</summary>
    private static unsafe void RunTiled(ref TiledKvCtx ctx, float* q, float* output, int seqQ,
                                        int numHeads, int numKvHeads, int headDim, int positionOffset,
                                        ComputeThreadPool? pool, int? slidingWindowSize, float softCap, float[]? sinks)
    {
        if (sinks is not null && sinks.Length < numHeads)
            throw new ArgumentException($"sinks must have at least numHeads ({numHeads}) entries.", nameof(sinks));

        ctx.Q = q;
        ctx.Output = output;
        ctx.SeqQ = seqQ;
        ctx.NumHeads = numHeads;
        ctx.GroupSize = numHeads / numKvHeads;
        ctx.HeadDim = headDim;
        ctx.QStride = numHeads * headDim;
        ctx.PositionOffset = positionOffset;
        ctx.TileSize = ComputeTileSize(headDim);
        ctx.SlidingWindowSize = slidingWindowSize ?? 0;
        ctx.SoftCap = softCap;

        fixed (float* sinksPtr = sinks)
        fixed (TiledKvCtx* ctxPtr = &ctx)
        {
            ctx.Sinks = sinksPtr;
            if (pool is null || numHeads < 2)
                TiledKvWorker((nint)ctxPtr, 0, 1);
            else
                pool.Dispatch((nint)ctxPtr, &TiledKvWorker);
        }
    }

    /// <summary>
    /// One attention call over cached rows split into two segments: <c>[0, Split)</c> read through
    /// <see cref="K0"/>/<see cref="V0"/> and <c>[Split, SeqKv)</c> through <see cref="K1"/>/<see cref="V1"/>
    /// (quantized history + F32 recent ring, or a single paged segment with <c>Split == SeqKv</c>).
    /// </summary>
    private unsafe struct TiledKvCtx
    {
        public float* Q;
        public float* Output;
        public KvRows K0;
        public KvRows V0;
        public KvRows K1;
        public KvRows V1;
        public int Split;
        public int SeqKv;
        public int SeqQ;
        public int NumHeads;
        public int GroupSize;
        public int HeadDim;
        public int QStride;
        public int PositionOffset;
        public int TileSize;
        public int SlidingWindowSize;
        public float Scale;
        /// <summary>Gemma-2/3 attention-logit soft-cap; 0 = disabled.</summary>
        public float SoftCap;
        /// <summary>Optional per-head sink logits [NumHeads] (gpt-oss). Null = no sinks.</summary>
        public float* Sinks;
        /// <summary>
        /// Rows with at most <see cref="OneShotMaxKeys"/> visible keys take the single-shot softmax, exactly as the contiguous
        /// path does, so F32 paged rows reproduce it bit for bit (#530). The quantized-cache path keeps tiling every row.
        /// </summary>
        public bool OneShot;
    }

    /// <summary>Query heads sharing one KV head that are processed together, so each cached row is decoded once for all of them.</summary>
    private const int MaxGroupHeads = 4;

    [SkipLocalsInit]
    private static unsafe void TiledKvWorker(nint ctxPtr, int threadIdx, int threadCount)
    {
        ref var ctx = ref Unsafe.AsRef<TiledKvCtx>((void*)ctxPtr);

        // Contiguous head ranges, as in the contiguous path: heads sharing a KV head stay on one worker.
        ComputeThreadPool.PartitionRange(ctx.NumHeads, threadIdx, threadCount, out int startHead, out int endHead);
        if (startHead >= ctx.NumHeads) return;

        int scoreStride = ctx.OneShot ? OneShotMaxKeys : MaxTileSize;
        float* scores = stackalloc float[MaxGroupHeads * scoreStride];
        float* rowScratch = stackalloc float[ctx.HeadDim];

        for (int h = startHead; h < endHead;)
        {
            // Run of consecutive heads on the same KV head (capped), never crossing the worker's range.
            int groupEnd = Math.Min(endHead, (h / ctx.GroupSize + 1) * ctx.GroupSize);
            int count = Math.Min(MaxGroupHeads, groupEnd - h);
            ExecuteHeadGroup(ref ctx, h, count, scores, scoreStride, rowScratch);
            h += count;
        }
    }

    /// <summary>
    /// Processes <paramref name="count"/> consecutive query heads that share one KV head. Per query row: softmax over the
    /// visible keys of segment 0 then segment 1 — single-shot for short rows when <see cref="TiledKvCtx.OneShot"/>, else
    /// online, tile by tile from the row's own visible start (tile boundaries depend on the query position only, #530).
    /// Each head's arithmetic is exactly the single-head sequence; only the K/V row decode is shared.
    /// </summary>
    [SkipLocalsInit]
    private static unsafe void ExecuteHeadGroup(ref TiledKvCtx ctx, int firstHead, int count, float* scores, int scoreStride,
                                                float* rowScratch)
    {
        int headDim = ctx.HeadDim;
        int kvHeadIdx = firstHead / ctx.GroupSize;
        int window = ctx.SlidingWindowSize;
        float* maxSoFar = stackalloc float[MaxGroupHeads];
        float* sumExp = stackalloc float[MaxGroupHeads];

        for (int i = 0; i < ctx.SeqQ; i++)
        {
            float* q = ctx.Q + (long)i * ctx.QStride + (long)firstHead * headDim;
            float* outRows = ctx.Output + (long)i * ctx.QStride + (long)firstHead * headDim;
            new Span<float>(outRows, count * headDim).Clear();

            int visibleEnd = Math.Min(ctx.SeqKv, ctx.PositionOffset + i + 1);
            int visibleStart = window > 0 ? Math.Max(0, ctx.PositionOffset + i - window + 1) : 0;
            if (visibleStart >= visibleEnd) continue;

            int split = Math.Clamp(ctx.Split, visibleStart, visibleEnd);
            if (ctx.OneShot && visibleEnd - visibleStart <= OneShotMaxKeys)
            {
                // Mirrors OneShotRow per head: score every visible key, soft-cap, softmax, then accumulate.
                int n = visibleEnd - visibleStart;
                int mid = split - visibleStart;
                ScoreRows(in ctx.K0, visibleStart, split, kvHeadIdx, q, count, scores, scoreStride, rowScratch, ctx.Scale, headDim);
                ScoreRows(in ctx.K1, split, visibleEnd, kvHeadIdx, q, count, scores + mid, scoreStride, rowScratch, ctx.Scale, headDim);
                for (int g = 0; g < count; g++)
                {
                    var row = new Span<float>(scores + g * scoreStride, n);
                    if (ctx.SoftCap > 0f)
                        ApplySoftCap(row, ctx.SoftCap);
                    float sink = ctx.Sinks is null ? float.NegativeInfinity : ctx.Sinks[firstHead + g];
                    if (!float.IsNegativeInfinity(sink))
                        SoftmaxRowWithSink(row, sink);
                    else
                        Softmax.ExecuteFast(row, row);
                }

                AccumulateRows(in ctx.V0, visibleStart, split, kvHeadIdx, scores, scoreStride, count, outRows, rowScratch, headDim);
                AccumulateRows(in ctx.V1, split, visibleEnd, kvHeadIdx, scores + mid, scoreStride, count, outRows, rowScratch, headDim);
                continue;
            }

            // Attention sink (gpt-oss): seed the online softmax as if a virtual key
            // with logit sinkLogit (and zero value vector) had already been processed.
            for (int g = 0; g < count; g++)
            {
                float sink = ctx.Sinks is null ? float.NegativeInfinity : ctx.Sinks[firstHead + g];
                bool hasSink = !float.IsNegativeInfinity(sink);
                maxSoFar[g] = hasSink ? sink : float.NegativeInfinity;
                sumExp[g] = hasSink ? 1f : 0f;
            }

            AttendSegment(in ctx.K0, in ctx.V0, visibleStart, split, kvHeadIdx, q, outRows, count,
                          scores, scoreStride, rowScratch, ref ctx, maxSoFar, sumExp);
            AttendSegment(in ctx.K1, in ctx.V1, split, visibleEnd, kvHeadIdx, q, outRows, count,
                          scores, scoreStride, rowScratch, ref ctx, maxSoFar, sumExp);

            for (int g = 0; g < count; g++)
            {
                if (sumExp[g] > 0f)
                {
                    var outRow = new Span<float>(outRows + g * headDim, headDim);
                    TensorPrimitives.Multiply(outRow, 1f / sumExp[g], outRow);
                }
            }
        }
    }

    /// <summary>Online-softmax accumulation of keys <c>[start, end)</c> of one segment, for each head of the group.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AttendSegment(in KvRows k, in KvRows v, int start, int end, int kvHeadIdx,
                                             float* q, float* outRows, int count, float* scores, int scoreStride,
                                             float* rowScratch, ref TiledKvCtx ctx, float* maxSoFar, float* sumExp)
    {
        int headDim = ctx.HeadDim;
        int tileSize = ctx.TileSize;
        float softCap = ctx.SoftCap;

        for (int tileBase = start; tileBase < end; tileBase += tileSize)
        {
            int tileLen = Math.Min(tileSize, end - tileBase);

            ScoreRows(in k, tileBase, tileBase + tileLen, kvHeadIdx, q, count, scores, scoreStride, rowScratch, ctx.Scale, headDim);

            for (int g = 0; g < count; g++)
            {
                var tile = new Span<float>(scores + g * scoreStride, tileLen);
                var outRow = new Span<float>(outRows + g * headDim, headDim);

                // Gemma 2/3 soft-cap on raw scores, before the running max (same placement as TiledRow).
                if (softCap > 0f)
                    ApplySoftCap(tile, softCap);

                // Online softmax update
                float tileMax = TensorPrimitives.Max(tile);
                float newMax = MathF.Max(maxSoFar[g], tileMax);
                float correction = FastMath.FastExp(maxSoFar[g] - newMax);

                if (correction < 1f)
                {
                    sumExp[g] *= correction;
                    TensorPrimitives.Multiply(outRow, correction, outRow);
                }

                sumExp[g] += FastMath.ExpSumAndStore(tile, tile, -newMax);
                maxSoFar[g] = newMax;
            }

            AccumulateRows(in v, tileBase, tileBase + tileLen, kvHeadIdx, scores, scoreStride, count, outRows, rowScratch, headDim);
        }
    }

    /// <summary>
    /// <c>scores[g * scoreStride + j] = dot(q_g, K[start + j]) * scale</c> for keys <c>[start, end)</c> and each of the
    /// <paramref name="count"/> heads; each K row is decoded once and paged blocks are resolved once per run.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void ScoreRows(in KvRows k, int start, int end, int kvHeadIdx, float* q, int count,
                                         float* scores, int scoreStride, float* rowScratch, float scale, int headDim)
    {
        long headOffset = (long)kvHeadIdx * k.HeadBytes;
        int rowBytes = k.RowBytes;
        KvRowFormat format = k.Format;
        for (int j = 0, n = end - start; j < n;)
        {
            byte* row = k.Run(start + j, n - j, out int run) + headOffset;
            for (int r = 0; r < run; r++, j++, row += rowBytes)
            {
                var kRow = new ReadOnlySpan<float>(LoadHeadRow(row, format, headDim, rowScratch), headDim);
                for (int g = 0; g < count; g++)
                    scores[g * scoreStride + j] = TensorPrimitives.Dot(new ReadOnlySpan<float>(q + g * headDim, headDim), kRow) * scale;
            }
        }
    }

    /// <summary>
    /// <c>out_g += weights[g * weightStride + j] * V[start + j]</c> for values <c>[start, end)</c> and each head, skipping
    /// zero weights; a V row is decoded once, and only when some head weights it.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void AccumulateRows(in KvRows v, int start, int end, int kvHeadIdx, float* weights, int weightStride,
                                              int count, float* outRows, float* rowScratch, int headDim)
    {
        long headOffset = (long)kvHeadIdx * v.HeadBytes;
        int rowBytes = v.RowBytes;
        KvRowFormat format = v.Format;
        for (int j = 0, n = end - start; j < n;)
        {
            byte* row = v.Run(start + j, n - j, out int run) + headOffset;
            for (int r = 0; r < run; r++, j++, row += rowBytes)
            {
                float* vRow = null;
                for (int g = 0; g < count; g++)
                {
                    float w = weights[g * weightStride + j];
                    if (w == 0f) continue;
                    vRow = vRow != null ? vRow : LoadHeadRow(row, format, headDim, rowScratch);
                    var outRow = new Span<float>(outRows + g * headDim, headDim);
                    TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(vRow, headDim), w, outRow, outRow);
                }
            }
        }
    }
}
