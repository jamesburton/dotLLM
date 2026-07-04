using System.Buffers;
using System.Globalization;
using System.Numerics.Tensors;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Cpu.Kernels;

namespace DotLLM.Models.Architectures;

/// <summary>
/// DiffusionGemma self-conditioning soft token-embedding (the vocab-side half of
/// <c>dg_canvas_embed</c>), shared by the CPU <see cref="TransformerModel"/> and the Vulkan
/// backend's host-side SC signal so both backends stay semantically identical (the CPU
/// implementation is the cross-backend oracle).
/// </summary>
/// <remarks>
/// <para>
/// <b>Dense reference</b> (GEMMA4-GRAPH-SPEC, source-confirmed against
/// <c>diffusion-gemma.cpp</c>): per canvas column <c>c</c>,
/// <c>soft[c] = Σ_v softmax(prev_logits[c])[v] · tok_embd[v]</c> — a softmax over the FULL
/// vocab followed by a <c>[canvas, vocab] × [vocab, hidden]</c> contraction. On the real 26B
/// (262 144-token vocab) this dominates the per-step cost (~47 GFLOP/step at canvas 32,
/// issue #121).
/// </para>
/// <para>
/// <b>Top-K sparsification</b> (issue #121, default K = 256 via
/// <see cref="DiffusionConfig.SelfCondTopK"/>): per canvas column, only the K highest
/// previous-step logits participate — softmax renormalised over that subset, soft-embed =
/// weighted sum of the K gathered embedding rows. The spec does not define a sample-reduce
/// contract, so the dense path (<c>K &lt;= 0</c>, or <c>K &gt;= vocab</c>) remains the exact
/// oracle and is byte-identical to the pre-#121 implementation.
/// </para>
/// <para>
/// The Gemma <c>sqrt(n_embd)</c> embedding scale and the downstream
/// <c>rms_norm · self_cond_pre_norm</c> + gated GeGLU MLP are applied by the callers — this
/// type only produces the raw probability-weighted embedding sum.
/// </para>
/// </remarks>
public static unsafe class SelfCondSoftEmbed
{
    /// <summary>Default top-K width when <see cref="DiffusionConfig.SelfCondTopK"/> is unavailable.</summary>
    public const int DefaultTopK = 256;

    /// <summary>
    /// Environment variable overriding the configured top-K at runtime
    /// (<c>DOTLLM_DG_SC_TOPK</c>). Parseable integer values take precedence over
    /// <see cref="DiffusionConfig.SelfCondTopK"/>; <c>&lt;= 0</c> selects the dense path.
    /// </summary>
    public const string TopKEnvVar = "DOTLLM_DG_SC_TOPK";

    /// <summary>
    /// Resolves the effective self-conditioning top-K: the <see cref="TopKEnvVar"/>
    /// environment variable when set and parseable, else
    /// <see cref="DiffusionConfig.SelfCondTopK"/>, else <see cref="DefaultTopK"/>.
    /// </summary>
    /// <param name="config">Diffusion config carrying the configured K (may be null).</param>
    public static int ResolveTopK(DiffusionConfig? config)
        => ResolveTopK(Environment.GetEnvironmentVariable(TopKEnvVar), config);

    /// <summary>
    /// Pure resolution core (unit-testable without touching process environment):
    /// <paramref name="envValue"/> wins when it parses as an integer, else
    /// <see cref="DiffusionConfig.SelfCondTopK"/>, else <see cref="DefaultTopK"/>.
    /// </summary>
    /// <param name="envValue">Raw environment-variable text (null/empty ⇒ not set).</param>
    /// <param name="config">Diffusion config carrying the configured K (may be null).</param>
    public static int ResolveTopK(string? envValue, DiffusionConfig? config)
    {
        if (!string.IsNullOrWhiteSpace(envValue)
            && int.TryParse(envValue, NumberStyles.Integer, CultureInfo.InvariantCulture, out int k))
            return k;
        return config?.SelfCondTopK ?? DefaultTopK;
    }

    /// <summary>
    /// Computes the self-conditioning soft token-embedding
    /// <c>soft[c] = Σ_v p(c, v) · tok_embd[v]</c> for every canvas column into
    /// <paramref name="soft"/> (<c>[canvasLen × hiddenSize]</c>, overwritten). With
    /// <paramref name="topK"/> in <c>(0, vocab)</c> the distribution <c>p(c, ·)</c> is the
    /// softmax renormalised over the K highest logits of column <c>c</c> (sparse path);
    /// otherwise it is the full-vocab softmax (dense reference path, byte-identical to the
    /// pre-#121 implementation).
    /// </summary>
    /// <param name="prevLogits">Previous denoise step's canvas logits <c>[canvasLen × vocab]</c> (post-softcap).</param>
    /// <param name="canvasLen">Number of canvas columns.</param>
    /// <param name="vocab">Vocabulary size (logit row width and embedding-table row count).</param>
    /// <param name="embPtr">Base pointer of the tied token-embedding table (row-major, <paramref name="vocab"/> rows).</param>
    /// <param name="embQt">Quantization type of the embedding table rows.</param>
    /// <param name="hiddenSize">Embedding row width (n_embd).</param>
    /// <param name="topK">Sparsification width; <c>&lt;= 0</c> or <c>&gt;= vocab</c> ⇒ dense.</param>
    /// <param name="soft">Output <c>[canvasLen × hiddenSize]</c>; cleared and accumulated in place.</param>
    public static void Compute(
        ReadOnlySpan<float> prevLogits, int canvasLen, int vocab,
        nint embPtr, QuantizationType embQt, int hiddenSize,
        int topK, Span<float> soft)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(prevLogits.Length, canvasLen * vocab, nameof(prevLogits));
        ArgumentOutOfRangeException.ThrowIfLessThan(soft.Length, canvasLen * hiddenSize, nameof(soft));

        soft = soft[..(canvasLen * hiddenSize)];
        soft.Clear();

        if (topK <= 0 || topK >= vocab)
            ComputeDense(prevLogits, canvasLen, vocab, embPtr, embQt, hiddenSize, soft);
        else
            ComputeSparse(prevLogits, canvasLen, vocab, embPtr, embQt, hiddenSize, topK, soft);
    }

    /// <summary>
    /// Dense reference path: full-vocab softmax per column, then a single sweep of the
    /// embedding table — each row is dequantized ONCE and scatter-accumulated into every
    /// canvas soft-vector weighted by that token's probability (ascending token order, so
    /// the per-column accumulation order is deterministic).
    /// </summary>
    private static void ComputeDense(
        ReadOnlySpan<float> prevLogits, int canvasLen, int vocab,
        nint embPtr, QuantizationType embQt, int hiddenSize, Span<float> soft)
    {
        float[] probsBuf = ArrayPool<float>.Shared.Rent(canvasLen * vocab);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(hiddenSize);
        try
        {
            for (int c = 0; c < canvasLen; c++)
                Softmax.Execute(
                    prevLogits.Slice(c * vocab, vocab),
                    probsBuf.AsSpan(c * vocab, vocab));

            var rowSpan = rowBuf.AsSpan(0, hiddenSize);
            for (int v = 0; v < vocab; v++)
            {
                DequantEmbeddingRow(embPtr, embQt, v, rowSpan, hiddenSize);
                for (int c = 0; c < canvasLen; c++)
                {
                    float w = probsBuf[c * vocab + v];
                    if (w == 0f) continue;
                    TensorPrimitives.MultiplyAdd(
                        rowSpan, w, soft.Slice(c * hiddenSize, hiddenSize),
                        soft.Slice(c * hiddenSize, hiddenSize));
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(probsBuf);
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Sparse top-K path: per canvas column, select the K highest logits (ties broken toward
    /// the lower token id), renormalise the softmax over that subset, and accumulate the K
    /// gathered embedding rows in ascending token order (same per-column accumulation order
    /// as the dense sweep restricted to the selected ids).
    /// </summary>
    private static void ComputeSparse(
        ReadOnlySpan<float> prevLogits, int canvasLen, int vocab,
        nint embPtr, QuantizationType embQt, int hiddenSize, int topK, Span<float> soft)
    {
        int[] idxBuf = ArrayPool<int>.Shared.Rent(topK);
        float[] probsBuf = ArrayPool<float>.Shared.Rent(topK);
        float[] rowBuf = ArrayPool<float>.Shared.Rent(hiddenSize);
        try
        {
            var indices = idxBuf.AsSpan(0, topK);
            var probs = probsBuf.AsSpan(0, topK);
            var rowSpan = rowBuf.AsSpan(0, hiddenSize);
            for (int c = 0; c < canvasLen; c++)
            {
                var logitsC = prevLogits.Slice(c * vocab, vocab);
                SelectTopK(logitsC, topK, indices);
                RenormSoftmax(logitsC, indices, probs);
                var softC = soft.Slice(c * hiddenSize, hiddenSize);
                for (int j = 0; j < topK; j++)
                {
                    DequantEmbeddingRow(embPtr, embQt, indices[j], rowSpan, hiddenSize);
                    TensorPrimitives.MultiplyAdd(rowSpan, probs[j], softC, softC);
                }
            }
        }
        finally
        {
            ArrayPool<int>.Shared.Return(idxBuf);
            ArrayPool<float>.Shared.Return(probsBuf);
            ArrayPool<float>.Shared.Return(rowBuf);
        }
    }

    /// <summary>
    /// Selects the indices of the <paramref name="k"/> largest values of
    /// <paramref name="logits"/> into <paramref name="indices"/> (first <paramref name="k"/>
    /// slots), ordered by ASCENDING index. Deterministic tie-break: equal values prefer the
    /// LOWER index (the selection is exactly the top-K under the total order
    /// "value descending, then index ascending").
    /// </summary>
    /// <param name="logits">Values to select from (length &gt; <paramref name="k"/>).</param>
    /// <param name="k">Number of indices to select (0 &lt; k &lt;= logits.Length).</param>
    /// <param name="indices">Receives the selected indices in ascending order (length &gt;= k).</param>
    public static void SelectTopK(ReadOnlySpan<float> logits, int k, Span<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfLessThanOrEqual(k, 0);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(k, logits.Length, nameof(k));
        ArgumentOutOfRangeException.ThrowIfLessThan(indices.Length, k, nameof(indices));

        // Bounded min-heap of the k best candidates. Heap order: entry a is "less than"
        // entry b when a.val < b.val, or a.val == b.val and a.idx > b.idx — so the root is
        // the weakest kept entry (smallest value; among equal values the HIGHEST index),
        // which yields the exact (value desc, index asc) top-K. Because the scan visits
        // indices in ascending order, a candidate that only TIES the root never displaces
        // it (the root's index is lower), preserving the lower-index preference.
        float[] valBuf = ArrayPool<float>.Shared.Rent(k);
        int[] idxHeapBuf = ArrayPool<int>.Shared.Rent(k);
        try
        {
            Span<float> hv = valBuf.AsSpan(0, k);
            Span<int> hi = idxHeapBuf.AsSpan(0, k);
            int count = 0;
            for (int v = 0; v < logits.Length; v++)
            {
                float x = logits[v];
                if (count < k)
                {
                    // Sift up.
                    int i = count++;
                    hv[i] = x; hi[i] = v;
                    while (i > 0)
                    {
                        int parent = (i - 1) >> 1;
                        if (!HeapLess(hv[i], hi[i], hv[parent], hi[parent])) break;
                        (hv[i], hv[parent]) = (hv[parent], hv[i]);
                        (hi[i], hi[parent]) = (hi[parent], hi[i]);
                        i = parent;
                    }
                }
                else if (HeapLess(hv[0], hi[0], x, v))
                {
                    // Replace root and sift down.
                    hv[0] = x; hi[0] = v;
                    int i = 0;
                    while (true)
                    {
                        int l = 2 * i + 1, r = l + 1, smallest = i;
                        if (l < k && HeapLess(hv[l], hi[l], hv[smallest], hi[smallest])) smallest = l;
                        if (r < k && HeapLess(hv[r], hi[r], hv[smallest], hi[smallest])) smallest = r;
                        if (smallest == i) break;
                        (hv[i], hv[smallest]) = (hv[smallest], hv[i]);
                        (hi[i], hi[smallest]) = (hi[smallest], hi[i]);
                        i = smallest;
                    }
                }
            }

            hi[..k].CopyTo(indices);
            indices[..k].Sort();
        }
        finally
        {
            ArrayPool<float>.Shared.Return(valBuf);
            ArrayPool<int>.Shared.Return(idxHeapBuf);
        }
    }

    /// <summary>Heap total order: (value ascending, index descending) — see <see cref="SelectTopK"/>.</summary>
    private static bool HeapLess(float aVal, int aIdx, float bVal, int bIdx)
        => aVal < bVal || (aVal == bVal && aIdx > bIdx);

    /// <summary>
    /// Softmax renormalised over the subset <paramref name="indices"/> of
    /// <paramref name="logits"/>: <c>probs[j] = exp(logits[indices[j]] − max) / Σ</c> where
    /// max and Σ range over the subset only. Writes <c>indices.Length</c> probabilities
    /// (summing to 1) into <paramref name="probs"/>.
    /// </summary>
    /// <param name="logits">Full logit row.</param>
    /// <param name="indices">Subset of logit indices (the selected top-K).</param>
    /// <param name="probs">Receives one probability per subset entry (length &gt;= indices.Length).</param>
    public static void RenormSoftmax(
        ReadOnlySpan<float> logits, ReadOnlySpan<int> indices, Span<float> probs)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(probs.Length, indices.Length, nameof(probs));
        float max = float.NegativeInfinity;
        for (int j = 0; j < indices.Length; j++)
            if (logits[indices[j]] > max) max = logits[indices[j]];

        float sum = 0f;
        for (int j = 0; j < indices.Length; j++)
        {
            float e = MathF.Exp(logits[indices[j]] - max);
            probs[j] = e;
            sum += e;
        }
        float inv = 1f / sum;
        for (int j = 0; j < indices.Length; j++)
            probs[j] *= inv;
    }

    /// <summary>
    /// Dequantizes one token-embedding row (raw, WITHOUT the Gemma <c>sqrt(n_embd)</c>
    /// scale — the caller folds it once per canvas column after accumulation).
    /// </summary>
    private static void DequantEmbeddingRow(
        nint embPtr, QuantizationType qt, int tokenId, Span<float> dest, int hiddenSize)
    {
        if (qt == QuantizationType.F32)
        {
            new ReadOnlySpan<float>((float*)embPtr + (long)tokenId * hiddenSize, hiddenSize).CopyTo(dest);
        }
        else if (qt == QuantizationType.F16)
        {
            TensorPrimitives.ConvertToSingle(
                new ReadOnlySpan<Half>((Half*)embPtr + (long)tokenId * hiddenSize, hiddenSize), dest);
        }
        else
        {
            long rowBytes = Dequantize.RowByteSize(hiddenSize, qt);
            Dequantize.ToFloat32(embPtr + (nint)((long)tokenId * rowBytes), hiddenSize, qt, dest);
        }
    }
}
