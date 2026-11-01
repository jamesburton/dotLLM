using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.ChunkReuse;

/// <summary>Outcome of <see cref="SelectiveRecomputer.Run"/>.</summary>
/// <param name="ForwardCalls">Number of <see cref="IModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, bool)"/> calls made (one per maximal run of consecutive selected positions).</param>
/// <param name="TokensRecomputed">Number of positions whose K/V were recomputed (the selection size).</param>
/// <param name="LastPositionLogits">
/// Next-token logits (<c>[vocab]</c>, copied) of the last context position <c>L-1</c>, computed over the fixed-up cache — only
/// when <c>L-1</c> was selected (it is then the last row of the last run); otherwise <c>null</c>. To get them without a
/// continuation, include <c>L-1</c> in the selection; with a continuation (query/suffix), forward it at <c>L..</c> on the cache
/// and read its last row instead.
/// </param>
public sealed record SelectiveRecomputeResult(int ForwardCalls, int TokensRecomputed, float[]? LastPositionLogits);

/// <summary>
/// Selective recompute (CacheBlend-style fix-up) of a naive-concatenation chunk cache over the public <see cref="IModel"/> and
/// <see cref="IKvCache"/>: the selected positions get fresh K/V at every layer, attending to the stale (per-chunk) K/V of the
/// non-selected positions and to the already refreshed K/V of earlier selected positions.
/// </summary>
/// <remarks>
/// <para>
/// The selection is split into maximal runs of consecutive positions and each run is one ordinary <c>Forward</c> with its
/// token ids, its true positions and the assembled cache, in ascending position order. A <c>Forward</c> writes the run's
/// rows (post-RoPE K, V) per layer before attending, and the CPU attention masks a query at <c>p</c> to keys <c>[0, p]</c>
/// with <c>p = positions[0] + i</c> — exact for a contiguous run, which is why runs are not merged (a non-contiguous
/// <c>Forward</c> would mask the later tokens as if they were consecutive). Because attention is causal, an earlier run never
/// sees a later one, and a later run sees the earlier runs' refreshed K/V at every layer: the result equals one-pass
/// layer-wise recompute of the whole selection.
/// </para>
/// <para>
/// Writing below <see cref="IKvCache.CurrentLength"/> is in place: <c>SimpleKvCache.Update</c> copies each row to its
/// position and sets <c>CurrentLength = max(CurrentLength, maxPos + 1)</c>, and the forward never truncates. Rows after a
/// run therefore stay intact (checked after every call: a cache whose length moves is rejected).
/// </para>
/// <para>Cost: one trunk pass per run, so scattered selections pay many small forwards (fine for quality studies, not TTFT).</para>
/// </remarks>
public static class SelectiveRecomputer
{
    /// <summary>
    /// Recomputes the K/V of <paramref name="selectedIndices"/> in <paramref name="cache"/>, which must already hold the
    /// assembled context of <paramref name="chunks"/> at positions <c>[0, L)</c> (e.g. from <see cref="ChunkAssembler"/>
    /// on an empty cache), <c>L = Σ chunk length</c>.
    /// </summary>
    /// <param name="model">The model that encoded the chunks.</param>
    /// <param name="cache">The assembled cache; updated in place. Rows at positions <c>&gt;= L</c> are left untouched.</param>
    /// <param name="chunks">The chunks in context order; their concatenated token ids give the token at each position.</param>
    /// <param name="selectedIndices">Global positions in <c>[0, L)</c> to recompute, strictly ascending. Empty is a no-op.</param>
    /// <returns>Calls made, tokens recomputed and, when <c>L-1</c> was selected, the last position's logits.</returns>
    /// <exception cref="ArgumentException">The selection is not strictly ascending or out of range, a chunk is null or empty,
    /// or the cache holds fewer than <c>L</c> positions. Nothing is written in that case.</exception>
    /// <exception cref="InvalidOperationException">The cache changed its length during a forward (it does not write in place).</exception>
    public static unsafe SelectiveRecomputeResult Run(IModel model, IKvCache cache, IReadOnlyList<ChunkKvEntry> chunks,
        ReadOnlySpan<int> selectedIndices)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentNullException.ThrowIfNull(chunks);

        int total = 0;
        for (int c = 0; c < chunks.Count; c++)
        {
            int n = chunks[c]?.TokenIds?.Length ?? 0;
            if (n == 0)
                throw new ArgumentException($"Chunk {c} is null or empty.", nameof(chunks));
            total += n;
        }
        for (int i = 0; i < selectedIndices.Length; i++)
        {
            int p = selectedIndices[i];
            if ((uint)p >= (uint)total)
                throw new ArgumentException($"Selected position {p} (index {i}) is outside the context [0, {total}).", nameof(selectedIndices));
            if (i > 0 && p <= selectedIndices[i - 1])
                throw new ArgumentException(
                    $"Selected positions must be strictly ascending; {p} follows {selectedIndices[i - 1]} at index {i}.", nameof(selectedIndices));
        }
        if (cache.CurrentLength < total)
            throw new ArgumentException(
                $"The cache holds {cache.CurrentLength} positions but the chunks span {total}; assemble them first.", nameof(cache));
        if (selectedIndices.IsEmpty)
            return new SelectiveRecomputeResult(0, 0, null);

        int[] tokens = new int[total];
        int offset = 0;
        foreach (ChunkKvEntry chunk in chunks)
        {
            chunk.TokenIds.CopyTo(tokens, offset);
            offset += chunk.TokenIds.Length;
        }

        int length = cache.CurrentLength;
        int calls = 0;
        float[]? lastLogits = null;
        int[] positions = new int[selectedIndices.Length];
        int start = 0;
        while (start < selectedIndices.Length)
        {
            int end = start + 1;
            while (end < selectedIndices.Length && selectedIndices[end] == selectedIndices[end - 1] + 1) end++;
            int first = selectedIndices[start], count = end - start;
            Span<int> runPositions = positions.AsSpan(start, count);
            for (int i = 0; i < count; i++) runPositions[i] = first + i;

            using (ITensor logits = model.Forward(tokens.AsSpan(first, count), runPositions, deviceId: -1, cache, lastTokenLogitsOnly: true))
            {
                calls++;
                if (first + count == total)
                {
                    int vocab = logits.Shape[1];
                    long row = logits.Shape[0] - 1;
                    lastLogits = new ReadOnlySpan<float>((float*)logits.DataPointer + row * vocab, vocab).ToArray();
                }
            }
            if (cache.CurrentLength != length)
                throw new InvalidOperationException(
                    $"The cache length moved from {length} to {cache.CurrentLength} during a recompute forward; "
                    + $"{cache.GetType().Name} does not overwrite rows in place.");
            start = end;
        }
        return new SelectiveRecomputeResult(calls, selectedIndices.Length, lastLogits);
    }
}
