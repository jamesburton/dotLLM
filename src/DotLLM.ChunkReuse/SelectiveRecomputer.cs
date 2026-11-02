using DotLLM.ChunkReuse.Backends;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Vulkan;

namespace DotLLM.ChunkReuse;

/// <summary>Outcome of <see cref="SelectiveRecomputer.Run"/> and <see cref="SelectiveRecomputer.Recompute(IModel, IChunkBackend, IKvCache, ReadOnlySpan{int}, ReadOnlySpan{int}, RecomputeMode, int)"/>.</summary>
/// <param name="ForwardCalls">Number of <see cref="IModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?, bool)"/> calls made (per-run: one per maximal run of consecutive selected positions; single-pass: one per batch of at most <c>maxRowsPerForward</c> selected positions).</param>
/// <param name="TokensRecomputed">Number of positions whose K/V were recomputed (the selection size).</param>
/// <param name="LastPositionLogits">
/// Next-token logits (<c>[vocab]</c>, copied) of the last context position <c>L-1</c>, computed over the fixed-up cache — only
/// when <c>L-1</c> was selected (it is then the last row of the last forward); otherwise <c>null</c>. To get them without a
/// continuation, include <c>L-1</c> in the selection; with a continuation (query/suffix), forward it at <c>L..</c> on the cache
/// and read its last row instead.
/// </param>
/// <param name="QueryPositionAttentionLayers">
/// Vulkan only: attention layers that ran on the per-query-position kernel (<c>ATTN_QPOS</c>, dotLLM #1534) during the recompute, from
/// <see cref="VulkanTransformerModel.AttentionQueryPositionsCount"/>; 0 on the CPU. Says which attention path the recompute took.
/// </param>
public sealed record SelectiveRecomputeResult(int ForwardCalls, int TokensRecomputed, float[]? LastPositionLogits, long QueryPositionAttentionLayers = 0);

/// <summary>How <see cref="SelectiveRecomputer.Recompute(IModel, IChunkBackend, IKvCache, ReadOnlySpan{int}, ReadOnlySpan{int}, RecomputeMode, int)"/> drives the forwards.</summary>
public enum RecomputeMode
{
    /// <summary>The backend's best mode: <see cref="SinglePass"/> on Vulkan, <see cref="PerRun"/> on the CPU.</summary>
    Auto,

    /// <summary>One <c>Forward</c> per maximal run of consecutive selected positions, in ascending order (any backend).</summary>
    PerRun,

    /// <summary>
    /// One <c>Forward(selectedIds, selectedPositions, cache)</c> of the whole (possibly non-contiguous) selection, split into ascending
    /// batches of at most <c>maxRowsPerForward</c> rows. Vulkan only: its per-query-position attention masks each row by its own position.
    /// </summary>
    SinglePass,
}

/// <summary>
/// Selective recompute (CacheBlend-style fix-up) of a naive-concatenation chunk cache over the public <see cref="IModel"/> and
/// <see cref="IKvCache"/>: the selected positions get fresh K/V at every layer, attending to the stale (per-chunk) K/V of the
/// non-selected positions and to the already refreshed K/V of earlier selected positions.
/// </summary>
/// <remarks>
/// <para>
/// <b>Per-run (the CPU path, and <see cref="Run"/>).</b> The selection is split into maximal runs of consecutive positions and each run is
/// one ordinary <c>Forward</c> with its token ids, its true positions and the assembled cache, in ascending position order. A
/// <c>Forward</c> writes the run's rows (post-RoPE K, V) per layer before attending, and the CPU attention masks a query at <c>p</c> to keys
/// <c>[0, p]</c> with <c>p = positions[0] + i</c> — exact for a contiguous run, which is why runs are not merged on the CPU (a non-contiguous
/// CPU <c>Forward</c> would mask the later tokens as if they were consecutive). Because attention is causal, an earlier run never sees a
/// later one, and a later run sees the earlier runs' refreshed K/V at every layer: the result equals one-pass layer-wise recompute of the
/// whole selection.
/// </para>
/// <para>
/// <b>Single pass (Vulkan).</b> The Vulkan model routes a forward whose positions are non-contiguous, or a run inside an already-filled cache,
/// through the per-query-position attention (dotLLM #1534), which masks each row by its own position. One forward of the whole selection
/// writes every selected row of a layer before that layer's attention, so a selected row at <c>p</c> sees the refreshed rows of the earlier
/// selected positions at the same layer and the stale rows elsewhere: the same one-pass layer-wise recompute, in one trunk pass. Batches of at
/// most <c>maxRowsPerForward</c> rows, in ascending order, keep the same semantics (a later batch sees the earlier ones' refreshed rows at
/// every layer) and bound the time of one GPU submit. The model must be a <see cref="VulkanTransformerModel"/>: after every forward that
/// needs the per-query-position kernel the recomputer checks <see cref="VulkanTransformerModel.AttentionQueryPositionsCount"/> moved by the
/// layer count and throws if it did not (a model loaded without <c>attention_f32_qpos.spv</c> falls back to the ordinary kernel, which masks
/// by <c>positions[0] + row</c>; the cache is then wrong and must be discarded).
/// </para>
/// <para>
/// Writing below <see cref="IKvCache.CurrentLength"/> is in place (<c>SimpleKvCache.Update</c>, <c>VulkanKvCache</c> row copies) and sets
/// <c>CurrentLength = max(CurrentLength, maxPos + 1)</c>; the forward never truncates. Rows after the context are left intact (checked after
/// every call: a cache whose length moves is rejected). A refused argument throws before anything is written; a failure during a forward
/// leaves the cache partly recomputed (undefined).
/// </para>
/// </remarks>
public static class SelectiveRecomputer
{
    /// <summary>
    /// Recomputes the K/V of <paramref name="selectedIndices"/> in <paramref name="cache"/>, which must already hold the
    /// assembled context of <paramref name="chunks"/> at positions <c>[0, L)</c> (e.g. from <see cref="ChunkAssembler"/>
    /// on an empty cache), <c>L = Σ chunk length</c>. CPU per-run path (the original API; device id -1).
    /// </summary>
    /// <param name="model">The model that encoded the chunks.</param>
    /// <param name="cache">The assembled cache; updated in place. Rows at positions <c>&gt;= L</c> are left untouched.</param>
    /// <param name="chunks">The chunks in context order; their concatenated token ids give the token at each position.</param>
    /// <param name="selectedIndices">Global positions in <c>[0, L)</c> to recompute, strictly ascending. Empty is a no-op.</param>
    /// <returns>Calls made, tokens recomputed and, when <c>L-1</c> was selected, the last position's logits.</returns>
    /// <exception cref="ArgumentException">The selection is not strictly ascending or out of range, a chunk is null or empty,
    /// or the cache holds fewer than <c>L</c> positions. Nothing is written in that case.</exception>
    /// <exception cref="InvalidOperationException">The cache changed its length during a forward (it does not write in place).</exception>
    public static SelectiveRecomputeResult Run(IModel model, IKvCache cache, IReadOnlyList<ChunkKvEntry> chunks,
        ReadOnlySpan<int> selectedIndices)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(cache);
        int[] tokens = ConcatTokens(chunks);
        Validate(cache, tokens.Length, selectedIndices);
        if (selectedIndices.IsEmpty)
            return new SelectiveRecomputeResult(0, 0, null);
        return PerRun(model, cache, tokens, selectedIndices, deviceId: -1, vulkan: null);
    }

    /// <summary>
    /// Backend-aware recompute of <paramref name="selectedIndices"/> over the assembled context of <paramref name="chunks"/> (see
    /// <see cref="Recompute(IModel, IChunkBackend, IKvCache, ReadOnlySpan{int}, ReadOnlySpan{int}, RecomputeMode, int)"/>).
    /// </summary>
    /// <param name="model">The model that encoded the chunks, on <paramref name="backend"/>'s device.</param>
    /// <param name="backend">The chunk backend of <paramref name="cache"/>.</param>
    /// <param name="cache">The assembled cache (a cache <paramref name="backend"/> accepts); updated in place.</param>
    /// <param name="chunks">The chunks in context order.</param>
    /// <param name="selectedIndices">Global positions in <c>[0, L)</c>, strictly ascending. Empty is a no-op.</param>
    /// <param name="mode">Per-run, single-pass, or the backend's default.</param>
    /// <param name="maxRowsPerForward">Single pass: most rows per forward (bounds one GPU submit). Ignored by per-run.</param>
    /// <returns>The outcome.</returns>
    public static SelectiveRecomputeResult Recompute(IModel model, IChunkBackend backend, IKvCache cache, IReadOnlyList<ChunkKvEntry> chunks,
        ReadOnlySpan<int> selectedIndices, RecomputeMode mode = RecomputeMode.Auto, int maxRowsPerForward = int.MaxValue)
        => Recompute(model, backend, cache, ConcatTokens(chunks), selectedIndices, mode, maxRowsPerForward);

    /// <summary>
    /// Backend-aware recompute of <paramref name="selectedIndices"/> in <paramref name="cache"/>, which must hold the assembled context
    /// <paramref name="contextTokens"/> at positions <c>[0, L)</c>. <see cref="RecomputeMode.Auto"/> is one forward of the whole selection on
    /// Vulkan (<see cref="RecomputeMode.SinglePass"/>) and the per-run loop on the CPU, bit-identical to <see cref="Run"/>.
    /// </summary>
    /// <param name="model">The model that encoded the context's chunks: a <see cref="VulkanTransformerModel"/> on the Vulkan backend, a host
    /// model on the CPU backend.</param>
    /// <param name="backend">The chunk backend of <paramref name="cache"/>; its <see cref="IChunkBackend.ForwardDeviceId"/> is used.</param>
    /// <param name="cache">The assembled cache (a cache <paramref name="backend"/> accepts); updated in place. Rows at positions
    /// <c>&gt;= L</c> are left untouched.</param>
    /// <param name="contextTokens">The token at each context position (the chunks' token ids concatenated).</param>
    /// <param name="selectedIndices">Global positions in <c>[0, L)</c>, strictly ascending. Empty is a no-op.</param>
    /// <param name="mode">Per-run, single-pass, or the backend's default.</param>
    /// <param name="maxRowsPerForward">Single pass: most rows per forward (ascending batches; bounds the time of one GPU submit). Per-run
    /// ignores it.</param>
    /// <returns>Calls made, tokens recomputed, the last position's logits when <c>L-1</c> was selected, and the per-query-position
    /// attention layer count.</returns>
    /// <exception cref="ArgumentException">Invalid selection, empty context or a cache shorter than <c>L</c>. Nothing is written.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="maxRowsPerForward"/> is not positive. Nothing is written.</exception>
    /// <exception cref="NotSupportedException">The cache is not one <paramref name="backend"/> accepts, the model does not match the backend,
    /// or <see cref="RecomputeMode.SinglePass"/> was asked of the CPU backend. Nothing is written.</exception>
    /// <exception cref="InvalidOperationException">The cache length moved during a forward, or a Vulkan forward that needed the
    /// per-query-position attention did not run it. The cache is then undefined.</exception>
    public static SelectiveRecomputeResult Recompute(IModel model, IChunkBackend backend, IKvCache cache, ReadOnlySpan<int> contextTokens,
        ReadOnlySpan<int> selectedIndices, RecomputeMode mode = RecomputeMode.Auto, int maxRowsPerForward = int.MaxValue)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(backend);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxRowsPerForward, 1);
        if (contextTokens.IsEmpty)
            throw new ArgumentException("The context is empty.", nameof(contextTokens));
        backend.RequireCache(cache);
        bool onVulkan = backend.Tag == ChunkBackendTags.Vulkan;
        VulkanTransformerModel? vulkan = model as VulkanTransformerModel;
        if (onVulkan && vulkan is null)
            throw new NotSupportedException(
                $"Recompute on the Vulkan backend needs a {nameof(VulkanTransformerModel)} (its per-query-position attention); got {model.GetType().Name}.");
        if (!onVulkan && vulkan is not null)
            throw new NotSupportedException($"The {backend.Tag} backend cannot drive a {nameof(VulkanTransformerModel)}; use the Vulkan chunk backend.");
        if (mode == RecomputeMode.Auto)
            mode = onVulkan ? RecomputeMode.SinglePass : RecomputeMode.PerRun;
        if (mode == RecomputeMode.SinglePass && !onVulkan)
            throw new NotSupportedException(
                $"Single-pass recompute needs per-query-position attention; the {backend.Tag} backend masks a forward's rows as consecutive "
                + "positions. Use RecomputeMode.PerRun (or Auto).");
        Validate(cache, contextTokens.Length, selectedIndices);
        if (selectedIndices.IsEmpty)
            return new SelectiveRecomputeResult(0, 0, null);

        return mode == RecomputeMode.SinglePass
            ? SinglePass(vulkan!, cache, contextTokens, selectedIndices, backend.ForwardDeviceId, maxRowsPerForward)
            : PerRun(model, cache, contextTokens, selectedIndices, backend.ForwardDeviceId, vulkan);
    }

    private static SelectiveRecomputeResult PerRun(IModel model, IKvCache cache, ReadOnlySpan<int> tokens, ReadOnlySpan<int> selectedIndices,
        int deviceId, VulkanTransformerModel? vulkan)
    {
        int total = tokens.Length;
        int length = cache.CurrentLength;
        int calls = 0;
        long qpos = 0;
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
            qpos += Forward(model, vulkan, cache, tokens.Slice(first, count), runPositions, deviceId, total, length, ref lastLogits);
            calls++;
            start = end;
        }
        return new SelectiveRecomputeResult(calls, selectedIndices.Length, lastLogits, qpos);
    }

    private static SelectiveRecomputeResult SinglePass(VulkanTransformerModel model, IKvCache cache, ReadOnlySpan<int> tokens,
        ReadOnlySpan<int> selectedIndices, int deviceId, int maxRowsPerForward)
    {
        int total = tokens.Length;
        int length = cache.CurrentLength;
        int calls = 0;
        long qpos = 0;
        float[]? lastLogits = null;
        for (int start = 0; start < selectedIndices.Length; start += maxRowsPerForward)
        {
            int count = Math.Min(maxRowsPerForward, selectedIndices.Length - start);
            ReadOnlySpan<int> positions = selectedIndices.Slice(start, count);
            int[] ids = new int[count];
            for (int i = 0; i < count; i++) ids[i] = tokens[positions[i]];
            qpos += Forward(model, model, cache, ids, positions, deviceId, total, length, ref lastLogits);
            calls++;
        }
        return new SelectiveRecomputeResult(calls, selectedIndices.Length, lastLogits, qpos);
    }

    /// <summary>
    /// One recompute forward; checks the cache length and, on Vulkan, that the per-query-position attention ran on every layer when the
    /// forward needs it (non-contiguous positions, or rows that do not end at the cache length). Returns the per-query-position layer count.
    /// </summary>
    private static unsafe long Forward(IModel model, VulkanTransformerModel? vulkan, IKvCache cache, ReadOnlySpan<int> ids,
        ReadOnlySpan<int> positions, int deviceId, int total, int length, ref float[]? lastLogits)
    {
        long before = vulkan?.AttentionQueryPositionsCount ?? 0;
        using (ITensor logits = model.Forward(ids, positions, deviceId, cache, lastTokenLogitsOnly: true))
        {
            if (positions[^1] == total - 1)
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
        if (vulkan is null)
            return 0;
        long ran = vulkan.AttentionQueryPositionsCount - before;
        bool tight = positions[^1] - positions[0] == positions.Length - 1 && positions[^1] + 1 == length;
        if (!tight && ran != vulkan.Config.NumLayers)
            throw new InvalidOperationException(
                $"A recompute forward of {positions.Length} rows at positions {positions[0]}..{positions[^1]} needed the per-query-position "
                + $"attention on all {vulkan.Config.NumLayers} layers but it ran on {ran} (was the model loaded without attention_f32_qpos.spv, "
                + "or does it have attention sinks?). The cache now holds wrongly masked rows; discard it.");
        return ran;
    }

    private static int[] ConcatTokens(IReadOnlyList<ChunkKvEntry> chunks)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        int total = 0;
        for (int c = 0; c < chunks.Count; c++)
        {
            int n = chunks[c]?.TokenIds?.Length ?? 0;
            if (n == 0)
                throw new ArgumentException($"Chunk {c} is null or empty.", nameof(chunks));
            total += n;
        }
        int[] tokens = new int[total];
        int offset = 0;
        foreach (ChunkKvEntry chunk in chunks)
        {
            chunk.TokenIds.CopyTo(tokens, offset);
            offset += chunk.TokenIds.Length;
        }
        return tokens;
    }

    private static void Validate(IKvCache cache, int total, ReadOnlySpan<int> selectedIndices)
    {
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
    }
}
