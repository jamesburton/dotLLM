using System.Diagnostics;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.ChunkCache;

/// <summary>What a recompute strategy did.</summary>
/// <param name="ForwardCalls">Forward passes made.</param>
/// <param name="TokensRecomputed">Positions whose K/V were recomputed.</param>
/// <param name="Elapsed">Wall time.</param>
internal sealed record RecomputeOutcome(int ForwardCalls, int TokensRecomputed, TimeSpan Elapsed);

/// <summary>
/// How <c>compare</c> recomputes the selected positions of an assembled (naive-concatenation) cache. The seam a backend plugs
/// into: today only the CPU per-run strategy exists on this branch; a single-pass Vulkan strategy (one forward over the whole
/// selection with a position-aware mask, jamesburton/dotLLM#1531) implements this interface when that work is merged, and every
/// timing <c>compare</c> prints names the strategy (<see cref="Path"/>) that produced it.
/// </summary>
internal interface IRecomputeStrategy
{
    /// <summary>Short name printed next to each timing, e.g. <c>cpu-per-run</c>.</summary>
    string Path { get; }

    /// <summary>One line describing the strategy.</summary>
    string Description { get; }

    /// <summary>Recomputes <paramref name="selection"/> (strictly ascending positions) in <paramref name="cache"/> in place.</summary>
    RecomputeOutcome Recompute(IModel model, IKvCache cache, IReadOnlyList<ChunkKvEntry> entries, int[] selection);
}

/// <summary>
/// The CPU path: <see cref="SelectiveRecomputer.Run"/>, one ordinary <c>Forward</c> per maximal run of consecutive selected
/// positions, in ascending order. Exact (equals one-pass layer-wise recompute) but slow for scattered selections: a chunk-starts
/// selection at budget b over k chunks pays about k forwards, each a full trunk pass.
/// </summary>
internal sealed class CpuPerRunRecompute : IRecomputeStrategy
{
    public string Path => "cpu-per-run";

    public string Description => "CPU, one Forward per contiguous run of selected positions (SelectiveRecomputer); scattered selections pay one trunk pass per run";

    public RecomputeOutcome Recompute(IModel model, IKvCache cache, IReadOnlyList<ChunkKvEntry> entries, int[] selection)
    {
        var sw = Stopwatch.StartNew();
        SelectiveRecomputeResult r = SelectiveRecomputer.Run(model, cache, entries, selection);
        return new RecomputeOutcome(r.ForwardCalls, r.TokensRecomputed, sw.Elapsed);
    }
}

/// <summary>
/// A read-only view of a store for <c>compare</c>: hits are read, misses are encoded by the assembler but not written (so a
/// comparison never needs the store lock or fills the store with chunks of a one-off selection). Run <c>build</c> first for
/// warm timings.
/// </summary>
internal sealed class ReadOnlyChunkStore(IChunkStore inner) : IChunkStore
{
    /// <summary>Puts that were dropped.</summary>
    public int DroppedPuts { get; private set; }

    public bool TryGet(ChunkKey key, out ChunkKvEntry entry) => inner.TryGet(key, out entry);

    public void Put(ChunkKey key, ChunkKvEntry entry) => DroppedPuts++;

    public bool Contains(ChunkKey key) => inner.Contains(key);

    public IEnumerable<ChunkKey> Keys() => inner.Keys();

    public long TotalBytes => inner.TotalBytes;

    public void Evict(ChunkKey key)
    {
        // Read-only: a rejected entry stays; the assembler then uses its fresh encode.
    }
}
