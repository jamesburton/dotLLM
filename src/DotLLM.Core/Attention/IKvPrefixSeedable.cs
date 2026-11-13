namespace DotLLM.Core.Attention;

/// <summary>
/// A KV cache that can take its first rows from another cache of the same kind before a prefill, without a forward (issue #1780):
/// the agent prefix policy seeds a fresh cache from a protected entry's shared rows instead of running a diverged request cold.
/// The copy is in the source's own element format (no widening) and leaves the source untouched.
/// </summary>
public interface IKvPrefixSeedable : IKvCache
{
    /// <summary>
    /// Copies rows <c>[0, <paramref name="rows"/>)</c> of every layer from <paramref name="source"/> into this cache and sets
    /// <see cref="IKvCache.CurrentLength"/> to <paramref name="rows"/>. Returns <see langword="false"/>, with this cache unchanged,
    /// when the source is not a compatible cache (another type, device, element format or geometry, compact sliding windows) or holds
    /// fewer rows, or this cache cannot hold them.
    /// </summary>
    /// <param name="source">The cache whose leading rows to copy; not modified.</param>
    /// <param name="rows">Rows to copy.</param>
    /// <returns>Whether the rows were copied.</returns>
    bool TrySeedPrefixFrom(IKvCache source, int rows);
}
