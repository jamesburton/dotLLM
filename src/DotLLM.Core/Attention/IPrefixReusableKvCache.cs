namespace DotLLM.Core.Attention;

/// <summary>
/// Optional capability of a KV-cache that a later request may adopt after the request that filled it finished: the cache
/// keeps its rows after the request ends, owns its storage exclusively (<see cref="IKvCache.HasExclusiveStorage"/>), and can
/// be cut back to a shorter length with <see cref="IKvCache.Rollback"/>. The engine's <c>PrefixCache</c> uses it to reuse the
/// longest common prefix of two sequential prompts without a paged block pool (the Vulkan single-request path, issue #1591).
/// </summary>
public interface IPrefixReusableKvCache : IKvCache
{
    /// <summary>
    /// True when <see cref="IKvCache.Rollback"/> to <paramref name="length"/> leaves every row below it intact: always for a
    /// full-history cache; for compact sliding-window layers only while they still hold the rows a query at that position needs.
    /// </summary>
    /// <param name="length">The length to cut back to.</param>
    /// <returns>Whether the rollback is safe.</returns>
    bool CanTruncateTo(int length);
}
