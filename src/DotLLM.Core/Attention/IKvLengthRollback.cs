namespace DotLLM.Core.Attention;

/// <summary>
/// Optional capability of a KV-cache whose <see cref="IKvCache.Rollback"/> is an exact length cut: every row below the new length
/// stays valid and the rows at or above it are simply rewritten by the next forward (a full-history cache, or a length-only handle
/// whose owning model re-syncs its own storage cursor to the handle's length on every forward). Unlike
/// <see cref="IPrefixReusableKvCache"/> it says nothing about adopting the cache across requests, so a hybrid cache that must not be
/// shared (<see cref="IKvCache.HasExclusiveStorage"/> false) can still declare it. A <c>GenerationSession</c> (issue #1919) rolls back
/// only caches that declare this, or <see cref="IPrefixReusableKvCache"/>; any other cache makes its <c>RollbackTo</c> refuse.
/// </summary>
public interface IKvLengthRollback : IKvCache
{
    /// <summary>True when <see cref="IKvCache.Rollback"/> to <paramref name="length"/> leaves every row below it intact.</summary>
    /// <param name="length">The length to cut back to.</param>
    bool CanTruncateTo(int length);
}
