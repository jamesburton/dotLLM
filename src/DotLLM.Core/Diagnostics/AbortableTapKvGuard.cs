using DotLLM.Core.Attention;

namespace DotLLM.Core.Diagnostics;

/// <summary>
/// Forward-entry gate for abortable hidden-state taps (#1920): proves, before anything is written, that an abort can restore the
/// KV-cache to its entry length. Registration cannot do this (it never sees the cache), so every host calls
/// <see cref="Checkpoint"/> at forward entry, and only while an abortable tap is attached.
/// </summary>
public static class AbortableTapKvGuard
{
    /// <summary>
    /// <c>ForwardBatch</c> refusal (#1920, review I-6): an abort of request i would leave requests 0..i-1 already committed, so an
    /// abortable tap is refused before any state changes, identically on every backend. Read-only taps are unaffected.
    /// </summary>
    /// <param name="taps">The snapshot the batch read once (null: no tap attached).</param>
    /// <exception cref="NotSupportedException">An abortable tap is attached.</exception>
    public static void RefuseAbortableBatch(HiddenStateTapRegistry.Snapshot? taps)
    {
        if (taps is { HasAbortable: true })
            throw new NotSupportedException(
                "An abortable hidden-state tap is not supported in ForwardBatch: aborting request i would leave requests 0..i-1 "
                + "committed and the rest unrun. Use Forward/TryForward per sequence, or a read-only tap. Nothing was modified.");
    }

    /// <summary>
    /// Cache-kind refusal (#1920, ruling 3 as narrowed 2026-11-23): an abortable tap is refused only when the cache itself DROPS rows
    /// (compact sliding-window ring/compaction, full-precision-window eviction), i.e. <see cref="IKvCache.SupportsAbortRollback"/> is
    /// false. A model with sliding-window attention layers on a full-history cache never drops rows and may abort. Shared by the
    /// registration overload (<c>RegisterAbortableTap(subscription, callback, cache)</c>) and <see cref="Checkpoint"/>.
    /// </summary>
    /// <param name="cache">The cache the tap will run against.</param>
    /// <exception cref="NotSupportedException">The cache cannot roll back after a window is crossed.</exception>
    public static void ThrowIfCacheCannotAbort(IKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if (!cache.SupportsAbortRollback)
            throw new NotSupportedException(
                $"An abortable hidden-state tap cannot be used with this {cache.GetType().Name}: it has a sliding-window or "
                + "full-precision window, so once the window is crossed an abort could not roll back to the entry length. "
                + "Refused now (independent of the current length) instead of failing mid-session; use a cache without windows "
                + "(KvGeometry.FromConfig) or a read-only tap. Nothing was modified.");
    }

    /// <summary>
    /// Returns the entry KV length to roll back to on abort, or throws <see cref="NotSupportedException"/> (before any KV, recurrent or
    /// device state is touched) when the cache cannot roll back to it now (probe: a no-op <see cref="IKvCache.Rollback"/>) or after
    /// this forward appends <paramref name="positions"/> (<see cref="IKvCache.CanRollbackAfterAppend"/>: a compact sliding-window
    /// cache drops rows during the append, after which the entry length is unreachable).
    /// </summary>
    /// <param name="cache">The forward's KV-cache (null: nothing to roll back, returns -1).</param>
    /// <param name="positions">Absolute positions this forward writes.</param>
    /// <returns>The cache length at entry, or -1 when <paramref name="cache"/> is null.</returns>
    /// <exception cref="NotSupportedException">The cache cannot undo this forward.</exception>
    public static int Checkpoint(IKvCache? cache, ReadOnlySpan<int> positions)
    {
        if (cache is null) return -1;
        ThrowIfCacheCannotAbort(cache);
        int entry = cache.CurrentLength;
        try { cache.Rollback(entry); }   // no-op probe: refuses a cache that cannot truncate at all, or already dropped rows
        catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException)
        {
            throw new NotSupportedException(
                $"An abortable hidden-state tap needs a KV-cache that supports Rollback; {cache.GetType().Name} refused.", ex);
        }
        if (positions.Length > 0)
        {
            int min = positions[0], max = positions[0];
            for (int i = 1; i < positions.Length; i++)
            {
                if (positions[i] < min) min = positions[i];
                if (positions[i] > max) max = positions[i];
            }
            if (!cache.CanRollbackAfterAppend(min, max))
                throw new NotSupportedException(
                    $"An abortable hidden-state tap cannot be used with this {cache.GetType().Name} (entry length {entry}, writing "
                    + $"positions {min}..{max}): its sliding-window layers would drop rows during this forward, so an abort could not "
                    + "roll back to the entry length. Use a cache without compact windows (KvGeometry.FromConfig) or a read-only tap. "
                    + "Nothing was modified.");
        }
        return entry;
    }
}
