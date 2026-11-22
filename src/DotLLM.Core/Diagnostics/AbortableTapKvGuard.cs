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
