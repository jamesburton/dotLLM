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
    /// Registration-time refusal (#1920, ruling 3): an abortable tap on a model whose config uses sliding-window attention is refused
    /// up front. Registration cannot see the per-call cache, so this is the config half; <see cref="Checkpoint"/> is the cache half.
    /// </summary>
    /// <param name="config">The model's configuration.</param>
    /// <exception cref="NotSupportedException">Some layer uses a sliding window.</exception>
    public static void ThrowIfWindowedConfig(DotLLM.Core.Models.ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        for (int l = 0; l < config.NumLayers; l++)
        {
            if (config.SlidingWindowOf(l) is int w && w > 0)
                throw new NotSupportedException(
                    $"An abortable hidden-state tap cannot be registered on a sliding-window model (layer {l} uses a window of {w}): "
                    + "an abort could not always roll the windowed KV-cache back. Use a read-only tap, or a model without sliding windows.");
        }
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
        if (!cache.SupportsAbortRollback)
            throw new NotSupportedException(
                $"An abortable hidden-state tap cannot be used with this {cache.GetType().Name}: it has a sliding-window or "
                + "full-precision window, so once the window is crossed an abort could not roll back to the entry length. "
                + "Refused now (independent of the current length) instead of failing mid-session; use a cache without windows "
                + "(KvGeometry.FromConfig) or a read-only tap. Nothing was modified.");
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
