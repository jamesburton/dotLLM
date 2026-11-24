using DotLLM.Core.Attention;
using DotLLM.Core.Tensors;

namespace DotLLM.Core.Diagnostics;

/// <summary>
/// A model whose forward pass delivers hidden-state samples to registered taps (#1920). Taps only read: the residual stream is
/// never modified (injection stays with <see cref="IResidualModifier"/>). With no tap attached the cost is one null check per layer.
/// A model that does not implement this interface is refused by type; a registration that names an unsupported
/// <see cref="HookPoint"/> throws <see cref="NotSupportedException"/>.
/// </summary>
public interface IHiddenStateTapHost
{
    /// <summary>Registers a read-only tap. Dispose the result to unregister.</summary>
    /// <exception cref="NotSupportedException">A point (or this model's architecture) is not supported.</exception>
    /// <exception cref="ArgumentException">A layer index is out of range, or no point is named.</exception>
    IDisposable RegisterTap(HiddenStateTapSubscription subscription, Action<HiddenStateSample> onSample);

    /// <summary>
    /// Registers a tap that may stop the forward pass by returning <see cref="HookResult.AbortForward"/>
    /// (<see cref="HookResult.Continue"/> otherwise; <c>Replace</c> is rejected with <see cref="InvalidOperationException"/>).
    /// On abort the model rolls the KV-cache back to its length at forward entry, restores recurrent state from a checkpoint taken
    /// at entry, and skips the remaining layers and the LM head. Refused with <see cref="NotSupportedException"/>, before any state is
    /// touched: at the first abortable forward when the supplied cache drops rows (a compact sliding-window or full-precision-window
    /// cache, <see cref="IKvCache.SupportsAbortRollback"/> false); in <c>ForwardBatch</c>; and with an MTP/DFlash state. A model with
    /// sliding-window attention layers on a full-history cache is allowed (no rows are dropped). Registration cannot see the per-call
    /// cache; use <see cref="RegisterAbortableTap(HiddenStateTapSubscription, Func{HiddenStateSample, HookResult}, IKvCache)"/> to
    /// validate the cache kind up front.
    /// A throwing callback (any tap) restores KV and recurrent state to forward entry before the exception propagates.
    /// </summary>
    IDisposable RegisterAbortableTap(HiddenStateTapSubscription subscription, Func<HiddenStateSample, HookResult> onSample);

    /// <summary>
    /// As <see cref="RegisterAbortableTap(HiddenStateTapSubscription, Func{HiddenStateSample, HookResult})"/>, but also validates the
    /// cache kind immediately, before anything is registered: refused when <paramref name="cache"/> drops rows (compact sliding-window or
    /// full-precision-window cache, <see cref="IKvCache.SupportsAbortRollback"/> false), because an abort could not restore it.
    /// A full-history cache is accepted even for a model with sliding-window layers. Default implementation validates, then registers.
    /// </summary>
    /// <param name="subscription">What to sample.</param>
    /// <param name="onSample">The callback.</param>
    /// <param name="cache">The cache the tap will run against.</param>
    /// <exception cref="NotSupportedException">The cache cannot roll back; the message names the cache type.</exception>
    IDisposable RegisterAbortableTap(HiddenStateTapSubscription subscription, Func<HiddenStateSample, HookResult> onSample, IKvCache cache)
    {
        AbortableTapKvGuard.ThrowIfCacheCannotAbort(cache);
        return RegisterAbortableTap(subscription, onSample);
    }

    /// <summary>
    /// Forward pass that reports an abort as <see langword="false"/> instead of throwing <see cref="ForwardAbortedException"/>
    /// (the hot-path form for per-token early exit).
    /// </summary>
    /// <returns><see langword="true"/> with <paramref name="logits"/> when the pass completed; <see langword="false"/> with <paramref name="abort"/> when aborted.</returns>
    bool TryForward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                    out ITensor? logits, out ForwardAbortInfo abort);
}
