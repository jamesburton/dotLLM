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
    /// at entry (only while an abortable tap is attached), and skips the remaining layers and the LM head.
    /// </summary>
    IDisposable RegisterAbortableTap(HiddenStateTapSubscription subscription, Func<HiddenStateSample, HookResult> onSample);

    /// <summary>
    /// Forward pass that reports an abort as <see langword="false"/> instead of throwing <see cref="ForwardAbortedException"/>
    /// (the hot-path form for per-token early exit).
    /// </summary>
    /// <returns><see langword="true"/> with <paramref name="logits"/> when the pass completed; <see langword="false"/> with <paramref name="abort"/> when aborted.</returns>
    bool TryForward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache,
                    out ITensor? logits, out ForwardAbortInfo abort);
}
