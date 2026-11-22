namespace DotLLM.Core.Diagnostics;

/// <summary>Which rows of the activation a hidden-state tap receives.</summary>
public enum TapRows
{
    /// <summary>Only the last row of the forward (decode-friendly; a prefill copies one row).</summary>
    Last,

    /// <summary>Every row of the forward, one sample per row.</summary>
    All,
}

/// <summary>
/// What a hidden-state tap observes (#1920): where in the forward pass, which layers, which rows.
/// </summary>
/// <param name="Layers">
/// Layer indices to observe for <see cref="HookPoint.PostLayer"/>; <see langword="null"/> means every layer.
/// Validated against the model at registration. Ignored for the points that are not per-layer.
/// </param>
/// <param name="Points">
/// Points to observe. Supported: <see cref="HookPoint.PostEmbedding"/>, <see cref="HookPoint.PostLayer"/>,
/// <see cref="HookPoint.PreFinalNorm"/> and <see cref="HookPoint.PreLmHead"/> (post-final-norm: the exact input of the LM head).
/// Any other point throws <see cref="NotSupportedException"/> at registration.
/// </param>
/// <param name="Rows">Which rows to deliver; default <see cref="TapRows.Last"/>.</param>
/// <param name="BufferProvider">
/// Optional caller-owned destination: called with the float count needed (the hidden size) and must return a
/// <see cref="Memory{T}"/> of at least that length, which then backs <see cref="HiddenStateSample.Data"/>.
/// When <see langword="null"/> a pooled buffer is used.
/// </param>
public sealed record HiddenStateTapSubscription(
    int[]? Layers,
    HookPoint[] Points,
    TapRows Rows = TapRows.Last,
    Func<int, Memory<float>>? BufferProvider = null);

/// <summary>
/// One delivered activation row. <see cref="Data"/> is pooled or caller-owned and valid only for the duration of the callback.
/// </summary>
/// <param name="Point">Where in the forward pass the row was taken.</param>
/// <param name="LayerIndex">Layer index for <see cref="HookPoint.PostLayer"/>; -1 for every other point.</param>
/// <param name="RowIndex">Row within the forward's token rows (0-based).</param>
/// <param name="TokenPosition">Absolute position of the token the row belongs to.</param>
/// <param name="SequenceId">Sequence identifier (index within a batch; 0 for single-sequence forwards).</param>
/// <param name="Step">Per-model counter of forwards that ran with a tap attached (0-based).</param>
/// <param name="Data">The hidden-size activation row, F32 (the public hook contract).</param>
public readonly record struct HiddenStateSample(
    HookPoint Point,
    int LayerIndex,
    int RowIndex,
    int TokenPosition,
    int SequenceId,
    int Step,
    ReadOnlyMemory<float> Data);

/// <summary>Where an aborted forward stopped.</summary>
/// <param name="LayerIndex">Layer index of the abort site (-1 for non-layer points).</param>
/// <param name="Point">The point whose tap returned <see cref="HookResult.AbortForward"/>.</param>
public readonly record struct ForwardAbortInfo(int LayerIndex, HookPoint Point);

/// <summary>
/// Thrown by the legacy <c>Forward</c> entry points when an abortable tap aborted the pass. The model has already rolled its
/// KV-cache and recurrent state back to their values at entry. Use <see cref="IHiddenStateTapHost.TryForward"/> to avoid the exception.
/// </summary>
public sealed class ForwardAbortedException : Exception
{
    /// <summary>Creates the exception for <paramref name="info"/>.</summary>
    public ForwardAbortedException(ForwardAbortInfo info)
        : base($"Forward aborted by a hidden-state tap at {info.Point} (layer {info.LayerIndex}).")
    {
        Layer = info.LayerIndex;
        Point = info.Point;
    }

    /// <summary>Layer index of the abort site (-1 for non-layer points).</summary>
    public int Layer { get; }

    /// <summary>The point whose tap aborted.</summary>
    public HookPoint Point { get; }
}
