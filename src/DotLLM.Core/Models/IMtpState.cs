namespace DotLLM.Core.Models;

/// <summary>
/// Per-sequence state for Multi-Token Prediction (MTP) self-speculative decoding (issue #253).
/// </summary>
/// <remarks>
/// <para>
/// MTP ships a lightweight extra prediction head in the <em>same</em> GGUF checkpoint as the main
/// (trunk) model — see <see cref="ModelConfig.NextnPredictLayers"/>. The head predicts several
/// future tokens from the trunk's own final hidden state, and the trunk verifies them in one extra
/// batched forward pass, exactly like <see cref="DotLLM.Core.Attention.IAttentionStrategy"/>-style
/// two-model speculative decoding — except there is no second <see cref="IModel"/>: the "draft" is
/// a single extra transformer block sharing the trunk's own weights file, with its own tiny
/// KV-cache (llama.cpp PR ggml-org/llama.cpp#22673 reports under ~10% memory overhead for it).
/// </para>
/// <para>
/// This state container is the small additional piece two-model speculative decoding doesn't need:
/// it carries the MTP head's own KV-cache (sized for exactly the trailing MTP block(s), not the
/// trunk) plus the "pending hidden" handoff row that seeds the head's next autoregressive step.
/// Mirrors llama.cpp's <c>common_speculative_state_draft_mtp</c> (<c>common/speculative.cpp</c>):
/// the trunk's pre-final-norm hidden state at each verified position is captured into
/// <see cref="CapturedHiddenRows"/>, and <see cref="SeedFromCapturedRow"/> selects the row matching
/// the last accepted position — the equivalent of llama.cpp's <c>accept()</c> picking
/// <c>i_h = min(n_accepted, n_rows - 1)</c> out of <c>verify_h</c>.
/// </para>
/// </remarks>
public interface IMtpState : IDisposable
{
    /// <summary>Number of MTP-head decode steps advanced so far (drives the MTP head's own KV-cache length).</summary>
    int CurrentLength { get; }

    /// <summary>
    /// Rolls the MTP head's own KV-cache back to <paramref name="length"/> steps, discarding any
    /// speculative (rejected) MTP-internal state beyond that point. Mirrors
    /// <see cref="DotLLM.Core.Attention.IKvCache.Rollback"/> but sized for the MTP head only —
    /// the trunk's own KV-cache is rolled back separately by the caller.
    /// </summary>
    void Rollback(int length);

    /// <summary>
    /// Captured pre-final-norm hidden state rows from the most recent target-model verify-phase
    /// forward pass, row-major <c>[<see cref="CapturedRowCount"/>, hiddenSize]</c> — one row per
    /// verified position, in the same order as that forward call's <c>tokenIds</c>/<c>positions</c>.
    /// Populated only by <see cref="IModel"/> implementations with <c>SupportsMtp == true</c>, and
    /// only when this state is passed into that model's MTP-aware <c>Forward</c> overload.
    /// </summary>
    ReadOnlySpan<float> CapturedHiddenRows { get; }

    /// <summary>Number of valid rows in <see cref="CapturedHiddenRows"/>. 0 until a verify-phase forward populates it.</summary>
    int CapturedRowCount { get; }

    /// <summary>Hidden-state width of each row in <see cref="CapturedHiddenRows"/> (the model's <see cref="ModelConfig.HiddenSize"/>).</summary>
    int HiddenSize { get; }

    /// <summary>
    /// Seeds the MTP head's next-step "pending hidden" input from
    /// <c>CapturedHiddenRows[rowIndex]</c> — call after accept/reject with the row matching the
    /// last accepted (or bonus) verified position. This is the hand-off that lets the next
    /// speculation round's first <c>ForwardMtp</c> call seed from a hidden state the trunk model
    /// actually verified, rather than one of the (possibly rejected) MTP head's own speculative
    /// hidden states.
    /// </summary>
    /// <param name="rowIndex">Row index into <see cref="CapturedHiddenRows"/>, clamped to <c>[0, CapturedRowCount)</c> by the caller.</param>
    void SeedFromCapturedRow(int rowIndex);
}
