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
/// Mirrors llama.cpp's <c>common_speculative_impl_draft_mtp</c> (<c>common/speculative.cpp</c>):
/// the trunk's post-final-norm hidden state at each forwarded position is captured into
/// <see cref="CapturedHiddenRows"/>, every forwarded token is absorbed into the head's KV-cache
/// (indexed by sequence position) paired with the hidden state of the position before it, and
/// <see cref="SeedFromCapturedRow"/> selects the row matching the last accepted position — the
/// equivalent of llama.cpp's <c>pending_h</c> hand-off (issue #469).
/// </para>
/// </remarks>
public interface IMtpState : IDisposable
{
    /// <summary>Number of sequence positions the MTP head's KV-cache covers (its next write slot).</summary>
    int CurrentLength { get; }

    /// <summary>
    /// Rolls the MTP head's own KV-cache back to <paramref name="length"/> steps, discarding any
    /// speculative (rejected) MTP-internal state beyond that point. Mirrors
    /// <see cref="DotLLM.Core.Attention.IKvCache.Rollback"/> but sized for the MTP head only —
    /// the trunk's own KV-cache is rolled back separately by the caller.
    /// </summary>
    void Rollback(int length);

    /// <summary>
    /// Captured post-final-norm hidden state rows from the most recent target-model
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
    /// Seeds the MTP head's next-step "pending hidden" input — and the carried row the next absorb
    /// pairs its first token with — from <c>CapturedHiddenRows[rowIndex]</c>. Call after
    /// accept/reject with the row of the last accepted verified position. This is the hand-off that lets the next
    /// speculation round's first <c>ForwardMtp</c> call seed from a hidden state the trunk model
    /// actually verified, rather than one of the (possibly rejected) MTP head's own speculative
    /// hidden states.
    /// </summary>
    /// <param name="rowIndex">Row index into <see cref="CapturedHiddenRows"/>, clamped to <c>[0, CapturedRowCount)</c> by the caller.</param>
    void SeedFromCapturedRow(int rowIndex);

    /// <summary>
    /// Marks the head at its current length so a later request can resume from here (issue #1088): the recurrent
    /// prefix snapshot keeps this state alive with the trunk's KV, and a prefix hit then drafts without re-absorbing
    /// the cached prefix. The mark records <see cref="CurrentLength"/> and the trunk hidden row the next absorb pairs
    /// its first token with. Rows <c>[0, CurrentLength)</c> must stay untouched until the resume, which holds because
    /// every later write is at or beyond the mark.
    /// </summary>
    /// <returns>An opaque mark for <see cref="TryResumeAt"/>, or <see langword="null"/> when this state cannot resume.</returns>
    object? MarkPrefix() => null;

    /// <summary>
    /// Rolls the head back to a mark taken by <see cref="MarkPrefix"/> on this same state and restores the hidden row the
    /// next absorb pairs with, so the next trunk <c>Forward</c> absorbs from the mark exactly as a cold run would.
    /// </summary>
    /// <param name="mark">A mark from <see cref="MarkPrefix"/>.</param>
    /// <param name="maxSequenceLength">The trunk KV's current <c>MaxLength</c>: the head's ceiling is raised to it, since a prefix hit may have grown the trunk's cache.</param>
    /// <returns><see langword="false"/> when the mark is not this state's, or the head no longer covers it; the state is then unchanged.</returns>
    bool TryResumeAt(object mark, int maxSequenceLength) => false;
}

/// <summary>
/// The resume point <see cref="IMtpState.MarkPrefix"/> records (issue #1088): the head's length and the trunk hidden row
/// at the last absorbed position (<c>h_{Length-1}</c>, zero when nothing was absorbed).
/// </summary>
/// <param name="Owner">The state the mark belongs to; another state's rows are not these.</param>
/// <param name="Length">The head's <see cref="IMtpState.CurrentLength"/> when marked.</param>
/// <param name="CarryHidden">A copy of the hidden row the next absorb pairs its first token with.</param>
public sealed record MtpPrefixMark(IMtpState Owner, int Length, float[] CarryHidden)
{
    /// <summary>True when <paramref name="mark"/> is a mark of <paramref name="state"/> that its committed rows still cover.</summary>
    /// <param name="mark">The candidate mark.</param>
    /// <param name="state">The state being resumed.</param>
    /// <param name="hiddenSize">The state's hidden size, which the carried row must match.</param>
    /// <returns>The typed mark, or <see langword="null"/> when it cannot be resumed.</returns>
    public static MtpPrefixMark? For(object mark, IMtpState state, int hiddenSize)
        => mark is MtpPrefixMark m && ReferenceEquals(m.Owner, state) && m.Length <= state.CurrentLength
           && m.CarryHidden.Length == hiddenSize
            ? m : null;
}
