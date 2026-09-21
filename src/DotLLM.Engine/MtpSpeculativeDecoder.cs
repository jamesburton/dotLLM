using System.Buffers;
using System.Diagnostics;
using System.Numerics.Tensors;
using DotLLM.Core.Attention;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Constraints;
using DotLLM.Engine.Samplers;

namespace DotLLM.Engine;

/// <summary>
/// Implements Multi-Token Prediction (MTP) self-speculative decoding (issue #253): the same
/// draft-verify-accept shape as <see cref="SpeculativeDecoder"/>, but the "draft" phase calls the
/// target model's own <see cref="IModel.ForwardMtp"/> (its lightweight extra head) instead of a
/// second model's <see cref="IModel.Forward(ReadOnlySpan{int}, ReadOnlySpan{int}, int, IKvCache?)"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Greedy-only, matching both this project's existing gate and llama.cpp's own current MTP
/// maturity.</b> <see cref="SpeculativeDecoder"/> already restricts probabilistic (modified
/// rejection sampling) acceptance to a future change (issue #121) because <c>q</c>/<c>p</c> must
/// come from the same post-transform distribution the sampler pipeline actually draws from. MTP
/// inherits that same constraint AND, independently, llama.cpp's own merged MTP draft
/// implementation (<c>common_speculative_state_draft_mtp::draft()</c>, PR
/// ggml-org/llama.cpp#22673) currently hardcodes <c>sparams.top_k = 1</c> for the MTP head's own
/// sampler with an explicit <c>// TODO: re-enable top_k == 10 and utilize p_min spec param</c> —
/// i.e. upstream's own MTP draft is greedy-argmax today too. This decoder matches that: it throws
/// for <c>greedy: false</c> for the same reason <see cref="SpeculativeDecoder"/> does.
/// </para>
/// <para>
/// <b>Distributional correctness.</b> In greedy mode the accepted sequence is identical to what
/// argmax-decoding the target model (without MTP) would produce: every accepted or corrected
/// token in <c>outputBuffer</c> is always the target model's OWN argmax at that position (MTP
/// never gets to inject a token the target didn't independently agree with) — see
/// <c>MtpSpeculativeDecoderTests</c> for the test that demonstrates token-for-token equivalence
/// against plain greedy decode of the same (synthetic) model.
/// </para>
/// <para>
/// <b>One trunk forward per round, matching llama.cpp's <c>draft-mtp</c> (issue #469).</b> The
/// head is trained to predict <c>x_{p+1}</c> from the pair <c>(h_{p-1}, embed(x_p))</c>: the
/// hidden state that PREDICTED <c>x_p</c>, plus <c>x_p</c> itself. So a round needs no forward of
/// <c>lastToken</c> before drafting — its partner <c>h_{position-1}</c> is the last committed row of
/// the previous verify batch (or of the prefill), which <see cref="IMtpState.SeedFromCapturedRow"/>
/// leaves in the state. The round then drafts <c>d1..dK</c>, and verifies <c>[lastToken, d1..dK]</c>
/// in a single forward of K+1 rows: row <c>i</c> checks draft <c>i+1</c>, the first mismatch is
/// replaced by the target's own token, and row K supplies the bonus when every draft holds.
/// </para>
/// <para>
/// An earlier version paired <c>(h_p, embed(x_p))</c> — citing <c>graph_mtp</c>, which does no
/// shifting; llama.cpp's caller does. That made draft 1 re-predict a token the trunk had already
/// produced, and forced a separate single-token forward of <c>lastToken</c> every round: two trunk
/// forwards per round, which is why MTP never beat plain decode. Teacher-forced on Bonsai 2 27B,
/// the useful draft was right 50% of the time under the old convention and 74% under this one.
/// </para>
/// <para>
/// <b>The head's own KV-cache holds the whole sequence.</b> Every trunk forward that carries the
/// <see cref="IMtpState"/> also absorbs its tokens into the head, so the head attends over the full
/// history rather than only this round's drafts. Draft steps write speculative rows after the
/// committed prefix; they are rolled back before the verify forward absorbs the real ones.
/// </para>
/// <para>
/// <b>Recurrent trunks (issue #287).</b> A Gated DeltaNet/Mamba state is a sequential recurrence
/// with no positions to roll back, so the verify forward is bracketed by a recurrent-state
/// checkpoint; on a rejection the state is restored and the committed prefix replayed (without the
/// MTP state — the head already absorbed those rows). llama.cpp avoids that replay with per-token
/// state snapshots; that remains a follow-up here.
/// </para>
/// </remarks>
public sealed class MtpSpeculativeDecoder : IMtpSpeculativeDecoder
{
    private readonly bool _greedy;

    /// <summary>
    /// Creates a new MTP self-speculative decoder.
    /// </summary>
    /// <param name="greedy">Must be <c>true</c> — see the greedy-only remarks on this type.</param>
    /// <exception cref="NotSupportedException">Thrown when <paramref name="greedy"/> is <c>false</c>.</exception>
    public MtpSpeculativeDecoder(bool greedy)
    {
        if (!greedy)
        {
            throw new NotSupportedException(
                "Probabilistic MTP self-speculative decoding is not yet distributionally correct " +
                "under the sampler pipeline, and llama.cpp's own merged MTP draft implementation is " +
                "greedy-only today too (top_k=1, see the type remarks). Use greedy mode.");
        }
        _greedy = greedy;
    }

    /// <inheritdoc/>
    public SpeculativeResult DraftAndVerify(
        IModel targetModel,
        IKvCache kvCacheTarget,
        IMtpState mtpState,
        SamplerPipeline pipeline,
        List<int> generatedIds,
        IDecodingConstraint? constraint,
        int position,
        int vocabSize,
        int numCandidates,
        Span<int> outputBuffer)
    {
        if (!targetModel.SupportsMtp)
            throw new ArgumentException(
                $"{targetModel.GetType().Name} does not support MTP (SupportsMtp=false). " +
                "Check SupportsMtp before constructing an MtpSpeculativeDecoder round.",
                nameof(targetModel));

        // The verify forward writes positions position..position+k, so k is bounded by the target
        // KV-cache (the MTP state is sized for the whole sequence by the caller).
        int maxPos = kvCacheTarget.MaxLength;
        int k = Math.Min(numCandidates, maxPos - position - 1);
        if (k <= 0)
            return default;

        int lastToken = generatedIds[^1];

        long draftTicks = 0;
        long verifyTicks = 0;

        IDecodingConstraint? draftConstraint = constraint?.Clone();
        int[] draftTokens = ArrayPool<int>.Shared.Rent(k);
        object? gdnCheckpoint = null;

        try
        {
            // ── Draft: the head's pending hidden is h_{position-1}, carried from the previous
            //    round's last accepted verify row (or the prefill's last row), and the first step
            //    pairs it with lastToken at `position` — llama.cpp's (pending_h, id_last). Each
            //    later step pairs the head's own output with the token it just drafted. ──
            int originalGenCount = generatedIds.Count;
            int draftToken = lastToken;
            try
            {
                for (int i = 0; i < k; i++)
                {
                    long fwdStart = Stopwatch.GetTimestamp();
                    using ITensor draftLogits = targetModel.ForwardMtp(mtpState, draftToken, position + i);
                    draftTicks += Stopwatch.GetTimestamp() - fwdStart;

                    unsafe
                    {
                        var logitSpan = new Span<float>((void*)draftLogits.DataPointer, vocabSize);
                        if (draftConstraint != null)
                            TokenMaskApplier.Apply(logitSpan, draftConstraint.GetAllowedTokens());
                        draftToken = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)logitSpan);
                    }

                    draftTokens[i] = draftToken;
                    draftConstraint?.Advance(draftToken);
                    generatedIds.Add(draftToken);
                }
            }
            finally
            {
                if (generatedIds.Count > originalGenCount)
                    generatedIds.RemoveRange(originalGenCount, generatedIds.Count - originalGenCount);
            }

            // The draft steps wrote speculative rows into the head's KV-cache from `position`
            // onward; the verify forward below absorbs the real ones in their place.
            mtpState.Rollback(position);

            // Issue #287: the verify forward advances a recurrent (GDN) trunk for every row before
            // accept/reject is known, and a sequential recurrence has no positions to roll back.
            // Checkpoint first so a partial rejection can restore and replay the accepted prefix.
            gdnCheckpoint = targetModel.SupportsRecurrentStateCheckpoint
                ? targetModel.CheckpointRecurrentState()
                : null;

            // ── Verify: ONE trunk forward over [lastToken, d1..dk] at position..position+k.
            //    Row i predicts the token after position+i, so it checks draft i+1; row k is the
            //    bonus when every draft is accepted. Passing mtpState absorbs the batch into the
            //    head (pairing each token with the previous row), exactly like the prefill. ──
            int verifyLen = k + 1;
            Span<int> verifyTokens = verifyLen <= 16 ? stackalloc int[verifyLen] : new int[verifyLen];
            Span<int> verifyPositions = verifyLen <= 16 ? stackalloc int[verifyLen] : new int[verifyLen];
            verifyTokens[0] = lastToken;
            verifyPositions[0] = position;
            for (int i = 0; i < k; i++)
            {
                verifyTokens[i + 1] = draftTokens[i];
                verifyPositions[i + 1] = position + i + 1;
            }

            long verifyStart = Stopwatch.GetTimestamp();
            using ITensor targetLogits = targetModel.Forward(
                verifyTokens, verifyPositions, deviceId: -1, kvCacheTarget, adapter: null, mtpState);
            verifyTicks += Stopwatch.GetTimestamp() - verifyStart;

            int accepted = 0;       // drafts accepted
            int acceptedCount = 0;  // tokens written to outputBuffer (accepted drafts + correction/bonus)
            unsafe
            {
                nint basePtr = targetLogits.DataPointer;
                for (int row = 0; row <= k; row++)
                {
                    var rowLogits = new Span<float>(
                        (void*)(basePtr + (long)row * vocabSize * sizeof(float)), vocabSize);
                    if (constraint != null)
                        TokenMaskApplier.Apply(rowLogits, constraint.GetAllowedTokens());
                    int targetArgmax = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)rowLogits);

                    outputBuffer[acceptedCount++] = targetArgmax;
                    constraint?.Advance(targetArgmax);

                    if (row == k || draftTokens[row] != targetArgmax)
                        break;      // correction (or bonus): the target's own token ends the round
                    accepted++;
                }
            }

            // Tokens [lastToken, d1..d_accepted] at position..position+accepted are now committed.
            int committedEnd = position + accepted + 1;
            if (committedEnd <= kvCacheTarget.CurrentLength)
                kvCacheTarget.Rollback(committedEnd);
            if (committedEnd <= mtpState.CurrentLength)
                mtpState.Rollback(committedEnd);
            // The next round pairs its first token (the correction/bonus just emitted) with the
            // hidden state of the last committed position: verify row `accepted`.
            mtpState.SeedFromCapturedRow(accepted);

            if (accepted < k && gdnCheckpoint is not null)
            {
                // The recurrent trunk advanced through rejected rows. Restore it and replay the
                // committed prefix — WITHOUT mtpState, the head already absorbed those rows.
                targetModel.RestoreRecurrentState(gdnCheckpoint);
                kvCacheTarget.Rollback(position);
                using ITensor _ = targetModel.Forward(
                    verifyTokens.Slice(0, accepted + 1), verifyPositions.Slice(0, accepted + 1),
                    deviceId: -1, kvCacheTarget, adapter: null, mtpState: null);
            }

            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
        }
        finally
        {
            (gdnCheckpoint as IDisposable)?.Dispose();
            ArrayPool<int>.Shared.Return(draftTokens);
        }
    }
}
