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
/// <b>Greedy and sampled.</b> The draft is always the head's argmax (a point mass), so exact speculative sampling is cheap: a draft <c>d</c> at a
/// verify row with prepared distribution <c>p</c> (processors, temperature, top-k/p/min-p applied by the <see cref="SamplerPipeline"/>) is accepted with
/// probability <c>p(d)</c>, and a rejected position is resampled from <c>p</c> with <c>d</c> removed and renormalised; the all-accepted bonus row is
/// sampled from its <c>p</c>. Every emitted token is then distributed exactly as the target alone would sample it (for <c>x != d</c>:
/// <c>(1 - p(d)) * p(x) / (1 - p(d)) = p(x)</c>). With a greedy pipeline this degenerates to accept-iff-argmax and the accepted sequence is identical
/// to argmax-decoding the target without MTP; <c>MtpSpeculativeDecoderTests</c> checks both token-for-token and distributionally.
/// A sampled run is NOT bit-reproducible against the same seed with MTP off (different random draws), only equal in distribution.
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
/// MTP state — the head already absorbed those rows). A model that reports
/// <see cref="IModel.SupportsRecurrentRowSnapshots"/> instead records its state after every verify
/// row and rolls back to row <c>accepted</c> directly, like llama.cpp's <c>n_rs_seq</c> (issue
/// #473): no checkpoint copy before the verify and no replay forward after a rejection.
/// </para>
/// </remarks>
public sealed class MtpSpeculativeDecoder : IMtpSpeculativeDecoder
{
    /// <summary>Env switch that forces checkpoint + replay even on snapshot-capable models.</summary>
    internal const string DisableRowSnapshotsEnvVar = "DOTLLM_MTP_GDN_SNAPSHOTS";

    /// <summary>
    /// Use per-row recurrent snapshots when the model supports them (issue #473). Defaults to on
    /// unless <c>DOTLLM_MTP_GDN_SNAPSHOTS=0</c>; tests and the A/B harness flip it in-process.
    /// </summary>
    internal bool UseRecurrentRowSnapshots { get; set; } =
        Environment.GetEnvironmentVariable(DisableRowSnapshotsEnvVar) != "0";

    /// <summary>Env switch that forces full-logits draft steps even on models with a native draft argmax.</summary>
    internal const string DisableDraftArgMaxEnvVar = "DOTLLM_MTP_DRAFT_ARGMAX";

    /// <summary>
    /// Take the draft token from <see cref="IModel.ForwardMtpArgMax"/> when the draft is unconstrained
    /// and the model reports <see cref="IModel.SupportsMtpArgMax"/> (issue #486) — one int from the
    /// device instead of a full-vocabulary logits row per draft step. Defaults to on unless
    /// <c>DOTLLM_MTP_DRAFT_ARGMAX=0</c>; tests flip it in-process. A constrained draft always takes
    /// full logits, because the mask must be applied before the argmax.
    /// </summary>
    internal bool UseDraftArgMax { get; set; } =
        Environment.GetEnvironmentVariable(DisableDraftArgMaxEnvVar) != "0";

    /// <summary>Draft steps whose token came from <see cref="IModel.ForwardMtpArgMax"/>.</summary>
    internal int DraftArgMaxSteps { get; private set; }

    /// <summary>Rounds whose rejection was rolled back by a row snapshot instead of a replay.</summary>
    internal int ReplaysAvoided { get; private set; }

    /// <summary>Replay forwards issued after a rejection (checkpoint path).</summary>
    internal int Replays { get; private set; }

    private int _traceRounds, _traceTokens;
    private long _traceDraft, _traceVerify, _traceRound;

    /// <summary><c>DOTLLM_MTP_TRACE=1</c> (#1220): every 16 rounds, the mean draft / verify / whole-round wall time and tokens per round.</summary>
    private void TraceRound(long draftTicks, long verifyTicks, long roundTicks, int tokens)
    {
        _traceRounds++; _traceTokens += tokens; _traceDraft += draftTicks; _traceVerify += verifyTicks; _traceRound += roundTicks;
        if (_traceRounds % 16 != 0) return;
        double f = 1000.0 / Stopwatch.Frequency / _traceRounds;
        Console.Error.WriteLine($"[mtp] rounds={_traceRounds} tokens/round={(double)_traceTokens / _traceRounds:F2} draft_ms={_traceDraft * f:F1} " +
                                $"verify_ms={_traceVerify * f:F1} round_ms={_traceRound * f:F1}");
    }

    /// <summary>
    /// Creates a new MTP self-speculative decoder.
    /// </summary>
    /// <param name="greedy">Kept for source compatibility. The decoder follows the <see cref="SamplerPipeline"/> it is given: a greedy pipeline
    /// accepts a draft iff it is the target's argmax (output identical to plain greedy decode); a stochastic one accepts with probability
    /// <c>p(draft)</c> and resamples a rejected position from the residual, so the output is distributed exactly as plain sampling (#900).</param>
    public MtpSpeculativeDecoder(bool greedy = true)
    {
        _ = greedy;
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
        => DraftAndVerify(targetModel, kvCacheTarget, mtpState, pipeline, generatedIds, constraint, position, vocabSize, numCandidates, outputBuffer,
                          allowZeroDraft: false);

    /// <summary>
    /// The same round with an explicit switch for the calibration round (#820): with <paramref name="allowZeroDraft"/> a <paramref name="numCandidates"/>
    /// of 0 drafts nothing, verifies the last token alone and commits one token - a plain decode step that keeps the MTP state in sync, whose wall time is
    /// the request's plain-decode reference. Without it 0 candidates return an empty result, as before.
    /// </summary>
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
        Span<int> outputBuffer,
        bool allowZeroDraft)
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
        if (k < 0 || (k == 0 && !(allowZeroDraft && numCandidates == 0)))
            return default;

        int lastToken = generatedIds[^1];

        long roundStart = EngineClock.GetTimestamp();
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
            // Issue #486: an unconstrained greedy draft needs only the argmax, which a GPU model can
            // reduce on the device. The decoder is greedy-only, so the constraint is the only gate.
            bool draftArgMax = draftConstraint is null && UseDraftArgMax && targetModel.SupportsMtpArgMax;
            try
            {
                for (int i = 0; i < k; i++)
                {
                    long fwdStart = EngineClock.GetTimestamp();
                    if (draftArgMax)
                    {
                        draftToken = targetModel.ForwardMtpArgMax(mtpState, draftToken, position + i);
                        draftTicks += EngineClock.GetTimestamp() - fwdStart;
                        DraftArgMaxSteps++;
                    }
                    else
                    {
                        using ITensor draftLogits = targetModel.ForwardMtp(mtpState, draftToken, position + i);
                        draftTicks += EngineClock.GetTimestamp() - fwdStart;

                        unsafe
                        {
                            var logitSpan = new Span<float>((void*)draftLogits.DataPointer, vocabSize);
                            if (draftConstraint != null)
                                TokenMaskApplier.Apply(logitSpan, draftConstraint.GetAllowedTokens());
                            draftToken = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)logitSpan);
                        }
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
            // Either record the state after every row (#473) or checkpoint first so a partial
            // rejection can restore and replay the accepted prefix.
            bool rowSnapshots = UseRecurrentRowSnapshots && targetModel.SupportsRecurrentRowSnapshots;
            gdnCheckpoint = !rowSnapshots && targetModel.SupportsRecurrentStateCheckpoint
                ? targetModel.CheckpointRecurrentState()
                : null;
            if (!rowSnapshots && gdnCheckpoint is null && targetModel.RequiresPerSequenceState && targetModel.SupportsRecurrentStateCheckpoint == false)
                throw new NotSupportedException(
                    $"{targetModel.GetType().Name} has a recurrent trunk but offers neither per-row snapshots nor a state checkpoint " +
                    $"(DOTLLM_MTP_GDN_SNAPSHOTS=0?): a rejected draft would leave the recurrent state advanced past the committed text.");

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

            long verifyStart = EngineClock.GetTimestamp();
            using ITensor targetLogits = rowSnapshots
                ? targetModel.ForwardWithRecurrentSnapshots(
                    verifyTokens, verifyPositions, deviceId: -1, kvCacheTarget, mtpState)
                : targetModel.Forward(
                    verifyTokens, verifyPositions, deviceId: -1, kvCacheTarget, adapter: null, mtpState);
            verifyTicks += EngineClock.GetTimestamp() - verifyStart;

            int accepted = 0;       // drafts accepted
            int acceptedCount = 0;  // tokens written to outputBuffer (accepted drafts + correction/bonus)
            // Penalty processors read the generated history: every emitted token joins it for the next row (a row is only reached when the
            // drafts before it were accepted, so the history is exactly the sequence plain decoding would have had). Removed again below.
            int historyBase = generatedIds.Count;
            bool stochastic = pipeline.IsStochastic;
            unsafe
            {
                nint basePtr = targetLogits.DataPointer;
                try
                {
                    for (int row = 0; row <= k; row++)
                    {
                        var rowLogits = new Span<float>(
                            (void*)(basePtr + (long)row * vocabSize * sizeof(float)), vocabSize);
                        if (constraint != null)
                            TokenMaskApplier.Apply(rowLogits, constraint.GetAllowedTokens());

                        int emitted;
                        bool draftHeld;
                        if (!stochastic && !pipeline.HasProcessors)
                        {
                            // Plain greedy: the target's own argmax (the path every earlier version took, bit for bit).
                            emitted = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)rowLogits);
                            draftHeld = row < k && draftTokens[row] == emitted;
                        }
                        else
                        {
                            pipeline.PrepareDistribution(rowLogits, generatedIds);
                            if (row == k)
                            {
                                emitted = pipeline.SamplePrepared(rowLogits, generatedIds.Count);   // every draft held: the bonus token
                                draftHeld = false;
                            }
                            else
                            {
                                draftHeld = pipeline.TryAcceptDraft(rowLogits, draftTokens[row], out emitted, generatedIds.Count);
                            }
                        }

                        outputBuffer[acceptedCount++] = emitted;
                        constraint?.Advance(emitted);
                        generatedIds.Add(emitted);

                        if (!draftHeld)
                            break;      // correction (or bonus): the target's own token ends the round
                        accepted++;
                    }
                }
                finally
                {
                    if (generatedIds.Count > historyBase)
                        generatedIds.RemoveRange(historyBase, generatedIds.Count - historyBase);
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

            if (accepted < k && rowSnapshots)
            {
                // The state after verify row `accepted` is exactly what replaying rows 0..accepted
                // would produce; the attention KV was already rolled back above.
                targetModel.RestoreRecurrentStateToRow(accepted);
                ReplaysAvoided++;
            }
            else if (accepted < k && gdnCheckpoint is not null)
            {
                Replays++;
                // The recurrent trunk advanced through rejected rows. Restore it and replay the
                // committed prefix — WITHOUT mtpState, the head already absorbed those rows.
                targetModel.RestoreRecurrentState(gdnCheckpoint);
                kvCacheTarget.Rollback(position);
                using ITensor _ = targetModel.Forward(
                    verifyTokens.Slice(0, accepted + 1), verifyPositions.Slice(0, accepted + 1),
                    deviceId: -1, kvCacheTarget, adapter: null, mtpState: null);
            }

            long roundTicks = EngineClock.GetTimestamp() - roundStart;
            if (MtpAdaptiveGate.TraceEnabled)
                TraceRound(draftTicks, verifyTicks, roundTicks, acceptedCount);
            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k, roundTicks);
        }
        finally
        {
            (gdnCheckpoint as IDisposable)?.Dispose();
            ArrayPool<int>.Shared.Return(draftTokens);
        }
    }
}
