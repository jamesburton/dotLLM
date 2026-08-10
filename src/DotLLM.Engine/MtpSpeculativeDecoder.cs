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
/// <b>The "catchup" forward — why every round starts by re-processing <c>lastToken</c>, and why
/// the verify batch does NOT re-submit it (issue #253 CUDA follow-up, fixed 2026-08-07).</b>
/// The MTP head's first draft step needs <c>mtpState</c> seeded with the trunk's own
/// hidden state <em>after</em> processing <c>lastToken</c> (the pairing invariant, confirmed
/// against llama.cpp's <c>graph_mtp</c>: <c>h_input</c> and <c>tok_embd</c> in the same MTP call
/// must come from the <em>same</em> position — <c>h</c> after token <c>T</c> pairs with
/// <c>embed(T)</c> to predict <c>T+1</c>). Whichever token becomes <c>lastToken</c> for a new
/// round — a corrected token (its argmax differed from what was fed to the previous round's
/// verify batch) or a bonus token (sampled from logits, never fed as an input at all) — has, by
/// construction, <em>never been forwarded through the trunk as an input</em>: there is no row in
/// the previous round's verify batch whose hidden state reflects it. So every round begins with a
/// single-token trunk forward of <c>lastToken</c> (with <c>mtpState</c> capture) purely
/// to obtain that hidden state before the MTP draft loop can start.
/// </para>
/// <para>
/// The catchup call's own logits are ALSO this round's "position 0" comparison basis for
/// <c>draftTokens[0]</c> — reused directly (<c>catchupArgmax</c>), rather than re-submitting
/// <c>lastToken</c> as row 0 of the verify batch the way <see cref="SpeculativeDecoder"/>'s
/// two-model verify phase does. An earlier version of this method DID re-submit it, reasoning
/// (correctly, but incompletely) that <see cref="IKvCache"/> is position-indexed so re-writing the
/// same token at the same position is a no-op on cache contents. <b>That reasoning does not extend
/// to recurrent trunk layers</b> (Gated DeltaNet / Mamba, exactly the token-mixing kind
/// <see cref="ModelConfig.HybridLayout"/> hybrid architectures use — the real MTP target,
/// Qwen3.6-27B/Bonsai-27B, IS one): their state is a pure sequential recurrence, not
/// position-indexed, so forwarding the same token through it a second time double-advances that
/// state and corrupts every subsequent decode step. This was caught empirically by a CUDA
/// integration test driving the real (GDN-containing) <c>Qwen3HybridDense</c> model through this
/// decoder and comparing against plain greedy decode of the same model — a comparison the original
/// mock-model unit tests could not catch because their mock used a non-recurrent architecture.
/// Skipping the redundant row also removes the "doubles the trunk's per-round single-token
/// cost" overhead the original version explicitly traded away as a documented simplification —
/// fixing the bug turned out to be strictly cheaper, not a tradeoff.
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

        // Clamp K to remaining target KV-cache capacity and the MTP head's own KV-cache depth.
        int maxPos = kvCacheTarget.MaxLength;
        int k = Math.Min(numCandidates, maxPos - position - 1);
        if (k <= 0)
            return default;

        int lastToken = generatedIds[^1];

        long draftTicks = 0;
        long verifyTicks = 0;

        // Fresh MTP head KV-cache for this round — see the type remarks on why this is safe.
        mtpState.Rollback(0);

        // ── Catchup (see type remarks): seed mtpState with h-after-lastToken before drafting.
        //    ALSO capture this call's own logits' argmax — this IS position 0's verify comparison
        //    basis (see the "no redundant re-forward" remarks below), so the verify batch never
        //    resubmits lastToken. ──
        long catchupStart = Stopwatch.GetTimestamp();
        int catchupArgmax;
        using (ITensor catchupLogits = targetModel.Forward(
                   [lastToken], [position], deviceId: -1, kvCacheTarget, adapter: null, mtpState))
        {
            mtpState.SeedFromCapturedRow(0);
            unsafe
            {
                var span = new Span<float>((void*)catchupLogits.DataPointer, vocabSize);
                if (constraint != null)
                    TokenMaskApplier.Apply(span, constraint.GetAllowedTokens());
                catchupArgmax = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)span);
            }
        }
        verifyTicks += Stopwatch.GetTimestamp() - catchupStart;

        IDecodingConstraint? draftConstraint = constraint?.Clone();
        int[] draftTokens = ArrayPool<int>.Shared.Rent(k);

        try
        {
            // ── Draft Phase: MTP head autoregressively drafts K tokens against its own tiny
            //    KV-cache, seeded from the target model's own hidden state — no second model. ──
            int originalGenCount = generatedIds.Count;
            int draftToken = lastToken;
            try
            {
                for (int i = 0; i < k; i++)
                {
                    int pos = position + i;

                    long fwdStart = Stopwatch.GetTimestamp();
                    using ITensor draftLogits = targetModel.ForwardMtp(mtpState, draftToken, pos);
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

            // ── Accept/reject position 0 against the catchup call's own argmax — NOT a fresh
            //    verify-batch row. Re-submitting lastToken as a verify-batch row (the original
            //    design) would be byte-redundant for attention/KV-cache math (causally
            //    independent of later batch rows) but subtly WRONG for recurrent (GDN/Mamba)
            //    trunk layers: their state is a pure sequential recurrence, not position-indexed,
            //    so forwarding the same token through it twice per round double-advances that
            //    state and corrupts every subsequent decode step. catchupArgmax IS the same
            //    computation a fresh "row 0" would have produced (identical input token, position,
            //    and preceding KV-cache/recurrent state) — reusing it is both correct and, as a
            //    side effect, removes the redundant single-token forward the original design paid
            //    every round. ──
            int acceptedCount = 0;
            if (draftTokens[0] == catchupArgmax)
            {
                outputBuffer[acceptedCount++] = draftTokens[0];
                constraint?.Advance(draftTokens[0]);
            }
            else
            {
                outputBuffer[acceptedCount++] = catchupArgmax;
                constraint?.Advance(catchupArgmax);
                // Rejected before the verify batch ever ran (only the always-legitimate catchup
                // forward has touched the trunk so far this round) — nothing to restore.
                RollbackState(targetModel, kvCacheTarget, position, acceptedCount,
                    outputBuffer, gdnCheckpoint: null, rejected: false);
                return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
            }

            // ── Verify Phase (single batched forward pass over ALL of draftTokens[0..k-1] at
            //    position+1..position+k — this is the pre-fix verify batch with ONLY the leading
            //    lastToken row dropped; draftTokens[0] itself still needs to appear as an INPUT
            //    token here even though its own value was already resolved via catchupArgmax
            //    above, because it is what row m needs to predict draftTokens[m+1]. Row m (0-based)
            //    predicts the token after position+m+1, i.e. it is the comparison basis for
            //    draftTokens[m+1] (m=0..k-2); the LAST row (m=k-1, input=draftTokens[k-1]) doubles
            //    as the bonus-token source when every draft token is accepted, mirroring
            //    SpeculativeDecoder's verify shape. ──
            int verifyLen = k;
            Span<int> verifyTokens = verifyLen <= 16 ? stackalloc int[verifyLen] : new int[verifyLen];
            Span<int> verifyPositions = verifyLen <= 16 ? stackalloc int[verifyLen] : new int[verifyLen];
            for (int i = 0; i < verifyLen; i++)
            {
                verifyTokens[i] = draftTokens[i];
                verifyPositions[i] = position + i + 1;
            }

            // k is already clamped to maxPos - position - 1 above, so position + k <= maxPos - 1
            // and every verify position here (<= position + k) is guaranteed in-range — no
            // additional clamp needed (unlike the pre-fix code, which clamped defensively against
            // an off-by-one that can no longer occur with this narrower verify range).
            //
            // Issue #287: this batched forward advances the target model's recurrent (GDN) trunk
            // state — if it has one — for every one of draftTokens[0..k-1], before we know which
            // will end up accepted (rows past the eventual rejection point get rolled back on the
            // KV-cache side below, but a pure sequential recurrence has no position addressing to
            // undo that the same way). Checkpoint immediately before this call so a partial
            // rejection can restore + replay exactly the genuinely-accepted prefix. (The position-0
            // catchup-vs-draftTokens[0] comparison above never reaches this point on rejection, so
            // it needs no checkpoint of its own — nothing has touched the trunk beyond the always-
            // legitimate catchup forward yet.)
            object? gdnCheckpoint = targetModel.SupportsRecurrentStateCheckpoint
                ? targetModel.CheckpointRecurrentState()
                : null;

            long verifyStart = Stopwatch.GetTimestamp();
            using ITensor targetLogits = targetModel.Forward(
                verifyTokens, verifyPositions, deviceId: -1, kvCacheTarget, adapter: null);
            verifyTicks += Stopwatch.GetTimestamp() - verifyStart;

            unsafe
            {
                nint basePtr = targetLogits.DataPointer;

                // Rows 0..k-2 verify draftTokens[1..k-1]; row k-1 is reserved for the bonus token
                // below and is never itself an accept/reject comparison target.
                for (int i = 0; i < verifyLen - 1; i++)
                {
                    int draftTok = draftTokens[i + 1];
                    var targetLogitSpan = new Span<float>(
                        (void*)(basePtr + (long)i * vocabSize * sizeof(float)), vocabSize);

                    if (constraint != null)
                        TokenMaskApplier.Apply(targetLogitSpan, constraint.GetAllowedTokens());

                    int targetArgmax = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)targetLogitSpan);
                    if (draftTok == targetArgmax)
                    {
                        outputBuffer[acceptedCount++] = draftTok;
                        constraint?.Advance(draftTok);
                    }
                    else
                    {
                        outputBuffer[acceptedCount++] = targetArgmax;
                        constraint?.Advance(targetArgmax);
                        RollbackState(targetModel, kvCacheTarget, position, acceptedCount,
                            outputBuffer, gdnCheckpoint, rejected: true);
                        return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
                    }
                }

                // All K accepted — sample bonus token from the LAST verify row's own argmax
                // (predicts position+k+1, exactly matching the pre-fix design's bonus semantics).
                var bonusLogitSpan = new Span<float>(
                    (void*)(basePtr + (long)(verifyLen - 1) * vocabSize * sizeof(float)), vocabSize);

                if (constraint != null)
                    TokenMaskApplier.Apply(bonusLogitSpan, constraint.GetAllowedTokens());

                int bonusToken = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)bonusLogitSpan);
                outputBuffer[acceptedCount++] = bonusToken;
            }

            // All K drafted tokens were accepted (the bonus token was sampled-only, never forwarded
            // through the trunk) — the target's recurrent state already reflects exactly the
            // accepted history, so no GDN restore is needed here.
            RollbackState(targetModel, kvCacheTarget, position, acceptedCount,
                outputBuffer, gdnCheckpoint, rejected: false);
            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(draftTokens);
        }
    }

    /// <summary>
    /// Rolls the (position-indexed) target KV-cache back to the accepted boundary, and — issue
    /// #287 — restores the target model's recurrent (GDN) trunk state from
    /// <paramref name="gdnCheckpoint"/> and replays exactly the genuinely-accepted draft-token
    /// prefix when <paramref name="rejected"/> is <see langword="true"/>. No-op for models that
    /// don't report <see cref="IModel.SupportsRecurrentStateCheckpoint"/> (<paramref name="gdnCheckpoint"/>
    /// is null for them) and for the all-accepted round (nothing to undo).
    /// </summary>
    /// <param name="targetModel">The target model.</param>
    /// <param name="kvCacheTarget">Target model's KV-cache.</param>
    /// <param name="position">Sequence position this round started drafting from.</param>
    /// <param name="acceptedCount">
    /// Tokens written to <paramref name="outputBuffer"/> this round. When <paramref name="rejected"/>
    /// is true, the LAST of these is always a corrected substitute that was never itself fed
    /// through the trunk as an input — so exactly <c>acceptedCount - 1</c> of the leading entries
    /// are the draft tokens genuinely forwarded AND accepted, which is what gets replayed.
    /// </param>
    /// <param name="outputBuffer">This round's accepted/corrected output tokens, in order.</param>
    /// <param name="gdnCheckpoint">
    /// Recurrent-state snapshot captured before the verify forward, or null when the target model
    /// doesn't support checkpointing.
    /// </param>
    /// <param name="rejected">True when this round ended in a rejection (vs. all K accepted).</param>
    private static void RollbackState(
        IModel targetModel, IKvCache kvCacheTarget, int position, int acceptedCount,
        ReadOnlySpan<int> outputBuffer, object? gdnCheckpoint, bool rejected)
    {
        int acceptedEnd = position + acceptedCount;
        if (acceptedEnd <= kvCacheTarget.CurrentLength)
            kvCacheTarget.Rollback(acceptedEnd);

        if (!rejected || gdnCheckpoint is null)
            return;

        targetModel.RestoreRecurrentState(gdnCheckpoint);

        int replayCount = acceptedCount - 1;
        if (replayCount <= 0)
            return;

        Span<int> replayPositions = replayCount <= 16 ? stackalloc int[replayCount] : new int[replayCount];
        for (int i = 0; i < replayCount; i++)
            replayPositions[i] = position + i + 1;

        using ITensor _ = targetModel.Forward(outputBuffer.Slice(0, replayCount), replayPositions, deviceId: -1, kvCacheTarget);
    }
}
