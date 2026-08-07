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
/// <b>The "catchup" forward — why every round starts by re-processing <c>lastToken</c>.</b>
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
/// to obtain that hidden state before the MTP draft loop can start. This re-forward is safe and
/// idempotent — <c>lastToken</c> already occupies <c>position</c> in
/// <c>kvCacheTarget</c> from the prior round (or the initial prefill); <see cref="IKvCache"/>
/// is position-indexed (<c>Rollback</c>'s doc: "Allocated memory is retained and overwritten on
/// subsequent Update calls"), so writing the same token at the same position again is a no-op on
/// the cache contents. The verify batch afterward still includes <c>lastToken</c> as its own row 0
/// (matching <see cref="SpeculativeDecoder"/>'s shape) for the comparison-basis logits it needs —
/// a second harmless re-forward of the same token. This doubles the trunk's per-round
/// single-token-equivalent cost relative to a maximally-optimized implementation that reuses the
/// catchup call's own logits as the row-0 comparison basis; documented here as a known,
/// correctness-first simplification rather than silently traded away.
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

        // ── Catchup (see type remarks): seed mtpState with h-after-lastToken before drafting. ──
        long catchupStart = Stopwatch.GetTimestamp();
        using (ITensor catchupLogits = targetModel.Forward(
                   [lastToken], [position], deviceId: -1, kvCacheTarget, adapter: null, mtpState))
        {
            mtpState.SeedFromCapturedRow(0);
            _ = catchupLogits; // logits themselves unused here — only the hidden-state side effect matters
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

            // ── Verify Phase (single batched forward pass over the target model — identical
            //    shape to SpeculativeDecoder's verify phase; row 0 redundantly re-forwards
            //    lastToken, matching the catchup step above — see type remarks). ──
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

            int actualVerifyLen = Math.Min(verifyLen, maxPos - position);
            if (actualVerifyLen < 1)
                return default;

            long verifyStart = Stopwatch.GetTimestamp();
            using ITensor targetLogits = targetModel.Forward(
                verifyTokens.Slice(0, actualVerifyLen),
                verifyPositions.Slice(0, actualVerifyLen),
                deviceId: -1, kvCacheTarget, adapter: null);
            verifyTicks += Stopwatch.GetTimestamp() - verifyStart;

            // ── Accept/Reject Phase (greedy argmax — see type remarks for the correctness argument) ──
            int acceptedCount = 0;

            unsafe
            {
                nint basePtr = targetLogits.DataPointer;

                for (int i = 0; i < Math.Min(k, actualVerifyLen); i++)
                {
                    int draftTok = draftTokens[i];
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
                        RollbackKvCache(kvCacheTarget, position, acceptedCount);
                        return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
                    }
                }

                // All K accepted — sample bonus token from the target's own argmax.
                if (actualVerifyLen > k)
                {
                    var bonusLogitSpan = new Span<float>(
                        (void*)(basePtr + (long)k * vocabSize * sizeof(float)), vocabSize);

                    if (constraint != null)
                        TokenMaskApplier.Apply(bonusLogitSpan, constraint.GetAllowedTokens());

                    int bonusToken = TensorPrimitives.IndexOfMax((ReadOnlySpan<float>)bonusLogitSpan);
                    outputBuffer[acceptedCount++] = bonusToken;
                }
            }

            RollbackKvCache(kvCacheTarget, position, acceptedCount);
            return new SpeculativeResult(acceptedCount, draftTicks, verifyTicks, k);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(draftTokens);
        }
    }

    private static void RollbackKvCache(IKvCache kvCacheTarget, int position, int acceptedCount)
    {
        int acceptedEnd = position + acceptedCount;
        if (acceptedEnd <= kvCacheTarget.CurrentLength)
            kvCacheTarget.Rollback(acceptedEnd);
    }
}
