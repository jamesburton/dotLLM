using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Engine.Samplers;

namespace DotLLM.Engine;

/// <summary>
/// Multi-Token Prediction (MTP) self-speculative decoding (issue #253): the target model's own
/// lightweight extra head drafts K candidate tokens from the target model's own hidden state, and
/// the target model verifies them in one extra batched forward pass — no second model, no second
/// KV-cache-of-the-full-model, just the target model's own tiny <see cref="IMtpState"/>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why a parallel interface instead of an <see cref="ISpeculativeDecoder"/> overload.</b>
/// <see cref="ISpeculativeDecoder.DraftAndVerify"/>'s signature is built around two independent
/// models with two independent full-model KV-caches
/// (<c>DraftAndVerify(targetModel, draftModel, kvCacheTarget, kvCacheDraft, ...)</c>). MTP has no
/// second <see cref="IModel"/> — the "draft" is a single extra transformer block sharing the
/// target model's own weights file (<see cref="ModelConfig.NextnPredictLayers"/>), and its state
/// is an <see cref="IMtpState"/> (a KV-cache sized for just that one block, not a full model), not
/// a second <see cref="DotLLM.Core.Attention.IKvCache"/>. Forcing MTP through the two-model
/// signature would mean either threading a fake "draft model" wrapper that doesn't behave like a
/// real <see cref="IModel"/> (its forward pass needs the TARGET's hidden state as input, which no
/// <c>IModel.Forward</c> overload exposes as an output) or overloading
/// <c>kvCacheDraft</c>'s meaning to sometimes be a full <see cref="DotLLM.Core.Attention.IKvCache"/>
/// and sometimes a tiny per-layer <see cref="IMtpState"/> — both erode the existing interface's
/// clarity for its actual (two-model) use case. A parallel interface keeps both designs honest
/// about what they need, while sharing the same <see cref="SpeculativeResult"/> return shape and
/// the same greedy-only correctness gate as <see cref="SpeculativeDecoder"/> (see remarks there and
/// on <see cref="MtpSpeculativeDecoder"/>).
/// </para>
/// </remarks>
public interface IMtpSpeculativeDecoder
{
    /// <summary>
    /// Drafts candidate tokens with the target model's own MTP head and verifies them with the
    /// target model's normal forward pass. On rejection, rolls back the KV-cache and constraint
    /// state to the last accepted position. Every call starts by re-seeding
    /// <paramref name="mtpState"/> from a fresh single-token "catchup" forward of the current
    /// <c>lastToken</c> — see the implementation's remarks for why that is necessary (neither a
    /// corrected nor a bonus token from the previous round has ever been forwarded through the
    /// trunk as an input, so its hidden state must be (re)computed before drafting can start).
    /// </summary>
    /// <param name="targetModel">The model — must have <see cref="IModel.SupportsMtp"/> true.</param>
    /// <param name="kvCacheTarget">KV-cache for the target model's normal (trunk) forward pass.</param>
    /// <param name="mtpState">The MTP head's own state (tiny KV-cache + pending hidden handoff), from <see cref="IModel.CreateMtpState"/>.</param>
    /// <param name="pipeline">Sampling pipeline for token selection.</param>
    /// <param name="generatedIds">All previously generated token IDs (for repetition penalty).</param>
    /// <param name="constraint">Optional decoding constraint (cloned before drafting, rolled back on rejection).</param>
    /// <param name="position">Current sequence position (prompt length + decoded so far).</param>
    /// <param name="vocabSize">Vocabulary size of the target model.</param>
    /// <param name="numCandidates">Number of draft tokens to propose (K).</param>
    /// <param name="outputBuffer">Caller-owned buffer for accepted token IDs (must be at least K+1 elements).</param>
    /// <returns>Result containing accepted count and timing information. Accepted tokens are in <paramref name="outputBuffer"/>.</returns>
    SpeculativeResult DraftAndVerify(
        IModel targetModel,
        DotLLM.Core.Attention.IKvCache kvCacheTarget,
        IMtpState mtpState,
        SamplerPipeline pipeline,
        List<int> generatedIds,
        IDecodingConstraint? constraint,
        int position,
        int vocabSize,
        int numCandidates,
        Span<int> outputBuffer);
}
