using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Constraints;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.Constraints;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

public sealed partial class TextGenerator
{
    // A recurrent model's state lives in the model instance, so the owner is tracked per model, across generators.
    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<IModel, object> s_recurrentOwners = new();

    /// <summary>
    /// Opens a resumable single-sequence <see cref="GenerationSession"/> (issue #1919): prefill once, generate until a boundary,
    /// append observation tokens, roll back, all without re-prefilling the committed prefix.
    /// </summary>
    /// <param name="options">Sampling options (temperature, seed, penalties) and the per-segment <see cref="InferenceOptions.MaxTokens"/>.
    /// <see cref="InferenceOptions.ResponseFormat"/> (when not Text) builds the session's initial constraint exactly as <c>Generate</c> does
    /// (JSON object, schema, regex, grammar); <see cref="GenerationSession.SetConstraint"/> replaces it later.
    /// <see cref="InferenceOptions.RepeatGuardLimit"/> follows the plain generator
    /// (the limit-th identical sampled token in a row ends the segment, token kept, <see cref="FinishReason.Length"/>); the run survives pauses and
    /// rollback restores it, but an observation append or prefill ends it (intended: those tokens are supplied, not sampled).</param>
    /// <param name="provenanceTag">Identity of the weights, tokenizer, precision, modifiers and knowledge the session's cached state belongs to.</param>
    /// <param name="currentProvenance">Live view of the provenance the orchestrator is running under (adapters, LoRA, knowledge snapshot).
    /// Every session operation compares it with <paramref name="provenanceTag"/> and throws <see cref="InvalidOperationException"/> naming the
    /// differing fields, so cached KV / recurrent state is never reused under a changed identity. Null = no automatic check.</param>
    /// <param name="maxContextTokens">KV capacity in tokens. 0 (default) = start at <see cref="DefaultInitialContextTokens"/> (capped at the model maximum) and
    /// grow on demand with <c>IKvCache.TryGrowTo</c> up to the model's maximum sequence length; a positive value is a fixed capacity allocated now (use it
    /// for a cache that cannot grow). Must not exceed the model's maximum sequence length.</param>
    /// <param name="checkpointBudgetBytes">Bytes of recurrent-state checkpoints a recurrent session may hold for rollback (plus the logits copies); 0 =
    /// <see cref="GenerationSession.DefaultCheckpointBudgetBytes"/> (1 GiB). <b>On a GPU these are device memory (VRAM)</b>, not host RAM.</param>
    /// <remarks>
    /// A recurrent (GDN) model keeps its state in the model instance, so only one session may be live per such model and
    /// <c>Generate*</c> on this generator throws while it is open. Dispose the session to release the KV-cache and detach.
    /// </remarks>
    public GenerationSession CreateSession(InferenceOptions options, CacheProvenanceTag provenanceTag,
        Func<CacheProvenanceTag>? currentProvenance = null, int maxContextTokens = 0, long checkpointBudgetBytes = 0)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(provenanceTag);
        ArgumentOutOfRangeException.ThrowIfNegative(maxContextTokens);
        ArgumentOutOfRangeException.ThrowIfNegative(checkpointBudgetBytes);
        int modelMax = _model.Config.MaxSequenceLength;
        if (maxContextTokens > modelMax)
            throw new ArgumentOutOfRangeException(nameof(maxContextTokens), maxContextTokens, $"Exceeds the model's maximum sequence length of {modelMax}.");
        ThrowIfOptionsUnsupported(options);
        // Hold the generation gate across check + owner registration + state reset so a concurrent Generate cannot slip in between (#1931 review M-1).
        if (!_generationGate.Wait(0))
            throw new InvalidOperationException("A Generate call is running on this generator; open the session when it has finished.");
        try
        {
            return CreateSessionLocked(options, provenanceTag, currentProvenance, maxContextTokens, modelMax, checkpointBudgetBytes);
        }
        finally { _generationGate.Release(); }
    }

    /// <summary>KV capacity a default session starts with (and grows from).</summary>
    public const int DefaultInitialContextTokens = 4096;

    /// <summary>
    /// Test hook (issue #1919): when positive, a default session (<c>maxContextTokens</c> 0) starts at this many tokens instead of
    /// <see cref="DefaultInitialContextTokens"/> and still grows to the model maximum. Lets a test with a small context window reach the
    /// growth path that a real model reaches at 4096.
    /// </summary>
    internal int InitialContextTokensForTest { get; set; }

    /// <summary>
    /// Fields of <see cref="InferenceOptions"/> a session cannot honour are refused rather than silently ignored (see docs/GENERATION_SESSION.md for the table).
    /// </summary>
    private static void ThrowIfOptionsUnsupported(InferenceOptions options)
    {
        if (options.StopConditions is not null)
            throw new NotSupportedException("options.StopConditions is not supported on a session; pass the stop condition to GenerateUntilBoundaryAsync per segment.");
        if (options.Logprobs || options.TopLogprobs > 0)
            throw new NotSupportedException("options.Logprobs / TopLogprobs are not supported on a session; read the raw logits with CopyNextLogits.");
    }

    private GenerationSession CreateSessionLocked(InferenceOptions options, CacheProvenanceTag provenanceTag,
        Func<CacheProvenanceTag>? currentProvenance, int maxContextTokens, int modelMax, long checkpointBudgetBytes)
    {
        bool recurrent = _model.RequiresPerSequenceState;
        if (recurrent && !s_recurrentOwners.TryAdd(_model, this))
            throw new InvalidOperationException(
                "A GenerationSession is already open on this recurrent model; its state is model-owned, so only one session may be live.");

        IKvCache? kv = null;
        try
        {
            int size = maxContextTokens > 0 ? maxContextTokens : Math.Min(InitialContextTokensForTest > 0 ? InitialContextTokensForTest : DefaultInitialContextTokens, modelMax);
            int ceiling = maxContextTokens > 0 ? maxContextTokens : modelMax;
            kv = AllocateKvCache(size);
            if (recurrent) _model.ResetSequenceState();   // a previous request left model-owned state behind
            IDecodingConstraint? initial = CreateResponseConstraint(options.ResponseFormat);
            return new GenerationSession(this, options, provenanceTag, currentProvenance, kv, ceiling,
                checkpointBudgetBytes > 0 ? checkpointBudgetBytes : GenerationSession.DefaultCheckpointBudgetBytes, initial);
        }
        catch
        {
            kv?.Dispose();
            if (recurrent) s_recurrentOwners.Remove(_model);
            throw;
        }
    }

    internal IModel SessionModel => _model;
    internal ITokenizer SessionTokenizer => _tokenizer;

    internal ITensor SessionForwardPrefill(int[] tokens, int start, int length, IKvCache kv, CancellationToken cancellationToken)
    {
        _requestImages = null;   // sessions carry no images (#2035)
        return ForwardPrefill(tokens, start, length, kv, adapter: null, cancellationToken: cancellationToken);
    }

    internal void ReleaseSession()
    {
        if (_model.RequiresPerSequenceState) s_recurrentOwners.Remove(_model);
    }

    private void ThrowIfRecurrentSessionOpen()
    {
        if (_model.RequiresPerSequenceState && s_recurrentOwners.TryGetValue(_model, out _))
            throw new InvalidOperationException(
                "A GenerationSession is open on this recurrent model and owns its state; dispose it before calling Generate on this generator.");
    }
}
