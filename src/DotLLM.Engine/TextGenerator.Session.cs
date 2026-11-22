using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

public sealed partial class TextGenerator
{
    private int _liveSessions;

    /// <summary>
    /// Opens a resumable single-sequence <see cref="GenerationSession"/> (issue #1919): prefill once, generate until a boundary,
    /// append observation tokens, roll back, all without re-prefilling the committed prefix.
    /// </summary>
    /// <param name="options">Sampling options (temperature, seed, penalties) and the per-segment <see cref="InferenceOptions.MaxTokens"/>.
    /// <see cref="InferenceOptions.ResponseFormat"/> is not supported here.</param>
    /// <param name="provenanceTag">Identity of the weights, tokenizer, precision, modifiers and knowledge the session's cached state belongs to.</param>
    /// <param name="maxContextTokens">KV capacity in tokens; 0 (default) = the model's maximum sequence length.</param>
    /// <remarks>
    /// A recurrent (GDN) model keeps its state in the model instance, so only one session may be live per such model and
    /// <c>Generate*</c> on this generator throws while it is open. Dispose the session to release the KV-cache and detach.
    /// </remarks>
    public GenerationSession CreateSession(InferenceOptions options, CacheProvenanceTag provenanceTag, int maxContextTokens = 0)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(provenanceTag);
        ArgumentOutOfRangeException.ThrowIfNegative(maxContextTokens);
        if (options.ResponseFormat is not null and not ResponseFormat.Text)
            throw new NotSupportedException("GenerationSession does not support options.ResponseFormat; constrain it with SetConstraint (issue #1921).");
        if (_generationGate.CurrentCount == 0)
            throw new InvalidOperationException("A Generate call is running on this generator; open the session when it has finished.");

        bool recurrent = _model.RequiresPerSequenceState;
        if (Interlocked.Increment(ref _liveSessions) > 1 && recurrent)
        {
            Interlocked.Decrement(ref _liveSessions);
            throw new InvalidOperationException(
                "A GenerationSession is already open on this recurrent model; its state is model-owned, so only one session may be live.");
        }

        IKvCache? kv = null;
        try
        {
            int size = maxContextTokens > 0
                ? Math.Min(maxContextTokens, _model.Config.MaxSequenceLength)
                : _model.Config.MaxSequenceLength;
            kv = AllocateKvCache(size);
            if (recurrent) _model.ResetSequenceState();   // a previous request left model-owned state behind
            return new GenerationSession(this, options, provenanceTag, kv);
        }
        catch
        {
            kv?.Dispose();
            Interlocked.Decrement(ref _liveSessions);
            throw;
        }
    }

    internal IModel SessionModel => _model;
    internal ITokenizer SessionTokenizer => _tokenizer;

    internal ITensor SessionForwardPrefill(int[] tokens, int start, int length, IKvCache kv, CancellationToken cancellationToken)
        => ForwardPrefill(tokens, start, length, kv, adapter: null, cancellationToken: cancellationToken);

    internal void ReleaseSession() => Interlocked.Decrement(ref _liveSessions);

    private void ThrowIfRecurrentSessionOpen()
    {
        if (_model.RequiresPerSequenceState && Volatile.Read(ref _liveSessions) > 0)
            throw new InvalidOperationException(
                "A GenerationSession is open on this recurrent model and owns its state; dispose it before calling Generate on this generator.");
    }
}
