using DotLLM.Core.Sampling;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Tokenizers;

namespace DotLLM.Engine.Samplers;

/// <summary>
/// <c>ignore_eos</c> / <c>min_tokens</c> (#1345), llama.cpp semantics: the end-of-generation tokens get a logit of -infinity so
/// they are never sampled, either for the whole request (<c>ignore_eos</c>) or until <c>min_tokens</c> tokens exist. Masking
/// (rather than ignoring the stop) keeps the EOG token out of the output and the context, and it is a plain logit processor, so
/// the speculative/MTP verify rows (which pass the growing generated list per row) pick the same token as plain decoding.
/// </summary>
public sealed class EosSuppressionProcessor : ILogitProcessor
{
    private readonly int[] _ids;
    private readonly bool _always;
    private readonly int _minTokens;

    /// <summary>Creates the processor.</summary>
    /// <param name="tokenizer">Supplies the end-of-generation token set.</param>
    /// <param name="ignoreEos">Mask for the whole request.</param>
    /// <param name="minTokens">Mask while fewer than this many tokens have been generated.</param>
    public EosSuppressionProcessor(ITokenizer tokenizer, bool ignoreEos, int minTokens)
        : this(EndOfGenerationTokens.Resolve(tokenizer), ignoreEos, minTokens) { }

    internal EosSuppressionProcessor(int[] endOfGenerationIds, bool ignoreEos, int minTokens)
    {
        _ids = endOfGenerationIds;
        _always = ignoreEos;
        _minTokens = minTokens;
    }

    /// <inheritdoc/>
    public void Process(Span<float> logits, IReadOnlyList<int> previousTokens, ProcessorContext context)
    {
        if (!_always && previousTokens.Count >= _minTokens)
            return;
        foreach (int id in _ids)
            if ((uint)id < (uint)logits.Length)
                logits[id] = float.NegativeInfinity;
    }
}
