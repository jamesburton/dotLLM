using DotLLM.Core.Sampling;

namespace DotLLM.Engine.Samplers.StopConditions;

/// <summary>
/// Stops generation when the end-of-sequence token is produced.
/// The EOS token is excluded from the output.
/// </summary>
public sealed class EosStopCondition : IStopCondition
{
    private readonly int _eosTokenId;
    private readonly int[]? _extraIds;

    /// <summary>
    /// Creates a new EOS stop condition.
    /// </summary>
    /// <param name="eosTokenId">The end-of-sequence token ID.</param>
    public EosStopCondition(int eosTokenId)
    {
        _eosTokenId = eosTokenId;
    }

    /// <summary>
    /// Creates a stop condition that fires on any of <paramref name="endOfGenerationIds"/>
    /// (see <see cref="EndOfGenerationTokens"/>).
    /// </summary>
    /// <param name="endOfGenerationIds">Non-empty set of end-of-generation token ids; the first is the declared EOS.</param>
    public EosStopCondition(int[] endOfGenerationIds)
    {
        _eosTokenId = endOfGenerationIds[0];
        _extraIds = endOfGenerationIds.Length > 1 ? endOfGenerationIds : null;
    }

    /// <inheritdoc/>
    public StopResult ShouldStop(int tokenId, IReadOnlyList<int> generatedTokens, string decodedText)
    {
        if (tokenId == _eosTokenId)
            return StopResult.Stop;
        return _extraIds is not null && Array.IndexOf(_extraIds, tokenId) >= 0 ? StopResult.Stop : StopResult.Continue;
    }
}
