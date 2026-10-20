using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Engine.PromptCache;

namespace DotLLM.Engine;

public sealed partial class TextGenerator
{
    private readonly SemaphoreSlim _generationGate = new(1, 1);
    private RecurrentPrefixSnapshotCache? _tieredSnapshots;
    private long _legacyPrefixHits, _legacyPrefixMisses;

    /// <summary>Current recurrent prefix cache occupancy and counters.</summary>
    public RecurrentPrefixCacheStatistics RecurrentPrefixStatistics => _tieredSnapshots?.Statistics
        ?? new() { Enabled = _recurrentPrefixEnabled, Entries = _recurrentPrefix is null ? 0 : 1, MaxEntries = 1, Hits = _legacyPrefixHits, VramHits = _legacyPrefixHits, Misses = _legacyPrefixMisses };
    /// <summary>Tier used for the most recent request's prefix hit.</summary>
    public string? LastPrefixCacheTier { get; private set; }
    /// <summary>Milliseconds spent restoring the most recent request's prefix.</summary>
    public double LastPrefixRestoreMilliseconds { get; private set; }

    /// <summary>Generates a response, serializing model-owned state against other calls on this generator.</summary>
    public InferenceResponse Generate(string prompt, InferenceOptions? options = null,
        Action<int>? onTokenGenerated = null, ILoraAdapter? adapter = null,
        Action<ReadOnlySpan<char>>? onTextGenerated = null, CancellationToken cancellationToken = default)
    {
        _generationGate.Wait(cancellationToken);
        LastPrefixCacheTier = null; LastPrefixRestoreMilliseconds = 0;
        try { return GenerateCore(prompt, options, onTokenGenerated, adapter, onTextGenerated, cancellationToken); }
        finally { _generationGate.Release(); }
    }

    /// <summary>Streams a response while exclusively owning the model's live recurrent and KV state.</summary>
    public async IAsyncEnumerable<GenerationToken> GenerateStreamingTokensAsync(string prompt, InferenceOptions? options = null,
        [EnumeratorCancellation] CancellationToken cancellationToken = default, ILoraAdapter? adapter = null)
    {
        await _generationGate.WaitAsync(cancellationToken).ConfigureAwait(false);
        LastPrefixCacheTier = null; LastPrefixRestoreMilliseconds = 0;
        try
        {
            await foreach (var token in GenerateStreamingTokensCoreAsync(prompt, options, cancellationToken, adapter).ConfigureAwait(false))
                yield return token;
        }
        finally { _generationGate.Release(); }
    }

    private (Core.Attention.IKvCache KvCache, int CachedTokenCount, bool OwnsKvCache) ResolveTieredRecurrentPrefix(
        int[] promptIds, int promptLen, int maxTokens, out int snapshotAt)
    {
        int needed = Math.Min(_model.Config.MaxSequenceLength, checked(promptLen + maxTokens));
        var kv = AllocateKvCache(needed);
        int cached = _tieredSnapshots!.TryRestore(promptIds, kv, out string? tier, out double milliseconds);
        LastPrefixCacheTier = tier; LastPrefixRestoreMilliseconds = milliseconds;
        if (cached == 0)
        {
            _model.ResetSequenceState();
            if (kv.CurrentLength > 0) kv.Rollback(0);
        }
        int boundary = promptLen - 1;
        snapshotAt = boundary >= RecurrentPrefixMinTokens && (cached == 0 || boundary >= cached + RecurrentPrefixMinTokens) ? boundary : -1;
        return (kv, cached, true);
    }
}


