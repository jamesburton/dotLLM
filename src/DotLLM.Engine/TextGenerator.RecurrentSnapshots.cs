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
        int[] promptIds, int promptLen, int maxTokens, string? generationPromptSuffix, out int snapshotAt)
    {
        int[]? previous = _recurrentPrefixLastPrompt;
        int needed = Math.Min(_model.Config.MaxSequenceLength, checked(promptLen + maxTokens));
        var kv = AllocateKvCache(needed);
        int cached;
        string? tier;
        double milliseconds;
        try
        {
            // This path never grows a KV: it restores into a fresh one. When the device refuses that KV's storage
            // (DeviceMemoryAdmissionException) the restore rethrows rather than missing, because a cold prefill needs the same
            // storage; the request then fails with the context error (issue #1093), not a full re-prefill into the same refusal.
            cached = _tieredSnapshots!.TryRestore(promptIds, kv, out tier, out milliseconds);
        }
        catch
        {
            kv.Dispose();
            throw;
        }
        _recurrentPrefixLastPrompt = promptIds;
        LastPrefixCacheTier = tier; LastPrefixRestoreMilliseconds = milliseconds;
        if (cached == 0)
        {
            _model.ResetSequenceState();
            if (kv.CurrentLength > 0) kv.Rollback(0);
        }
        // The snapshot bets that the next prompt extends this one, so it goes one token short of the end. With a
        // generation-prompt hint (issue #1093) it goes just before the generation prompt instead, which the next turn shares
        // even when it re-renders the rest; an exact extension of the previous prompt (an agent loop) keeps the later end.
        int? suffixBoundary = RecurrentSnapshotBoundary.FromGenerationSuffix(promptIds, _tokenizer, generationPromptSuffix);
        bool extendsPrevious = previous is not null && promptLen >= previous.Length
            && promptIds.AsSpan(0, previous.Length).SequenceEqual(previous);
        int boundary = RecurrentSnapshotBoundary.Choose(
            suffixBoundary is null || extendsPrevious ? promptLen - 1 : -1, suffixBoundary, promptLen);
        snapshotAt = boundary >= RecurrentPrefixMinTokens && (cached == 0 || boundary >= cached + RecurrentPrefixMinTokens) ? boundary : -1;
        return (kv, cached, true);
    }
}


