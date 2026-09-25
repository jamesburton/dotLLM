using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Server;

/// <summary>
/// Point-in-time snapshot of a resident-but-not-currently-active model's live objects plus
/// keep-alive bookkeeping (#369). Reactivating a snapshot is a cheap field-copy back onto the
/// active <see cref="ServerState"/> — no disk I/O, no GGUF re-parse, no weight reload. Only the
/// continuous-batch scheduler (if any) is rebuilt on reactivation; it's in-memory-only and cheap
/// to reconstruct, and only one model's scheduler runs at a time in this v1 (see
/// <c>docs/perf/ISSUE_369_MODEL_KEEPALIVE.md</c> for the concurrency scoping rationale).
/// </summary>
public sealed class ResidentModelSnapshot : IDisposable
{
    /// <summary>Canonical routing key — the model's <see cref="ServerOptions.ModelId"/>.</summary>
    public required string Key { get; init; }

    /// <summary>Options this model was loaded with (device, quant, cache types, etc.).</summary>
    public required ServerOptions Options { get; init; }

    public ModelConfig? Config { get; init; }
    public IToolCallParser? ToolCallParser { get; init; }
    public KvCacheConfig KvCacheConfig { get; init; }
    public Func<ModelConfig, int, IKvCache>? KvCacheFactory { get; init; }
    public PagedKvCacheFactory? PagedFactory { get; init; }

    /// <summary>GPU-resident paged KV block-pool factory (issue #252), non-null only for a CUDA
    /// model loaded with <c>--paged</c>. Independent of <see cref="PagedFactory"/> (the CPU pool) —
    /// a given resident model uses at most one of the two, never both.</summary>
    public DotLLM.Cuda.CudaPagedKvCacheFactory? CudaPagedFactory { get; init; }
    public PrefixCache? PrefixCache { get; init; }
    public PrefixTrieManager? PrefixTrieManager { get; init; }
    public IModel? Model { get; init; }
    public ITokenizer? Tokenizer { get; init; }
    public IChatTemplate? ChatTemplate { get; init; }
    public TextGenerator? Generator { get; init; }
    public DiffusionTextGenerator? DiffusionGenerator { get; init; }
    public string LoadedModelPath { get; init; } = "";
    public GgufFile? CurrentGguf { get; init; }
    public IModel? DraftModel { get; init; }
    public string DraftModelPath { get; init; } = "";
    public GgufFile? DraftGguf { get; init; }

    /// <summary>Approximate resident footprint (GGUF file size) used for eviction budget accounting.</summary>
    public long EstimatedBytes { get; init; }

    /// <summary>Per-model keep-alive override in seconds. Null = server default. 0 = unload after
    /// each use. Negative = never auto-unload.</summary>
    public double? KeepAliveSecondsOverride { get; set; }

    /// <summary>Wall-clock time this model was last activated/used. Drives idle-unload and LRU eviction.</summary>
    public DateTimeOffset LastUsedUtc { get; set; }

    /// <summary>Disposes every owned live object.</summary>
    public void Dispose()
    {
        PrefixCache?.Dispose();
        PrefixTrieManager?.Dispose();
        PagedFactory?.Dispose();
        CudaPagedFactory?.Dispose();
        DraftModel?.Dispose();
        DraftGguf?.Dispose();
        Model?.Dispose();
        CurrentGguf?.Dispose();
    }
}

/// <summary>
/// Observability snapshot of a resident model's lifecycle state, used by
/// <c>GET /v1/models</c> (#369).
/// </summary>
public sealed record ResidentModelInfo
{
    public required string Key { get; init; }
    public required bool IsActive { get; init; }
    public required DateTimeOffset LastUsedUtc { get; init; }
    public required double EffectiveKeepAliveSeconds { get; init; }
    public required long EstimatedBytes { get; init; }

    /// <summary>Seconds until auto-unload, or null when the keep-alive is negative (never expires).</summary>
    public double? ExpiresInSeconds { get; init; }
}

/// <summary>
/// Tracks N concurrently-resident models (#369): stashed (inactive but loaded) snapshots plus
/// bookkeeping for the currently-active model (which lives directly on the owning
/// <see cref="ServerState"/>). Enforces a configurable count/byte budget with simple LRU eviction.
/// </summary>
/// <remarks>
/// Default-compatible by construction: <see cref="MaxResidentModels"/> defaults to 1, so
/// <see cref="EnforceBudget"/> always evicts anything else the instant a second model activates —
/// reproducing the original single-model hot-swap-disposes-the-old-one behavior for callers who
/// never opt into a larger budget.
/// </remarks>
public sealed class ModelResidencyManager : IDisposable
{
    private readonly object _lock = new();
    private readonly Dictionary<string, ResidentModelSnapshot> _resident = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>Maximum number of models resident at once, counting the active one. Default 1
    /// (single-model hot-swap; matches pre-#369 behavior).</summary>
    public int MaxResidentModels { get; set; } = 1;

    /// <summary>Total byte budget across all resident models (active + stashed). 0 = unlimited
    /// (only <see cref="MaxResidentModels"/> bounds residency).</summary>
    public long MemoryBudgetBytes { get; set; }

    /// <summary>Server-wide default keep-alive in seconds (ollama parity default: 300 = 5 min).
    /// 0 = unload after each use. Negative = never auto-unload.</summary>
    public double DefaultKeepAliveSeconds { get; set; } = 300;

    /// <summary>Number of stashed (inactive) resident models.</summary>
    public int StashedCount { get { lock (_lock) return _resident.Count; } }

    /// <summary>Stashes a snapshot for later reactivation. Does not evict — call
    /// <see cref="EnforceBudget"/> afterward to apply the budget.</summary>
    public void Stash(ResidentModelSnapshot snapshot)
    {
        lock (_lock) { _resident[snapshot.Key] = snapshot; }
    }

    /// <summary>Removes and returns a stashed snapshot by key, or null if not resident.</summary>
    public ResidentModelSnapshot? TryTake(string key)
    {
        lock (_lock) { return _resident.Remove(key, out var s) ? s : null; }
    }

    /// <summary>Whether a model with the given key is currently stashed (not counting the active one).</summary>
    public bool Contains(string key)
    {
        lock (_lock) { return _resident.ContainsKey(key); }
    }

    /// <summary>
    /// Evicts least-recently-used stashed models (disposing them) until the total resident set —
    /// stashed entries plus one implicit slot for the active model sized <paramref name="activeBytes"/>
    /// — fits within <see cref="MaxResidentModels"/> and <see cref="MemoryBudgetBytes"/>. Returns the
    /// evicted snapshots (already disposed) for logging.
    /// </summary>
    public List<ResidentModelSnapshot> EnforceBudget(long activeBytes)
    {
        var evicted = new List<ResidentModelSnapshot>();
        lock (_lock)
        {
            while (_resident.Count > 0)
            {
                long totalBytes = activeBytes + _resident.Values.Sum(s => s.EstimatedBytes);
                int totalCount = _resident.Count + 1; // +1 = the active model's slot
                bool overCount = totalCount > Math.Max(1, MaxResidentModels);
                bool overBudget = MemoryBudgetBytes > 0 && totalBytes > MemoryBudgetBytes;
                if (!overCount && !overBudget) break;

                var oldest = _resident.Values.OrderBy(s => s.LastUsedUtc).First();
                _resident.Remove(oldest.Key);
                evicted.Add(oldest);
            }
        }
        foreach (var e in evicted) e.Dispose();
        return evicted;
    }

    /// <summary>
    /// Removes and disposes stashed models whose keep-alive has elapsed. Does not touch the active
    /// model — callers check that separately against <see cref="ServerState.LastUsedUtc"/> since it
    /// requires the request gate.
    /// </summary>
    public List<ResidentModelSnapshot> SweepExpired(DateTimeOffset now)
    {
        var expired = new List<ResidentModelSnapshot>();
        lock (_lock)
        {
            foreach (var (key, snap) in _resident.ToArray())
            {
                double keepAlive = snap.KeepAliveSecondsOverride ?? DefaultKeepAliveSeconds;
                if (keepAlive < 0) continue; // never expire
                if ((now - snap.LastUsedUtc).TotalSeconds >= keepAlive)
                {
                    _resident.Remove(key);
                    expired.Add(snap);
                }
            }
        }
        foreach (var e in expired) e.Dispose();
        return expired;
    }

    /// <summary>Observability snapshot of all stashed models (does not include the active one —
    /// callers merge that in separately from <see cref="ServerState"/>).</summary>
    public IReadOnlyList<ResidentModelInfo> Snapshot(DateTimeOffset now)
    {
        lock (_lock)
        {
            return _resident.Values.Select(s =>
            {
                double keepAlive = s.KeepAliveSecondsOverride ?? DefaultKeepAliveSeconds;
                double? expiresIn = keepAlive < 0 ? null : Math.Max(0, keepAlive - (now - s.LastUsedUtc).TotalSeconds);
                return new ResidentModelInfo
                {
                    Key = s.Key,
                    IsActive = false,
                    LastUsedUtc = s.LastUsedUtc,
                    EffectiveKeepAliveSeconds = keepAlive,
                    EstimatedBytes = s.EstimatedBytes,
                    ExpiresInSeconds = expiresIn,
                };
            }).ToList();
        }
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        lock (_lock)
        {
            foreach (var s in _resident.Values) s.Dispose();
            _resident.Clear();
        }
    }
}
