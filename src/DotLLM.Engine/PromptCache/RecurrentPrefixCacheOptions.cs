namespace DotLLM.Engine.PromptCache;

/// <summary>Bounds independent recurrent prefix retention. Defaults preserve the legacy single-slot path.</summary>
public sealed record RecurrentPrefixCacheOptions
{
    /// <summary>Maximum number of retained token boundaries.</summary>
    public int MaxEntries { get; init; } = 1;
    /// <summary>Maximum pinned host bytes. Zero disables host retention.</summary>
    public long HostBudgetBytes { get; init; }
    /// <summary>Private snapshot spill directory. Null disables disk retention.</summary>
    public string? DiskDirectory { get; init; }
    /// <summary>Maximum owned disk bytes including file headers and temporary writes.</summary>
    public long DiskBudgetBytes { get; init; }

    /// <summary>
    /// Mid-history resume (issue #1286): checkpoint the recurrent state every this many prefill tokens, at absolute positions
    /// that are multiples of it, so a request that diverges from the cached history (an agent trimming an old tool result)
    /// resumes from the deepest checkpoint at or before the divergence instead of re-prefilling everything. 0 (default)
    /// disables it. Applies to the single-slot path, whose KV rows are shared by every checkpoint along one history; each
    /// checkpoint copies only the recurrent state (<see cref="DotLLM.Core.Models.IModel.RecurrentStateCheckpointBytes"/>).
    /// </summary>
    public int StepTokens { get; init; }

    /// <summary>Most step checkpoints kept (the deepest are kept); default 8, as azllm's <c>--prompt-cache-step</c>.</summary>
    public int MaxStepCheckpoints { get; init; } = 8;

    /// <summary>Byte cap over the retained step checkpoints. 0 (default) leaves <see cref="MaxStepCheckpoints"/> as the only bound.</summary>
    public long StepBudgetBytes { get; init; }
}

/// <summary>Recurrent prefix cache occupancy and lifetime counters.</summary>
public sealed record RecurrentPrefixCacheStatistics
{
    /// <summary>Whether recurrent prefix reuse is enabled.</summary>
    public bool Enabled { get; init; }
    /// <summary>Retained snapshot count.</summary>
    public int Entries { get; init; }
    /// <summary>Configured maximum count.</summary>
    public int MaxEntries { get; init; }
    /// <summary>Pinned host bytes currently retained.</summary>
    public long HostBytes { get; init; }
    /// <summary>Configured pinned host budget.</summary>
    public long HostBudgetBytes { get; init; }
    /// <summary>Owned disk snapshot bytes.</summary>
    public long DiskBytes { get; init; }
    /// <summary>Configured disk budget.</summary>
    public long DiskBudgetBytes { get; init; }
    /// <summary>Number of retained disk snapshots.</summary>
    public int DiskEntries { get; init; }
    /// <summary>Disk snapshot hits.</summary>
    public long DiskHits { get; init; }
    /// <summary>Implemented disk I/O path, or null when unavailable.</summary>
    public string? DiskTransport { get; init; }
    /// <summary>Total successful hits.</summary>
    public long Hits { get; init; }
    /// <summary>Device snapshot hits.</summary>
    public long VramHits { get; init; }
    /// <summary>Host snapshot hits.</summary>
    public long HostHits { get; init; }
    /// <summary>Requests without a restorable match.</summary>
    public long Misses { get; init; }
    /// <summary>Capacity or budget evictions.</summary>
    public long Evictions { get; init; }
    /// <summary>Failed restores discarded as misses.</summary>
    public long RestoreFailures { get; init; }
    /// <summary>Mid-history step checkpoints currently retained (issue #1286).</summary>
    public int StepCheckpoints { get; init; }
    /// <summary>Bytes held by the retained step checkpoints.</summary>
    public long StepCheckpointBytes { get; init; }
}
