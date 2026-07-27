using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.Engine.KvCache;
using DotLLM.Engine.PromptCache;
using DotLLM.Engine.Scheduler;
using DotLLM.Models.Gguf;
using DotLLM.Server.RateLimiting;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;

namespace DotLLM.Server;

/// <summary>
/// Shared server state: loaded model, concurrency control, mutable configuration.
/// Endpoints access live model/generator instances through this object.
/// May start "bare" (no model loaded) when <c>dotllm serve</c> is run without a model argument.
/// </summary>
/// <remarks>
/// <b>Multi-model residency (#369):</b> at any time exactly one model's live objects are mirrored
/// onto this object's flat properties (the "active" model) — every existing endpoint keeps reading
/// <see cref="Model"/>/<see cref="Generator"/>/etc. unmodified. Additional models can be resident
/// but inactive, tracked as <see cref="ResidentModelSnapshot"/> instances inside <see cref="Residency"/>.
/// <see cref="EnsureActiveAsync"/> activates a resident (or freshly-loaded) model by key, swapping
/// it into these flat fields; the previously-active model is stashed (or evicted, depending on the
/// configured budget) rather than always disposed. Idle-unload and LRU eviction are driven by
/// <see cref="LastUsedUtc"/> / <see cref="KeepAliveSecondsOverride"/> plus the same fields on each
/// stashed snapshot.
/// </remarks>
public sealed class ServerState : IDisposable
{
    private readonly SemaphoreSlim _requestGate = new(1, 1);

    /// <summary>Server startup options (updated on model swap).</summary>
    public required ServerOptions Options { get; set; }

    /// <summary>Model configuration (null when no model loaded).</summary>
    public ModelConfig? Config { get; set; }

    /// <summary>Tool call parser for the loaded model.</summary>
    public IToolCallParser? ToolCallParser { get; set; }

    /// <summary>KV-cache configuration.</summary>
    public KvCacheConfig KvCacheConfig { get; set; }

    /// <summary>KV-cache factory for the loaded model/device.</summary>
    public Func<ModelConfig, int, IKvCache>? KvCacheFactory { get; set; }

    /// <summary>Paged KV-cache factory (non-null when paged mode is active). Owns the shared block pool.</summary>
    public PagedKvCacheFactory? PagedFactory { get; set; }

    /// <summary>Prefix cache for prompt caching (null when disabled).</summary>
    public PrefixCache? PrefixCache { get; set; }

    /// <summary>Cross-request prefix trie manager (Step 37). Non-null when paged KV-cache is active.</summary>
    public PrefixTrieManager? PrefixTrieManager { get; set; }

    /// <summary>Whether a model is loaded and ready to accept requests.</summary>
    public bool IsReady { get; set; }

    // ── Live instances (nullable — null when no model loaded) ──

    /// <summary>Currently loaded model.</summary>
    public IModel? Model { get; set; }

    /// <summary>Tokenizer for the loaded model.</summary>
    public ITokenizer? Tokenizer { get; set; }

    /// <summary>Chat template for the loaded model.</summary>
    public IChatTemplate? ChatTemplate { get; set; }

    /// <summary>Text generator wired to the current model.</summary>
    public TextGenerator? Generator { get; set; }

    /// <summary>
    /// Masked text-diffusion generator wired to the current model. Non-null
    /// <b>only</b> when the loaded model is a diffusion model
    /// (<see cref="ModelConfig.DiffusionConfig"/> is non-null). When set, chat
    /// completions route through this generator instead of <see cref="Generator"/>;
    /// the autoregressive path (this being null) is unchanged.
    /// </summary>
    public DiffusionTextGenerator? DiffusionGenerator { get; set; }

    /// <summary>
    /// Async continuous-batch scheduler. When non-null, endpoint handlers route concurrent
    /// requests through <see cref="ContinuousBatchSchedulerService.EnqueueAsync"/> instead of
    /// serialising through <see cref="ExecuteAsync"/>. Falls back to the direct-generator path
    /// when null (e.g. quantized KV-cache, hybrid/CUDA models).
    /// </summary>
    public ContinuousBatchSchedulerService? Scheduler { get; set; }

    /// <summary>
    /// Cancellation source for <see cref="Scheduler"/>'s background run-loop. Cancelled at
    /// shutdown so the loop exits cleanly.
    /// </summary>
    public CancellationTokenSource? SchedulerLoopCts { get; set; }

    /// <summary>Task driving <see cref="ContinuousBatchSchedulerService.RunLoopAsync"/>. Awaited at shutdown.</summary>
    public Task? SchedulerLoopTask { get; set; }

    /// <summary>Mutable sampling parameter defaults (changeable from the UI).</summary>
    public SamplingDefaults SamplingDefaults { get; set; } = new();

    /// <summary>Path of the currently loaded GGUF file.</summary>
    public string LoadedModelPath { get; set; } = "";

    /// <summary>Open GGUF file handle (disposed on model swap).</summary>
    public GgufFile? CurrentGguf { get; set; }

    /// <summary>Draft model for speculative decoding (null when disabled).</summary>
    public IModel? DraftModel { get; set; }

    /// <summary>Path of the loaded draft model GGUF file.</summary>
    public string DraftModelPath { get; set; } = "";

    /// <summary>Open draft GGUF file handle (disposed on model swap).</summary>
    public GgufFile? DraftGguf { get; set; }

    /// <summary>
    /// Process-wide LoRA adapter registry (singleton). Set by
    /// <see cref="ServerStartup"/> and shared between admin endpoints
    /// (<c>POST /v1/lora/load</c>) and the inference pipeline.
    /// </summary>
    public ILoraAdapterRegistry? LoraRegistry { get; set; }

    /// <summary>
    /// Per-API-key rate limiter manager. <c>null</c> when rate limiting is
    /// disabled or not configured (see <see cref="ServerOptions.RateLimit"/>).
    /// Owned by <see cref="ServerState"/> and disposed at server shutdown.
    /// </summary>
    public RateLimitManager? RateLimitManager { get; set; }

    /// <summary>
    /// Multi-model residency tracker (#369). Always non-null; default
    /// <see cref="ModelResidencyManager.MaxResidentModels"/> of 1 reproduces the original
    /// single-model hot-swap behavior.
    /// </summary>
    public ModelResidencyManager Residency { get; init; } = new();

    /// <summary>
    /// Wall-clock time the active model was last used to serve a request (or activated). Drives
    /// idle-unload for the active model (#369). Stashed models track this on their own
    /// <see cref="ResidentModelSnapshot"/>.
    /// </summary>
    public DateTimeOffset LastUsedUtc { get; set; } = DateTimeOffset.UtcNow;

    /// <summary>
    /// Per-model keep-alive override in seconds for the active model. Null = use
    /// <see cref="ModelResidencyManager.DefaultKeepAliveSeconds"/>. 0 = unload after each use.
    /// Negative = never auto-unload. Set via the <c>keep_alive</c> request/load field (#369).
    /// </summary>
    public double? KeepAliveSecondsOverride { get; set; }

    /// <summary>
    /// Approximate resident footprint (GGUF file size on disk) of the active model. Used for
    /// eviction budget accounting (#369).
    /// </summary>
    public long EstimatedBytes { get; set; }

    /// <summary>
    /// Host-shutdown token, wired by <see cref="ServerStartup.BuildApp"/>. Used by
    /// <see cref="StartSchedulerLoop"/> so schedulers rebuilt on model reactivation (#369) also
    /// stop cleanly at shutdown, same as the initially-loaded one.
    /// </summary>
    public CancellationToken ShutdownToken { get; set; } = CancellationToken.None;

    /// <summary>
    /// Executes a request with sequential access control.
    /// Only one request is processed at a time (Step 35 adds batching).
    /// </summary>
    public async Task ExecuteAsync(Func<Task> work, CancellationToken ct)
    {
        await _requestGate.WaitAsync(ct);
        try { await work(); }
        finally { _requestGate.Release(); }
    }

    /// <summary>
    /// Loads or swaps a model under the request gate.
    /// Blocks new requests during the swap. The previously-active model (if any) is stashed into
    /// <see cref="Residency"/> rather than unconditionally disposed (#369) — whether it survives
    /// depends on <see cref="ModelResidencyManager.MaxResidentModels"/> /
    /// <see cref="ModelResidencyManager.MemoryBudgetBytes"/>, evaluated right after
    /// <paramref name="loadAction"/> completes (once the new model's <see cref="EstimatedBytes"/>
    /// is known). Default configuration (<c>MaxResidentModels = 1</c>) evicts it immediately,
    /// reproducing the original hot-swap-always-disposes behavior exactly.
    /// </summary>
    public async Task SwapModelAsync(Func<Task> loadAction, CancellationToken ct)
    {
        await _requestGate.WaitAsync(ct);
        IsReady = false;
        try
        {
            await StopSchedulerAsync().ConfigureAwait(false);
            StashOrDropActiveLocked();

            await loadAction();
            IsReady = true;
            LastUsedUtc = DateTimeOffset.UtcNow;

            Residency.EnforceBudget(EstimatedBytes);
        }
        finally { _requestGate.Release(); }
    }

    /// <summary>
    /// Ensures the model identified by <paramref name="requestedModel"/> is the active one,
    /// activating it with no reload cost if it's already resident (#369). Returns an error message
    /// on failure (model not found / load exception), or <see langword="null"/> on success.
    /// </summary>
    /// <param name="requestedModel">
    /// Model key (<see cref="ServerOptions.ModelId"/>) or a resolvable model argument (file path /
    /// HuggingFace repo id). Null/empty means "whatever is/was active" — reloads it if it idled out.
    /// </param>
    /// <param name="keepAliveOverride">
    /// When non-null, sets this model's keep-alive override going forward (seconds; 0 = unload
    /// after each use, negative = never).
    /// </param>
    /// <param name="ct">Cancellation token for the load/reload, if one is needed.</param>
    public async Task<string?> EnsureActiveAsync(string? requestedModel, double? keepAliveOverride, CancellationToken ct)
    {
        if (MatchesActiveAndReady(requestedModel))
        {
            Touch(keepAliveOverride);
            return null;
        }

        await _requestGate.WaitAsync(ct);
        try
        {
            if (MatchesActiveAndReady(requestedModel))
            {
                Touch(keepAliveOverride);
                return null;
            }

            string targetKey = string.IsNullOrWhiteSpace(requestedModel) ? Options.ModelId : requestedModel!;
            if (string.IsNullOrEmpty(targetKey) || targetKey == "none")
                return "No model loaded and no model specified";

            var snapshot = Residency.TryTake(targetKey);
            if (snapshot is not null)
            {
                await StopSchedulerAsync().ConfigureAwait(false);
                StashOrDropActiveLocked();
                ActivateSnapshotLocked(snapshot, keepAliveOverride);
                Residency.EnforceBudget(EstimatedBytes);
                return null;
            }

            string reloadPath;
            ServerOptions loadOptions;
            long incomingBytes;
            if (string.Equals(targetKey, Options.ModelId, StringComparison.OrdinalIgnoreCase) && !string.IsNullOrEmpty(LoadedModelPath))
            {
                // Same model as before, just idled out - reload from the same path/options.
                reloadPath = LoadedModelPath;
                loadOptions = Options;
                incomingBytes = EstimatedBytes > 0 ? EstimatedBytes : SafeFileLength(reloadPath);
            }
            else
            {
                var resolvedPath = ServerStartup.ResolveModelPath(targetKey, quant: null);
                if (resolvedPath is null)
                    return $"Model not found: {targetKey}";
                reloadPath = resolvedPath;
                loadOptions = Options with
                {
                    Model = targetKey,
                    Quant = null,
                    ModelId = Path.GetFileNameWithoutExtension(resolvedPath),
                };
                incomingBytes = SafeFileLength(resolvedPath);
            }

            var loaded = await Task.Run(() => ServerStartup.LoadModel(reloadPath, loadOptions), ct).ConfigureAwait(false);

            await StopSchedulerAsync().ConfigureAwait(false);
            StashOrDropActiveLocked();
            InstallLoadedModelLocked(loaded, incomingBytes, keepAliveOverride);
            Residency.EnforceBudget(EstimatedBytes);
            return null;
        }
        catch (Exception ex)
        {
            IsReady = Model is not null;
            return ex.Message;
        }
        finally { _requestGate.Release(); }
    }

    /// <summary>
    /// Lists every resident model (active + stashed) for observability (<c>GET /v1/models</c>, #369).
    /// </summary>
    public IReadOnlyList<ResidentModelInfo> ListResidentModels()
    {
        var now = DateTimeOffset.UtcNow;
        var result = new List<ResidentModelInfo>();
        if (IsReady && Model is not null)
        {
            double keepAlive = KeepAliveSecondsOverride ?? Residency.DefaultKeepAliveSeconds;
            double? expiresIn = keepAlive < 0 ? null : Math.Max(0, keepAlive - (now - LastUsedUtc).TotalSeconds);
            result.Add(new ResidentModelInfo
            {
                Key = Options.ModelId,
                IsActive = true,
                LastUsedUtc = LastUsedUtc,
                EffectiveKeepAliveSeconds = keepAlive,
                EstimatedBytes = EstimatedBytes,
                ExpiresInSeconds = expiresIn,
            });
        }
        result.AddRange(Residency.Snapshot(now));
        return result;
    }

    /// <summary>
    /// One idle-sweep pass (#369): disposes stashed models past their keep-alive unconditionally
    /// (they're not in use by any in-flight request), and — only if the request gate is free right
    /// now, so a busy generation is never interrupted — deactivates the active model if it's past
    /// its keep-alive too. Intended to be called periodically by <see cref="RunIdleSweepLoopAsync"/>.
    /// </summary>
    public async Task SweepIdleAsync()
    {
        Residency.SweepExpired(DateTimeOffset.UtcNow);

        if (!IsReady || Model is null) return;
        if (!IsActiveExpired()) return;

        if (!await _requestGate.WaitAsync(0).ConfigureAwait(false)) return; // busy - try again next tick
        try
        {
            if (!IsReady || Model is null) return;
            if (!IsActiveExpired()) return;

            await StopSchedulerAsync().ConfigureAwait(false);
            DisposeActiveLiveFieldsKeepPathAndOptions();
            IsReady = false;
        }
        finally { _requestGate.Release(); }
    }

    /// <summary>Background loop calling <see cref="SweepIdleAsync"/> on <paramref name="interval"/>. Runs
    /// until <paramref name="ct"/> is cancelled (host shutdown).</summary>
    public async Task RunIdleSweepLoopAsync(TimeSpan interval, CancellationToken ct)
    {
        using var timer = new PeriodicTimer(interval);
        try
        {
            while (await timer.WaitForNextTickAsync(ct).ConfigureAwait(false))
                await SweepIdleAsync().ConfigureAwait(false);
        }
        catch (OperationCanceledException) { /* shutdown */ }
    }

    /// <summary>
    /// Starts (or restarts) the background run-loop for <see cref="Scheduler"/>, linked to
    /// <see cref="ShutdownToken"/>. No-op when <see cref="Scheduler"/> is null.
    /// </summary>
    public void StartSchedulerLoop()
    {
        if (Scheduler is null) return;
        var cts = CancellationTokenSource.CreateLinkedTokenSource(ShutdownToken);
        var scheduler = Scheduler;
        SchedulerLoopCts = cts;
        SchedulerLoopTask = Task.Run(() => scheduler.RunLoopAsync(cts.Token));
    }

    private bool MatchesActiveAndReady(string? requestedModel) =>
        IsReady && Model is not null &&
        (string.IsNullOrWhiteSpace(requestedModel) || string.Equals(Options.ModelId, requestedModel, StringComparison.OrdinalIgnoreCase));

    private void Touch(double? keepAliveOverride)
    {
        LastUsedUtc = DateTimeOffset.UtcNow;
        if (keepAliveOverride.HasValue)
            KeepAliveSecondsOverride = keepAliveOverride;
    }

    private bool IsActiveExpired()
    {
        double effective = KeepAliveSecondsOverride ?? Residency.DefaultKeepAliveSeconds;
        if (effective < 0) return false; // never expire
        return (DateTimeOffset.UtcNow - LastUsedUtc).TotalSeconds >= effective;
    }

    private static long SafeFileLength(string path) => ServerStartup.SafeFileLength(path);

    /// <summary>
    /// Moves the currently-active model's live objects into a <see cref="ResidentModelSnapshot"/>
    /// and stashes it in <see cref="Residency"/> (subject to later <see cref="ModelResidencyManager.EnforceBudget"/>
    /// eviction). Detaches the flat fields without disposing — ownership transfers to the snapshot.
    /// No-op when nothing is currently loaded. Caller must hold <see cref="_requestGate"/> and have
    /// already stopped the scheduler.
    /// </summary>
    private void StashOrDropActiveLocked()
    {
        if (Model is null) return;

        var snapshot = new ResidentModelSnapshot
        {
            Key = Options.ModelId,
            Options = Options,
            Config = Config,
            ToolCallParser = ToolCallParser,
            KvCacheConfig = KvCacheConfig,
            KvCacheFactory = KvCacheFactory,
            PagedFactory = PagedFactory,
            PrefixCache = PrefixCache,
            PrefixTrieManager = PrefixTrieManager,
            Model = Model,
            Tokenizer = Tokenizer,
            ChatTemplate = ChatTemplate,
            Generator = Generator,
            DiffusionGenerator = DiffusionGenerator,
            LoadedModelPath = LoadedModelPath,
            CurrentGguf = CurrentGguf,
            DraftModel = DraftModel,
            DraftModelPath = DraftModelPath,
            DraftGguf = DraftGguf,
            EstimatedBytes = EstimatedBytes,
            KeepAliveSecondsOverride = KeepAliveSecondsOverride,
            LastUsedUtc = LastUsedUtc,
        };
        Residency.Stash(snapshot);

        // Ownership transferred to the snapshot above - detach without disposing.
        Config = null;
        ToolCallParser = null;
        KvCacheFactory = null;
        PagedFactory = null;
        PrefixCache = null;
        PrefixTrieManager = null;
        Model = null;
        Tokenizer = null;
        ChatTemplate = null;
        Generator = null;
        DiffusionGenerator = null;
        CurrentGguf = null;
        DraftModel = null;
        DraftGguf = null;
        KeepAliveSecondsOverride = null;
    }

    /// <summary>Reactivates a stashed snapshot onto the flat fields and rebuilds its scheduler (cheap —
    /// in-memory only). Caller must hold <see cref="_requestGate"/>.</summary>
    private void ActivateSnapshotLocked(ResidentModelSnapshot snapshot, double? keepAliveOverride)
    {
        Options = snapshot.Options;
        Config = snapshot.Config;
        ToolCallParser = snapshot.ToolCallParser;
        KvCacheConfig = snapshot.KvCacheConfig;
        KvCacheFactory = snapshot.KvCacheFactory;
        PagedFactory = snapshot.PagedFactory;
        PrefixCache = snapshot.PrefixCache;
        PrefixTrieManager = snapshot.PrefixTrieManager;
        Model = snapshot.Model;
        Tokenizer = snapshot.Tokenizer;
        ChatTemplate = snapshot.ChatTemplate;
        Generator = snapshot.Generator;
        DiffusionGenerator = snapshot.DiffusionGenerator;
        LoadedModelPath = snapshot.LoadedModelPath;
        CurrentGguf = snapshot.CurrentGguf;
        DraftModel = snapshot.DraftModel;
        DraftModelPath = snapshot.DraftModelPath;
        DraftGguf = snapshot.DraftGguf;
        EstimatedBytes = snapshot.EstimatedBytes;
        KeepAliveSecondsOverride = keepAliveOverride ?? snapshot.KeepAliveSecondsOverride;
        IsReady = true;
        LastUsedUtc = DateTimeOffset.UtcNow;

        RebuildSchedulerLocked();
    }

    /// <summary>Installs a freshly-<see cref="ServerStartup.LoadModel"/>-ed state onto the flat fields
    /// (used by <see cref="EnsureActiveAsync"/>'s fresh-load path). Caller must hold the request gate.</summary>
    private void InstallLoadedModelLocked(ServerState loaded, long estimatedBytes, double? keepAliveOverride)
    {
        Options = loaded.Options;
        Config = loaded.Config;
        ToolCallParser = loaded.ToolCallParser;
        KvCacheConfig = loaded.KvCacheConfig;
        KvCacheFactory = loaded.KvCacheFactory;
        PagedFactory = loaded.PagedFactory;
        PrefixCache = loaded.PrefixCache;
        PrefixTrieManager = loaded.PrefixTrieManager;
        Model = loaded.Model;
        Tokenizer = loaded.Tokenizer;
        ChatTemplate = loaded.ChatTemplate;
        Generator = loaded.Generator;
        DiffusionGenerator = loaded.DiffusionGenerator;
        Scheduler = loaded.Scheduler;
        LoadedModelPath = loaded.LoadedModelPath;
        CurrentGguf = loaded.CurrentGguf;
        DraftModel = loaded.DraftModel;
        DraftModelPath = loaded.DraftModelPath;
        DraftGguf = loaded.DraftGguf;
        EstimatedBytes = estimatedBytes;
        KeepAliveSecondsOverride = keepAliveOverride;
        IsReady = true;
        LastUsedUtc = DateTimeOffset.UtcNow;

        if (loaded.LoraRegistry is not null && !ReferenceEquals(loaded.LoraRegistry, LoraRegistry))
            loaded.LoraRegistry.Dispose();

        StartSchedulerLoop();
    }

    private void RebuildSchedulerLocked()
    {
        Scheduler = null;
        SchedulerLoopCts = null;
        SchedulerLoopTask = null;

        if (PagedFactory is null || KvCacheFactory is null || DraftModel is not null || Model is null || Tokenizer is null)
            return;

        var schedulerOptions = ServerStartup.ResolveSchedulerOptions(Options);
        Scheduler = new ContinuousBatchSchedulerService(
            Model, Tokenizer, KvCacheFactory, options: schedulerOptions, pagedPool: PagedFactory.Pool);
        StartSchedulerLoop();
    }

    /// <summary>
    /// Cancels the scheduler's run loop and awaits its exit. Idempotent.
    /// </summary>
    public async Task StopSchedulerAsync()
    {
        var cts = SchedulerLoopCts;
        var task = SchedulerLoopTask;
        if (cts is not null)
        {
            try { cts.Cancel(); } catch { /* already disposed */ }
        }
        if (task is not null)
        {
            try { await task.ConfigureAwait(false); }
            catch (OperationCanceledException) { /* expected */ }
            catch { /* loop swallows other errors */ }
        }
        Scheduler?.Dispose();
        cts?.Dispose();
        Scheduler = null;
        SchedulerLoopCts = null;
        SchedulerLoopTask = null;
    }

    /// <summary>Disposes the active model's live objects but keeps <see cref="Options"/> and
    /// <see cref="LoadedModelPath"/> intact so a later <see cref="EnsureActiveAsync"/> call can
    /// lazily reload it. Used by the idle-unload sweep (#369).</summary>
    private void DisposeActiveLiveFieldsKeepPathAndOptions()
    {
        PrefixCache?.Dispose();
        PrefixTrieManager?.Dispose();
        PagedFactory?.Dispose();
        DraftModel?.Dispose();
        DraftGguf?.Dispose();
        Model?.Dispose();
        CurrentGguf?.Dispose();

        Config = null;
        ToolCallParser = null;
        KvCacheFactory = null;
        PagedFactory = null;
        PrefixCache = null;
        PrefixTrieManager = null;
        Model = null;
        Tokenizer = null;
        ChatTemplate = null;
        Generator = null;
        DiffusionGenerator = null;
        CurrentGguf = null;
        DraftModel = null;
        DraftGguf = null;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        try { StopSchedulerAsync().GetAwaiter().GetResult(); }
        catch { /* shutdown best-effort */ }
        RateLimitManager?.Dispose();
        PrefixCache?.Dispose();
        PrefixTrieManager?.Dispose();
        PagedFactory?.Dispose();
        DraftModel?.Dispose();
        DraftGguf?.Dispose();
        Model?.Dispose();
        CurrentGguf?.Dispose();
        LoraRegistry?.Dispose();
        Residency.Dispose();
        _requestGate.Dispose();
    }
}

/// <summary>
/// Immutable sampling parameter defaults that can be changed from the UI.
/// These serve as defaults when the per-request body does not specify a value.
/// Replaced atomically via <c>with</c> expressions to avoid torn reads.
/// </summary>
public sealed record SamplingDefaults
{
    /// <summary>Sampling temperature. 0 = greedy.</summary>
    public float Temperature { get; init; } = 0.0f;

    /// <summary>Top-P (nucleus) sampling threshold.</summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>Top-K sampling. 0 = disabled.</summary>
    public int TopK { get; init; }

    /// <summary>Min-P sampling threshold. 0 = disabled.</summary>
    public float MinP { get; init; }

    /// <summary>Repetition penalty factor. 1.0 = disabled.</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Maximum tokens to generate per response.</summary>
    public int MaxTokens { get; init; } = 2048;

    /// <summary>Random seed for reproducibility. Null = non-deterministic.</summary>
    public int? Seed { get; init; }
}
