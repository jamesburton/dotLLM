using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Engine;
using DotLLM.HuggingFace;
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

    /// <summary>Who holds <see cref="_requestGate"/> (<c>req#3 generate</c>), for the lock-wait log line (#1021). Null when free.</summary>
    private volatile string? _gateHolder;

    /// <summary>When <see cref="_gateHolder"/> took the gate (UTC ticks); 0 when free (#1023).</summary>
    private long _gateHeldSinceTicks;

    /// <summary>Requests currently waiting for <see cref="_requestGate"/> (#1023).</summary>
    private int _gateWaiters;

    /// <summary>The model being loaded right now (id, start, triggering request), or null (#1023).</summary>
    private volatile LoadingInfo? _loading;

    private sealed record LoadingInfo(string ModelId, DateTimeOffset SinceUtc, string? Request);

    /// <summary>
    /// How long a request that just activated a model keeps first claim on it (#1023): long enough to render its prompt and come back
    /// for the gate, short enough that a request that never comes back cannot stall the queue.
    /// </summary>
    internal static readonly TimeSpan ActivationGrantWindow = TimeSpan.FromSeconds(30);

    /// <summary>
    /// The request that just activated the current model and has not yet generated on it (#1023). Until it does (or ends, or the
    /// window passes), a queued request that needs a DIFFERENT model yields the gate to it instead of swapping the model away: without
    /// this, FIFO order handed the gate to an older waiter first, which swapped the model back, so each queued request reloaded twice.
    /// </summary>
    private volatile ActivationGrant? _activationGrant;

    private sealed class ActivationGrant(ServerLog.RequestScope request, string modelId, DateTimeOffset until)
    {
        private readonly TaskCompletionSource _released = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public ServerLog.RequestScope Request { get; } = request;
        public string ModelId { get; } = modelId;
        public DateTimeOffset Until { get; } = until;
        public Task Released => _released.Task;
        public void Release() => _released.TrySetResult();
        public bool IsLive => !Released.IsCompleted && !Request.Ended.IsCompleted && DateTimeOffset.UtcNow < Until;
    }

    /// <summary>
    /// Probe for a CUDA device's (free, total) bytes, used to make room on the device before a load (#1023). Null result = unknown.
    /// Settable for tests; defaults to NVML through <see cref="DotLLM.Cuda.CudaMemoryInfo"/>.
    /// </summary>
    internal Func<int, (long Free, long Total)?> CudaMemoryProbe { get; set; } =
        ordinal => DotLLM.Cuda.CudaMemoryInfo.TryGet(ordinal, out long free, out long total) ? (free, total) : null;

    /// <summary>The loader an activation uses (<see cref="ServerStartup.LoadModel"/>); settable so tests can observe the state at load time.</summary>
    internal Func<string, ServerOptions, ServerState> ModelLoader { get; set; } = ServerStartup.LoadModel;

    /// <summary>Server startup options (updated on model swap).</summary>
    public required ServerOptions Options { get; set; }

    /// <summary>Set when <c>--device auto</c> had to fall back to the CPU after a GPU load failed (#733); surfaced via /props.</summary>
    public string? DeviceFallbackWarning { get; set; }

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

    /// <summary>GPU-resident paged KV block-pool factory (issue #252), non-null only for a CUDA
    /// model loaded with <c>--paged</c>. Not consulted by <see cref="RebuildSchedulerLocked"/> —
    /// the continuous-batch scheduler runs through <c>IModel.ForwardBatch</c>, which CUDA does not
    /// yet override (see <c>docs/perf/CUDA_PAGED_ATTENTION_DESIGN.md</c>), so a CUDA model with
    /// paged KV-cache still serves each request through the per-request <see cref="TextGenerator"/>
    /// path, same as before this field existed.</summary>
    public DotLLM.Cuda.CudaPagedKvCacheFactory? CudaPagedFactory { get; set; }

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
    /// serialising through <see cref="ExecuteAsync(Func{Task}, CancellationToken)"/>. Falls back to the direct-generator path
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

    /// <summary>The active model's profile (#716), looked up by its model id, or null when it was not loaded through a profile name.</summary>
    public ModelProfile? ActiveProfile =>
        ModelProfileStore.Resolve(Options.ModelId) is { } r ? ModelProfileStore.Merge(r.Chain) : null;

    /// <summary>
    /// <see cref="SamplingDefaults"/> overlaid with the active profile's values: what a request that omits a parameter gets. The global defaults
    /// (<c>/v1/config</c>) are never modified by a profile.
    /// </summary>
    public SamplingDefaults EffectiveSamplingDefaults => ActiveProfile is { } p ? SamplingDefaults.OverlayProfile(p) : SamplingDefaults;

    /// <summary>Path of the currently loaded GGUF file.</summary>
    public string LoadedModelPath { get; set; } = "";

    /// <summary>Open GGUF file handle (disposed on model swap).</summary>
    public GgufFile? CurrentGguf { get; set; }

    /// <summary>True when MTP self-speculative decoding is actually engaging requests for the loaded model (#757).</summary>
    public bool MtpActive { get; set; }

    /// <summary>Short reason for <see cref="MtpActive"/> (<c>active</c>, <c>off (--no-mtp)</c>, <c>skipped (...)</c>, ...), surfaced in /props.</summary>
    public string? MtpStatus { get; set; }

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
    /// Operator-curated enable/disable list (#454). Always non-null; empty by default, so every
    /// model is loadable unless explicitly disabled via <c>POST /v1/models/disable</c>.
    /// </summary>
    public ModelCatalog Catalog { get; init; } = new();

    /// <summary>
    /// Background model-download jobs (#454, <c>POST /v1/models/pull</c>). Always non-null;
    /// idle until a pull is requested.
    /// </summary>
    public ModelPullManager PullManager { get; init; } = new();

    /// <summary>
    /// Live idle-sweep interval in seconds (#454). Mutable so <c>PUT /v1/settings</c> takes effect
    /// without a restart: <see cref="RunIdleSweepLoopAsync"/> re-reads this on every tick and
    /// re-arms its <see cref="PeriodicTimer"/>. Seeded from
    /// <see cref="ServerOptions.IdleSweepInterval"/>, and deliberately NOT stored back on
    /// <see cref="Options"/> — that record is replaced wholesale on every model swap
    /// (<see cref="ActivateSnapshotLocked"/>, <c>POST /v1/models/load</c>), which would silently
    /// revert any runtime setting kept there.
    /// </summary>
    public double IdleSweepIntervalSeconds { get; set; } = 5;

    /// <summary>
    /// Host-shutdown token, wired by <see cref="ServerStartup.BuildApp"/>. Used by
    /// <see cref="StartSchedulerLoop"/> so schedulers rebuilt on model reactivation (#369) also
    /// stop cleanly at shutdown, same as the initially-loaded one.
    /// </summary>
    public CancellationToken ShutdownToken { get; set; } = CancellationToken.None;

    /// <summary>
    /// Executes a request with sequential access control: one direct-generator request at a time,
    /// and never concurrently with a continuous-batch scheduler step.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>The request gate alone is not enough (#461).</b> When a scheduler is running it drives
    /// forward passes on the same model from its own background loop, deliberately <i>outside</i>
    /// this gate — batching is the whole point. The model's scratch buffers are shared mutable
    /// state, so a direct-generator forward running beside a scheduler step is not a numerical
    /// wobble: it tears down the shared <c>ComputeThreadPool</c> and the process dies with
    /// <c>CountdownEvent ... below zero</c>.
    /// </para>
    /// <para>
    /// The scheduler lease is therefore taken here, inside the gate, rather than left to each
    /// caller to remember — which is the follow-up #461 asked for. Every direct forward in the
    /// server already routes through this method, so the protection is now structural instead of
    /// a convention: <b>streaming chat and completions always take the direct-generator path</b>,
    /// so before this they raced the loop on every request that overlapped a batched one.
    /// </para>
    /// <para>
    /// <b>Cost, stated:</b> the lease is held for the whole of <paramref name="work"/>, so a long
    /// streaming generation stalls in-flight batched requests for its duration. That is the
    /// correct trade — the two cannot run at once in any case — but it is a real throughput change
    /// for a mixed workload, not a free fix. Callers that need finer granularity should take the
    /// lease themselves around the forward only.
    /// </para>
    /// <para>
    /// Lock order is gate-then-lease, matching what the embeddings path already documented. Never
    /// acquire them the other way round, and never call this from inside a held lease — the
    /// semaphores are not reentrant.
    /// </para>
    /// </remarks>
    public async Task ExecuteAsync(Func<Task> work, CancellationToken ct)
    {
        await AcquireGateAsync("generate", ct).ConfigureAwait(false);
        try
        {
            using var lease = Scheduler is { } scheduler
                ? await scheduler.AcquireModelAsync(ct)
                : null;
            await work();
        }
        finally { ReleaseGate(); }
    }

    /// <summary>
    /// <see cref="ExecuteAsync(Func{Task}, CancellationToken)"/> for a request admitted against <paramref name="admittedModelId"/>:
    /// once the gate is held, the model is re-checked and the CURRENT generator is handed to <paramref name="work"/> (#1023).
    /// </summary>
    /// <remarks>
    /// A request captures the generator when it is admitted, then may wait behind another request that swaps the model. Running on
    /// the captured instance would use a disposed (or a different) model. So after acquiring the gate: if the admitted model is still
    /// the active, ready one its current generator is used; otherwise it is re-activated under the gate already held (a stashed one
    /// is a field swap, an evicted one is reloaded), and when that fails a <see cref="ModelUnavailableException"/> is thrown.
    /// </remarks>
    /// <param name="admittedModelId">The <see cref="ServerOptions.ModelId"/> the request was admitted (and its prompt rendered) for.</param>
    /// <param name="work">The generation, given the generator to use. It must not use a generator captured earlier.</param>
    /// <param name="ct">Cancels the gate wait and any re-activation.</param>
    /// <returns>A task that completes when <paramref name="work"/> has.</returns>
    /// <exception cref="ModelUnavailableException">The admitted model is no longer active and could not be re-activated.</exception>
    public async Task ExecuteAsync(string admittedModelId, Func<TextGenerator, Task> work, CancellationToken ct)
    {
        await AcquireGateHonouringGrantAsync("generate", () => !ActiveIs(admittedModelId), ct).ConfigureAwait(false);
        try
        {
            if (_activationGrant is { } grant && ReferenceEquals(grant.Request, ServerLog.CurrentRequest))
            {
                _activationGrant = null;   // the activating request is generating now: its claim is used up
                grant.Release();
            }
            var generator = await RevalidateLockedAsync(admittedModelId, ct).ConfigureAwait(false);
            using var lease = Scheduler is { } scheduler
                ? await scheduler.AcquireModelAsync(ct)
                : null;
            await work(generator);
        }
        finally { ReleaseGate(); }
    }

    private bool ActiveIs(string modelId) =>
        IsReady && Generator is not null && string.Equals(Options.ModelId, modelId, StringComparison.OrdinalIgnoreCase);

    /// <summary>
    /// <see cref="AcquireGateAsync"/>, but when another request holds a live activation grant (it just loaded the current model and
    /// has not generated yet) and this caller would swap the model away (<paramref name="wouldSwap"/>), the gate is released again and
    /// the caller waits until that request generates, ends, or the grant window passes (#1023). On return the gate is held; on an
    /// exception it is not.
    /// </summary>
    private async Task AcquireGateHonouringGrantAsync(string purpose, Func<bool> wouldSwap, CancellationToken ct)
    {
        await AcquireGateAsync(purpose, ct).ConfigureAwait(false);
        while (_activationGrant is { } grant && grant.IsLive
               && !ReferenceEquals(grant.Request, ServerLog.CurrentRequest) && wouldSwap())
        {
            ServerLog.Info($"yielding to {grant.Request.Tag}, which just activated {grant.ModelId} and has not generated yet");
            ReleaseGate();
            var remaining = grant.Until - DateTimeOffset.UtcNow;
            using (var wake = CancellationTokenSource.CreateLinkedTokenSource(ct))
            {
                var timeout = Task.Delay(remaining > TimeSpan.Zero ? remaining : TimeSpan.Zero, wake.Token);
                await Task.WhenAny(grant.Released, grant.Request.Ended, timeout).ConfigureAwait(false);
                wake.Cancel();
            }
            ct.ThrowIfCancellationRequested();
            await AcquireGateAsync(purpose, ct).ConfigureAwait(false);
        }
    }

    /// <summary>The active generator when <paramref name="modelId"/> is the active, ready model; otherwise re-activates it. Gate held.</summary>
    private async Task<TextGenerator> RevalidateLockedAsync(string modelId, CancellationToken ct)
    {
        if (IsReady && Generator is { } current && string.Equals(Options.ModelId, modelId, StringComparison.OrdinalIgnoreCase))
            return current;

        ServerLog.Info($"model changed while queued (active: {(IsReady ? Options.ModelId : "none")}); re-activating {modelId}");
        string? error = await ActivateLockedAsync(modelId, keepAliveOverride: null, ct).ConfigureAwait(false);
        if (error is not null || Generator is null)
            throw new ModelUnavailableException(modelId, error ?? "the model has no text generator");
        return Generator;
    }

    /// <summary>
    /// What the server is doing right now (#1023): <c>loading</c> (a model load is in progress), <c>busy</c> (a request holds the
    /// execution lock), <c>idle</c>, or <c>no_model</c> - with the load target and start time, the lock holder and since when, and
    /// how many requests are queued. Served by <c>GET /v1/status</c> and embedded in <c>/props</c>.
    /// </summary>
    /// <returns>A point-in-time snapshot.</returns>
    public Models.ServerStatusResponse GetStatus()
    {
        var now = DateTimeOffset.UtcNow;
        var loading = _loading;
        string? holder = _gateHolder;
        long heldTicks = Interlocked.Read(ref _gateHeldSinceTicks);
        DateTimeOffset? heldSince = holder is not null && heldTicks > 0 ? new DateTimeOffset(heldTicks, TimeSpan.Zero) : null;
        bool ready = IsReady && Model is not null;

        return new Models.ServerStatusResponse
        {
            State = loading is not null ? "loading" : holder is not null ? "busy" : ready ? "idle" : "no_model",
            IsReady = ready,
            ModelId = ready ? Options.ModelId : null,
            LoadingModelId = loading?.ModelId,
            LoadingSince = loading?.SinceUtc,
            LoadingElapsedSeconds = loading is null ? null : Math.Round((now - loading.SinceUtc).TotalSeconds, 1),
            LoadingRequest = loading?.Request,
            BusyWith = holder,
            BusySince = heldSince,
            BusyElapsedSeconds = heldSince is { } s ? Math.Round((now - s).TotalSeconds, 1) : null,
            WaitingRequests = Volatile.Read(ref _gateWaiters),
        };
    }

    /// <summary>
    /// One line for an SSE keep-alive comment (#1023): what a queued request is waiting for, e.g.
    /// <c>loading owner/repo/x.gguf (started 2026-10-16T14:12:54Z, 73s)</c> or <c>busy: req#3 generate (41s), 2 queued</c>.
    /// </summary>
    /// <returns>The description.</returns>
    public string DescribeWait()
    {
        var s = GetStatus();
        return s.State switch
        {
            "loading" => $"loading {s.LoadingModelId} (started {s.LoadingSince:yyyy-MM-ddTHH:mm:ssZ}, {s.LoadingElapsedSeconds:F0}s)",
            "busy" => $"busy: {s.BusyWith} ({s.BusyElapsedSeconds:F0}s)" + (s.WaitingRequests > 0 ? $", {s.WaitingRequests} queued" : ""),
            _ => "the server",
        };
    }

    /// <summary>
    /// Takes <see cref="_requestGate"/>, logging (#1021) when it is not immediately free: who holds it, then how long the wait took.
    /// An uncontended acquire logs nothing.
    /// </summary>
    private async Task AcquireGateAsync(string purpose, CancellationToken ct)
    {
        string me = ServerLog.RequestTag is { } tag ? $"{tag} {purpose}" : purpose;
        if (!_requestGate.Wait(0))
        {
            ServerLog.Info($"waiting for the server lock ({purpose}); held by {_gateHolder ?? "?"}");
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            Interlocked.Increment(ref _gateWaiters);
            try
            {
                await _requestGate.WaitAsync(ct).ConfigureAwait(false);
            }
            catch (OperationCanceledException)
            {
                ServerLog.Info($"gave up waiting for the server lock after {ServerLog.Duration(System.Diagnostics.Stopwatch.GetElapsedTime(start))} (cancelled)");
                throw;
            }
            finally { Interlocked.Decrement(ref _gateWaiters); }
            ServerLog.Info($"acquired the server lock after {ServerLog.Duration(System.Diagnostics.Stopwatch.GetElapsedTime(start))}");
        }
        MarkGateHeld(me);
    }

    private void MarkGateHeld(string holder)
    {
        Interlocked.Exchange(ref _gateHeldSinceTicks, DateTimeOffset.UtcNow.UtcTicks);
        _gateHolder = holder;
    }

    /// <summary>Releases <see cref="_requestGate"/> taken by <see cref="AcquireGateAsync"/>.</summary>
    private void ReleaseGate()
    {
        _gateHolder = null;
        Interlocked.Exchange(ref _gateHeldSinceTicks, 0);
        _requestGate.Release();
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
    /// <remarks>
    /// (#1023) When <paramref name="incomingBytes"/> is given, the budget is enforced with the incoming model in the active slot
    /// BEFORE <paramref name="loadAction"/> runs, so a model the budget would evict anyway (always, by default) is freed first rather
    /// than sharing device memory with its replacement during the load.
    /// </remarks>
    /// <param name="loadAction">Loads the new model and installs it on this state.</param>
    /// <param name="ct">Cancels the gate wait; checked again before anything is freed.</param>
    /// <param name="incomingModelId">The id being loaded, for <see cref="GetStatus"/>; null = unknown.</param>
    /// <param name="incomingBytes">The incoming model's size; 0 = unknown (no budget enforcement before the load).</param>
    /// <returns>A task that completes when the swap has.</returns>
    public async Task SwapModelAsync(Func<Task> loadAction, CancellationToken ct, string? incomingModelId = null, long incomingBytes = 0)
    {
        await AcquireGateAsync("model load", ct).ConfigureAwait(false);
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        try
        {
            ct.ThrowIfCancellationRequested();   // (#1023) the caller has gone: do not free or load anything
            IsReady = false;
            _loading = new LoadingInfo(incomingModelId ?? "(explicit load)", DateTimeOffset.UtcNow, ServerLog.RequestTag);
            ServerLog.Info($"model load started (POST /v1/models/load){(incomingModelId is null ? "" : ": " + incomingModelId)}");
            await StopSchedulerAsync().ConfigureAwait(false);
            StashOrDropActiveLocked();
            if (incomingBytes > 0)
                FreeForIncomingLocked(incomingBytes, Options.Device);

            await loadAction();
            IsReady = true;
            LastUsedUtc = DateTimeOffset.UtcNow;
            ServerLog.Info($"model loaded: {Options.ModelId} in {ServerLog.Duration(System.Diagnostics.Stopwatch.GetElapsedTime(start))}");

            Residency.EnforceBudget(EstimatedBytes);
        }
        catch (Exception ex)
        {
            ServerLog.Info($"model load failed after {ServerLog.Duration(System.Diagnostics.Stopwatch.GetElapsedTime(start))}: {ex.Message}");
            throw;
        }
        finally
        {
            _loading = null;
            ReleaseGate();
        }
    }

    /// <summary>
    /// Makes room for an incoming model BEFORE it loads (#1023). The active model must already be stashed. First the residency budget
    /// is enforced with the incoming model in the active slot - by default (<c>MaxResidentModels = 1</c>) that evicts every stashed
    /// model, the old active one included, so its memory is free before the new load starts. Then, for a CUDA target, stashed models
    /// are evicted least-recently-used first until the incoming weights fit in the device's FREE memory (with the usual headroom).
    /// Whether the model fits at all is decided by the device plan in <see cref="ServerStartup.LoadModel"/>, which now sees free memory.
    /// </summary>
    private void FreeForIncomingLocked(long incomingBytes, string? device)
    {
        foreach (var evicted in Residency.EnforceBudget(incomingBytes))
            ServerLog.Info($"evicted {evicted.Key} before loading (residency budget)");

        if (CudaOrdinalFor(device) is not { } ordinal) return;
        while (Residency.StashedCount > 0 && CudaMemoryProbe(ordinal) is { } mem
               && incomingBytes + (long)(mem.Total * (1 - DeviceSelector.CudaFitFraction)) > mem.Free)
        {
            if (Residency.EvictLeastRecentlyUsed() is not { } evicted) break;
            ServerLog.Info($"evicted {evicted.Key} before loading: {incomingBytes / (1024.0 * 1024 * 1024):F1} GiB needs room on gpu:{ordinal} "
                + $"({mem.Free / (1024.0 * 1024 * 1024):F1} GiB free)");
        }
    }

    /// <summary>The CUDA ordinal a load on <paramref name="device"/> would use: explicit <c>gpu[:N]</c>, or 0 for <c>auto</c>; null otherwise.</summary>
    private static int? CudaOrdinalFor(string? device) =>
        DeviceSelector.IsAuto(device) ? 0
        : DeviceSpec.TryParse(device, out var spec, out _) && spec.Kind == DeviceKind.Cuda ? spec.Ordinal
        : null;

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
        if (ServerLog.RequestTag is not null)
            ServerLog.Info($"model={(string.IsNullOrWhiteSpace(requestedModel) ? "(active)" : requestedModel)}");

        if (MatchesActiveAndReady(requestedModel))
        {
            Touch(keepAliveOverride);
            return null;
        }

        await AcquireGateHonouringGrantAsync("model activation", () => !MatchesActiveAndReady(requestedModel), ct).ConfigureAwait(false);
        try
        {
            bool swapping = !MatchesActiveAndReady(requestedModel);
            string? error = await ActivateLockedAsync(requestedModel, keepAliveOverride, ct).ConfigureAwait(false);

            // (#1023) This request changed the active model and will come back for the gate to generate: give it first claim, so a
            // waiter that needs another model does not swap this one away (and force a second reload) in between.
            if (error is null && swapping && ServerLog.CurrentRequest is { } me)
            {
                _activationGrant?.Release();
                _activationGrant = new ActivationGrant(me, Options.ModelId, DateTimeOffset.UtcNow + ActivationGrantWindow);
            }
            return error;
        }
        finally { ReleaseGate(); }
    }

    /// <summary>
    /// The body of <see cref="EnsureActiveAsync"/>, run with the gate already held (also by <see cref="RevalidateLockedAsync"/>).
    /// Returns an error message, or null on success.
    /// </summary>
    /// <remarks>
    /// A fresh load (#1023): cancellation is checked before anything is freed; the active model is stashed and the budget enforced
    /// BEFORE the load (free-before-load - it used to load the new model next to the old one and evict afterwards); <see cref="IsReady"/>
    /// is false and <see cref="GetStatus"/> reports <c>loading</c> for the whole load.
    /// </remarks>
    private async Task<string?> ActivateLockedAsync(string? requestedModel, double? keepAliveOverride, CancellationToken ct)
    {
        long loadStart = 0;
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

            // (#454) A disabled model may not be (re)activated. Checked here rather than only in
            // the endpoint so the implicit activation driven by a chat request's `model` field is
            // covered too, not just the explicit POST /v1/models/load path.
            if (!Catalog.IsEnabled(targetKey))
                return $"Model is disabled: {targetKey}";

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
                var resolvedPath = await ServerStartup.ResolveOrPullAsync(targetKey, quant: null, Options.AutoPull, ct).ConfigureAwait(false);
                if (resolvedPath is null)
                    return ServerStartup.NotFoundMessage(targetKey);

                // (#454) Re-check the catalog against the key this load would actually produce.
                // The check above only saw the raw request string, so a request naming a repo id
                // or a file path ("org/Repo-GGUF", "C:/models/foo.gguf") would otherwise load a
                // model whose catalog key ("foo") is disabled. Deliberately before LoadModel, so a
                // disabled model is never parsed, let alone mapped into memory.
                var resolvedKey = ServerStartup.ModelIdFor(targetKey, resolvedPath);
                if (!Catalog.IsEnabled(resolvedKey))
                    return $"Model is disabled: {resolvedKey}";

                reloadPath = resolvedPath;
                var profile = ModelProfileStore.Resolve(targetKey) is { } rp ? ModelProfileStore.Merge(rp.Chain) : null;
                loadOptions = Options with
                {
                    Model = targetKey,
                    Quant = null,
                    ModelId = ServerStartup.ModelIdFor(targetKey, resolvedPath),
                    Device = profile?.Device ?? Options.Device,
                    GpuLayers = profile?.GpuLayers ?? Options.GpuLayers,
                };
                if (profile?.KeepAlive is { } profileKeepAlive) keepAliveOverride ??= profileKeepAlive;
                incomingBytes = SafeFileLength(resolvedPath);
            }

            // (#1023) A queued request whose client has gone must not free the active model or start a load.
            ct.ThrowIfCancellationRequested();

            ServerLog.Info($"loading {loadOptions.ModelId} from {reloadPath} (active: {(Model is null ? "none" : Options.ModelId)})");
            loadStart = System.Diagnostics.Stopwatch.GetTimestamp();
            _loading = new LoadingInfo(loadOptions.ModelId, DateTimeOffset.UtcNow, ServerLog.RequestTag);

            // (#1023) Free BEFORE loading: stash the active model, then let the budget evict with the incoming model in the active
            // slot (by default that disposes the old model now, not after the new one has been loaded beside it).
            await StopSchedulerAsync().ConfigureAwait(false);
            StashOrDropActiveLocked();
            IsReady = false;
            FreeForIncomingLocked(incomingBytes, loadOptions.Device);
            ct.ThrowIfCancellationRequested();

            var loaded = await Task.Run(() => ModelLoader(reloadPath, loadOptions), ct).ConfigureAwait(false);

            InstallLoadedModelLocked(loaded, incomingBytes, keepAliveOverride);
            Residency.EnforceBudget(EstimatedBytes);
            ServerLog.Info($"loaded {Options.ModelId} in {ServerLog.Duration(System.Diagnostics.Stopwatch.GetElapsedTime(loadStart))}");
            return null;
        }
        catch (OperationCanceledException) when (ct.IsCancellationRequested)
        {
            ServerLog.Info(loadStart != 0 ? "load abandoned: the requesting client disconnected" : "activation abandoned: the requesting client disconnected");
            IsReady = Model is not null;
            return "Request cancelled";
        }
        catch (Exception ex)
        {
            if (loadStart != 0)
                ServerLog.Info($"load failed after {ServerLog.Duration(System.Diagnostics.Stopwatch.GetElapsedTime(loadStart))}: {ex.Message}");
            IsReady = Model is not null;
            return ex.Message;
        }
        finally { _loading = null; }
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
        _gateHolder = "idle sweep";
        try
        {
            if (!IsReady || Model is null) return;
            if (!IsActiveExpired()) return;

            ServerLog.Info($"keep-alive expired: unloading {Options.ModelId}");
            await StopSchedulerAsync().ConfigureAwait(false);
            DisposeActiveLiveFieldsKeepPathAndOptions();
            IsReady = false;
        }
        finally { ReleaseGate(); }
    }

    /// <summary>Background loop calling <see cref="SweepIdleAsync"/> on <paramref name="interval"/>. Runs
    /// until <paramref name="ct"/> is cancelled (host shutdown).</summary>
    public async Task RunIdleSweepLoopAsync(TimeSpan interval, CancellationToken ct)
    {
        IdleSweepIntervalSeconds = interval.TotalSeconds;
        using var timer = new PeriodicTimer(interval);
        try
        {
            while (await timer.WaitForNextTickAsync(ct).ConfigureAwait(false))
            {
                await SweepIdleAsync().ConfigureAwait(false);

                // (#454) Re-read the mutable interval each tick so PUT /v1/settings takes effect
                // without a restart. PeriodicTimer.Period is settable; clamped to a sane floor so
                // a bad value cannot turn this into a spin loop.
                var desired = TimeSpan.FromSeconds(Math.Clamp(IdleSweepIntervalSeconds, 0.1, 3600));
                if (desired != timer.Period) timer.Period = desired;
            }
        }
        catch (OperationCanceledException) { /* shutdown */ }
    }

    /// <summary>
    /// Explicitly unloads resident models (#454) — the HTTP-reachable counterpart of the
    /// keep-alive sweep. Returns the keys actually unloaded.
    /// </summary>
    /// <param name="key">
    /// Model key to unload. Null/empty means "the active model". Ignored when
    /// <paramref name="all"/> is true.
    /// </param>
    /// <param name="all">When true, unloads the active model and every stashed one.</param>
    /// <param name="ct">Cancellation for the request-gate wait.</param>
    /// <remarks>
    /// Unloading the active model takes the request gate and therefore <b>waits behind an
    /// in-flight generation</b> rather than interrupting it. It reuses exactly the same
    /// stop-scheduler-then-dispose sequence as <see cref="SweepIdleAsync"/>, and likewise keeps
    /// <see cref="Options"/>/<see cref="LoadedModelPath"/> so a later request can lazily reload
    /// the same model. Stashed models are not in use by any in-flight request, so they are
    /// disposed without the gate — same as <see cref="ModelResidencyManager.SweepExpired"/>.
    /// </remarks>
    public async Task<IReadOnlyList<string>> UnloadAsync(string? key, bool all, CancellationToken ct)
    {
        var unloaded = new List<string>();

        if (all)
        {
            foreach (var info in Residency.Snapshot(DateTimeOffset.UtcNow))
            {
                if (Residency.TryTake(info.Key) is { } snap) { snap.Dispose(); unloaded.Add(info.Key); }
            }
        }
        else if (!string.IsNullOrWhiteSpace(key) && Residency.TryTake(key!) is { } stashed)
        {
            stashed.Dispose();
            unloaded.Add(key!);
            return unloaded;
        }

        bool targetsActive = all
            || string.IsNullOrWhiteSpace(key)
            || string.Equals(Options.ModelId, key, StringComparison.OrdinalIgnoreCase);

        if (targetsActive && Model is not null)
        {
            await AcquireGateAsync("unload", ct).ConfigureAwait(false);
            try
            {
                if (Model is not null)
                {
                    ServerLog.Info($"unloading {Options.ModelId}");
                    await StopSchedulerAsync().ConfigureAwait(false);
                    DisposeActiveLiveFieldsKeepPathAndOptions();
                    IsReady = false;
                    unloaded.Add(Options.ModelId);
                }
            }
            finally { ReleaseGate(); }
        }

        return unloaded;
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
            CudaPagedFactory = CudaPagedFactory,
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
            MtpActive = MtpActive,
            MtpStatus = MtpStatus,
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
        CudaPagedFactory = null;
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
        CudaPagedFactory = snapshot.CudaPagedFactory;
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
        MtpActive = snapshot.MtpActive;
        MtpStatus = snapshot.MtpStatus;
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
        CudaPagedFactory = loaded.CudaPagedFactory;
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
        MtpActive = loaded.MtpActive;
        MtpStatus = loaded.MtpStatus;
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
        Generator?.ClearRecurrentPrefixCache();   // releases the snapshot's KV cache + state checkpoint
        PrefixCache?.Dispose();
        PrefixTrieManager?.Dispose();
        PagedFactory?.Dispose();
        CudaPagedFactory?.Dispose();
        DraftModel?.Dispose();
        DraftGguf?.Dispose();
        Model?.Dispose();
        CurrentGguf?.Dispose();

        Config = null;
        ToolCallParser = null;
        KvCacheFactory = null;
        PagedFactory = null;
        CudaPagedFactory = null;
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
        PullManager.Dispose();
        PrefixCache?.Dispose();
        PrefixTrieManager?.Dispose();
        PagedFactory?.Dispose();
        CudaPagedFactory?.Dispose();
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

    /// <summary>Stop sequences always applied in addition to the request's (from a model profile). Null = none.</summary>
    public IReadOnlyList<string>? StopSequences { get; init; }

    /// <summary>Returns these defaults with every value the <paramref name="profile"/> sets replacing the global one.</summary>
    public SamplingDefaults OverlayProfile(ModelProfile profile) => this with
    {
        Temperature = profile.Temperature ?? Temperature,
        TopP = profile.TopP ?? TopP,
        TopK = profile.TopK ?? TopK,
        MinP = profile.MinP ?? MinP,
        RepetitionPenalty = profile.RepeatPenalty ?? RepetitionPenalty,
        MaxTokens = profile.MaxTokens ?? MaxTokens,
        Seed = profile.Seed ?? Seed,
        StopSequences = profile.Stop is { Length: > 0 } ? profile.Stop : StopSequences,
    };
}
