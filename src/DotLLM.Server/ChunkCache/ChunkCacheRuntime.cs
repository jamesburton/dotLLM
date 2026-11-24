using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Identity;
using DotLLM.Models.Gguf;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// Per-server chunk-cache state: the (read-only) chunk store and the identity of the loaded model, computed once per model
/// (single flight; a failure is cached too). Thread-safe. Holds no request data.
/// </summary>
internal sealed class ChunkCacheRuntime : IDisposable
{
    private ChunkStoreService? _writer;

    /// <summary>The store's sole writer (<c>--chunk-store-write</c>), once <see cref="StartWriter"/> ran; else null (read-only).</summary>
    internal ChunkStoreService? Writer => Volatile.Read(ref _writer);

    /// <summary>
    /// Starts the sole writer for <paramref name="state"/>'s options at server startup (once; later calls return the running
    /// one). Null when <c>--chunk-store-write</c> is off.
    /// </summary>
    /// <param name="state">The server.</param>
    /// <param name="host">The model the worker encodes on; null = the server's own (<see cref="ServerChunkEncodeHost"/>).</param>
    /// <returns>The writer or null.</returns>
    /// <exception cref="InvalidServerOptionException">Another server owns the store, or it cannot be created.</exception>
    internal ChunkStoreService? StartWriter(ServerState state, IChunkEncodeHost? host = null)
    {
        var options = state.Options.ChunkCache;
        if (!options.Enabled || !options.StoreWrite)
            return null;
        lock (_storeGate)
        {
            return _writer ??= ChunkStoreService.Start(options, host ?? new ServerChunkEncodeHost(state));
        }
    }

    private ChunkStoreService? _retired;

    /// <summary>Whether a stopped writer's worker is still inside an encode (the model must not be freed under it).</summary>
    internal bool WriterStillRunning => Volatile.Read(ref _retired) is { WorkerStopped: false };

    /// <summary>Stops the writer (worker and ownership; see <see cref="ChunkStoreService.Dispose"/>).</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref _writer, null) is { } writer)
        {
            writer.Dispose();
            Volatile.Write(ref _retired, writer);
        }
    }

    private readonly object _storeGate = new();
    private readonly object _identityGate = new();
    private IChunkStore? _store;
    private string? _storeRootOpened;
    private object? _identityKey;
    private ServerModelIdentity? _identity;

    /// <summary>Slots for concurrent <c>explain</c> calls (<see cref="Endpoints.ChunkCacheEndpoint.MaxConcurrentExplains"/>).</summary>
    internal SemaphoreSlim ExplainGate { get; } = new(Endpoints.ChunkCacheEndpoint.MaxConcurrentExplains, Endpoints.ChunkCacheEndpoint.MaxConcurrentExplains);

    /// <summary>
    /// How long an <c>explain</c> body may take to arrive (<see cref="Endpoints.ChunkCacheEndpoint.ExplainBodyReadTimeoutSeconds"/>);
    /// a slower body is <c>408</c>. Settable for tests.
    /// </summary>
    internal TimeSpan ExplainBodyReadTimeout
    {
        get => _explainBodyReadTimeout ?? Endpoints.ChunkCacheEndpoint.BodyReadTimeout; // one deadline for every POST body unless a test set this one
        set => _explainBodyReadTimeout = value;
    }

    private TimeSpan? _explainBodyReadTimeout;

    private int _explainBodyReadsInFlight;
    private int _inferenceCallerResolutions;

    /// <summary><c>explain</c> bodies being read right now (test observability: proves the server sees a stalled body).</summary>
    internal int ExplainBodyReadsInFlight => Volatile.Read(ref _explainBodyReadsInFlight);

    /// <summary>
    /// Inference requests whose chunk-cache caller was resolved (the key looked up and hashed). Only annotated requests
    /// pay that; the counter lets a test prove an unannotated one does not.
    /// </summary>
    internal int InferenceCallerResolutions => Volatile.Read(ref _inferenceCallerResolutions);

    internal void EnterExplainBodyRead() => Interlocked.Increment(ref _explainBodyReadsInFlight);

    internal void ExitExplainBodyRead() => Interlocked.Decrement(ref _explainBodyReadsInFlight);

    internal void CountInferenceCallerResolution() => Interlocked.Increment(ref _inferenceCallerResolutions);
    /// <summary>Slots for concurrent <c>prewarm</c> / <c>evict</c> calls (<see cref="Endpoints.ChunkCacheEndpoint.MaxConcurrentWrites"/>).</summary>
    internal SemaphoreSlim WriteGate { get; } = new(Endpoints.ChunkCacheEndpoint.MaxConcurrentWrites, Endpoints.ChunkCacheEndpoint.MaxConcurrentWrites);

    private readonly HashSet<string> _writeNamespaces = new(StringComparer.Ordinal);

    /// <summary>Takes the namespace's single write slot (one prewarm / evict call per namespace at a time); false when taken.</summary>
    /// <param name="ns">The caller's namespace.</param>
    /// <returns>Whether the slot was free.</returns>
    internal bool TryEnterWriteSlot(string ns)
    {
        lock (_writeNamespaces)
            return _writeNamespaces.Add(ns);
    }

    /// <summary>Releases the namespace's write slot.</summary>
    /// <param name="ns">The caller's namespace.</param>
    internal void LeaveWriteSlot(string ns)
    {
        lock (_writeNamespaces)
            _writeNamespaces.Remove(ns);
    }

    /// <summary>A store injected by the host or a test; wins over <c>--chunk-store</c>.</summary>
    internal IChunkStore? StoreOverride { get; set; }

    /// <summary>An identity injected by a test; wins over the computed one.</summary>
    internal ServerModelIdentity? IdentityOverride { get; set; }

    /// <summary>How an identity is computed from a loaded GGUF (a seam for tests); <see cref="ServerModelIdentity.FromGguf"/>.</summary>
    internal Func<GgufFile, DotLLM.Core.Models.ModelConfig, string, ServerModelIdentity> IdentityFactory { get; set; } = ServerModelIdentity.FromGguf;

    /// <summary>
    /// The store to look chunks up in. <paramref name="unavailable"/> is true when a store is configured but cannot be used
    /// (its root does not exist or cannot be opened): every chunk is then a miss with <c>chunk_store_unavailable</c>.
    /// Null with <paramref name="unavailable"/> false: no store configured (every chunk is a miss, no warning).
    /// </summary>
    /// <param name="options">The server's chunk-cache options.</param>
    /// <param name="unavailable">Whether the configured store is unusable.</param>
    /// <returns>The store, or null.</returns>
    public IChunkStore? Store(ChunkCacheOptions options, out bool unavailable)
    {
        unavailable = false;
        if (StoreOverride is { } injected)
            return injected;
        string? root = options.StoreRoot;
        if (root is null)
            return null;
        lock (_storeGate)
        {
            try
            {
                // Never created here: a missing root is a deployment error to report, also if it disappears later.
                if (!Directory.Exists(root))
                {
                    unavailable = true;
                    return null;
                }
                if (_store is null || !string.Equals(_storeRootOpened, root, StringComparison.Ordinal))
                {
                    _store = FileChunkStore.OpenExisting(root);
                    _storeRootOpened = root;
                }
                return _store;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
            {
                unavailable = true;
                return null;
            }
        }
    }

    /// <summary>
    /// The identity of the model <paramref name="state"/> serves, computed once per loaded model under a lock (concurrent
    /// first requests hash the model once). Never throws: any failure gives an unsupported identity, which is cached for
    /// that model too.
    /// </summary>
    /// <param name="state">The server state (its model, configuration, GGUF and device).</param>
    /// <returns>The identity; unsupported when chunk reuse cannot apply to the model or backend.</returns>
    public ServerModelIdentity IdentityFor(ServerState state)
    {
        if (IdentityOverride is { } injected)
            return injected;
        if (UnsupportedReason(state) is { } reason)
            return ServerModelIdentity.Unsupported(reason);

        GgufFile gguf = state.CurrentGguf!;
        string backend = ServerModelIdentity.BackendTagFor(state.Options.ResolvedDevice ?? state.Options.Device)!;
        lock (_identityGate)
        {
            if (_identity is not null && ReferenceEquals(_identityKey, gguf)
                && (!_identity.Supported || string.Equals(_identity.Backend, backend, StringComparison.Ordinal)))
                return _identity;

            ServerModelIdentity identity;
            try
            {
                // The overlay and ternary adapters are applied to the loaded model (the encoder runs on it), so their content is
                // part of its identity: a separate key space from the base model's and from any other overlay or adapter set (#1915).
                identity = IdentityFactory(gguf, state.Config!, backend).WithModifiers(ServerModifiers(state));
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                // E.g. an unreadable or malformed GGUF, a geometry the fingerprint cannot describe: never a 500.
                ServerLog.Info($"chunk_cache: model identity unavailable ({ex.GetType().Name}); annotations are ignored for this model");
                identity = ServerModelIdentity.Unsupported("model identity unavailable");
            }
            _identityKey = gguf;
            _identity = identity;
            return identity;
        }
    }

    /// <summary>
    /// The identity a request is looked up under: <see cref="IdentityFor(ServerState)"/>, plus the request's LoRA adapter folded in
    /// when it names one (#1915). A LoRA request therefore never shares keys with base-model entries (no wrong hit), and the same
    /// adapter content always lands on the same keys. An unsupported base identity stays unsupported.
    /// </summary>
    /// <param name="state">The server state.</param>
    /// <param name="requestAdapter">The request's LoRA adapter, or null.</param>
    /// <returns>The identity; never throws (a digest failure is an unsupported identity).</returns>
    public ServerModelIdentity IdentityFor(ServerState state, DotLLM.Core.Lora.ILoraAdapter? requestAdapter)
    {
        var identity = IdentityFor(state);
        if (requestAdapter is null || !identity.Supported)
            return identity;
        try
        {
            return identity.WithModifiers([(ModifierSetDigest.KindLora, "", ModifierSetDigest.OfLora(requestAdapter))]);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            ServerLog.Info($"chunk_cache: LoRA identity unavailable ({ex.GetType().Name}); annotations are ignored for this request");
            return ServerModelIdentity.Unsupported("LoRA adapter identity unavailable");
        }
    }

    /// <summary>
    /// How long the store-wide figures of <c>GET /v1/chunk-cache</c> (a walk over every entry file) are reused. The route
    /// has no slot and a keyless local caller may call it freely, so the walk must not run per request.
    /// </summary>
    internal TimeSpan StoreFiguresTtl { get; set; } = TimeSpan.FromSeconds(5);

    private readonly Lock _figuresGate = new();
    private (long Entries, long Bytes) _figures;
    private long _figuresAtMs = long.MinValue;
    private int _figureWalks;

    /// <summary>Directory walks done for the store-wide figures (test observability).</summary>
    internal int StoreFigureWalks => Volatile.Read(ref _figureWalks);

    /// <summary>The store-wide figures, walked at most once per <see cref="StoreFiguresTtl"/> (one walk at a time).</summary>
    /// <param name="walk">Counts the store's entry files and bytes.</param>
    /// <returns>Entries and bytes, at most <see cref="StoreFiguresTtl"/> old.</returns>
    internal (long Entries, long Bytes) StoreFigures(Func<(long Entries, long Bytes)> walk)
    {
        lock (_figuresGate)
        {
            long now = Environment.TickCount64;
            if (_figuresAtMs != long.MinValue && StoreFiguresTtl > TimeSpan.Zero && now - _figuresAtMs < StoreFiguresTtl.TotalMilliseconds)
                return _figures;
            _figures = walk();
            _figuresAtMs = now;
            Interlocked.Increment(ref _figureWalks);
            return _figures;
        }
    }

    /// <summary>
    /// Why chunk reuse cannot apply to the served model (spec §5.5: recurrent / hybrid models, a backend without assembly),
    /// or null.
    /// </summary>
    internal static string? UnsupportedReason(ServerState state)
    {
        if (state.Config is null || state.Model is null)
            return "no model loaded";
        if (state.DiffusionGenerator is not null)
            return "diffusion model";
        if (ServerStartup.IsHybridOrRecurrentCpuModel(state.Config, state.Model) || state.Model.RequiresPerSequenceState)
            return "recurrent or hybrid model";
        if (ServerModelIdentity.BackendTagFor(state.Options.ResolvedDevice ?? state.Options.Device) is null)
            return "backend without chunk assembly";
        if (state.CurrentGguf is null)
            return "no GGUF identity";
        return null;
    }

    /// <summary>
    /// What changed the loaded weights without changing the base GGUF (<c>--tensor-overlay</c>, each <c>--ternary-adapter</c> package), as
    /// (kind, id, content digest) from <see cref="ServerState.WeightModifiers"/>: digests taken at load time from the loaded bytes, so
    /// nothing is read from disk here and a file replaced since load cannot change the identity (#1915). Ids are empty: content decides.
    /// </summary>
    /// <param name="state">The server state.</param>
    /// <returns>The modifiers (empty for a base model).</returns>
    /// <exception cref="InvalidOperationException">The options name an overlay or adapter but no digests were computed at load.</exception>
    internal static IReadOnlyList<(string Kind, string Id, string ContentDigest)> ServerModifiers(ServerState state)
    {
        if (state.WeightModifiers is { } computed)
            return computed;
        if (state.Options.TensorOverlayPath is not null || state.Options.TernaryAdapterPaths is { Count: > 0 })
            throw new InvalidOperationException("The overlay and adapter digests were not computed at load.");
        return [];
    }
}
