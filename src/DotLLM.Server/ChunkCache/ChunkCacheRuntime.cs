using DotLLM.ChunkReuse.Store;
using DotLLM.Models.Gguf;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// Per-server chunk-cache state: the (read-only) chunk store and the identity of the loaded model, computed once per model
/// (single flight; a failure is cached too). Thread-safe. Holds no request data.
/// </summary>
internal sealed class ChunkCacheRuntime
{
    private readonly object _storeGate = new();
    private readonly object _identityGate = new();
    private IChunkStore? _store;
    private string? _storeRootOpened;
    private object? _identityKey;
    private ServerModelIdentity? _identity;

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
                identity = IdentityFactory(gguf, state.Config!, backend);
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
}
