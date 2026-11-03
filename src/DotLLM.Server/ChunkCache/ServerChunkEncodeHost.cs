using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.Backends;
using DotLLM.ChunkReuse.CodeCorpus;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// The encode worker's view of a running server (<see cref="IChunkEncodeHost"/>): the model is entered only through
/// <see cref="ServerState.TryAcquireIdleModel"/> (the request gate plus the scheduler's model lease, only when idle), and
/// chunks are encoded exactly as the <c>chunkcache</c> CLI encodes them: <c>ChunkTokenization.Encode</c> (ruling 21), an
/// isolated forward at positions <c>0..n-1</c> on the served model (<see cref="ChunkEncoder"/>), F32 rows, the entry stamped
/// with the served model's fingerprint and backend tag, so a CLI reader and this server's own lookups agree.
/// </summary>
/// <param name="state">The server.</param>
internal sealed class ServerChunkEncodeHost(ServerState state) : IChunkEncodeHost
{
    /// <inheritdoc/>
    public IDisposable? TryEnterIdle() => state.TryAcquireIdleModel();

    /// <inheritdoc/>
    public long ActivityStamp() => state.ActivityStamp();

    /// <inheritdoc/>
    public bool IsBusy() => state.IsBusyForBackgroundWork();

    /// <inheritdoc/>
    public ChunkKvEntry? Encode(ChunkEncodeItem item)
    {
        ArgumentNullException.ThrowIfNull(item);
        var identity = state.ChunkCacheRuntime.IdentityFor(state); // under the lease: the model cannot change meanwhile
        if (!identity.Supported || !string.Equals(identity.Backend, ChunkBackendTags.Cpu, StringComparison.Ordinal))
            return null; // only CPU entries are written until the Vulkan schema question is decided (plan Task 15)
        if (state.Model is not { } model || state.Tokenizer is not { } tokenizer)
            return null;
        int[] ids = ChunkTokenization.Encode(tokenizer, item.Text);
        if (ids.Length == 0 || ids.Length > ChunkStoreService.MaxEncodeTokens || ids.Length > model.Config.MaxSequenceLength)
            return null;
        if (identity.KeyOf(ids, item.Namespace) != item.Key)
            return null; // another model (or tokenizer) is served now: this item's key is not ours to fill
        return new ChunkEncoder(identity.Fingerprint).Encode(model, new CpuChunkBackend(model.Config), ids);
    }
}
