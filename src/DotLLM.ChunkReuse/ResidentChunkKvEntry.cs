using DotLLM.Core.Attention;

namespace DotLLM.ChunkReuse;

/// <summary>
/// A chunk's K/V kept where it was encoded: the backend cache of the isolated prefill (rows <c>0..n-1</c>, keys post-RoPE at those
/// positions). The device-resident counterpart of <see cref="ChunkKvEntry"/>: assembly copies its rows cache to cache
/// (<see cref="Backends.IChunkBackend.CopyRows"/>; on Vulkan device to device, no host hop) and then re-rotates the keys. Owns the cache;
/// after <see cref="Dispose"/> the entry refuses use (<see cref="ObjectDisposedException"/>) rather than reading a freed device buffer.
/// </summary>
public sealed class ResidentChunkKvEntry : IDisposable
{
    private readonly int[] _tokenIds;
    private readonly IKvCache _cache;

    /// <summary>Wraps <paramref name="cache"/>, which holds exactly the chunk's rows at <c>0..n-1</c>.</summary>
    /// <param name="modelFingerprint">Identity of the model the rows come from.</param>
    /// <param name="tokenIds">The chunk's token ids (copied).</param>
    /// <param name="cache">The encode cache; owned from here on.</param>
    /// <param name="backend">Tag of the backend that owns <paramref name="cache"/>.</param>
    public ResidentChunkKvEntry(string modelFingerprint, ReadOnlySpan<int> tokenIds, IKvCache cache, string backend)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelFingerprint);
        ArgumentNullException.ThrowIfNull(cache);
        ArgumentException.ThrowIfNullOrEmpty(backend);
        if (tokenIds.IsEmpty) throw new ArgumentException("A chunk must contain at least one token.", nameof(tokenIds));
        if (cache.CurrentLength != tokenIds.Length)
            throw new ArgumentException($"The cache holds {cache.CurrentLength} rows for a {tokenIds.Length}-token chunk.", nameof(cache));
        ModelFingerprint = modelFingerprint;
        _tokenIds = tokenIds.ToArray();
        _cache = cache;
        Backend = backend;
    }

    /// <summary>Identity of the model the rows come from.</summary>
    public string ModelFingerprint { get; }

    /// <summary>The chunk's token ids.</summary>
    public ReadOnlySpan<int> TokenIds => _tokenIds;

    /// <summary>Number of tokens (rows).</summary>
    public int Length => _tokenIds.Length;

    /// <summary>The backend cache holding the rows at <c>0..Length-1</c>.</summary>
    /// <exception cref="ObjectDisposedException">The entry was disposed.</exception>
    public IKvCache Cache
    {
        get
        {
            ObjectDisposedException.ThrowIf(IsDisposed, this);
            return _cache;
        }
    }

    /// <summary>Tag of the backend that owns <see cref="Cache"/>.</summary>
    public string Backend { get; }

    /// <summary>True after <see cref="Dispose"/>; the assembler refuses a disposed entry before writing anything.</summary>
    public bool IsDisposed { get; private set; }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (IsDisposed) return;
        IsDisposed = true;
        _cache.Dispose();
    }
}
