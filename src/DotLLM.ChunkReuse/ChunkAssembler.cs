using DotLLM.ChunkReuse.Backends;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.ChunkReuse;

/// <summary>
/// Writes cached chunks into a KV-cache back to back, starting at the cache's current length, re-rotating each chunk's keys
/// from its encoded positions <c>0..n-1</c> to its new offset (<see cref="RopeShift"/>). Values are copied unchanged. The
/// result is the E1 "naive concatenation" cache: per-chunk KV with correct positions but no cross-chunk attention.
/// </summary>
public sealed class ChunkAssembler
{
    private readonly ModelConfig _config;
    private readonly RopeParameters _rope;
    private readonly IChunkBackend _backend;

    /// <summary>Creates a CPU assembler (<see cref="CpuChunkBackend"/>) for chunks produced by a model with <paramref name="config"/>.</summary>
    /// <param name="config">The model configuration (RoPE parameters, head dim, NoPE layers).</param>
    /// <exception cref="NotSupportedException">The model's RoPE variant cannot be re-rotated (see <see cref="RopeParameters.FromConfig"/>).</exception>
    public ChunkAssembler(ModelConfig config)
        : this(config, new CpuChunkBackend(config ?? throw new ArgumentNullException(nameof(config))))
    {
    }

    /// <summary>Creates an assembler that places chunks through <paramref name="backend"/>.</summary>
    /// <param name="config">The model configuration (RoPE parameters, head dim, NoPE layers).</param>
    /// <param name="backend">Where the destination cache lives; only entries carrying its <see cref="IChunkBackend.Tag"/> are placed.</param>
    /// <exception cref="NotSupportedException">The model's RoPE variant cannot be re-rotated (see <see cref="RopeParameters.FromConfig"/>).</exception>
    public ChunkAssembler(ModelConfig config, IChunkBackend backend)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(backend);
        _config = config;
        _rope = RopeParameters.FromConfig(config);
        // A backend bound to other RoPE parameters would refuse the first shift after a chunk was already written: refuse up front.
        if (backend.Rope is RopeParameters bound && bound != _rope)
            throw new ArgumentException(
                $"The backend shifts with RoPE parameters {bound}, but this assembler's config gives {_rope}; they must be the same model.",
                nameof(backend));
        _backend = backend;
    }

    /// <summary>The RoPE parameters keys are re-rotated with.</summary>
    public RopeParameters Rope => _rope;

    /// <summary>The backend chunks are placed through.</summary>
    public IChunkBackend Backend => _backend;

    /// <summary>
    /// Appends <paramref name="chunks"/> to <paramref name="destination"/> in list order at running offsets: each chunk's rows are
    /// written at its offset and its keys are then re-rotated in the cache by that offset. The destination is vetted by the backend
    /// and every chunk is validated before anything is written, so a rejected call leaves the cache unchanged; a failure while writing
    /// (device or allocation error) rolls the cache back to its start length. Entries are not modified, so the same entry may appear
    /// several times.
    /// </summary>
    /// <param name="destination">A cache the backend accepts (<see cref="IChunkBackend.RequireCache"/>; CPU: an exact host-F32 <c>SimpleKvCache</c>).</param>
    /// <param name="chunks">Chunks to place, in order.</param>
    /// <param name="expectedFingerprint">Fingerprint every chunk must carry (ordinal comparison).</param>
    /// <exception cref="NotSupportedException">The backend cannot write the destination exactly, or the layer map differs.</exception>
    /// <exception cref="ArgumentException">A chunk has the wrong fingerprint or backend, is empty, has the wrong shape, or the chunks do not fit.</exception>
    public void Assemble(IKvCache destination, IReadOnlyList<ChunkKvEntry> chunks, string expectedFingerprint)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        (IPerLayerKvCache perLayer, int start) = Validate(destination, chunks.Count, expectedFingerprint, c =>
        {
            ChunkKvEntry chunk = chunks[c] ?? throw new ArgumentException($"Chunk {c} is null.", nameof(chunks));
            return (chunk.ModelFingerprint, chunk.Backend, chunk.TokenIds?.Length ?? 0);
        });
        int layers = perLayer.LayerCount;
        for (int c = 0; c < chunks.Count; c++)
        {
            ChunkKvEntry chunk = chunks[c];
            int n = chunk.TokenIds.Length;
            if (chunk.KeysPerLayer is null || chunk.ValuesPerLayer is null
                || chunk.KeysPerLayer.Length != layers || chunk.ValuesPerLayer.Length != layers)
                throw new ArgumentException($"Chunk {c} does not carry K/V for the destination's {layers} layers.", nameof(chunks));
            for (int l = 0; l < layers; l++)
            {
                long expected = (long)n * perLayer.KvStrideOf(l);
                if (chunk.KeysPerLayer[l]?.Length != expected || chunk.ValuesPerLayer[l]?.Length != expected)
                    throw new ArgumentException(
                        $"Chunk {c} layer {l}: expected {expected} K and V floats ({n} tokens x stride {perLayer.KvStrideOf(l)}).", nameof(chunks));
            }
        }

        int offset = start;
        try
        {
            foreach (ChunkKvEntry chunk in chunks)
            {
                int n = chunk.TokenIds.Length;
                for (int l = 0; l < layers; l++)
                    _backend.WriteRows(destination, l, offset, chunk.KeysPerLayer[l], chunk.ValuesPerLayer[l]);
                // Keys were encoded at 0..n-1; rotate the placed rows by the offset (NoPE layers are skipped by the backend).
                if (offset != 0)
                    _backend.ShiftKeys(destination, offset, n, offset, _rope);
                offset += n;
            }
        }
        catch
        {
            // Row writes advance the cache length as they go: never leave a partly assembled, partly unshifted cache behind.
            destination.Rollback(start);
            throw;
        }
    }

    /// <summary>
    /// As <see cref="Assemble(IKvCache, IReadOnlyList{ChunkKvEntry}, string)"/>, for chunks still held in their encode caches: the rows
    /// are copied cache to cache (<see cref="IChunkBackend.CopyRows"/>; on Vulkan device to device), then the keys are re-rotated by the
    /// offset. Fingerprint, backend tag, capacity, disposal and the source caches' type, layers, strides and lengths are validated before
    /// anything is written. A source cache on another device cannot be detected up front (no public device identity on
    /// <c>VulkanKvCache</c>); it, and any device error, fails during the copy and the destination is rolled back to its start length.
    /// </summary>
    /// <param name="destination">A cache the backend accepts.</param>
    /// <param name="chunks">Resident chunks to place, in order; they are not modified.</param>
    /// <param name="expectedFingerprint">Fingerprint every chunk must carry (ordinal comparison).</param>
    /// <exception cref="NotSupportedException">The backend cannot write the destination or read a source exactly, or the layer map differs.</exception>
    /// <exception cref="ArgumentException">A chunk has the wrong fingerprint or backend, the wrong shape, or the chunks do not fit.</exception>
    public void Assemble(IKvCache destination, IReadOnlyList<ResidentChunkKvEntry> chunks, string expectedFingerprint)
    {
        ArgumentNullException.ThrowIfNull(chunks);
        (IPerLayerKvCache perLayer, int start) = Validate(destination, chunks.Count, expectedFingerprint, c =>
        {
            ResidentChunkKvEntry chunk = chunks[c] ?? throw new ArgumentException($"Chunk {c} is null.", nameof(chunks));
            ObjectDisposedException.ThrowIf(chunk.IsDisposed, chunk);
            return (chunk.ModelFingerprint, chunk.Backend, chunk.Length);
        });
        int layers = perLayer.LayerCount;
        for (int c = 0; c < chunks.Count; c++)
        {
            ResidentChunkKvEntry chunk = chunks[c];
            if (ReferenceEquals(chunk.Cache, destination))
                throw new ArgumentException($"Chunk {c}'s own cache is the destination; assemble into a separate cache.", nameof(destination));
            _backend.RequireCache(chunk.Cache);
            if (chunk.Cache is not IPerLayerKvCache src || src.LayerCount != layers)
                throw new ArgumentException($"Chunk {c}'s cache does not hold the destination's {layers} layers.", nameof(chunks));
            for (int l = 0; l < layers; l++)
            {
                if (src.KvStrideOf(l) != perLayer.KvStrideOf(l))
                    throw new ArgumentException(
                        $"Chunk {c} layer {l}: stride {src.KvStrideOf(l)}, destination {perLayer.KvStrideOf(l)}.", nameof(chunks));
            }
            if (chunk.Cache.CurrentLength < chunk.Length)
                throw new ArgumentException($"Chunk {c}'s cache holds {chunk.Cache.CurrentLength} rows for {chunk.Length} tokens.", nameof(chunks));
        }

        int offset = start;
        try
        {
            foreach (ResidentChunkKvEntry chunk in chunks)
            {
                int n = chunk.Length;
                for (int l = 0; l < layers; l++)
                    _backend.CopyRows(chunk.Cache, l, 0, destination, offset, n);
                if (offset != 0)
                    _backend.ShiftKeys(destination, offset, n, offset, _rope);
                offset += n;
            }
        }
        catch
        {
            destination.Rollback(start);
            throw;
        }
    }

    // Checks shared by both forms: the backend vets the destination; the layer map and strides; per chunk the fingerprint, the backend tag
    // (a mismatch is a miss, never a silent cross-backend placement) and non-emptiness; the total capacity.
    private (IPerLayerKvCache PerLayer, int Start) Validate(IKvCache destination, int count, string expectedFingerprint,
        Func<int, (string Fingerprint, string Backend, int Tokens)> describe)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(expectedFingerprint);
        _backend.RequireCache(destination);
        if (destination is not IPerLayerKvCache perLayer)
            throw new NotSupportedException($"Chunk assembly needs per-layer KV strides; {destination.GetType().Name} has none.");

        int layers = perLayer.LayerCount;
        if (layers != _config.NumLayers)
            throw new NotSupportedException(
                $"The destination holds {layers} KV layers for a {_config.NumLayers}-layer model (shared-KV layer map); not supported.");
        int headDim = _config.HeadDim;
        for (int l = 0; l < layers; l++)
        {
            if (perLayer.KvStrideOf(l) % headDim != 0)
                throw new ArgumentException($"Layer {l} KV stride {perLayer.KvStrideOf(l)} is not a multiple of head dim {headDim}.", nameof(destination));
        }
        int start = destination.CurrentLength;
        long total = start;
        for (int c = 0; c < count; c++)
        {
            (string fingerprint, string backend, int n) = describe(c);
            if (!string.Equals(fingerprint, expectedFingerprint, StringComparison.Ordinal))
                throw new ArgumentException($"Chunk {c} has model fingerprint '{fingerprint}', expected '{expectedFingerprint}'.", "chunks");
            if (!string.Equals(backend, _backend.Tag, StringComparison.Ordinal))
                throw new ArgumentException(
                    $"Chunk {c} was encoded by backend '{backend}', but this assembler places through backend '{_backend.Tag}'; "
                    + "convert it explicitly or treat it as a miss.", "chunks");
            if (n == 0)
                throw new ArgumentException($"Chunk {c} is empty.", "chunks");
            total += n;
        }
        if (total > destination.MaxLength)
            throw new ArgumentException($"Chunks need positions up to {total - 1} but the destination holds {destination.MaxLength}.", "chunks");
        return (perLayer, start);
    }
}
