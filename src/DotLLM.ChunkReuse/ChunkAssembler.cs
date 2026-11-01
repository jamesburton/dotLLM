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
        _backend = backend;
    }

    /// <summary>The RoPE parameters keys are re-rotated with.</summary>
    public RopeParameters Rope => _rope;

    /// <summary>The backend chunks are placed through.</summary>
    public IChunkBackend Backend => _backend;

    /// <summary>
    /// Appends <paramref name="chunks"/> to <paramref name="destination"/> in list order at running offsets: each chunk's rows are
    /// written at its offset and its keys are then re-rotated in the cache by that offset. The destination is vetted by the backend
    /// and every chunk is validated before anything is written, so a rejected call leaves the cache unchanged. Entries are not
    /// modified, so the same entry may appear several times.
    /// </summary>
    /// <param name="destination">A cache the backend accepts (<see cref="IChunkBackend.RequireCache"/>; CPU: an exact host-F32 <c>SimpleKvCache</c>).</param>
    /// <param name="chunks">Chunks to place, in order.</param>
    /// <param name="expectedFingerprint">Fingerprint every chunk must carry (ordinal comparison).</param>
    /// <exception cref="NotSupportedException">The backend cannot write the destination exactly, or the layer map differs.</exception>
    /// <exception cref="ArgumentException">A chunk has the wrong fingerprint or backend, is empty, has the wrong shape, or the chunks do not fit.</exception>
    public void Assemble(IKvCache destination, IReadOnlyList<ChunkKvEntry> chunks, string expectedFingerprint)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(expectedFingerprint);
        _backend.RequireCache(destination);
        if (destination is not IPerLayerKvCache perLayer)
            throw new NotSupportedException($"Chunk assembly needs per-layer KV strides; {destination.GetType().Name} has none.");

        int layers = perLayer.LayerCount;
        if (layers != _config.NumLayers)
            throw new NotSupportedException(
                $"The destination holds {layers} KV layers for a {_config.NumLayers}-layer model (shared-KV layer map); not supported.");
        int headDim = _config.HeadDim;
        int start = destination.CurrentLength;
        long total = start;
        for (int c = 0; c < chunks.Count; c++)
        {
            ChunkKvEntry chunk = chunks[c] ?? throw new ArgumentException($"Chunk {c} is null.", nameof(chunks));
            if (!string.Equals(chunk.ModelFingerprint, expectedFingerprint, StringComparison.Ordinal))
                throw new ArgumentException(
                    $"Chunk {c} has model fingerprint '{chunk.ModelFingerprint}', expected '{expectedFingerprint}'.", nameof(chunks));
            if (!string.Equals(chunk.Backend, _backend.Tag, StringComparison.Ordinal))
                throw new ArgumentException(
                    $"Chunk {c} was encoded by backend '{chunk.Backend}', but this assembler places through backend '{_backend.Tag}'; "
                    + "convert it explicitly or treat it as a miss.", nameof(chunks));
            int n = chunk.TokenIds?.Length ?? 0;
            if (n == 0)
                throw new ArgumentException($"Chunk {c} is empty.", nameof(chunks));
            if (chunk.KeysPerLayer is null || chunk.ValuesPerLayer is null
                || chunk.KeysPerLayer.Length != layers || chunk.ValuesPerLayer.Length != layers)
                throw new ArgumentException($"Chunk {c} does not carry K/V for the destination's {layers} layers.", nameof(chunks));
            for (int l = 0; l < layers; l++)
            {
                int stride = perLayer.KvStrideOf(l);
                if (stride % headDim != 0)
                    throw new ArgumentException($"Layer {l} KV stride {stride} is not a multiple of head dim {headDim}.", nameof(destination));
                long expected = (long)n * stride;
                if (chunk.KeysPerLayer[l]?.Length != expected || chunk.ValuesPerLayer[l]?.Length != expected)
                    throw new ArgumentException(
                        $"Chunk {c} layer {l}: expected {expected} K and V floats ({n} tokens x stride {stride}).", nameof(chunks));
            }
            total += n;
        }
        if (total > destination.MaxLength)
            throw new ArgumentException(
                $"Chunks need positions up to {total - 1} but the destination holds {destination.MaxLength}.", nameof(chunks));

        int offset = start;
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
}
