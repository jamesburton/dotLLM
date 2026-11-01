using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

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

    /// <summary>Creates an assembler for chunks produced by a model with <paramref name="config"/>.</summary>
    /// <param name="config">The model configuration (RoPE parameters, head dim, NoPE layers).</param>
    /// <exception cref="NotSupportedException">The model's RoPE variant cannot be re-rotated (see <see cref="RopeParameters.FromConfig"/>).</exception>
    public ChunkAssembler(ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
        _rope = RopeParameters.FromConfig(config);
    }

    /// <summary>The RoPE parameters keys are re-rotated with.</summary>
    public RopeParameters Rope => _rope;

    /// <summary>
    /// Appends <paramref name="chunks"/> to <paramref name="destination"/> in list order at running offsets. Every chunk is
    /// validated before anything is written, so a rejected call leaves the cache unchanged. Entries are not modified (keys
    /// are shifted in a copy), so the same entry may appear several times.
    /// </summary>
    /// <param name="destination">An exact host-F32 cache (<see cref="IKvCache.SupportsExactF32PrefixCopy"/>, e.g. <c>SimpleKvCache</c>).</param>
    /// <param name="chunks">Chunks to place, in order.</param>
    /// <param name="expectedFingerprint">Fingerprint every chunk must carry (ordinal comparison).</param>
    /// <exception cref="NotSupportedException">The destination stores KV lossily (quantized, F16, paged) or not as host F32 rows.</exception>
    /// <exception cref="ArgumentException">A chunk has the wrong fingerprint, is empty, has the wrong shape, or the chunks do not fit.</exception>
    public unsafe void Assemble(IKvCache destination, IReadOnlyList<ChunkKvEntry> chunks, string expectedFingerprint)
    {
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(chunks);
        ArgumentNullException.ThrowIfNull(expectedFingerprint);
        if (!destination.SupportsExactF32PrefixCopy || destination is not IPerLayerKvCache perLayer)
            throw new NotSupportedException(
                $"Chunk assembly needs an exact host-F32 KV-cache; {destination.GetType().Name} stores KV lossily or off-host.");

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
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = offset + i;

            for (int l = 0; l < layers; l++)
            {
                int stride = perLayer.KvStrideOf(l);
                float[] keys = chunk.KeysPerLayer[l];
                if (offset != 0 && !_config.IsNoRopeLayer(l))
                {
                    keys = (float[])keys.Clone();
                    RopeShift.Apply(keys, headDim, offset, in _rope);
                }

                fixed (float* kPtr = keys)
                fixed (float* vPtr = chunk.ValuesPerLayer[l])
                {
                    var kRef = new TensorRef(n, stride, DType.Float32, -1, (nint)kPtr);
                    var vRef = new TensorRef(n, stride, DType.Float32, -1, (nint)vPtr);
                    destination.Update(kRef, vRef, positions, l);
                }
            }
            offset += n;
        }
    }
}
