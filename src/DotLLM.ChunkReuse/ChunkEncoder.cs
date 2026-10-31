using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;

namespace DotLLM.ChunkReuse;

/// <summary>
/// Encodes a token chunk into a <see cref="ChunkKvEntry"/>: an isolated prefill at positions <c>0..n-1</c> on a fresh F32
/// <see cref="SimpleKvCache"/>, then a copy of every layer's K/V rows.
/// </summary>
public sealed class ChunkEncoder
{
    /// <summary>Creates an encoder that stamps entries with <paramref name="modelFingerprint"/>.</summary>
    /// <param name="modelFingerprint">Identity of the model the entries come from (<see cref="ChunkFingerprint.Compute"/>).</param>
    public ChunkEncoder(string modelFingerprint)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelFingerprint);
        ModelFingerprint = modelFingerprint;
    }

    /// <summary>The fingerprint written into every entry.</summary>
    public string ModelFingerprint { get; }

    /// <summary>Runs <paramref name="tokenIds"/> through <paramref name="model"/> in isolation and captures its per-layer K/V.</summary>
    /// <param name="model">The model; must write host F32 rows into a <see cref="SimpleKvCache"/> (the CPU <c>TransformerModel</c>).</param>
    /// <param name="tokenIds">The chunk; not empty.</param>
    /// <returns>The entry, keys post-RoPE at <c>0..n-1</c>.</returns>
    /// <exception cref="ArgumentException">The chunk is empty or longer than the model context.</exception>
    public ChunkKvEntry Encode(IModel model, ReadOnlySpan<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (tokenIds.IsEmpty)
            throw new ArgumentException("A chunk must contain at least one token.", nameof(tokenIds));
        int n = tokenIds.Length;
        if (n > model.Config.MaxSequenceLength)
            throw new ArgumentException($"Chunk of {n} tokens exceeds the model context {model.Config.MaxSequenceLength}.", nameof(tokenIds));

        var geometry = KvGeometry.FromConfig(model.Config, compactSlidingWindows: false);
        using var cache = new SimpleKvCache(geometry, n);
        int[] positions = new int[n];
        for (int i = 0; i < n; i++) positions[i] = i;

        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, cache, lastTokenLogitsOnly: true))
        {
            // Only the cache side effect is wanted.
        }

        if (cache.CurrentLength != n)
            throw new InvalidOperationException($"The model wrote {cache.CurrentLength} cache rows for a {n}-token chunk.");

        int layers = cache.NumLayers;
        var keys = new float[layers][];
        var values = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            keys[l] = cache.KeysSpan(l).ToArray();
            values[l] = cache.ValuesSpan(l).ToArray();
        }
        return new ChunkKvEntry(ModelFingerprint, tokenIds.ToArray(), keys, values);
    }
}
