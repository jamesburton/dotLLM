using DotLLM.ChunkReuse.Backends;
using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;

namespace DotLLM.ChunkReuse;

/// <summary>
/// Encodes a token chunk: an isolated prefill at positions <c>0..n-1</c> in one forward into a fresh exact-F32 cache of a chunk backend,
/// then either a host copy of every layer's K/V rows (<see cref="ChunkKvEntry"/>) or the cache itself (<see cref="ResidentChunkKvEntry"/>).
/// Entries are stamped with the backend's tag.
/// </summary>
public sealed class ChunkEncoder
{
    /// <summary>Creates an encoder that stamps entries with <paramref name="modelFingerprint"/>.</summary>
    /// <param name="modelFingerprint">Identity of the model the entries come from (<see cref="ChunkFingerprint.Compute(DotLLM.Core.Models.ModelConfig, string, string, string, string)"/>).</param>
    public ChunkEncoder(string modelFingerprint)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelFingerprint);
        ModelFingerprint = modelFingerprint;
    }

    /// <summary>The fingerprint written into every entry.</summary>
    public string ModelFingerprint { get; }

    /// <summary>Encodes on the CPU backend (<see cref="CpuChunkBackend"/>; the model must be the CPU <c>TransformerModel</c>).</summary>
    /// <param name="model">The CPU model.</param>
    /// <param name="tokenIds">The chunk; not empty.</param>
    /// <returns>The entry, keys post-RoPE at <c>0..n-1</c>, tagged <see cref="ChunkBackendTags.Cpu"/>.</returns>
    /// <exception cref="ArgumentException">The chunk is empty or longer than the model context.</exception>
    public ChunkKvEntry Encode(IModel model, ReadOnlySpan<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(model);
        return Encode(model, new CpuChunkBackend(model.Config), tokenIds);
    }

    /// <summary>Runs <paramref name="tokenIds"/> through <paramref name="model"/> on <paramref name="backend"/> and copies its per-layer K/V to the host.</summary>
    /// <param name="model">A model whose forward writes <paramref name="backend"/>'s caches (CPU <c>TransformerModel</c>, or a Vulkan model on the backend's device).</param>
    /// <param name="backend">The chunk backend; its tag is stamped on the entry.</param>
    /// <param name="tokenIds">The chunk; not empty.</param>
    /// <returns>The entry, keys post-RoPE at <c>0..n-1</c>.</returns>
    /// <exception cref="ArgumentException">The chunk is empty or longer than the model context.</exception>
    public ChunkKvEntry Encode(IModel model, IChunkBackend backend, ReadOnlySpan<int> tokenIds)
    {
        using IKvCache cache = Prefill(model, backend, tokenIds);
        int n = tokenIds.Length;
        var perLayer = (IPerLayerKvCache)cache;
        int layers = perLayer.LayerCount;
        var keys = new float[layers][];
        var values = new float[layers][];
        for (int l = 0; l < layers; l++)
        {
            int elems = n * perLayer.KvStrideOf(l);
            keys[l] = new float[elems];
            values[l] = new float[elems];
            backend.ReadRows(cache, l, 0, n, keys[l], values[l]);
        }
        return new ChunkKvEntry(ModelFingerprint, tokenIds.ToArray(), keys, values, backend.Tag);
    }

    /// <summary>Like <see cref="Encode(IModel, IChunkBackend, ReadOnlySpan{int})"/> but keeps the rows in the backend cache (no host copy).</summary>
    /// <param name="model">A model whose forward writes <paramref name="backend"/>'s caches.</param>
    /// <param name="backend">The chunk backend; its tag is stamped on the entry.</param>
    /// <param name="tokenIds">The chunk; not empty.</param>
    /// <returns>The resident entry; the caller disposes it.</returns>
    public ResidentChunkKvEntry EncodeResident(IModel model, IChunkBackend backend, ReadOnlySpan<int> tokenIds)
    {
        IKvCache cache = Prefill(model, backend, tokenIds);
        try
        {
            return new ResidentChunkKvEntry(ModelFingerprint, tokenIds, cache, backend.Tag);
        }
        catch
        {
            cache.Dispose();
            throw;
        }
    }

    private static IKvCache Prefill(IModel model, IChunkBackend backend, ReadOnlySpan<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(backend);
        if (tokenIds.IsEmpty)
            throw new ArgumentException("A chunk must contain at least one token.", nameof(tokenIds));
        int n = tokenIds.Length;
        if (n > model.Config.MaxSequenceLength)
            throw new ArgumentException($"Chunk of {n} tokens exceeds the model context {model.Config.MaxSequenceLength}.", nameof(tokenIds));

        IKvCache cache = backend.CreateCache(n);
        try
        {
            if (cache is not IPerLayerKvCache)
                throw new NotSupportedException($"Chunk encoding needs per-layer KV strides; {cache.GetType().Name} has none.");
            int[] positions = new int[n];
            for (int i = 0; i < n; i++) positions[i] = i;
            using (ITensor logits = model.Forward(tokenIds, positions, backend.ForwardDeviceId, cache, lastTokenLogitsOnly: true))
            {
                // Only the cache side effect is wanted.
            }
            if (cache.CurrentLength != n)
                throw new InvalidOperationException($"The model wrote {cache.CurrentLength} cache rows for a {n}-token chunk.");
            return cache;
        }
        catch
        {
            cache.Dispose();
            throw;
        }
    }
}
