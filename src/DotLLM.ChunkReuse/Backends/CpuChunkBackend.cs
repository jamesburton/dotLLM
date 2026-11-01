using DotLLM.Core.Attention;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;

namespace DotLLM.ChunkReuse.Backends;

/// <summary>
/// The CPU reference backend: host F32 <see cref="SimpleKvCache"/> rows written through <see cref="IKvCache.Update(TensorRef, TensorRef, ReadOnlySpan{int}, int)"/>
/// and keys re-rotated on the host with <see cref="RopeShift"/> (the same float operations as the assembler before the seam, so
/// its results are bit-identical to that path).
/// </summary>
public sealed class CpuChunkBackend : IChunkBackend
{
    private readonly ModelConfig _config;

    /// <summary>Creates the backend for a model with <paramref name="config"/>.</summary>
    /// <param name="config">The model configuration (KV geometry, head dim, NoPE layers).</param>
    public CpuChunkBackend(ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        _config = config;
    }

    /// <inheritdoc/>
    public string Tag => ChunkBackendTags.Cpu;

    /// <inheritdoc/>
    public IKvCache CreateCache(int maxSeqLen)
        => new SimpleKvCache(KvGeometry.FromConfig(_config, compactSlidingWindows: false), maxSeqLen);

    /// <inheritdoc/>
    /// <remarks>Accepts a <see cref="SimpleKvCache"/> without compact sliding windows (exact host F32 rows).</remarks>
    public void RequireCache(IKvCache cache)
    {
        ArgumentNullException.ThrowIfNull(cache);
        if (cache is not SimpleKvCache { SupportsExactF32PrefixCopy: true })
            throw new NotSupportedException(
                $"Chunk assembly on the CPU backend needs an exact host-F32 SimpleKvCache; {cache.GetType().Name} stores KV lossily, off-host or windowed.");
    }

    /// <inheritdoc/>
    public unsafe void WriteRows(IKvCache cache, int layer, int startPos, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
    {
        SimpleKvCache c = Checked(cache);
        int stride = c.KvStrideOf(layer);
        int n = RowCount(k, v, stride);
        int[] positions = new int[n];
        for (int i = 0; i < n; i++) positions[i] = startPos + i;
        fixed (float* kPtr = k)
        fixed (float* vPtr = v)
        {
            c.Update(new TensorRef(n, stride, DType.Float32, -1, (nint)kPtr), new TensorRef(n, stride, DType.Float32, -1, (nint)vPtr),
                positions, layer);
        }
    }

    /// <inheritdoc/>
    public void ReadRows(IKvCache cache, int layer, int startPos, int count, Span<float> k, Span<float> v)
    {
        SimpleKvCache c = Checked(cache);
        int stride = c.KvStrideOf(layer);
        if (startPos < 0 || count <= 0 || (long)startPos + count > c.CurrentLength)
            throw new ArgumentOutOfRangeException(nameof(count), $"rows [{startPos}, {startPos + count}) exceed CurrentLength {c.CurrentLength}.");
        int elems = count * stride;
        c.KeysSpan(layer).Slice(startPos * stride, elems).CopyTo(k);
        c.ValuesSpan(layer).Slice(startPos * stride, elems).CopyTo(v);
    }

    /// <inheritdoc/>
    public void ShiftKeys(IKvCache cache, int startPos, int count, int delta, RopeParameters rope)
    {
        SimpleKvCache c = Checked(cache);
        if (delta == 0) return;
        for (int l = 0; l < c.NumLayers; l++)
        {
            if (_config.IsNoRopeLayer(l)) continue;
            int elems = count * c.KvStrideOf(l);
            var k = new float[elems];
            var v = new float[elems];
            ReadRows(c, l, startPos, count, k, v);
            RopeShift.Apply(k, _config.HeadDim, delta, in rope);
            WriteRows(c, l, startPos, k, v);
        }
    }

    private SimpleKvCache Checked(IKvCache cache)
    {
        RequireCache(cache);
        return (SimpleKvCache)cache;
    }

    internal static int RowCount(ReadOnlySpan<float> k, ReadOnlySpan<float> v, int stride)
    {
        if (k.Length != v.Length || k.IsEmpty || k.Length % stride != 0)
            throw new ArgumentException($"K and V must hold the same whole number of rows of {stride} floats; got K={k.Length}, V={v.Length}.");
        return k.Length / stride;
    }
}
