using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Tensors;

namespace DotLLM.Cpu.Kernels;

/// <summary>
/// Row-level access to a quantised engine KV cache (<see cref="IQuantizedKvCache"/>: Q8_0 / Q4_0 rows plus an optional fp32 window of
/// the most recent rows) for the Qwen4-Exp QSA layers, whose sparse attention reads an arbitrary, non-contiguous set of cached rows
/// per query (#841).
/// </summary>
/// <remarks>
/// The generic quantised attention kernel streams contiguous tiles; QSA instead gathers the rows the indexer selected, so the rows
/// are dequantised one by one into a scratch (up to ~2k rows of <c>numKvHeads * headDim</c> floats per query token). The rows of the
/// chunk being processed never go through the quantiser on the read side: they are read from the fp32 projections the layer just
/// computed, so a chunk is attended at full precision and only rows from EARLIER chunks carry quantisation error.
/// </remarks>
public static unsafe class Qwen4ExpKvRows
{
    /// <summary>
    /// Validates that <paramref name="cache"/> can carry QSA rows: both K and V quantised (a mixed fp32/quantised cache does not store the
    /// fp32 side's evicted rows).
    /// </summary>
    public static void RequireSupported(IQuantizedKvCache cache)
    {
        if (cache.KeyDType == KvCacheDType.F32 || cache.ValueDType == KvCacheDType.F32)
            throw new NotSupportedException(
                $"qwen4exp QSA layers need both K and V quantised in a quantised KV cache (got K={cache.KeyDType}, V={cache.ValueDType}); " +
                "use --cache-type-k and --cache-type-v together, or an fp32 cache.");
    }

    /// <summary>
    /// <c>cache.Update</c> for QSA rows. A quantised cache with an fp32 window of W rows evicts the oldest rows into the quantised store
    /// at the START of each update, from the ring; an update longer than W would quantise ring slots that were never written. Such
    /// updates are therefore split into sub-chunks of at most W rows (W = 0, pure-quantised, needs no split).
    /// </summary>
    public static void Update(IKvCache cache, float* k, float* v, int rows, int stride, ReadOnlySpan<int> positions, int slot)
    {
        int step = rows;
        if (cache is IQuantizedKvCache q && q.WindowCapacity > 0) step = Math.Min(rows, q.WindowCapacity);
        for (int off = 0; off < rows; off += step)
        {
            int n = Math.Min(step, rows - off);
            var kRef = new TensorRef(n, stride, DType.Float32, -1, (nint)(k + (long)off * stride));
            var vRef = new TensorRef(n, stride, DType.Float32, -1, (nint)(v + (long)off * stride));
            cache.Update(kRef, vRef, positions.Slice(off, n), slot);
        }
    }

    /// <summary>
    /// Reads cached row <paramref name="pos"/> of <paramref name="slot"/> as fp32 (dequantising it when it is in the quantised region,
    /// copying it when it is still in the fp32 window).
    /// </summary>
    public static void ReadRow(IQuantizedKvCache cache, int slot, int pos, int stride, float* kDst, float* vDst)
    {
        if (pos < cache.QuantizedLength)
        {
            Dequant((byte*)cache.GetQuantizedKeysPtr(slot) + (long)pos * cache.KeyQuantizedRowBytesOf(slot), kDst, stride, cache.KeyDType);
            Dequant((byte*)cache.GetQuantizedValuesPtr(slot) + (long)pos * cache.ValueQuantizedRowBytesOf(slot), vDst, stride, cache.ValueDType);
        }
        else
        {
            int w = cache.WindowCapacity;
            if (w <= 0) throw new InvalidOperationException($"Row {pos} is not in the quantised region ({cache.QuantizedLength} rows) and the cache has no window.");
            int ring = pos % w;
            new ReadOnlySpan<float>((float*)cache.GetWindowKeysPtr(slot) + (long)ring * stride, stride).CopyTo(new Span<float>(kDst, stride));
            new ReadOnlySpan<float>((float*)cache.GetWindowValuesPtr(slot) + (long)ring * stride, stride).CopyTo(new Span<float>(vDst, stride));
        }
    }

    /// <summary>
    /// Fills <paramref name="kDst"/> / <paramref name="vDst"/> (<c>[keyIdx.Length, stride]</c>) with the rows at the positions in
    /// <paramref name="keyIdx"/>: positions below <paramref name="chunkStart"/> come from <paramref name="cache"/>, positions in the
    /// current chunk from its fp32 projections <paramref name="chunkK"/> / <paramref name="chunkV"/>.
    /// </summary>
    public static void Gather(IQuantizedKvCache cache, int slot, ReadOnlySpan<int> keyIdx, int chunkStart, int stride,
                              ReadOnlySpan<float> chunkK, ReadOnlySpan<float> chunkV, Span<float> kDst, Span<float> vDst)
    {
        fixed (float* kp = kDst) fixed (float* vp = vDst)
        {
            for (int i = 0; i < keyIdx.Length; i++)
            {
                int p = keyIdx[i];
                float* kd = kp + (long)i * stride, vd = vp + (long)i * stride;
                if (p >= chunkStart)
                {
                    chunkK.Slice((p - chunkStart) * stride, stride).CopyTo(new Span<float>(kd, stride));
                    chunkV.Slice((p - chunkStart) * stride, stride).CopyTo(new Span<float>(vd, stride));
                }
                else ReadRow(cache, slot, p, stride, kd, vd);
            }
        }
    }

    private static void Dequant(byte* src, float* dst, int n, KvCacheDType t)
    {
        if (t == KvCacheDType.Q8_0) KvQuantize.Q8_0ToF32(src, dst, n);
        else if (t == KvCacheDType.Q4_0) KvQuantize.Q4_0ToF32(src, dst, n);
        else throw new NotSupportedException($"Unsupported KV dtype {t}.");
    }
}
