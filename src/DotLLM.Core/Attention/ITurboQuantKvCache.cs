namespace DotLLM.Core.Attention;

/// <summary>
/// A quantized KV cache whose K and V history is stored as TurboQuant codes plus a per-head-vector norm and is read in place:
/// attention decodes <c>norm * centroid[code]</c> per tile in registers and never builds a decoded history (#1892).
/// The code buffers returned by <see cref="IQuantizedKvCache.GetQuantizedKeysPtr"/> / <see cref="IQuantizedKvCache.GetQuantizedValuesPtr"/>
/// follow <see cref="Layout"/>: per (position, kv head), <see cref="TurboQuantLayout.CodeUintsPerVec"/> little-endian words of packed codes.
/// </summary>
public interface ITurboQuantKvCache : IQuantizedKvCache
{
    /// <summary>The shared storage geometry (head dim, kv heads, MSE bits, norm format).</summary>
    TurboQuantLayout Layout { get; }

    /// <summary>Norm buffer of the key side for <paramref name="layer"/>: <c>[position, kvHead]</c> norms in <see cref="TurboQuantLayout.Norm"/> format.</summary>
    nint GetKeyNormsPtr(int layer);

    /// <summary>Norm buffer of the value side for <paramref name="layer"/>.</summary>
    nint GetValueNormsPtr(int layer);

    /// <summary>The Lloyd-Max centroids (already scaled by <c>1/sqrt(headDim)</c>), ascending, <c>1 &lt;&lt; Layout.MseBits</c> entries.</summary>
    ReadOnlySpan<float> Centroids { get; }

    /// <summary>The K-side rotation signs (+1 / -1), length <c>headDim</c>; used to rotate the query into code space.</summary>
    ReadOnlySpan<sbyte> SignsK { get; }

    /// <summary>The V-side rotation signs (+1 / -1), length <c>headDim</c>; used to rotate the attention output back.</summary>
    ReadOnlySpan<sbyte> SignsV { get; }
}
