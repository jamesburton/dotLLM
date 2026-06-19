namespace DotLLM.Core.Attention;

/// <summary>
/// Optional capability for a KV-cache that exposes its per-layer row geometry.
/// Implemented by caches that can hold distinct per-layer KV row widths (Gemma-4
/// sliding vs global layers). The forward path uses this to validate a supplied
/// cache against the model's <see cref="KvGeometry"/> without binding to a concrete
/// cache type (keeping the universal <see cref="IKvCache"/> seam geometry-free).
/// </summary>
public interface IPerLayerKvCache
{
    /// <summary>Number of layers this cache holds.</summary>
    int LayerCount { get; }

    /// <summary>The cached K/V row width (FP32 elements) for <paramref name="layerIndex"/>.</summary>
    int KvStrideOf(int layerIndex);
}
