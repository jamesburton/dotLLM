namespace DotLLM.Core.Attention;

/// <summary>
/// A KV cache whose storage lives in device memory and that can say how much (issue #1848). The engine reads it to decide whether
/// evicting retained prefix-cache entries could make a refused allocation fit, instead of emptying the cache for nothing.
/// </summary>
public interface IDeviceResidentKvCache
{
    /// <summary>Device bytes this cache holds now (all its buffers, freed on dispose).</summary>
    long DeviceBytes { get; }
}
