namespace DotLLM.Core.Attention;

/// <summary>
/// Capability for a KV cache that can read and write per-layer K/V row ranges at arbitrary positions through host F32
/// memory (chunk-level KV reuse: place a cached chunk at an offset, read rows back for re-rotation or inspection).
/// Unlike <see cref="IHostStagedKvCache"/> it addresses <c>[startPosition, startPosition + count)</c>, not <c>[0, length)</c>.
/// </summary>
/// <remarks>
/// Host rows are always FP32 <c>[count, kvStride(layer)]</c>. A cache that stores a narrower element type (Vulkan F16)
/// rounds on write and widens on read, so a write-then-read round trip is exact only to that format's precision.
/// K rows are stored post-RoPE: writing rows at a different position than they were computed for needs a RoPE delta shift.
/// </remarks>
public interface IPositionedHostKvCache
{
    /// <summary>
    /// Writes <paramref name="count"/> rows of layer <paramref name="layerIndex"/> at positions
    /// <c>[startPosition, startPosition + count)</c>; the visible length becomes <c>max(CurrentLength, startPosition + count)</c>.
    /// </summary>
    void WriteRows(int layerIndex, int startPosition, int count, ReadOnlySpan<float> keys, ReadOnlySpan<float> values);

    /// <summary>Reads rows <c>[startPosition, startPosition + count)</c> of layer <paramref name="layerIndex"/> (must lie below <c>CurrentLength</c>) as FP32.</summary>
    void ReadRows(int layerIndex, int startPosition, int count, Span<float> keys, Span<float> values);
}
