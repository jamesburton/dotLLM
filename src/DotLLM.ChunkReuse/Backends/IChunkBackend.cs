using DotLLM.Core.Attention;

namespace DotLLM.ChunkReuse.Backends;

/// <summary>
/// Where chunk K/V lives and how it is placed: the seam between the chunk-reuse algorithms (assembly, later encode, recompute and
/// probe) and a compute backend. The CPU backend (<see cref="CpuChunkBackend"/>) is the reference; the Vulkan backend
/// (<see cref="VulkanChunkBackend"/>) moves rows through <see cref="IPositionedHostKvCache"/> and shifts keys on the device.
/// </summary>
/// <remarks>
/// Rows are host F32 <c>[count, kvStride(layer)]</c>; K is post-RoPE. A backend writes only caches it vetted with
/// <see cref="RequireCache"/>. Entries carry the <see cref="Tag"/> of the backend that encoded them, and an entry is only
/// assembled by a backend with the same tag (CPU and Vulkan K/V differ by a numerical floor).
/// </remarks>
public interface IChunkBackend
{
    /// <summary>Backend identity stamped on entries and fingerprints (<see cref="ChunkBackendTags"/>).</summary>
    string Tag { get; }

    /// <summary>Creates an empty exact-F32 KV-cache of this backend for the backend's model.</summary>
    /// <param name="maxSeqLen">Positions the cache holds.</param>
    /// <returns>The cache.</returns>
    IKvCache CreateCache(int maxSeqLen);

    /// <summary>Throws <see cref="NotSupportedException"/> unless this backend can write <paramref name="cache"/> exactly.</summary>
    /// <param name="cache">The destination.</param>
    void RequireCache(IKvCache cache);

    /// <summary>Writes rows <c>[startPos, startPos + k.Length / stride)</c> of <paramref name="layer"/>; K and V have equal length.</summary>
    /// <param name="cache">A cache accepted by <see cref="RequireCache"/>.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="startPos">First position written.</param>
    /// <param name="k">K rows, a whole number of rows.</param>
    /// <param name="v">V rows, same length as <paramref name="k"/>.</param>
    void WriteRows(IKvCache cache, int layer, int startPos, ReadOnlySpan<float> k, ReadOnlySpan<float> v);

    /// <summary>Reads <paramref name="count"/> rows of <paramref name="layer"/> starting at <paramref name="startPos"/> as F32.</summary>
    /// <param name="cache">A cache accepted by <see cref="RequireCache"/>.</param>
    /// <param name="layer">Layer index.</param>
    /// <param name="startPos">First position read (rows must lie below <see cref="IKvCache.CurrentLength"/>).</param>
    /// <param name="count">Rows to read.</param>
    /// <param name="k">Receives K, at least <c>count × stride</c> floats.</param>
    /// <param name="v">Receives V, at least <c>count × stride</c> floats.</param>
    void ReadRows(IKvCache cache, int layer, int startPos, int count, Span<float> k, Span<float> v);

    /// <summary>
    /// Re-rotates the K rows <c>[startPos, startPos + count)</c> of every RoPE layer (NoPE layers are left alone) by
    /// <paramref name="delta"/> positions, in place; V is untouched. The rows must have been written.
    /// </summary>
    /// <param name="cache">A cache accepted by <see cref="RequireCache"/>.</param>
    /// <param name="startPos">First row.</param>
    /// <param name="count">Rows.</param>
    /// <param name="delta">Position delta (may be negative).</param>
    /// <param name="rope">Plain-RoPE parameters of the model (<see cref="RopeParameters.FromConfig"/>).</param>
    void ShiftKeys(IKvCache cache, int startPos, int count, int delta, RopeParameters rope);
}

/// <summary>Backend tags; the same literals as the <c>backend</c> argument of the chunk store key.</summary>
public static class ChunkBackendTags
{
    /// <summary>The CPU reference backend.</summary>
    public const string Cpu = "cpu";

    /// <summary>The Vulkan backend.</summary>
    public const string Vulkan = "vulkan";
}
