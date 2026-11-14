namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Minimal lookup surface shared by single-file
/// (<see cref="SafetensorsFile"/>) and multi-shard
/// (<see cref="MultiShardSafetensorsFile"/>) safetensors readers. Lets
/// weight loaders accept either shape without branching on the concrete
/// type.
/// </summary>
/// <remarks>
/// <para>
/// The interface intentionally mirrors exactly what consumers (dense
/// transformer loader, Mamba-3 loader) actually call: enumeration,
/// by-name lookup, zero-copy data access, and optional original file locations
/// for direct device loading. Container header details stay on the concrete type.
/// </para>
/// <para>
/// Implementations must keep the backing memory-mapped regions alive
/// until <see cref="System.IDisposable.Dispose"/> is called — the
/// pointers returned by <see cref="GetTensorPointer(string)"/> and
/// spans returned by <see cref="GetTensorSpan(string)"/> are only valid
/// for that lifetime.
/// </para>
/// </remarks>
public interface ISafetensorsTensorSource : IDisposable
{
    /// <summary>Tensor descriptors in declaration order (flat union across all shards).</summary>
    IReadOnlyList<SafetensorsTensorDescriptor> Tensors { get; }

    /// <summary>Tensor descriptors indexed by name for O(1) lookup.</summary>
    IReadOnlyDictionary<string, SafetensorsTensorDescriptor> TensorsByName { get; }

    /// <summary>
    /// Returns a pointer to the first byte of the named tensor's raw data
    /// in its owning memory-mapped region. Throws if the tensor is unknown.
    /// </summary>
    nint GetTensorPointer(string name);

    /// <summary>Returns the original file range for direct device loading, without reading tensor data.</summary>
    /// <remarks>In-memory or transformed sources must reject this operation instead of returning stale backing bytes.</remarks>
    (string Path, long FileOffset) GetTensorFileLocation(string name)
        => throw new NotSupportedException("This SafeTensors source does not expose original file ranges for direct device loading.");

    /// <summary>
    /// Returns a <see cref="ReadOnlySpan{Byte}"/> over the raw bytes of the
    /// named tensor's data. Valid until this source is disposed. Throws if
    /// the tensor is unknown or its byte count exceeds <c>Int32.MaxValue</c>.
    /// </summary>
    ReadOnlySpan<byte> GetTensorSpan(string name);
}
