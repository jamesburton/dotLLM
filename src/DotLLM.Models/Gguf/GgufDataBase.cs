namespace DotLLM.Models.Gguf;

/// <summary>
/// Resolves tensor data addresses of a <see cref="GgufFile"/> — single file or split (<c>-0000N-of-0000M</c>) set
/// alike (issue #756). Replaces the <c>DataBasePointer + descriptor.DataOffset</c> idiom, which is only valid when
/// all tensor data lives in one mapping. Loaders resolve each tensor once at load time; the resulting pointers are
/// what the hot paths use, so there is no steady-state cost.
/// </summary>
public readonly struct GgufDataBase
{
    private readonly GgufFile? _file;
    private readonly nint _base;

    /// <summary>Creates a resolver for <paramref name="file"/>.</summary>
    public GgufDataBase(GgufFile file) => _file = file;

    private GgufDataBase(nint contiguousBase) { _file = null; _base = contiguousBase; }

    /// <summary>
    /// Resolver over ONE contiguous tensor-data region (offsets are relative to <paramref name="dataBase"/>) - for callers that
    /// hold a raw mapping rather than a <see cref="GgufFile"/>, such as tests with hand-built descriptors.
    /// </summary>
    public static GgufDataBase FromContiguousBase(nint dataBase) => new(dataBase);

    /// <summary>Address of the first byte of <paramref name="tensor"/>'s data inside its shard's mapping.</summary>
    [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
    public nint Of(in GgufTensorDescriptor tensor) => _file is null ? _base + (nint)tensor.DataOffset : _file.TensorDataPointer(in tensor);
}
