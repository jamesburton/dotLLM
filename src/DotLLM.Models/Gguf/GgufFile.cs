using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Represents an opened GGUF file with parsed metadata, tensor descriptors, and memory-mapped tensor data.
/// Owns the memory-mapped file resources and must be disposed when no longer needed.
/// </summary>
public sealed unsafe class GgufFile : IDisposable
{
    /// <summary>
    /// How the tensor data section is mapped. <see cref="MemoryMappedFileAccess.Read"/> by
    /// default; <c>DOTLLM_GGUF_MAP_COW=1</c> selects
    /// <see cref="MemoryMappedFileAccess.CopyOnWrite"/>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>READ THIS BEFORE FLIPPING THE DEFAULT — this mapping is shared by every backend.</b>
    /// </para>
    /// <para>
    /// <b>Why the switch exists.</b> The Vulkan zero-copy weight import
    /// (<c>VK_EXT_external_memory_host</c>, issues #508/#507) is refused outright on read-only
    /// pages: measured on gfx1151/amdvlk, <c>vkAllocateMemory</c> returns
    /// <c>VK_ERROR_INVALID_EXTERNAL_HANDLE</c> (-1000072003) for a
    /// <see cref="MemoryMappedFileAccess.Read"/> view and succeeds for the identical bytes in a
    /// <see cref="MemoryMappedFileAccess.CopyOnWrite"/> view or in anonymous read-write memory.
    /// Because every GGUF is mapped read-only, the import has aliased <b>zero bytes of every
    /// real model</b> since it was written; the existing import tests all pass because they
    /// import <c>NativeMemory.AlignedAlloc</c> pages. See
    /// <c>VulkanHostImportMmapAccessModeTests</c>.
    /// </para>
    /// <para>
    /// <b>What copy-on-write costs</b>, measured on the same box over a 256 MiB mapping:
    /// creating the view reserves commit charge equal to the mapped size <i>up front</i>
    /// (+256 MiB), reading every page adds <b>no</b> private commit (the pages stay shared with
    /// the page cache, working set grows as normal), and the Vulkan import adds <b>no</b>
    /// private commit either — the driver pins without breaking copy-on-write. So it does not
    /// duplicate the weights, which was the thing that would have made it pointless.
    /// </para>
    /// <para>
    /// <b>What it risks.</b> (1) The up-front commit reservation is the whole file: a 30 GB
    /// checkpoint reserves 30 GB of commit charge, so mapping can fail on a machine with a
    /// small or disabled pagefile where the read-only map would have succeeded. (2) A stray
    /// write no longer throws — it silently privatises a page instead of faulting. That is
    /// still strictly safer than <see cref="MemoryMappedFileAccess.ReadWrite"/>, which would
    /// write the corruption through to the checkpoint on disk; CoW cannot touch the file.
    /// (3) Cross-process page-cache sharing is preserved, since CoW pages stay shared until
    /// written.
    /// </para>
    /// <para>
    /// It is opt-in rather than the default precisely because of (1): the CPU path depends on
    /// this mapping and gains nothing from the change, so the whole engine should not take a
    /// commit-charge regression for one backend's optimisation until the trade has been
    /// measured on the large checkpoints.
    /// </para>
    /// </remarks>
    public static MemoryMappedFileAccess MappingAccess { get; } =
        string.Equals(Environment.GetEnvironmentVariable("DOTLLM_GGUF_MAP_COW"), "1", StringComparison.Ordinal)
            ? MemoryMappedFileAccess.CopyOnWrite
            : MemoryMappedFileAccess.Read;

    private MemoryMappedFile? _mmf;
    private MemoryMappedViewAccessor? _accessor;
    private byte* _basePointer;
    private bool _disposed;

    /// <summary>Parsed GGUF header.</summary>
    public GgufHeader Header { get; }

    /// <summary>Typed metadata accessor.</summary>
    public GgufMetadata Metadata { get; }

    /// <summary>Ordered list of tensor descriptors as they appear in the file.</summary>
    public IReadOnlyList<GgufTensorDescriptor> Tensors { get; }

    /// <summary>Tensor descriptors indexed by name for fast lookup.</summary>
    public IReadOnlyDictionary<string, GgufTensorDescriptor> TensorsByName { get; }

    /// <summary>
    /// Pointer to the start of the tensor data section. Individual tensor data is at
    /// <c>DataBasePointer + tensor.DataOffset</c>.
    /// Returns <see cref="nint.Zero"/> if the file contains no tensors.
    /// </summary>
    public nint DataBasePointer { get; }

    /// <summary>Byte offset of the tensor data section from the start of the file.</summary>
    public long DataSectionOffset { get; }

    private GgufFile(
        GgufHeader header,
        GgufMetadata metadata,
        IReadOnlyList<GgufTensorDescriptor> tensors,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensorsByName,
        long dataSectionOffset,
        nint dataBasePointer,
        MemoryMappedFile? mmf,
        MemoryMappedViewAccessor? accessor,
        byte* basePointer)
    {
        Header = header;
        Metadata = metadata;
        Tensors = tensors;
        TensorsByName = tensorsByName;
        DataSectionOffset = dataSectionOffset;
        DataBasePointer = dataBasePointer;
        _mmf = mmf;
        _accessor = accessor;
        _basePointer = basePointer;
    }

    /// <summary>
    /// Opens and parses a GGUF file. The tensor data section is memory-mapped for zero-copy access.
    /// </summary>
    /// <param name="filePath">Path to the GGUF file.</param>
    /// <returns>A <see cref="GgufFile"/> instance. Caller owns disposal.</returns>
    /// <exception cref="FileNotFoundException">File does not exist.</exception>
    /// <exception cref="InvalidDataException">File is not a valid GGUF file.</exception>
    public static GgufFile Open(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"GGUF file not found: {filePath}", filePath);

        GgufHeader header;
        Dictionary<string, GgufMetadataValue> rawMetadata;
        List<GgufTensorDescriptor> tensors;
        long streamPositionAfterInfos;

        long fileLength;
        using (var fs = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.Read))
        using (var reader = new BinaryReader(fs))
        {
            header = GgufReader.ReadHeader(reader);
            rawMetadata = GgufReader.ReadMetadata(reader, header);
            tensors = GgufReader.ReadTensorInfos(reader, header);
            streamPositionAfterInfos = fs.Position;
            fileLength = fs.Length;
        }

        var metadata = new GgufMetadata(rawMetadata);

        // Alignment: default 32, overridable via general.alignment.
        uint alignment = metadata.GetUInt32OrDefault("general.alignment", 32);
        if (alignment == 0 || !BitOperations.IsPow2(alignment))
            throw new InvalidDataException(
                $"GGUF alignment must be a power of 2, got {alignment}.");

        long dataSectionOffset = AlignUp(streamPositionAfterInfos, alignment);

        // Validate tensor data fits within the file.
        long dataSectionLength = fileLength - dataSectionOffset;
        foreach (var tensor in tensors)
        {
            long tensorBytes = tensor.QuantizationType.ComputeByteCount(tensor.Shape.ElementCount);
            long endOffset = (long)tensor.DataOffset + tensorBytes;
            if (endOffset > dataSectionLength)
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' data extends beyond file boundary " +
                    $"(offset {tensor.DataOffset}, size {tensorBytes}, " +
                    $"data section size {dataSectionLength}).");
        }

        var tensorsByName = new Dictionary<string, GgufTensorDescriptor>(tensors.Count);
        foreach (var tensor in tensors)
            tensorsByName[tensor.Name] = tensor;

        // Memory-map the file for tensor data access.
        MemoryMappedFile? mmf = null;
        MemoryMappedViewAccessor? accessor = null;
        byte* basePointer = null;
        nint dataBasePointer = nint.Zero;

        if (header.TensorCount > 0)
        {
            try
            {
                MemoryMappedFileAccess access = MappingAccess;
                mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, access);
                accessor = mmf.CreateViewAccessor(0, 0, access);
                accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);
                dataBasePointer = (nint)(basePointer + accessor.PointerOffset + dataSectionOffset);
            }
            catch
            {
                if (basePointer != null)
                    accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
                accessor?.Dispose();
                mmf?.Dispose();
                throw;
            }
        }

        return new GgufFile(
            header,
            metadata,
            tensors.AsReadOnly(),
            tensorsByName,
            dataSectionOffset,
            dataBasePointer,
            mmf,
            accessor,
            basePointer);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        if (_basePointer != null)
        {
            _accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
            _basePointer = null;
        }

        _accessor?.Dispose();
        _accessor = null;

        _mmf?.Dispose();
        _mmf = null;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long AlignUp(long value, uint alignment)
    {
        long mask = alignment - 1;
        return (value + mask) & ~mask;
    }
}
