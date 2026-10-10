using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Text.RegularExpressions;
using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Represents an opened GGUF file with parsed metadata, tensor descriptors, and memory-mapped tensor data.
/// Owns the memory-mapped file resources and must be disposed when no longer needed.
/// </summary>
public sealed class GgufFile : IDisposable
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


    private readonly Shard[] _shards;
    private readonly Dictionary<string, int>? _shardByTensor;
    private readonly nint _dataBasePointer;
    private bool _disposed;

    /// <summary>Parsed GGUF header (of the first shard when the file is a split set).</summary>
    public GgufHeader Header { get; }

    /// <summary>
    /// Typed metadata accessor. For a split set this is the first shard's metadata, which holds
    /// every key except the per-shard <c>split.*</c> bookkeeping (llama.cpp's
    /// <c>llama-gguf-split</c> writes the model KVs to shard 1 only).
    /// </summary>
    public GgufMetadata Metadata { get; }

    /// <summary>
    /// Ordered list of tensor descriptors: file order for a single file, shard order then file
    /// order for a split set. <see cref="GgufTensorDescriptor.DataOffset"/> is always relative to
    /// the data section of the shard that owns the tensor — never to a unified address space; use
    /// <see cref="TensorDataPointer(in GgufTensorDescriptor)"/> to resolve a pointer.
    /// </summary>
    public IReadOnlyList<GgufTensorDescriptor> Tensors { get; }

    /// <summary>Tensor descriptors indexed by name for fast lookup (unified across all shards).</summary>
    public IReadOnlyDictionary<string, GgufTensorDescriptor> TensorsByName { get; }

    /// <summary>
    /// True when this file is a multi-shard (<c>-0000N-of-0000M.gguf</c>) set opened through its
    /// first shard (issue #756).
    /// </summary>
    public bool IsSplit => _shards.Length > 1;

    /// <summary>Number of shard files backing this logical GGUF (1 for an ordinary file).</summary>
    public int ShardCount => _shards.Length;

    /// <summary>Paths of the shard files in shard order (a single entry for an ordinary file).</summary>
    public IReadOnlyList<string> ShardPaths { get; }

    /// <summary>
    /// Pointer to the start of the tensor data section. Individual tensor data is at
    /// <c>DataBasePointer + tensor.DataOffset</c>. Returns <see cref="nint.Zero"/> if the file
    /// contains no tensors.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    /// The file is a split set (<see cref="IsSplit"/>): the shards are separate mappings, so there is no
    /// single base pointer and returning one would make every <c>DataBasePointer + DataOffset</c> read the
    /// wrong bytes without any error. Resolve pointers with
    /// <see cref="TensorDataPointer(in GgufTensorDescriptor)"/> instead.
    /// </exception>
    public nint DataBasePointer
    {
        get
        {
            if (_shards.Length > 1)
                throw new InvalidOperationException(
                    $"This GGUF is a {_shards.Length}-shard split set whose tensor data lives in separate mappings, so it has no " +
                    "single DataBasePointer (DataBasePointer + DataOffset would address the wrong bytes). Resolve each tensor with " +
                    "GgufFile.TensorDataPointer(...). Loader support for split GGUFs is tracked in issue #756.");
            return _dataBasePointer;
        }
    }

    /// <summary>Byte offset of the (first shard's) tensor data section from the start of its file.</summary>
    public long DataSectionOffset { get; }

    /// <summary>
    /// Byte length of the tensor data section — exactly the range <see cref="DataBasePointer"/>
    /// maps, and so the length a page-residency census of the mapping must use. For a split set it is the
    /// SUM of every shard's data section (the model's total weight bytes), which no single mapping covers.
    /// </summary>
    /// <remarks>
    /// Exposed because the alternative, reopening the file to ask its length, does not work:
    /// <see cref="FileInfo.Length"/> on a symlink reports the reparse point (every model in the HF
    /// hub cache is reached through one), and a second <c>File.OpenRead</c> hits a sharing violation
    /// whenever the mapping is copy-on-write — that mode needs write access, so the mapping holds the
    /// file with <c>FileShare.None</c>. This value is already known at parse time; use it.
    /// </remarks>
    public long DataSectionLength { get; }

    /// <summary>
    /// Resolves the in-memory address of a tensor's data, for single files and split sets alike.
    /// </summary>
    /// <param name="tensor">A descriptor taken from <see cref="Tensors"/> / <see cref="TensorsByName"/>.</param>
    /// <returns>Address of the first byte of the tensor's data inside its shard's mapping.</returns>
    /// <exception cref="KeyNotFoundException">No tensor of that name exists.</exception>
    public nint TensorDataPointer(in GgufTensorDescriptor tensor)
    {
        if (_shardByTensor is null)
            return _dataBasePointer + (nint)tensor.DataOffset;

        if (!_shardByTensor.TryGetValue(tensor.Name, out int shard))
            throw new KeyNotFoundException($"GGUF tensor '{tensor.Name}' not found in any shard.");
        return _shards[shard].DataBase + (nint)tensor.DataOffset;
    }

    /// <summary>Resolves the in-memory address of the named tensor's data.</summary>
    /// <exception cref="KeyNotFoundException">No tensor of that name exists.</exception>
    public nint TensorDataPointer(string tensorName)
    {
        if (!TensorsByName.TryGetValue(tensorName, out var tensor))
            throw new KeyNotFoundException($"GGUF tensor '{tensorName}' not found.");
        return TensorDataPointer(in tensor);
    }

    /// <summary>Zero-based index of the shard that stores the named tensor (always 0 for an ordinary file).</summary>
    /// <exception cref="KeyNotFoundException">No tensor of that name exists.</exception>
    public int GetTensorShardIndex(string tensorName)
    {
        if (!TensorsByName.ContainsKey(tensorName))
            throw new KeyNotFoundException($"GGUF tensor '{tensorName}' not found.");
        return _shardByTensor is null ? 0 : _shardByTensor[tensorName];
    }

    private GgufFile(
        GgufHeader header,
        GgufMetadata metadata,
        IReadOnlyList<GgufTensorDescriptor> tensors,
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensorsByName,
        Dictionary<string, int>? shardByTensor,
        Shard[] shards)
    {
        Header = header;
        Metadata = metadata;
        Tensors = tensors;
        TensorsByName = tensorsByName;
        _shardByTensor = shardByTensor;
        _shards = shards;
        ShardPaths = Array.ConvertAll(shards, static s => s.Path);
        DataSectionOffset = shards[0].DataSectionOffset;
        long total = 0;
        foreach (var s in shards) total += s.DataSectionLength;
        DataSectionLength = total;
        _dataBasePointer = shards[0].DataBase;
    }

    /// <summary>
    /// Opens and parses a GGUF file. The tensor data section is memory-mapped for zero-copy access.
    /// A split set (<c>name-00001-of-0000M.gguf</c>, as written by <c>llama-gguf-split</c> and shipped for
    /// every model over ~50 GB) is opened through its FIRST shard: the sibling shards are discovered by the
    /// naming convention, validated against <c>split.no</c> / <c>split.count</c> / <c>split.tensors.count</c>,
    /// and exposed as one logical tensor table (issue #756).
    /// </summary>
    /// <param name="filePath">Path to the GGUF file (the first shard for a split set).</param>
    /// <returns>A <see cref="GgufFile"/> instance. Caller owns disposal.</returns>
    /// <exception cref="FileNotFoundException">File does not exist, or a sibling shard of a split set is missing
    /// (the exception's <see cref="FileNotFoundException.FileName"/> is the missing shard's path).</exception>
    /// <exception cref="InvalidDataException">File is not a valid GGUF file, or a split set is inconsistent.</exception>
    public static GgufFile Open(string filePath)
    {
        if (!File.Exists(filePath))
            throw new FileNotFoundException($"GGUF file not found: {filePath}", filePath);

        ParsedShard first = ParseShard(filePath);
        var parsed = new List<ParsedShard> { first };

        int splitCount = ReadSplitInt(first.Metadata, "split.count", filePath) ?? 1;
        if (splitCount > 1)
        {
            ValidateFirstShard(first, filePath, splitCount, out string dir, out string prefix);
            for (int i = 2; i <= splitCount; i++)
            {
                string sibling = Path.Combine(dir, $"{prefix}-{i:D5}-of-{splitCount:D5}.gguf");
                if (!File.Exists(sibling))
                    throw new FileNotFoundException(
                        $"GGUF split set '{filePath}' declares {splitCount} shards but shard {i} is missing: {sibling}", sibling);

                ParsedShard shard = ParseShard(sibling);
                ValidateSiblingShard(shard, sibling, expectedNo: i - 1, splitCount);
                parsed.Add(shard);
            }

            ValidateSplitTensorTable(parsed, filePath);
        }

        // Unified tensor table, in shard order.
        var tensors = new List<GgufTensorDescriptor>();
        Dictionary<string, int>? shardByTensor = splitCount > 1 ? new Dictionary<string, int>() : null;
        for (int si = 0; si < parsed.Count; si++)
        {
            foreach (var t in parsed[si].Tensors)
            {
                tensors.Add(t);
                shardByTensor?.Add(t.Name, si);
            }
        }

        var tensorsByName = new Dictionary<string, GgufTensorDescriptor>(tensors.Count);
        foreach (var tensor in tensors)
            tensorsByName[tensor.Name] = tensor;

        // Memory-map every shard that carries tensor data (a metadata-only first shard maps nothing).
        var shards = new Shard[parsed.Count];
        try
        {
            for (int si = 0; si < parsed.Count; si++)
                shards[si] = Shard.Map(parsed[si]);
        }
        catch
        {
            foreach (var s in shards) s?.Dispose();
            throw;
        }

        return new GgufFile(first.Header, first.Metadata, tensors.AsReadOnly(), tensorsByName, shardByTensor, shards);
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;

        foreach (var shard in _shards)
            shard.Dispose();
    }

    // ───────────────────────────── shard parsing ─────────────────────────────

    private sealed record ParsedShard(
        string Path,
        GgufHeader Header,
        GgufMetadata Metadata,
        List<GgufTensorDescriptor> Tensors,
        long DataSectionOffset,
        long DataSectionLength);

    /// <summary>Parses one GGUF file's header, metadata and tensor table and bounds-checks every tensor against ITS data section.</summary>
    private static ParsedShard ParseShard(string filePath)
    {
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
        GgufReader.ReclassifyUpstreamQ2_0(tensors, alignment, dataSectionLength);   // #823: type id 42 = PQ2_0 or upstream Q2_0, told apart by extent
        foreach (var tensor in tensors)
        {
            long tensorBytes = tensor.QuantizationType.ComputeByteCount(tensor.Shape.ElementCount);
            long endOffset = (long)tensor.DataOffset + tensorBytes;
            if (endOffset > dataSectionLength)
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' data extends beyond file boundary " +
                    $"(offset {tensor.DataOffset}, size {tensorBytes}, " +
                    $"data section size {dataSectionLength}) in '{filePath}'.");
        }

        GgufReader.ValidatePq2_0Layout(tensors, alignment, dataSectionLength);

        return new ParsedShard(filePath, header, metadata, tensors, dataSectionOffset, dataSectionLength);
    }

    private static readonly Regex s_shardName = new(
        @"^(?<prefix>.+)-(?<no>\d{5})-of-(?<count>\d{5})\.gguf$",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    /// <summary>Checks the file handed to <see cref="Open"/> really is shard 1 of a well-named set; yields the sibling name stem.</summary>
    private static void ValidateFirstShard(ParsedShard first, string filePath, int splitCount, out string dir, out string prefix)
    {
        string fileName = Path.GetFileName(filePath);
        Match m = s_shardName.Match(fileName);
        if (!m.Success)
            throw new InvalidDataException(
                $"GGUF '{filePath}' declares split.count={splitCount} but its name does not follow the " +
                "'<name>-0000N-of-0000M.gguf' convention, so the other shards cannot be located.");

        int fileNo = int.Parse(m.Groups["no"].Value);
        int fileCount = int.Parse(m.Groups["count"].Value);
        prefix = m.Groups["prefix"].Value;
        dir = Path.GetDirectoryName(Path.GetFullPath(filePath)) ?? ".";

        if (fileCount != splitCount)
            throw new InvalidDataException(
                $"GGUF '{filePath}' is named as one of {fileCount} shards but its split.count metadata says {splitCount}.");

        int? splitNo = ReadSplitInt(first.Metadata, "split.no", filePath);
        if (splitNo is null || splitNo.Value + 1 != fileNo)
            throw new InvalidDataException(
                $"GGUF '{filePath}' is named shard {fileNo} of {fileCount} but its split.no metadata is " +
                $"{(splitNo is null ? "absent" : splitNo.Value.ToString())} (zero-based; expected {fileNo - 1}).");

        if (fileNo != 1)
            throw new InvalidDataException(
                $"GGUF '{filePath}' is shard {fileNo} of {fileCount}; open the first shard " +
                $"'{prefix}-{1:D5}-of-{fileCount:D5}.gguf' instead (only it carries the model metadata).");
    }

    private static void ValidateSiblingShard(ParsedShard shard, string path, int expectedNo, int splitCount)
    {
        int? no = ReadSplitInt(shard.Metadata, "split.no", path);
        int? count = ReadSplitInt(shard.Metadata, "split.count", path);
        if (count is null || count.Value != splitCount)
            throw new InvalidDataException(
                $"GGUF shard '{path}' has split.count {(count is null ? "absent" : count.Value.ToString())}, expected {splitCount}.");
        if (no is null || no.Value != expectedNo)
            throw new InvalidDataException(
                $"GGUF shard '{path}' has split.no {(no is null ? "absent" : no.Value.ToString())}, expected {expectedNo} " +
                "(shards were renamed or mixed from different sets).");
    }

    private static void ValidateSplitTensorTable(List<ParsedShard> shards, string filePath)
    {
        var seen = new Dictionary<string, string>();
        long total = 0;
        foreach (var shard in shards)
        {
            foreach (var t in shard.Tensors)
            {
                if (!seen.TryAdd(t.Name, shard.Path))
                    throw new InvalidDataException(
                        $"Tensor '{t.Name}' appears in both '{seen[t.Name]}' and '{shard.Path}' of split set '{filePath}'.");
                total++;
            }
        }

        int? declared = ReadSplitInt(shards[0].Metadata, "split.tensors.count", filePath);
        if (declared is not null && declared.Value != total)
            throw new InvalidDataException(
                $"GGUF split set '{filePath}' declares split.tensors.count={declared.Value} but its {shards.Count} shards " +
                $"contain {total} tensors (a shard is truncated, replaced, or from a different quantization).");
    }

    /// <summary>Reads an integer-typed <c>split.*</c> key (u16 in files written by llama.cpp, i32 for the tensor count).</summary>
    private static int? ReadSplitInt(GgufMetadata metadata, string key, string filePath)
    {
        if (!metadata.TryGetValue(key, out var v))
            return null;

        long value = v.Value switch
        {
            byte b => b,
            sbyte sb => sb,
            ushort us => us,
            short s => s,
            uint ui => ui,
            int i => i,
            ulong ul when ul <= int.MaxValue => (long)ul,
            long l => l,
            _ => throw new InvalidDataException(
                $"GGUF '{filePath}' metadata '{key}' has non-integer type {v.Type}."),
        };
        if (value < 0 || value > int.MaxValue)
            throw new InvalidDataException($"GGUF '{filePath}' metadata '{key}' value {value} is out of range.");
        return (int)value;
    }

    // ───────────────────────────── shard mapping ─────────────────────────────

    /// <summary>Owns one shard file's memory mapping.</summary>
    private sealed unsafe class Shard : IDisposable
    {
        private MemoryMappedFile? _mmf;
        private MemoryMappedViewAccessor? _accessor;
        private byte* _basePointer;

        public string Path { get; }
        public long DataSectionOffset { get; }
        public long DataSectionLength { get; }
        public nint DataBase { get; private set; }

        private Shard(string path, long dataSectionOffset, long dataSectionLength)
        {
            Path = path;
            DataSectionOffset = dataSectionOffset;
            DataSectionLength = dataSectionLength;
        }

        public static Shard Map(ParsedShard parsed)
        {
            var shard = new Shard(parsed.Path, parsed.DataSectionOffset, parsed.DataSectionLength);
            if (parsed.Header.TensorCount == 0)
                return shard;

            try
            {
                MemoryMappedFileAccess access = MappingAccess;
                shard._mmf = MemoryMappedFile.CreateFromFile(parsed.Path, FileMode.Open, null, 0, access);
                shard._accessor = shard._mmf.CreateViewAccessor(0, 0, access);
                byte* basePointer = null;
                shard._accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePointer);
                shard._basePointer = basePointer;
                shard.DataBase = (nint)(basePointer + shard._accessor.PointerOffset + parsed.DataSectionOffset);
                return shard;
            }
            catch
            {
                shard.Dispose();
                throw;
            }
        }

        public void Dispose()
        {
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
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long AlignUp(long value, uint alignment)
    {
        long mask = alignment - 1;
        return (value + mask) & ~mask;
    }
}
