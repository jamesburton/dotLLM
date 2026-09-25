namespace DotLLM.Models.SafeTensors;

/// <summary>
/// Shard-aware safetensors reader. Owns one
/// <see cref="SafetensorsFile"/> per shard declared in a
/// <c>model.safetensors.index.json</c> sidecar, and exposes a flat
/// per-tensor lookup surface that consumers of
/// <see cref="ISafetensorsTensorSource"/> cannot distinguish from the
/// single-file case.
/// </summary>
/// <remarks>
/// <para>
/// HuggingFace's <c>save_pretrained</c> splits models above
/// <c>max_shard_size</c> (default 5 GiB) across files named
/// <c>model-00001-of-00005.safetensors</c>, <c>model-00002-of-…</c>, etc.
/// Every tensor lives in exactly one shard, and the sibling
/// <c>model.safetensors.index.json</c> enumerates which. Opening a
/// multi-shard checkpoint means opening each shard's header + mmap
/// region, then indexing tensors by name back to their owning shard so
/// consumers can look up a pointer without caring about the split.
/// </para>
/// <para>
/// <b>Duplicate handling.</b> If the index's <c>weight_map</c> points at
/// a tensor name in a specific shard, that is authoritative even if a
/// second shard happens to redeclare the same tensor. Shards that
/// contain tensors not mentioned in the index are still accepted (those
/// tensors become addressable by name) — this tolerates HF checkpoints
/// whose index lags slightly behind the on-disk shards. Conflicting
/// redeclarations in non-index shards throw, because the tensor data at
/// a given name would otherwise be ambiguous.
/// </para>
/// <para>
/// <b>Lifetime.</b> Disposing this object releases every underlying
/// <see cref="SafetensorsFile"/>. Any <see cref="nint"/> or span returned
/// from this source becomes invalid at that point.
/// </para>
/// </remarks>
public sealed class MultiShardSafetensorsFile : ISafetensorsTensorSource
{
    private readonly SafetensorsFile[] _shards;
    private readonly Dictionary<string, int> _tensorShardIndex;
    private bool _disposed;

    /// <summary>Number of shards this view spans.</summary>
    public int ShardCount => _shards.Length;

    /// <summary>The directory that contained the index + shards.</summary>
    public string RootDirectory { get; }

    /// <summary>The parsed index, if this view was opened from one.</summary>
    public SafetensorsIndex? Index { get; }

    /// <inheritdoc/>
    public IReadOnlyList<SafetensorsTensorDescriptor> Tensors { get; }

    /// <inheritdoc/>
    public IReadOnlyDictionary<string, SafetensorsTensorDescriptor> TensorsByName { get; }

    private MultiShardSafetensorsFile(
        string rootDirectory,
        SafetensorsIndex? index,
        SafetensorsFile[] shards,
        Dictionary<string, int> tensorShardIndex,
        IReadOnlyList<SafetensorsTensorDescriptor> tensors,
        IReadOnlyDictionary<string, SafetensorsTensorDescriptor> tensorsByName)
    {
        RootDirectory = rootDirectory;
        Index = index;
        _shards = shards;
        _tensorShardIndex = tensorShardIndex;
        Tensors = tensors;
        TensorsByName = tensorsByName;
    }

    /// <summary>
    /// Opens every shard named by <paramref name="index"/>, resolves
    /// <paramref name="index"/>'s <c>weight_map</c> against the actual
    /// tensors each shard declares, and returns a ready-to-query
    /// multi-shard view.
    /// </summary>
    /// <param name="indexFilePath">Absolute path to <c>model.safetensors.index.json</c>.</param>
    /// <param name="index">Parsed index contents.</param>
    /// <returns>An opened <see cref="MultiShardSafetensorsFile"/>. Caller owns disposal.</returns>
    /// <exception cref="FileNotFoundException">A shard named in the index is missing on disk.</exception>
    /// <exception cref="InvalidDataException">
    /// A tensor the index maps to shard X is not physically present in shard X, or two shards
    /// redeclare the same tensor name while the index disagrees about the owning shard.
    /// </exception>
    public static MultiShardSafetensorsFile Open(string indexFilePath, SafetensorsIndex index)
    {
        ArgumentNullException.ThrowIfNull(indexFilePath);
        ArgumentNullException.ThrowIfNull(index);

        string? dir = Path.GetDirectoryName(indexFilePath);
        if (string.IsNullOrEmpty(dir))
            throw new InvalidDataException(
                $"Could not determine parent directory of index file '{indexFilePath}'.");

        var shardNames = index.DistinctShardFileNames();
        if (shardNames.Count == 0)
            throw new InvalidDataException(
                "model.safetensors.index.json weight_map references zero shards.");

        return OpenInternal(dir, index, shardNames);
    }

    /// <summary>
    /// Convenience overload: reads the index JSON from
    /// <paramref name="indexFilePath"/> and then opens every shard.
    /// </summary>
    public static MultiShardSafetensorsFile Open(string indexFilePath)
    {
        var index = SafetensorsIndex.Load(indexFilePath);
        return Open(indexFilePath, index);
    }

    private static MultiShardSafetensorsFile OpenInternal(
        string dir,
        SafetensorsIndex? index,
        IReadOnlyList<string> shardNames)
    {
        var shards = new SafetensorsFile[shardNames.Count];
        var nameToShard = new Dictionary<string, int>(StringComparer.Ordinal);
        var flatTensors = new List<SafetensorsTensorDescriptor>();
        var flatByName = new Dictionary<string, SafetensorsTensorDescriptor>(StringComparer.Ordinal);

        try
        {
            // Open every shard.
            for (int i = 0; i < shardNames.Count; i++)
            {
                string shardPath = Path.Combine(dir, shardNames[i]);
                if (!File.Exists(shardPath))
                    throw new FileNotFoundException(
                        $"Safetensors shard '{shardNames[i]}' declared by the index is missing at '{shardPath}'.",
                        shardPath);
                shards[i] = SafetensorsFile.Open(shardPath);
            }

            // If we have an index, prefer its ownership; otherwise use the shard
            // enumeration order (first-seen wins).
            if (index is not null)
            {
                // First pass: apply the index's authoritative mapping.
                foreach (var (tensorName, shardFile) in index.WeightMap)
                {
                    int shardIdx = FindShardIndex(shardNames, shardFile);
                    if (shardIdx < 0)
                        throw new InvalidDataException(
                            $"index.json claims tensor '{tensorName}' lives in shard '{shardFile}', "
                            + "which is not present in the distinct-shard list (bug).");
                    if (!shards[shardIdx].TensorsByName.TryGetValue(tensorName, out var desc))
                        throw new InvalidDataException(
                            $"index.json claims tensor '{tensorName}' lives in shard '{shardFile}', "
                            + "but that shard does not declare it.");
                    nameToShard[tensorName] = shardIdx;
                    flatByName[tensorName] = desc;
                    flatTensors.Add(desc);
                }

                // Second pass: pull in any tensors a shard declares that the
                // index omitted (some HF exports under-document pad/norm
                // tensors). Conflicting redeclarations throw.
                for (int i = 0; i < shards.Length; i++)
                {
                    foreach (var desc in shards[i].Tensors)
                    {
                        if (nameToShard.TryGetValue(desc.Name, out int existingShard))
                        {
                            if (existingShard == i)
                                continue; // already recorded by the index pass
                            throw new InvalidDataException(
                                $"Tensor '{desc.Name}' appears in shard '{shardNames[existingShard]}' "
                                + $"(per index.json) and also in shard '{shardNames[i]}' — ambiguous.");
                        }
                        nameToShard[desc.Name] = i;
                        flatByName[desc.Name] = desc;
                        flatTensors.Add(desc);
                    }
                }
            }
            else
            {
                // No index: walk shards in order, first-seen wins, conflicts throw.
                for (int i = 0; i < shards.Length; i++)
                {
                    foreach (var desc in shards[i].Tensors)
                    {
                        if (nameToShard.ContainsKey(desc.Name))
                            throw new InvalidDataException(
                                $"Tensor '{desc.Name}' appears in multiple shards "
                                + "and no index.json resolves the ambiguity.");
                        nameToShard[desc.Name] = i;
                        flatByName[desc.Name] = desc;
                        flatTensors.Add(desc);
                    }
                }
            }

            return new MultiShardSafetensorsFile(dir, index, shards, nameToShard, flatTensors, flatByName);
        }
        catch
        {
            for (int i = 0; i < shards.Length; i++)
                shards[i]?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Opens a directory that contains a set of shard files whose
    /// ownership is inferred from the on-disk layout rather than an
    /// explicit index. Useful when an index.json is absent but the caller
    /// knows which shards to include. Pass the shard filenames in the
    /// order they should be resolved; conflicts on tensor names throw.
    /// </summary>
    internal static MultiShardSafetensorsFile OpenWithoutIndex(
        string directory,
        IReadOnlyList<string> shardFileNames)
    {
        ArgumentNullException.ThrowIfNull(directory);
        ArgumentNullException.ThrowIfNull(shardFileNames);
        if (shardFileNames.Count == 0)
            throw new ArgumentException("Need at least one shard.", nameof(shardFileNames));
        return OpenInternal(directory, index: null, shardFileNames);
    }

    private static int FindShardIndex(IReadOnlyList<string> shardNames, string target)
    {
        for (int i = 0; i < shardNames.Count; i++)
        {
            if (string.Equals(shardNames[i], target, StringComparison.Ordinal))
                return i;
        }
        return -1;
    }

    /// <inheritdoc/>
    public nint GetTensorPointer(string name)
    {
        if (!_tensorShardIndex.TryGetValue(name, out int shardIdx))
            throw new KeyNotFoundException(
                $"Multi-shard safetensors view has no tensor named '{name}'.");
        return _shards[shardIdx].GetTensorPointer(name);
    }

    /// <inheritdoc/>
    public ReadOnlySpan<byte> GetTensorSpan(string name)
    {
        if (!_tensorShardIndex.TryGetValue(name, out int shardIdx))
            throw new KeyNotFoundException(
                $"Multi-shard safetensors view has no tensor named '{name}'.");
        return _shards[shardIdx].GetTensorSpan(name);
    }

    /// <summary>
    /// Returns the 0-based shard index that owns <paramref name="name"/>.
    /// Used by tests and diagnostics.
    /// </summary>
    public int GetShardIndexFor(string name)
    {
        if (!_tensorShardIndex.TryGetValue(name, out int idx))
            throw new KeyNotFoundException(
                $"Multi-shard safetensors view has no tensor named '{name}'.");
        return idx;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        for (int i = 0; i < _shards.Length; i++)
            _shards[i]?.Dispose();
    }
}
