namespace DotLLM.HuggingFace;

/// <summary>A parsed model reference: <c>owner/repo[:tag]</c>, <c>owner/repo/file.gguf</c>, a bare name, or a path.</summary>
/// <param name="RepoId"><c>owner/repo</c> when the reference names a Hub repo; otherwise null.</param>
/// <param name="Filename">An explicit GGUF filename (inside the repo), or the bare name for a name-only reference.</param>
/// <param name="Tag">A quantisation tag (<c>Q4_K_M</c>) after a colon; <c>latest</c> and empty mean "the default".</param>
public readonly record struct ModelReference(string? RepoId, string? Filename, string? Tag)
{
    /// <summary>True for <c>owner/repo</c> forms that could be fetched from the Hub.</summary>
    public bool IsRepo => RepoId is not null;
}

/// <summary>
/// One resolver for every place a model is named (CLI <c>run/chat/serve</c>, server requests, <c>/v1/models/load</c>) — issue #714.
/// </summary>
/// <remarks>
/// <para>
/// Accepted references: a path to a <c>.gguf</c>; <c>owner/repo</c> (default quant); <c>owner/repo:Q4_K_M</c> (llama.cpp / ollama style tag);
/// <c>owner/repo/file.gguf</c>; <c>hf.co/owner/repo[:tag]</c> and <c>hf://owner/repo</c>; and a bare name, which matches a local file's stem
/// (the key <c>GET /v1/models</c> reports) or filename.
/// </para>
/// <para>
/// Local models are enumerated from BOTH the flat mirror (<c>~/.dotllm/models/{owner}/{repo}/*.gguf</c>) and the Hugging Face hub cache
/// (<c>models--owner--repo/snapshots/*/*.gguf</c>), so files fetched by <c>hf download</c> or llama.cpp are found without a re-pull. The same
/// file reachable through both is listed once.
/// </para>
/// </remarks>
public static class ModelResolver
{
    /// <summary>Default-quant preference, best first. Mirrors ollama's choice of Q4_K_M as the everyday default.</summary>
    private static readonly string[] QuantPreference =
    [
        "Q4_K_M", "Q4_K_S", "Q4_K_XL", "Q4_0", "IQ4_XS", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0", "Q3_K_M", "Q2_K", "BF16", "F16", "F32",
    ];

    /// <summary>The real file behind a path: follows symlinks (hf_hub snapshots on many setups) to the final target; the path itself otherwise.</summary>
    public static string ResolveLinks(string path)
    {
        try { return new FileInfo(path).ResolveLinkTarget(returnFinalTarget: true)?.FullName ?? path; }
        catch (IOException) { return path; }
    }

    /// <summary>File length that sees through symlinks (a link's own <see cref="FileInfo.Length"/> is not the target's).</summary>
    public static long FileLength(string path) => new FileInfo(ResolveLinks(path)).Length;

    /// <summary>Parses a reference. Never touches the file system or the network.</summary>
    public static ModelReference Parse(string arg)
    {
        string s = arg.Trim();
        foreach (string prefix in new[] { "hf://", "huggingface.co/", "hf.co/", "https://huggingface.co/", "https://hf.co/" })
            if (s.StartsWith(prefix, StringComparison.OrdinalIgnoreCase)) { s = s[prefix.Length..]; break; }

        string? tag = null;
        int colon = s.LastIndexOf(':');
        // A drive-letter colon ("C:\x.gguf") is not a tag separator.
        if (colon > 1 && !s[(colon + 1)..].Contains('/') && !s[(colon + 1)..].Contains('\\'))
        {
            tag = s[(colon + 1)..];
            s = s[..colon];
            if (tag.Length == 0 || tag.Equals("latest", StringComparison.OrdinalIgnoreCase)) tag = null;
        }

        string[] parts = s.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (parts.Length >= 3 && parts[^1].EndsWith(".gguf", StringComparison.OrdinalIgnoreCase))
            return new ModelReference($"{parts[0]}/{parts[1]}", string.Join('/', parts[2..]), tag);
        if (parts.Length == 2 && !s.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) && !Path.IsPathRooted(s))
            return new ModelReference($"{parts[0]}/{parts[1]}", null, tag);
        return new ModelReference(null, s, tag);
    }

    /// <summary>Every local GGUF (mirror + hub cache), de-duplicated by repo and filename. Never throws.</summary>
    public static List<LocalModel> EnumerateLocal(string? modelsDir = null, string? cacheRoot = null, bool includeOllama = false, string? ollamaRoot = null)
    {
        var byKey = new Dictionary<string, LocalModel>(StringComparer.OrdinalIgnoreCase);
        void Add(LocalModel m)
        {
            if (Path.GetFileName(m.Filename).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase)) return;   // multimodal projector, not a model
            string key = m.RepoId + "|" + m.Filename;
            if (!byKey.TryGetValue(key, out var have) || m.DownloadedAt > have.DownloadedAt) byKey[key] = m;
        }

        try { foreach (var m in HuggingFaceDownloader.ListLocalModels(modelsDir)) Add(m); } catch { /* unreadable mirror: skip */ }

        try
        {
            cacheRoot ??= HubCache.CacheRoot;
            if (Directory.Exists(cacheRoot))
            {
                foreach (string repoDir in Directory.GetDirectories(cacheRoot, "models--*"))
                {
                    string folder = Path.GetFileName(repoDir)["models--".Length..];
                    int sep = folder.IndexOf("--", StringComparison.Ordinal);
                    if (sep <= 0) continue;
                    string repoId = folder[..sep] + "/" + folder[(sep + 2)..];
                    string snaps = Path.Combine(repoDir, "snapshots");
                    if (!Directory.Exists(snaps)) continue;
                    foreach (string snap in Directory.GetDirectories(snaps))
                        foreach (string file in Directory.GetFiles(snap, "*.gguf", SearchOption.AllDirectories))
                        {
                            var info = new FileInfo(file);
                            long size = info.Length;
                            // hf_hub snapshots are symlinks on many setups: FileInfo.Length of a link is not the target's size.
                            if (info.LinkTarget is not null)
                                size = info.ResolveLinkTarget(returnFinalTarget: true) is FileInfo t && t.Exists ? t.Length : 0;
                            if (size == 0) continue;   // dangling link / partial file
                            Add(new LocalModel(repoId, Path.GetRelativePath(snap, file).Replace('\\', '/'), file, size, info.LastWriteTimeUtc));
                        }
                }
            }
        }
        catch { /* unreadable cache: skip */ }

        if (includeOllama)
        {
            // Models of an existing ollama installation, read in place (no copy): repo "ollama/{name}", file "{name}-{tag}.gguf".
            foreach (var om in OllamaStore.ListAll(ollamaRoot))
            {
                string safe = om.Ref.Name.Replace('/', '_');
                Add(new LocalModel("ollama/" + safe, $"{safe}-{om.Ref.Tag}.gguf", om.BlobPath, om.SizeBytes, om.ModifiedAt));
            }
        }

        return byKey.Values.OrderBy(m => m.RepoId, StringComparer.OrdinalIgnoreCase).ThenBy(m => m.Filename, StringComparer.OrdinalIgnoreCase).ToList();
    }

    /// <summary>
    /// Resolves a reference to a local GGUF path, or null when nothing local matches (the caller may then pull).
    /// <paramref name="quant"/> (the legacy <c>--quant</c> option) is equivalent to a <c>:tag</c>.
    /// </summary>
    public static string? ResolveLocal(string arg, string? quant = null, string? modelsDir = null, string? cacheRoot = null,
        bool includeOllama = false, string? ollamaRoot = null)
    {
        string? found = ResolveLocalCore(arg, quant, modelsDir, cacheRoot, includeOllama, ollamaRoot);
        if (found is not null || !includeOllama) return found;

        // An ollama reference ("llama3.2:3b", "ollama:user/model:tag"): our own pulled copy first, then the ollama store in place.
        if (OllamaRef.TryParse(arg) is { } o)
        {
            string pulled = OllamaRegistry.TargetPath(o, modelsDir);
            if (File.Exists(pulled)) return pulled;
            return OllamaStore.TryFind(o, ollamaRoot)?.BlobPath;
        }
        return null;
    }

    private static string? ResolveLocalCore(string arg, string? quant, string? modelsDir, string? cacheRoot, bool includeOllama, string? ollamaRoot)
    {
        if (string.IsNullOrWhiteSpace(arg)) return null;
        // An absolute path to an existing GGUF is accepted whatever its extension: ollama blobs are named sha256-<hex>.
        if (File.Exists(arg) && (arg.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) || (Path.IsPathRooted(arg) && HasGgufMagic(arg)))) return Path.GetFullPath(arg);

        var r = Parse(arg);
        string? tag = quant ?? r.Tag;
        var all = EnumerateLocal(modelsDir, cacheRoot, includeOllama, ollamaRoot);

        if (r.IsRepo)
        {
            var inRepo = all.Where(m => m.RepoId.Equals(r.RepoId, StringComparison.OrdinalIgnoreCase)).ToList();
            if (r.Filename is not null)
            {
                var exact = inRepo.FirstOrDefault(m => m.Filename.Equals(r.Filename, StringComparison.OrdinalIgnoreCase));
                return exact?.FullPath;
            }
            return ChooseFile(inRepo, tag)?.FullPath;
        }

        // Bare name: the file stem /v1/models reports, or the full filename.
        string name = r.Filename ?? arg;
        var named = all.Where(m =>
            Path.GetFileNameWithoutExtension(m.Filename).Equals(name, StringComparison.OrdinalIgnoreCase)
            || Path.GetFileName(m.Filename).Equals(name, StringComparison.OrdinalIgnoreCase)).ToList();
        return ChooseFile(named, tag)?.FullPath;
    }

    private static bool HasGgufMagic(string path)
    {
        try
        {
            using var fs = File.OpenRead(path);
            Span<byte> m = stackalloc byte[4];
            return fs.Read(m) == 4 && m[0] == (byte)'G' && m[1] == (byte)'G' && m[2] == (byte)'U' && m[3] == (byte)'F';
        }
        catch (IOException) { return false; }
        catch (UnauthorizedAccessException) { return false; }
    }

    /// <summary>
    /// Pulls an ollama reference from the registry into the model store and saves a profile named <c>name:tag</c> carrying its system prompt and
    /// parameters (an existing profile of that name is left alone). Returns the GGUF path.
    /// </summary>
    public static async Task<string> PullOllamaAsync(
        OllamaRef reference, OllamaRegistry registry, IProgress<(long bytesDownloaded, long? totalBytes)>? progress, CancellationToken ct,
        string? modelsDir = null, string? profilesDir = null)
    {
        var (path, profile) = await registry.PullAsync(reference, progress, ct, modelsDir).ConfigureAwait(false);
        if (ModelProfileStore.TryGet(reference.ToString(), profilesDir) is null)
            ModelProfileStore.Save(reference.ToString(), profile, profilesDir);
        return path;
    }

    /// <summary>
    /// Picks one file from a repo's candidates: those whose name contains <paramref name="tag"/> (when given), then the best by
    /// <see cref="QuantPreference"/>, then the largest. Shard 2..N of a split GGUF is never chosen over shard 1.
    /// </summary>
    public static LocalModel? ChooseFile(IReadOnlyList<LocalModel> candidates, string? tag)
    {
        var pool = candidates.Where(m => !IsLaterShard(m.Filename)).ToList();
        if (tag is not null) pool = pool.Where(m => m.Filename.Contains(tag, StringComparison.OrdinalIgnoreCase)).ToList();
        if (pool.Count == 0) return null;
        return pool.OrderBy(m => QuantRank(m.Filename)).ThenByDescending(m => m.SizeBytes).First();
    }

    /// <summary>Same selection over bare repo file entries (name, size) - used before a download.</summary>
    public static string? ChooseRemoteFile(IEnumerable<(string Path, long Size)> files, string? tag)
    {
        var pool = files
            .Where(f => f.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase)
                        && !System.IO.Path.GetFileName(f.Path).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase)
                        && !IsLaterShard(f.Path))
            .ToList();
        if (tag is not null) pool = pool.Where(f => f.Path.Contains(tag, StringComparison.OrdinalIgnoreCase)).ToList();
        if (pool.Count == 0) return null;
        return pool.OrderBy(f => QuantRank(f.Path)).ThenByDescending(f => f.Size).First().Path;
    }

    private static int QuantRank(string filename)
    {
        for (int i = 0; i < QuantPreference.Length; i++)
            if (filename.Contains(QuantPreference[i], StringComparison.OrdinalIgnoreCase)) return i;
        return QuantPreference.Length;
    }

    private static bool IsLaterShard(string filename)
    {
        // "model-00002-of-00005.gguf": only shard 1 is a valid entry point.
        var m = System.Text.RegularExpressions.Regex.Match(filename, @"-(\d{5})-of-(\d{5})\.gguf$", System.Text.RegularExpressions.RegexOptions.IgnoreCase);
        return m.Success && int.Parse(m.Groups[1].Value) > 1;
    }

    /// <summary>
    /// Downloads the reference's file into the hub cache (resumable) and returns the mirror path. The file is chosen by tag / default quant
    /// from the repo's GGUF listing, unless the reference names it.
    /// </summary>
    /// <exception cref="InvalidOperationException">The reference is not a Hub repo, or the repo has no matching GGUF.</exception>
    public static async Task<string> PullAsync(
        ModelReference reference, string? quant, HuggingFaceClient client, HuggingFaceDownloader downloader,
        IProgress<(long bytesDownloaded, long? totalBytes)>? progress, CancellationToken ct,
        string? cacheRoot = null, string? modelsDir = null)
    {
        if (!reference.IsRepo)
            throw new InvalidOperationException("Only 'owner/repo' references can be pulled from the Hugging Face Hub.");

        string repo = reference.RepoId!;
        string? file = reference.Filename;
        if (file is null)
        {
            var listing = await client.ListGgufFilesAsync(repo, ct).ConfigureAwait(false);
            file = ChooseRemoteFile(listing.Select(f => (f.Path, f.Size)), quant ?? reference.Tag)
                ?? throw new InvalidOperationException(
                    $"No matching .gguf in '{repo}'" + ((quant ?? reference.Tag) is { } t ? $" for tag '{t}'" : "") + ".");
        }

        var result = await downloader.DownloadToHubCacheAsync(repo, file, cacheRoot: cacheRoot, modelsDir: modelsDir, progress: progress, cancellationToken: ct).ConfigureAwait(false);
        return result.ModelPath;
    }

    /// <summary>
    /// Deletes one local model everywhere it is linked: the mirror entry, the hub-cache snapshot entries, and the hub-cache blob - the blob only
    /// when exactly one blob in the repo matches the file (same length and first MiB), so another tool's data is never guessed at.
    /// Returns the bytes released, 0 when nothing was removed.
    /// </summary>
    public static long DeleteLocal(LocalModel model, string? modelsDir = null, string? cacheRoot = null)
    {
        cacheRoot ??= HubCache.CacheRoot;
        string repoDir = HubCache.RepoDirectory(model.RepoId, cacheRoot);
        string blobsDir = Path.Combine(repoDir, "blobs");

        // Identify the blob BEFORE unlinking: with hard links the snapshot entry carries no pointer to it.
        string? blob = null;
        if (File.Exists(model.FullPath) && Directory.Exists(blobsDir))
        {
            var target = new FileInfo(model.FullPath).LinkTarget;   // symlinked snapshot (non-Windows)
            if (target is not null)
                blob = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(model.FullPath)!, target));
            else
            {
                var matches = Directory.GetFiles(blobsDir).Where(b => !b.EndsWith(".incomplete", StringComparison.Ordinal)
                    && new FileInfo(b).Length == model.SizeBytes && SameHead(b, model.FullPath)).ToList();
                if (matches.Count == 1) blob = matches[0];
            }
        }

        long freed = 0;
        void Remove(string path)
        {
            try { if (File.Exists(path)) { File.Delete(path); freed = model.SizeBytes; } } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }

        Remove(HubCache.MirrorPath(model.RepoId, model.Filename, modelsDir));
        Remove(model.FullPath);
        string snaps = Path.Combine(repoDir, "snapshots");
        if (Directory.Exists(snaps))
            foreach (string snap in Directory.GetDirectories(snaps))
                Remove(Path.Combine(snap, model.Filename.Replace('/', Path.DirectorySeparatorChar)));
        if (blob is not null) Remove(blob);

        PruneEmpty(Path.GetDirectoryName(HubCache.MirrorPath(model.RepoId, model.Filename, modelsDir)), stopAt: modelsDir ?? HuggingFaceDownloader.DefaultModelsDirectory);
        if (Directory.Exists(snaps))
        {
            foreach (string snap in Directory.GetDirectories(snaps)) PruneEmpty(snap, stopAt: snaps);
            if (!Directory.EnumerateFileSystemEntries(snaps).Any() && (!Directory.Exists(blobsDir) || !Directory.EnumerateFileSystemEntries(blobsDir).Any()))
            {
                try { Directory.Delete(repoDir, recursive: true); } catch (IOException) { }
            }
        }
        return freed;
    }

    private static bool SameHead(string a, string b)
    {
        const int n = 1 << 20;
        try
        {
            using var fa = File.OpenRead(a); using var fb = File.OpenRead(b);
            var ba = new byte[n]; var bb = new byte[n];
            int ra = fa.Read(ba, 0, n), rb = fb.Read(bb, 0, n);
            return ra == rb && ba.AsSpan(0, ra).SequenceEqual(bb.AsSpan(0, rb));
        }
        catch (IOException) { return false; }
    }

    /// <summary>Removes <paramref name="dir"/> and its empty parents, stopping before <paramref name="stopAt"/>.</summary>
    public static void PruneEmptyDirectories(string? dir, string stopAt) => PruneEmpty(dir, stopAt);

    private static void PruneEmpty(string? dir, string stopAt)
    {
        string stop = Path.GetFullPath(stopAt);
        while (dir is not null && Directory.Exists(dir) && Path.GetFullPath(dir).Length > stop.Length
               && !Directory.EnumerateFileSystemEntries(dir).Any())
        {
            try { Directory.Delete(dir); } catch (IOException) { break; }
            dir = Path.GetDirectoryName(dir);
        }
    }
}
