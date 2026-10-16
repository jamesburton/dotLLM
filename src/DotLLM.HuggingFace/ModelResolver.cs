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
    public static long FileLength(string path) =>
        SplitGguf.IsFirstShard(Path.GetFileName(path)) ? SplitGguf.TotalLength(path) : new FileInfo(ResolveLinks(path)).Length;

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

    /// <summary>
    /// The id a local model is listed and served under: <c>owner/repo/file.gguf</c> for a Hub/mirror file, so two repos that both ship
    /// <c>q4_k_m.gguf</c> are distinguishable (a bare stem told a client nothing about which model it was). It is an accepted reference
    /// (<see cref="Parse"/>), so a client can send it straight back. Ollama-store entries keep their file stem; they are addressed by ollama name.
    /// </summary>
    public static string IdFor(LocalModel model) =>
        model.RepoId.StartsWith("ollama/", StringComparison.OrdinalIgnoreCase)
            ? Path.GetFileNameWithoutExtension(model.Filename)
            : $"{model.RepoId}/{model.Filename}";

    /// <summary>
    /// <see cref="IdFor(LocalModel)"/> for a resolved file path, derived from the file's place in the stores so it is the SAME id however
    /// the file was reached (#1021): the listed model at that path; else the hub-cache layout (<c>models--o--r/snapshots/rev/rel</c> ->
    /// <c>o/r/rel</c>) or the mirror layout (<c>{modelsDir}/o/r/f</c> -> the listed id of that repo's file, else <c>o/r/f</c>). A file
    /// outside the stores gets its full path - never a bare stem, which a client would persist and which could later name another file.
    /// </summary>
    /// <param name="path">The resolved GGUF path.</param>
    /// <param name="modelsDir">The flat mirror root; the default when null.</param>
    /// <param name="cacheRoot">The hub cache root; the default when null.</param>
    /// <returns>The stable id.</returns>
    public static string IdForPath(string path, string? modelsDir = null, string? cacheRoot = null)
    {
        string full;
        try { full = Path.GetFullPath(path); } catch { return path; }
        var all = EnumerateLocal(modelsDir, cacheRoot);
        var match = all.FirstOrDefault(m => string.Equals(SafeFullPath(m.FullPath), full, StringComparison.OrdinalIgnoreCase));
        if (match is not null) return IdFor(match);

        if (TryRelativeTo(full, cacheRoot ?? HubCache.CacheRoot, out var hub) && hub.Length >= 4
            && hub[0].StartsWith("models--", StringComparison.Ordinal) && hub[1] == "snapshots")
        {
            string folder = hub[0]["models--".Length..];
            int sep = folder.IndexOf("--", StringComparison.Ordinal);
            if (sep > 0)
                return $"{folder[..sep]}/{folder[(sep + 2)..]}/{string.Join('/', hub[3..])}";
        }

        if (TryRelativeTo(full, modelsDir ?? HuggingFaceDownloader.DefaultModelsDirectory, out var mirror) && mirror.Length >= 3)
        {
            string repo = $"{mirror[0]}/{mirror[1]}";
            long length = SafeLength(full);
            // The flat mirror keeps only the file name; the listed (snapshot) copy of the same file may sit in a subfolder.
            var same = all.FirstOrDefault(m => m.RepoId.Equals(repo, StringComparison.OrdinalIgnoreCase)
                && Path.GetFileName(m.Filename).Equals(mirror[^1], StringComparison.OrdinalIgnoreCase) && m.SizeBytes == length);
            return same is not null ? IdFor(same) : $"{repo}/{string.Join('/', mirror[2..])}";
        }

        return full;
    }

    private static bool TryRelativeTo(string full, string root, out string[] segments)
    {
        segments = [];
        try
        {
            string rel = Path.GetRelativePath(Path.GetFullPath(root), full);
            if (rel.StartsWith("..", StringComparison.Ordinal) || Path.IsPathRooted(rel)) return false;
            segments = rel.Split(['/', '\\'], StringSplitOptions.RemoveEmptyEntries);
            return true;
        }
        catch { return false; }
    }

    private static long SafeLength(string path)
    {
        try { return FileLength(path); } catch { return -1; }
    }

    /// <summary>True when <paramref name="name"/> addresses <paramref name="model"/> by its id, file stem, or filename.</summary>
    public static bool IsNamedBy(LocalModel model, string name) =>
        IdFor(model).Equals(name, StringComparison.OrdinalIgnoreCase)
        || model.Filename.Equals(name, StringComparison.OrdinalIgnoreCase)
        || Path.GetFileNameWithoutExtension(model.Filename).Equals(name, StringComparison.OrdinalIgnoreCase)
        || Path.GetFileName(model.Filename).Equals(name, StringComparison.OrdinalIgnoreCase);

    private static string SafeFullPath(string path)
    {
        try { return Path.GetFullPath(path); } catch { return path; }
    }

    /// <summary>Every local GGUF (mirror + hub cache), de-duplicated by repo and filename. Never throws.</summary>
    public static List<LocalModel> EnumerateLocal(string? modelsDir = null, string? cacheRoot = null, bool includeOllama = false, string? ollamaRoot = null)
    {
        var byKey = new Dictionary<string, LocalModel>(StringComparer.OrdinalIgnoreCase);
        void Add(LocalModel m)
        {
            if (Path.GetFileName(m.Filename).StartsWith("mmproj", StringComparison.OrdinalIgnoreCase)) return;   // multimodal projector, not a model
            if (SplitGguf.IsLaterShard(m.Filename)) return;   // shard 2..N is part of the set listed under shard 1 (#756)
            if (SplitGguf.IsFirstShard(m.Filename)) m = m with { SizeBytes = SplitGguf.TotalLength(m.FullPath) };
            // (#1021) One file, two listings: the flat mirror keeps only the file NAME while the hub snapshot keeps its subfolder
            // ("Q4_K_M/model-00001-of-00002.gguf"), so key on repo + name + size, not the relative path - otherwise the same file is
            // listed twice and a bare name reports it as ambiguous with itself.
            string key = m.RepoId + "|" + Path.GetFileName(m.Filename) + "|" + m.SizeBytes;
            if (!byKey.TryGetValue(key, out var have) || Preferred(m, have)) byKey[key] = m;
        }

        // The snapshot's path-preserving entry wins, so the id (owner/repo/sub/file.gguf) is the same however the file is reached;
        // between equally deep entries, the newer (unchanged behaviour).
        static bool Preferred(LocalModel candidate, LocalModel have)
        {
            bool deep = candidate.Filename.Contains('/'), haveDeep = have.Filename.Contains('/');
            return deep != haveDeep ? deep : candidate.DownloadedAt > have.DownloadedAt;
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
    /// <exception cref="AmbiguousModelReferenceException">
    /// A bare name (file stem or filename, no repo) matches more than one local model; the exception lists their ids (#1021).
    /// </exception>
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
                if (exact is not null) return exact.FullPath;
                // An id minted from the flat mirror (no subfolder) still names the one file of that name in the repo.
                var byName = inRepo.Where(m => Path.GetFileName(m.Filename).Equals(r.Filename, StringComparison.OrdinalIgnoreCase)).ToList();
                return byName.Count == 1 ? byName[0].FullPath : null;
            }
            return ChooseFile(inRepo, tag)?.FullPath;
        }

        // Bare name: a file stem (the id /v1/models reported before ids were repo-qualified), or the full filename.
        // Unlike a repo reference (one model, several quants: picking the default quant is the documented behaviour), several
        // matches here are DIFFERENT models that merely share a filename - e.g. two repos both shipping q4_k_m.gguf. Picking one
        // (it used to be the largest) loads a model the caller never named, so it is an error that lists the unambiguous ids (#1021).
        string name = r.Filename ?? arg;
        var named = Eligible(all.Where(m => IsNamedBy(m, name)), tag)
            .GroupBy(IdFor, StringComparer.OrdinalIgnoreCase).Select(g => g.First()).ToList();   // one id = one model, never ambiguous with itself
        if (named.Count > 1)
            throw new AmbiguousModelReferenceException(arg, named.Select(IdFor).ToList());
        return ChooseFile(named, tag)?.FullPath;
    }

    /// <summary>The candidates <see cref="ChooseFile"/> chooses among: shard 2..N dropped, then filtered to <paramref name="tag"/> when given.</summary>
    private static List<LocalModel> Eligible(IEnumerable<LocalModel> candidates, string? tag)
    {
        var pool = candidates.Where(m => !IsLaterShard(m.Filename));
        if (tag is not null) pool = pool.Where(m => m.Filename.Contains(tag, StringComparison.OrdinalIgnoreCase));
        return pool.ToList();
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
        var pool = Eligible(candidates, tag);
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

    private static bool IsLaterShard(string filename) => SplitGguf.IsLaterShard(filename);

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

        // A split GGUF (name-00001-of-0000M.gguf) is useless without its siblings: fetch every shard, open from shard 1 (#756).
        var result = await downloader.DownloadModelToHubCacheAsync(repo, file, cacheRoot: cacheRoot, modelsDir: modelsDir, progress: progress, cancellationToken: ct).ConfigureAwait(false);
        return result.ModelPath;
    }

    /// <summary>
    /// Deletes one local model everywhere it is linked: the mirror entry, the hub-cache snapshot entries, and the hub-cache blob - the blob only
    /// when exactly one blob in the repo matches the file (same length and first MiB), so another tool's data is never guessed at.
    /// Returns the bytes released, 0 when nothing was removed.
    /// </summary>
    public static long DeleteLocal(LocalModel model, string? modelsDir = null, string? cacheRoot = null, bool firstShardOwnSize = false)
    {
        cacheRoot ??= HubCache.CacheRoot;
        if (!firstShardOwnSize && SplitGguf.IsFirstShard(model.Filename))
        {
            // Release shards 2..N first (each is its own blob), then shard 1 with its own size.
            long released = 0;
            string dir = Path.GetDirectoryName(model.FullPath) ?? ".";
            foreach (string shard in SplitGguf.ShardNames(model.Filename).Skip(1))
            {
                string shardPath = Path.Combine(dir, Path.GetFileName(shard));
                long len = File.Exists(shardPath) ? new FileInfo(ResolveLinks(shardPath)).Length : 0;
                released += DeleteLocal(new LocalModel(model.RepoId, shard, shardPath, len, model.DownloadedAt), modelsDir, cacheRoot);
            }
            long own = File.Exists(model.FullPath) ? new FileInfo(ResolveLinks(model.FullPath)).Length : 0;
            return released + DeleteLocal(model with { SizeBytes = own }, modelsDir, cacheRoot, firstShardOwnSize: true);
        }
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
