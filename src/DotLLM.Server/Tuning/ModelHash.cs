using System.Security.Cryptography;
using System.Text;
using DotLLM.Models.Gguf;

namespace DotLLM.Server.Tuning;

/// <summary>
/// Identity of the model a tuning result belongs to (#1902): a GGUF file's content hash, the blob id of a Hugging Face hub-cache file, or
/// a digest over a safetensors directory.
/// </summary>
public static class ModelHash
{
    /// <summary>The hash of a model path: a GGUF file, a hub-cache file, or a safetensors directory.</summary>
    /// <param name="modelPath">File or directory.</param>
    /// <returns>A stable identifier; changes when the content changes.</returns>
    public static string Of(string modelPath)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelPath);
        if (Directory.Exists(modelPath)) return OfDirectory(modelPath);
        return OfFile(modelPath);
    }

    /// <summary>
    /// Digest of the load-time modifiers that change what the weights compute: a tensor overlay, ternary adapters, a separate MTP head and
    /// a DFlash drafter (#1951). Null when none is set. Per-request LoRA is not included.
    /// </summary>
    /// <returns>The digest, or null.</returns>
    public static string? Modifiers(string? overlayPath, IReadOnlyList<string>? ternaryAdapters, string? mtpHeadPath, string? dflashPath)
    {
        var parts = new List<string>();
        if (!string.IsNullOrWhiteSpace(overlayPath)) parts.Add("overlay\t" + OfFile(overlayPath));
        if (ternaryAdapters is not null)
            foreach (string a in ternaryAdapters.Where(a => !string.IsNullOrWhiteSpace(a))) parts.Add("ternary\t" + OfFile(a));
        if (!string.IsNullOrWhiteSpace(mtpHeadPath)) parts.Add("mtp\t" + OfFile(mtpHeadPath));
        if (!string.IsNullOrWhiteSpace(dflashPath)) parts.Add("dflash\t" + OfFile(dflashPath));
        if (parts.Count == 0) return null;
        parts.Sort(StringComparer.Ordinal);
        return Sha256Hex(string.Join('\n', parts));
    }

    private static string OfFile(string path)
    {
        if (TryHubBlobId(path) is { } id) return "hf:" + id;   // first: BaseHash resolves links, and Windows refuses some snapshot symlinks
        return DotLLM.Models.Gguf.TensorOverlay.BaseHash.Compute(path);
    }

    private static string OfDirectory(string dir)
    {
        var lines = new List<string>();
        foreach (string f in Directory.EnumerateFiles(dir))
        {
            string name = Path.GetFileName(f);
            bool wanted = name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
                          || name.Equals("config.json", StringComparison.OrdinalIgnoreCase)
                          || name.StartsWith("tokenizer", StringComparison.OrdinalIgnoreCase);
            if (!wanted) continue;
            lines.Add($"{name}\t{new FileInfo(f).Length}\t{OfFile(f)}");
        }
        lines.Sort(StringComparer.Ordinal);
        return Sha256Hex(string.Join('\n', lines));
    }

    /// <summary>
    /// The blob id of a Hugging Face hub-cache file, read from the directory layout without hashing: a file under <c>blobs/</c> is its own
    /// id; a <c>snapshots/&lt;rev&gt;/...</c> entry is a link to a blob (read, not followed), or a hardlink or copy that is matched to a
    /// blob of the same length and modification time. Null when the path is not in a hub cache or no blob is found.
    /// </summary>
    internal static string? TryHubBlobId(string path)
    {
        try
        {
            var file = new FileInfo(path);
            string full = file.FullName;
            string[] segs = full.Split(Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar);
            int repo = Array.FindLastIndex(segs, s => s.StartsWith("models--", StringComparison.Ordinal)
                                                        || s.StartsWith("datasets--", StringComparison.Ordinal));
            if (repo < 0 || repo + 1 >= segs.Length) return null;
            string kind = segs[repo + 1];
            if (kind == "blobs" && repo + 2 < segs.Length) return segs[^1];

            if (kind != "snapshots") return null;
            if (file.LinkTarget is { } target)
            {
                string leaf = Path.GetFileName(target.TrimEnd('/', '\\'));
                if (IsBlobName(leaf)) return leaf;
            }
            var repoDir = file.Directory;
            while (repoDir is not null && !repoDir.Name.StartsWith("models--", StringComparison.Ordinal)
                   && !repoDir.Name.StartsWith("datasets--", StringComparison.Ordinal)) repoDir = repoDir.Parent;
            if (repoDir is null) return null;
            string blobs = Path.Combine(repoDir.FullName, "blobs");
            if (!Directory.Exists(blobs)) return null;
            var match = new DirectoryInfo(blobs).EnumerateFiles()
                .Where(b => b.Length == file.Length && b.LastWriteTimeUtc == file.LastWriteTimeUtc && IsBlobName(b.Name))
                .Take(2).ToList();
            return match.Count == 1 ? match[0].Name : null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            return null;
        }
    }

    private static bool IsBlobName(string n) =>
        (n.Length == 64 || n.Length == 40) && n.All(Uri.IsHexDigit);

    internal static string Sha256Hex(string text) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();
}
