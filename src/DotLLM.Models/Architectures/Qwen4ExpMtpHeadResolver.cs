using DotLLM.Core.Models;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Finds and attaches the qwen4exp MTP draft head (issue #820). The released checkpoint ships the head as a SEPARATE GGUF
/// (<c>mtp-Qwen3.8-Flash-Next-Q8_0.gguf</c>, 4.1 GB) beside the quantisation folders of the trunk, so the usual case is an
/// auto-detected sibling file; an explicit path always wins.
/// </summary>
public static class Qwen4ExpMtpHeadResolver
{
    /// <summary>
    /// Looks for an <c>mtp-*.gguf</c> next to <paramref name="modelPath"/>: first in its directory, then in the directory above it
    /// (the HF snapshot layout puts the head at the snapshot root and the shards in a quantisation sub-folder). With several
    /// candidates the largest file wins (the highest-precision head).
    /// </summary>
    /// <param name="modelPath">Path of the trunk GGUF (any shard) or its directory.</param>
    /// <returns>The head's path, or null when none is found.</returns>
    public static string? Find(string modelPath)
    {
        if (string.IsNullOrEmpty(modelPath)) return null;
        string? dir = Directory.Exists(modelPath) ? modelPath : Path.GetDirectoryName(Path.GetFullPath(modelPath));
        for (int up = 0; up < 2 && dir is not null; up++, dir = Path.GetDirectoryName(dir))
        {
            if (!Directory.Exists(dir)) continue;
            string? best = null;
            long bestSize = -1;
            foreach (string f in Directory.EnumerateFiles(dir, "mtp-*.gguf"))
            {
                long size = new FileInfo(f).Length;
                if (size > bestSize) { best = f; bestSize = size; }
            }
            if (best is not null) return best;
        }
        return null;
    }

    /// <summary>
    /// Attaches the MTP head to <paramref name="model"/> when it is a CPU <see cref="Qwen4ExpTransformerModel"/> without a head yet.
    /// </summary>
    /// <param name="model">The loaded model; any other type is left alone.</param>
    /// <param name="modelPath">Trunk path used for auto-detection.</param>
    /// <param name="explicitPath">An explicit head path (<c>--mtp-head</c>), which skips detection; null for auto-detection.</param>
    /// <returns>The attached head's path, or null when nothing was attached.</returns>
    /// <exception cref="FileNotFoundException"><paramref name="explicitPath"/> does not exist.</exception>
    public static string? TryAttach(IModel model, string modelPath, string? explicitPath = null)
    {
        if (model is not Qwen4ExpTransformerModel q4 || q4.SupportsMtp) return null;
        string? path = explicitPath;
        if (path is not null)
        {
            if (!File.Exists(path)) throw new FileNotFoundException("MTP head GGUF not found.", path);
        }
        else
        {
            path = Find(modelPath);
            if (path is null) return null;
        }
        q4.AttachMtpHead(path);
        return path;
    }
}
