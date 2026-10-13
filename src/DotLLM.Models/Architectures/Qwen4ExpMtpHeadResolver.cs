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
    /// Smallest trunk (summed GGUF bytes) for which an auto-detected head is attached; 0 = every trunk. Before the 2..8-row GEMVs for the K-quant / IQ dense
    /// projections (#902) the ISTA GSQ-RCO files (37-44 GiB) lost to plain decode with the head (verify rows cost 100+ ms at 2 rows), so this was 50 GiB; with them
    /// every measured file wins at K = 3-4 (see <c>docs/SPECULATIVE.md</c>), and the adaptive gate still falls back per request. The head costs ~2 GiB of device
    /// memory, so a trunk that leaves no room declines it at attach (<c>EnsureDeviceHeadroom</c>). Override with <c>DOTLLM_MTP_AUTO_ATTACH_MIN_GIB</c>.
    /// </summary>
    public static long AutoAttachMinTrunkBytes { get; } =
        double.TryParse(Environment.GetEnvironmentVariable("DOTLLM_MTP_AUTO_ATTACH_MIN_GIB"), System.Globalization.NumberStyles.Float,
            System.Globalization.CultureInfo.InvariantCulture, out double gib) && gib >= 0 ? (long)(gib * (1L << 30)) : 0;

    /// <summary>True when an auto-detected head is worth attaching to a trunk of <paramref name="trunkBytes"/> (see <see cref="AutoAttachMinTrunkBytes"/>).</summary>
    public static bool ShouldAutoAttach(long trunkBytes) => trunkBytes >= AutoAttachMinTrunkBytes;

    /// <summary>
    /// Attaches the MTP head to <paramref name="model"/> when it is a qwen4exp model (CPU oracle or Vulkan) without a head yet.
    /// </summary>
    /// <param name="model">The loaded model; any other type is left alone.</param>
    /// <param name="modelPath">Trunk path used for auto-detection.</param>
    /// <param name="explicitPath">An explicit head path (<c>--mtp-head</c>), which skips detection; null for auto-detection.</param>
    /// <param name="trunkBytes">Summed GGUF bytes of the trunk; when given (&gt;= 0) an auto-detected head is attached only if
    /// <see cref="ShouldAutoAttach"/> (an explicit path always attaches). -1 = no size rule.</param>
    /// <returns>The attached head's path, or null when nothing was attached.</returns>
    /// <exception cref="FileNotFoundException"><paramref name="explicitPath"/> does not exist.</exception>
    public static string? TryAttach(IModel model, string modelPath, string? explicitPath = null, long trunkBytes = -1)
    {
        if (model is not IMtpHeadAttachable q4 || q4.HasMtpHead) return null;
        if (!q4.CanAttachMtpHead)
        {
            if (explicitPath is not null)
                throw new NotSupportedException("An MTP head cannot be attached while the routed experts are tiered (DOTLLM_VK_EXPERT_SLOTS): a verify window of 5 rows touches 31-33 experts per layer.");
            return null;
        }
        string? path = explicitPath;
        if (path is not null)
        {
            if (!File.Exists(path)) throw new FileNotFoundException("MTP head GGUF not found.", path);
        }
        else
        {
            if (trunkBytes >= 0 && !ShouldAutoAttach(trunkBytes)) return null;
            path = Find(modelPath);
            if (path is null) return null;
        }
        q4.AttachMtpHead(path);
        return path;
    }
}
