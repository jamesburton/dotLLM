using System.Text.RegularExpressions;

namespace DotLLM.HuggingFace;

/// <summary>
/// Naming helpers for split GGUF sets (<c>name-00001-of-00003.gguf</c>, llama.cpp <c>gguf-split</c> convention; issue #756).
/// The engine opens a set through shard 1 and discovers the siblings itself, so the resolver's job is to make sure
/// every shard is present side by side and that shard 1 is the entry point.
/// </summary>
public static class SplitGguf
{
    private static readonly Regex ShardName = new(
        @"^(?<prefix>.+)-(?<no>\d{5})-of-(?<count>\d{5})\.gguf$",
        RegexOptions.IgnoreCase | RegexOptions.CultureInvariant | RegexOptions.Compiled);

    /// <summary>Parses a shard filename or relative path; false for an ordinary single file.</summary>
    public static bool TryParse(string filename, out string prefix, out int no, out int count)
    {
        prefix = ""; no = 0; count = 0;
        var m = ShardName.Match(filename);
        if (!m.Success) return false;
        prefix = m.Groups["prefix"].Value;
        no = int.Parse(m.Groups["no"].Value);
        count = int.Parse(m.Groups["count"].Value);
        return count > 1;
    }

    /// <summary>True for shard 2..N of a set.</summary>
    public static bool IsLaterShard(string filename) => TryParse(filename, out _, out int no, out _) && no > 1;

    /// <summary>True for shard 1 of a set of two or more.</summary>
    public static bool IsFirstShard(string filename) => TryParse(filename, out _, out int no, out _) && no == 1;

    /// <summary>Names (same directory prefix as <paramref name="filename"/>) of all shards of the set, shard 1 first; just the input for a single file.</summary>
    public static IReadOnlyList<string> ShardNames(string filename)
    {
        if (!TryParse(filename, out string prefix, out _, out int count)) return [filename];
        var names = new string[count];
        for (int i = 1; i <= count; i++)
            names[i - 1] = $"{prefix}-{i:D5}-of-{count:D5}.gguf";
        return names;
    }

    /// <summary>Sum of the sizes of every shard file present beside <paramref name="firstShardPath"/> (links followed); the file's own size for a single file.</summary>
    public static long TotalLength(string firstShardPath)
    {
        static long Len(string p)
        {
            try
            {
                if (File.ResolveLinkTarget(p, returnFinalTarget: true) is FileInfo t) return t.Length;
                return new FileInfo(p).Length;
            }
            catch (IOException) { return 0; }
        }

        string dir = Path.GetDirectoryName(Path.GetFullPath(firstShardPath)) ?? ".";
        string name = Path.GetFileName(firstShardPath);
        long total = 0;
        foreach (string shard in ShardNames(name)) total += Len(Path.Combine(dir, shard));
        return total;
    }
}
