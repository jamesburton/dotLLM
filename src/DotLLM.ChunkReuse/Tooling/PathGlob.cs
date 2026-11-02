using System.Text;
using System.Text.RegularExpressions;

namespace DotLLM.ChunkReuse.Tooling;

/// <summary>
/// A path glob over corpus paths (relative, '/' separated, as <see cref="CodeCorpus.SourceFile.Path"/>).
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item><c>*</c> matches any characters inside one path segment, <c>?</c> one character other than '/', <c>**</c> as a whole
/// segment any number of segments (zero included). Every other character is literal (<c>[</c>, <c>{</c>, <c>+</c>, <c>(</c>…).</item>
/// <item>A pattern with no '/' matches the file name at any depth (<c>*.cs</c> = <c>**/*.cs</c>); a trailing '/' means
/// everything below that directory (<c>src/App/</c> = <c>src/App/**</c>).</item>
/// <item>A leading <c>!</c> negates (see <see cref="SelectionRequest.Paths"/>).</item>
/// <item>'\' is a separator (Windows), <c>.</c> and empty segments are dropped. Patterns that are rooted (<c>/x</c>,
/// <c>C:\x</c>, <c>c:x</c>, UNC) or contain a <c>..</c> segment are rejected: globs are relative to the corpus root and never
/// leave it. For the same reason a path outside the root (<c>../x</c>, or rooted, as a scan can produce for linked files) never
/// matches any glob.</item>
/// <item>Case-insensitive on Windows, ordinal elsewhere (as <see cref="Graph.RoslynGraphBuilder.PathComparer"/>).</item>
/// <item>Matching uses a non-backtracking regex, so a pattern cannot make it slow.</item>
/// </list>
/// </remarks>
public sealed class PathGlob
{
    private readonly Regex _regex;

    private PathGlob(string pattern, bool negated, Regex regex)
    {
        Pattern = pattern;
        Negated = negated;
        _regex = regex;
    }

    /// <summary>The normalised pattern (without <c>!</c>; '/' separated).</summary>
    public string Pattern { get; }

    /// <summary>True for a <c>!pattern</c>.</summary>
    public bool Negated { get; }

    /// <summary>Parses <paramref name="pattern"/>.</summary>
    /// <exception cref="ArgumentException">Empty, rooted, or with a <c>..</c> segment.</exception>
    public static PathGlob Parse(string pattern)
    {
        ArgumentNullException.ThrowIfNull(pattern);
        string p = pattern.Trim();
        bool negated = p.StartsWith('!');
        if (negated) p = p[1..].TrimStart();
        p = p.Replace('\\', '/');
        if (p.Length == 0 || p.Trim('/', '.').Length == 0)
            throw new ArgumentException($"Empty path glob '{pattern}'.", nameof(pattern));
        if (p.StartsWith('/') || (p.Length >= 2 && p[1] == ':') || Path.IsPathRooted(p))
            throw new ArgumentException($"Path glob '{pattern}' is rooted; globs are relative to the corpus root.", nameof(pattern));

        bool directory = p.EndsWith('/');
        var segments = p.Split('/', StringSplitOptions.RemoveEmptyEntries).Where(s => s != ".").ToList();
        if (segments.Contains(".."))
            throw new ArgumentException($"Path glob '{pattern}' has a '..' segment; globs never leave the corpus root.", nameof(pattern));
        if (segments.Count == 0)
            throw new ArgumentException($"Empty path glob '{pattern}'.", nameof(pattern));
        if (directory) segments.Add("**");
        else if (segments.Count == 1 && segments[0] != "**") segments.Insert(0, "**");

        var sb = new StringBuilder("^");
        for (int i = 0; i < segments.Count; i++)
        {
            bool last = i == segments.Count - 1;
            string seg = segments[i];
            if (seg == "**")
            {
                sb.Append(last ? ".*" : "(?:[^/]+/)*");
                continue;
            }
            foreach (char c in seg)
            {
                sb.Append(c switch
                {
                    '*' => "[^/]*",
                    '?' => "[^/]",
                    _ => Regex.Escape(c.ToString()),
                });
            }
            if (!last) sb.Append('/');
        }
        sb.Append('$');
        RegexOptions options = RegexOptions.CultureInvariant | RegexOptions.NonBacktracking;
        if (OperatingSystem.IsWindows()) options |= RegexOptions.IgnoreCase;
        return new PathGlob(string.Join('/', segments), negated, new Regex(sb.ToString(), options));
    }

    /// <summary>True when <paramref name="path"/> (relative; '\' accepted) matches. A path outside the root never matches.</summary>
    public bool IsMatch(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        string p = path.Replace('\\', '/');
        if (IsOutsideRoot(p)) return false;
        return _regex.IsMatch(p);
    }

    /// <summary>True for a rooted path or one that starts with a <c>..</c> segment.</summary>
    internal static bool IsOutsideRoot(string slashPath) =>
        slashPath.StartsWith('/') || (slashPath.Length >= 2 && slashPath[1] == ':') || Path.IsPathRooted(slashPath)
        || slashPath == ".." || slashPath.StartsWith("../", StringComparison.Ordinal);

    /// <inheritdoc/>
    public override string ToString() => (Negated ? "!" : "") + Pattern;
}
