namespace DotLLM.ChunkReuse.Tooling;

/// <summary>
/// The secrets deny-list (integration spec §10): paths whose text is never chunked by <c>build</c>, never encoded by
/// prewarm, and never admitted for a background encode by the server when an annotation names the path in
/// <c>source.path</c>. The default list is <see cref="DefaultPatterns"/>.
/// </summary>
/// <remarks>
/// <list type="bullet">
/// <item>A pattern ending in <c>/**</c> denies every path with a <b>directory</b> segment matching what precedes it
/// (<c>.git/**</c>: anything inside a <c>.git</c> directory, at any depth).</item>
/// <item>Any other pattern is matched against the file name (the last segment) at any depth: <c>*</c> matches any run of
/// characters, <c>?</c> one character; everything else is literal.</item>
/// <item>Matching is always case-insensitive (ordinal ignore-case), on every platform: a deny-list errs towards denying.
/// Both '/' and '\' separate segments; empty and <c>.</c> segments are ignored. Rooted paths and <c>..</c> segments are
/// matched as given (never rejected: a deny-list must not be bypassed by an unusual spelling).</item>
/// </list>
/// The server cannot see files: it matches only the <c>source.path</c> a client sends, and trusts <c>no_store</c>.
/// </remarks>
public sealed class ChunkDenyList
{
    /// <summary>
    /// The default patterns: spec §10 widened by the Task 15 review ruling (every <c>appsettings*.json</c>, key stores,
    /// credential dot-files, <c>.ssh/</c>, <c>secrets.*</c>, <c>*.env</c>, <c>credentials*</c>).
    /// </summary>
    public static readonly IReadOnlyList<string> DefaultPatterns =
    [
        ".env*", "*.env", "*.pem", "*.key", "*.pfx", "*.p12", "id_*", "id_rsa*", "secrets.*", "appsettings*.json",
        "credentials*", ".npmrc", ".netrc", ".pgpass", ".git/**", ".ssh/**",
    ];

    /// <summary>The default deny-list.</summary>
    public static ChunkDenyList Default { get; } = new(DefaultPatterns);

    private readonly string[] _fileNamePatterns;
    private readonly string[] _directoryPatterns;

    /// <summary>Creates a deny-list from patterns (see the class remarks).</summary>
    /// <param name="patterns">The patterns; blank entries are ignored.</param>
    /// <exception cref="ArgumentException">A pattern contains a separator other than a trailing <c>/**</c>.</exception>
    public ChunkDenyList(IEnumerable<string> patterns)
    {
        ArgumentNullException.ThrowIfNull(patterns);
        var files = new List<string>();
        var dirs = new List<string>();
        foreach (string raw in patterns)
        {
            if (string.IsNullOrWhiteSpace(raw)) continue;
            string p = raw.Trim();
            if (p.EndsWith("/**", StringComparison.Ordinal))
            {
                string dir = p[..^3];
                if (dir.Length == 0 || dir.IndexOfAny(['/', '\\']) >= 0)
                    throw new ArgumentException($"Deny pattern '{p}': a directory pattern is one segment followed by '/**'.", nameof(patterns));
                dirs.Add(dir);
            }
            else
            {
                if (p.IndexOfAny(['/', '\\']) >= 0)
                    throw new ArgumentException($"Deny pattern '{p}': a file-name pattern has no separator.", nameof(patterns));
                files.Add(p);
            }
        }
        _fileNamePatterns = [.. files];
        _directoryPatterns = [.. dirs];
    }

    /// <summary>
    /// Whether chunk text looks like a secret by content: a PEM private-key header (<c>-----BEGIN </c> ... <c>PRIVATE
    /// KEY-----</c> on one line, e.g. RSA, EC, OPENSSH, ENCRYPTED). Certificates and public keys are not secrets.
    /// </summary>
    /// <param name="text">The chunk text.</param>
    /// <returns>True when it should never be stored.</returns>
    public static bool LooksLikeSecret(string? text)
    {
        if (string.IsNullOrEmpty(text)) return false;
        int at = 0;
        while ((at = text.IndexOf("-----BEGIN ", at, StringComparison.Ordinal)) >= 0)
        {
            int eol = text.IndexOf('\n', at);
            int end = eol < 0 ? text.Length : eol;
            int key = text.IndexOf("PRIVATE KEY-----", at, end - at, StringComparison.Ordinal);
            if (key >= 0) return true;
            at = end;
        }
        return false;
    }

    /// <summary>Whether <paramref name="path"/> is denied.</summary>
    /// <param name="path">A path (relative or not, either separator).</param>
    /// <returns>True when a pattern matches; false for a null or empty path (nothing to match).</returns>
    public bool IsDenied(string? path)
    {
        if (string.IsNullOrEmpty(path)) return false;
        if (path.Contains("::", StringComparison.Ordinal)) return true; // an alternate data stream: never a plain file
        var segments = new List<string>();
        foreach (string raw in path.Split(['/', '\\'], StringSplitOptions.RemoveEmptyEntries))
        {
            if (raw is "." or "..")
            {
                if (raw == "..") segments.Add(raw);
                continue;
            }
            string seg = raw;
            int colon = seg.IndexOf(':');
            if (colon >= 0) seg = seg[..colon]; // "secrets.json:stream" (or a drive letter)
            seg = seg.TrimEnd('.', ' ');        // Windows ignores trailing dots and spaces
            if (seg.Length > 0) segments.Add(seg);
        }
        if (segments.Count == 0) return false;
        string name = segments[^1];
        foreach (string p in _fileNamePatterns)
            if (Matches(p, name)) return true;
        for (int i = 0; i < segments.Count - 1; i++)
            foreach (string p in _directoryPatterns)
                if (Matches(p, segments[i])) return true;
        for (int i = 0; i < segments.Count; i++)
            if (ShortNameMayBeDenied(segments[i], i == segments.Count - 1)) return true;
        return false;
    }

    // An 8.3 short name ("SECRET~1.JSO"): denied when its stem and a pattern's literal start (leading '.' dropped, as the
    // short name drops it) could be the same name. Errs towards denying.
    private bool ShortNameMayBeDenied(string segment, bool isFileName)
    {
        int tilde = segment.IndexOf('~');
        if (tilde < 1 || tilde + 1 >= segment.Length || !char.IsAsciiDigit(segment[tilde + 1])) return false;
        string stem = segment[..tilde].TrimStart('.');
        if (stem.Length == 0) return false;
        IEnumerable<string> patterns = isFileName ? _fileNamePatterns.Concat(_directoryPatterns) : _directoryPatterns;
        foreach (string p in patterns)
        {
            int wild = p.IndexOfAny(['*', '?']);
            string literal = (wild < 0 ? p : p[..wild]).TrimStart('.');
            if (literal.Length == 0) continue;
            if (literal.StartsWith(stem, StringComparison.OrdinalIgnoreCase) || stem.StartsWith(literal, StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    // Glob over one segment: '*' any run, '?' one char, the rest literal (case-insensitive). Iterative with one backtrack point.
    private static bool Matches(string pattern, string text)
    {
        int p = 0, t = 0, star = -1, mark = 0;
        while (t < text.Length)
        {
            if (p < pattern.Length && (pattern[p] == '?' || char.ToUpperInvariant(pattern[p]) == char.ToUpperInvariant(text[t])))
            {
                p++;
                t++;
            }
            else if (p < pattern.Length && pattern[p] == '*')
            {
                star = p++;
                mark = t;
            }
            else if (star >= 0)
            {
                p = star + 1;
                t = ++mark;
            }
            else
            {
                return false;
            }
        }
        while (p < pattern.Length && pattern[p] == '*') p++;
        return p == pattern.Length;
    }
}
