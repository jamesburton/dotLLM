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
        "credentials*", ".npmrc", ".netrc", ".pgpass", "*.keystore", "*.jks", "kubeconfig*", "*.tfvars", "*.tfstate",
        ".git/**", ".ssh/**", ".kube/**", ".docker/**",
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
    /// Whether chunk text looks like a secret by content: a PEM armour line (<c>-----BEGIN </c> or <c>-----END </c>) that
    /// says <c>PRIVATE KEY</c> anywhere on it (RSA, EC, OPENSSH, ENCRYPTED, PGP <c>PRIVATE KEY BLOCK</c>). Certificates and
    /// public keys are not secrets. Known limit: a key body the chunker split into a chunk with neither armour line (base64
    /// only) is not detected; harnesses set <c>no_store</c>, and the path deny-list covers key files.
    /// </summary>
    /// <param name="text">The chunk text.</param>
    /// <returns>True when it should never be stored.</returns>
    public static bool LooksLikeSecret(string? text)
    {
        if (string.IsNullOrEmpty(text)) return false;
        int at = 0;
        while ((at = text.IndexOf("-----", at, StringComparison.Ordinal)) >= 0)
        {
            int eol = text.IndexOf('\n', at);
            int end = eol < 0 ? text.Length : eol;
            var line = text.AsSpan(at, end - at);
            if ((line.StartsWith("-----BEGIN ", StringComparison.Ordinal) || line.StartsWith("-----END ", StringComparison.Ordinal))
                && line.Contains("PRIVATE KEY", StringComparison.Ordinal))
                return true;
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
        // Each segment has one spelling, or two when it holds a colon: the text before it ("secrets.json:stream") and the
        // text after it (a drive-relative "C:secrets.json"). Every spelling is checked (errs towards denying).
        var segments = new List<string[]>();
        foreach (string raw in path.Split(['/', '\\'], StringSplitOptions.RemoveEmptyEntries))
        {
            if (raw is "." or "..")
            {
                if (raw == "..") segments.Add([raw]);
                continue;
            }
            int colon = raw.IndexOf(':');
            string[] spellings = colon < 0 ? [raw] : [raw[..colon], raw[(colon + 1)..]];
            string[] kept = spellings.Select(x => x.TrimEnd('.', ' ')) // Windows ignores trailing dots and spaces
                .Where(x => x.Length > 0).ToArray();
            if (kept.Length > 0) segments.Add(kept);
        }
        if (segments.Count == 0) return false;
        foreach (string name in segments[^1])
            foreach (string p in _fileNamePatterns)
                if (Matches(p, name)) return true;
        for (int i = 0; i < segments.Count - 1; i++)
            foreach (string seg in segments[i])
                foreach (string p in _directoryPatterns)
                    if (Matches(p, seg)) return true;
        for (int i = 0; i < segments.Count; i++)
            foreach (string seg in segments[i])
                if (ShortNameMayBeDenied(seg, i == segments.Count - 1)) return true;
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
        // NTFS's hashed form after four collisions ("SE1A2B~1.JSO": two characters and four hex digits): the stem says
        // nothing, so deny when the short extension could be a denied extension.
        if (isFileName && stem.Length == 6 && stem[2..].All(char.IsAsciiHexDigit))
        {
            int dot = segment.LastIndexOf('.');
            string ext = dot > tilde ? segment[(dot + 1)..] : "";
            if (ext.Length > 0)
            {
                foreach (string p in _fileNamePatterns)
                {
                    int pdot = p.LastIndexOf('.');
                    string pext = pdot < 0 ? "" : p[(pdot + 1)..];
                    if (pext.Length > 0 && pext.IndexOfAny(['*', '?']) < 0 && pext.StartsWith(ext, StringComparison.OrdinalIgnoreCase))
                        return true;
                }
            }
        }
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
