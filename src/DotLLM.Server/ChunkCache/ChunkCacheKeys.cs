using System.Security.Cryptography;
using System.Text;
using DotLLM.ChunkReuse.Store;

namespace DotLLM.Server.ChunkCache;

/// <summary>
/// The API keys of <c>--chunk-cache-api-keys &lt;file&gt;</c>: who may use the chunk-cache control plane
/// (<c>/v1/chunk-cache/*</c>) and in which store namespace. This class keeps SHA-256 digests of the keys only, never the
/// key text, and never prints either. (Elsewhere in the server the raw presented key does live in memory: the rate
/// limiter partitions its buckets by it.)
/// </summary>
/// <remarks>
/// <para>
/// <b>File format.</b> UTF-8 text, one key per line. Blank lines are skipped, and so is every line whose first non-blank
/// character is <c>#</c> (a comment): a key can therefore never start with <c>#</c>. Each line is trimmed (spaces and
/// tabs) and split on spaces / tabs: the key, then optionally (in any order, each at most once) the role word
/// <c>admin</c> (store-wide figures and operations, from a later task) and <c>ns=&lt;namespace&gt;</c>.
/// <list type="bullet">
/// <item>A key is printable ASCII without spaces (<c>!</c> to <c>~</c>); any other character on a line (a non-ASCII letter,
/// a control character, a no-break space) refuses the file.</item>
/// <item>A key shorter than <see cref="MinLength"/> characters refuses the file unless <c>--chunk-cache-allow-short-keys</c>,
/// which accepts it with a warning: a namespace derived from a key appears in the store's file names, so a short key can be
/// recovered offline from them.</item>
/// <item><c>ns=&lt;namespace&gt;</c> maps the key to that namespace instead of <see cref="ChunkNamespace.ForApiKey"/>:
/// <see cref="ChunkNamespace.IsValid"/> and not empty (so never the reserved <c>default</c>). Two keys may name the same
/// namespace on purpose (a team, or a rotated key keeping the old key's entries). A named namespace is not derived from
/// the key, so it says nothing about the key.</item>
/// <item>Repeated keys, an unknown field and a file with no key are refused.</item>
/// </list>
/// Errors name the file and the line number, never the key or the namespace.
/// </para>
/// <para>
/// <b>Matching</b> (<see cref="TryMatch(string, out bool, out string?)"/>) is constant-time in the key text: the presented
/// key is hashed and its digest is compared with every configured digest by
/// <see cref="CryptographicOperations.FixedTimeEquals"/>, without stopping at a match, so neither the position of a
/// matching key nor a common prefix shows in the time taken.
/// </para>
/// </remarks>
public sealed class ChunkCacheKeys
{
    /// <summary>Keys shorter than this many characters refuse startup unless <c>--chunk-cache-allow-short-keys</c>.</summary>
    public const int MinLength = 32;

    /// <summary>The role word that marks an admin key.</summary>
    public const string AdminRole = "admin";

    /// <summary>The prefix of the explicit-namespace field.</summary>
    public const string NamespaceField = "ns=";

    private readonly byte[][] _digests;
    private readonly bool[] _admin;
    private readonly string?[] _namespaces;

    private ChunkCacheKeys(byte[][] digests, bool[] admin, string?[] namespaces, IReadOnlyList<string> warnings)
    {
        _digests = digests;
        _admin = admin;
        _namespaces = namespaces;
        Warnings = warnings;
    }

    /// <summary>Configured keys.</summary>
    public int Count => _digests.Length;

    /// <summary>Configured admin keys.</summary>
    public int AdminCount => _admin.Count(a => a);

    /// <summary>Startup warnings (short keys accepted by the override), naming line numbers only.</summary>
    public IReadOnlyList<string> Warnings { get; }

    /// <summary>Reads and parses a keys file.</summary>
    /// <param name="path">The file.</param>
    /// <param name="allowShortKeys"><c>--chunk-cache-allow-short-keys</c>: accept keys under <see cref="MinLength"/> characters, with a warning.</param>
    /// <returns>The keys.</returns>
    /// <exception cref="ArgumentException">The file cannot be read, or a line is invalid (the message names the line, never a key).</exception>
    public static ChunkCacheKeys Load(string path, bool allowShortKeys = false)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        string[] lines;
        try
        {
            lines = File.ReadAllLines(path, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: false));
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException
                                       or System.Security.SecurityException or ArgumentException)
        {
            throw new ArgumentException($"--chunk-cache-api-keys: cannot read '{path}' ({ex.GetType().Name}).", nameof(path), ex);
        }
        return Parse(lines, path, allowShortKeys);
    }

    /// <summary>Parses the lines of a keys file (see the class remarks for the format).</summary>
    /// <param name="lines">The file's lines.</param>
    /// <param name="source">The file name, for messages.</param>
    /// <param name="allowShortKeys">Accept keys under <see cref="MinLength"/> characters, with a warning; else refuse them.</param>
    /// <returns>The keys.</returns>
    /// <exception cref="ArgumentException">A line is invalid, a key is short (without the override) or repeats, or there is no key.</exception>
    public static ChunkCacheKeys Parse(IReadOnlyList<string> lines, string source, bool allowShortKeys)
    {
        ArgumentNullException.ThrowIfNull(lines);
        var digests = new List<byte[]>();
        var admin = new List<bool>();
        var namespaces = new List<string?>();
        var warnings = new List<string>();
        for (int i = 0; i < lines.Count; i++)
        {
            int lineNo = i + 1;
            string raw = lines[i] ?? "";
            // Refuse before trimming: a non-ASCII space (U+00A0, U+3000) must not be trimmed away silently.
            foreach (char c in raw)
            {
                if (c != '\t' && (c < 0x20 || c > 0x7E))
                    throw new ArgumentException($"--chunk-cache-api-keys: {source}, line {lineNo}: a key must be printable ASCII; this line has another character (the key is not shown).");
            }
            string line = raw.Trim(' ', '\t');
            if (line.Length == 0 || line[0] == '#')
                continue;

            string[] fields = line.Split([' ', '\t'], StringSplitOptions.RemoveEmptyEntries);
            string key = fields[0];
            bool isAdmin = false;
            string? ns = null;
            for (int f = 1; f < fields.Length; f++)
            {
                if (string.Equals(fields[f], AdminRole, StringComparison.Ordinal) && !isAdmin)
                {
                    isAdmin = true;
                }
                else if (fields[f].StartsWith(NamespaceField, StringComparison.Ordinal) && ns is null)
                {
                    ns = fields[f][NamespaceField.Length..];
                    if (ns.Length == 0 || !ChunkNamespace.IsValid(ns))
                        throw new ArgumentException($"--chunk-cache-api-keys: {source}, line {lineNo}: {NamespaceField} must be 1-{ChunkNamespace.MaxLength} characters of [a-z0-9._-] other than '{ChunkNamespace.ReservedDefaultName}'.");
                }
                else
                {
                    throw new ArgumentException($"--chunk-cache-api-keys: {source}, line {lineNo}: expected '<key> [{AdminRole}] [{NamespaceField}<namespace>]' (a key contains no spaces; each field at most once).");
                }
            }

            byte[] digest = Digest(key);
            if (digests.Any(d => CryptographicOperations.FixedTimeEquals(d, digest)))
                throw new ArgumentException($"--chunk-cache-api-keys: {source}, line {lineNo}: repeats an earlier key.");
            if (key.Length < MinLength)
            {
                if (!allowShortKeys)
                    throw new ArgumentException($"--chunk-cache-api-keys: {source}, line {lineNo}: key shorter than {MinLength} characters (guessable offline from the store's file names; use 32+ random bytes, or start with --chunk-cache-allow-short-keys).");
                warnings.Add($"--chunk-cache-api-keys: {source}, line {lineNo}: key shorter than {MinLength} characters, accepted because of --chunk-cache-allow-short-keys (guessable offline from the store's file names).");
            }
            digests.Add(digest);
            admin.Add(isAdmin);
            namespaces.Add(ns);
        }
        if (digests.Count == 0)
            throw new ArgumentException($"--chunk-cache-api-keys: {source} contains no key.");
        return new ChunkCacheKeys([.. digests], [.. admin], [.. namespaces], warnings);
    }

    /// <summary>Whether <paramref name="presented"/> (trimmed) is a configured key (see the overload).</summary>
    /// <param name="presented">The key from the request.</param>
    /// <param name="isAdmin">Whether the matching key is an admin key.</param>
    /// <returns>Whether it matches.</returns>
    public bool TryMatch(string presented, out bool isAdmin) => TryMatch(presented, out isAdmin, out _);

    /// <summary>
    /// Whether <paramref name="presented"/> (trimmed) is a configured key. Constant-time in the key text (see the class
    /// remarks).
    /// </summary>
    /// <param name="presented">The key from the request.</param>
    /// <param name="isAdmin">Whether the matching key is an admin key.</param>
    /// <param name="namespace">The matching key's explicit namespace (<c>ns=</c>), or null when it uses the derived one.</param>
    /// <returns>Whether it matches.</returns>
    public bool TryMatch(string presented, out bool isAdmin, out string? @namespace)
    {
        ArgumentNullException.ThrowIfNull(presented);
        byte[] digest = Digest(presented.Trim(' ', '\t'));
        int match = -1;
        for (int i = 0; i < _digests.Length; i++)
        {
            bool equal = CryptographicOperations.FixedTimeEquals(_digests[i], digest);
            match = equal ? i : match;
        }
        isAdmin = match >= 0 && _admin[match];
        @namespace = match >= 0 ? _namespaces[match] : null;
        return match >= 0;
    }

    private static byte[] Digest(string key) => SHA256.HashData(Encoding.UTF8.GetBytes(key));

    /// <inheritdoc />
    public override string ToString() => $"ChunkCacheKeys {{ Count = {Count}, AdminCount = {AdminCount} }}";
}
