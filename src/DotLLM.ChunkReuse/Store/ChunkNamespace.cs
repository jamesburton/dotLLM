using System.Security.Cryptography;
using System.Text;

namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// Store partitions (<see cref="ChunkKey.Create"/>'s <c>namespace</c>). The namespace is hashed into every key, so entries written in
/// one namespace are never hits in another, and an index (whose producer identity is the key of no tokens) saved under one
/// namespace loads empty under another. A server with any API key, or more than one client, uses one namespace per key
/// (<see cref="ForApiKey"/>); a namespace-free store (<see cref="Default"/>) is for single-user local servers only.
/// </summary>
public static class ChunkNamespace
{
    /// <summary>The single-user namespace. It is hashed like any other (an empty length-prefixed string), not skipped.</summary>
    public const string Default = "";

    private const int DigestHexChars = 16;

    /// <summary>Longest valid namespace, in characters (all ASCII, so also in bytes).</summary>
    public const int MaxLength = 64;

    /// <summary>
    /// The one reserved name, <c>default</c>: it stands for <see cref="Default"/> in index file names
    /// (<see cref="ChunkIndex.DefaultPath(string, string, string, string, string)"/>), so no namespace may be called that (else the
    /// default namespace and a namespace named <c>default</c> would share, and clobber, one index file).
    /// </summary>
    public const string ReservedDefaultName = "default";

    /// <summary>
    /// True when <paramref name="value"/> is a valid namespace: <c>^(|k-[0-9a-f]{16}|[a-z0-9._-]{1,64})$</c> except the reserved
    /// <see cref="ReservedDefaultName"/>, i.e. the empty <see cref="Default"/>, a <see cref="ForApiKey"/> value, or 1-64
    /// characters of <c>[a-z0-9._-]</c>. Valid namespaces are plain ASCII, so they can go into a key, a file name, a JSON ledger
    /// or a log line without escaping, and no two distinct strings encode to the same bytes.
    /// </summary>
    /// <param name="value">Candidate; null is not valid.</param>
    /// <returns>Whether it is valid.</returns>
    public static bool IsValid(string? value)
    {
        if (value is null || value.Length > MaxLength || value == ReservedDefaultName) return false;
        foreach (char c in value)
            if (c is not ((>= 'a' and <= 'z') or (>= '0' and <= '9') or '.' or '_' or '-')) return false;
        return true; // k-[0-9a-f]{16} is a subset of [a-z0-9._-]{1,64}
    }

    /// <summary>Throws unless <paramref name="value"/> is a valid namespace (<see cref="IsValid"/>).</summary>
    /// <param name="value">Candidate.</param>
    /// <param name="paramName">Parameter name reported in the exception.</param>
    /// <exception cref="ArgumentNullException"><paramref name="value"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="value"/> is not a valid namespace.</exception>
    public static void Validate(string value, string paramName = "namespace")
    {
        ArgumentNullException.ThrowIfNull(value, paramName);
        if (!IsValid(value))
            throw new ArgumentException(
                $"A chunk namespace is empty, k-<16 hex> or 1-{MaxLength} characters of [a-z0-9._-] other than '{ReservedDefaultName}'; got {value.Length} characters that are not.",
                paramName);
    }

    /// <summary>
    /// The namespace of an API key: <c>"k-"</c> + the first 16 lowercase hex characters of SHA-256 over the key's UTF-8 bytes.
    /// It holds a digest, never the key, is stable across processes, and never equals <see cref="Default"/>.
    /// </summary>
    /// <param name="apiKey">The caller's API key; not empty (a caller without a key uses <see cref="Default"/>).</param>
    /// <returns>The namespace, e.g. <c>k-e0dbaa0c6455768b</c>.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="apiKey"/> is null.</exception>
    /// <exception cref="ArgumentException"><paramref name="apiKey"/> is empty.</exception>
    public static string ForApiKey(string apiKey)
    {
        ArgumentException.ThrowIfNullOrEmpty(apiKey);
        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        SHA256.HashData(Encoding.UTF8.GetBytes(apiKey), digest);
        return "k-" + Convert.ToHexStringLower(digest[..(DigestHexChars / 2)]);
    }
}
