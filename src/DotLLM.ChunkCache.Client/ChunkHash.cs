// SPDX-License-Identifier: MIT
using System.Security.Cryptography;
using System.Text;

namespace DotLLM.ChunkCache.Client;

/// <summary>The wire hash of a chunk (spec 5.2, 5.3 item 1).</summary>
public static class ChunkHash
{
    /// <summary>The prefix of every wire hash.</summary>
    public const string Prefix = "sha256:";

    /// <summary>
    /// <c>"sha256:" + lowercase hex</c> of the UTF-8 bytes of <paramref name="text"/>, with no normalisation: CRLF vs LF,
    /// a BOM and trailing whitespace all change the hash.
    /// </summary>
    /// <remarks>
    /// <paramref name="text"/> is the <b>decoded</b> string (a .NET string, or a JSON string value after its escapes are
    /// resolved), never the escaped wire bytes. A lone surrogate encodes as U+FFFD (EF BF BD), the default
    /// <see cref="System.Text.Encoding.UTF8"/> replacement; contract ruling 17 requires the server to do the same.
    /// </remarks>
    public static string Of(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        SHA256.HashData(Encoding.UTF8.GetBytes(text), digest);
        // One code path for net8.0 and net10.0 (the tests run on net10.0 only, so a TFM branch would go unexercised).
        return Prefix + Convert.ToHexString(digest).ToLowerInvariant();
    }
}
