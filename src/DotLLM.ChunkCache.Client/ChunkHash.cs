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
        return OfSpan(text.AsSpan());
    }

    /// <summary>The same hash over a slice of a string (the body handler hashes spans in place).</summary>
    internal static string OfSpan(ReadOnlySpan<char> text)
    {
        Span<byte> digest = stackalloc byte[SHA256.HashSizeInBytes];
        byte[] utf8 = System.Buffers.ArrayPool<byte>.Shared.Rent(Encoding.UTF8.GetMaxByteCount(text.Length));
        try
        {
            int n = Encoding.UTF8.GetBytes(text, utf8);
            SHA256.HashData(utf8.AsSpan(0, n), digest);
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(utf8);
        }

        // One code path for net8.0 and net10.0 (the tests run on net10.0 only, so a TFM branch would go unexercised).
        return Prefix + Convert.ToHexString(digest).ToLowerInvariant();
    }
}
