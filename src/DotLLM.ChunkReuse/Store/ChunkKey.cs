using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;

namespace DotLLM.ChunkReuse.Store;

/// <summary>
/// Physical element type of stored K/V rows. Not <c>DotLLM.Core.Configuration.KvCacheDType</c>: its <c>F32</c> means "full
/// precision", which is F32 on CPU but F16 on GPU, so it cannot tell two physically different encodings apart.
/// </summary>
public enum KvDtype
{
    /// <summary>32-bit float rows.</summary>
    F32 = 0,

    /// <summary>16-bit float rows.</summary>
    F16 = 1,

    /// <summary>Q8_0 blocks.</summary>
    Q8_0 = 2,

    /// <summary>Q4_0 blocks.</summary>
    Q4_0 = 3,
}

/// <summary>
/// Content address of one encoded chunk: SHA-256 (lowercase hex) over every input that changes the chunk's K/V. Two chunks get
/// the same key only when producer and tokens are identical; the key holds no file path, so identical chunk text in two files
/// shares one entry.
/// </summary>
/// <param name="Hex">64 lowercase hex characters.</param>
public readonly record struct ChunkKey(string Hex)
{
    // Bumped when the canonical encoding below changes; old keys then simply never match.
    private const string Scheme = "chunk-key/v1";

    /// <summary>Computes the key.</summary>
    /// <param name="modelFingerprint">Model identity (<see cref="ChunkFingerprint.Compute"/>).</param>
    /// <param name="tokenizerId">Tokenizer identity.</param>
    /// <param name="rope">RoPE parameters the stored keys were rotated with.</param>
    /// <param name="engineVersion">Engine/KV-format version; every version that may change K/V bits must differ.</param>
    /// <param name="backend">Backend that produced the K/V, e.g. <c>cpu</c> or <c>vulkan</c> (their F32 K/V differ by a noise floor).</param>
    /// <param name="kvDtype">Physical element type of the stored K/V.</param>
    /// <param name="tokenIds">The chunk's token ids.</param>
    /// <returns>The key.</returns>
    public static ChunkKey Create(string modelFingerprint, string tokenizerId, RopeParameters rope, string engineVersion,
        string backend, KvDtype kvDtype, ReadOnlySpan<int> tokenIds)
    {
        ArgumentNullException.ThrowIfNull(modelFingerprint);
        ArgumentNullException.ThrowIfNull(tokenizerId);
        ArgumentNullException.ThrowIfNull(engineVersion);
        ArgumentException.ThrowIfNullOrEmpty(backend);

        // Canonical encoding: length-prefixed UTF-8 strings and fixed-width little-endian integers, so no value can forge a
        // field boundary.
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        AppendString(hash, Scheme);
        AppendString(hash, modelFingerprint);
        AppendString(hash, tokenizerId);
        AppendInt(hash, rope.RopeDim);
        AppendInt(hash, BitConverter.SingleToInt32Bits(rope.Theta));
        AppendInt(hash, (int)rope.Type);
        AppendString(hash, engineVersion);
        AppendString(hash, backend);
        AppendInt(hash, (int)kvDtype);
        AppendInt(hash, tokenIds.Length);
        if (BitConverter.IsLittleEndian)
        {
            hash.AppendData(MemoryMarshal.AsBytes(tokenIds));
        }
        else
        {
            foreach (int id in tokenIds) AppendInt(hash, id);
        }
        return new ChunkKey(Convert.ToHexStringLower(hash.GetHashAndReset()));
    }

    /// <summary>True when <paramref name="hex"/> is a well-formed key (64 lowercase hex characters).</summary>
    /// <param name="hex">Candidate.</param>
    /// <returns>Whether it is well formed.</returns>
    public static bool IsWellFormed(string? hex)
    {
        if (hex is null || hex.Length != 64) return false;
        foreach (char c in hex)
            if (c is not ((>= '0' and <= '9') or (>= 'a' and <= 'f'))) return false;
        return true;
    }

    /// <inheritdoc/>
    public override string ToString() => Hex;

    private static void AppendInt(IncrementalHash hash, int value)
    {
        Span<byte> buf = stackalloc byte[4];
        BinaryPrimitives.WriteInt32LittleEndian(buf, value);
        hash.AppendData(buf);
    }

    private static void AppendString(IncrementalHash hash, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        AppendInt(hash, bytes.Length);
        hash.AppendData(bytes);
    }
}
