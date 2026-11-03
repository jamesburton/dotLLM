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

/// <summary>How a stored chunk's K/V were computed relative to the rest of a context.</summary>
public enum ChunkPositioning
{
    /// <summary>Encoded alone at positions <c>0..n-1</c> with no preceding context (<see cref="ChunkEncoder"/>); the only kind the
    /// assemblers place and re-rotate.</summary>
    Isolated = 0,

    /// <summary>Computed with preceding context (e.g. a selective-recompute output). Not interchangeable with isolated K/V; the
    /// positioning is in the key so such entries never answer an isolated lookup.</summary>
    Contextual = 1,
}

/// <summary>
/// Content address of one encoded chunk: SHA-256 (lowercase hex) over every input that changes the chunk's K/V, plus the store
/// partition (<see cref="ChunkNamespace"/>). Two chunks get the same key only when namespace, producer and tokens are identical;
/// the key holds no file path, so identical chunk text in two files of one namespace shares one entry. Entries an assembler places must be <see cref="ChunkPositioning.Isolated"/> encodes (<see cref="ChunkEncoder"/>
/// at <c>0..n-1</c>, no preceding context); K/V computed with context belong under a <see cref="ChunkPositioning.Contextual"/> key.
/// </summary>
/// <param name="Hex">64 lowercase hex characters.</param>
public readonly record struct ChunkKey(string Hex)
{
    // Bumped when the canonical encoding below changes; old keys then simply never match.
    private const string Scheme = "chunk-key/v4"; // v3: positioning; v4: namespace

    /// <summary>Computes the key.</summary>
    /// <param name="modelFingerprint">
    /// The real model fingerprint (<see cref="ChunkFingerprint.Compute(DotLLM.Core.Models.ModelConfig, string, string, string, string, int)"/>), not a label: RoPE scaling fields (YaRN/NTK factors,
    /// original context, betas), NoPE layers and the KV layout reach the key only through it.
    /// </param>
    /// <param name="tokenizerId">Tokenizer identity.</param>
    /// <param name="rope">RoPE parameters the stored keys were rotated with.</param>
    /// <param name="backend">Backend that produced the K/V, e.g. <c>cpu</c> or <c>vulkan</c> (their F32 K/V differ by a noise floor).</param>
    /// <param name="kvDtype">Physical element type of the stored K/V.</param>
    /// <param name="tokenIds">The chunk's token ids.</param>
    /// <param name="kvOutputVersion">K/V schema version (<see cref="ChunkReuseSchema.KvOutputVersion"/>); callers pass the default except in tests.</param>
    /// <param name="positioning">How the K/V were computed; store entries are <see cref="ChunkPositioning.Isolated"/> encodes.</param>
    /// <param name="namespace">Store partition (<see cref="ChunkNamespace"/>): <see cref="ChunkNamespace.Default"/> for a single-user
    /// store, <see cref="ChunkNamespace.ForApiKey"/> per API key; must be valid (<see cref="ChunkNamespace.IsValid"/>). Always hashed,
    /// so equal inputs in two namespaces never share a key.</param>
    /// <exception cref="ArgumentException">The namespace is not valid (<see cref="ChunkNamespace.IsValid"/>).</exception>
    /// <returns>The key.</returns>
    public static ChunkKey Create(string modelFingerprint, string tokenizerId, RopeParameters rope, string backend, KvDtype kvDtype,
        ReadOnlySpan<int> tokenIds, int kvOutputVersion = ChunkReuseSchema.KvOutputVersion, ChunkPositioning positioning = ChunkPositioning.Isolated,
        string @namespace = ChunkNamespace.Default)
    {
        ArgumentNullException.ThrowIfNull(modelFingerprint);
        ArgumentNullException.ThrowIfNull(tokenizerId);
        ArgumentException.ThrowIfNullOrEmpty(backend);
        ChunkNamespace.Validate(@namespace, nameof(@namespace));

        // Canonical encoding: length-prefixed UTF-8 strings and fixed-width little-endian integers, so no value can forge a
        // field boundary. The namespace is always written, the default "" as a zero length, so no namespace collapses to v3.
        // It is encoded strictly (a lone surrogate throws instead of becoming U+FFFD, which would let two namespaces share
        // keys); the validity rule above already admits ASCII only, so this is defence in depth.
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        AppendString(hash, Scheme);
        AppendString(hash, @namespace, StrictUtf8);
        AppendString(hash, modelFingerprint);
        AppendString(hash, tokenizerId);
        AppendInt(hash, rope.RopeDim);
        AppendInt(hash, BitConverter.SingleToInt32Bits(rope.Theta));
        AppendInt(hash, (int)rope.Type);
        AppendInt(hash, kvOutputVersion);
        AppendString(hash, backend);
        AppendInt(hash, (int)kvDtype);
        AppendInt(hash, (int)positioning);
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

    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    private static void AppendString(IncrementalHash hash, string value) => AppendString(hash, value, Encoding.UTF8);

    private static void AppendString(IncrementalHash hash, string value, Encoding encoding)
    {
        byte[] bytes = encoding.GetBytes(value);
        AppendInt(hash, bytes.Length);
        hash.AppendData(bytes);
    }
}
