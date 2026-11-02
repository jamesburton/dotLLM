using System.Text.Json.Serialization;
using DotLLM.ChunkReuse.CodeCorpus;

namespace DotLLM.ChunkReuse.Store;

/// <summary>How a chunk's boundaries were chosen.</summary>
public enum ChunkKind
{
    /// <summary>Fixed-size chunks: token slices of the file (<see cref="FixedSizeChunker"/>) or text-first pieces
    /// (<see cref="LinesChunker"/>); boundaries can fall mid-line.</summary>
    Fixed = 0,

    /// <summary>One member per chunk from the Roslyn syntax tree (<see cref="SymbolChunker"/>); whole lines.</summary>
    Symbol = 1,

    /// <summary>Any other chunker (its name is not <c>symbol-*</c>, <c>fixed-*</c> or <c>lines-*</c>).</summary>
    Other = 2,
}

/// <summary>
/// A lightweight reference to one chunk of a source file: where it is (path, 1-based inclusive lines, character offsets) and,
/// when known, its size and store key. It carries no token ids, so it is cheap to keep for a whole repository, to persist and
/// to pass between the index, the selection tools and the CLI; the chunk's K/V is looked up in an <see cref="IChunkStore"/>
/// by <see cref="KeyHex"/>.
/// </summary>
/// <param name="FilePath">Corpus-relative path, '/' separated (as <see cref="SourceFile.Path"/>).</param>
/// <param name="StartLine">Line of the chunk's first character (1-based).</param>
/// <param name="EndLine">Line of the chunk's last character (1-based, inclusive).</param>
/// <param name="StartOffset">Character offset of the chunk's first character in the file text (inclusive).</param>
/// <param name="EndOffset">Character offset just past the chunk's last character (exclusive).</param>
/// <param name="Kind">The chunking that produced it.</param>
/// <remarks>
/// <para>
/// <b>Identity.</b> <see cref="Id"/> is a <i>location</i> (path, lines, offset), not content: a chunk whose text is unchanged
/// gets a new <see cref="Id"/> when an edit above it moves its lines, and two locations with the same text share one
/// <see cref="KeyHex"/>. Use <see cref="KeyHex"/> for "same K/V" and <see cref="Id"/> for "same place"; never use
/// <see cref="Id"/> as a cache key.
/// </para>
/// <para>
/// <b>Equality.</b> Record equality compares every field, including <see cref="TokenCount"/> and <see cref="KeyHex"/>, so a
/// reference made before tokenisation (both null) is not equal to the index's reference for the same place; compare by
/// <see cref="Id"/> for that.
/// </para>
/// <para>
/// <b>JSON.</b> Serialise with <see cref="ChunkRefJsonContext"/> (camelCase, <see cref="Kind"/> as its name, the computed
/// <see cref="Key"/> and <see cref="Id"/> omitted; trimming/AOT safe).
/// </para>
/// </remarks>
public sealed record ChunkRef(string FilePath, int StartLine, int EndLine, int StartOffset, int EndOffset, ChunkKind Kind)
{
    /// <summary>Number of token ids in the chunk, or null when not known (e.g. a selection made before tokenisation).</summary>
    public int? TokenCount { get; init; }

    /// <summary>The chunk's <see cref="ChunkKey.Hex"/>, or null when not known.</summary>
    public string? KeyHex { get; init; }

    // Stored inverted so the default (eligible) is the field's default: equality and JSON both see one representation.
    private readonly bool _notWireEligible;

    /// <summary>
    /// Whether the chunk may be sent on the wire as text (<see cref="Chunk.WireEligible"/>); default true. Serialised as
    /// <c>wireEligible</c> only when false (see <see cref="WireEligibleJson"/>).
    /// </summary>
    [JsonIgnore]
    public bool WireEligible
    {
        get => !_notWireEligible;
        init => _notWireEligible = !value;
    }

    /// <summary>
    /// JSON form of <see cref="WireEligible"/>: <c>false</c>, or null (omitted) when eligible. A nullable carrier because a
    /// source-generated context drops initializers of <c>init</c> properties, so JSON without the field (written before
    /// Phase I Task 5) must read as eligible through null. Use <see cref="WireEligible"/> in code.
    /// </summary>
    [JsonPropertyName("wireEligible")]
    [System.ComponentModel.EditorBrowsable(System.ComponentModel.EditorBrowsableState.Never)]
    public bool? WireEligibleJson
    {
        get => _notWireEligible ? false : null;
        init => _notWireEligible = value == false;
    }

    /// <summary>The key, when <see cref="KeyHex"/> is known.</summary>
    [JsonIgnore]
    public ChunkKey? Key => KeyHex is null ? null : new ChunkKey(KeyHex);

    /// <summary>Location id, same format as <see cref="Chunk.Id"/>: <c>path:startLine-endLine@startOffset</c> (not content identity).</summary>
    [JsonIgnore]
    public string Id => $"{FilePath}:{StartLine}-{EndLine}@{StartOffset}";

    /// <summary>A reference to <paramref name="chunk"/>, with its token count and (optionally) its key.</summary>
    /// <param name="chunk">The chunk.</param>
    /// <param name="kind">The chunking that produced it.</param>
    /// <param name="key">Its store key, if computed.</param>
    /// <returns>The reference.</returns>
    public static ChunkRef From(Chunk chunk, ChunkKind kind, ChunkKey? key = null)
    {
        ArgumentNullException.ThrowIfNull(chunk);
        return new ChunkRef(chunk.FilePath, chunk.StartLine, chunk.EndLine, chunk.StartOffset, chunk.EndOffset, kind)
        {
            TokenCount = chunk.TokenIds.Length,
            KeyHex = key?.Hex,
            WireEligible = chunk.WireEligible,
        };
    }

    /// <summary>The kind of the chunks <paramref name="chunker"/> produces.</summary>
    /// <param name="chunker">A chunker.</param>
    /// <returns>The kind by name prefix (see <see cref="KindOfName"/>).</returns>
    public static ChunkKind KindOf(IChunker chunker)
    {
        ArgumentNullException.ThrowIfNull(chunker);
        return KindOfName(chunker.Name);
    }

    /// <summary>The kind for a chunker name (<c>symbol-*</c>; <c>fixed-*</c> and <c>lines-*</c> are <see cref="ChunkKind.Fixed"/>;
    /// anything else is <see cref="ChunkKind.Other"/>).</summary>
    /// <param name="chunkerName">Chunker name.</param>
    /// <returns>The kind.</returns>
    public static ChunkKind KindOfName(string chunkerName)
    {
        ArgumentNullException.ThrowIfNull(chunkerName);
        if (chunkerName.StartsWith("symbol-", StringComparison.Ordinal)) return ChunkKind.Symbol;
        if (chunkerName.StartsWith("fixed-", StringComparison.Ordinal) || chunkerName.StartsWith("lines-", StringComparison.Ordinal)) return ChunkKind.Fixed;
        return ChunkKind.Other;
    }
}

/// <summary>
/// Source-generated JSON contract for <see cref="ChunkRef"/> (camelCase; <see cref="ChunkKind"/> written as its name;
/// <see cref="ChunkRef.Key"/> and <see cref="ChunkRef.Id"/> are computed and not written).
/// </summary>
[JsonSourceGenerationOptions(PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase, UseStringEnumConverter = true,
    DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
[JsonSerializable(typeof(ChunkRef))]
[JsonSerializable(typeof(List<ChunkRef>))]
public sealed partial class ChunkRefJsonContext : JsonSerializerContext;
