using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.ChunkReuse;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Store;
using DotLLM.Core.Configuration;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// Phase I Task 5 (L6), symbol chunker and index side: an over-long line's token slices are not wire-eligible, and the
/// persisted flag survives <see cref="ChunkRef"/> JSON and <see cref="ChunkIndex"/> files (including indexes written before
/// the flag). Split out of <see cref="WireChunkingTests"/> (core) because it needs the symbol chunker and the index.
/// </summary>
public sealed class WireChunkingIndexTests : IDisposable
{
    private static readonly RopeParameters Rope = new(128, 5_000_000f, RoPEType.NeoX);
    private readonly string _root = Path.Combine(Path.GetTempPath(), "dotllm-wirechunking-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { if (Directory.Exists(_root)) Directory.Delete(_root, recursive: true); }
        catch (IOException) { /* best effort */ }
        catch (UnauthorizedAccessException) { }
    }

    private static ChunkKey Key(ReadOnlySpan<int> ids) => ChunkKey.Create("fp-wire", "tok", Rope, "cpu", KvDtype.F32, ids);

    // One over-long line (the string field, > 40 chars) between two ordinary members.
    private const string WithLongLine = """
        namespace Demo;

        public sealed class Holder
        {
            public int One() => 1;

            public const string Long = "0123456789012345678901234567890123456789012345678901234567890123456789";

            public int Two() => 2;
        }

        """;

    private const string Calc = WireChunkingTests.Calc;

    private static string Text(SourceFile f, Chunk c) => f.Text[c.StartOffset..c.EndOffset];

    // ------------------------------------------------------------------------------------------------ chunkers

    [Fact]
    public void SymbolChunker_OverlongLineSlices_NotWireEligible()
    {
        var file = new SourceFile("Holder.cs", WithLongLine.ReplaceLineEndings("\n"));
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 40).Chunk(file);
        int longLine = 1 + file.Text[..file.Text.IndexOf("Long =", StringComparison.Ordinal)].Count(ch => ch == '\n');

        List<Chunk> slices = chunks.Where(c => c.StartLine == longLine && c.EndLine == longLine).ToList();
        Assert.True(slices.Count >= 2, $"the {file.Text.Split('\n')[longLine - 1].Length}-char line is cut into token slices");
        Assert.All(slices, c => Assert.False(c.WireEligible, $"slice {c.Id} must not be wire-eligible"));
        List<Chunk> others = chunks.Except(slices).ToList();
        Assert.NotEmpty(others);
        Assert.All(others, c => Assert.True(c.WireEligible, $"whole-line chunk {c.Id} is wire-eligible"));
    }

    [Fact]
    public void SymbolChunker_NormalChunks_WireEligible()
    {
        var file = new SourceFile("Calc.cs", Calc.ReplaceLineEndings("\n"));
        IReadOnlyList<Chunk> chunks = new SymbolChunker(new CharTokenizer(), 512).Chunk(file);
        Assert.True(chunks.Count >= 2);
        Assert.All(chunks, c => Assert.True(c.WireEligible));
        // And they do round-trip, so the flag is not just a default.
        Assert.Equal(0, WireChunking.VerifyRoundTrip(chunks, c => Text(file, c), new CharTokenizer()).Failed);
    }

    [Fact]
    public void ChunkRef_KindOf_Lines_IsFixed()
    {
        Assert.Equal(ChunkKind.Fixed, ChunkRef.KindOf(new LinesChunker(new CharTokenizer())));
        Assert.Equal(ChunkKind.Fixed, ChunkRef.KindOfName("lines-64"));
        Assert.Equal(ChunkKind.Other, ChunkRef.KindOfName("lines")); // no size suffix: not this chunker
    }

    // ------------------------------------------------------------------------------------------------ persistence

    [Fact]
    public void ChunkRef_Json_RoundTripsWireEligible()
    {
        var slice = new Chunk("a.cs", 3, 3, [1, 2, 3]) { StartOffset = 10, EndOffset = 13, WireEligible = false };
        ChunkRef notEligible = ChunkRef.From(slice, ChunkKind.Symbol, Key([1, 2, 3]));
        Assert.False(notEligible.WireEligible);
        string json = JsonSerializer.Serialize(notEligible, ChunkRefJsonContext.Default.ChunkRef);
        Assert.Contains("\"wireEligible\":false", json, StringComparison.Ordinal);
        ChunkRef back = JsonSerializer.Deserialize(json, ChunkRefJsonContext.Default.ChunkRef)!;
        Assert.False(back.WireEligible);
        Assert.Equal(notEligible, back);

        ChunkRef eligible = ChunkRef.From(slice with { WireEligible = true }, ChunkKind.Symbol, Key([1, 2, 3]));
        Assert.True(eligible.WireEligible);
        ChunkRef eligibleBack = JsonSerializer.Deserialize(JsonSerializer.Serialize(eligible, ChunkRefJsonContext.Default.ChunkRef), ChunkRefJsonContext.Default.ChunkRef)!;
        Assert.True(eligibleBack.WireEligible);
        Assert.Equal(eligible, eligibleBack);
        Assert.NotEqual(notEligible, eligible);

        // JSON without the field (written before Task 5) reads as eligible: the default survives source-generated deserialisation.
        ChunkRef legacy = JsonSerializer.Deserialize("""{"filePath":"x.txt","startLine":1,"endLine":1,"startOffset":0,"endOffset":1,"kind":"Symbol"}""",
            ChunkRefJsonContext.Default.ChunkRef)!;
        Assert.True(legacy.WireEligible);
        Assert.Equal(new ChunkRef("x.txt", 1, 1, 0, 1, ChunkKind.Symbol), legacy);
        Assert.True(new ChunkRef("x.txt", 1, 1, 0, 1, ChunkKind.Symbol).WireEligible);
    }

    [Fact]
    public void ChunkIndex_SaveLoad_RoundTripsWireEligible()
    {
        var file = new SourceFile("Holder.cs", WithLongLine.ReplaceLineEndings("\n"));
        var index = new ChunkIndex(Key);
        index.Update([file], new SymbolChunker(new CharTokenizer(), 40));
        IReadOnlyList<ChunkRef> refs = index.ChunksOf("Holder.cs");
        Assert.Contains(refs, r => !r.WireEligible);
        Assert.Contains(refs, r => r.WireEligible);

        string path = Path.Combine(_root, "index.json");
        index.Save(path);
        JsonNode doc = JsonNode.Parse(File.ReadAllText(path))!;
        Assert.All(doc["files"]![0]!["chunks"]!.AsArray(), c => Assert.NotNull(c!["wireEligible"]));
        ChunkIndex loaded = ChunkIndex.Load(path, Key);
        Assert.Null(loaded.LoadNote);
        Assert.Equal(refs, loaded.ChunksOf("Holder.cs"));
    }

    /// <summary>Saves an index of <paramref name="file"/> made with <paramref name="chunker"/>, then removes every
    /// <c>wireEligible</c> field, as an index written before Task 5.</summary>
    private string LegacyIndex(SourceFile file, IChunker chunker, out ChunkIndex original)
    {
        original = new ChunkIndex(Key);
        original.Update([file], chunker);
        string path = Path.Combine(_root, "legacy-" + chunker.Name + ".json");
        original.Save(path);
        JsonNode doc = JsonNode.Parse(File.ReadAllText(path))!;
        foreach (JsonNode? f in doc["files"]!.AsArray())
            foreach (JsonNode? c in f!["chunks"]!.AsArray())
                Assert.True(c!.AsObject().Remove("wireEligible"));
        File.WriteAllText(path, doc.ToJsonString());
        return path;
    }

    [Fact]
    public void ChunkIndex_LegacySymbolIndex_WholeLineChunks_LoadEligible_AndRechunkOnNextUpdate()
    {
        var file = new SourceFile("Calc.cs", Calc.ReplaceLineEndings("\n"));
        var tok = new CharTokenizer();
        string path = LegacyIndex(file, new SymbolChunker(tok, 512), out ChunkIndex original);

        ChunkIndex loaded = ChunkIndex.Load(path, Key);
        Assert.Equal(original.ChunksOf("Calc.cs"), loaded.ChunksOf("Calc.cs"));
        Assert.All(loaded.ChunksOf("Calc.cs"), r => Assert.True(r.WireEligible));

        // Saving it before an update keeps it legacy on disk.
        string resaved = Path.Combine(_root, "legacy-resaved.json");
        loaded.Save(resaved);
        Assert.All(JsonNode.Parse(File.ReadAllText(resaved))!["files"]![0]!["chunks"]!.AsArray(), c => Assert.Null(c!["wireEligible"]));
        loaded = ChunkIndex.Load(resaved, Key);

        // Written under the pre-Task-5b tokenisation: the next update re-chunks the file even though its text is unchanged.
        int before = tok.EncodeCalls;
        IndexDiff diff = loaded.Update([file], new SymbolChunker(tok, 512));
        Assert.True(tok.EncodeCalls > before, "a legacy entry is re-chunked on the next update");
        Assert.Empty(diff.Added);
        Assert.Empty(diff.Removed);
        Assert.Equal(original.ChunksOf("Calc.cs").Count, diff.Unchanged.Count);

        // A current index is not re-chunked.
        string current = Path.Combine(_root, "current.json");
        loaded.Save(current);
        ChunkIndex again = ChunkIndex.Load(current, Key);
        before = tok.EncodeCalls;
        again.Update([file], new SymbolChunker(tok, 512));
        Assert.Equal(before, tok.EncodeCalls);
    }

    [Fact]
    public void ChunkIndex_LegacySymbolIndex_WithTokenSlices_FileDropped()
    {
        var file = new SourceFile("Holder.cs", WithLongLine.ReplaceLineEndings("\n"));
        var calc = new SourceFile("Calc.cs", Calc.ReplaceLineEndings("\n"));
        var original = new ChunkIndex(Key);
        original.Update([file, calc], new SymbolChunker(new CharTokenizer(), 40));
        string path = Path.Combine(_root, "legacy-slices.json");
        original.Save(path);
        JsonNode doc = JsonNode.Parse(File.ReadAllText(path))!;
        foreach (JsonNode? f in doc["files"]!.AsArray())
            foreach (JsonNode? c in f!["chunks"]!.AsArray()) c!.AsObject().Remove("wireEligible");
        File.WriteAllText(path, doc.ToJsonString());

        ChunkIndex loaded = ChunkIndex.Load(path, Key);
        Assert.Empty(loaded.ChunksOf("Holder.cs"));          // its slices cannot be told apart from eligible chunks: re-chunk
        Assert.NotEmpty(loaded.ChunksOf("Calc.cs"));
        Assert.NotNull(loaded.LoadNote);
        Assert.Contains("Holder.cs", loaded.LoadNote, StringComparison.Ordinal);
    }

    [Fact]
    public void ChunkIndex_LegacyNonSymbolIndex_FilesDropped()
    {
        // Dropped by chunker kind, not only for shared lines: both indexes hold one whole-file chunk (no line is shared).
        var file = new SourceFile("Calc.cs", Calc.ReplaceLineEndings("\n"));
        string path = LegacyIndex(file, new FixedSizeChunker(new CharTokenizer(), 4096), out ChunkIndex fixedIndex);
        Assert.Single(fixedIndex.ChunksOf("Calc.cs"));
        ChunkIndex loaded = ChunkIndex.Load(path, Key);
        Assert.Empty(loaded.ChunksOf("Calc.cs"));
        Assert.Contains("Calc.cs", loaded.LoadNote, StringComparison.Ordinal);

        string lines = LegacyIndex(file, new LinesChunker(new CharTokenizer(), 4096), out ChunkIndex linesIndex);
        Assert.Single(linesIndex.ChunksOf("Calc.cs"));
        Assert.Empty(ChunkIndex.Load(lines, Key).ChunksOf("Calc.cs"));

        // Several whole-line chunks of a non-symbol chunker (cuts at line ends only) are dropped too.
        string multi = LegacyIndex(file, new LinesChunker(new CharTokenizer(), 120), out ChunkIndex multiIndex);
        IReadOnlyList<ChunkRef> refs = multiIndex.ChunksOf("Calc.cs");
        Assert.True(refs.Count >= 2);
        Assert.All(refs.Skip(1).Zip(refs), p => Assert.True(p.First.StartLine > p.Second.EndLine, "whole-line cuts"));
        Assert.Empty(ChunkIndex.Load(multi, Key).ChunksOf("Calc.cs"));
    }
}
