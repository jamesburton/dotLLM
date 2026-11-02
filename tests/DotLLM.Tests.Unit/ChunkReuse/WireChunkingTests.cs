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
/// Phase I Task 5 (L6): which chunks may travel on the wire. A chunk is wire-eligible when a server that receives its text and
/// tokenises it gets exactly the chunk's ids: text-first chunks (symbol, lines) are; token-id slices (an over-long line's
/// pieces, <see cref="FixedSizeChunker"/>) are not. Also the wire hash (Task 0 fixtures), the chunker allow-list and the
/// persisted flag (<see cref="ChunkRef"/> JSON, <see cref="ChunkIndex"/> files, including indexes written before the flag).
/// </summary>
public sealed class WireChunkingTests : IDisposable
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

    private const string Calc = """
        namespace Demo;

        public sealed class Calc
        {
            public int Add(int a, int b)
            {
                return a + b;
            }

            public int Neg(int a)
            {
                return -a;
            }
        }

        """;

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
    public void FixedSizeChunker_IdSlices_NotWireEligible_LinesChunker_Eligible()
    {
        var file = new SourceFile("Calc.cs", Calc.ReplaceLineEndings("\n"));
        Assert.All(new FixedSizeChunker(new CharTokenizer(), 16).Chunk(file), c => Assert.False(c.WireEligible));
        Assert.All(new LinesChunker(new CharTokenizer(), 16).Chunk(file), c => Assert.True(c.WireEligible));
    }

    // ------------------------------------------------------------------------------------------------ round trip

    [Fact]
    public void VerifyRoundTrip_IdSlicedFixedChunk_FailsAtMultibyteCut()
    {
        // ByteTokenizer: one id per UTF-8 byte. 'é' is 2 bytes (C3 A9); slices of 4 ids cut it after "caf" + C3.
        var file = new SourceFile("cafe.txt", "café\nnaïve\n");
        var tok = new ByteTokenizer();
        IReadOnlyList<Chunk> slices = new FixedSizeChunker(tok, 4).Chunk(file);
        RoundTripReport report = WireChunking.VerifyRoundTrip(slices, c => Text(file, c), tok);
        Assert.Equal(slices.Count, report.Checked);
        Assert.True(report.Failed >= 1);
        Assert.Contains(slices[0].Id, report.FailedIds);
        Assert.Equal(report.Failed, report.FailedIds.Count);

        // The same text cut text-first round-trips everywhere.
        IReadOnlyList<Chunk> lines = new LinesChunker(tok, 4).Chunk(file);
        RoundTripReport text = WireChunking.VerifyRoundTrip(lines, c => Text(file, c), tok);
        Assert.Equal((lines.Count, 0), (text.Checked, text.Failed));
        Assert.Empty(text.FailedIds);
    }

    [Fact]
    public void VerifyRoundTrip_SameLengthDifferentIds_Fails()
    {
        // A chunk whose ids have the right count but the wrong values (kills "compare lengths only").
        var chunk = new Chunk("x.txt", 1, 1, [(int)'a', (int)'c']) { StartOffset = 0, EndOffset = 2 };
        RoundTripReport report = WireChunking.VerifyRoundTrip([chunk], _ => "ab", new CharTokenizer());
        Assert.Equal((1, 1), (report.Checked, report.Failed));
        Assert.Equal([chunk.Id], report.FailedIds);
    }

    [Fact]
    public void VerifyRoundTrip_Lines512_AllPass()
    {
        // Multi-byte text (2-, 3- and 4-byte UTF-8, CRLF) long enough for several 512-id pieces, many cut mid-line.
        var sb = new StringBuilder();
        for (int i = 0; sb.Length < 6000; i++)
            sb.Append("résumé ").Append(i).Append(" — naïve ☃ 😀 café").Append(i % 3 == 0 ? "\r\n" : i % 7 == 0 ? "" : "\n");
        var file = new SourceFile("multi.txt", sb.ToString());
        var tok = new ByteTokenizer();
        var chunker = new LinesChunker(tok);
        Assert.Equal("lines-512", chunker.Name);
        IReadOnlyList<Chunk> chunks = chunker.Chunk(file);
        Assert.True(chunks.Count >= 10);
        RoundTripReport report = WireChunking.VerifyRoundTrip(chunks, c => Text(file, c), tok);
        Assert.Equal(chunks.Count, report.Checked);
        Assert.Equal(0, report.Failed);
        Assert.Empty(report.FailedIds);
    }

    // ------------------------------------------------------------------------------------------------ hash

    [Fact]
    public void Hash_IsSha256OfUtf8_LowercaseHex()
    {
        Assert.Equal("sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", WireChunking.Hash("abc"));
        Assert.Equal("sha256:" + Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes("é😀\r\n"))), WireChunking.Hash("é😀\r\n"));
        // Ruling 17: a lone surrogate hashes as U+FFFD (Encoding.UTF8's replacement).
        Assert.Equal(WireChunking.Hash("a�b"), WireChunking.Hash("a\uD800b"));
    }

    [Fact]
    public void Hash_MatchesFixtures()
    {
        int checkedSpans = 0;

        // inline.json: every annotated text part carries the hash of its (decoded) text.
        using (JsonDocument inline = Fixture("inline.json"))
        {
            foreach (JsonElement part in Parts(inline))
            {
                if (!part.TryGetProperty("chunk", out JsonElement chunk)) continue;
                Assert.Equal(chunk.GetProperty("hash").GetString(), WireChunking.Hash(part.GetProperty("text").GetString()!));
                checkedSpans++;
            }
        }

        // nonascii-crlf.json: manifest spans with UTF-8 byte offsets (a span after an emoji and 'é'), and a whole CRLF part.
        using (JsonDocument doc = Fixture("nonascii-crlf.json"))
        {
            JsonElement messages = doc.RootElement.GetProperty("messages");
            foreach (JsonElement span in doc.RootElement.GetProperty("chunk_cache").GetProperty("chunks").EnumerateArray())
            {
                JsonElement at = span.GetProperty("at");
                JsonElement content = messages[at.GetProperty("message").GetInt32()].GetProperty("content");
                string partText = (content.ValueKind == JsonValueKind.String ? content : content[at.GetProperty("part").GetInt32()].GetProperty("text")).GetString()!;
                byte[] bytes = Encoding.UTF8.GetBytes(partText);
                int start = at.TryGetProperty("start", out JsonElement s) ? s.GetInt32() : 0;
                int end = at.TryGetProperty("end", out JsonElement e) ? e.GetInt32() : bytes.Length;
                string spanText = Encoding.UTF8.GetString(bytes, start, end - start);
                Assert.Equal(span.GetProperty("hash").GetString(), WireChunking.Hash(spanText));
                checkedSpans++;
            }
        }
        Assert.Equal(4, checkedSpans);
    }

    [Fact]
    public void Fixtures_MatchContractSha256()
    {
        string dir = FixtureDir();
        string[] lines = File.ReadAllLines(Path.Combine(dir, "fixtures.sha256"));
        Assert.NotEmpty(lines);
        foreach (string line in lines.Where(l => l.Length > 0))
        {
            string[] parts = line.Split("  ", 2);
            Assert.Equal(parts[0], Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(Path.Combine(dir, parts[1])))));
        }
    }

    private static string FixtureDir() => Path.Combine(AppContext.BaseDirectory, "ChunkCache", "Fixtures", "v1");

    private static JsonDocument Fixture(string name) => JsonDocument.Parse(File.ReadAllBytes(Path.Combine(FixtureDir(), name)));

    private static IEnumerable<JsonElement> Parts(JsonDocument doc)
    {
        foreach (JsonElement m in doc.RootElement.GetProperty("messages").EnumerateArray())
            if (m.TryGetProperty("content", out JsonElement c) && c.ValueKind == JsonValueKind.Array)
                foreach (JsonElement p in c.EnumerateArray()) yield return p;
    }

    // ------------------------------------------------------------------------------------------------ allow-list, kind

    [Fact]
    public void Lines512_IsPendingNotEligible()
    {
        Assert.Equal(["symbol-512"], WireChunking.Eligible.Order());
        Assert.Equal(["lines-512"], WireChunking.PendingE4c.Order());
        Assert.True(WireChunking.IsWireChunker("symbol-512"));
        Assert.False(WireChunking.IsWireChunker("lines-512"));
        Assert.False(WireChunking.IsWireChunker("fixed-512"));
        Assert.False(WireChunking.IsWireChunker("symbol-256"));
        Assert.Equal("lines-512", new LinesChunker(new CharTokenizer()).Name);
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
