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
/// tokenises it gets exactly the chunk's ids: text-first chunks (lines) are; token-id slices (<see cref="FixedSizeChunker"/>)
/// are not. Also the wire hash and the chunker allow-list. The symbol chunker's and the index's side are in
/// <c>WireChunkingIndexTests</c>; the contract-fixture hashes in <c>WireChunkingFixtureTests</c>.
/// </summary>
public sealed class WireChunkingTests
{
    internal const string Calc = """
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
}
