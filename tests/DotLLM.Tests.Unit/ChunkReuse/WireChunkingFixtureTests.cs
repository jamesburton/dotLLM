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
/// Phase I Task 0/5: the wire hash (<see cref="WireChunking.Hash"/>) of every annotated span in the chunk-cache contract
/// fixtures, and the fixtures' own <c>fixtures.sha256</c>. Split out of <see cref="WireChunkingTests"/> because it needs both
/// the library and the contract fixtures (<c>ChunkCache/Fixtures/v1</c>).
/// </summary>
public sealed class WireChunkingFixtureTests
{
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
}
