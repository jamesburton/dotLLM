using System.Text;
using System.Text.Json.Nodes;
using Xunit;

// Test code: handlers, requests and helpers are handed to HttpClient / HttpMessageInvoker chains that each test
// disposes (ownership the analyzer cannot follow), and continuations may resume on xUnit's context.
#pragma warning disable IDISP001, IDISP004, IDISP017, MA0004

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary><see cref="SseChunkCacheScanner"/>: pass-through, lazy, chunk-boundary independent.</summary>
public sealed class SseScannerTests
{
    private static string OpenAiStream(string newline = "\n") =>
        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"the word \\\"chunk_cache\\\" in a token\"}}]}" + newline + newline
        + "data: {\"choices\":[],\"nested\":{\"chunk_cache\":{\"v\":1}}}" + newline + newline
        + "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"é😀\"}}]}" + newline + newline
        + "data: " + Fixtures.Json("response-stream-final.json").ToJsonString() + newline + newline
        + "data: [DONE]" + newline + newline;

    private static string AnthropicStream() =>
        "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\"}}\n\n"
        + "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n"
        + "event: message_delta\ndata: " + Fixtures.Json("response-anthropic-delta.json").ToJsonString() + "\n\n"
        + "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";

    private static (byte[] Out, List<JsonObject> Reports, SseChunkCacheScanner Scanner) Scan(string sse, int readSize)
    {
        var reports = new List<JsonObject>();
        var scanner = new SseChunkCacheScanner(new TrickleStream(Encoding.UTF8.GetBytes(sse), readSize), reports.Add);
        var ms = new MemoryStream();
        var buf = new byte[readSize];
        int n;
        while ((n = scanner.Read(buf, 0, buf.Length)) > 0)
            ms.Write(buf, 0, n);
        return (ms.ToArray(), reports, scanner);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(4096)]
    public void OpenAiFinalChunk_AnyReadSize(int readSize)
    {
        string sse = OpenAiStream();
        (byte[] output, List<JsonObject> reports, _) = Scan(sse, readSize);

        Assert.Equal(Encoding.UTF8.GetBytes(sse), output);
        JsonObject report = Assert.Single(reports);
        Assert.True(JsonNode.DeepEquals(Fixtures.Json("response-stream-final.json")["chunk_cache"], report));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(4096)]
    public void AnthropicMessageDelta(int readSize)
    {
        (byte[] _, List<JsonObject> reports, _) = Scan(AnthropicStream(), readSize);

        Assert.True(JsonNode.DeepEquals(Fixtures.Json("response-anthropic-delta.json")["chunk_cache"], Assert.Single(reports)));
    }

    [Fact]
    public void CrlfLineEndings()
    {
        (byte[] _, List<JsonObject> reports, _) = Scan(OpenAiStream("\r\n"), 11);
        Assert.Single(reports);
    }

    /// <summary>
    /// Lazy: only events whose bytes contain <c>"chunk_cache"</c> are parsed: here a nested (not top-level) member and the
    /// final chunk. A token containing the quoted word is escaped on the wire (<c>\"</c>) and never matches.
    /// </summary>
    [Fact]
    public void OnlyCandidateEventsAreParsed()
    {
        (_, _, SseChunkCacheScanner scanner) = Scan(OpenAiStream(), 64);
        Assert.Equal(2, scanner.ParsedEvents);
    }

    [Fact]
    public void NoReport_NoCallback_AndFinalLineWithoutNewlineStillSeen()
    {
        (_, List<JsonObject> none, _) = Scan("data: {\"choices\":[]}\n\ndata: [DONE]\n\n", 8);
        Assert.Empty(none);

        string unterminated = "data: " + Fixtures.Json("response-stream-final.json").ToJsonString();
        (_, List<JsonObject> one, _) = Scan(unterminated, 16);
        Assert.Single(one);
    }

    [Fact]
    public async Task AsyncRead_PassesThrough()
    {
        string sse = AnthropicStream();
        var reports = new List<JsonObject>();
        await using var scanner = new SseChunkCacheScanner(new TrickleStream(Encoding.UTF8.GetBytes(sse), 13), reports.Add);
        var ms = new MemoryStream();
        await scanner.CopyToAsync(ms);
        Assert.Equal(sse, Encoding.UTF8.GetString(ms.ToArray()));
        Assert.Single(reports);
    }

    /// <summary>A stream that returns at most <c>size</c> bytes per read.</summary>
    private sealed class TrickleStream(byte[] data, int size) : Stream
    {
        private int _pos;

        public override bool CanRead => true;

        public override bool CanSeek => false;

        public override bool CanWrite => false;

        public override long Length => data.Length;

        public override long Position { get => _pos; set => throw new NotSupportedException(); }

        public override void Flush() { }

        public override int Read(byte[] buffer, int offset, int count)
        {
            int n = Math.Min(Math.Min(count, size), data.Length - _pos);
            Array.Copy(data, _pos, buffer, offset, n);
            _pos += n;
            return n;
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();

        public override void SetLength(long value) => throw new NotSupportedException();

        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }
}
