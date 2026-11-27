using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// OpenAI array content with <c>chunk</c> annotations (user ruling 2026-11-27, #2035, superseding chunk-cache ruling 19). Every request goes
/// through the server's source-generated <see cref="ServerJsonContext"/> (never reflection), because the CLAUDE.md rule on STJ
/// source generation is exactly about this path. Each test names the mutant it kills (the part's <c>chunk</c> dropped at read time, the
/// engine offset of a later part wrong, <c>at.part</c> read as 0, the image part counted as text).
/// </summary>
public sealed class ChunkCacheOpenAiPartsTests
{
    private static readonly ChunkCacheOptions On = new(true, ChunkCacheOptions.DefaultMaxChunks);

    private static string Sha(string text) => ChunkAnnotationParserTests.Sha(text);

    private static ChatCompletionRequest Bind(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;

    private static string Chunk(string id, string text) => $$$"""{"v":1,"id":"{{{id}}}","hash":"{{{Sha(text)}}}"}""";

    private static ParsedChunkRequest ParseOk(string json, ChunkCacheOptions? options = null)
    {
        var request = Bind(json);
        var parsed = ChunkAnnotationParser.Parse(request, options ?? On, "");
        Assert.NotNull(parsed);
        return parsed;
    }

    private const string TinyPng = "data:image/png;base64,iVBORw0KGgo=";

    /// <summary>
    /// Kills: the converter dropping a part's <c>chunk</c> (Parse returns null, 0 spans) and a wrong engine offset (the "not_carried"
    /// defect, or the span text taken from the wrong place). The annotated part is the THIRD text part, so its offset is the length of
    /// both earlier parts, with an image part in between that must not count.
    /// </summary>
    [Fact]
    public void InlineChunkOnALaterTextPart_IsHonoured_WithTheEngineOffsetOfTheEarlierTextParts()
    {
        string json = $$$"""
            {"messages":[{"role":"user","content":[
              {"type":"text","text":"ab"},
              {"type":"image_url","image_url":{"url":"{{{TinyPng}}}"}},
              {"type":"text","text":"cd"},
              {"type":"text","text":"héllo wörld","chunk":{{{Chunk("x", "héllo wörld")}}} }]}]}
            """;
        var parsed = ParseOk(json);

        Assert.Empty(parsed.Warnings);
        Assert.Empty(parsed.Defects);
        var span = Assert.Single(parsed.Spans);
        Assert.Equal("héllo wörld", span.Text);
        Assert.Equal("x", span.Id);
        Assert.Equal(0, span.Message);
        Assert.Equal(3, span.Part);                          // index in the content array (the image counts as a part)
        Assert.Equal(0, span.StartByte);
        Assert.Equal(Encoding.UTF8.GetByteCount("héllo wörld"), span.EndByte);
        Assert.Equal(0, span.Engine.EngineMessage);
        Assert.Equal(4, span.Engine.StartChar);              // "ab" + "cd": the image part adds nothing
        Assert.Equal(4 + "héllo wörld".Length, span.Engine.EndChar);

        var engine = RequestConverter.ToMessages(Bind(json).Messages);
        Assert.Equal("abcdhéllo wörld", engine[0].Content);
        Assert.Equal(span.Text, engine[0].Content.Substring(span.Engine.StartChar, span.Engine.EndChar - span.Engine.StartChar));
    }

    /// <summary>
    /// Same semantics as the string-content form: the same text annotated inline on a part and through the manifest on string content
    /// gives the same span (id, text, byte range); only the placement differs.
    /// </summary>
    [Fact]
    public void InlineOnAPart_MatchesTheStringContentManifestForm()
    {
        const string body = "hello world";
        var viaPart = ParseOk($$$"""
            {"messages":[{"role":"user","content":[{"type":"text","text":"{{{body}}}","chunk":{{{Chunk("x", body)}}} }]}]}
            """);
        var viaString = ParseOk($$$"""
            {"messages":[{"role":"user","content":"{{{body}}}"}],
             "chunk_cache":{"v":1,"chunks":[{"id":"x","hash":"{{{Sha(body)}}}","at":{"message":0,"part":0}}]}}
            """);

        var a = Assert.Single(viaPart.Spans);
        var b = Assert.Single(viaString.Spans);
        Assert.Equal((b.Id, b.Hash, b.Text, b.StartByte, b.EndByte, b.Engine), (a.Id, a.Hash, a.Text, a.StartByte, a.EndByte, a.Engine));
    }

    /// <summary>
    /// Manifest <c>at.part</c> over multi-part content: <c>part</c> is the index in the content array. Kills: <c>part</c> read as 0
    /// (a span would land in part 0's text and fail its hash), an offset missing the earlier parts, and the image counted as text.
    /// </summary>
    [Fact]
    public void ManifestAtPart_AddressesTheContentArray_StartEndAreBytesInThatPart()
    {
        const string p2 = "xx-é-needle-yy";
        string needle = "needle";
        int start = Encoding.UTF8.GetByteCount("xx-é-");
        string json = $$$"""
            {"messages":[{"role":"system","content":"sys"},{"role":"user","content":[
              {"type":"text","text":"first part"},
              {"type":"image_url","image_url":{"url":"{{{TinyPng}}}"}},
              {"type":"text","text":"{{{p2}}}"}]}],
             "chunk_cache":{"v":1,"chunks":[
               {"id":"whole0","hash":"{{{Sha("first part")}}}","at":{"message":1,"part":0}},
               {"id":"needle","hash":"{{{Sha(needle)}}}","at":{"message":1,"part":2,"start":{{{start}}},"end":{{{start + Encoding.UTF8.GetByteCount(needle)}}} }}]}}
            """;
        var parsed = ParseOk(json);

        Assert.Empty(parsed.Defects);
        Assert.Equal(2, parsed.Spans.Count);
        var whole = parsed.Spans[0];
        Assert.Equal((1, 0, "first part", 0, 10), (whole.Message, whole.Part, whole.Text, whole.Engine.StartChar, whole.Engine.EndChar));
        var found = parsed.Spans[1];
        Assert.Equal((1, 2, needle), (found.Message, found.Part, found.Text));
        Assert.Equal(1, found.Engine.EngineMessage);
        Assert.Equal("first part".Length + "xx-é-".Length, found.Engine.StartChar);   // earlier TEXT parts only; é is one UTF-16 char
        Assert.Equal(found.Engine.StartChar + needle.Length, found.Engine.EndChar);
    }

    /// <summary>The manifest's failure modes on array content are the existing ones: the image part is not text; an index past the array is out of range.</summary>
    [Fact]
    public void ManifestAtPart_ImagePartAndOutOfRange_AreDefects()
    {
        string json = $$$"""
            {"messages":[{"role":"user","content":[
              {"type":"text","text":"abc"},{"type":"image_url","image_url":{"url":"{{{TinyPng}}}"}}]}],
             "chunk_cache":{"v":1,"chunks":[
               {"id":"img","hash":"{{{Sha("abc")}}}","at":{"message":0,"part":1}},
               {"id":"far","hash":"{{{Sha("abc")}}}","at":{"message":0,"part":2}}]}}
            """;
        var parsed = ParseOk(json);
        Assert.Empty(parsed.Spans);
        Assert.Equal(["non_text", "part_out_of_range"], parsed.Defects.Select(d => d.Kind));
    }

    /// <summary>An inline <c>chunk</c> on an image part is a non-text defect (as on the Anthropic surface), not a crash and not silently accepted.</summary>
    [Fact]
    public void InlineChunkOnAnImagePart_IsANonTextDefect()
    {
        string json = $$$"""
            {"messages":[{"role":"user","content":[
              {"type":"text","text":"abc"},{"type":"image_url","image_url":{"url":"{{{TinyPng}}}"},"chunk":{{{Chunk("i", "abc")}}} }]}]}
            """;
        var parsed = ParseOk(json);
        Assert.Empty(parsed.Spans);
        Assert.Equal("non_text", Assert.Single(parsed.Defects).Kind);
    }

    /// <summary>Inline defects are the chunk's own: a wrong hash is <c>chunk_hash_mismatch</c>, strict mode makes it the request's error.</summary>
    [Fact]
    public void InlineChunk_WrongHash_IsHashMismatch_AndStrictIsTheRequestError()
    {
        string json = """
            {"messages":[{"role":"user","content":[{"type":"text","text":"abc","chunk":{"v":1,"id":"x","hash":"sha256:00"}}]}],
             "chunk_cache":{"v":1,"strict":true}}
            """;
        var parsed = ParseOk(json);
        Assert.Empty(parsed.Spans);
        Assert.Equal(ChunkCacheCodes.HashMismatch, Assert.Single(parsed.Warnings).Code);
        Assert.NotNull(parsed.Error);
    }

    /// <summary>Inline and manifest together are the existing conflict (ruling 2), on this surface too.</summary>
    [Fact]
    public void InlineAndManifest_AreAConflict()
    {
        string json = $$$"""
            {"messages":[{"role":"user","content":[{"type":"text","text":"abc","chunk":{{{Chunk("x", "abc")}}} }]}],
             "chunk_cache":{"v":1,"chunks":[{"id":"y","hash":"{{{Sha("abc")}}}","at":{"message":0,"part":0}}]}}
            """;
        var parsed = ParseOk(json);
        Assert.Empty(parsed.Spans);
        Assert.Equal(ChunkCacheCodes.ManifestConflict, Assert.Single(parsed.Warnings).Code);
    }

    /// <summary>An inline annotation alone (no <c>chunk_cache</c> object) is enough, and without the flag it is ignored (null parse), never an error.</summary>
    [Fact]
    public void InlineWithoutChunkCacheObject_IsAnnotated_AndFlagOffIgnoresIt()
    {
        string json = $$$"""{"messages":[{"role":"user","content":[{"type":"text","text":"abc","chunk":{{{Chunk("x", "abc")}}} }]}]}""";
        var request = Bind(json);
        Assert.True(ChunkAnnotationParser.HasAnnotation(request));
        Assert.Single(ChunkAnnotationParser.Parse(request, On, "")!.Spans);
        Assert.Null(ChunkAnnotationParser.Parse(request, ChunkCacheOptions.Disabled, ""));
        Assert.False(ChunkAnnotationParser.HasAnnotation(Bind("""{"messages":[{"role":"user","content":[{"type":"text","text":"abc"}]}]}""")));
    }

    /// <summary>
    /// The part's <c>chunk</c> survives the source-generated context both ways (read and write), nulls are not annotations, and a plain
    /// string content still reads as a string. CLAUDE.md's STJ rule: this goes through <c>ServerJsonContext</c>, the context the server uses.
    /// </summary>
    [Fact]
    public void RoundTrip_ThroughTheServerContext_KeepsThePartChunk()
    {
        const string json = """{"messages":[{"role":"user","content":[{"type":"text","text":"a","chunk":null},{"type":"text","text":"b","chunk":{"v":1,"id":"k"}}]}]}""";
        var request = Bind(json);
        var parts = request.Messages[0].Content!.Parts!;
        Assert.Null(parts[0].Chunk);
        Assert.Equal("k", parts[1].Chunk!.Value.GetProperty("id").GetString());

        string written = JsonSerializer.Serialize(request, ServerJsonContext.Default.ChatCompletionRequest);
        var again = JsonNode.Parse(written)!["messages"]![0]!["content"]!.AsArray();
        Assert.Null(again[0]!["chunk"]);
        Assert.Equal("k", again[1]!["chunk"]!["id"]!.GetValue<string>());

        var str = Bind("""{"messages":[{"role":"user","content":"plain"}]}""");
        Assert.Null(str.Messages[0].Content!.Parts);
        Assert.Equal("plain", str.Messages[0].Content!.Text);
    }

    // ---------------------------------------------------------------------------------------- the locator with image parts

    /// <summary>
    /// A message with an image renders its <c>ContentParts</c> (not <c>Content</c>), so the locator's sentinels must be written into the parts.
    /// With a template that renders each text part between markers and the image as one literal, an annotated span in the second text part is
    /// found at its rendered position. Kills: sentinels applied to <c>Content</c> only (the span is then not found in the render).
    /// </summary>
    [Fact]
    public void Locator_FindsASpan_InAMessageThatCarriesImageParts()
    {
        const string template = "{% for m in messages %}{% if m['content'] is string %}[{{ m['content'] }}]{% else %}{% for p in m['content'] %}"
            + "{% if p['type'] == 'text' %}<{{ p['text'] }}>{% else %}(IMG){% endif %}{% endfor %}{% endif %}{% endfor %}";
        var tpl = ServerPinTests.ServerTemplateFromText(template);
        const string span = "needle";
        // The span is "needle", bytes [7, 13) of part 2, addressed through the manifest.
        string json = $$$"""
            {"messages":[{"role":"user","content":[
              {"type":"text","text":"head "},
              {"type":"image_url","image_url":{"url":"{{{TinyPng}}}"}},
              {"type":"text","text":"before needle after"}]}],
             "chunk_cache":{"v":1,"chunks":[{"id":"n","hash":"{{{Sha(span)}}}","at":{"message":0,"part":2,"start":7,"end":13}}]}}
            """;
        var request = Bind(json);
        var parsed = ChunkAnnotationParser.Parse(request, On, "")!;
        var s = Assert.Single(parsed.Spans);

        var engine = RequestConverter.ToMessages(request.Messages);
        Assert.NotNull(engine[0].ContentParts);
        var options = DotLLM.Server.ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);
        string rendered = tpl.Apply(engine, options);
        Assert.Equal("<head >(IMG)<before needle after>", rendered);

        var located = ChunkSpanLocator.Locate(tpl, engine, options, rendered, parsed.Spans, 0);
        var l = Assert.Single(located.Located);
        Assert.Equal(span, rendered[l.RenderedStartChar..l.RenderedEndChar]);
        Assert.Empty(located.Warnings);
        Assert.Equal(s.Id, l.Span.Id);
    }
}
