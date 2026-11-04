using System.Text;
using System.Text.Json;
using DotLLM.Server;
using DotLLM.Server.ChunkCache;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// <see cref="ChunkSpanLocator"/>: chunk spans located in the rendered prompt (spec §5.3 item 5, contract ruling 20),
/// with the repo's real Qwen3-4B-Instruct-2507 and Llama-3.1 templates built exactly as the server builds them (Task 9's
/// <see cref="ServerPinTests.ServerTemplate"/>: GGUF factory, the #1339 guard tokenizer).
/// </summary>
public sealed class ChunkSpanLocatorTests
{
    private const string Qwen3 = ServerPinTests.Qwen3Fixture;
    private const string Llama31 = ServerPinTests.Llama31Fixture;
    private const string Altered = ChunkCacheCodes.TemplateAltered;

    private static readonly ChatTemplateOptions Options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);

    private static ChatMessage Msg(string role, string content) => new() { Role = role, Content = content };

    private static ChatMessage Tool(string content) => new() { Role = "tool", ToolCallId = "t1", Content = content };

    /// <summary>A span over the <paramref name="occurrence"/>-th copy of <paramref name="text"/> in engine message <paramref name="m"/>.</summary>
    private static ChunkSpanRef Span(IReadOnlyList<ChatMessage> messages, int m, string text, string id = "c", int occurrence = 0, int offset = 0)
    {
        string content = messages[m + offset].Content;
        int at = -1;
        for (int i = 0; i <= occurrence; i++)
            at = content.IndexOf(text, at + 1, StringComparison.Ordinal);
        Assert.True(at >= 0, $"'{text}' not in message {m}");
        return new ChunkSpanRef(m, 0, 0, Encoding.UTF8.GetByteCount(text), id, "sha256:-", null, false, false, text,
            new EngineSpan(m, at, at + text.Length));
    }

    private static (LocatedSpans Result, string Rendered) Locate(string fixture, IReadOnlyList<ChatMessage> messages,
        params ChunkSpanRef[] spans) => Locate(ServerPinTests.ServerTemplate(fixture), messages, 0, spans);

    private static (LocatedSpans Result, string Rendered) Locate(IChatTemplate template, IReadOnlyList<ChatMessage> messages,
        int offset, params ChunkSpanRef[] spans)
    {
        string rendered = template.Apply(messages, Options);
        return (ChunkSpanLocator.Locate(template, messages, Options, rendered, spans, offset), rendered);
    }

    private static string[] Codes(LocatedSpans r) => r.Warnings.Select(w => $"{w.Code}x{w.Count}").ToArray();

    /// <summary>
    /// Stripping the U+FDD0 markers from the located range gives the span text, and the range starts and ends on span
    /// characters (never on a marker). The markers inside follow the whole-message marking, so the range is NOT always
    /// <c>MarkUntrustedSpecials(text)</c>: a spelling crossing the span's end leaves its marker inside the range.
    /// </summary>
    private static void AssertExact(string rendered, LocatedSpan l)
    {
        string got = rendered[l.RenderedStartChar..l.RenderedEndChar];
        Assert.Equal(l.Span.Text, ControlTokenGuard.Strip(got));
        Assert.Equal(l.Span.Text[0], got[0]);
        Assert.Equal(l.Span.Text[^1], got[^1]);
    }

    // ------------------------------------------------------------------------------------------------ plan Step 1

    [Fact]
    public void Qwen3_MidPartSpan_Located()
    {
        const string code = "public int Add(int a, int b) => a + b;\n";
        var messages = new[] { Msg("system", "sys"), Msg("user", "// src/A.cs (lines 1-1)\n" + code + "explain") };
        var (r, rendered) = Locate(Qwen3, messages, Span(messages, 1, code));

        var l = Assert.Single(r.Located);
        Assert.Empty(r.Warnings);
        AssertExact(rendered, l);
        Assert.Equal(rendered.IndexOf(code, StringComparison.Ordinal), l.RenderedStartChar);
    }

    /// <summary>Qwen3 renders tool content verbatim (Task 9, F4): a span in a tool result is located.</summary>
    [Fact]
    public void Qwen3_ToolResultSpan_Located()
    {
        const string code = "class A\n{\n    string S = \"q\\n\";\n}\n";
        var messages = new[] { Msg("user", "read A"), Tool(code) };
        var (r, rendered) = Locate(Qwen3, messages, Span(messages, 1, code));
        AssertExact(rendered, Assert.Single(r.Located));
        Assert.Empty(r.Warnings);
    }

    /// <summary>
    /// Llama-3.1 trims user content (F5): a span touching either edge of the message with whitespace there is not in the
    /// render as sent, so it is ignored. The same span text away from the edge is located.
    /// </summary>
    [Theory]
    [InlineData("class A {}\n", "", "")]               // whole part, trailing newline trimmed
    [InlineData("class A {}\n", "explain:\n", "")]     // at the end
    [InlineData("\n  class A {}", "", "\nexplain")]    // at the start, leading whitespace trimmed
    public void Llama31_SpanAtPartEdge_TrimmedIsTemplateAltered(string span, string before, string after)
    {
        var messages = new[] { Msg("user", before + span + after) };
        var (r, _) = Locate(Llama31, messages, Span(messages, 0, span));
        Assert.Empty(r.Located);
        Assert.Equal([$"{Altered}x1"], Codes(r));

        var inside = new[] { Msg("user", "glue:\n" + span + "\nmore") };
        var (ok, rendered) = Locate(Llama31, inside, Span(inside, 0, span));
        AssertExact(rendered, Assert.Single(ok.Located));
    }

    /// <summary>
    /// The advised layout (spec §5.3 item 5): a glue header ending in a newline, then the span ending in a newline, then
    /// more text. Llama-3.1's render starts with the BOS text, so the offsets must count it.
    /// </summary>
    [Fact]
    public void Llama31_SpanAfterGlueEndingNewline_Located()
    {
        const string code = "public sealed class A\n{\n    int X;\n}\n";
        var messages = new[] { Msg("system", "be brief"), Msg("user", "// src/A.cs (lines 1-4)\n" + code + "explain A") };
        var (r, rendered) = Locate(Llama31, messages, Span(messages, 1, code));

        Assert.StartsWith("<|begin_of_text|>", rendered, StringComparison.Ordinal);
        var l = Assert.Single(r.Located);
        AssertExact(rendered, l);
        Assert.Equal(rendered.IndexOf(code, StringComparison.Ordinal), l.RenderedStartChar);
    }

    [Theory]
    [InlineData(Qwen3)]
    [InlineData(Llama31)]
    public void TwoSpansSamePart_BothLocated(string fixture)
    {
        // Two separated spans, and two adjacent spans with no glue between them.
        var messages = new[] { Msg("user", "h\nA1 a\nB2 b\nC3 c\nD4 d\nend") };
        var spans = new[]
        {
            Span(messages, 0, "A1 a\n", "a"), Span(messages, 0, "C3 c\n", "c"), Span(messages, 0, "D4 d\n", "d"),
        };
        var (r, rendered) = Locate(fixture, messages, spans);
        Assert.Equal(["a", "c", "d"], r.Located.Select(l => l.Span.Id));
        Assert.Empty(r.Warnings);
        foreach (var l in r.Located)
            AssertExact(rendered, l);
        Assert.Equal(r.Located[1].RenderedEndChar, r.Located[2].RenderedStartChar);
    }

    /// <summary>
    /// Caller text cannot fake a sentinel: a forged copy of the exact sentinel (nonce known to the test only) makes the
    /// sentinel appear twice and the span is ignored, never located at the forged copy; text full of the sentinel frame
    /// characters and sentinel-shaped strings with another nonce changes nothing.
    /// </summary>
    [Fact]
    public void SentinelInUserText_CannotCollide()
    {
        var template = ServerPinTests.ServerTemplate(Qwen3);
        const string code = "int F() => 1;\n";
        string forged = "n0:0\"\\\n";
        var messages = new[] { Msg("user", "x " + forged + " y"), Msg("user", "a\n" + code + "b") };
        string rendered = template.Apply(messages, Options);
        var r = ChunkSpanLocator.Locate(template, messages, Options, rendered, [Span(messages, 1, code)], 0, "n0");
        Assert.Empty(r.Located);
        Assert.Equal([$"{Altered}x1"], Codes(r));

        // A forged sentinel inside the span itself is replaced with the span: it is ordinary span text, located exactly.
        var inSpan = new[] { Msg("user", "a\n" + code + forged + "\nb") };
        string rendered2 = template.Apply(inSpan, Options);
        var r2 = ChunkSpanLocator.Locate(template, inSpan, Options, rendered2, [Span(inSpan, 0, code + forged)], 0, "n0");
        var l2 = Assert.Single(r2.Located);
        AssertExact(rendered2, l2);
        Assert.Equal(rendered2.IndexOf(code, StringComparison.Ordinal), l2.RenderedStartChar);

        // A forged sentinel for a second span index in caller text: span 1's sentinel appears twice, span 0 is unaffected.
        var two = new[] { Msg("user", "a\n" + code + "b\nint G() => 2;\nc"), Msg("user", "n0:1\"\\\n") };
        string rendered4 = template.Apply(two, Options);
        var r4 = ChunkSpanLocator.Locate(template, two, Options, rendered4,
            [Span(two, 0, code, "zero"), Span(two, 0, "int G() => 2;\n", "one")], 0, "n0");
        Assert.Equal(["zero"], r4.Located.Select(l => l.Span.Id));
        Assert.Equal([$"{Altered}x1"], Codes(r4));

        // Frame characters and other nonces everywhere, random nonce: located at the right place.
        string noise = "n0:0\"\\\n" + Guid.NewGuid().ToString("N") + ":0\"\\\n";
        var noisy = new[] { Msg("user", noise + "\n" + noise + code + noise + "\n" + noise) };
        var (ok, rendered3) = Locate(template, noisy, 0, Span(noisy, 0, noise + code + noise));
        AssertExact(rendered3, Assert.Single(ok.Located));
    }

    /// <summary>
    /// Property: over random conversations (system / user / assistant / tool, non-BMP text, CRLF, special spellings,
    /// several spans per message), every located range in the real render equals the marked span text, and on Qwen3 every
    /// span is located. On Llama-3.1 spans in user / system messages away from the edges are located and every tool span
    /// is altered (attribution: a broken tool sentinel never costs a user span).
    /// </summary>
    [Theory]
    [InlineData(Qwen3, 1)]
    [InlineData(Qwen3, 2)]
    [InlineData(Llama31, 3)]
    [InlineData(Llama31, 4)]
    public void RenderedOffsets_SubstringEqualsSpanText(string fixture, int seed)
    {
        var template = ServerPinTests.ServerTemplate(fixture);
        string[] pieces = ["alpha", "beta ", "\n", "\r\n", " ", "é", "😀", "x=\"q\"", "\\", "\t", "<think>", "</think>",
            "<|im_end|>", "<tool_call>", "<|eot_id|>", "<|start_header_id|>", "{", "}", "int", ";"];
        var rng = new Random(seed);
        int located = 0, altered = 0;
        for (int round = 0; round < 60; round++)
        {
            var messages = new List<ChatMessage>();
            var spans = new List<ChunkSpanRef>();
            var expectLocated = new List<bool>();
            int count = rng.Next(1, 6);
            for (int m = 0; m < count; m++)
            {
                string role = m == 0 && rng.Next(3) == 0 ? "system" : rng.Next(3) == 0 ? "tool" : "user";
                var sb = new StringBuilder("hdr:");
                var ranges = new List<(int, int)>();
                int nSpans = rng.Next(0, 4);
                for (int s = 0; s < nSpans; s++)
                {
                    sb.Append(pieces[rng.Next(pieces.Length)]).Append("|g|");
                    int start = sb.Length;
                    int len = rng.Next(1, 8);
                    for (int p = 0; p < len; p++)
                        sb.Append(pieces[rng.Next(pieces.Length)]);
                    ranges.Add((start, sb.Length));
                }
                sb.Append("|tail");
                string content = sb.ToString();
                messages.Add(role == "tool" ? Tool(content) : Msg(role, content));
                foreach (var (a, b) in ranges)
                {
                    spans.Add(new ChunkSpanRef(m, 0, 0, 1, $"s{spans.Count}", "sha256:-", null, false, false, content[a..b],
                        new EngineSpan(m, a, b)));
                    expectLocated.Add(fixture == Qwen3 || role != "tool");
                }
            }

            string rendered = template.Apply(messages, Options);
            var r = ChunkSpanLocator.Locate(template, messages, Options, rendered, spans, 0);
            var ids = r.Located.Select(l => l.Span.Id).ToHashSet();
            for (int k = 0; k < spans.Count; k++)
                Assert.True(expectLocated[k] == ids.Contains(spans[k].Id), $"{fixture} seed {seed} round {round} span {k}");
            foreach (var l in r.Located)
                AssertExact(rendered, l);
            located += r.Located.Count;
            altered += spans.Count - r.Located.Count;
        }
        Assert.True(located > 50, $"only {located} located");
        if (fixture == Llama31)
            Assert.True(altered > 10, $"only {altered} altered");
    }

    /// <summary>
    /// A span holding special-token spellings (<c>&lt;think&gt;</c>, <c>&lt;|im_end|&gt;</c>) is rendered as the marked copy
    /// (#1339): it is located by comparing with the markers skipped, and its range is the marked text.
    /// </summary>
    [Fact]
    public void SpecialSpellingSpan_LocatedViaMarkedText()
    {
        const string code = "// <think> is a tag\nvar s = \"<|im_end|>\";\n";
        var messages = new[] { Msg("user", "file:\n" + code + "done") };
        var (r, rendered) = Locate(Qwen3, messages, Span(messages, 0, code));

        Assert.DoesNotContain(code, rendered, StringComparison.Ordinal); // the render holds the marked copy only
        var l = Assert.Single(r.Located);
        Assert.Contains(ControlTokenGuard.Marker, rendered[l.RenderedStartChar..l.RenderedEndChar]);
        AssertExact(rendered, l);
        // No spelling crosses this span's edges, so here the range is exactly the marked copy.
        Assert.Equal(ServerPinTests.Tok.MarkUntrustedSpecials(code), rendered[l.RenderedStartChar..l.RenderedEndChar]);
    }

    /// <summary>
    /// Review S1: a special spelling crossing a span edge. The whole message is marked, so the marker lands after the
    /// spelling's first character, wherever the span boundary is.
    /// <list type="bullet">
    /// <item>Crossing the END (<c>q &lt;|im_</c> of <c>q &lt;|im_end|&gt; z</c>): located; the range holds the marker
    /// (<c>q &lt;﷐|im_</c>), which <c>MarkUntrustedSpecials</c> of the span alone would not insert.</item>
    /// <item>Crossing the START (<c>|im_end|&gt; z</c>): the marker sits right before the span's first character and is
    /// NOT in the range.</item>
    /// </list>
    /// Character location only: such a span's tokens differ from the prompt's at the seam (the token check, Task 13).
    /// </summary>
    [Theory]
    [InlineData(Qwen3)]
    [InlineData(Llama31)]
    public void SpellingStraddlingSpanEdge_LocatedByCharacters(string fixture)
    {
        var messages = new[] { Msg("user", "h\nq <|im_end|> z\nmore") };
        var (r, rendered) = Locate(fixture, messages, Span(messages, 0, "q <|im_", "end"));
        var l = Assert.Single(r.Located);
        AssertExact(rendered, l);
        Assert.Equal("q <﷐|im_", rendered[l.RenderedStartChar..l.RenderedEndChar]);
        Assert.NotEqual(ServerPinTests.Tok.MarkUntrustedSpecials(l.Span.Text), rendered[l.RenderedStartChar..l.RenderedEndChar]);

        var (r2, rendered2) = Locate(fixture, messages, Span(messages, 0, "|im_end|> z\n", "start"));
        var l2 = Assert.Single(r2.Located);
        AssertExact(rendered2, l2);
        Assert.Equal(ControlTokenGuard.Marker, rendered2[l2.RenderedStartChar - 1]);
        Assert.Equal("|im_end|> z\n", rendered2[l2.RenderedStartChar..l2.RenderedEndChar]);
    }

    // ------------------------------------------------------------------------------------------------ assistant turns (review S2)

    /// <summary>
    /// Qwen3 assistant turns. The guard leaves the template's own reasoning tags LIVE in assistant content (#1339
    /// exemption), and the template rewrites content that contains <c>&lt;/think&gt;</c> (reasoning split off, leading
    /// newlines stripped). Pinned today:
    /// <list type="bullet">
    /// <item>a mid-content span with a live <c>&lt;think&gt;</c> is LOCATED by characters, with no marker in the range;
    /// its tokens differ from the chunk's (a control token in the prompt, plain text in the store), so the token check
    /// (Task 13) is the only gate;</item>
    /// <item>a span in the reasoning part of a historical turn is dropped by the template: altered;</item>
    /// <item>a span in the answer part, away from the stripped newlines, is located;</item>
    /// <item>an answer span starting with the newlines the template strips: altered.</item>
    /// </list>
    /// </summary>
    [Fact]
    public void Qwen3_AssistantSpans()
    {
        var live = new[]
        {
            Msg("user", "q1"), Msg("assistant", "ok:\n<think> stays live here\nend"), Msg("user", "q2"),
        };
        var (r, rendered) = Locate(Qwen3, live, Span(live, 1, "<think> stays live here\n"));
        var l = Assert.Single(r.Located);
        AssertExact(rendered, l);
        Assert.DoesNotContain(ControlTokenGuard.Marker, rendered[l.RenderedStartChar..l.RenderedEndChar]);

        var split = new[]
        {
            Msg("user", "q1"), Msg("assistant", "<think>\nreason span\n</think>\n\nanswer span\ntail"), Msg("user", "q2"),
        };
        var (r2, rendered2) = Locate(Qwen3, split,
            Span(split, 1, "reason span\n", "reason"), Span(split, 1, "answer span\n", "answer"));
        Assert.Equal(["answer"], r2.Located.Select(x => x.Span.Id));
        Assert.Equal([$"{Altered}x1"], Codes(r2));
        AssertExact(rendered2, r2.Located[0]);

        var (r3, _) = Locate(Qwen3, split, Span(split, 1, "\n\nanswer span\n", "stripped"));
        Assert.Empty(r3.Located);
        Assert.Equal([$"{Altered}x1"], Codes(r3));
    }

    /// <summary>Llama-3.1 assistant content is trimmed like user content: an edge span is altered, a mid span located.</summary>
    [Fact]
    public void Llama31_AssistantSpans_TrimmedEdgeAltered_MidLocated()
    {
        var messages = new[] { Msg("user", "q"), Msg("assistant", "\n edge span\nmid span\nend"), Msg("user", "q2") };
        var (r, rendered) = Locate(Llama31, messages,
            Span(messages, 1, "\n edge span\n", "edge"), Span(messages, 1, "mid span\n", "mid"));
        Assert.Equal(["mid"], r.Located.Select(x => x.Span.Id));
        Assert.Equal([$"{Altered}x1"], Codes(r));
        AssertExact(rendered, r.Located[0]);
    }

    /// <summary>
    /// Llama-3.1 with tools (review N5): the template moves the first user message after the tool JSON
    /// (<c>tools_in_user_message</c>) and trims it. A span inside it is still located; an edge span is altered.
    /// </summary>
    [Fact]
    public void Llama31_ToolsInFirstUserMessage_SpanLocated()
    {
        var template = ServerPinTests.ServerTemplate(Llama31);
        var options = ReasoningSupport.BuildTemplateOptions(
            [new ToolDefinition("read", "Read a file", """{"type":"object","properties":{"path":{"type":"string"}}}""")],
            null, null, null, false);
        var messages = new[] { Msg("user", "  edge span\nglue:\nmid span\nend") };
        string rendered = template.Apply(messages, options);
        Assert.Contains("Given the following functions", rendered, StringComparison.Ordinal);
        var r = ChunkSpanLocator.Locate(template, messages, options, rendered,
            [Span(messages, 0, "  edge span\n", "edge"), Span(messages, 0, "mid span\n", "mid")], 0);
        Assert.Equal(["mid"], r.Located.Select(x => x.Span.Id));
        Assert.Equal([$"{Altered}x1"], Codes(r));
        AssertExact(rendered, r.Located[0]);
    }

    /// <summary>A caller U+FDD0 inside a span is dropped at encode (Task 9, F10): that span is ignored, its neighbour is not.</summary>
    [Fact]
    public void CallerFdd0InSpan_Ignored()
    {
        var messages = new[] { Msg("user", "h\na﷐b\nok span\nend") };
        var (r, rendered) = Locate(Qwen3, messages, Span(messages, 0, "a﷐b\n", "fdd0"), Span(messages, 0, "ok span\n", "ok"));
        Assert.Equal(["ok"], r.Located.Select(l => l.Span.Id));
        Assert.Equal([$"{Altered}x1"], Codes(r));
        AssertExact(rendered, r.Located[0]);
    }

    /// <summary>
    /// Llama-3.1 JSON-encodes tool content (F6): every tool-result span is altered, even one whose own characters need no
    /// escaping (the sentinel's probe characters break), while a user span in the same request is located.
    /// </summary>
    [Theory]
    [InlineData("class A\n{\n    string S = \"q\";\n}\n")]
    [InlineData("plain words only")]
    public void Llama31_ToolResultSpan_AlwaysTemplateAltered(string code)
    {
        var messages = new[] { Msg("user", "read:\nuser span\nnow"), Tool("// A.cs\n" + code + "\n// end"), Tool(code) };
        var (r, rendered) = Locate(Llama31, messages,
            Span(messages, 1, code, "tool-mid"), Span(messages, 2, code, "tool-whole"), Span(messages, 0, "user span\n", "user"));
        Assert.Equal(["user"], r.Located.Select(l => l.Span.Id));
        Assert.Equal([$"{Altered}x2"], Codes(r));
        AssertExact(rendered, r.Located[0]);
    }

    /// <summary>
    /// Attribution: a trimmed edge span costs only itself. Good spans after it (proven from the end), and good spans
    /// between two trimmed ones (tried again in a second render without them), are located.
    /// </summary>
    [Theory]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void Llama31_GoodSpansNextToTrimmedSpans_Located(bool leadingBad, bool trailingBad)
    {
        var messages = new List<ChatMessage>();
        var spans = new List<ChunkSpanRef>();
        if (leadingBad)
        {
            messages.Add(Msg("user", "  lead span\nrest"));
            spans.Add(Span(messages, messages.Count - 1, "  lead span\n", "lead"));
        }
        messages.Add(Msg("user", "a\ngood one\nb"));
        spans.Add(Span(messages, messages.Count - 1, "good one\n", "good1"));
        messages.Add(Msg("user", "x\ngood two\ny"));
        spans.Add(Span(messages, messages.Count - 1, "good two\n", "good2"));
        if (trailingBad)
        {
            messages.Add(Msg("user", "m\ntrail span\n"));
            spans.Add(Span(messages, messages.Count - 1, "trail span\n", "trail"));
        }

        var (r, rendered) = Locate(ServerPinTests.ServerTemplate(Llama31), messages, 0, [.. spans]);
        Assert.Equal(["good1", "good2"], r.Located.Select(l => l.Span.Id));
        Assert.Equal([$"{Altered}x{(leadingBad ? 1 : 0) + (trailingBad ? 1 : 0)}"], Codes(r));
        foreach (var l in r.Located)
            AssertExact(rendered, l);
    }

    // ------------------------------------------------------------------------------------------------ end to end, ruling 31

    /// <summary>
    /// Anthropic end to end: parser output (engine positions after the converter's joins and tool_result split) located in
    /// the real render. Qwen3 locates the text block and the tool result; Llama-3.1 only the text block.
    /// </summary>
    [Theory]
    [InlineData(Qwen3, 2)]
    [InlineData(Llama31, 1)]
    public void Anthropic_ParsedSpans_LocatedInTheRender(string fixture, int expected)
    {
        const string code = "class A {}\n";
        string body = $$$"""
            {"max_tokens":8,"messages":[
              {"role":"user","content":[{"type":"text","text":"// A.cs\n"},{"type":"text","text":"{{{Esc(code)}}}","chunk":{"v":1,"id":"text","hash":"{{{ChunkAnnotationParserTests.Sha(code)}}}"}},{"type":"text","text":"explain"}]},
              {"role":"assistant","content":[{"type":"tool_use","id":"t","name":"read","input":{}}]},
              {"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":[{"type":"text","text":"{{{Esc(code)}}}","chunk":{"v":1,"id":"tool","hash":"{{{ChunkAnnotationParserTests.Sha(code)}}}"}}]},{"type":"text","text":"go"}]}]}
            """;
        var request = JsonSerializer.Deserialize(body, ServerJsonContext.Default.AnthropicMessagesRequest)!;
        var parsed = ChunkAnnotationParser.Parse(request, new ChunkCacheOptions(true, 16), "")!;
        Assert.Equal(2, parsed.Spans.Count);

        var template = ServerPinTests.ServerTemplate(fixture);
        var messages = AnthropicConverter.ToMessages(request);
        string rendered = template.Apply(messages, Options);
        var r = ChunkSpanLocator.Locate(template, messages, Options, rendered, parsed.Spans, 0);
        Assert.Equal(expected, r.Located.Count);
        Assert.Equal("text", r.Located[0].Span.Id);
        foreach (var l in r.Located)
            AssertExact(rendered, l);
    }

    private static string Esc(string s) => JsonEncodedText.Encode(s).ToString();

    /// <summary>
    /// Ruling 31: the parser's engine indices are counted before <see cref="ProfileSystemPrompt.Apply"/> prepends a profile
    /// system message; the caller passes the offset. With it, the span is located; without it, the span is not where its
    /// position says (<c>chunk_span_invalid</c>), never a wrong location.
    /// </summary>
    [Fact]
    public void ProfileSystemPrompt_OffsetIsAdded()
    {
        const string code = "int F() => 1;\n";
        string body = $$$"""
            {"messages":[{"role":"user","content":"a\n{{{Esc(code)}}}b"}],
             "chunk_cache":{"chunks":[{"id":"c","hash":"{{{ChunkAnnotationParserTests.Sha(code)}}}","at":{"message":0,"part":0,"start":2,"end":{{{2 + code.Length}}}}}]}}
            """;
        var request = JsonSerializer.Deserialize(body, ServerJsonContext.Default.ChatCompletionRequest)!;
        var parsed = ChunkAnnotationParser.Parse(request, new ChunkCacheOptions(true, 16), "")!;
        var engine = RequestConverter.ToMessages(request.Messages).ToArray();
        var messages = ProfileSystemPrompt.Apply("You are a profile.", engine);
        Assert.Equal(engine.Length + 1, messages.Length);

        var template = ServerPinTests.ServerTemplate(Qwen3);
        string rendered = template.Apply(messages, Options);
        var withOffset = ChunkSpanLocator.Locate(template, messages, Options, rendered, parsed.Spans, messages.Length - engine.Length);
        AssertExact(rendered, Assert.Single(withOffset.Located));

        var without = ChunkSpanLocator.Locate(template, messages, Options, rendered, parsed.Spans, 0);
        Assert.Empty(without.Located);
        Assert.Equal([$"{ChunkCacheCodes.SpanInvalid}x1"], Codes(without));
    }

    // ------------------------------------------------------------------------------------------------ hostile input, cost

    /// <summary>Positions that do not hold the span text, out-of-range messages, and overlaps: invalid, never a throw.</summary>
    [Fact]
    public void InconsistentPositions_AreSpanInvalid_NotAThrow()
    {
        var messages = new[] { Msg("user", "hello world\nmore") };
        ChunkSpanRef Raw(int m, int a, int b, string text, string id) =>
            new(m, 0, 0, 1, id, "sha256:-", null, false, false, text, new EngineSpan(m, a, b));
        var (r, rendered) = Locate(Qwen3, messages,
            Raw(5, 0, 5, "hello", "out-of-range"), Raw(-1, 0, 5, "hello", "negative"), Raw(0, 0, 5, "world", "wrong-text"),
            Raw(0, 3, 99, "lo world", "past-end"), Raw(0, 4, 2, "", "reversed"),
            Raw(0, 0, 5, "hello", "first"), Raw(0, 3, 8, "lo wo", "overlaps-first"));
        Assert.Equal(["first"], r.Located.Select(l => l.Span.Id));
        Assert.Equal([$"{ChunkCacheCodes.SpanInvalid}x6"], Codes(r));
        AssertExact(rendered, r.Located[0]);
    }

    /// <summary>A template that throws, drops or duplicates content on the sentinel render: every span ignored, no throw.</summary>
    [Fact]
    public void HostileTemplates_IgnoreSpans_NeverThrow()
    {
        var messages = new[] { Msg("user", "a\nspan\nb") };
        var spans = new[] { Span(messages, 0, "span\n") };
        var real = ServerPinTests.ServerTemplate(Qwen3);
        string rendered = real.Apply(messages, Options);

        foreach (IChatTemplate t in new IChatTemplate[]
                 {
                     new FakeTemplate(_ => throw new InvalidOperationException("boom")),
                     new FakeTemplate(_ => throw new JinjaException("raise_exception")),
                     new FakeTemplate(_ => "constant"),
                     new FakeTemplate(ms => ms[0].Content + ms[0].Content),
                 })
        {
            var r = ChunkSpanLocator.Locate(t, messages, Options, rendered, spans, 0);
            Assert.Empty(r.Located);
            Assert.Equal([$"{Altered}x1"], Codes(r));
        }

        // A lone surrogate in a span (in-process callers only; HTTP cannot carry one): handled like any other text.
        var lone = new[] { Msg("user", "a\nx\uD800y\nb") };
        var (lr, lrendered) = Locate(Qwen3, lone, Span(lone, 0, "x\uD800y\n"));
        AssertExact(lrendered, Assert.Single(lr.Located));
    }

    /// <summary>
    /// Cost bound (render counts; no wall-clock asserts): 4,096 spans on a 4 MiB message are located with ONE sentinel
    /// render (not one per span);
    /// 4,096 Llama-3.1 messages each with an edge span (each altered by the trim) stop after
    /// <see cref="ChunkSpanLocator.MaxSentinelRenders"/> renders.
    /// </summary>
    [Fact]
    public void CostBound_ManySpans_BoundedRenders()
    {
        const int n = 4096;
        var line = new string('x', 1000) + "\n";
        var sb = new StringBuilder("hdr\n");
        var ranges = new List<(int, int)>();
        for (int i = 0; i < n; i++)
        {
            ranges.Add((sb.Length, sb.Length + line.Length));
            sb.Append(line);
        }
        sb.Append("tail");
        var messages = new[] { Msg("user", sb.ToString()) };
        var spans = ranges.Select((r, i) => new ChunkSpanRef(0, 0, 0, 1, $"s{i}", "sha256:-", null, false, false, line,
            new EngineSpan(0, r.Item1, r.Item2))).ToArray();

        var qwen = new FakeTemplate(ServerPinTests.ServerTemplate(Qwen3));
        string rendered = qwen.Inner!.Apply(messages, Options);
        var r = ChunkSpanLocator.Locate(qwen, messages, Options, rendered, spans, 0);
        Assert.Equal(n, r.Located.Count);
        Assert.Equal(1, qwen.Calls);

        var many = Enumerable.Range(0, n).Select(i => Msg("user", $"m{i}\nspan {i}\n")).ToArray();
        var edgeSpans = many.Select((m, i) => Span(many, i, $"span {i}\n", $"e{i}")).ToArray();
        var llama = new FakeTemplate(ServerPinTests.ServerTemplate(Llama31));
        string lrendered = llama.Inner!.Apply(many, Options);
        var lr = ChunkSpanLocator.Locate(llama, many, Options, lrendered, edgeSpans, 0);
        Assert.Empty(lr.Located);
        Assert.Equal([$"{Altered}x{n}"], Codes(lr));
        Assert.Equal(ChunkSpanLocator.MaxSentinelRenders, llama.Calls);
    }

    /// <summary>Delegating or scripted template that counts its renders.</summary>
    private sealed class FakeTemplate : IChatTemplate
    {
        private readonly Func<IReadOnlyList<ChatMessage>, string>? _render;

        public FakeTemplate(IChatTemplate inner) => Inner = inner;

        public FakeTemplate(Func<IReadOnlyList<ChatMessage>, string> render) => _render = render;

        public IChatTemplate? Inner { get; }

        public int Calls { get; private set; }

        public string Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options)
        {
            Calls++;
            return Inner is not null ? Inner.Apply(messages, options) : _render!(messages);
        }
    }
}
