using System.Net;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Models.Gguf;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tests.Unit.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.Bpe;
using DotLLM.Tokenizers.ChatTemplates;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// Chunk-cache Task 9 (S1): characterisation pins of what the server does TODAY with the three things the
/// <c>chunk_cache</c> wire convention leans on — how multi-part content reaches the chat template, how special-token
/// spellings inside message content are tokenized, and what happens to fields the server does not know. No production
/// code changes; each fact is the observed behaviour of unmodified <c>dev</c>, and later tasks (span location, the
/// annotation parser, exact/shadow) depend on it. If one of these starts failing, the convention's assumptions moved:
/// update <c>docs/CHUNK_CACHE.md</c> ("Pinned server behaviour") together with the test.
/// </summary>
/// <remarks>
/// Templates are the GGUF <c>tokenizer.chat_template</c> strings of the HF-cache models
/// <c>bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF</c> and <c>bartowski/Meta-Llama-3.1-8B-Instruct-GGUF</c>, already in the
/// repo as fixtures and verified byte-identical to the GGUF metadata (see <see cref="RealGguf_TemplateFixture_IsTheGgufMetadataTemplate"/>).
/// The tokenizer is a synthetic character-level vocabulary carrying the two families' real special-token spellings, so
/// one plain character is one token and one recognised special spelling is one token: counts discriminate exactly.
/// </remarks>
public sealed class ServerPinTests
{
    // ------------------------------------------------------------------------------------------------ fixtures

    internal static string FixtureTemplate(string name) => File.ReadAllText(Path.Combine(
        AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", name));

    internal const string Qwen3Fixture = "qwen3-4b-instruct-2507-chat-template.jinja";
    internal const string Llama31Fixture = "llama-3.1-chat-template.jinja";

    // The special spellings (GGUF token type 3/4, length > 1) of the two models, from their GGUF vocabularies.
    internal static readonly string[] Specials =
    [
        // Qwen3-4B-Instruct-2507 (all 26)
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
        "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
        "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
        "<think>", "</think>",
        // Llama-3.1-8B-Instruct (the named ones; the 248 reserved_special_token_N are omitted)
        "<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eom_id|>", "<|eot_id|>",
        "<|python_tag|>", "<|finetune_right_pad_id|>",
    ];

    private static readonly Lazy<(BpeTokenizer Tok, BpeTokenizer Plain, Dictionary<string, int> Ids)> Vocab = new(BuildVocab);

    private static (BpeTokenizer, BpeTokenizer, Dictionary<string, int>) BuildVocab()
    {
        var tokens = new List<string> { "<unk>" };
        var types = new List<int> { 2 };
        for (char c = ' '; c <= '~'; c++)
        {
            tokens.Add(c == ' ' ? "▁" : c.ToString());
            types.Add(1);
        }
        tokens.Add("\n");
        types.Add(1);
        var ids = new Dictionary<string, int>();
        foreach (string s in Specials)
        {
            ids[s] = tokens.Count;
            tokens.Add(s);
            types.Add(3);
        }
        var scores = new float[tokens.Count];
        var tok = BpeTokenizer.CreateSentencePiece(tokens.ToArray(), scores, types.ToArray(),
            bosId: ids["<|begin_of_text|>"], eosId: ids["<|im_end|>"], addBosSpace: false);
        // Same vocabulary with every special demoted to an ordinary piece: the oracle for "tokenized as plain text".
        var plain = BpeTokenizer.CreateSentencePiece(tokens.ToArray(), scores, tokens.Select(_ => 1).ToArray(),
            bosId: 0, eosId: 0, addBosSpace: false);
        return (tok, plain, ids);
    }

    internal static BpeTokenizer Tok => Vocab.Value.Tok;
    private static BpeTokenizer PlainTok => Vocab.Value.Plain;
    private static int Id(string spelling) => Vocab.Value.Ids[spelling];

    private static GgufMetadata Metadata(string template)
    {
        var data = new GgufTestData(version: 3);
        data.AddString("tokenizer.chat_template", template);
        using var stream = new MemoryStream(data.Build());
        using var reader = new BinaryReader(stream);
        var header = GgufReader.ReadHeader(reader);
        return new GgufMetadata(GgufReader.ReadMetadata(reader, header));
    }

    /// <summary>The template exactly as <c>ServerStartup</c> builds it: GGUF factory, model tokenizer as the #1339 guard.</summary>
    internal static JinjaChatTemplate ServerTemplate(string fixture)
        => GgufChatTemplateFactory.TryCreate(Metadata(FixtureTemplate(fixture)), Tok, Architecture.Llama)!;

    /// <summary>The render call both chat endpoints make (no tools, no reasoning switches, generation prompt on).</summary>
    private static string Render(IChatTemplate template, IReadOnlyList<ChatMessage> messages)
    {
        var options = ReasoningSupport.BuildTemplateOptions(null, null, null, null, false);
        Assert.True(ReasoningSupport.TryApply(template, messages, options, out string prompt, out string? error, out _), error);
        return prompt;
    }

    private static ChatMessage Msg(string role, string content) => new() { Role = role, Content = content };
    private static int CountOf(int[] ids, int id) => ids.Count(x => x == id);

    private static AnthropicMessagesRequest ParseAnthropic(string json)
        => JsonSerializer.Deserialize(json, ServerJsonContext.Default.AnthropicMessagesRequest)!;

    private static ChatCompletionRequest ParseOpenAi(string json)
        => JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;

    // ------------------------------------------------------------------------------------------------ (a) multi-part content

    /// <summary>
    /// Anthropic <c>text</c> blocks of one message are concatenated by <see cref="AnthropicConverter"/> BEFORE the template
    /// sees them. Pin: the separator is one <c>'\n'</c> between blocks (spec assumption was "no separator").
    /// </summary>
    [Fact]
    public void Anthropic_TextBlocks_AreJoinedWithOneNewline()
    {
        var request = ParseAnthropic("""
            {"max_tokens":8,"messages":[{"role":"user","content":[
              {"type":"text","text":"// src/A.cs (lines 1-2)\n"},
              {"type":"text","text":"class A {}\n"},
              {"type":"text","text":"explain"}]}]}
            """);

        var messages = AnthropicConverter.ToMessages(request);

        Assert.Single(messages);
        Assert.Equal("// src/A.cs (lines 1-2)\n" + "\n" + "class A {}\n" + "\n" + "explain", messages[0].Content);
    }

    /// <summary>
    /// The separator is written only when text has already accumulated, so an EMPTY block contributes no separator while
    /// nothing precedes it, and one separator (plus its empty text) after. Offsets of later parts depend on this.
    /// </summary>
    [Theory]
    [InlineData("""[{"type":"text","text":""},{"type":"text","text":"b"}]""", "b")]
    [InlineData("""[{"type":"text","text":"a"},{"type":"text","text":""},{"type":"text","text":"b"}]""", "a\n\nb")]
    [InlineData("""[{"type":"text","text":"a"},{"type":"text","text":""}]""", "a\n")]
    public void Anthropic_EmptyTextBlock_SeparatorRule(string content, string expected)
    {
        var request = ParseAnthropic($$"""{"max_tokens":8,"messages":[{"role":"user","content":{{content}}}]}""");
        Assert.Equal(expected, AnthropicConverter.ToMessages(request)[0].Content);
    }

    /// <summary>
    /// A <c>tool_result</c> block (where file reads land in an agent loop) ends the current turn: text before it becomes
    /// its own message, the result becomes a <c>tool</c> message whose array content is joined with <c>'\n'</c>, and text
    /// after it starts a new message of the original role. So an Anthropic (message, part) index does NOT map 1:1 onto
    /// the engine's message list.
    /// </summary>
    [Fact]
    public void Anthropic_ToolResult_ArrayContentJoinedWithNewline_AndSplitsTheTurn()
    {
        var request = ParseAnthropic("""
            {"max_tokens":8,"messages":[
              {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"read","input":{"path":"A.cs"}}]},
              {"role":"user","content":[
                {"type":"text","text":"see:"},
                {"type":"tool_result","tool_use_id":"t1","content":[{"type":"text","text":"A1"},{"type":"text","text":"A2"}]},
                {"type":"text","text":"done"}]}]}
            """);

        var messages = AnthropicConverter.ToMessages(request);

        Assert.Equal(new[] { "assistant", "user", "tool", "user" }, messages.Select(m => m.Role));
        Assert.Equal("see:", messages[1].Content);
        Assert.Equal("A1\nA2", messages[2].Content);
        Assert.Equal("t1", messages[2].ToolCallId);
        Assert.Equal("done", messages[3].Content);
    }

    /// <summary>A top-level <c>system</c> block array is joined the same way (<c>'\n'</c>).</summary>
    [Fact]
    public void Anthropic_SystemBlocks_AreJoinedWithOneNewline()
    {
        var request = ParseAnthropic("""
            {"max_tokens":8,"system":[{"type":"text","text":"S1"},{"type":"text","text":"S2"}],
             "messages":[{"role":"user","content":"u"}]}
            """);
        var messages = AnthropicConverter.ToMessages(request);
        Assert.Equal("system", messages[0].Role);
        Assert.Equal("S1\nS2", messages[0].Content);
    }

    /// <summary>
    /// OpenAI <c>messages[i].content</c> is a <c>string?</c> on the DTO: an array of content parts does not deserialize.
    /// Over HTTP this is a 400 from model binding (see <see cref="ServerPinHttpTests"/>). Spec assumption was "accepted,
    /// parts joined with no separator".
    /// </summary>
    [Fact]
    public void OpenAi_ArrayContent_DoesNotDeserialize()
    {
        const string body = """{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}""";
        Assert.Throws<JsonException>(() => ParseOpenAi(body));
    }

    /// <summary>
    /// Qwen3-4B-Instruct-2507: user and tool content are emitted verbatim between fixed literals — no trim, no
    /// escaping. Leading/trailing whitespace and newlines of the content survive.
    /// </summary>
    [Fact]
    public void Qwen3_UserAndToolContent_AreRenderedVerbatim()
    {
        string prompt = Render(ServerTemplate(Qwen3Fixture),
        [
            Msg("user", "  \nuser text\n  "),
            new ChatMessage { Role = "tool", ToolCallId = "t1", Content = "\n tool body \"q\" \n" },
        ]);

        Assert.Equal(
            "<|im_start|>user\n  \nuser text\n  <|im_end|>\n" +
            "<|im_start|>user\n<tool_response>\n\n tool body \"q\" \n\n</tool_response><|im_end|>\n" +
            "<|im_start|>assistant\n",
            prompt);
    }

    /// <summary>
    /// Llama-3.1: user (and assistant) content is <c>| trim</c>med — .NET <c>string.Trim()</c>, every leading and
    /// trailing whitespace character — while interior whitespace is kept. A span touching either edge of a message is
    /// altered by the template.
    /// </summary>
    [Fact]
    public void Llama31_UserContent_IsTrimmed_InteriorKept()
    {
        string prompt = Render(ServerTemplate(Llama31Fixture), [Msg("user", "\n  a\n  b  \n")]);
        Assert.EndsWith(
            "<|start_header_id|>user<|end_header_id|>\n\n" + "a\n  b" + "<|eot_id|>" +
            "<|start_header_id|>assistant<|end_header_id|>\n\n",
            prompt, StringComparison.Ordinal);
    }

    /// <summary>Llama-3.1: the system message is trimmed too (<c>messages[0]['content']|trim</c>).</summary>
    [Fact]
    public void Llama31_SystemContent_IsTrimmed()
    {
        string prompt = Render(ServerTemplate(Llama31Fixture), [Msg("system", "\n sys \n"), Msg("user", "u")]);
        Assert.Contains("Today Date: 26 Jul 2024\n\n" + "sys" + "<|eot_id|>", prompt, StringComparison.Ordinal);
    }

    /// <summary>
    /// Llama-3.1: a <c>tool</c> message's content goes through <c>| tojson</c> (in Jinja2 a string is iterable), so the
    /// rendered prompt holds a quoted, JSON-escaped string: <c>\n</c> becomes the two characters <c>\\n</c>, quotes are
    /// escaped. Tool results — where an agent's file reads land — are therefore NEVER byte-identical in the render.
    /// </summary>
    [Fact]
    public void Llama31_ToolContent_IsJsonStringEncoded()
    {
        string prompt = Render(ServerTemplate(Llama31Fixture),
        [
            Msg("user", "q"),
            new ChatMessage { Role = "tool", ToolCallId = "t1", Content = "line1\n\"q\" <x>" },
        ]);
        Assert.Contains(
            "<|start_header_id|>ipython<|end_header_id|>\n\n" + "\"line1\\n\\\"q\\\" <x>\"" + "<|eot_id|>",
            prompt, StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ (b) special tokens

    /// <summary>
    /// #1339 (ControlTokenGuard): a special-token spelling inside message content is tokenized as PLAIN TEXT on the chat
    /// path — the template marks it with U+FDD0 after its first character, encode strips the marker. Consequence pinned
    /// here: the rendered prompt string does NOT contain such content verbatim (it contains the marked copy), so a
    /// substring / sentinel search over the render must account for the markers.
    /// </summary>
    [Theory]
    [InlineData(Qwen3Fixture, "a<|im_start|>b<think>c</tool_response>d")]
    [InlineData(Llama31Fixture, "a<|eot_id|>b<|start_header_id|>c")]
    public void ContentPath_SpecialSpellings_ArePlainText_AndTheRenderCarriesMarkers(string fixture, string content)
    {
        var template = ServerTemplate(fixture);
        string attacked = Render(template, [Msg("user", content)]);
        string benign = Render(template, [Msg("user", new string('x', content.Length))]);
        int[] attackedIds = Tok.Encode(attacked);
        int[] benignIds = Tok.Encode(benign);

        foreach (string s in Specials)
            Assert.True(CountOf(attackedIds, Id(s)) == CountOf(benignIds, Id(s)), $"{fixture}: content smuggled a {s}");
        Assert.Equal(benignIds.Length, attackedIds.Length);   // one token per character, exactly like the filler

        Assert.DoesNotContain(content, attacked, StringComparison.Ordinal);
        Assert.Contains(Tok.MarkUntrustedSpecials(content), attacked, StringComparison.Ordinal);
        Assert.Contains(content, ControlTokenGuard.Strip(attacked), StringComparison.Ordinal);
    }

    /// <summary>
    /// The chunkers' encode (<see cref="ITokenizer.EncodeRaw"/>, used by SymbolChunker / FixedSizeChunker) keeps
    /// scan-everything semantics: a special spelling in chunk text IS a control token there. So the chunk store's ids
    /// and the chat path's ids differ for any chunk containing a special spelling, unless the chunk text is marked first
    /// (<c>EncodeRaw(MarkUntrustedSpecials(text))</c> equals the plain-text encoding). Spec assumption was that EncodeRaw
    /// does not parse specials.
    /// </summary>
    [Fact]
    public void ChunkerEncodeRaw_ParsesSpecialSpellings_GuardedEncodeIsPlain()
    {
        const string text = "x<think>y<|im_end|>z";
        int[] raw = Tok.EncodeRaw(text);
        Assert.Equal(1, CountOf(raw, Id("<think>")));
        Assert.Equal(1, CountOf(raw, Id("<|im_end|>")));

        Assert.Equal(PlainTok.EncodeRaw(text), Tok.EncodeRaw(Tok.MarkUntrustedSpecials(text)));
    }

    /// <summary>
    /// U+FDD0 (the guard's marker) supplied by the caller is stripped on encode: content "a﷐b" tokenizes exactly
    /// like "ab". Chunk text carrying U+FDD0 therefore has different bytes but identical tokens.
    /// </summary>
    [Fact]
    public void CallerSuppliedMarker_IsDroppedAtEncode()
    {
        var template = ServerTemplate(Qwen3Fixture);
        Assert.Equal(
            Tok.Encode(Render(template, [Msg("user", "ab")])),
            Tok.Encode(Render(template, [Msg("user", "a" + ControlTokenGuard.Marker + "b")])));
    }

    // ------------------------------------------------------------------------------------------------ (c) unknown fields

    /// <summary>
    /// OpenAI chat: an unknown top-level <c>chunk_cache</c> object (and an unknown field on a message object) is skipped
    /// by source-generated deserialization; the rendered prompt is byte-identical to the unannotated request.
    /// </summary>
    [Fact]
    public void OpenAi_UnknownTopLevelAndMessageFields_AreIgnored_PromptUnchanged()
    {
        var plain = ParseOpenAi("""{"messages":[{"role":"user","content":"hello"}]}""");
        var annotated = ParseOpenAi("""
            {"messages":[{"role":"user","content":"hello","chunk":{"v":1,"id":"x","hash":"sha256:00"}}],
             "chunk_cache":{"v":1,"policy":"shadow","chunks":[{"id":"x","hash":"sha256:00","at":{"message":0,"part":0}}]}}
            """);

        var template = ServerTemplate(Qwen3Fixture);
        Assert.Equal(
            Render(template, RequestConverter.ToMessages(plain.Messages)),
            Render(template, RequestConverter.ToMessages(annotated.Messages)));
        Assert.Null(RequestValidator.ValidateChatRequest(annotated));
    }

    /// <summary>
    /// Anthropic: a top-level <c>chunk_cache</c> and a <c>chunk</c> field on a <c>text</c> block are both ignored —
    /// validation passes (it inspects only <c>type</c>), the flattened messages and the render are unchanged.
    /// </summary>
    [Fact]
    public void Anthropic_ChunkCacheAndChunkOnTextBlock_AreIgnored()
    {
        var plain = ParseAnthropic("""
            {"max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"hdr\n"},{"type":"text","text":"body\n"}]}]}
            """);
        var annotated = ParseAnthropic("""
            {"max_tokens":8,"chunk_cache":{"v":1,"policy":"exact"},
             "messages":[{"role":"user","content":[{"type":"text","text":"hdr\n"},
               {"type":"text","text":"body\n","chunk":{"v":1,"id":"b","hash":"sha256:00"},"cache_control":{"type":"ephemeral"}}]}]}
            """);

        Assert.Null(MessagesEndpoint.ValidateRequest(annotated));
        var template = ServerTemplate(Qwen3Fixture);
        Assert.Equal(Render(template, AnthropicConverter.ToMessages(plain)), Render(template, AnthropicConverter.ToMessages(annotated)));
    }

    /// <summary>Anthropic: an unknown content block TYPE is a 400 (<c>unsupported content block type</c>), #449.</summary>
    [Fact]
    public void Anthropic_UnknownBlockType_IsRejected()
    {
        var request = ParseAnthropic("""
            {"max_tokens":8,"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"chunk_ref","id":"x"}]}]}
            """);
        Assert.Equal("messages[0].content[1].type: unsupported content block type 'chunk_ref'", MessagesEndpoint.ValidateRequest(request));
    }

    /// <summary>
    /// ...but the blocks INSIDE a <c>tool_result</c>'s content array are not type-checked: any block with a string
    /// <c>text</c> is joined in (whatever its type), any other block is silently dropped.
    /// </summary>
    [Fact]
    public void Anthropic_ToolResultInnerBlocks_AreNotTypeChecked()
    {
        var request = ParseAnthropic("""
            {"max_tokens":8,"messages":[
              {"role":"assistant","content":[{"type":"tool_use","id":"t1","name":"read","input":{}}]},
              {"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":""}},
                {"type":"bogus","text":"x"},
                {"type":"text","text":"y"}]}]}]}
            """);

        Assert.Null(MessagesEndpoint.ValidateRequest(request));
        Assert.Equal("x\ny", AnthropicConverter.ToMessages(request).Single(m => m.Role == "tool").Content);
    }

    // ------------------------------------------------------------------------------------------------ real GGUFs (CPU, tokenizer only)

    private static string? FindGguf(string org, string repo, params string[] names)
        => TestFixtureResolver.ResolveFile([], org, repo, names, null).Path;

    private static readonly string? QwenGguf = FindGguf("bartowski", "Qwen_Qwen3-4B-Instruct-2507-GGUF",
        "Qwen_Qwen3-4B-Instruct-2507-IQ4_XS.gguf", "Qwen_Qwen3-4B-Instruct-2507-*.gguf");
    private static readonly string? LlamaGguf = FindGguf("bartowski", "Meta-Llama-3.1-8B-Instruct-GGUF",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", "Meta-Llama-3.1-8B-Instruct-*.gguf");

    public static IEnumerable<object[]> RealModels() =>
    [
        ["qwen3", Qwen3Fixture, "<|im_start|>", "<think>"],
        ["llama31", Llama31Fixture, "<|eot_id|>", "<|start_header_id|>"],
    ];

    private static string? GgufFor(string which) => string.Equals(which, "qwen3", StringComparison.Ordinal) ? QwenGguf : LlamaGguf;

    /// <summary>The repo fixtures are the models' GGUF <c>tokenizer.chat_template</c>, byte for byte.</summary>
    [SkippableTheory]
    [MemberData(nameof(RealModels))]
    public void RealGguf_TemplateFixture_IsTheGgufMetadataTemplate(string which, string fixture, string s1, string s2)
    {
        string? path = GgufFor(which);
        Skip.If(path is null, $"{which} GGUF not in the local model stores / HF cache.");
        using var gguf = GgufFile.Open(path!);
        Assert.Equal(FixtureTemplate(fixture), gguf.Metadata.GetStringOrDefault("tokenizer.chat_template", null!));
        _ = (s1, s2);
    }

    /// <summary>
    /// Substitution on real data: the models' own GGUF tokenizer and template, exactly as ServerStartup builds them.
    /// Special spellings in user content add no control ids on the chat path, while EncodeRaw of the bare spelling is
    /// the single control id (the chunkers' path).
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(RealModels))]
    public void RealGguf_ContentPathIsPlain_EncodeRawParses(string which, string fixture, string s1, string s2)
    {
        string? path = GgufFor(which);
        Skip.If(path is null, $"{which} GGUF not in the local model stores / HF cache.");
        using var gguf = GgufFile.Open(path!);
        var tok = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        var template = GgufChatTemplateFactory.TryCreate(gguf.Metadata, tok, Architecture.Llama)!;
        _ = fixture;

        int id1 = Assert.Single(tok.EncodeRaw(s1));
        int id2 = Assert.Single(tok.EncodeRaw(s2));

        int[] attacked = tok.Encode(Render(template, [Msg("user", $"a {s1} b {s2} c")]));
        int[] benign = tok.Encode(Render(template, [Msg("user", "a b c")]));
        Assert.Equal(CountOf(benign, id1), CountOf(attacked, id1));
        Assert.Equal(CountOf(benign, id2), CountOf(attacked, id2));
    }
}

/// <summary>
/// The same pins over real HTTP (the production JSON binding via <see cref="ServiceCollectionExtensions.AddDotLLM"/>):
/// what reaches the handler and what is refused before it. A non-computing model with a 1-token context makes every
/// chat request that reaches the handler stop at the prompt-length check, whose 400 message carries the prompt's token
/// count — so "reached the handler" and "rendered the same prompt" are both observable without generating.
/// </summary>
public sealed class ServerPinHttpTests : IAsyncLifetime
{
    private WebApplication? _app;
    private ServerState? _state;
    private HttpClient _http = new();

    public async Task InitializeAsync()
    {
        var model = new NonComputingModel(maxSeq: 1);
        _state = new ServerState
        {
            Options = new ServerOptions { Model = "test-model", ModelId = "test-model" },
            Model = model,
            Config = model.Config,
            Tokenizer = ServerPinTests.Tok,
            ChatTemplate = ServerPinTests.ServerTemplate(ServerPinTests.Qwen3Fixture),
            Generator = new TextGenerator(model, ServerPinTests.Tok),
            IsReady = true,
        };
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddDotLLM(_state);
        _app = builder.Build();
        _app.MapDotLLMEndpoints(serveUi: false);
        await _app.StartAsync();
        _http = new HttpClient { BaseAddress = new Uri(_app.Urls.First()) };
    }

    public async Task DisposeAsync()
    {
        _http.Dispose();
        if (_app is not null) await _app.DisposeAsync();
        _state?.Dispose();
    }

    private async Task<(HttpStatusCode Status, string Body)> PostAsync(string path, string body)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        request.Headers.Add("anthropic-version", "2023-06-01");
        using var resp = await _http.SendAsync(request);
        return (resp.StatusCode, await resp.Content.ReadAsStringAsync());
    }

    private static readonly Regex PromptTokens = new(@"prompt \((?<n>\d+) tokens\)", RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(1));

    /// <summary>Token count of the rendered prompt, read from the handler's context-length 400.</summary>
    private async Task<int> ChatPromptTokensAsync(string body)
    {
        var (status, text) = await PostAsync("/v1/chat/completions", body);
        Assert.True(status == HttpStatusCode.BadRequest, $"{(int)status} {text}");
        Assert.Contains("context_length_exceeded", text, StringComparison.Ordinal);
        return int.Parse(PromptTokens.Match(text).Groups["n"].Value, System.Globalization.CultureInfo.InvariantCulture);
    }

    private async Task<int> CountTokensAsync(string body)
    {
        var (status, text) = await PostAsync("/v1/messages/count_tokens", body);
        Assert.True(status == HttpStatusCode.OK, $"{(int)status} {text}");
        using var doc = JsonDocument.Parse(text);
        return doc.RootElement.GetProperty("input_tokens").GetInt32();
    }

    /// <summary>
    /// OpenAI chat with array <c>content</c>: refused by model binding with a bare 400 and an EMPTY body (no OpenAI error
    /// envelope) — the handler never runs. Inline <c>chunk</c> on an OpenAI text part is therefore unreachable today.
    /// </summary>
    [Fact]
    public async Task OpenAi_ArrayContent_Is400FromBinding_EmptyBody()
    {
        var (status, text) = await PostAsync("/v1/chat/completions",
            """{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b","chunk":{"v":1}}]}]}""");
        Assert.Equal(HttpStatusCode.BadRequest, status);
        Assert.Equal("", text);
    }

    /// <summary>Control for the test above: string content with unknown fields reaches the handler and renders the same prompt.</summary>
    [Fact]
    public async Task OpenAi_UnknownFields_ReachTheHandler_SamePromptTokens()
    {
        int plain = await ChatPromptTokensAsync("""{"messages":[{"role":"user","content":"hello"}]}""");
        int annotated = await ChatPromptTokensAsync("""
            {"messages":[{"role":"user","content":"hello","chunk":{"v":1}}],
             "chunk_cache":{"v":1,"policy":"shadow","chunks":[{"id":"x","hash":"sha256:00","at":{"message":0,"part":0}}]}}
            """);
        Assert.Equal(plain, annotated);
    }

    /// <summary>
    /// Anthropic count_tokens (shares BuildPrompt with /v1/messages): chunk fields are ignored and two text blocks cost
    /// exactly what the single string "a\nb" costs — one character-token more than "ab".
    /// </summary>
    [Fact]
    public async Task Anthropic_CountTokens_ChunkFieldsIgnored_BlocksJoinedWithNewline()
    {
        int joined = await CountTokensAsync("""{"messages":[{"role":"user","content":"a\nb"}]}""");
        int concatenated = await CountTokensAsync("""{"messages":[{"role":"user","content":"ab"}]}""");
        int blocks = await CountTokensAsync("""
            {"chunk_cache":{"v":1},"messages":[{"role":"user","content":[
              {"type":"text","text":"a"},{"type":"text","text":"b","chunk":{"v":1,"id":"b","hash":"sha256:00"}}]}]}
            """);
        Assert.Equal(joined, blocks);
        Assert.Equal(concatenated + 1, blocks);
    }

    /// <summary>Anthropic: an unknown block type is a 400 <c>invalid_request_error</c> in the Anthropic envelope.</summary>
    [Fact]
    public async Task Anthropic_UnknownBlockType_Is400InvalidRequest()
    {
        var (status, text) = await PostAsync("/v1/messages/count_tokens",
            """{"messages":[{"role":"user","content":[{"type":"chunk_ref","id":"x"}]}]}""");
        Assert.Equal(HttpStatusCode.BadRequest, status);
        using var doc = JsonDocument.Parse(text);
        Assert.Equal("error", doc.RootElement.GetProperty("type").GetString());
        Assert.Equal("invalid_request_error", doc.RootElement.GetProperty("error").GetProperty("type").GetString());
        Assert.Contains("unsupported content block type 'chunk_ref'", doc.RootElement.GetProperty("error").GetProperty("message").GetString(), StringComparison.Ordinal);
    }

    private sealed class NonComputingModel(int maxSeq) : IModel
    {
        public ModelConfig Config { get; } = new()
        {
            Architecture = Architecture.Llama,
            VocabSize = 512,
            HiddenSize = 8,
            IntermediateSize = 16,
            NumLayers = 1,
            NumAttentionHeads = 2,
            NumKvHeads = 1,
            HeadDim = 4,
            MaxSequenceLength = maxSeq,
        };

        public long ComputeMemoryBytes => 0;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId) =>
            throw new InvalidOperationException("ServerPinHttpTests never generates.");

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache) =>
            throw new InvalidOperationException("ServerPinHttpTests never generates.");

        public void Dispose() { }
    }
}
