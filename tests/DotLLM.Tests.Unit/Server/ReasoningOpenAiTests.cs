using System.Text;
using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using DotLLM.Engine.Samplers.StopConditions;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ChatTemplates;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Tokenizers.ToolCallParsers;
using Microsoft.AspNetCore.Http;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#767: reasoning / thinking handling on the OpenAI-compatible surface.</summary>
public sealed class ReasoningOpenAiTests
{
    private static ChatCompletionRequest ParseRequest(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.ChatCompletionRequest)!;

    private static ReasoningPlan Open(ReasoningFormat f = ReasoningFormat.Auto) =>
        ReasoningSupport.Plan(f, null, constrained: false, "<|im_start|>assistant\n<think>\n", out _);

    private static ReasoningPlan Closed() =>
        ReasoningSupport.Plan(ReasoningFormat.Auto, null, constrained: false, "<|im_start|>assistant\n<think>\n\n</think>\n\n", out _);

    // ---------------------------------------------------------------- request surface / STJ guard

    [Fact]
    public void NewRequestFields_AreNullWhenOmitted_NotDefaulted()
    {
        // STJ source-gen drops initializers on `init` properties (CLAUDE.md "JSON DTO Rules"): every new
        // field must be nullable, and omission must be distinguishable from `false`.
        var r = ParseRequest("""{"messages":[{"role":"user","content":"hi"}]}""");
        Assert.Null(r.EnableThinking);
        Assert.Null(r.ChatTemplateKwargs);
        Assert.Null(r.ReasoningEffort);
        Assert.Null(r.ReasoningFormat);
        Assert.Null(r.Messages[0].ReasoningContent);
        Assert.Null(r.Messages[0].Reasoning);
    }

    [Fact]
    public void NewRequestFields_AreBound()
    {
        var r = ParseRequest("""
            {"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"yo","reasoning_content":"hmm"}],
             "enable_thinking":false,"reasoning_effort":"low","reasoning_format":"deepseek",
             "chat_template_kwargs":{"preserve_thinking":true,"x":{"a":[1,2]}}}
            """);
        Assert.False(r.EnableThinking);
        Assert.Equal("low", r.ReasoningEffort);
        Assert.Equal("deepseek", r.ReasoningFormat);
        Assert.True(r.ChatTemplateKwargs!["preserve_thinking"].GetBoolean());
        Assert.Equal("hmm", r.Messages[1].ReasoningContent);

        var msgs = RequestConverter.ToMessages(r.Messages);
        Assert.Equal("hmm", msgs[1].ReasoningContent);
        Assert.Null(msgs[0].ReasoningContent);
    }

    [Fact]
    public void ReasoningAliasOnRequestMessage_MapsToReasoningContent()
    {
        var r = ParseRequest("""{"messages":[{"role":"assistant","content":"a","reasoning":"why"}]}""");
        Assert.Equal("why", RequestConverter.ToMessages(r.Messages)[0].ReasoningContent);
    }

    // ---------------------------------------------------------------- template options

    [Fact]
    public void TemplateOptions_ExplicitEnableThinkingWinsOverKwarg()
    {
        using var doc = JsonDocument.Parse("""{"enable_thinking": true}""");
        var kwargs = doc.RootElement.EnumerateObject().ToDictionary(p => p.Name, p => p.Value.Clone());
        Assert.False(ReasoningSupport.BuildTemplateOptions(null, false, null, kwargs, false).EnableThinking);
        Assert.True(ReasoningSupport.BuildTemplateOptions(null, null, null, kwargs, false).EnableThinking);
    }

    [Fact]
    public void TemplateOptions_ConstrainedDefaultsThinkingOff_ButExplicitOnIsKept()
    {
        Assert.False(ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: true).EnableThinking);
        Assert.True(ReasoningSupport.BuildTemplateOptions(null, true, null, null, constrained: true).EnableThinking);
        Assert.Null(ReasoningSupport.BuildTemplateOptions(null, null, null, null, constrained: false).EnableThinking);
    }

    [Fact]
    public void TemplateOptions_EffortNone_MeansThinkingOff_AndIsNotPassedToTheTemplate()
    {
        var o = ReasoningSupport.BuildTemplateOptions(null, null, "none", null, false);
        Assert.False(o.EnableThinking);
        Assert.Null(o.ReasoningEffort);
        Assert.Equal("low", ReasoningSupport.BuildTemplateOptions(null, null, "low", null, false).ReasoningEffort);
    }

    [Fact]
    public void TryApply_TemplateRejectingEffort_IsAClientErrorNamingTheParam()
    {
        string src = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Tokenizers", "ChatTemplates", "Fixtures", "qwen3.8-27b-chat-template.jinja"));
        var template = new JinjaChatTemplate(src, "<|endoftext|>", "<|im_end|>");
        var options = ReasoningSupport.BuildTemplateOptions(null, null, "high", null, false);
        bool ok = ReasoningSupport.TryApply(template, [new ChatMessage { Role = "user", Content = "q" }], options,
            out _, out string? error, out string? param);
        Assert.False(ok);
        Assert.Equal("reasoning_effort", param);
        Assert.Contains("Unexpected reasoning effort", error, StringComparison.Ordinal);
    }

    // ---------------------------------------------------------------- plan

    [Fact]
    public void Plan_ConstrainedOrNone_Disabled()
    {
        Assert.False(ReasoningSupport.Plan(ReasoningFormat.Auto, null, constrained: true, "x<think>\n", out _).Enabled);
        Assert.False(ReasoningSupport.Plan(ReasoningFormat.None, null, constrained: false, "x<think>\n", out _).Enabled);
        Assert.False(ReasoningSupport.Plan(ReasoningFormat.Auto, "none", constrained: false, "x<think>\n", out _).Enabled);
    }

    [Fact]
    public void Plan_RequestFormatOverridesServer_AndUnknownIsAnError()
    {
        Assert.True(ReasoningSupport.Plan(ReasoningFormat.None, "deepseek", false, "p", out _).Enabled);
        ReasoningSupport.Plan(ReasoningFormat.Auto, "bogus", false, "p", out string? err);
        Assert.NotNull(err);
    }

    [Fact]
    public void Plan_Gate_DescribesTheBlock_AndIsAbsentWhenSplittingIsOff()
    {
        var opts = new InferenceOptions { StopSequences = ["\n\n", "<|im_end|>"] };

        var opened = Open().Gate(opts, ReasoningSupport.UngatedStops).ReasoningStopGate!;
        Assert.True(opened.StartsInside);
        Assert.True(opened.OpenOnlyAtStart);
        Assert.Equal("</think>", opened.CloseTag);

        // Template did not open a block: the model may still open its own, so the gate is present but starts outside.
        Assert.False(Closed().Gate(opts, ReasoningSupport.UngatedStops).ReasoningStopGate!.StartsInside);

        Assert.False(Open(ReasoningFormat.Deepseek).Gate(opts, ReasoningSupport.UngatedStops).ReasoningStopGate!.OpenOnlyAtStart);
        Assert.Null(ReasoningPlan.Disabled.Gate(opts, ReasoningSupport.UngatedStops).ReasoningStopGate);
    }

    // ---------------------------------------------------------------- non-streaming

    private sealed class MarkerToolParser : IToolCallParser
    {
        public ToolCall[]? TryParse(string generatedText) =>
            generatedText.Contains("<tool_call>", StringComparison.Ordinal)
                ? [new ToolCall("call_1", "get_weather", "{}")]
                : null;
        public bool IsToolCallStart(string text) => text.Contains("<tool_call>", StringComparison.Ordinal);
    }

    private static InferenceResponse Response(string text, FinishReason finish = FinishReason.Stop) => new()
    {
        GeneratedTokenIds = [], Text = text, FinishReason = finish, PromptTokenCount = 5, GeneratedTokenCount = 9,
    };

    private static readonly ToolDefinition[] Tools = [new("get_weather", "w", "{}")];

    private static ChatChoiceDto Build(string text, ReasoningPlan plan, IToolCallParser? parser = null,
        FinishReason finish = FinishReason.Stop)
        => Build(text, plan, parser, finish, tok: null, out _);

    private static ChatChoiceDto Build(string text, ReasoningPlan plan, IToolCallParser? parser,
        FinishReason finish, ITokenizer? tok, out int reasoningTokens)
        => ChatCompletionEndpoint.BuildChoice(
            ParseRequest("""{"messages":[{"role":"user","content":"hi"}]}"""),
            new InferenceOptions(), parser is null ? null : Tools, parser, plan, tok,
            Response(text, finish), 0, out reasoningTokens);

    [Fact]
    public void NonStreaming_ReasoningTokens_CountTheRawPrefixThroughTheClosingTag()
    {
        // Char-per-token stub: "think</think>" is 13 raw chars up to and including the tag.
        Build("think</think>\n\nAnswer", Open(), null, FinishReason.Stop, new CharTokenizerStub(), out int tokens);
        Assert.Equal("think</think>".Length, tokens);
        Build("plain", Closed(), null, FinishReason.Stop, new CharTokenizerStub(), out int none);
        Assert.Equal(0, none);
    }

    [Fact]
    public void NonStreaming_SplitsReasoningFromContent()
    {
        var choice = Build("We need to answer.\n</think>\n\nThe capital of France is Paris.", Open());
        Assert.Equal("We need to answer.", choice.Message.ReasoningContent);
        Assert.Equal("The capital of France is Paris.", choice.Message.Content);
        Assert.Equal("stop", choice.FinishReason);
    }

    [Fact]
    public void NonStreaming_ClosedBlockInPrompt_NoReasoningField_ContentVerbatim()
    {
        var choice = Build("Paris.", Closed());
        Assert.Null(choice.Message.ReasoningContent);
        Assert.Equal("Paris.", choice.Message.Content);
        string json = JsonSerializer.Serialize(choice, ServerJsonContext.Default.ChatChoiceDto);
        Assert.DoesNotContain("reasoning_content", json, StringComparison.Ordinal);
    }

    [Fact]
    public void NonStreaming_FormatNone_KeepsTheRawOutput()
    {
        var plan = ReasoningSupport.Plan(ReasoningFormat.None, null, false, "x<think>\n", out _);
        var choice = Build("hmm</think>\n\nAnswer", plan);
        Assert.Equal("hmm</think>\n\nAnswer", choice.Message.Content);
        Assert.Null(choice.Message.ReasoningContent);
    }

    [Fact]
    public void NonStreaming_MaxTokensMidThought_AllReasoning_EmptyContent()
    {
        var choice = Build("still thinking...", Open(), finish: FinishReason.Length);
        Assert.Equal("still thinking...", choice.Message.ReasoningContent);
        Assert.Equal("", choice.Message.Content);
        Assert.Equal("length", choice.FinishReason);
    }

    [Fact]
    public void NonStreaming_ToolCallQuotedInReasoning_IsNotACall()
    {
        var choice = Build("I could emit <tool_call> but no.\n</think>\n\nIt is sunny.", Open(), new MarkerToolParser());
        Assert.Null(choice.Message.ToolCalls);
        Assert.Equal("It is sunny.", choice.Message.Content);
        Assert.Contains("<tool_call>", choice.Message.ReasoningContent, StringComparison.Ordinal);
        Assert.Equal("stop", choice.FinishReason);
    }

    [Fact]
    public void NonStreaming_ToolCallInAnswer_IsDetected_ReasoningKept()
    {
        var choice = Build("need weather\n</think>\n\n<tool_call>x</tool_call>", Open(), new MarkerToolParser());
        Assert.NotNull(choice.Message.ToolCalls);
        Assert.Null(choice.Message.Content);
        Assert.Equal("need weather", choice.Message.ReasoningContent);
        Assert.Equal("tool_calls", choice.FinishReason);
    }

    private sealed class CharTokenizerStub : ITokenizer
    {
        public int CountTokens(string text) => text.Length;
        public int VocabSize => 0;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public int[] Encode(string text) => throw new NotSupportedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotSupportedException();
        public string DecodeToken(int tokenId) => throw new NotSupportedException();
    }

    // ---------------------------------------------------------------- streaming

    private static async IAsyncEnumerable<GenerationToken> Tokens(params string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(i, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    private static async Task<List<JsonElement>> StreamAsync(
        string requestJson, ReasoningPlan plan, IToolCallParser? parser, params string[] pieces)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            ParseRequest(requestJson), ctx, _ => Tokens(pieces), (work, _) => work(),
            "chatcmpl-test", "m", parser is null ? null : Tools, parser, plan, CancellationToken.None);

        var frames = new List<JsonElement>();
        foreach (string block in Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string data = block["data: ".Length..];
            if (data == "[DONE]") continue;
            frames.Add(JsonDocument.Parse(data).RootElement.Clone());
        }
        return frames;
    }

    private static (string Reasoning, string Content) Collect(List<JsonElement> frames)
    {
        var r = new StringBuilder();
        var c = new StringBuilder();
        foreach (var f in frames)
        {
            foreach (var choice in f.GetProperty("choices").EnumerateArray())
            {
                var d = choice.GetProperty("delta");
                if (d.TryGetProperty("reasoning_content", out var rc)) r.Append(rc.GetString());
                if (d.TryGetProperty("content", out var cc)) c.Append(cc.GetString());
            }
        }
        return (r.ToString(), c.ToString());
    }

    private const string Req = """{"messages":[{"role":"user","content":"hi"}],"stream":true}""";

    [Fact]
    public async Task Streaming_TagSplitAcrossTokens_ReasoningThenContent_NoTagLeak()
    {
        var frames = await StreamAsync(Req, Open(), null,
            "We need", " to answer.", "\n</th", "ink>", "\n\n", "The capital", " is Paris.");

        var (reasoning, content) = Collect(frames);
        Assert.Equal("We need to answer.", reasoning);
        Assert.Equal("The capital is Paris.", content);
        Assert.DoesNotContain("think", reasoning + content, StringComparison.Ordinal);

        // Reasoning deltas precede content deltas, and never share a delta.
        int firstContent = frames.FindIndex(f => f.GetProperty("choices").GetArrayLength() > 0
            && f.GetProperty("choices")[0].GetProperty("delta").TryGetProperty("content", out _));
        int lastReasoning = frames.FindLastIndex(f => f.GetProperty("choices").GetArrayLength() > 0
            && f.GetProperty("choices")[0].GetProperty("delta").TryGetProperty("reasoning_content", out _));
        Assert.True(lastReasoning < firstContent);
        foreach (var f in frames.Where(f => f.GetProperty("choices").GetArrayLength() > 0))
        {
            var d = f.GetProperty("choices")[0].GetProperty("delta");
            Assert.False(d.TryGetProperty("reasoning_content", out _) && d.TryGetProperty("content", out _));
        }
    }

    [Fact]
    public async Task Streaming_StreamedEqualsNonStreamed()
    {
        string[] pieces = ["Hmm,", " so ", "</", "think", ">", "\n\n", "Final", " answer."];
        var (r, c) = Collect(await StreamAsync(Req, Open(), null, pieces));
        var choice = Build(string.Concat(pieces), Open());
        Assert.Equal(choice.Message.ReasoningContent, r);
        Assert.Equal(choice.Message.Content, c);
    }

    [Fact]
    public async Task Streaming_ModelOpensThinkItself_WhenTemplateDidNot()
    {
        var (r, c) = Collect(await StreamAsync(Req, Closed() /* not opened by prompt */, null,
            "<thi", "nk>", "plan", "</think>", "Done."));
        // Closed() has Auto + prompt not open: a leading <think> from the model is still honoured.
        Assert.Equal("plan", r);
        Assert.Equal("Done.", c);
    }

    [Fact]
    public async Task Streaming_PlainAnswerWithoutThink_PassesThroughUnchanged()
    {
        var (r, c) = Collect(await StreamAsync(Req, Closed(), null, "Hel", "lo ", "wor", "ld"));
        Assert.Equal("", r);
        Assert.Equal("Hello world", c);
    }

    [Fact]
    public async Task Streaming_PlanDisabled_EmitsRawTokensAsContent()
    {
        var frames = await StreamAsync(Req, ReasoningPlan.Disabled, null, "a</think>", "b");
        var (r, c) = Collect(frames);
        Assert.Equal("", r);
        Assert.Equal("a</think>b", c);
    }

    [Fact]
    public async Task Streaming_ToolCallInsideReasoning_IsNotDetected()
    {
        var frames = await StreamAsync(Req, Open(), new MarkerToolParser(),
            "maybe <tool_call>", " no", "</think>", "Sunny.");
        var last = frames.Last(f => f.GetProperty("choices").GetArrayLength() > 0);
        var choice = last.GetProperty("choices")[0];
        Assert.Equal("stop", choice.GetProperty("finish_reason").GetString());
        Assert.False(choice.GetProperty("delta").TryGetProperty("tool_calls", out _));
    }

    [Fact]
    public async Task Streaming_UsageChunk_CountsReasoningTokens()
    {
        var frames = await StreamAsync(
            """{"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}""",
            Open(), null, "think ", "more", "</think>", "\n\n", "Hi");
        var usage = frames.Last(f => f.GetProperty("choices").GetArrayLength() == 0).GetProperty("usage");
        Assert.Equal(5, usage.GetProperty("completion_tokens").GetInt32());
        // "think ", "more", "</think>" (the closing tag token counts as reasoning).
        Assert.Equal(3, usage.GetProperty("completion_tokens_details").GetProperty("reasoning_tokens").GetInt32());
    }

    [Fact]
    public async Task Streaming_NoReasoning_UsageHasNoDetails()
    {
        var frames = await StreamAsync(
            """{"messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}""",
            Closed(), null, "Hi", " there");
        var usage = frames.Last(f => f.GetProperty("choices").GetArrayLength() == 0).GetProperty("usage");
        Assert.False(usage.TryGetProperty("completion_tokens_details", out _));
    }

    // ---------------------------------------------------------------- engine stop gate

    private static StopResult Run(StopStringCondition c, string tail) => c.ShouldStop(0, [], tail.AsSpan());

    private static readonly StopGate Inside = new("<think>", "</think>", StartsInside: true, OpenOnlyAtStart: true);
    private static readonly StopGate Outside = new("<think>", "</think>", StartsInside: false, OpenOnlyAtStart: true);

    [Fact]
    public void StopGate_PromptOpenedBlock_StopInsideReasoningDoesNotFire_AfterCloseItDoes()
    {
        var gate = new StopStringCondition("\n\n", Inside);
        Assert.Equal(StopResult.Continue, Run(gate, "first paragraph\n\n"));          // inside the thinking
        Assert.Equal(StopResult.Continue, Run(gate, "more thinking</think>"));
        Assert.Equal(StopResult.Continue, Run(gate, "king</think>\n\n"));             // the template separator is not answer text
        Assert.Equal(StopResult.Continue, Run(gate, "</think>\n\nAnswer"));
        Assert.Equal(StopResult.Stop, Run(gate, "</think>\n\nAnswer\n\n"));          // a real stop in the answer
    }

    [Fact]
    public void StopGate_ModelOpensBlockItself_Suspends_ThenResumesAfterClose()
    {
        // Original Qwen3 template: the prompt does NOT open <think>, the model emits it first.
        var gate = new StopStringCondition("\n\n", Outside);
        Assert.Equal(StopResult.Continue, Run(gate, "<th"));                          // could still be the open tag
        Assert.Equal(StopResult.Continue, Run(gate, "<think>"));
        Assert.Equal(StopResult.Continue, Run(gate, "<think>\nOkay.\n\n"));          // stop string inside the thinking
        Assert.Equal(StopResult.Continue, Run(gate, "a long thought that scrolled the open tag away\n\n"));
        Assert.Equal(StopResult.Continue, Run(gate, "</think>"));
        Assert.Equal(StopResult.Stop, Run(gate, "</think>\n\nFive\n\n"));
    }

    [Fact]
    public void StopGate_NoThinkingAtAll_StopsBehaveAsBefore()
    {
        var gate = new StopStringCondition("END", Outside);
        Assert.Equal(StopResult.Continue, Run(gate, "Hello"));
        Assert.Equal(StopResult.Stop, Run(gate, "Hello END"));
    }

    [Fact]
    public void StopGate_AutoIgnoresALiteralThinkLaterInTheAnswer_DeepseekHonoursIt()
    {
        var auto = new StopStringCondition("END", Outside);
        Assert.Equal(StopResult.Continue, Run(auto, "Use "));
        Assert.Equal(StopResult.Stop, Run(auto, "Use <think> tags END"));   // not at the start: not a block

        var anywhere = new StopStringCondition("END", new StopGate("<think>", "</think>", false, OpenOnlyAtStart: false));
        Assert.Equal(StopResult.Continue, Run(anywhere, "Use "));
        Assert.Equal(StopResult.Continue, Run(anywhere, "Use <think> inner END"));
    }

    [Fact]
    public void StopGate_NoGate_BehavesExactlyLikeTheOriginal()
    {
        var plain = new StopStringCondition("END");
        Assert.Equal(StopResult.Stop, Run(plain, "abcEND"));
        Assert.Equal(StopResult.Continue, Run(plain, "abc"));
    }

    [Fact]
    public void StopGate_CreateAll_UsesFreshStatePerCall_AndExemptsControlTokens()
    {
        var options = new InferenceOptions
        {
            StopSequences = ["STOP", "<|im_end|>"],
            ReasoningStopGate = Inside,
            StopSequencesUngated = ["<|im_end|>"],
        };
        var a = StopStringCondition.CreateAll(options);
        var b = StopStringCondition.CreateAll(options);
        Assert.Equal(StopResult.Continue, Run(a[0], "thinking STOP"));           // gated
        Assert.Equal(StopResult.Stop, Run(a[1], "thinking <|im_end|>"));         // ungated control token
        Run(a[0], "x</think>");                                                   // leaves the block in a[0] only
        Assert.Equal(StopResult.Stop, Run(a[0], "ans STOP"));
        Assert.Equal(StopResult.Continue, Run(b[0], "ans STOP"));                 // sibling sequence unaffected
    }
}
