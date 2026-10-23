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

/// <summary>
/// #798: Harmony (gpt-oss) and Gemma-4 channel markup through the server surfaces: reasoning goes to
/// <c>reasoning_content</c>, the answer to <c>content</c>, recipient messages to <c>tool_calls</c>, markup
/// never leaks, streamed == non-streamed, and Harmony's <c>&lt;|end|&gt;</c> is not a stop string.
/// </summary>
public sealed class ChannelMarkupEndpointTests
{
    // Minimal templates that carry each family's marker tokens (all the markup detector reads).
    private static readonly JinjaChatTemplate HarmonyTemplate =
        new("{{ '<|start|>assistant<|channel|>final<|message|>' }}", "", "");
    private static readonly JinjaChatTemplate GemmaTemplate =
        new("{{ '<|channel>thought\\n' }}", "", "");
    private static readonly JinjaChatTemplate ThinkTemplate = new("{{ '<think>' }}", "", "");

    private const string HarmonyPrompt = "<|start|>user<|message|>hi<|end|><|start|>assistant";
    private const string GemmaPrompt = "<|turn>user\nhi<turn|>\n<|turn>model\n";

    private static ReasoningPlan HarmonyPlan(bool constrained = false, ReasoningFormat f = ReasoningFormat.Auto)
        => ReasoningSupport.Plan(f, null, constrained, HarmonyPrompt, out _, HarmonyTemplate);

    private static ReasoningPlan GemmaPlan()
        => ReasoningSupport.Plan(ReasoningFormat.Auto, null, false, GemmaPrompt, out _, GemmaTemplate);

    private static readonly ToolDefinition[] Tools =
        [new("get_weather", "w", """{"type":"object","properties":{"city":{"type":"string"}}}""")];

    private const string HarmonyAnalysisFinal =
        "<|channel|>analysis<|message|>The user greets me. Reply briefly.<|end|>"
        + "<|start|>assistant<|channel|>final<|message|>Hello! How can I help?";

    private const string HarmonyToolTurn =
        "<|channel|>analysis<|message|>Need the weather for Paris.<|end|>"
        + "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Paris\"}";

    // ---------------------------------------------------------------- plan / stops

    [Fact]
    public void Plan_DetectsMarkupFromTheTemplate()
    {
        Assert.Equal(ReasoningMarkup.Harmony, HarmonyPlan().Markup);
        Assert.Equal(ReasoningMarkup.Gemma4Channel, GemmaPlan().Markup);
        Assert.Equal(ReasoningMarkup.Think,
            ReasoningSupport.Plan(ReasoningFormat.Auto, null, false, "<think>\n", out _, ThinkTemplate).Markup);
        Assert.Equal(ReasoningMarkup.Think,
            ReasoningSupport.Plan(ReasoningFormat.Auto, null, false, "<think>\n", out _).Markup);
    }

    [Fact]
    public void Harmony_EndIsNotAStopString_AnywhereItWouldBeApplied()
    {
        // "<|end|>" closes the ANALYSIS message; as a stop string it ended the turn before any answer (#798).
        string[] common = ["<|im_end|>", "<|eot_id|>", "<|eom_id|>", "<|end|>", "</s>", "</tool_call>"];
        Assert.DoesNotContain("<|end|>", HarmonyPlan().FilterStops(common));
        Assert.Contains("<|end|>", ReasoningSupport.Plan(ReasoningFormat.Auto, null, false, "<think>\n", out _, ThinkTemplate).FilterStops(common));

        var gated = HarmonyPlan().Gate(new InferenceOptions(), ReasoningSupport.UngatedStops);
        Assert.DoesNotContain("<|end|>", gated.StopSequencesUngated!);
        Assert.Contains("<|im_end|>", gated.StopSequencesUngated!);
    }

    [Fact]
    public void Harmony_ConstrainedRequest_DisablesSplitting_ButStillFiltersStops()
    {
        var plan = HarmonyPlan(constrained: true);
        Assert.False(plan.Enabled);
        Assert.DoesNotContain("<|end|>", plan.FilterStops(["<|end|>", "</s>"]));
    }

    [Fact]
    public void Harmony_StopGate_SuspendsStopStringsDuringAnalysis_ThenResumes()
    {
        var gate = HarmonyPlan().Gate(new InferenceOptions(), ReasoningSupport.UngatedStops).ReasoningStopGate!;
        Assert.Equal("<|channel|>analysis<|message|>", gate.OpenTag);
        Assert.Equal("<|end|>", gate.CloseTag);
        var stop = new StopStringCondition("STOP", gate);
        Assert.Equal(StopResult.Continue, stop.ShouldStop(0, [], "<|channel|>analysis<|message|>think STOP".AsSpan()));
        Assert.Equal(StopResult.Stop, stop.ShouldStop(0, [], "<|channel|>analysis<|message|>think<|end|><|start|>assistant<|channel|>final<|message|>ok STOP".AsSpan()));
    }

    [Fact]
    public void Gemma_StopGate_UsesTheChannelTags()
    {
        var gate = GemmaPlan().Gate(new InferenceOptions(), ReasoningSupport.UngatedStops).ReasoningStopGate!;
        Assert.Equal("<|channel>thought", gate.OpenTag);
        Assert.Equal("<channel|>", gate.CloseTag);
        Assert.False(gate.StartsInside);
        // Template ending in an open channel (after a tool response).
        var open = ReasoningSupport.Plan(ReasoningFormat.Auto, null, false, "<|turn>model\n<|channel>thought\n", out _, GemmaTemplate);
        Assert.True(open.PromptOpened);
    }

    [Fact]
    public void EndOfGeneration_IncludesCallAndReturn()
    {
        var tok = new VocabTokenizer("<|return|>", "<|call|>", "<|end|>", "<|channel|>");
        var ids = EndOfGenerationTokens.Resolve(tok);
        Assert.Contains(tok.IdOf("<|call|>"), ids);
        Assert.Contains(tok.IdOf("<|return|>"), ids);
        Assert.DoesNotContain(tok.IdOf("<|end|>"), ids);       // delimits the analysis message: never an EOG
    }

    // ---------------------------------------------------------------- non-streaming

    private static InferenceResponse Response(string text, FinishReason finish = FinishReason.Stop) => new()
    {
        GeneratedTokenIds = [], Text = text, FinishReason = finish, PromptTokenCount = 5, GeneratedTokenCount = 9,
    };

    private static ChatChoiceDto Build(string text, ReasoningPlan plan, IToolCallParser? parser = null)
        => ChatCompletionEndpoint.BuildChoice(
            JsonSerializer.Deserialize("""{"messages":[{"role":"user","content":"hi"}]}""", ServerJsonContext.Default.ChatCompletionRequest)!,
            new InferenceOptions(), parser is null ? null : Tools, parser, plan, null, Response(text), 0, out _);

    [Fact]
    public void Harmony_NonStreaming_AnalysisToReasoning_FinalToContent()
    {
        var c = Build(HarmonyAnalysisFinal, HarmonyPlan());
        Assert.Equal("The user greets me. Reply briefly.", c.Message.ReasoningContent);
        Assert.Equal("Hello! How can I help?", c.Message.Content);
        Assert.Equal("stop", c.FinishReason);
    }

    [Fact]
    public void Harmony_NonStreaming_CommentaryRecipient_BecomesAToolCall()
    {
        var c = Build(HarmonyToolTurn, HarmonyPlan(), new HarmonyToolCallParser());
        Assert.Equal("Need the weather for Paris.", c.Message.ReasoningContent);
        Assert.Null(c.Message.Content);
        var call = Assert.Single(c.Message.ToolCalls!);
        Assert.Equal("get_weather", call.Function.Name);
        Assert.Equal("{\"city\":\"Paris\"}", call.Function.Arguments);
        Assert.Equal("tool_calls", c.FinishReason);
    }

    [Fact]
    public void Harmony_NonStreaming_FormatNone_KeepsRawOutput()
    {
        var plan = ReasoningSupport.Plan(ReasoningFormat.None, null, false, HarmonyPrompt, out _, HarmonyTemplate);
        var c = Build(HarmonyAnalysisFinal, plan);
        Assert.Equal(HarmonyAnalysisFinal, c.Message.Content);
        Assert.Null(c.Message.ReasoningContent);
    }

    [Fact]
    public void Harmony_NonStreaming_CutOffInAnalysis_AllReasoning_LengthFinish()
    {
        var c = ChatCompletionEndpoint.BuildChoice(
            JsonSerializer.Deserialize("""{"messages":[{"role":"user","content":"hi"}]}""", ServerJsonContext.Default.ChatCompletionRequest)!,
            new InferenceOptions(), null, null, HarmonyPlan(), null,
            Response("<|channel|>analysis<|message|>thinking and thinking", FinishReason.Length), 0, out _);
        Assert.Equal("thinking and thinking", c.Message.ReasoningContent);
        Assert.Equal("", c.Message.Content);
        Assert.Equal("length", c.FinishReason);
    }

    [Fact]
    public void Gemma_NonStreaming_ThoughtToReasoning_NoMarkupInContent()
    {
        var c = Build("<|channel>thought\nThe user greets me.\n<channel|>Hello!", GemmaPlan());
        Assert.Equal("The user greets me.", c.Message.ReasoningContent);
        Assert.Equal("Hello!", c.Message.Content);
    }

    [Fact]
    public void Gemma_NonStreaming_ToolCallAfterThought()
    {
        var c = Build("<|channel>thought\nneed weather<channel|><|tool_call>call:get_weather{city:<|\"|>Paris<|\"|>}<tool_call|>",
            GemmaPlan(), new Gemma4ToolCallParser());
        Assert.Equal("need weather", c.Message.ReasoningContent);
        var call = Assert.Single(c.Message.ToolCalls!);
        Assert.Equal("get_weather", call.Function.Name);
    }

    // ---------------------------------------------------------------- streaming (OpenAI)

    private static async IAsyncEnumerable<GenerationToken> Tokens(string[] pieces)
    {
        for (int i = 0; i < pieces.Length; i++)
        {
            await Task.Yield();
            yield return new GenerationToken(i, pieces[i], i == pieces.Length - 1 ? FinishReason.Stop : null);
        }
    }

    private sealed record Streamed(string Reasoning, string Content, JsonElement? ToolCalls, string? Finish, List<JsonElement> Frames);

    private static async Task<Streamed> StreamAsync(string[] pieces, ReasoningPlan plan, IToolCallParser? parser)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            JsonSerializer.Deserialize("""{"messages":[{"role":"user","content":"hi"}],"stream":true}""", ServerJsonContext.Default.ChatCompletionRequest)!,
            ctx, _ => Tokens(pieces), (work, _) => work(), "chatcmpl-test", "m",
            parser is null ? null : Tools, parser, plan, CancellationToken.None);

        var frames = new List<JsonElement>();
        var r = new StringBuilder();
        var c = new StringBuilder();
        JsonElement? calls = null;
        string? finish = null;
        foreach (string block in Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            string data = block["data: ".Length..];
            if (data == "[DONE]") continue;
            var f = JsonDocument.Parse(data).RootElement.Clone();
            frames.Add(f);
            foreach (var choice in f.GetProperty("choices").EnumerateArray())
            {
                var d = choice.GetProperty("delta");
                if (d.TryGetProperty("reasoning_content", out var rc)) r.Append(rc.GetString());
                if (d.TryGetProperty("content", out var cc)) c.Append(cc.GetString());
                if (d.TryGetProperty("tool_calls", out var tc)) calls = tc.Clone();
                if (choice.TryGetProperty("finish_reason", out var fr) && fr.ValueKind == JsonValueKind.String) finish = fr.GetString();
            }
        }
        return new Streamed(r.ToString(), c.ToString(), calls, finish, frames);
    }

    /// <summary>One piece per character: the hardest tokenisation for a markup state machine.</summary>
    private static string[] Chars(string s) => s.Select(ch => ch.ToString()).ToArray();

    /// <summary>Realistic tokenisation: each special marker is one token, words are small pieces.</summary>
    private static string[] HarmonyTokens(string s)
    {
        var parts = new List<string>();
        int i = 0;
        while (i < s.Length)
        {
            if (s[i] == '<' && s.IndexOf("|>", i, StringComparison.Ordinal) is var e and > 0 && s[i + 1] == '|')
            {
                parts.Add(s.Substring(i, e + 2 - i));
                i = e + 2;
                continue;
            }
            int j = i + 1;
            while (j < s.Length && s[j] != '<' && j - i < 4)
                j++;
            parts.Add(s.Substring(i, j - i));
            i = j;
        }
        return [.. parts];
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Harmony_Streaming_SplitsChannels_NoMarkupLeaks(bool charPerToken)
    {
        var pieces = charPerToken ? Chars(HarmonyAnalysisFinal) : HarmonyTokens(HarmonyAnalysisFinal);
        var s = await StreamAsync(pieces, HarmonyPlan(), null);
        Assert.Equal("The user greets me. Reply briefly.", s.Reasoning);
        Assert.Equal("Hello! How can I help?", s.Content);
        Assert.DoesNotContain("<|", s.Reasoning + s.Content, StringComparison.Ordinal);
        Assert.Equal("stop", s.Finish);

        // streamed == non-streamed
        var c = Build(HarmonyAnalysisFinal, HarmonyPlan());
        Assert.Equal(c.Message.ReasoningContent, s.Reasoning);
        Assert.Equal(c.Message.Content, s.Content);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Harmony_Streaming_ToolCall_ArrivesInFinalChunk_NotInContent(bool charPerToken)
    {
        var pieces = charPerToken ? Chars(HarmonyToolTurn) : HarmonyTokens(HarmonyToolTurn);
        var s = await StreamAsync(pieces, HarmonyPlan(), new HarmonyToolCallParser());
        Assert.Equal("Need the weather for Paris.", s.Reasoning);
        Assert.Equal("", s.Content);                                  // no "<|channel|>commentary to=..." leak
        Assert.Equal("tool_calls", s.Finish);
        var call = s.ToolCalls!.Value[0];
        Assert.Equal("get_weather", call.GetProperty("function").GetProperty("name").GetString());
        Assert.Equal("{\"city\":\"Paris\"}", call.GetProperty("function").GetProperty("arguments").GetString());
    }

    [Fact]
    public async Task Harmony_Streaming_PreambleStreams_ThenToolCall()
    {
        const string turn =
            "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>commentary<|message|>Checking now.<|end|>"
            + "<|start|>assistant<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{\"city\":\"Rome\"}";
        var s = await StreamAsync(HarmonyTokens(turn), HarmonyPlan(), new HarmonyToolCallParser());
        Assert.Equal("Checking now.", s.Content);
        Assert.Equal("tool_calls", s.Finish);
        Assert.Single(s.ToolCalls!.Value.EnumerateArray());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Gemma_Streaming_ThoughtSplit_NoMarkupLeaks(bool charPerToken)
    {
        const string text = "<|channel>thought\nThe user greets me.\n<channel|>Hello! How can I help?";
        string[] pieces = charPerToken ? Chars(text) : ["<|channel>", "thought", "\n", "The user", " greets me.", "\n", "<channel|>", "Hello!", " How can", " I help?"];
        var s = await StreamAsync(pieces, GemmaPlan(), null);
        Assert.Equal("The user greets me.", s.Reasoning);
        Assert.Equal("Hello! How can I help?", s.Content);
        Assert.DoesNotContain("channel", s.Reasoning + s.Content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task Gemma_Streaming_ToolCallAfterThought_IsSuppressedFromContent()
    {
        string[] pieces = ["<|channel>", "thought", "\nneed weather\n", "<channel|>", "<|tool_call>", "call:get_weather{city:", "<|\"|>", "Paris", "<|\"|>", "}", "<tool_call|>"];
        var s = await StreamAsync(pieces, GemmaPlan(), new Gemma4ToolCallParser());
        Assert.Equal("need weather", s.Reasoning);
        Assert.Equal("", s.Content);
        Assert.Equal("tool_calls", s.Finish);
    }

    // ---------------------------------------------------------------- streaming (Anthropic)

    private static async Task<List<(string Event, JsonElement Data)>> MessagesAsync(string[] pieces, ReasoningPlan plan, IToolCallParser? parser)
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Response.Body = body;
        await MessagesEndpoint.WriteMessageStreamAsync(
            ctx, _ => Tokens(pieces), (work, _) => work(), parser, null,
            "msg_t", "m", promptTokenCount: 3, CancellationToken.None, forcedToolCall: false, plan,
            tools: parser is null ? null : Tools);
        var frames = new List<(string, JsonElement)>();
        foreach (string block in Encoding.UTF8.GetString(body.ToArray()).Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
        {
            var lines = block.Split('\n', StringSplitOptions.RemoveEmptyEntries);
            frames.Add((lines[0]["event: ".Length..], JsonDocument.Parse(lines[1]["data: ".Length..]).RootElement.Clone()));
        }
        return frames;
    }

    [Fact]
    public async Task Harmony_Anthropic_ThinkingBlockThenText()
    {
        var f = await MessagesAsync(HarmonyTokens(HarmonyAnalysisFinal), HarmonyPlan(), null);
        string thinking = string.Concat(f.Where(x => x.Event == "content_block_delta" && x.Data.GetProperty("delta").GetProperty("type").GetString() == "thinking_delta")
            .Select(x => x.Data.GetProperty("delta").GetProperty("thinking").GetString()));
        string text = string.Concat(f.Where(x => x.Event == "content_block_delta" && x.Data.GetProperty("delta").GetProperty("type").GetString() == "text_delta")
            .Select(x => x.Data.GetProperty("delta").GetProperty("text").GetString()));
        Assert.Equal("The user greets me. Reply briefly.", thinking);
        Assert.Equal("Hello! How can I help?", text);
    }

    [Fact]
    public async Task Harmony_Anthropic_ToolUse()
    {
        var f = await MessagesAsync(HarmonyTokens(HarmonyToolTurn), HarmonyPlan(), new HarmonyToolCallParser());
        var tool = f.Single(x => x.Event == "content_block_start"
            && x.Data.GetProperty("content_block").GetProperty("type").GetString() == "tool_use");
        Assert.Equal("get_weather", tool.Data.GetProperty("content_block").GetProperty("name").GetString());
        Assert.Equal("tool_use", f.Last(x => x.Event == "message_delta").Data.GetProperty("delta").GetProperty("stop_reason").GetString());
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>Tokenizer stub whose vocabulary is exactly the given token texts (id = index).</summary>
    private sealed class VocabTokenizer(params string[] vocab) : ITokenizer
    {
        public int IdOf(string text) => Array.IndexOf(vocab, text);
        public int VocabSize => vocab.Length;
        public int BosTokenId => 0;
        public int EosTokenId => 0;
        public string DecodeToken(int tokenId) => vocab[tokenId];
        public int CountTokens(string text) => text.Length;
        public int[] Encode(string text) => throw new NotSupportedException();
        public string Decode(ReadOnlySpan<int> tokenIds) => throw new NotSupportedException();
    }
}
