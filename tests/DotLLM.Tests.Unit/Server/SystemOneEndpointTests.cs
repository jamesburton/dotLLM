using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Core.Sampling;
using DotLLM.Engine;
using DotLLM.Engine.Decisions;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #708: the Jev-compatible <c>/v1/systemone</c> endpoint, driven through a scripted runner (no model).
/// </summary>
public sealed class SystemOneEndpointTests
{
    private sealed class CharTokenizer : ITokenizer
    {
        public int VocabSize => 256;
        public int BosTokenId => 1;
        public int EosTokenId => 2;
        public string DecodeToken(int tokenId) => ((char)tokenId).ToString();
        public int[] Encode(string text) => text.Select(c => (int)c).ToArray();
        public string Decode(ReadOnlySpan<int> tokenIds) => string.Concat(tokenIds.ToArray().Select(i => (char)i));
        public string Decode(ReadOnlySpan<int> tokenIds, bool stripBosSpace) => Decode(tokenIds);
        public int CountTokens(string text) => text.Length;
    }

    private sealed class ChatMlTemplate : IChatTemplate
    {
        public string Apply(IReadOnlyList<ChatMessage> messages, ChatTemplateOptions options) =>
            string.Concat(messages.Select(m => $"<|im_start|>{m.Role}\n{m.Content}<|im_end|>\n"))
            + (options.AddGenerationPrompt ? "<|im_start|>assistant\n<think>\n" : "");
    }

    /// <summary>Scripted logits per label letter; the runner feeds them to the capture processor like the generator would.</summary>
    private sealed class Script
    {
        public Dictionary<char, float> Logits { get; } = new();
        public List<string> Prompts { get; } = [];
        public float Default { get; set; } = -10f;

        public Task<InferenceResponse> Run(string prompt, InferenceOptions options, CancellationToken ct)
        {
            Prompts.Add(prompt);
            Assert.Equal(1, options.MaxTokens);
            var logits = new float[256];
            Array.Fill(logits, Default);
            foreach (var (c, v) in Logits) logits[c] = v;
            options.LogitProcessors![0].Process(logits, [], new ProcessorContext(1f, 0, 0));
            return Task.FromResult(new InferenceResponse
            {
                GeneratedTokenIds = [65],
                Text = "A",
                FinishReason = FinishReason.Length,
                PromptTokenCount = prompt.Length,
                GeneratedTokenCount = 1,
            });
        }
    }

    private static SystemOneRequest Parse(string json) =>
        JsonSerializer.Deserialize(json, ServerJsonContext.Default.SystemOneRequest)!;

    private static Task<SystemOneResponse> Answer(Script s, string json, double temperature = 1.0, int orderings = 1) =>
        SystemOneEndpoint.AnswerAsync(Parse(json),
            new DecisionEvaluator(new CharTokenizer(), new ChatMlTemplate(), s.Run) { Temperature = temperature, Orderings = orderings },
            "test-model", 100_000, default);

    [Fact]
    public void Math_Softmax_Confidence_ExpectedIndex()
    {
        double[] p = DecisionMath.Softmax([2f, 2f, 2f, 2f]);
        Assert.All(p, x => Assert.Equal(0.25, x, 12));
        Assert.Equal(0.0, DecisionMath.Confidence(p), 12);                 // uniform -> 0
        Assert.Equal(1.0, DecisionMath.Confidence([1.0, 0.0, 0.0]), 12);   // certain -> 1
        Assert.Equal(1.5, DecisionMath.ExpectedIndex(p), 12);
        Assert.Equal(1.0, DecisionMath.Softmax([1000f, 999f, 3f]).Sum(), 12);   // stable at large logits
    }

    [Fact]
    public void UserJson_IsTheTev1Contract()
    {
        using var doc = JsonDocument.Parse("\"Purchase was 12 days ago.\"");
        string json = DecisionEvaluator.BuildUserJson(doc.RootElement, "Is the return within the window?",
            DecisionEvaluator.BuildOptions([("yes", "Yes."), ("no", "No.")]));
        Assert.Equal(
            "{\"state\":\"Purchase was 12 days ago.\",\"question\":\"Is the return within the window?\",\"options\":[" +
            "{\"label\":\"A\",\"key\":\"yes\",\"description\":\"Yes.\"},{\"label\":\"B\",\"key\":\"no\",\"description\":\"No.\"}]}", json);
    }

    [Fact]
    public async Task Prompt_HasSystemPrompt_AndClosedThinkingBlock()
    {
        var s = new Script();
        s.Logits['A'] = 5f;
        await Answer(s, """{"state":"x","questions":{"q":{"type":"noul","instructions":"Is it?"}}}""");
        string prompt = s.Prompts.Single();
        Assert.StartsWith("<|im_start|>system\n" + DecisionEvaluator.SystemPrompt, prompt);
        Assert.EndsWith("<|im_start|>assistant\n<think>\n\n</think>\n\n", prompt);
    }

    [Fact]
    public async Task Choice_ReturnsArgmaxName_AndNormalisedProbabilities()
    {
        var s = new Script();
        s.Logits['A'] = 1f; s.Logits['B'] = 4f; s.Logits['C'] = 2f;
        var r = await Answer(s, """
            {"model":"jev-latest","state":{"order":{"total":12}},"questions":{"route":{"type":"choice","instructions":"Where?",
             "criteria":{"billing":"Billing issue","tech":"Technical issue","other":"Anything else"}}}}
            """);
        var a = r.Answers!["route"];
        Assert.Equal("choice", a.Type);
        Assert.Equal("tech", a.Choice);
        Assert.Equal(["billing", "tech", "other"], a.Probabilities!.Keys);          // request order preserved
        Assert.Equal(1.0, a.Probabilities.Values.Sum(), 9);
        Assert.True(a.Probabilities["tech"] > a.Probabilities["other"] && a.Probabilities["other"] > a.Probabilities["billing"]);
        Assert.InRange(a.Confidence!.Value, 0.0, 1.0);
        Assert.Equal("test-model", r.Model);
        Assert.Equal(0, r.Usage!.OutputTokens);
        Assert.Contains("\"state\":{\"order\":{\"total\":12}}", s.Prompts.Single());   // non-string state embedded verbatim
    }

    [Fact]
    public async Task Noul_IsProbabilityOfTrue()
    {
        var s = new Script();
        s.Logits['A'] = 3f; s.Logits['B'] = 0f;
        var r = await Answer(s, """{"state":"s","questions":{"ok":{"type":"noul","instructions":"Is it ok?"}}}""");
        double expected = Math.Exp(3) / (Math.Exp(3) + 1);
        Assert.Equal(expected, r.Answers!["ok"].Noul!.Value, 9);
        Assert.Null(r.Answers["ok"].Choice);
    }

    [Fact]
    public async Task Score_ReturnsLevelProbabilities_Expectation_AndLegend()
    {
        var s = new Script();
        s.Logits['A'] = 0f; s.Logits['B'] = 0f; s.Logits['C'] = 6f;
        var r = await Answer(s, """{"state":"s","questions":{"sev":{"type":"score","instructions":"How severe?","criteria":["none","minor","critical"]}}}""");
        var a = r.Answers!["sev"];
        Assert.Equal("score", a.Type);
        Assert.Equal(["0", "1", "2"], a.Probabilities!.Keys);
        Assert.InRange(a.Score!.Value, 1.9, 2.0);
        Assert.Equal("critical", a.Legend!["2"]);
    }

    [Fact]
    public async Task SeveralQuestions_RunInOrder_AndSumUsage()
    {
        var s = new Script();
        s.Logits['A'] = 1f;
        var r = await Answer(s, """
            {"state":"s","questions":{"a":{"type":"noul","instructions":"one"},"b":{"type":"noul","instructions":"two"}}}
            """);
        Assert.Equal(["a", "b"], r.Answers!.Keys);
        Assert.Equal(2, s.Prompts.Count);
        Assert.Equal(s.Prompts.Sum(p => (long)p.Length), r.Usage!.InputTokens);
        // The state comes first, so the prefix caches can reuse it across questions.
        string firstPrefix = s.Prompts[0][..s.Prompts[0].IndexOf("\"question\"", StringComparison.Ordinal)];
        Assert.StartsWith(firstPrefix, s.Prompts[1]);
    }

    [Theory]
    [InlineData("""{"state":"s","questions":{}}""")]
    [InlineData("""{"state":"s"}""")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"choice","instructions":"x","criteria":{"only":"one"}}}}""")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"choice","instructions":"x"}}}""")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"score","instructions":"x","criteria":["a"]}}}""")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"wat","instructions":"x"}}}""")]
    [InlineData("""{"state":"s","questions":{"q":{"type":"noul"}}}""")]
    public async Task InvalidRequests_AreRejectedBefore_AnyForwardPass(string json)
    {
        var s = new Script();
        var ex = await Assert.ThrowsAsync<SystemOneEndpoint.SystemOneException>(() => Answer(s, json));
        Assert.Equal(422, ex.Status);
        Assert.Empty(s.Prompts);
    }

    /// <summary>
    /// Replays the parsing rules of the Microsoft Agent Framework <c>TypeSafeProtocol.ParseResponse</c> (PR microsoft/agent-framework#8563) against
    /// our serialized response: a non-empty <c>model</c>, an object per question id carrying a matching <c>type</c>, a probability for EVERY requested
    /// choice / level within 0..1, a chosen name that is one of the requested choices, a score within 0..levels-1, 0-indexed level keys.
    /// </summary>
    [Fact]
    public async Task Response_SatisfiesTheAgentFrameworkProviderParser()
    {
        var s = new Script();
        s.Logits['A'] = 2f; s.Logits['B'] = 1f; s.Logits['C'] = 0.5f;
        var r = await Answer(s, """
            {"model":"jev-latest","state":"A customer wrote in about a late parcel.","questions":{
              "urgent":{"type":"noul","instructions":"Is this urgent?"},
              "queue":{"type":"choice","instructions":"Which queue?","criteria":{"shipping":"Shipping","billing":"Billing","other":"Other"}},
              "sev":{"type":"score","instructions":"Severity?","criteria":["low","medium","high"]}}}
            """);
        string wire = JsonSerializer.Serialize(r, ServerJsonContext.Default.SystemOneResponse);
        using var doc = JsonDocument.Parse(wire);
        var root = doc.RootElement;

        Assert.False(string.IsNullOrWhiteSpace(root.GetProperty("model").GetString()));
        var answers = root.GetProperty("answers");

        var urgent = answers.GetProperty("urgent");
        Assert.Equal("noul", urgent.GetProperty("type").GetString());
        Assert.InRange(urgent.GetProperty("noul").GetDouble(), 0.0, 1.0);

        var queue = answers.GetProperty("queue");
        Assert.Equal("choice", queue.GetProperty("type").GetString());
        string[] choices = ["shipping", "billing", "other"];
        Assert.Contains(queue.GetProperty("choice").GetString(), choices);
        foreach (string c in choices)
            Assert.InRange(queue.GetProperty("probabilities").GetProperty(c).GetDouble(), 0.0, 1.0);
        Assert.InRange(queue.GetProperty("confidence").GetDouble(), 0.0, 1.0);

        var sev = answers.GetProperty("sev");
        Assert.Equal("score", sev.GetProperty("type").GetString());
        for (int i = 0; i < 3; i++)
            Assert.InRange(sev.GetProperty("probabilities").GetProperty(i.ToString()).GetDouble(), 0.0, 1.0);
        Assert.InRange(sev.GetProperty("score").GetDouble(), 0.0, 2.0);
        Assert.Equal(JsonValueKind.Object, sev.GetProperty("legend").ValueKind);

        var usage = root.GetProperty("usage");
        Assert.Equal(JsonValueKind.Number, usage.GetProperty("input_tokens").ValueKind);
        Assert.Equal(JsonValueKind.Number, usage.GetProperty("output_tokens").ValueKind);
        Assert.DoesNotContain("null", wire);   // unset fields are omitted, not written as null
    }


    [Fact]
    public async Task Temperature_ScalesLogits_AndPreservesArgmax()
    {
        var s = new Script();
        s.Logits['A'] = 2f; s.Logits['B'] = 0f;
        const string q = """{"state":"s","questions":{"ok":{"type":"noul","instructions":"Is it ok?"}}}""";
        double raw = (await Answer(s, q)).Answers!["ok"].Noul!.Value;
        double sharp = (await Answer(s, q, temperature: 0.5)).Answers!["ok"].Noul!.Value;
        double soft = (await Answer(s, q, temperature: 2.0)).Answers!["ok"].Noul!.Value;
        Assert.Equal(1 / (1 + Math.Exp(-2.0)), raw, 9);
        Assert.Equal(1 / (1 + Math.Exp(-4.0)), sharp, 9);   // logits / 0.5
        Assert.Equal(1 / (1 + Math.Exp(-1.0)), soft, 9);    // logits / 2
        Assert.True(sharp > raw && raw > soft && soft > 0.5);
    }

    [Fact]
    public async Task Orderings2_CancelsAPositionBias_AndRunsTwoForwardPasses()
    {
        // A model that always prefers label "A": with one ordering the first option wins; averaged over forward + reversed it is a tie.
        var s = new Script();
        s.Logits['A'] = 2f; s.Logits['B'] = 0f;
        const string q = """{"state":"s","questions":{"c":{"type":"choice","instructions":"Pick","criteria":{"x":"X","y":"Y"}}}}""";
        var single = await Answer(s, q);
        Assert.Equal("x", single.Answers!["c"].Choice);
        Assert.True(single.Answers["c"].Probabilities!["x"] > 0.85);

        s.Prompts.Clear();
        var both = await Answer(s, q, orderings: 2);
        Assert.Equal(2, s.Prompts.Count);
        Assert.Equal(0.5, both.Answers!["c"].Probabilities!["x"], 9);
        Assert.Equal(0.5, both.Answers["c"].Probabilities!["y"], 9);
        // The reversed prompt lists y first, labelled A again, with each option's own key/description kept.
        Assert.Contains("\"options\":[{\"label\":\"A\",\"key\":\"y\",\"description\":\"Y\"},{\"label\":\"B\",\"key\":\"x\"", s.Prompts[1]);
        Assert.Equal(s.Prompts.Sum(p => (long)p.Length), both.Usage!.InputTokens);
    }

    [Fact]
    public async Task Orderings2_AveragesOptionLogitsInOriginalOrder_AndSkipsScore()
    {
        var s = new Script();
        s.Logits['A'] = 0f; s.Logits['B'] = 0f; s.Logits['C'] = 4f;
        // Score levels are ordinal: reversing them would change their meaning, so only one pass runs.
        await Answer(s, """{"state":"s","questions":{"sev":{"type":"score","instructions":"Severity?","criteria":["low","mid","high"]}}}""", orderings: 2);
        Assert.Single(s.Prompts);
    }

    [Fact]
    public void Request_Deserialises_StringAndObjectState()
    {
        Assert.Equal(JsonValueKind.String, Parse("""{"state":"hi","questions":{}}""").State.ValueKind);
        Assert.Equal(JsonValueKind.Object, Parse("""{"state":{"a":1},"questions":{}}""").State.ValueKind);
        Assert.Equal(JsonValueKind.Undefined, Parse("""{"questions":{}}""").State.ValueKind);
    }
}
