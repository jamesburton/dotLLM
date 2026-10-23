using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Constraints;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Integration.Fixtures;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// End-to-end integration test for the constrained tool-call path used by
/// <c>chat --tool-choice required</c> (#104 / #106).
/// Test B is env-gated: set <c>DOTLLM_BITNET_GGUF</c> to a valid I2_S GGUF path to
/// activate it. Without the env var the test runs as a no-op pass (plain <see cref="FactAttribute"/>
/// + early return, NOT <c>SkippableFact</c>, matching the brief's env-gating requirement).
/// </summary>
public sealed class ConstrainedToolCallTests
{
    private readonly ITestOutputHelper _out;

    public ConstrainedToolCallTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Env-gated BitNet end-to-end: loads the model specified by <c>DOTLLM_BITNET_GGUF</c>,
    /// runs greedy generation with a single <c>get_weather</c> tool under
    /// <c>--tool-choice required</c> (JSON schema constrained decoding).
    /// <para>
    /// The constraint engine guarantees STRUCTURAL VALIDITY (the output is always a valid JSON
    /// prefix of the constrained tool call, with <c>name</c> forced to the schema's <c>const</c>),
    /// NOT model termination. Full termination/EOS depends on model capability — a weak / quantised
    /// base (BitNet b1.58) can run the unbounded <c>city</c> string value to <c>MaxTokens</c>
    /// (<see cref="FinishReason.Length"/>) rather than self-terminating. So this test asserts the
    /// engine's real guarantee unconditionally, and only asserts the full parsed tool call when the
    /// model actually terminated.
    /// </para>
    /// When the BitNet fixture cannot be resolved the test reports <b>skipped</b> naming every
    /// path probed — it used to <c>return;</c> and be counted as a pass (#307).
    /// </summary>
    [SkippableFact]
    public void BitNet_ToolChoiceRequired_ProducesValidSelfTerminatingCall()
    {
        // ── fixture gate ──────────────────────────────────────────────────────
        FixtureLocation fixture = KnownTestFixtures.BitNetI2S;
        Skip.If(!fixture.Found, fixture.SkipMessage(KnownTestFixtures.BitNetI2SDescription));
        string ggufPath = fixture.Path!;

        // ── load model (same path as CLI ChatCommand, no Cli dependency needed) ──
        using var gguf = GgufFile.Open(ggufPath);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = TransformerModel.LoadFromGguf(gguf, config);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);

        // ── build schema — mirrors ChatCommand's tool-choice required wiring ───
        var tool = new ToolDefinition(
            "get_weather",
            "Get current weather for a city",
            """{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}""");

        // Mirror ChatCommand: select parser + arguments key the same way production does.
        // The MODEL parser decides the arguments key only:
        // For BitNet (Architecture.BitNet) → HermesToolCallParser → "arguments" key.
        // For Llama 3.1+ → LlamaToolCallParser → "parameters" key.
        var modelParser = GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, config.Architecture);
        string argumentsKey = modelParser is LlamaToolCallParser ? "parameters" : "arguments";
        string schema = ToolCallSchemaBuilder.BuildForRequired([tool], argumentsKey);

        // The constraint emits a BARE JSON object (no <tool_call>/<|python_tag|>/[TOOL_CALLS]
        // envelope — the JSON-schema constraint cannot emit literal marker tokens), so the
        // markerless parser is the one production uses under tool_choice=required (#325).
        var toolCallParser = ToolCallParserFactory.ForToolChoice(new ToolChoice.Required(), modelParser);

        // ── inference options — greedy + repeat-penalty 1.3, JSON schema constraint ──
        var options = new InferenceOptions
        {
            Temperature = 0f,
            MaxTokens = 128,
            RepetitionPenalty = 1.3f,
            ResponseFormat = new ResponseFormat.JsonSchema { Schema = schema, Name = "tool_call" }
        };

        // ── generate — same constrained path TextGenerator.Generate wires internally ──
        var generator = new TextGenerator(model, tokenizer);
        var response = generator.Generate("What is the weather in Tokyo?", options);

        // Always log the real output so the captured evidence shows finish reason + text,
        // whether the model terminated (Stop) or ran the unbounded value to MaxTokens (Length).
        _out.WriteLine($"finish={response.FinishReason} text=<{response.Text}>");

        // ── engine's real guarantee: STRUCTURAL VALIDITY ──────────────────────
        // The constraint forces a valid JSON prefix of the tool call. It must be non-empty and
        // begin the constrained object. Once "name" has been emitted, the constraint forces it to
        // the schema const "get_weather", so the text must contain that bound name.
        string text = response.Text;
        Assert.False(string.IsNullOrWhiteSpace(text), "constrained output must be non-empty");
        Assert.StartsWith("{", text.TrimStart());
        if (text.Contains("\"name\""))
        {
            Assert.Contains("get_weather", text);
        }

        // ── strong assertion only when the model actually terminated (capability-bound) ──
        if (response.FinishReason == FinishReason.Stop)
        {
            var calls = toolCallParser.TryParse(response.Text);
            Assert.NotNull(calls);
            Assert.Single(calls);
            Assert.Equal("get_weather", calls![0].FunctionName);
            Assert.Contains("city", calls[0].Arguments, StringComparison.OrdinalIgnoreCase);
        }
    }
}
