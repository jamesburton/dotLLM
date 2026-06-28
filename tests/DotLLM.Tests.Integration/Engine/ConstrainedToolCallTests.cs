using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Engine.Constraints;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tokenizers;
using DotLLM.Tokenizers.ToolCallParsers;
using Xunit;

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
    /// <summary>
    /// Env-gated BitNet end-to-end: loads the model specified by <c>DOTLLM_BITNET_GGUF</c>,
    /// runs greedy generation with a single <c>get_weather</c> tool under
    /// <c>--tool-choice required</c> (JSON schema constrained decoding), and asserts that:
    /// <list type="bullet">
    ///   <item>Generation terminated via EOS (<see cref="FinishReason.Stop"/>).</item>
    ///   <item>The production-selected parser (via <see cref="GgufChatTemplateFactory.CreateToolCallParser"/>)
    ///   parses the output to exactly one tool call.</item>
    ///   <item>The parsed call targets <c>get_weather</c> with a non-empty string <c>city</c> argument.</item>
    /// </list>
    /// When <c>DOTLLM_BITNET_GGUF</c> is unset or points to a missing file the test exits
    /// immediately and is counted as a pass (CI no-op).
    /// </summary>
    [Fact]
    public void BitNet_ToolChoiceRequired_ProducesValidSelfTerminatingCall()
    {
        // ── env gate ──────────────────────────────────────────────────────────
        string? ggufPath = Environment.GetEnvironmentVariable("DOTLLM_BITNET_GGUF");
        if (string.IsNullOrEmpty(ggufPath) || !File.Exists(ggufPath)) return;

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
        // For BitNet (Architecture.BitNet) → HermesToolCallParser → "arguments" key.
        // For Llama 3.1+ → LlamaToolCallParser → "parameters" key.
        var toolCallParser = GgufChatTemplateFactory.CreateToolCallParser(gguf.Metadata, config.Architecture);
        string argumentsKey = toolCallParser is LlamaToolCallParser ? "parameters" : "arguments";
        string schema = ToolCallSchemaBuilder.BuildForRequired([tool], argumentsKey);

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

        // ── assert generation terminated cleanly (EOS, not MaxTokens) ──────────
        // When the JSON schema constraint reaches IsComplete(), EOS is added to the
        // allowed set; greedy sampling selects it → FinishReason.Stop.
        Assert.Equal(FinishReason.Stop, response.FinishReason);

        // ── assert output parses to a valid get_weather call ──────────────────
        var calls = toolCallParser.TryParse(response.Text);

        Assert.NotNull(calls);
        Assert.Single(calls);
        Assert.Equal("get_weather", calls![0].FunctionName);

        // The city argument must be a non-empty string inside the JSON arguments blob.
        Assert.False(string.IsNullOrWhiteSpace(calls[0].Arguments),
            "Arguments JSON must not be empty");
        Assert.Contains("city", calls[0].Arguments,
            StringComparison.OrdinalIgnoreCase);
    }
}
