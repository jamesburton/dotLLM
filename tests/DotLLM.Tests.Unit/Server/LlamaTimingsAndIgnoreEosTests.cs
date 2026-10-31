using System.Text.Json;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>#1345: llama.cpp-shaped <c>timings</c> fields and the <c>ignore_eos</c> / <c>min_tokens</c> request fields.</summary>
public sealed class LlamaTimingsAndIgnoreEosTests
{
    private static JsonElement Serialize(TimingsDto dto)
        => JsonDocument.Parse(JsonSerializer.Serialize(dto, ServerJsonContext.Default.TimingsDto)).RootElement;

    [Fact]
    public void Timings_CarryLlamaCppFieldNames_BesideTheDotLlmOnes()
    {
        var timings = new InferenceTimings
        {
            PrefillTokenCount = 120, CachedTokenCount = 20, PrefillTimeMs = 500,    // 100 prefilled in 500 ms
            DecodeTokenCount = 49, DecodeTimeMs = 980,
        };

        var json = Serialize(TimingsDto.From(timings));

        Assert.Equal(20, json.GetProperty("cache_n").GetInt32());
        Assert.Equal(100, json.GetProperty("prompt_n").GetInt32());
        Assert.Equal(500, json.GetProperty("prompt_ms").GetDouble());
        Assert.Equal(5, json.GetProperty("prompt_per_token_ms").GetDouble(), 6);
        Assert.Equal(200, json.GetProperty("prompt_per_second").GetDouble(), 6);
        Assert.Equal(49, json.GetProperty("predicted_n").GetInt32());
        Assert.Equal(980, json.GetProperty("predicted_ms").GetDouble());
        Assert.Equal(20, json.GetProperty("predicted_per_token_ms").GetDouble(), 6);
        Assert.Equal(50, json.GetProperty("predicted_per_second").GetDouble(), 6);

        // added beside, not replacing
        Assert.Equal(500, json.GetProperty("prefill_time_ms").GetDouble());
        Assert.Equal(200, json.GetProperty("prefill_tokens_per_sec").GetDouble(), 6);
        Assert.Equal(50, json.GetProperty("decode_tokens_per_sec").GetDouble(), 6);
        Assert.Equal(20, json.GetProperty("cached_tokens").GetInt32());
    }

    [Fact]
    public void Timings_WithNothingPrefilledOrDecoded_AreZeroNotNaN()
    {
        var json = Serialize(TimingsDto.From(new InferenceTimings { PrefillTokenCount = 10, CachedTokenCount = 10 }));
        Assert.Equal(0, json.GetProperty("prompt_n").GetInt32());
        Assert.Equal(0, json.GetProperty("prompt_per_token_ms").GetDouble());
        Assert.Equal(0, json.GetProperty("predicted_per_token_ms").GetDouble());
        Assert.Equal(0, json.GetProperty("predicted_per_second").GetDouble());
    }

    [Fact]
    public void BlockingResponses_SerializeTimingsBesideUsage_AndOmitItWhenAbsent()
    {
        var withTimings = new CompletionResponse
        {
            Id = "x", Model = "m", Choices = [],
            Usage = new UsageDto { PromptTokens = 1, CompletionTokens = 2, TotalTokens = 3 },
            Timings = TimingsDto.From(new InferenceTimings { PrefillTokenCount = 1, PrefillTimeMs = 2, DecodeTokenCount = 1, DecodeTimeMs = 4 }),
        };
        var root = JsonDocument.Parse(JsonSerializer.Serialize(withTimings, ServerJsonContext.Default.CompletionResponse)).RootElement;
        Assert.True(root.TryGetProperty("usage", out _));
        Assert.Equal(1, root.GetProperty("timings").GetProperty("prompt_n").GetInt32());

        var chat = new ChatCompletionResponse
        {
            Id = "x", Model = "m", Choices = [],
            Usage = new UsageDto { PromptTokens = 1, CompletionTokens = 2, TotalTokens = 3 },
        };
        var chatRoot = JsonDocument.Parse(JsonSerializer.Serialize(chat, ServerJsonContext.Default.ChatCompletionResponse)).RootElement;
        Assert.False(chatRoot.TryGetProperty("timings", out _));
    }

    [Fact]
    public void Requests_DeserializeIgnoreEosAndMinTokens_AndAbsentMeansNull()
    {
        var chat = JsonSerializer.Deserialize("""{"messages":[{"role":"user","content":"hi"}],"ignore_eos":true,"min_tokens":7}""",
            ServerJsonContext.Default.ChatCompletionRequest)!;
        Assert.True(chat.IgnoreEos);
        Assert.Equal(7, chat.MinTokens);

        var raw = JsonSerializer.Deserialize("""{"prompt":"hi","ignore_eos":true,"min_tokens":3}""", ServerJsonContext.Default.CompletionRequest)!;
        Assert.True(raw.IgnoreEos);
        Assert.Equal(3, raw.MinTokens);

        var empty = JsonSerializer.Deserialize("""{"prompt":"hi"}""", ServerJsonContext.Default.CompletionRequest)!;
        Assert.Null(empty.IgnoreEos);
        Assert.Null(empty.MinTokens);
    }

    [Fact]
    public void Converter_MapsIgnoreEosAndMinTokens_OnChatAndCompletions()
    {
        var defaults = new SamplingDefaults();
        var chat = JsonSerializer.Deserialize("""{"messages":[{"role":"user","content":"hi"}],"ignore_eos":true,"min_tokens":7}""",
            ServerJsonContext.Default.ChatCompletionRequest)!;
        var chatOptions = RequestConverter.ToInferenceOptions(chat, [], defaults, DotLLM.Core.Configuration.ThreadingConfig.Auto);
        Assert.True(chatOptions.IgnoreEos);
        Assert.Equal(7, chatOptions.MinTokens);

        var raw = JsonSerializer.Deserialize("""{"prompt":"hi","ignore_eos":true,"min_tokens":3}""", ServerJsonContext.Default.CompletionRequest)!;
        var rawOptions = RequestConverter.ToInferenceOptions(raw, defaults, DotLLM.Core.Configuration.ThreadingConfig.Auto);
        Assert.True(rawOptions.IgnoreEos);
        Assert.Equal(3, rawOptions.MinTokens);

        var plain = RequestConverter.ToInferenceOptions(
            JsonSerializer.Deserialize("""{"prompt":"hi"}""", ServerJsonContext.Default.CompletionRequest)!, defaults,
            DotLLM.Core.Configuration.ThreadingConfig.Auto);
        Assert.False(plain.IgnoreEos);
        Assert.Equal(0, plain.MinTokens);
    }

    [Fact]
    public void Validator_RejectsNegativeMinTokens()
    {
        var chat = JsonSerializer.Deserialize("""{"messages":[{"role":"user","content":"hi"}],"min_tokens":-1}""",
            ServerJsonContext.Default.ChatCompletionRequest)!;
        Assert.Equal("min_tokens must not be negative", RequestValidator.ValidateChatRequest(chat));

        var raw = JsonSerializer.Deserialize("""{"prompt":"hi","min_tokens":-1}""", ServerJsonContext.Default.CompletionRequest)!;
        Assert.Equal("min_tokens must not be negative", RequestValidator.ValidateCompletionRequest(raw));
    }
}
