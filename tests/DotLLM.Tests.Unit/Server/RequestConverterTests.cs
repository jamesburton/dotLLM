using DotLLM.Core.Configuration;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Tests for <see cref="RequestConverter"/> wiring of logit_bias / frequency_penalty /
/// presence_penalty (and the DRY / top-n-sigma request fields) into <see cref="InferenceOptions"/>.
/// </summary>
public class RequestConverterTests
{
    private static readonly SamplingDefaults Defaults = new();
    private static readonly ThreadingConfig Threading = ThreadingConfig.Auto;

    [Fact]
    public void ToInferenceOptions_Chat_FrequencyAndPresencePenalty_Wired()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
            FrequencyPenalty = 0.5f,
            PresencePenalty = 0.25f,
        };

        var options = RequestConverter.ToInferenceOptions(request, [], Defaults, Threading);

        Assert.Equal(0.5f, options.FrequencyPenalty);
        Assert.Equal(0.25f, options.PresencePenalty);
    }

    [Fact]
    public void ToInferenceOptions_Chat_LogitBias_ParsedToTokenIdKeyedDictionary()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
            LogitBias = new Dictionary<string, float> { ["1234"] = -100f, ["5"] = 5.5f },
        };

        var options = RequestConverter.ToInferenceOptions(request, [], Defaults, Threading);

        Assert.NotNull(options.LogitBias);
        Assert.Equal(-100f, options.LogitBias![1234]);
        Assert.Equal(5.5f, options.LogitBias![5]);
    }

    [Fact]
    public void ToInferenceOptions_Chat_NoLogitBias_IsNull()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
        };

        var options = RequestConverter.ToInferenceOptions(request, [], Defaults, Threading);

        Assert.Null(options.LogitBias);
    }

    [Fact]
    public void ToInferenceOptions_Chat_TopNSigmaAndDryFields_Wired()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
            TopNSigma = 1.5f,
            DryMultiplier = 0.8f,
            DryBase = 2.0f,
            DryAllowedLength = 3,
            DryPenaltyLastN = 256,
            DrySequenceBreakers = ["\n", "."],
        };

        var options = RequestConverter.ToInferenceOptions(request, [], Defaults, Threading);

        Assert.Equal(1.5f, options.TopNSigma);
        Assert.Equal(0.8f, options.DryMultiplier);
        Assert.Equal(2.0f, options.DryBase);
        Assert.Equal(3, options.DryAllowedLength);
        Assert.Equal(256, options.DryPenaltyLastN);
        Assert.Equal(["\n", "."], options.DrySequenceBreakers);
    }

    [Fact]
    public void ToInferenceOptions_Chat_TopNSigmaDefault_IsDisabled()
    {
        var request = new ChatCompletionRequest
        {
            Messages = [new ChatMessageDto { Role = "user", Content = "hi" }],
        };

        var options = RequestConverter.ToInferenceOptions(request, [], Defaults, Threading);

        Assert.True(options.TopNSigma < 0f);
        Assert.Equal(0f, options.DryMultiplier);
    }

    [Fact]
    public void ToInferenceOptions_Completion_FrequencyAndPresencePenalty_Wired()
    {
        var request = new CompletionRequest
        {
            Prompt = "hi",
            FrequencyPenalty = 0.3f,
            PresencePenalty = 0.1f,
        };

        var options = RequestConverter.ToInferenceOptions(request, Defaults, Threading);

        Assert.Equal(0.3f, options.FrequencyPenalty);
        Assert.Equal(0.1f, options.PresencePenalty);
    }

    [Fact]
    public void ToInferenceOptions_Completion_LogitBias_ParsedToTokenIdKeyedDictionary()
    {
        var request = new CompletionRequest
        {
            Prompt = "hi",
            LogitBias = new Dictionary<string, float> { ["42"] = -1.0f },
        };

        var options = RequestConverter.ToInferenceOptions(request, Defaults, Threading);

        Assert.NotNull(options.LogitBias);
        Assert.Equal(-1.0f, options.LogitBias![42]);
    }

    [Fact]
    public void ParseLogitBias_NonNumericKeys_Skipped()
    {
        var raw = new Dictionary<string, float> { ["not_a_token_id"] = 5f, ["10"] = 2f };

        var result = RequestConverter.ParseLogitBias(raw);

        Assert.NotNull(result);
        Assert.Single(result!);
        Assert.Equal(2f, result![10]);
    }

    [Fact]
    public void ParseLogitBias_NullOrEmpty_ReturnsNull()
    {
        Assert.Null(RequestConverter.ParseLogitBias(null));
        Assert.Null(RequestConverter.ParseLogitBias(new Dictionary<string, float>()));
    }
}
