using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Server;
using DotLLM.Server.Endpoints;
using DotLLM.Server.Models;
using DotLLM.Tokenizers.Reasoning;
using DotLLM.Vulkan;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.DependencyInjection;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

/// <summary>
/// Issue #1324: qwen4exp on Vulkan sized its K/V caches for 8192 tokens while the server advertised the checkpoint's window, so a long
/// thinking reply (prompt + generation = 12,894) threw an unhandled <see cref="NotSupportedException"/> out of the forward. The window now
/// follows the capacity, overflow is clamped / refused up front, the model's backstop is a 400 <c>context_length_exceeded</c>, and a stream
/// fault is reported in-stream. These tests are pure logic (no device).
/// </summary>
public sealed class Qwen4ExpContextWindowTests
{
    [Theory]
    [InlineData(2051, 32768, 40000, 32768)]   // everything fits: the ceiling
    [InlineData(2051, 32768, 20000, 19968)]   // largest granule-aligned capacity under the budget
    [InlineData(2051, 32768, 3000, 2816)]
    [InlineData(2051, 32768, 100, 2051)]      // nothing above the floor fits: the floor (the caller reports the plan)
    [InlineData(2051, 2051, 5, 2051)]         // ceiling == floor
    [InlineData(4096, 2051, 0, 4096)]         // inverted range cannot go below the floor
    public void ChooseCapacity_IsTheLargestFittingGranule(int floor, int ceiling, int maxFitting, int expected)
    {
        int got = VulkanQwen4ExpTransformerModel.ChooseCapacity(floor, ceiling, c => c <= maxFitting);
        Assert.Equal(expected, got);
        if (expected > floor && expected < ceiling)
            Assert.True(maxFitting - expected < VulkanQwen4ExpTransformerModel.CapacityGranule, "a larger granule would still have fit");
    }

    [Fact]
    public void AdvertisedWindow_ClampsMaxTokens_SoGenerationEndsWithLengthNotACrash()
    {
        // The reported failure: 8192-token capacity, a ~9K-token prompt/history plus a 12K thinking budget.
        var tokenizer = new FixedLengthTokenizer(promptTokens: 6000);
        const int window = 8192;
        string? error = RequestValidator.ValidatePromptLength("p", tokenizer, window, requestedMaxTokens: 12_000, out int effective, out int prompt);
        Assert.Null(error);
        Assert.Equal(6000, prompt);
        Assert.Equal(window - 6000, effective);   // generation stops at the window: finish_reason "length"
    }

    [Fact]
    public void APromptPastTheWindow_IsAContextLengthExceeded400()
    {
        var tokenizer = new FixedLengthTokenizer(promptTokens: 9000);
        string? error = RequestValidator.ValidatePromptLength("p", tokenizer, 8192, 100, out int effective, out _);
        Assert.NotNull(error);
        Assert.Contains("8192", error);
        Assert.Equal(0, effective);
    }

    [Fact]
    public async Task AMidStreamRefusal_ReachesTheHandler_AsContextLengthExceeded()
    {
        var (ctx, body) = Context();
        var app = new ApplicationBuilder(new ServiceCollection().BuildServiceProvider());
        ModelUnavailableException.UseModelUnavailableHandler(app);
        app.Run(c => ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, c, _ => Tokens(new DeviceMemoryAdmissionException("window is 8192; this call would reach 12894")),
            (work, _) => work(), "chatcmpl-1", "m", tools: null, toolParser: null, ReasoningPlan.Disabled, CancellationToken.None));
        await app.Build()(ctx);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.Contains("context_length_exceeded", raw);
        Assert.Contains("12894", raw);
    }

    [Fact]
    public async Task AnyOtherMidStreamFault_IsAnErrorEventAndDone_NotAnUnhandledException()
    {
        var (ctx, body) = Context();
        ctx.Features.Set<IHttpResponseFeature>(new StartedResponseFeature());   // the role chunk has gone out: no status left to set
        await ChatCompletionEndpoint.WriteChatStreamAsync(
            new ChatCompletionRequest { Messages = [], Stream = true }, ctx, _ => Tokens(new NotSupportedException("kaboom")),
            (work, _) => work(), "chatcmpl-1", "m", tools: null, toolParser: null, ReasoningPlan.Disabled, CancellationToken.None);

        string raw = Encoding.UTF8.GetString(body.ToArray());
        Assert.Contains("kaboom", raw);
        Assert.Contains("\"type\":\"error\"", raw);
        Assert.EndsWith("data: [DONE]\n\n", raw);
    }

    [Fact]
    public void Serve_HandsTheCtxSizeAndF16KvToTheModelBeforeItLoads()
    {
        var oldKv = VulkanQwen4ExpTransformerModel.KvStorage;
        var oldReq = VulkanQwen4ExpTransformerModel.RequestedContextTokens;
        string? oldEnv = Environment.GetEnvironmentVariable("DOTLLM_VK_QWEN4EXP_KV");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_QWEN4EXP_KV", null);
            VulkanQwen4ExpTransformerModel.KvStorage = VulkanKvStorage.F32;
            var opts = ServerOptions.Parse(["--model", "m.gguf", "--ctx-size", "20000"]);
            ServerStartup.PrepareQwen4ExpVulkan(Cfg(Architecture.Qwen4Exp), opts);
            Assert.Equal(20000, VulkanQwen4ExpTransformerModel.RequestedContextTokens);
            Assert.Equal(VulkanKvStorage.F16, VulkanQwen4ExpTransformerModel.KvStorage);

            // An explicit DOTLLM_VK_QWEN4EXP_KV (even f32) is the user's choice and is left alone; other architectures are untouched.
            Environment.SetEnvironmentVariable("DOTLLM_VK_QWEN4EXP_KV", "f32");
            VulkanQwen4ExpTransformerModel.KvStorage = VulkanKvStorage.F32;
            ServerStartup.PrepareQwen4ExpVulkan(Cfg(Architecture.Qwen4Exp), opts);
            Assert.Equal(VulkanKvStorage.F32, VulkanQwen4ExpTransformerModel.KvStorage);
            VulkanQwen4ExpTransformerModel.RequestedContextTokens = null;
            ServerStartup.PrepareQwen4ExpVulkan(Cfg(Architecture.Llama), opts);
            Assert.Null(VulkanQwen4ExpTransformerModel.RequestedContextTokens);
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_VK_QWEN4EXP_KV", oldEnv);
            VulkanQwen4ExpTransformerModel.KvStorage = oldKv;
            VulkanQwen4ExpTransformerModel.RequestedContextTokens = oldReq;
        }
    }

    private static DotLLM.Core.Models.ModelConfig Cfg(Architecture a) => new()
    {
        Architecture = a, VocabSize = 8, HiddenSize = 8, IntermediateSize = 8, NumLayers = 1, NumAttentionHeads = 1, NumKvHeads = 1, HeadDim = 8, MaxSequenceLength = 1024,
    };

    private static async IAsyncEnumerable<GenerationToken> Tokens(Exception thenThrow)
    {
        await Task.Yield();
        yield return new GenerationToken(0, "tok", null);
        throw thenThrow;
    }

    private static (DefaultHttpContext Ctx, MemoryStream Body) Context()
    {
        var ctx = new DefaultHttpContext();
        var body = new MemoryStream();
        ctx.Features.Set<IHttpResponseBodyFeature>(new StreamResponseBodyFeature(body));
        ctx.Request.Path = "/v1/chat/completions";
        return (ctx, body);
    }

    private sealed class StartedResponseFeature : HttpResponseFeature
    {
        public override bool HasStarted => true;
    }

    private sealed class FixedLengthTokenizer(int promptTokens) : DotLLM.Tokenizers.ITokenizer
    {
        public int CountTokens(string text) => promptTokens;
        public int[] Encode(string text) => new int[promptTokens];
        public string Decode(ReadOnlySpan<int> tokenIds) => "";
        public int VocabSize => 8;
        public int BosTokenId => 0;
        public int EosTokenId => 1;
        public string DecodeToken(int tokenId) => "";
    }
}
