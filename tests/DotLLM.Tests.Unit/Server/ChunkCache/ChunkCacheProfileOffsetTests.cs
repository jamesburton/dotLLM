using System.Text.Json.Nodes;
using DotLLM.HuggingFace;
using DotLLM.Server.ChunkCache;
using Xunit;

namespace DotLLM.Tests.Unit.Server.ChunkCache;

/// <summary>
/// #1632: when the active model profile prepends a system prompt, the engine messages shift by one. Both endpoints pass
/// that offset to the span locator, so annotated spans are still found where they are in the rendered prompt.
/// </summary>
[Collection("SequentialFileIO")] // DOTLLM_PROFILES_DIR is process-global
public sealed class ChunkCacheProfileOffsetTests : IDisposable
{
    private static readonly string ChunkA =
        "public static int Add(int a, int b)\n{\n    return a + b; // the sum of both operands\n}\n";

    private static readonly string User = "Read this file.\n// A.cs\n" + ChunkA + "What does Add return?";

    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-chunk-profile-" + Guid.NewGuid().ToString("N")[..12]);
    private readonly string? _prev = Environment.GetEnvironmentVariable("DOTLLM_PROFILES_DIR");
    private readonly string _model;

    public ChunkCacheProfileOffsetTests()
    {
        Environment.SetEnvironmentVariable("DOTLLM_PROFILES_DIR", Path.Combine(_scratch, "profiles"));
        _model = ChunkCacheTestServer.WriteModel(Path.Combine(_scratch, "tiny.gguf"));
    }

    public void Dispose()
    {
        Environment.SetEnvironmentVariable("DOTLLM_PROFILES_DIR", _prev);
        try { Directory.Delete(_scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static JsonObject Manifest() => new()
    {
        ["v"] = 1,
        ["policy"] = "shadow",
        ["chunks"] = new JsonArray(ChunkCacheTestServer.Entry("a", 0, User, ChunkA)),
    };

    [Theory]
    [InlineData("openai", false)]
    [InlineData("openai", true)]
    [InlineData("anthropic", false)]
    [InlineData("anthropic", true)]
    public async Task ProfileSystemPrompt_SpansAreStillLocated(string surface, bool withProfile)
    {
        if (withProfile)
            ModelProfileStore.Save("tiny", new ModelProfile { From = "owner/repo:Q4_K_M", System = "You are a profile." });
        await using var server = await ChunkCacheTestServer.StartAsync(_model, ChunkCacheOptions.Create(true, null));
        Assert.Equal(withProfile, server.State.ActiveProfile?.System is "You are a profile.");

        (string Role, string Content)[] conversation = [("user", User)];
        string body = surface == "openai"
            ? ChunkCacheTestServer.OpenAi(conversation, Manifest())
            : ChunkCacheTestServer.Anthropic(conversation, Manifest());
        var response = await server.PostAsync(surface == "openai" ? "/v1/chat/completions" : "/v1/messages", body);
        Assert.Equal(200, response.Status);
        var report = JsonNode.Parse(response.Body)!["chunk_cache"]!;
        Assert.Equal(1, report["chunks"]!["annotated"]!.GetValue<int>());
        // Located and verified, so the span is a miss against the empty store (not ignored as altered or invalid).
        Assert.Equal(0, report["chunks"]!["ignored"]!.GetValue<int>());
        Assert.Equal(1, report["chunks"]!["miss"]!.GetValue<int>());
        Assert.True(report["warnings"] is null || report["warnings"]!.AsArray().Count == 0, report.ToJsonString());
    }
}
