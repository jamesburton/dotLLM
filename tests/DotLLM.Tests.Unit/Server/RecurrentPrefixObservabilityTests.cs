using System.Text.Json;
using System.Text;
using Microsoft.AspNetCore.Http;
using DotLLM.Engine;
using DotLLM.Engine.PromptCache;
using DotLLM.Server.Endpoints;
using DotLLM.Server;
using DotLLM.Server.Models;
using Xunit;

namespace DotLLM.Tests.Unit.Server;

public sealed class RecurrentPrefixObservabilityTests
{
    [Fact]
    public void Timings_ExposeSnapshotRestoreTimeEvenWhenZero()
    {
        string json = JsonSerializer.Serialize(new TimingsDto { CachedTokens = 12 }, ServerJsonContext.Default.TimingsDto);
        using var document = JsonDocument.Parse(json);
        Assert.True(document.RootElement.TryGetProperty("cache_restore_ms", out var restore));
        Assert.Equal(0, restore.GetDouble());
        Assert.Equal(12, document.RootElement.GetProperty("cached_tokens").GetInt32());
    }

    [Fact]
    public void UsageChunk_ExposesStandardPromptTokenDetails()
    {
        var chunk = DotLLM.Server.Endpoints.ChatCompletionEndpoint.BuildUsageChunk("req", "model", 20, 2);
        string json = JsonSerializer.Serialize(chunk, ServerJsonContext.Default.ChatCompletionChunk);
        using var document = JsonDocument.Parse(json);
        var usage = document.RootElement.GetProperty("usage");
        Assert.True(usage.TryGetProperty("prompt_tokens_details", out var details));
        Assert.Equal(0, details.GetProperty("cached_tokens").GetInt32());
    }
    [Fact]
    public void Timings_MapHostHitWithoutLosingExistingMeasurements()
    {
        var dto = TimingsDto.From(new InferenceTimings
        {
            PrefillTokenCount = 20, CachedTokenCount = 12, PrefillTimeMs = 8,
            CacheTier = "host", CacheRestoreMilliseconds = 1.25,
        });
        using var document = JsonDocument.Parse(JsonSerializer.Serialize(dto, ServerJsonContext.Default.TimingsDto));
        Assert.Equal("host", document.RootElement.GetProperty("cache_tier").GetString());
        Assert.Equal(1.25, document.RootElement.GetProperty("cache_restore_ms").GetDouble());
        Assert.Equal(12, document.RootElement.GetProperty("cached_tokens").GetInt32());
    }

    [Theory]
    [InlineData(12, 12)]
    [InlineData(99, 20)]
    [InlineData(-1, 0)]
    public void UsageChunk_ReportsBoundedCachedTokens(int cached, int expected)
    {
        var chunk = ChatCompletionEndpoint.BuildUsageChunk("req", "model", 20, 2, cachedTokens: cached);
        Assert.Equal(expected, chunk.Usage!.PromptTokensDetails!.CachedTokens);
        Assert.Equal(22, chunk.Usage.TotalTokens);
    }

    [Fact]
    public void RecurrentOnlyCache_ReportsEnabledAndHostCounters()
    {
        var stats = PromptCacheEndpoint.BuildStats(null, new RecurrentPrefixCacheStatistics
        {
            Enabled = true, Entries = 2, HostBytes = 1024, HostHits = 3, HostBudgetBytes = 4096,
        });
        using var document = JsonDocument.Parse(JsonSerializer.Serialize(stats, ServerJsonContext.Default.PromptCacheStatsResponse));
        Assert.True(document.RootElement.GetProperty("enabled").GetBoolean());
        var recurrent = document.RootElement.GetProperty("recurrent");
        Assert.Equal(3, recurrent.GetProperty("host_hits").GetInt64());
        Assert.Equal(1024, recurrent.GetProperty("host_bytes").GetInt64());
        Assert.False(PromptCacheEndpoint.BuildStats(null, null).Enabled);
    }

    [Fact]
    public void MultiChoiceUsage_CountsPromptCacheOnce()
    {
        var response = new InferenceResponse
        {
            GeneratedTokenIds = [1, 2], Text = "ok", FinishReason = default,
            PromptTokenCount = 20, GeneratedTokenCount = 2,
            Timings = new InferenceTimings { CachedTokenCount = 12 },
        };
        var usage = ChatCompletionEndpoint.BuildMultiChoiceUsage([response, response]);
        Assert.Equal(20, usage.PromptTokens);
        Assert.Equal(4, usage.CompletionTokens);
        Assert.Equal(12, usage.PromptTokensDetails!.CachedTokens);
    }
    [Fact]
    public async Task CompletionStream_ReportsCachedUsageAndRestoreTiming()
    {
        using var body = new MemoryStream();
        var context = new DefaultHttpContext { Response = { Body = body } };
        await CompletionEndpoint.WriteCompletionStreamAsync(context, _ => CachedTokens(), (work, _) => work(),
            "req", "model", true, CancellationToken.None);
        string raw = Encoding.UTF8.GetString(body.ToArray());
        var chunks = raw.Split("\n\n", StringSplitOptions.RemoveEmptyEntries)
            .Where(line => line.StartsWith("data: {", StringComparison.Ordinal))
            .Select(line => JsonDocument.Parse(line[6..])).ToArray();
        try
        {
            var timing = chunks.Single(chunk => chunk.RootElement.TryGetProperty("timings", out _)).RootElement.GetProperty("timings");
            Assert.Equal("host", timing.GetProperty("cache_tier").GetString());
            Assert.Equal(1.25, timing.GetProperty("cache_restore_ms").GetDouble());
            var usage = chunks[^1].RootElement.GetProperty("usage");
            Assert.Equal(20, usage.GetProperty("prompt_tokens").GetInt32());
            Assert.Equal(12, usage.GetProperty("prompt_tokens_details").GetProperty("cached_tokens").GetInt32());
        }
        finally { foreach (var chunk in chunks) chunk.Dispose(); }
    }

    private static async IAsyncEnumerable<GenerationToken> CachedTokens()
    {
        await Task.CompletedTask;
        yield return new GenerationToken(1, "ok", default,
            new InferenceTimings { PrefillTokenCount = 20, CachedTokenCount = 12, CacheTier = "host", CacheRestoreMilliseconds = 1.25 });
    }
}
