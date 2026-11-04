using System.Diagnostics;
using System.Net;
using System.Reflection;
using System.Text;
using System.Text.Json.Nodes;
using Xunit;
using Xunit.Abstractions;

// Test code: handlers and requests are owned by the HttpClient chains each test disposes.
#pragma warning disable IDISP001, IDISP004, IDISP017, MA0004

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>Timing tests run alone, after the parallel classes, so other tests' load does not count as overhead.</summary>
[CollectionDefinition(nameof(TimingCollection), DisableParallelization = true)]
public sealed class TimingCollection;

/// <summary>
/// Spec 12.5: handler and decorator overhead under 5 ms for 100 spans in a 200 KB body. The timing test is a benchmark
/// (<c>Category=Benchmark</c>, filtered out of CI like the repository's other benchmarks, and skipped outside Release);
/// its non-timing twin keeps the same body in the default suite.
/// </summary>
[Collection(nameof(TimingCollection))]
public sealed class OverheadTests(ITestOutputHelper output)
{
    private sealed record Workload(ChunkSpanRegistry Registry, List<(string Role, string Content)> Messages, byte[] Body);

    private static Workload Build()
    {
        var registry = new ChunkSpanRegistry();
        var messages = new List<(string Role, string Content)> { ("system", "You are a coding agent.") };
        var rng = new Random(7);
        for (int i = 0; i < 50; i++)
        {
            var sb = new StringBuilder();
            while (sb.Length < 4000)
                sb.Append("    var x").Append(rng.Next()).Append(" = Compute(é, ").Append(i).Append(");\n");
            string text = sb.ToString();
            int mid = text.IndexOf('\n', 2000) + 1;
            registry.Register(text, [new ChunkSpan($"f{i}:a", ChunkHash.Of(text[..mid]), 0, mid), new ChunkSpan($"f{i}:b", ChunkHash.Of(text[mid..]), mid)]);
            messages.Add(("tool", text));
        }

        byte[] body = Pipeline.StringBody([.. messages]);
        Assert.InRange(body.Length, 200_000, 260_000);
        return new Workload(registry, messages, body);
    }

    /// <summary>What the decorator does per call: look every content text up in the registry.</summary>
    private static List<ChunkPart> Lookup(Workload w)
    {
        var parts = new List<ChunkPart>();
        foreach ((_, string content) in w.Messages)
        {
            if (w.Registry.TryFind(content, out string? t, out IReadOnlyList<ChunkSpan> spans))
                parts.Add(new ChunkPart(t, spans));
        }

        return parts;
    }

    /// <summary>The benchmark's body, without the clock: all 100 spans are sent, two per tool message, in order.</summary>
    [Fact]
    public async Task Overhead_Workload_AllSpansSent()
    {
        Workload w = Build();
        var capture = CaptureHandler.Ok();
        using HttpMessageInvoker invoker = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = capture });
        using var scope = ChunkManifestScope.Begin(Lookup(w), new ChunkRequestOptions());
        using HttpResponseMessage _ = await invoker.SendAsync(Pipeline.Post(w.Body), CancellationToken.None);

        Assert.Equal(100, scope.Outcome!.SentSpans);
        JsonArray chunks = JsonNode.Parse(Pipeline.SplicedManifest(w.Body, capture.LastBody))!["chunks"]!.AsArray();
        Assert.Equal(100, chunks.Count);
        Assert.Equal(Enumerable.Range(1, 50).SelectMany(i => new[] { i, i }), chunks.Select(c => (int)c!["at"]!["message"]!));
        Assert.Equal(Enumerable.Range(0, 50).SelectMany(i => new[] { $"f{i}:a", $"f{i}:b" }), chunks.Select(c => (string)c!["id"]!));
    }

    /// <summary>
    /// The handler alone (through <see cref="HttpMessageInvoker"/>) plus the decorator's lookups, against the same
    /// terminal without the handler: the median of 120 interleaved (baseline, measured) pairs after 30 warm-up pairs.
    /// Release only: a Debug build measures the JIT's unoptimised code, not the library.
    /// </summary>
    [SkippableFact]
    [Trait("Category", "Benchmark")]
    public async Task Overhead_Under5ms()
    {
        Skip.If(typeof(ChunkCacheBodyHandler).Assembly.GetCustomAttribute<DebuggableAttribute>()?.IsJITOptimizerDisabled == true,
            "timing runs only in a Release build (dotnet test -c Release --filter Category=Benchmark)");

        Workload w = Build();
        HttpResponseMessage Ok(HttpRequestMessage _, byte[]? __) => new(HttpStatusCode.OK);
        using HttpMessageInvoker plain = new(new CaptureHandler(Ok));
        using HttpMessageInvoker handled = new(new ChunkCacheBodyHandler(Pipeline.Caps()) { InnerHandler = new CaptureHandler(Ok) });

        var diffs = new List<double>();
        var sw = new Stopwatch();
        for (int round = 0; round < 150; round++)
        {
            sw.Restart();
            using (await plain.SendAsync(Pipeline.Post(w.Body), CancellationToken.None)) { }
            double baseline = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            using (var scope = ChunkManifestScope.Begin(Lookup(w), new ChunkRequestOptions()))
            using (await handled.SendAsync(Pipeline.Post(w.Body), CancellationToken.None))
            {
                if (scope.Outcome!.SentSpans != 100) Assert.Fail("not every span was sent");
            }

            double measured = sw.Elapsed.TotalMilliseconds;
            if (round >= 30) diffs.Add(measured - baseline);
        }

        diffs.Sort();
        double median = diffs[diffs.Count / 2];
        output.WriteLine($"median overhead {median:F3} ms over {diffs.Count} pairs (p10 {diffs[diffs.Count / 10]:F3}, p90 {diffs[diffs.Count * 9 / 10]:F3})");
        Assert.True(median < 5.0, $"median overhead {median:F2} ms");
    }
}
