using System.Diagnostics;
using System.Net;
using System.Text;
using Xunit;
using Xunit.Abstractions;

// Test code: handlers and requests are owned by the HttpClient chains each test disposes.
#pragma warning disable IDISP001, IDISP004, IDISP017, MA0004

namespace DotLLM.ChunkCache.Client.Tests;

/// <summary>Timing tests run alone, after the parallel classes, so other tests' load does not count as overhead.</summary>
[CollectionDefinition(nameof(TimingCollection), DisableParallelization = true)]
public sealed class TimingCollection;

/// <summary>Spec 12.5: handler and decorator overhead under 5 ms.</summary>
[Collection(nameof(TimingCollection))]
public sealed class OverheadTests(ITestOutputHelper output)
{
    /// <summary>The handler and the decorator's lookup add under 5 ms for 100 spans in a 200 KB body (median of 20).</summary>
    [Fact]
    public async Task Overhead_Under5ms()
    {
        var registry = new ChunkSpanRegistry();
        var messages = new List<(string, string)> { ("system", "You are a coding agent.") };
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

        byte[] input = Pipeline.StringBody([.. messages]);
        Assert.InRange(input.Length, 200_000, 260_000);
        using HttpClient http = Pipeline.Client(new CaptureHandler((_, _) => new HttpResponseMessage(HttpStatusCode.OK)), new ChunkCacheBodyHandler(Pipeline.Caps()));
        using HttpClient plain = new(new CaptureHandler((_, _) => new HttpResponseMessage(HttpStatusCode.OK)));

        var samples = new List<double>();
        for (int round = 0; round < 23; round++)
        {
            var sw = Stopwatch.StartNew();
            using (HttpResponseMessage _ = await plain.SendAsync(Pipeline.Post(input))) { }
            double baseline = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            var parts = new List<ChunkPart>();
            foreach ((_, string content) in messages)
            {
                if (registry.TryFind(content, out string? t, out IReadOnlyList<ChunkSpan> spans))
                    parts.Add(new ChunkPart(t, spans));
            }

            using (var scope = ChunkManifestScope.Begin(parts, new ChunkRequestOptions()))
            {
                using HttpResponseMessage _ = await http.SendAsync(Pipeline.Post(input));
                Assert.Equal(100, scope.Outcome!.SentSpans);
            }

            if (round >= 3) samples.Add(sw.Elapsed.TotalMilliseconds - baseline);
        }

        samples.Sort();
        double median = samples[samples.Count / 2];
        output.WriteLine($"median overhead {median:F3} ms (min {samples[0]:F3}, max {samples[^1]:F3})");
        Assert.True(median < 5.0, $"median overhead {median:F2} ms");
    }

}
