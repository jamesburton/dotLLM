using System.Text.Json;
using DotLLM.Core.Configuration;
using DotLLM.Cpu;
using DotLLM.Engine.Evaluation;
using DotLLM.Models;
using DotLLM.Models.Evaluation;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Evaluation;

/// <summary>
/// <c>--dump-routing</c> (#1706) on synthetic MoE GGUFs (<see cref="SyntheticGraniteGguf"/>, arch olmoe / granitemoe), which
/// load through the plain CPU <c>TransformerModel</c> and the shared router.
/// </summary>
[Collection("MoeRoutingProbe")]   // the probe is process-global
public sealed class RoutingDumpTests : IDisposable
{
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), "dotllm-routing-" + Guid.NewGuid().ToString("N"));

    public RoutingDumpTests() => Directory.CreateDirectory(_scratch);

    public void Dispose()
    {
        MoeRoutingProbe.Sink = null;
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    private (string Doc, SyntheticGraniteConfig Cfg) Run(SyntheticGraniteConfig cfg, int[][] docs, bool withDump = true)
    {
        string path = Path.Combine(_scratch, $"{cfg.Arch}.gguf");
        SyntheticGraniteGguf.Write(path, cfg);
        var (model, gguf, _) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            var pm = new BackendPerplexityModel(model, -1, returnsAllRows: true);
            var routingOut = new StringWriter();
            using var dump = new RoutingDump(routingOut, expectedMoeLayers: cfg.Layers);
            DocumentEvaluator.Run(pm, docs, default, new StringWriter(), null, withDump ? dump : null);
            return (routingOut.ToString(), cfg);
        }
    }

    private static readonly int[] DocA = [1, 17, 5, 44, 9, 71, 30, 8];
    private static readonly int[] DocB = [1, 17, 5, 44, 9, 71, 30, 8, 55, 12, 3];

    private static List<JsonElement> Lines(string text, out List<JsonDocument> owners)
    {
        owners = [];
        var l = new List<JsonElement>();
        foreach (string line in text.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            var d = JsonDocument.Parse(line);
            owners.Add(d);
            l.Add(d.RootElement);
        }
        return l;
    }

    [Fact]
    public void Olmoe_RoutingShape_LayersExpertsAndWeights()
    {
        var cfg = new SyntheticGraniteConfig { Arch = "olmoe", Layers = 3, Experts = 6, ExpertsUsed = 3, EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0 };
        var (text, _) = Run(cfg, [DocA, DocB]);
        var lines = Lines(text, out var owners);
        try
        {
            Assert.Equal(2, lines.Count);
            var sums = new List<float>();
            foreach (var (o, n) in new[] { (lines[0], DocA.Length), (lines[1], DocB.Length) })
            {
                Assert.Equal(n, o.GetProperty("n_tokens").GetInt32());
                Assert.Equal(3, o.GetProperty("top_k").GetInt32());
                var layers = o.GetProperty("layers").EnumerateArray().ToList();
                Assert.Equal(new[] { 0, 1, 2 }, layers.Select(l => l.GetProperty("layer").GetInt32()));
                foreach (var layer in layers)
                {
                    var experts = layer.GetProperty("experts").EnumerateArray().ToList();
                    var weights = layer.GetProperty("weights").EnumerateArray().ToList();
                    Assert.Equal(n, experts.Count);
                    Assert.Equal(n, weights.Count);
                    for (int t = 0; t < n; t++)
                    {
                        int[] ids = experts[t].EnumerateArray().Select(e => e.GetInt32()).ToArray();
                        float[] w = weights[t].EnumerateArray().Select(e => e.GetSingle()).ToArray();
                        Assert.Equal(3, ids.Length);
                        Assert.All(ids, id => Assert.InRange(id, 0, 5));
                        Assert.Equal(3, ids.Distinct().Count());
                        Assert.All(w, x => Assert.True(x > 0f));
                        // olmoe = softmax WITHOUT top-k renormalisation: kept mass is strictly below 1,
                        // and slot order is descending router score.
                        Assert.True(w.Sum() <= 1f + 1e-5f);
                        sums.Add(w.Sum());
                        Assert.Equal(w.OrderByDescending(x => x), w);
                    }
                }
            }
            // No renormalisation: somewhere the kept mass is visibly below 1 (a renormalised dump would be exactly 1 everywhere).
            Assert.True(sums.Min() < 1f - 1e-3f, $"min kept mass {sums.Min()}");
        }
        finally { owners.ForEach(d => d.Dispose()); }
    }

    [Fact]
    public void Granitemoe_GateWeightsAreRenormalised_SoTheDumpIsThePostNormalisationWeight()
    {
        var cfg = new SyntheticGraniteConfig { Arch = "granitemoe", Layers = 2, Experts = 6, ExpertsUsed = 3 };
        var (text, _) = Run(cfg, [DocA]);
        var lines = Lines(text, out var owners);
        try
        {
            foreach (var layer in lines[0].GetProperty("layers").EnumerateArray())
                foreach (var row in layer.GetProperty("weights").EnumerateArray())
                    Assert.Equal(1.0, row.EnumerateArray().Sum(e => e.GetDouble()), 4);
        }
        finally { owners.ForEach(d => d.Dispose()); }
    }

    [Fact]
    public void Routing_IsCausal_PrefixRowsDoNotChangeWhenTokensAreAppended()
    {
        var cfg = new SyntheticGraniteConfig { Arch = "olmoe", Layers = 3, Experts = 6, ExpertsUsed = 3, EmbeddingScale = 0, AttentionScale = 0, ResidualScale = 0, LogitScale = 0 };
        var (text, _) = Run(cfg, [DocA, DocB]);
        var lines = Lines(text, out var owners);
        try
        {
            var la = lines[0].GetProperty("layers").EnumerateArray().ToList();
            var lb = lines[1].GetProperty("layers").EnumerateArray().ToList();
            for (int l = 0; l < la.Count; l++)
            {
                var ea = la[l].GetProperty("experts").EnumerateArray().ToList();
                var eb = lb[l].GetProperty("experts").EnumerateArray().ToList();
                for (int t = 0; t < DocA.Length; t++)
                    Assert.Equal(ea[t].GetRawText(), eb[t].GetRawText());
            }
            // ...and routing really does vary with position/layer (the check above is not satisfied by a constant).
            var distinctRows = la.SelectMany(x => x.GetProperty("experts").EnumerateArray().Select(r => r.GetRawText())).Distinct().Count();
            Assert.True(distinctRows > 3);
        }
        finally { owners.ForEach(d => d.Dispose()); }
    }

    [Fact]
    public void DenseModel_IsAnError_NotAnEmptyFile()
    {
        var cfg = new SyntheticGraniteConfig { Arch = "granite" };
        var ex = Assert.Throws<InvalidOperationException>(() => Run(cfg, [DocA]));
        Assert.Contains("No MoE routing was captured", ex.Message);
    }

    [Fact]
    public void ProbeIsOffAfterwards_AndRoutingDoesNotChangeTheScores()
    {
        var cfg = new SyntheticGraniteConfig { Arch = "olmoe", Layers = 2, Experts = 6, ExpertsUsed = 2 };
        string path = Path.Combine(_scratch, "olmoe-scores.gguf");
        SyntheticGraniteGguf.Write(path, cfg);
        var (model, gguf, _) = ModelLoader.LoadFromGguf(path, ThreadingConfig.SingleThreaded);
        using (gguf)
        using (model)
        {
            var pm = new BackendPerplexityModel(model, -1, true);
            var off = DocumentEvaluator.Run(pm, [DocA], default, new StringWriter());
            using var dump = new RoutingDump(new StringWriter());
            var on = DocumentEvaluator.Run(pm, [DocA], default, new StringWriter(), null, dump);
            Assert.Equal(off[0].Score.NllSum, on[0].Score.NllSum);      // bit-equal
            Assert.Null(MoeRoutingProbe.Sink);
        }
    }
}

[CollectionDefinition("MoeRoutingProbe", DisableParallelization = true)]
public sealed class MoeRoutingProbeCollection;
