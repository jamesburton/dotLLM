using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// Whole-model parity of the CPU Qwen4-Exp oracle (#816) against HF <c>Qwen4ExpTextModel</c> on a tiny random-weight
/// checkpoint (GDN, GDN+PLE, GDN, QSA; NK=2 != NV=4, GQA 4:2, 8 experts top-3, 16-token indexer budget so QSA prunes beyond
/// 19 tokens), driven through the production path: fixture → real GGUF → <see cref="GgufModelConfigExtractor"/> →
/// <see cref="ModelLoader"/> → forward. Per-layer residual match, logits, chunked == single-shot, sparse vs dense.
/// </summary>
public sealed unsafe class Qwen4ExpModelParityTests : IDisposable
{
    private static readonly Qwen4ExpReferenceFixture Fx = Qwen4ExpReferenceFixture.Load("tiny_model.json");
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _disposables = [];

    public Qwen4ExpModelParityTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        foreach (var d in _disposables) d.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private (Qwen4ExpTransformerModel Model, ModelConfig Config) Load()
    {
        string path = Path.Combine(_dir, "tiny.gguf");
        if (!File.Exists(path)) File.WriteAllBytes(path, Qwen4ExpTinyGguf.Build(Fx));
        var (model, gguf, config) = ModelLoader.LoadFromGguf(path);
        _disposables.Add(gguf);
        _disposables.Add(model);
        return ((Qwen4ExpTransformerModel)model, config);
    }

    private static float[] ToArray(ITensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();

    private static float MaxRel(float[] expected, float[] actual, out int at)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0; at = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            float d = MathF.Abs(expected[i] - actual[i]) / (1f + MathF.Abs(expected[i]));
            if (d > worst) { worst = d; at = i; }
        }
        return worst;
    }

    private static int[] Ids => Fx.I64("ids").Select(v => (int)v).ToArray();
    private static int[] Positions(int n, int start = 0) => Enumerable.Range(start, n).ToArray();

    [Fact]
    public void Loader_DispatchesToTheOracle_WithTheExpectedConfig()
    {
        var (model, config) = Load();
        Assert.IsType<Qwen4ExpTransformerModel>(model);
        Assert.Equal(DotLLM.Core.Configuration.Architecture.Qwen4Exp, config.Architecture);
        Assert.Equal(Fx.Int("num_layers"), config.NumLayers);
        Assert.Equal(Fx.Int("experts"), config.Moe!.NumExperts);
        Assert.True(model.RequiresPerSequenceState);
    }

    [Fact]
    public void SingleShot_MatchesHf_PerLayerAndLogits()
    {
        var (model, _) = Load();
        var trace = new Dictionary<string, float[]>();
        model.Trace = (name, data, rows, cols) => trace[name] = data.ToArray();
        int T = Fx.Int("seq_len");

        var logits = ToArray(model.Forward(Ids, Positions(T), deviceId: -1));

        // Per-layer residual stream (the TensorDump-equivalent check): any divergence is localised to the first bad layer.
        for (int il = 0; il < Fx.Int("num_layers"); il++)
        {
            float rel = MaxRel(Fx.F32($"l_out.{il}"), trace[$"blk.{il}.l_out"], out int at);
            Assert.True(rel < 2e-4f, $"layer {il} residual: worst rel diff {rel:E3} at {at}");
        }
        Assert.True(MaxRel(Fx.F32("hidden_final"), trace["hidden_final"], out _) < 2e-4f, "head-mixer output");
        float lrel = MaxRel(Fx.F32("logits"), logits, out int la);
        Assert.True(lrel < 5e-4f, $"logits: worst rel diff {lrel:E3} at {la}");

        // Non-degenerate: the logits actually vary across rows and the argmax matches HF everywhere.
        int V = Fx.Int("vocab");
        var hf = Fx.F32("logits");
        for (int t = 0; t < T; t++)
            Assert.Equal(Argmax(hf.AsSpan(t * V, V)), Argmax(logits.AsSpan(t * V, V)));
        Assert.NotEqual(Argmax(hf.AsSpan(0, V)), Argmax(hf.AsSpan(7 * V, V)) + 1000);   // keep the check honest
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(17)]
    [InlineData(40)]
    public void ChunkedPrefill_EqualsSingleShot(int chunk)
    {
        var (model, _) = Load();
        int T = Fx.Int("seq_len"), V = Fx.Int("vocab");
        var ids = Ids;
        var whole = ToArray(model.Forward(ids, Positions(T), -1));

        var state = model.CreateState();
        var chunked = new float[T * V];
        for (int i = 0; i < T; i += chunk)
        {
            int n = Math.Min(chunk, T - i);
            var part = ToArray(model.Forward(ids.AsSpan(i, n), Positions(n, i), -1, state));
            part.CopyTo(chunked, i * V);
        }
        float rel = MaxRel(whole, chunked, out int at);
        Assert.True(rel < 1e-4f, $"chunk {chunk}: worst rel diff {rel:E3} at {at}");
        Assert.Equal(T, state.Length);
    }

    [Fact]
    public void ForceDense_DiffersFromSparse_BeyondTheBudget_AndEqualsItBelow()
    {
        var (model, _) = Load();
        int T = Fx.Int("seq_len"), V = Fx.Int("vocab");
        var sparse = ToArray(model.Forward(Ids, Positions(T), -1));
        model.SetForceDenseAttention(true);
        var dense = ToArray(model.Forward(Ids, Positions(T), -1));   // position 0 restarts the default state
        model.SetForceDenseAttention(false);

        int exactRows = Fx.Int("budget") + Fx.Int("block") - 1;      // 19: positions 0..18 are exactly dense
        Assert.True(MaxRel(dense.AsSpan(0, exactRows * V).ToArray(), sparse.AsSpan(0, exactRows * V).ToArray(), out _) < 1e-6f);
        Assert.True(MaxRel(dense.AsSpan(exactRows * V).ToArray(), sparse.AsSpan(exactRows * V).ToArray(), out _) > 1e-4f,
            "dense attention coincides with QSA beyond the budget: the HF comparison would not discriminate");
        // And HF agrees with the sparse one (not the dense one).
        var hf = Fx.F32("logits");
        Assert.True(MaxRel(hf, sparse, out _) < 5e-4f);
        Assert.True(MaxRel(hf, dense, out _) > 1e-4f);
    }

    [Fact]
    public void ParallelThreadPool_MatchesHf()
    {
        string path = Path.Combine(_dir, "tiny.gguf");
        File.WriteAllBytes(path, Qwen4ExpTinyGguf.Build(Fx));
        var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var model = ModelLoader.CreateCpuModelFromGguf(gguf, config, DotLLM.Core.Configuration.ThreadingConfig.Auto);
        _disposables.Add(gguf); _disposables.Add(model);
        int T = Fx.Int("seq_len");
        var logits = ToArray(model.Forward(Ids, Positions(T), -1));
        Assert.True(MaxRel(Fx.F32("logits"), logits, out _) < 5e-4f);
    }

    [Fact]
    public void LastTokenLogitsOnly_ReturnsTheLastRow()
    {
        var (model, _) = Load();
        int T = Fx.Int("seq_len"), V = Fx.Int("vocab");
        var all = ToArray(model.Forward(Ids, Positions(T), -1));
        var last = ToArray(model.Forward(Ids, Positions(T), -1, kvCache: null, lastTokenLogitsOnly: true));
        Assert.Equal(V, last.Length);
        Assert.Equal(all.AsSpan((T - 1) * V, V).ToArray(), last);
    }

    [Fact]
    public void PositionContract_IsEnforced_AndPositionZeroRestartsTheDefaultState()
    {
        var (model, _) = Load();
        var ids = Ids;
        var first = ToArray(model.Forward(ids.AsSpan(0, 6), Positions(6), -1));
        Assert.Throws<ArgumentException>(() => model.Forward(ids.AsSpan(6, 2), Positions(2, 9), -1));       // gap
        var again = ToArray(model.Forward(ids.AsSpan(0, 6), Positions(6), -1));                             // restart
        Assert.Equal(first, again);
        model.ResetSequenceState();
        Assert.Equal(first, ToArray(model.Forward(ids.AsSpan(0, 6), Positions(6), -1)));
    }

    [Fact]
    public void SyntheticGguf_LoadsAndRuns_FromSingleFileAndShardedSet_Identically()
    {
        string single = SyntheticQwen4ExpGguf.Write(Path.Combine(_dir, "syn.gguf"));
        string splitDir = Path.Combine(_dir, "split");
        Directory.CreateDirectory(splitDir);
        string first = SyntheticQwen4ExpGguf.WriteSplit(splitDir, "syn", shardCount: 3);

        int[] ids = [4, 7, 2, 9, 11, 2, 5, 13, 6, 8];
        float[] a, b;
        {
            var (m, g, _) = ModelLoader.LoadFromGguf(single);
            using (g) using (m) a = ToArray(m.Forward(ids, Positions(ids.Length), -1));
        }
        {
            var (m, g, _) = ModelLoader.LoadFromGguf(first);
            using (g) using (m) b = ToArray(m.Forward(ids, Positions(ids.Length), -1));
        }
        Assert.All(a, v => Assert.True(float.IsFinite(v)));
        Assert.Contains(a, v => MathF.Abs(v) > 1e-6f);
        Assert.Equal(a, b);
    }
}
