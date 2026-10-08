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
    public void Q8_0_F16_QuantizedCheckpoint_StaysCloseToHf()
    {
        // The real file stores hc down/up, projections and the router as Q8_0 and the table quantised: exercise those dispatch
        // paths (quant GEMMs, quant embedding rows, quant PLE table rows, 3D expert banks) on the tiny geometry.
        string path = Path.Combine(_dir, "tiny-q.gguf");
        File.WriteAllBytes(path, Qwen4ExpTinyGguf.Build(Fx, quantize: true));
        var (model, gguf, _) = ModelLoader.LoadFromGguf(path);
        _disposables.Add(gguf); _disposables.Add(model);
        int T = Fx.Int("seq_len"), V = Fx.Int("vocab");
        var logits = ToArray(model.Forward(Ids, Positions(T), -1));
        var hf = Fx.F32("logits");
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));

        double num = 0, den = 0;
        for (int i = 0; i < hf.Length; i++) { num += Math.Pow(hf[i] - logits[i], 2); den += Math.Pow(hf[i], 2); }
        double relRms = Math.Sqrt(num / den);
        Assert.True(relRms < 0.08, $"Q8_0 logits drifted from HF: relative RMS {relRms:F4}");
        int agree = 0;
        for (int t = 0; t < T; t++) if (Argmax(hf.AsSpan(t * V, V)) == Argmax(logits.AsSpan(t * V, V))) agree++;
        Assert.True(agree >= T * 3 / 4, $"top-1 agreement {agree}/{T}");
        Assert.True(relRms > 1e-5, "quantised path is bit-identical to F32: quantisation did not apply");
    }

    [Fact]
    public void RealBudget_ContextBeyond2051_SparseDiffersFromDense_AndChunkingHolds()
    {
        // Same tiny weights, but the REAL indexer budget (2048 tokens = 512 blocks) and a 2300-token context.
        string path = Path.Combine(_dir, "tiny-long.gguf");
        File.WriteAllBytes(path, Qwen4ExpTinyGguf.Build(Fx, contextLength: 4096, budgetTokens: 2048));
        var (m, gguf, _) = ModelLoader.LoadFromGguf(path);
        _disposables.Add(gguf); _disposables.Add(m);
        var model = (Qwen4ExpTransformerModel)m;
        const int T = 2300;
        int V = Fx.Int("vocab");
        var rng = new Random(5);
        var ids = Enumerable.Range(0, T).Select(_ => rng.Next(6, V)).ToArray();

        var sparse = ToArray(model.Forward(ids, Positions(T), -1, kvCache: null, lastTokenLogitsOnly: false));
        model.SetForceDenseAttention(true);
        var dense = ToArray(model.Forward(ids, Positions(T), -1, kvCache: null, lastTokenLogitsOnly: false));
        model.SetForceDenseAttention(false);

        int exact = 2048 + Fx.Int("block") - 1;   // 2051 tokens: positions 0..2050
        Assert.True(MaxRel(dense.AsSpan(0, exact * V).ToArray(), sparse.AsSpan(0, exact * V).ToArray(), out _) < 1e-6f,
            "rows within the budget must be exactly dense");
        Assert.True(MaxRel(dense.AsSpan(exact * V).ToArray(), sparse.AsSpan(exact * V).ToArray(), out _) > 1e-5f,
            "QSA never diverged from dense beyond 2051 tokens");

        var state = model.CreateState();
        var chunked = new float[T * V];
        for (int i = 0; i < T; i += 512)
        {
            int n = Math.Min(512, T - i);
            ToArray(model.Forward(ids.AsSpan(i, n), Positions(n, i), -1, state)).CopyTo(chunked, i * V);
        }
        Assert.True(MaxRel(sparse, chunked, out int at) < 1e-4f, $"chunked(512) vs single-shot at {at}");
    }

    [Fact]
    public void EngineKvCacheAndBatchedDispatch_AreRefused_NotSilentlyMisused()
    {
        var (model, _) = Load();
        var ids = Ids.AsSpan(0, 4).ToArray();
        using var cache = new DotLLM.Engine.KvCache.SimpleKvCache(1, 1, 8, 16);
        var ex = Assert.Throws<NotSupportedException>(() => model.Forward(ids, Positions(4), -1, cache));
        Assert.Contains("#817", ex.Message);
        Assert.Throws<NotSupportedException>(() => model.Forward(ids, Positions(4), -1, cache, true));
        var req = new SequenceForwardRequest { TokenIds = ids, Positions = Positions(4), KvCache = cache };
        Assert.Throws<NotSupportedException>(() => ((IModel)model).ForwardBatch([req, req], -1));
        // The refusal happened before any state was touched.
        Assert.Equal(4, ToArray(model.Forward(ids, Positions(4), -1)).Length / Fx.Int("vocab"));
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
