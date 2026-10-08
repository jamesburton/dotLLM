using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Qwen4Exp;

/// <summary>
/// #845: runtime LoRA on the Qwen4-Exp CPU path. The reference is the same checkpoint with the adapter MERGED into the weights
/// (<c>W' = W + (alpha / rank) * A B</c>, written into a second GGUF), so runtime == merged is an exact statement about the projection wiring:
/// every projection kind, the fused QSA <c>q | gate</c> output, the value-head-tiled GDN projections and the expert banks, with rank 6 and
/// non-square projections so a swapped A/B or a wrong dimension cannot cancel.
/// </summary>
public sealed unsafe class Qwen4ExpLoraTests : IDisposable
{
    private const int Rank = 6;
    private const float Alpha = 9f;                      // scale 1.5
    private static readonly Qwen4ExpReferenceFixture Fx = Qwen4ExpReferenceFixture.Load("tiny_model.json");

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-q4e-lora-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _d = [];

    public Qwen4ExpLoraTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        foreach (var x in _d) x.Dispose();
        try { Directory.Delete(_dir, true); } catch (IOException) { }
    }

    private (Qwen4ExpTransformerModel Model, ModelConfig Config) Load(string file, Func<string, float[], float[]>? edit = null)
    {
        string path = Path.Combine(_dir, file);
        if (!File.Exists(path)) File.WriteAllBytes(path, Qwen4ExpTinyGguf.Build(Fx, editTensor: edit));
        var (m, g, c) = ModelLoader.LoadFromGguf(path);
        _d.Add(g); _d.Add(m);
        return ((Qwen4ExpTransformerModel)m, c);
    }

    // ── adapter construction ──

    private static readonly Dictionary<string, string> TensorOf = new()
    {
        ["q_proj"] = "attn_q.weight", ["k_proj"] = "attn_k.weight", ["v_proj"] = "attn_v.weight", ["o_proj"] = "attn_output.weight",
        ["in_proj_qkv"] = "attn_qkv.weight", ["in_proj_z"] = "attn_gate.weight", ["in_proj_a"] = "ssm_alpha.weight",
        ["in_proj_b"] = "ssm_beta.weight", ["out_proj"] = "ssm_out.weight",
    };

    private sealed record Site(int Layer, string Proj, int In, int Out, float[] A, float[] B);   // A [out, rank], B [rank, in]

    private List<Site> Sites(ModelConfig cfg, int seed, bool experts = true)
    {
        var rng = new Random(seed);
        int hidden = cfg.HiddenSize, nH = cfg.NumAttentionHeads, d = cfg.HeadDim;
        var gdn = cfg.GdnConfig!.Value;
        int convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState, vDim = gdn.NVHead * gdn.DState;
        var sites = new List<Site>();
        float[] R(int n, float s) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * s; return a; }
        void Add(int layer, string proj, int inDim, int outDim)
            => sites.Add(new Site(layer, proj, inDim, outDim, R(outDim * Rank, 0.5f), R(Rank * inDim, 0.5f / MathF.Sqrt(inDim))));
        for (int l = 0; l < cfg.NumLayers; l++)
        {
            if (cfg.HybridLayout!.LayerKind[l] == HybridLayerKind.GatedDeltaNet)
            {
                Add(l, "in_proj_qkv", hidden, convDim); Add(l, "in_proj_z", hidden, vDim);
                Add(l, "in_proj_a", hidden, gdn.NVHead); Add(l, "in_proj_b", hidden, gdn.NVHead); Add(l, "out_proj", vDim, hidden);
            }
            else
            {
                int kv = cfg.HybridLayout.HeadCountKv[l] * d;
                Add(l, "q_proj", hidden, 2 * nH * d); Add(l, "k_proj", hidden, kv); Add(l, "v_proj", hidden, kv); Add(l, "o_proj", nH * d, hidden);
            }
        }
        if (experts)
        {
            int inter = cfg.Moe!.MoeIntermediateSize;
            Add(1, "mlp.experts.3.gate_proj", hidden, inter); Add(1, "mlp.experts.3.up_proj", hidden, inter); Add(2, "mlp.experts.5.down_proj", inter, hidden);
        }
        return sites;
    }

    private LoraAdapter Adapter(string name, IEnumerable<Site> sites)
    {
        var ad = new LoraAdapter(name, Rank, Alpha, sites.Select(s => s.Proj).Distinct().ToList());
        foreach (var s in sites)
        {
            nint a = LoraAdapter.AllocAligned(s.A.Length), b = LoraAdapter.AllocAligned(s.B.Length);
            s.A.AsSpan().CopyTo(new Span<float>((void*)a, s.A.Length));
            s.B.AsSpan().CopyTo(new Span<float>((void*)b, s.B.Length));
            ad.AddLayerWeights(s.Layer, s.Proj, new LoraLayerWeights(a, b, s.In, s.Out));
        }
        _d.Add(ad);
        return ad;
    }

    /// <summary>The fixture tensor edit that merges <paramref name="sites"/> into the weights: W[out, in] += scale * A[out, r] B[r, in].</summary>
    private static Func<string, float[], float[]> Merge(IEnumerable<Site> sites)
    {
        var byTensor = new Dictionary<string, List<(Site S, int Expert)>>();
        foreach (var s in sites)
        {
            string tensor; int expert = -1;
            if (s.Proj.StartsWith("mlp.experts.", StringComparison.Ordinal))
            {
                var parts = s.Proj.Split('.');                       // mlp.experts.{j}.{proj}
                expert = int.Parse(parts[2]);
                tensor = parts[3] switch { "gate_proj" => "ffn_gate_exps.weight", "up_proj" => "ffn_up_exps.weight", _ => "ffn_down_exps.weight" };
            }
            else tensor = TensorOf[s.Proj];
            string key = $"blk.{s.Layer}.{tensor}";
            if (!byTensor.TryGetValue(key, out var list)) byTensor[key] = list = [];
            list.Add((s, expert));
        }
        return (name, data) =>
        {
            if (!byTensor.TryGetValue(name, out var list)) return data;
            var w = (float[])data.Clone();
            float scale = Alpha / Rank;
            foreach (var (s, expert) in list)
            {
                long baseOff = expert < 0 ? 0 : (long)expert * s.Out * s.In;
                for (int o = 0; o < s.Out; o++)
                    for (int i = 0; i < s.In; i++)
                    {
                        float acc = 0;
                        for (int r = 0; r < Rank; r++) acc += s.A[o * Rank + r] * s.B[r * s.In + i];
                        w[baseOff + (long)o * s.In + i] += scale * acc;
                    }
            }
            return w;
        };
    }

    // ── helpers ──

    private static int[] Ids => Fx.I64("ids").Select(v => (int)v).ToArray();
    private static int[] Pos(int n, int start = 0) => Enumerable.Range(start, n).ToArray();
    private static float[] Rows(ITensor t) { using (t) return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray(); }

    private static float MaxRel(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float worst = 0, scale = 1e-12f;
        foreach (float v in a) scale = MathF.Max(scale, MathF.Abs(v));
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst / scale;
    }

    // ── tests ──

    [Fact]
    public void Runtime_EqualsMergedWeights_OnEveryProjectionKind_AndDiffersFromTheBase()
    {
        var (model, cfg) = Load("base.gguf");
        var sites = Sites(cfg, seed: 3);
        var adapter = Adapter("all", sites);
        var (merged, _) = Load("merged.gguf", Merge(sites));
        int T = Fx.Int("seq_len");

        var baseline = Rows(model.Forward(Ids, Pos(T), -1));
        var runtime = Rows(model.Forward(Ids, Pos(T), -1, kvCache: null, adapter));
        var reference = Rows(merged.Forward(Ids, Pos(T), -1));

        float rel = MaxRel(reference, runtime);
        float effect = MaxRel(reference, baseline);
        Assert.True(effect > 0.05f, $"the adapter barely changes the logits ({effect:E3}): the comparison would not discriminate");
        Assert.True(rel < 2e-3f * effect + 1e-4f, $"runtime vs merged: {rel:E3} (adapter effect {effect:E3})");
        Assert.True(rel * 50 < effect, "runtime LoRA must be far closer to the merged model than the un-adapted model is");
    }

    [Fact]
    public void EachSiteKind_IsWiredIndependently()
    {
        // One site kind at a time: a kind that were silently skipped (or wired to the wrong projection) would equal the base instead of the merge.
        var (model, cfg) = Load("base.gguf");
        var all = Sites(cfg, seed: 4);
        int T = 24;
        var ids = Ids.AsSpan(0, T).ToArray();
        var baseline = Rows(model.Forward(ids, Pos(T), -1));
        foreach (var kind in all.Select(s => s.Proj.StartsWith("mlp.experts.", StringComparison.Ordinal) ? "experts" : s.Proj).Distinct())
        {
            var subset = all.Where(s => (s.Proj.StartsWith("mlp.experts.", StringComparison.Ordinal) ? "experts" : s.Proj) == kind).ToList();
            var (merged, _) = Load($"merged-{kind}.gguf", Merge(subset));
            var want = Rows(merged.Forward(ids, Pos(T), -1));
            var got = Rows(model.Forward(ids, Pos(T), -1, kvCache: null, Adapter($"one-{kind}", subset)));
            float effect = MaxRel(want, baseline);
            Assert.True(effect > 1e-4f, $"{kind}: merged equals base ({effect:E3}); the site is not exercised by this input");
            Assert.True(MaxRel(want, got) < 0.05f * effect + 1e-5f, $"{kind}: runtime {MaxRel(want, got):E3} vs effect {effect:E3}");
        }
    }

    [Fact]
    public void SwitchingAdaptersPerCall_LeavesNoTraceOnTheModel()
    {
        var (model, cfg) = Load("base.gguf");
        var a1 = Adapter("one", Sites(cfg, 5)); var a2 = Adapter("two", Sites(cfg, 6));
        int T = 20;
        var ids = Ids.AsSpan(0, T).ToArray();

        float[] Run(Qwen4ExpTransformerModel m, ILoraAdapter? ad) => Rows(m.Forward(ids, Pos(T), -1, kvCache: null, ad));
        var (fresh, _) = Load("base.gguf");
        var baseFresh = Run(fresh, null);
        var one = Run(model, a1);
        var two = Run(model, a2);
        var baseAgain = Run(model, null);
        var oneAgain = Run(model, a1);

        Assert.NotEqual(one, two);
        Assert.NotEqual(baseFresh, one);
        Assert.Equal(baseFresh, baseAgain);              // bit-identical: no adapter state leaked into the model
        Assert.Equal(one, oneAgain);
        var (f2, _) = Load("base.gguf"); // a separate fresh model with the second adapter alone gives the same logits as after switching
        Assert.Equal(two, Run(f2, a2));
    }

    [Fact]
    public void EngineInterface_RoutesTheAdapter_InsteadOfIgnoringIt()
    {
        // IModel's default Forward(..., adapter) silently ignores the adapter; this model must override it.
        var (model, cfg) = Load("base.gguf");
        var ad = Adapter("iface", Sites(cfg, 7));
        int T = 16;
        var ids = Ids.AsSpan(0, T).ToArray();
        var viaInterface = Rows(((IModel)model).Forward(ids, Pos(T), -1, null, ad));
        var direct = Rows(model.Forward(ids, Pos(T), -1, kvCache: null, ad));
        var plain = Rows(model.Forward(ids, Pos(T), -1));
        Assert.Equal(direct, viaInterface);
        Assert.NotEqual(plain, viaInterface);
    }

    [Fact]
    public void ForwardBatch_AppliesEachRequestsOwnAdapter()
    {
        var (model, cfg) = Load("base.gguf");
        var a1 = Adapter("b1", Sites(cfg, 8)); var a2 = Adapter("b2", Sites(cfg, 9));
        var ids1 = Ids.AsSpan(0, 12).ToArray(); var ids2 = Ids.AsSpan(12, 12).ToArray();
        float[] Solo(int[] ids, ILoraAdapter? ad)
        {
            using var st = model.CreateState();
            using var t = model.Forward(ids, Pos(ids.Length), -1, st, null, lastTokenLogitsOnly: true, ad);
            return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
        }
        var want1 = Solo(ids1, a1); var want2 = Solo(ids2, a2);
        using var s1 = (Qwen4ExpSequenceState)model.CreateSequenceState()!; using var s2 = (Qwen4ExpSequenceState)model.CreateSequenceState()!;
        var res = model.ForwardBatch(
        [
            new SequenceForwardRequest { TokenIds = ids1, Positions = Pos(ids1.Length), KvCache = null!, GdnState = s1, Adapter = a1 },
            new SequenceForwardRequest { TokenIds = ids2, Positions = Pos(ids2.Length), KvCache = null!, GdnState = s2, Adapter = a2 },
        ], -1);
        try
        {
            Assert.Equal(want1, Rows2(res[0])); Assert.Equal(want2, Rows2(res[1]));
        }
        finally { foreach (var r in res) r.Dispose(); }
        Assert.NotEqual(want1, Solo(ids1, null));

        static float[] Rows2(ITensor t) => new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray();
    }

    // ── rejection ──

    private LoraAdapter OneSite(ModelConfig cfg, int layer, string proj, int inDim, int outDim)
        => Adapter("bad", [new Site(layer, proj, inDim, outDim, new float[outDim * Rank], new float[Rank * inDim])]);

    [Theory]
    [InlineData("shared_expert.gate_proj")]
    [InlineData("mlp.shared_expert.up_proj")]
    [InlineData("indexer.q_proj")]
    [InlineData("attn_hyper_connection.input_mix_weight_down")]
    [InlineData("gate_proj")]                       // dense-FFN name: qwen4exp has no dense FFN
    [InlineData("mlp.experts.99.gate_proj")]        // expert index out of range
    [InlineData("mlp.experts.1.foo_proj")]
    [InlineData("lm_head")]
    public void UnsupportedTargets_AreRejectedBeforeAnyCompute(string proj)
    {
        var (model, cfg) = Load("base.gguf");
        var bad = OneSite(cfg, 0, proj, cfg.HiddenSize, 32);
        int T = 8;
        var ex = Assert.Throws<NotSupportedException>(() => model.Forward(Ids.AsSpan(0, T), Pos(T), -1, null, bad));
        Assert.Contains("not supported on qwen4exp", ex.Message);
        Assert.Contains(proj, ex.Message);
        // the rejected call advanced nothing: the model still computes the same thing as a fresh one
        var (fresh, _) = Load("base.gguf");
        Assert.Equal(Rows(fresh.Forward(Ids.AsSpan(0, T), Pos(T), -1)), Rows(model.Forward(Ids.AsSpan(0, T), Pos(T), -1)));
    }

    [Fact]
    public void WrongLayerKind_WrongShape_AndOutOfRangeLayer_AreRejectedWithTheSite()
    {
        var (model, cfg) = Load("base.gguf");
        int hidden = cfg.HiddenSize;
        var gdn = cfg.GdnConfig!.Value;
        int convDim = (2 * gdn.NKHead + gdn.NVHead) * gdn.DState;
        int T = 6;
        int[] ids = Ids.AsSpan(0, T).ToArray();

        var qsaOnGdn = OneSite(cfg, 0, "q_proj", hidden, 2 * cfg.NumAttentionHeads * cfg.HeadDim);
        Assert.Contains("Gated-DeltaNet layer", Assert.Throws<ArgumentException>(() => model.Forward(ids, Pos(T), -1, null, qsaOnGdn)).Message);

        var gdnOnQsa = OneSite(cfg, 3, "in_proj_qkv", hidden, convDim);
        Assert.Contains("QSA attention layer", Assert.Throws<ArgumentException>(() => model.Forward(ids, Pos(T), -1, null, gdnOnQsa)).Message);

        // the QSA q_proj is the FUSED q|gate projection: a plain nH*d-wide factor (what a generic adapter would carry) must not pass
        var unfused = OneSite(cfg, 3, "q_proj", hidden, cfg.NumAttentionHeads * cfg.HeadDim);
        Assert.Contains("does not match", Assert.Throws<ArgumentException>(() => model.Forward(ids, Pos(T), -1, null, unfused)).Message);

        var wrongIn = OneSite(cfg, 1, "out_proj", hidden, 5);
        Assert.Contains("layer 1 projection 'out_proj'", Assert.Throws<ArgumentException>(() => model.Forward(ids, Pos(T), -1, null, wrongIn)).Message);

        var oob = OneSite(cfg, cfg.NumLayers, "o_proj", cfg.NumAttentionHeads * cfg.HeadDim, hidden);
        Assert.Contains("layers", Assert.Throws<ArgumentException>(() => model.Forward(ids, Pos(T), -1, null, oob)).Message);
    }
}
