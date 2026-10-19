using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Lfm2Moe;

public sealed unsafe class Lfm2LoraTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-lfm-lora-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _owned = [];
    internal static ModelConfig Config => new()
    {
        Architecture = Architecture.Lfm2Moe, HiddenSize = 32, VocabSize = 48, IntermediateSize = 64,
        NumLayers = 3, NumAttentionHeads = 4, NumKvHeads = 2, HeadDim = 8, MaxSequenceLength = 32,
        ShortConvCacheLength = 3, HybridLayout = new() { LayerKind = [HybridLayerKind.ShortConv, HybridLayerKind.Attention, HybridLayerKind.ShortConv], HeadCountKv = [0, 2, 0], FeedForwardLength = [64, 64, 64] },
        Moe = new() { NumExperts = 2, NumExpertsPerTok = 2, MoeIntermediateSize = 64, MlpOnlyLayers = [0], SigmoidGating = true, NormTopKProb = true }
    };
    public Lfm2LoraTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { foreach (var d in _owned) d.Dispose(); Directory.Delete(_dir, true); }
    internal static readonly string[] Sites = ["in_proj", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "mlp.experts.1.gate_proj", "mlp.experts.1.up_proj", "mlp.experts.1.down_proj"];
    private static (int Layer, int In, int Out, string Tensor, int Offset) Site(string site) => site switch
    {
        "in_proj" => (0, 32, 96, "shortconv.in_proj", 0), "out_proj" => (0, 32, 32, "shortconv.out_proj", 0),
        "q_proj" => (1, 32, 32, "attn_q", 0), "k_proj" => (1, 32, 16, "attn_k", 0), "v_proj" => (1, 32, 16, "attn_v", 0), "o_proj" => (1, 32, 32, "attn_output", 0),
        "gate_proj" => (0, 32, 64, "ffn_gate", 0), "up_proj" => (0, 32, 64, "ffn_up", 0), "down_proj" => (0, 64, 32, "ffn_down", 0),
        _ when site.EndsWith("down_proj") => (2, 64, 32, "ffn_down_exps", 2048),
        _ => (2, 32, 64, site.EndsWith("gate_proj") ? "ffn_gate_exps" : "ffn_up_exps", 2048)
    };
    internal static byte[] Build(Func<string, float[], float[]>? edit = null)
    {
        var w = new GgufWriter(); var rng = new Random(1027);
        void Add(string name, int[] shape, bool norm = false)
        {
            int n = shape.Aggregate(1, (a, b) => a * b);
            float[] values = Enumerable.Range(0, n).Select(_ => norm ? 1f : (float)(rng.NextDouble() * 2 - 1) * 0.12f).ToArray();
            if (edit is not null) values = edit(name, values);
            var bytes = new byte[n * 4]; Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            w.AddTensor(name, shape, (uint)QuantizationType.F32, bytes);
        }
        Add("token_embd.weight", [32, 48]); Add("token_embd_norm.weight", [32], true);
        for (int l = 0; l < 3; l++)
        {
            string p = $"blk.{l}."; Add(p + "attn_norm.weight", [32], true); Add(p + "ffn_norm.weight", [32], true);
            if (l != 1) { Add(p + "shortconv.in_proj.weight", [32, 96]); Add(p + "shortconv.out_proj.weight", [32, 32]); Add(p + "shortconv.conv.weight", [3, 32]); }
            else { Add(p + "attn_q.weight", [32, 32]); Add(p + "attn_k.weight", [32, 16]); Add(p + "attn_v.weight", [32, 16]); Add(p + "attn_output.weight", [32, 32]); Add(p + "attn_q_norm.weight", [8], true); Add(p + "attn_k_norm.weight", [8], true); }
            if (l == 0) { Add(p + "ffn_gate.weight", [32, 64]); Add(p + "ffn_up.weight", [32, 64]); Add(p + "ffn_down.weight", [64, 32]); }
            else { Add(p + "ffn_gate_inp.weight", [32, 2]); Add(p + "ffn_gate_exps.weight", [32, 64, 2]); Add(p + "ffn_up_exps.weight", [32, 64, 2]); Add(p + "ffn_down_exps.weight", [64, 32, 2]); }
        }
        return w.Build();
    }
    private Lfm2MoeTransformerModel Load(string name, Func<string, float[], float[]>? edit = null)
    {
        string path = Path.Combine(_dir, name); File.WriteAllBytes(path, Build(edit));
        var gguf = GgufFile.Open(path); var m = Lfm2MoeTransformerModel.LoadFromGguf(gguf, Config);
        _owned.Insert(0, gguf); _owned.Insert(0, m); return m;
    }
    private LoraAdapter Adapter(string site, int? layerOverride = null, int? inOverride = null, string? projectionOverride = null)
    {
        var s = Site(site); int input = inOverride ?? s.In;
        var ad = new LoraAdapter("test", 3, 4.5f, [site]);
        nint a = LoraAdapter.AllocAligned(s.Out * 3), b = LoraAdapter.AllocAligned(input * 3);
        for (int i = 0; i < s.Out * 3; i++) ((float*)a)[i] = MathF.Sin(i + 1) * 0.2f;
        for (int i = 0; i < input * 3; i++) ((float*)b)[i] = MathF.Cos(i + 1) * 0.15f;
        ad.AddLayerWeights(layerOverride ?? s.Layer, projectionOverride ?? site, new(a, b, input, s.Out)); _owned.Add(ad); return ad;
    }
    private static float[] Rows(ITensor t) { using (t) return new ReadOnlySpan<float>((void*)t.DataPointer, t.Shape[0] * t.Shape[1]).ToArray(); }
    private static float Error(float[] a, float[] b) => a.Zip(b).Max(p => MathF.Abs(p.First - p.Second));
    public static IEnumerable<object[]> ProjectionSites => Sites.Select(s => new object[] { s });
    [Theory, MemberData(nameof(ProjectionSites))]
    public void RuntimeMatchesMergedWeights_PrefillAndDecode(string site)
    {
        var model = Load("base.gguf"); var adapter = Adapter(site); var s = Site(site); var factors = adapter.GetLayerWeights(s.Layer, site)!.Value;
        var merged = Load("merged.gguf", (name, values) =>
        {
            if (name != $"blk.{s.Layer}.{s.Tensor}.weight") return values;
            for (int o = 0; o < s.Out; o++) for (int i = 0; i < s.In; i++)
                for (int r = 0; r < 3; r++) values[s.Offset + o * s.In + i] += 1.5f * ((float*)factors.AHandle)[o * 3 + r] * ((float*)factors.BHandle)[r * s.In + i];
            return values;
        });
        float[] baseline = Rows(model.Forward([1, 4, 7], [0, 1, 2], -1)); model.ResetSequenceState();
        float[] runtime = Rows(((IModel)model).Forward([1, 4, 7], [0, 1, 2], -1, null, adapter));
        float[] expected = Rows(merged.Forward([1, 4, 7], [0, 1, 2], -1));
        Assert.True(Error(expected, baseline) > 1e-5f, $"{site} has no measurable effect");
        Assert.True(Error(expected, runtime) < 2e-5f, $"{site}: {Error(expected, runtime)}");
        Assert.True(Error(Rows(merged.Forward([3], [3], -1)), Rows(((IModel)model).Forward([3], [3], -1, null, adapter))) < 2e-5f);
        model.ResetSequenceState(); Assert.Equal(baseline, Rows(model.Forward([1, 4, 7], [0, 1, 2], -1)));
    }
    [Fact]
    public void BatchHonorsEachAdapterAndSequenceState()
    {
        var model = Load("base.gguf"); var adapter = Adapter("in_proj");
        var expected = Rows(((IModel)model).Forward([1, 4], [0, 1], -1, null, adapter));
        model.ResetSequenceState(); var baseline = Rows(model.Forward([1, 4], [0, 1], -1));
        using var first = model.CreateSequenceState(); using var second = model.CreateSequenceState();
        var rows = model.ForwardBatch([
            new() { TokenIds = new[] { 1, 4 }, Positions = new[] { 0, 1 }, KvCache = null!, SsmState = (ISsmState)first!, Adapter = adapter },
            new() { TokenIds = new[] { 1, 4 }, Positions = new[] { 0, 1 }, KvCache = null!, SsmState = (ISsmState)second! }
        ], -1);
        Assert.Equal(expected, Rows(rows[0])); Assert.Equal(baseline, Rows(rows[1]));
    }
    [Fact] public void UnsupportedTargetsRejected()
    {
        var model = Load("base.gguf");
        Assert.Throws<NotSupportedException>(() => ((IModel)model).Forward([1], [0], -1, null, Adapter("in_proj", projectionOverride: "router")));
        Assert.Throws<NotSupportedException>(() => ((IModel)model).Forward([1], [0], -1, null, Adapter("in_proj", projectionOverride: "mlp.experts.99.gate_proj")));
        Assert.Throws<NotSupportedException>(() => ((IModel)model).Forward([1], [0], -1, null, Adapter("in_proj", layerOverride: 2, projectionOverride: "mlp.experts.01.gate_proj")));
    }
    [Fact]
    public void VulkanDirectForwardRejectsAdaptersBeforeAnyDeviceAccess()
    {
        // The guard must run before command recording or state mutation, including through IModel.
        var model = (IModel)System.Runtime.CompilerServices.RuntimeHelpers.GetUninitializedObject(typeof(DotLLM.Vulkan.VulkanLfm2MoeTransformerModel));
        Assert.Throws<NotSupportedException>(() => model.Forward([1], [0], -1, null, Adapter("q_proj")));
    }
    [Fact] public void WrongLayerAndShapeRejectedBeforeStateMutation()
    {
        var model = Load("base.gguf");
        Assert.Throws<ArgumentException>(() => ((IModel)model).Forward([1], [0], -1, null, Adapter("q_proj", layerOverride: 0)));
        Assert.Throws<ArgumentException>(() => ((IModel)model).Forward([1], [0], -1, null, Adapter("in_proj", inOverride: 31)));
        var untouched = Load("untouched.gguf");
        Assert.Equal(Rows(untouched.Forward([1, 4], [0, 1], -1)), Rows(model.Forward([1, 4], [0, 1], -1)));
    }
}
