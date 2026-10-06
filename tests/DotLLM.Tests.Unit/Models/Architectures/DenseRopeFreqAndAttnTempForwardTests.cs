using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Issue #743 model-level discrimination: a dense Llama forward must change when (a)
/// <c>rope_freqs.weight</c> factors are present and (b) Mistral-3 attention temperature is on —
/// and must NOT change at position 0 (angle 0; temperature factor exactly 1).
/// </summary>
public sealed class DenseRopeFreqAndAttnTempForwardTests : IDisposable
{
    private const int H = 16, L = 2, NH = 2, HD = 8, V = 8, FF = 24;
    private readonly string _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-743-{Guid.NewGuid():N}");

    public DenseRopeFreqAndAttnTempForwardTests() => Directory.CreateDirectory(_scratch);
    public void Dispose() { try { Directory.Delete(_scratch, true); } catch { } }

    private static ModelConfig Cfg(float temp = 0f, int floor = 0) => new()
    {
        Architecture = Architecture.Mistral,
        VocabSize = V, HiddenSize = H, IntermediateSize = FF, NumLayers = L,
        NumAttentionHeads = NH, NumKvHeads = NH, HeadDim = HD, MaxSequenceLength = 64,
        NormEpsilon = 1e-5f,
        RoPEConfig = new RoPEConfig(Theta: 10000f, DimensionCount: HD, Type: RoPEType.Norm),
        AttnTemperatureScale = temp, AttnTemperatureFloorScale = floor,
    };

    private static void Add(SafetensorsFixtureBuilder b, string name, int[] shape, float amp, int seed, float center = 0, float jitter = 0)
    {
        long n = 1; foreach (int d in shape) n *= d;
        var v = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            v[i] = jitter > 0 ? center + jitter * MathF.Cos(phi) : amp * MathF.Cos(phi);
        }
        b.AddFloat32(name, shape, v);
    }

    private string WriteFixture()
    {
        string path = Path.Combine(_scratch, "m.safetensors");
        var b = new SafetensorsFixtureBuilder();
        Add(b, "model.embed_tokens.weight", [V, H], 0.5f, 1);
        Add(b, "model.norm.weight", [H], 0, 2, 1f, 0.05f);
        Add(b, "lm_head.weight", [V, H], 0.5f, 3);
        for (int i = 0; i < L; i++)
        {
            int s = 10 * (i + 1); string p = $"model.layers.{i}";
            Add(b, $"{p}.input_layernorm.weight", [H], 0, s, 1f, 0.05f);
            Add(b, $"{p}.post_attention_layernorm.weight", [H], 0, s + 1, 1f, 0.05f);
            Add(b, $"{p}.self_attn.q_proj.weight", [NH * HD, H], 0.8f, s + 2);
            Add(b, $"{p}.self_attn.k_proj.weight", [NH * HD, H], 0.8f, s + 3);
            Add(b, $"{p}.self_attn.v_proj.weight", [NH * HD, H], 0.5f, s + 4);
            Add(b, $"{p}.self_attn.o_proj.weight", [H, NH * HD], 0.5f, s + 5);
            Add(b, $"{p}.mlp.gate_proj.weight", [FF, H], 0.3f, s + 6);
            Add(b, $"{p}.mlp.up_proj.weight", [FF, H], 0.3f, s + 7);
            Add(b, $"{p}.mlp.down_proj.weight", [H, FF], 0.3f, s + 8);
        }
        b.WriteTo(path);
        return path;
    }

    private float[] Run(ModelConfig cfg, float[]? factors, int[] ids, int[] pos)
    {
        using var sf = SafetensorsFile.Open(WriteFixture());
        var w = TransformerWeightsSafetensorsLoader.Load(sf, cfg, null);
        w.RopeFreqFactors = factors;
        using var model = TransformerModel.BuildFromPrebuiltWeights(w, cfg);
        using ITensor logits = model.Forward(ids, pos, deviceId: -1, kvCache: null);
        var r = new float[checked((int)logits.Shape.ElementCount)];
        unsafe { new ReadOnlySpan<float>((void*)logits.DataPointer, r.Length).CopyTo(r); }
        return r;
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        float m = 0; for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i])); return m;
    }

    private static readonly int[] Ids = [1, 2, 3, 4, 5, 6, 7, 0, 1, 2];
    private static readonly int[] Pos = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

    [Fact]
    public void RopeFreqFactors_ChangeMultiPositionLogits_ButNotPositionZero()
    {
        var f = new float[HD / 2]; for (int i = 0; i < f.Length; i++) f[i] = 1f + 3f * i; // 1,4,7,10
        Assert.Equal(Run(Cfg(), null, [2], [0]), Run(Cfg(), f, [2], [0]));
        Assert.True(MaxDiff(Run(Cfg(), null, Ids, Pos), Run(Cfg(), f, Ids, Pos)) > 1e-4f,
            "rope_freqs.weight had no effect on a dense model (the #743 bug).");
    }

    [Fact]
    public void AttnTemperature_ChangesLogitsBeyondFloor_ButNotBelowIt()
    {
        // floor = 4: positions 0..3 have factor exactly 1; 4+ get log(2)*scale+1 etc.
        Assert.Equal(Run(Cfg(), null, Ids[..4], Pos[..4]), Run(Cfg(0.5f, 4), null, Ids[..4], Pos[..4]));
        Assert.True(MaxDiff(Run(Cfg(), null, Ids, Pos), Run(Cfg(0.5f, 4), null, Ids, Pos)) > 1e-4f,
            "attention temperature had no effect past the floor scale.");
    }

    [Fact]
    public void AttnTemperatureAt_MatchesLlamaCppFormula()
    {
        var c = Cfg(0.1f, 8192);
        Assert.Equal(1.0f, c.AttnTemperatureAt(8191));
        Assert.Equal((float)(Math.Log(2.0) * 0.1 + 1.0), c.AttnTemperatureAt(8192), 6);
        Assert.Equal((float)(Math.Log(4.0) * 0.1 + 1.0), c.AttnTemperatureAt(3 * 8192 + 5), 6);
        Assert.Equal(1.0f, Cfg().AttnTemperatureAt(100000));
    }
}
