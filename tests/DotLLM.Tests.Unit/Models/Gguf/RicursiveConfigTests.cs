using System.Text.Json;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Cuda;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Gguf;

/// <summary>
/// Issue #1643 (S0 of Ripple-65B-Preview, #1642): <c>ricursive</c> config extraction (GGUF + HF), the tensor census with both community
/// value-residual names, BOS/EOG tokenizer wiring and the loud refusals. Synthetic fixture values are pairwise distinct and
/// non-degenerate (GQA 3, head_dim 16 != hidden/heads 8, window 8, two 4:1 periods, clamp 0.5, embed scale sqrt(48)); the real-file
/// values are covered by <see cref="RicursiveRealHeaderTests"/>. CPU-only: no device is created anywhere.
/// </summary>
public sealed class RicursiveConfigTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "dotllm-ricursive-" + Guid.NewGuid().ToString("N"));

    public RicursiveConfigTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best-effort cleanup */ }
    }

    private GgufFile OpenFile(SyntheticRicursiveOptions? o = null)
        => GgufFile.Open(SyntheticRicursiveGguf.Write(Path.Combine(_dir, Guid.NewGuid().ToString("N") + ".gguf"), o));

    private ModelConfig Extract(SyntheticRicursiveOptions? o = null)
    {
        using var f = OpenFile(o);
        return GgufModelConfigExtractor.Extract(f.Metadata);
    }

    private RicursiveCensusResult Census(SyntheticRicursiveOptions? o = null)
    {
        using var f = OpenFile(o);
        return RicursiveTensorCensus.Check(GgufModelConfigExtractor.Extract(f.Metadata), f);
    }

    // ───────────────────────── GGUF config ─────────────────────────

    [Fact]
    public void Gguf_Extracts_RicursiveShape()
    {
        var c = Extract();
        Assert.Equal(Architecture.Ricursive, c.Architecture);
        Assert.Equal(10, c.NumLayers);
        Assert.Equal(48, c.HiddenSize);
        Assert.Equal(40, c.IntermediateSize);
        Assert.Equal(6, c.NumAttentionHeads);
        Assert.Equal(2, c.NumKvHeads);
        Assert.Equal(16, c.HeadDim);
        Assert.Equal(32, c.VocabSize);
        Assert.Equal(64, c.MaxSequenceLength);
        Assert.Equal(1e-6f, c.NormEpsilon);
        Assert.Equal(AttentionType.GQA, c.AttentionType);
        Assert.Equal(ActivationFunction.SiLU, c.ActivationFunction);
        Assert.False(c.TiedEmbeddings);
        Assert.True(c.UseQkNorm);
        Assert.Equal(SyntheticRicursiveGguf.EmbedScale, c.EmbeddingScale);
        Assert.Equal(8, c.SlidingWindowSize);

        var rope = c.RoPEConfig!.Value;
        Assert.Equal(RoPEType.NeoX, rope.Type);
        Assert.Equal(16, rope.DimensionCount);   // full head: rotate_half over all 16 dims
        Assert.Equal(10000f, rope.Theta);
        Assert.Null(c.PartialRotaryFactor);

        var r = c.Ricursive!;
        Assert.Null(r.UseGatedAttention);        // GGUF has no key: the census decides from attn_gate
        Assert.Null(r.UseValueResidual);         // likewise from attn_vres_lambda / attn_value_res
        Assert.True(r.GateFullPrecision);
        Assert.True(r.FfnClampDeclared);
        Assert.All(r.FfnClamp, v => Assert.Equal(SyntheticRicursiveGguf.Clamp, v));
        Assert.Equal(10, r.FfnClamp.Count);
        Assert.False(r.EmbeddingScaleFolded);
    }

    public static IEnumerable<object[]> PatternSources() => [[true], [false]];

    [Theory]
    [MemberData(nameof(PatternSources))]
    public void Gguf_LayerPattern_SlidingRope_FullNoPe(bool writePattern)
    {
        // With the key (bool array) and without it (llama.cpp load_swa_pattern(ml, 5) default): full = layers 4 and 9 only.
        var c = Extract(new SyntheticRicursiveOptions { WritePattern = writePattern });
        for (int l = 0; l < c.NumLayers; l++)
        {
            bool full = l == 4 || l == 9;
            Assert.Equal(full ? null : 8, c.SlidingWindowOf(l));
            Assert.Equal(full, c.IsFullAttentionLayer(l));
            Assert.Equal(full, c.IsNoRopeLayer(l));
        }
        Assert.Equal([4, 9], c.NoRopeLayers);
    }

    [Fact]
    public void Gguf_ExplicitPattern_IsHonoured_NotTheDefault()
    {
        // A pattern that differs from the every-5th default must be read, not re-derived.
        bool[] p = [false, true, true, true, true, true, true, false, true, true];
        var c = Extract(new SyntheticRicursiveOptions { Pattern = p });
        Assert.Equal([0, 7], c.NoRopeLayers);
        Assert.Null(c.SlidingWindowOf(0));
        Assert.Equal(8, c.SlidingWindowOf(4));
    }

    [Fact]
    public void Gguf_CompactKvGeometry_WindowsOnlyTheSlidingSlots()
    {
        // The S1/S2 ring: sliding layers keep `window` rows, full (NoPE) layers the whole history.
        var c = Extract();
        var g = KvGeometry.FromConfig(c, compactSlidingWindows: true);
        Assert.True(g.HasSlidingWindows);
        for (int l = 0; l < c.NumLayers; l++)
            Assert.Equal(l % 5 == 4 ? 0 : 8, g.SlidingWindowOf(g.SlotOfLayer(l)));
    }

    [Fact]
    public void Gguf_RejectsBrokenPatterns()
    {
        Assert.Throws<InvalidDataException>(() => Extract(new SyntheticRicursiveOptions { Pattern = [true, true, true, false] }));    // short
        Assert.Throws<InvalidDataException>(() => Extract(new SyntheticRicursiveOptions { Pattern = new bool[10] }));                // no sliding layer
        bool[] other = SyntheticRicursiveGguf.DefaultPattern();
        other[2] = false;
        Assert.Throws<InvalidDataException>(() => Extract(new SyntheticRicursiveOptions { WriteRopePattern = true, RopePatternOverride = other }));
        // rope_pattern equal to the sliding pattern (PR #31692 files) is fine.
        Assert.Equal([4, 9], Extract(new SyntheticRicursiveOptions { WriteRopePattern = true }).NoRopeLayers);
    }

    [Fact]
    public void Gguf_SlidingWindowKey_IsRequired()
    {
        var ex = Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d.Remove("ricursive.attention.sliding_window")));
        Assert.Contains("sliding_window", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Gguf_ClampAbsent_IsTheExplicitNoClampVariant()
    {
        var r = Extract(new SyntheticRicursiveOptions { Clamp = SyntheticClampKey.Absent }).Ricursive!;
        Assert.False(r.FfnClampDeclared);
        Assert.All(r.FfnClamp, v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Gguf_ClampPerLayerArray_IsReadPerLayer()
    {
        var r = Extract(new SyntheticRicursiveOptions { Clamp = SyntheticClampKey.PerLayerArray }).Ricursive!;
        Assert.True(r.FfnClampDeclared);
        for (int l = 0; l < 10; l++)
            Assert.Equal(l == 3 ? 0f : 0.5f + 0.25f * (l % 2), r.FfnClampOf(l));
    }

    [Fact]
    public void Gguf_ClampAndEmbedScale_RejectInvalidValues()
    {
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["ricursive.swiglu_clamp"] = F32(-1f)));
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["ricursive.swiglu_clamp"] = F32(float.NaN)));
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["ricursive.embedding_scale"] = F32(0f)));
        Assert.Throws<InvalidDataException>(() => ExtractMutated(d => d["ricursive.attention.value_length"] = U32(8)));
    }

    [Fact]
    public void Gguf_EmbedScaleAbsent_MeansFolded_NotOne()
    {
        var c = Extract(new SyntheticRicursiveOptions { EmbeddingScale = null });
        Assert.Null(c.EmbeddingScale);
        Assert.True(c.Ricursive!.EmbeddingScaleFolded);
    }

    private static GgufMetadataValue U32(uint v) => new(GgufValueType.UInt32, v);
    private static GgufMetadataValue F32(float v) => new(GgufValueType.Float32, v);

    private ModelConfig ExtractMutated(Action<Dictionary<string, GgufMetadataValue>> mutate)
    {
        using var file = OpenFile();
        var dict = new Dictionary<string, GgufMetadataValue>(StringComparer.Ordinal);
        foreach (string key in file.Metadata.Keys)
        {
            file.Metadata.TryGetValue(key, out var v);
            dict[key] = v!;
        }
        mutate(dict);
        return GgufModelConfigExtractor.Extract(new GgufMetadata(dict));
    }

    // ───────────────────────── tensor census ─────────────────────────

    [Theory]
    [InlineData(RicursiveNames.VresLambda)]     // AtomicChat / llama.cpp PR #31692
    [InlineData(RicursiveNames.VresValueRes)]   // prometheusAIR
    public void Census_AcceptsBothValueResidualNames(string name)
    {
        var r = Census(new SyntheticRicursiveOptions { ValueResidualName = name });
        Assert.True(r.IsClean, r + "\n" + string.Join("\n", r.Missing.Concat(r.Unexpected).Concat(r.ShapeMismatches)));
        Assert.Equal(name, r.ValueResidualName);
        Assert.True(r.HasGate);
        Assert.True(r.HasOutputHead);
        Assert.Equal(3 + 10 * 15, r.TensorCount);   // real file: 1203 = 3 + 80 x 15
    }

    [Fact]
    public void Census_NoValueResidual_IsClean_AndReportsNone()
    {
        var r = Census(new SyntheticRicursiveOptions { ValueResidualName = null });
        Assert.True(r.IsClean, r.ToString());
        Assert.Null(r.ValueResidualName);
    }

    [Theory]
    [InlineData(RicursiveNames.VresLambda, 3)]
    [InlineData(RicursiveNames.VresValueRes, 7)]
    public void Census_RejectsAMixOfTheTwoNames(string name, int mixedLayer)
    {
        var r = Census(new SyntheticRicursiveOptions { ValueResidualName = name, MixedValueResidualLayer = mixedLayer });
        Assert.False(r.IsClean);
        string other = name == RicursiveNames.VresLambda ? RicursiveNames.VresValueRes : RicursiveNames.VresLambda;
        Assert.Contains($"blk.{mixedLayer}.{other}.weight", r.Unexpected);
        Assert.Contains($"blk.{mixedLayer}.{name}.weight", r.Missing);
    }

    [Fact]
    public void Census_RejectsUnknownValueResidualSpellings()
    {
        // Neither accepted name: the tensor is unknown, and with no accepted name on layer 0 there is no value residual at all.
        using var f = OpenFile(new SyntheticRicursiveOptions { ValueResidualName = null, ExtraTensor = "blk.0.attn_valres.weight" });
        var r = RicursiveTensorCensus.Check(GgufModelConfigExtractor.Extract(f.Metadata), f);
        Assert.False(r.IsClean);
        Assert.Contains("blk.0.attn_valres.weight", r.Unexpected);
    }

    [Fact]
    public void Census_ValueResidualOnSomeLayersOnly_IsMissing()
    {
        using var f = OpenFile();
        var listing = f.Tensors.Where(t => t.Name != "blk.6.attn_vres_lambda.weight").Select(t => (t.Name, t.Shape.Dimensions)).ToList();
        var r = RicursiveTensorCensus.Check(GgufModelConfigExtractor.Extract(f.Metadata), listing);
        Assert.Equal(["blk.6.attn_vres_lambda.weight"], r.Missing);
    }

    [Fact]
    public void Census_Detects_MissingAndMisShapedTensors()
    {
        foreach (string victim in new[] { "blk.9.post_ffw_norm.weight", "blk.4.attn_gate.weight", "blk.0.attn_k_norm.weight", "blk.2.ffn_up.weight" })
            Assert.Contains(victim, Census(new SyntheticRicursiveOptions { OmitTensor = victim }).Missing);
        foreach (string victim in new[] { "blk.1.attn_gate.weight", "blk.8.attn_value_res.weight", "token_embd.weight" })
        {
            var o = new SyntheticRicursiveOptions { WrongShapeTensor = victim, ValueResidualName = RicursiveNames.VresValueRes };
            Assert.Contains(Census(o).ShapeMismatches, m => m.StartsWith(victim + ":", StringComparison.Ordinal));
        }
    }

    [Fact]
    public void Census_GateIsAllOrNone()
    {
        Assert.True(Census(new SyntheticRicursiveOptions { WithGate = false }).IsClean);   // a gate-less model (use_gated_attention false)
        Assert.False(Census(new SyntheticRicursiveOptions { WithGate = false }).HasGate);
        using var f = OpenFile(new SyntheticRicursiveOptions { WithGate = false });
        var c = GgufModelConfigExtractor.Extract(f.Metadata);
        var declared = c with { Ricursive = c.Ricursive! with { UseGatedAttention = true } };   // HF says gated, file has none
        Assert.Contains("blk.0.attn_gate.weight", RicursiveTensorCensus.Check(declared, f).Missing);
    }

    [Fact]
    public void Census_OutputHeadIsOptional_TiedFallbackLikeLlamaCpp()
    {
        var r = Census(new SyntheticRicursiveOptions { WithOutputHead = false });
        Assert.True(r.IsClean, r.ToString());
        Assert.False(r.HasOutputHead);
    }

    [Fact]
    public void Census_Flags_ForeignTensors()
    {
        var r = Census(new SyntheticRicursiveOptions { ExtraTensor = "blk.3.ffn_gate_inp.weight" });
        Assert.Equal(["blk.3.ffn_gate_inp.weight"], r.Unexpected);
    }

    // ───────────────────────── tokenizer wiring ─────────────────────────

    [Theory]
    [InlineData(null)]    // AtomicChat: no add_bos_token key at all
    [InlineData(false)]   // prometheusAIR: explicit false
    public void Tokenizer_NeverAddsBos_AndStopsOnImEndAndEndOfText(bool? addBosKey)
    {
        using var f = OpenFile(new SyntheticRicursiveOptions { AddBosToken = addBosKey });
        Assert.False(GgufAddBosResolver.Resolve(f.Metadata));
        var tok = GgufBpeTokenizerFactory.Load(f.Metadata);
        Assert.False(tok.AddBosToken);
        Assert.Equal(SyntheticRicursiveGguf.ImEndId, tok.EosTokenId);
        Assert.Equal(SyntheticRicursiveGguf.EndOfTextId, tok.BosTokenId);
        Assert.Equal([SyntheticRicursiveGguf.EndOfTextId], tok.ExtraEndOfGenerationTokenIds);
        Assert.NotEqual(SyntheticRicursiveGguf.EndOfTextId, tok.Encode("abc").FirstOrDefault(-1));
    }

    [Fact]
    public void Tokenizer_BosDefault_IsArchScoped_EvenForALlama3PreType()
    {
        // A future ricursive conversion declaring a llama3-family pre-type must still not get BOS by default; an explicit key wins.
        using var f = OpenFile();
        var d = Meta(f);
        d["tokenizer.ggml.pre"] = new GgufMetadataValue(GgufValueType.String, "llama-bpe");
        Assert.False(GgufAddBosResolver.Resolve(new GgufMetadata(d)));
        d["tokenizer.ggml.add_bos_token"] = new GgufMetadataValue(GgufValueType.Bool, true);
        Assert.True(GgufAddBosResolver.Resolve(new GgufMetadata(d)));
        d["general.architecture"] = new GgufMetadataValue(GgufValueType.String, "llama");
        d.Remove("tokenizer.ggml.add_bos_token");
        Assert.True(GgufAddBosResolver.Resolve(new GgufMetadata(d)));   // control: the same metadata on llama does add BOS
    }

    [Fact]
    public void Tokenizer_EogWithoutEotKey_StillHasBothStops()
    {
        using var f = OpenFile(new SyntheticRicursiveOptions { WriteEot = false, AddBosToken = false });   // prometheusAIR shape
        var tok = GgufBpeTokenizerFactory.Load(f.Metadata);
        Assert.Equal(SyntheticRicursiveGguf.ImEndId, tok.EosTokenId);
        Assert.Equal([SyntheticRicursiveGguf.EndOfTextId], tok.ExtraEndOfGenerationTokenIds);
    }

    private static Dictionary<string, GgufMetadataValue> Meta(GgufFile f)
    {
        var d = new Dictionary<string, GgufMetadataValue>(StringComparer.Ordinal);
        foreach (string k in f.Metadata.Keys) { f.Metadata.TryGetValue(k, out var v); d[k] = v!; }
        return d;
    }

    // ───────────────────────── HF config ─────────────────────────

    private static string RealHfConfig =>
        File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "Models", "Gguf", "TestData", "ricursive-ripple-65b-preview-config.json"));

    [Fact]
    public void Hf_RealRipple65bConfig_Extracts()
    {
        // Verbatim ricursiveAI/Ripple-65B-Preview config.json (2026-11-05).
        var c = HfConfigExtractor.Extract(RealHfConfig);
        Assert.Equal(Architecture.Ricursive, c.Architecture);
        Assert.Equal(80, c.NumLayers);
        Assert.Equal(5120, c.HiddenSize);
        Assert.Equal(40960, c.IntermediateSize);
        Assert.Equal(80, c.NumAttentionHeads);
        Assert.Equal(8, c.NumKvHeads);
        Assert.Equal(128, c.HeadDim);
        Assert.Equal(100352, c.VocabSize);
        Assert.Equal(2097152, c.MaxSequenceLength);
        Assert.Equal(1e-6f, c.NormEpsilon);
        Assert.False(c.TiedEmbeddings);
        Assert.Equal(1024, c.SlidingWindowSize);
        Assert.Equal(71.55417527999327f, c.EmbeddingScale);
        Assert.Equal(RoPEType.NeoX, c.RoPEConfig!.Value.Type);
        Assert.Equal(128, c.RoPEConfig.Value.DimensionCount);
        Assert.Equal(Enumerable.Range(0, 16).Select(i => 5 * i + 4), c.NoRopeLayers);
        for (int l = 0; l < 80; l++)
            Assert.Equal(l % 5 == 4 ? null : 1024, c.SlidingWindowOf(l));
        var r = c.Ricursive!;
        Assert.True(r.UseGatedAttention);
        Assert.True(r.GateFullPrecision);
        Assert.True(r.UseValueResidual);
        Assert.True(r.FfnClampDeclared);
        Assert.All(r.FfnClamp, v => Assert.Equal(20f, v));
        Assert.False(r.EmbeddingScaleFolded);
    }

    [Fact]
    public void Hf_GgufAndHf_AgreeOnTheRealModel()
    {
        // Same model, two sources: the real AtomicChat header vs the real config.json.
        var hf = HfConfigExtractor.Extract(RealHfConfig);
        var gg = RicursiveRealHeaderTests.Load("atomicchat").Config;
        Assert.Equal(hf.NoRopeLayers, gg.NoRopeLayers);
        Assert.Equal(hf.PerLayerSlidingWindow, gg.PerLayerSlidingWindow);
        Assert.Equal(hf.EmbeddingScale!.Value, gg.EmbeddingScale!.Value, 1e-4f);
        Assert.Equal(hf.Ricursive!.FfnClamp, gg.Ricursive!.FfnClamp);
        Assert.Equal((hf.HiddenSize, hf.NumLayers, hf.NumAttentionHeads, hf.NumKvHeads, hf.HeadDim, hf.IntermediateSize, hf.VocabSize),
                     (gg.HiddenSize, gg.NumLayers, gg.NumAttentionHeads, gg.NumKvHeads, gg.HeadDim, gg.IntermediateSize, gg.VocabSize));
    }

    private static string Mutate(string json, Action<Dictionary<string, JsonElement>> edit)
    {
        var d = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(json)!;
        edit(d);
        return JsonSerializer.Serialize(d);
    }

    private static JsonElement J(string raw) => JsonDocument.Parse(raw).RootElement.Clone();

    [Fact]
    public void Hf_NullsAndDefaults_FollowTheVendorConfigClass()
    {
        string json = Mutate(RealHfConfig, d =>
        {
            d["embed_scale"] = J("null");
            d["ffn_clamp_threshold"] = J("null");
            d.Remove("use_valres");          // vendor default false
            d.Remove("layer_types");          // vendor default: full iff i % 5 == 4
            d.Remove("hybrid_layer_pattern");
        });
        var c = HfConfigExtractor.Extract(json);
        Assert.Null(c.EmbeddingScale);
        Assert.True(c.Ricursive!.EmbeddingScaleFolded);
        Assert.False(c.Ricursive.FfnClampDeclared);
        Assert.All(c.Ricursive.FfnClamp, v => Assert.Equal(0f, v));
        Assert.False(c.Ricursive.UseValueResidual);
        Assert.Equal(Enumerable.Range(0, 16).Select(i => 5 * i + 4), c.NoRopeLayers);
    }

    [Fact]
    public void Hf_LayerTypes_WinAndAreValidated()
    {
        string custom = Mutate(RealHfConfig, d =>
        {
            var types = Enumerable.Repeat("sliding_attention", 80).ToArray();
            types[0] = "full_attention";
            types[41] = "full_attention";
            d["layer_types"] = JsonSerializer.SerializeToElement(types);
            d.Remove("hybrid_layer_pattern");
        });
        Assert.Equal([0, 41], HfConfigExtractor.Extract(custom).NoRopeLayers);

        // A stored hybrid_layer_pattern that disagrees with layer_types is a corrupt config.
        string disagree = Mutate(RealHfConfig, d =>
        {
            var p = Enumerable.Range(0, 80).Select(i => i % 5 == 4 ? 0 : 1).ToArray();
            p[3] = 0;
            d["hybrid_layer_pattern"] = JsonSerializer.SerializeToElement(p);
        });
        Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(disagree));
        Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(Mutate(RealHfConfig, d => d["layer_types"] = J("[\"sliding_attention\"]"))));
        Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(Mutate(RealHfConfig, d => d["embed_scale"] = J("0"))));
        Assert.Throws<InvalidDataException>(() => HfConfigExtractor.Extract(Mutate(RealHfConfig, d => d["ffn_clamp_threshold"] = J("-1"))));
    }

    [Fact]
    public void Hf_SyntheticDirectory_RoundTripsAndPassesTheHfCensus()
    {
        string dir = SyntheticRicursiveGguf.WriteHfDirectory(Path.Combine(_dir, "hf"));
        var c = HfConfigExtractor.Extract(File.ReadAllText(Path.Combine(dir, "config.json")));
        var gg = Extract();
        Assert.Equal(gg.NoRopeLayers, c.NoRopeLayers);
        Assert.Equal(gg.PerLayerSlidingWindow, c.PerLayerSlidingWindow);
        Assert.Equal(gg.EmbeddingScale!.Value, c.EmbeddingScale!.Value, 1e-5f);
        Assert.Equal(gg.Ricursive!.FfnClamp, c.Ricursive!.FfnClamp);

        using var st = SafetensorsFile.Open(Path.Combine(dir, "model.safetensors"));
        var r = RicursiveTensorCensus.CheckHf(c, st.Tensors.Select(t => (t.Name, t.Shape)));
        Assert.True(r.IsClean, r + "\n" + string.Join("\n", r.Missing.Concat(r.Unexpected).Concat(r.ShapeMismatches)));
        Assert.Equal("self_attn.value_res_lambda", r.ValueResidualName);

        // use_valres false: the logits must be absent (the vendor model has no such parameter then).
        string noVres = SyntheticRicursiveGguf.WriteHfDirectory(Path.Combine(_dir, "hf-novres"), useValres: false);
        var c2 = HfConfigExtractor.Extract(File.ReadAllText(Path.Combine(noVres, "config.json")));
        using var st2 = SafetensorsFile.Open(Path.Combine(noVres, "model.safetensors"));
        Assert.True(RicursiveTensorCensus.CheckHf(c2, st2.Tensors.Select(t => (t.Name, t.Shape))).IsClean);
        // ...and a use_valres=true config over that directory reports every logit missing.
        Assert.Equal(10, RicursiveTensorCensus.CheckHf(c, st2.Tensors.Select(t => (t.Name, t.Shape))).Missing.Count);
    }

    // ───────────────────────── loud refusals ─────────────────────────

    [Fact]
    public void CpuDispatcher_RunsTheCensusThenRefuses()
    {
        using (var f = OpenFile())
        {
            var c = GgufModelConfigExtractor.Extract(f.Metadata);
            var ex = Assert.Throws<NotSupportedException>(() => ModelLoader.CreateCpuModelFromGguf(f, c));
            Assert.Contains("ricursive", ex.Message, StringComparison.Ordinal);
            Assert.Contains("backend: CPU", ex.Message, StringComparison.Ordinal);
        }
        using (var bad = OpenFile(new SyntheticRicursiveOptions { MixedValueResidualLayer = 5 }))
        {
            var c = GgufModelConfigExtractor.Extract(bad.Metadata);
            var ex = Assert.Throws<InvalidDataException>(() => ModelLoader.CreateCpuModelFromGguf(bad, c));
            Assert.Contains("blk.5.attn_value_res.weight", ex.Message, StringComparison.Ordinal);
        }
    }

    [Fact]
    public void GenericWeightLoader_And_GpuLoaders_Refuse()
    {
        using var f = OpenFile();
        var c = GgufModelConfigExtractor.Extract(f.Metadata);
        var w = Assert.Throws<NotSupportedException>(() => TransformerWeights.LoadFromGguf(f, c));
        Assert.Contains("ricursive", w.Message, StringComparison.Ordinal);
        var cuda = Assert.Throws<NotSupportedException>(() => CudaModelLoader.CreateFromGguf(f, c));
        Assert.Contains("backend: CUDA", cuda.Message, StringComparison.Ordinal);
        Assert.Throws<NotSupportedException>(() => CudaTransformerModel.RejectUnsupportedArchitecture(c));
        var vk = Assert.Throws<NotSupportedException>(() => VulkanTransformerModel.RejectUnsupportedArchitecture(c));
        Assert.Contains("ricursive", vk.Message, StringComparison.Ordinal);
        Assert.False(GpuOffloadPlanner.SupportsPartialOffload(Architecture.Ricursive));
        Assert.NotNull(TransformerWeights.DedicatedLoaderReason(c));
    }

    [Fact]
    public void SafetensorsDispatcher_Refuses()
    {
        string dir = SyntheticRicursiveGguf.WriteHfDirectory(Path.Combine(_dir, "hf-load"));
        var ex = Assert.Throws<NotSupportedException>(() => ModelLoader.LoadFromSafetensors(Path.Combine(dir, "model.safetensors")));
        Assert.Contains("ricursive", ex.Message, StringComparison.Ordinal);
    }
}
