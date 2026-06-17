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
/// End-to-end Gemma 3 forward-pass coverage. Exercises the four Gemma-family
/// mechanisms wired into <see cref="TransformerModel"/>:
/// <list type="bullet">
///   <item>Per-layer sliding-window dispatch (<see cref="ModelConfig.PerLayerSlidingWindow"/>) —
///         interleaved local / global pattern matching Gemma 3's
///         <c>sliding_window_pattern</c>.</item>
///   <item>Attention-logit soft-cap (<see cref="ModelConfig.AttnLogitSoftcap"/>) — Gemma 2 sets
///         this to 50.0; the parameter is wired regardless even though Gemma 3 leaves it null.</item>
///   <item>Final-logit soft-cap (<see cref="ModelConfig.FinalLogitSoftcap"/>) applied at
///         lm_head before sampling.</item>
///   <item>Query-pre-attn scalar override (<see cref="ModelConfig.QueryPreAttnScalar"/>) —
///         Gemma's alternative to the standard <c>1/sqrt(headDim)</c> scale.</item>
/// </list>
/// Uses a tiny synthetic safetensors fixture (HiddenSize=16, NumLayers=4,
/// NumHeads=2, VocabSize=8, IntermediateSize=24, sliding_window_pattern=2 so
/// layers 0 and 2 are sliding, layers 1 and 3 are full) following the
/// <see cref="TransformerModelMlaForwardTests"/> pattern.
/// </summary>
public sealed class TransformerModelGemma3ForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 4;
    private const int NumHeads = 2;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 8
    private const int IntermediateSize = 24;
    private const int SlidingWindow = 2;
    private const int SwPattern = 2;

    private readonly string _scratch;

    public TransformerModelGemma3ForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma3-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_Gemma3_AllMechanisms_FiniteLogits()
    {
        // The canonical Gemma 3 forward: per-layer sliding/full, attn soft-cap,
        // final soft-cap, and query-pre-attn-scalar all active simultaneously.
        // Asserts shape, finiteness, and non-degenerate variance — the standard
        // Gemma2/3 forward acceptance criterion.
        string path = Path.Combine(_scratch, "gemma3-all.safetensors");
        WriteFixture(path, seed: 42);

        ModelConfig config = BuildConfig(
            withAttnSoftcap: 50.0f,
            withFinalSoftcap: 30.0f,
            withQueryPreAttnScalar: HeadDim);

        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, config);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f, $"Logits degenerate: std={stats.StdDev}");

        // Final-logit soft-cap bounds every logit to (-cap, +cap). Verify the
        // mechanism actually fired by asserting the magnitude bound holds.
        Assert.True(stats.Min > -30.0f && stats.Max < 30.0f,
            $"Final-logit soft-cap did not clamp: min={stats.Min}, max={stats.Max}");
    }

    [Fact]
    public void Forward_Gemma3_FinalSoftcap_BoundsLogitMagnitude()
    {
        // Decisive evidence the final-logit soft-cap kernel runs: compare the
        // SAME forward pass with and without the cap. We run uncapped first and
        // derive the cap from the observed logit magnitude (cap = 0.5 × max|z|)
        // so the test is self-tuning and robust to fixture-magnitude changes —
        // the capped pass must then saturate every logit inside (-cap, +cap)
        // while the uncapped pass exceeds it.
        string path = Path.Combine(_scratch, "gemma3-finalcap.safetensors");
        WriteFixture(path, seed: 314, lmHeadAmplitude: 8.0f);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        float[] uncapped, capped;

        ModelConfig cfgUncapped = BuildConfig(
            withAttnSoftcap: null,
            withFinalSoftcap: null,
            withQueryPreAttnScalar: null);
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgUncapped))
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1))
        {
            uncapped = CopyLogits(l);
        }

        float maxAbs = 0f;
        foreach (float v in uncapped)
            if (MathF.Abs(v) > maxAbs) maxAbs = MathF.Abs(v);
        Assert.True(maxAbs > 1e-3f, $"Uncapped logits are degenerate (max|z|={maxAbs}).");
        float cap = 0.5f * maxAbs;

        ModelConfig cfgCapped = cfgUncapped with { FinalLogitSoftcap = cap };
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgCapped))
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1))
        {
            capped = CopyLogits(l);
        }

        // Capped: |z| < cap for every logit (z' = cap·tanh(z/cap) is strictly
        // bounded by cap).
        for (int i = 0; i < capped.Length; i++)
        {
            Assert.True(MathF.Abs(capped[i]) < cap,
                $"Capped logit[{i}]={capped[i]} should be inside ±{cap}");
        }

        // Uncapped: at least one logit exceeds the cap band (true by
        // construction, since cap = 0.5 × max|z|).
        bool anyExceeds = false;
        for (int i = 0; i < uncapped.Length; i++)
        {
            if (MathF.Abs(uncapped[i]) >= cap) { anyExceeds = true; break; }
        }
        Assert.True(anyExceeds,
            "Uncapped logits stayed inside the cap band — the soft-cap test is not discriminative.");
    }

    [Fact]
    public void Forward_Gemma3_PerLayerSlidingWindow_DiffersFromUniform()
    {
        // Discriminative: layers 0,2 are sliding (window=2) and layers 1,3
        // are full. Compare against a uniform-full configuration on the same
        // weights; the logits must differ on at least one element. A naive
        // mistake (e.g. ignoring PerLayerSlidingWindow and falling back to
        // SlidingWindowSize for every layer) would produce identical outputs
        // when SlidingWindowSize is null but PerLayerSlidingWindow imposes
        // a window — this test catches that.
        string path = Path.Combine(_scratch, "gemma3-perlayer.safetensors");
        WriteFixture(path, seed: 271);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        // Baseline: every layer full attention.
        ModelConfig cfgUniform = BuildConfig(
            withAttnSoftcap: null, withFinalSoftcap: null, withQueryPreAttnScalar: null);
        cfgUniform = cfgUniform with { PerLayerSlidingWindow = null, SlidingWindowSize = null };

        float[] uniformLogits;
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgUniform))
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1))
        {
            uniformLogits = CopyLogits(l);
        }

        // Gemma 3 pattern: layers 0,2 sliding (window=2); layers 1,3 full.
        ModelConfig cfgGemma3 = cfgUniform with
        {
            SlidingWindowSize = SlidingWindow,
            PerLayerSlidingWindow = new int?[NumLayers]
            {
                SlidingWindow, // layer 0 sliding
                null,          // layer 1 full
                SlidingWindow, // layer 2 sliding
                null,          // layer 3 full
            },
        };

        float[] gemma3Logits;
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgGemma3))
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1))
        {
            gemma3Logits = CopyLogits(l);
        }

        // Discrimination: at least one logit must differ by a non-trivial
        // amount. seqLen=5 > window=2 ensures the sliding layers actually
        // mask off-window keys.
        float maxDiff = 0f;
        for (int i = 0; i < uniformLogits.Length; i++)
        {
            float d = MathF.Abs(uniformLogits[i] - gemma3Logits[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Assert.True(maxDiff > 1e-4f,
            $"PerLayerSlidingWindow had no measurable effect (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_Gemma3_QueryPreAttnScalar_ChangesAttentionTemperature()
    {
        // Discriminative: changing QueryPreAttnScalar changes the attention
        // scale, which changes the softmax temperature, which changes the
        // logits. Two configurations differing only in QueryPreAttnScalar
        // (one default 1/sqrt(headDim), one with override) must produce
        // measurably different outputs.
        string path = Path.Combine(_scratch, "gemma3-qpas.safetensors");
        WriteFixture(path, seed: 7);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfgDefault = BuildConfig(
            withAttnSoftcap: null, withFinalSoftcap: null, withQueryPreAttnScalar: null);

        float[] defaultLogits;
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgDefault))
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1))
        {
            defaultLogits = CopyLogits(l);
        }

        // QueryPreAttnScalar = 4 * HeadDim => scale = 1/sqrt(4*headDim) which
        // is 0.5× the default 1/sqrt(headDim). Visibly different.
        ModelConfig cfgOverride = cfgDefault with { QueryPreAttnScalar = 4f * HeadDim };

        float[] overrideLogits;
        using (var sf = SafetensorsFile.Open(path))
        using (var model = TransformerModel.LoadFromSafetensors(sf, cfgOverride))
        using (ITensor l = model.Forward(tokenIds, positions, deviceId: -1))
        {
            overrideLogits = CopyLogits(l);
        }

        float maxDiff = 0f;
        for (int i = 0; i < defaultLogits.Length; i++)
        {
            float d = MathF.Abs(defaultLogits[i] - overrideLogits[i]);
            if (d > maxDiff) maxDiff = d;
        }
        Assert.True(maxDiff > 1e-4f,
            $"QueryPreAttnScalar override had no measurable effect (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_Gemma3_GeGLU_DiffersFromSwiGLU()
    {
        // Discriminative: the dense FFN must select FusedOps.GeGLUTanh when
        // ActivationFunction == GELUTanh and FusedOps.SwiGLU otherwise. Two
        // runs on the SAME Gemma fixture differing only in the activation
        // function must produce measurably different logits.
        // Large gate/up amplitude: GELU(tanh) and SiLU agree closely near zero
        // but diverge meaningfully once |gate| is O(1), so amplify the FFN input
        // to make the activation choice unambiguously discriminative.
        string path = Path.Combine(_scratch, "gemma3-geglu.safetensors");
        WriteFixture(path, seed: 99, gateUpAmplitude: 1.5f);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfgGeGLU = BuildConfig(
            withAttnSoftcap: null, withFinalSoftcap: null, withQueryPreAttnScalar: null,
            activation: ActivationFunction.GELUTanh);
        ModelConfig cfgSwiGLU = cfgGeGLU with { ActivationFunction = ActivationFunction.SiLU };

        float[] geglu = RunLogits(path, cfgGeGLU, tokenIds, positions);
        float[] swiglu = RunLogits(path, cfgSwiGLU, tokenIds, positions);

        float maxDiff = MaxAbsDiff(geglu, swiglu);
        Assert.True(maxDiff > 1e-4f,
            $"GeGLU vs SwiGLU produced no measurable difference (maxDiff={maxDiff}).");

        // Sanity: both finite.
        Assert.All(geglu, v => Assert.True(float.IsFinite(v)));
        Assert.All(swiglu, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Forward_Gemma3_EmbeddingScale_ChangesLogits()
    {
        // Discriminative: the sqrt(hidden_size) embedding scale must feed the
        // forward math. Same fixture, same config except EmbeddingScale =
        // sqrt(hidden) vs null — the logits must differ.
        string path = Path.Combine(_scratch, "gemma3-embscale.safetensors");
        WriteFixture(path, seed: 123);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfgNoScale = BuildConfig(
            withAttnSoftcap: null, withFinalSoftcap: null, withQueryPreAttnScalar: null,
            embeddingScale: null);
        ModelConfig cfgScaled = cfgNoScale with { EmbeddingScale = MathF.Sqrt(HiddenSize) };

        float[] noScale = RunLogits(path, cfgNoScale, tokenIds, positions);
        float[] scaled = RunLogits(path, cfgScaled, tokenIds, positions);

        float maxDiff = MaxAbsDiff(noScale, scaled);
        Assert.True(maxDiff > 1e-4f,
            $"Embedding scale had no measurable effect (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_Gemma3_FourNormLayout_DependsOnPostNorms()
    {
        // Discriminative: the post_attention_layernorm and post_feedforward_layernorm
        // weights (the two norms that the four-norm layout adds vs the legacy
        // two-norm Llama path) must feed the residual math. Two fixtures that
        // are byte-identical except for the post-attn / post-ffn norm weights
        // must produce different logits. (A two-norm implementation would ignore
        // those weights entirely and yield identical output.)
        string baseline = Path.Combine(_scratch, "gemma3-4norm-a.safetensors");
        string perturbed = Path.Combine(_scratch, "gemma3-4norm-b.safetensors");
        WriteFixture(baseline, seed: 555, postNormAmplitude: 0.02f);
        WriteFixture(perturbed, seed: 555, postNormAmplitude: 0.40f);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfg = BuildConfig(
            withAttnSoftcap: null, withFinalSoftcap: null, withQueryPreAttnScalar: null);

        float[] a = RunLogits(baseline, cfg, tokenIds, positions);
        float[] bb = RunLogits(perturbed, cfg, tokenIds, positions);

        float maxDiff = MaxAbsDiff(a, bb);
        Assert.True(maxDiff > 1e-4f,
            $"Post-attn / post-ffn norm weights had no measurable effect (maxDiff={maxDiff}). "
            + "The four-norm layout may have collapsed to the two-norm path.");
    }

    [Fact]
    public void Forward_Gemma3_WithQkNorm_FiniteAndUsesNormWeights()
    {
        // Per-head Q/K RMSNorm: when q_norm/k_norm tensors are present the loader
        // populates QNormWeight/KNormWeight and the existing apply path runs.
        // Compare a fixture WITH QK-norm tensors against one WITHOUT — the
        // outputs must differ (the norm weights change Q/K before RoPE).
        string withQk = Path.Combine(_scratch, "gemma3-qknorm-on.safetensors");
        string noQk = Path.Combine(_scratch, "gemma3-qknorm-off.safetensors");
        WriteFixture(withQk, seed: 808, writeQkNorm: true);
        WriteFixture(noQk, seed: 808, writeQkNorm: false);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfg = BuildConfig(
            withAttnSoftcap: null, withFinalSoftcap: null, withQueryPreAttnScalar: null);

        float[] on = RunLogits(withQk, cfg, tokenIds, positions);
        float[] off = RunLogits(noQk, cfg, tokenIds, positions);

        Assert.All(on, v => Assert.True(float.IsFinite(v)));
        float maxDiff = MaxAbsDiff(on, off);
        Assert.True(maxDiff > 1e-4f,
            $"Per-head Q/K RMSNorm had no measurable effect (maxDiff={maxDiff}).");
    }

    // ───────────────────────── helpers ─────────────────────────

    private static float[] RunLogits(string path, ModelConfig config, int[] tokenIds, int[] positions)
    {
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, config);
        using ITensor l = model.Forward(tokenIds, positions, deviceId: -1);
        return CopyLogits(l);
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float maxDiff = 0f;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++)
        {
            float d = MathF.Abs(a[i] - b[i]);
            if (d > maxDiff) maxDiff = d;
        }
        return maxDiff;
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig(
        float? withAttnSoftcap, float? withFinalSoftcap, float? withQueryPreAttnScalar,
        Architecture architecture = Architecture.Gemma3,
        ActivationFunction activation = ActivationFunction.GELUTanh,
        float? embeddingScale = null)
    {
        var rope = new RoPEConfig(
            Theta: 10000.0f,
            DimensionCount: HeadDim,
            Type: RoPEType.NeoX);

        // Per-layer pattern: layers 0 and 2 are sliding (window=2); layers 1
        // and 3 are full. Matches sliding_window_pattern=2.
        var perLayer = new int?[NumLayers]
        {
            SlidingWindow,
            null,
            SlidingWindow,
            null,
        };

        return new ModelConfig
        {
            Architecture = architecture,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            // Gemma uses GeGLU (tanh-approximate GELU) in the dense FFN; the
            // forward path selects FusedOps.GeGLUTanh when ActivationFunction ==
            // GELUTanh and FusedOps.SwiGLU otherwise. Discriminative coverage of
            // this branch lives in Forward_Gemma3_GeGLU_DiffersFromSwiGLU.
            ActivationFunction = activation,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = SlidingWindow,
            PerLayerSlidingWindow = perLayer,
            AttnLogitSoftcap = withAttnSoftcap,
            FinalLogitSoftcap = withFinalSoftcap,
            QueryPreAttnScalar = withQueryPreAttnScalar,
            EmbeddingScale = embeddingScale,
            MlaConfig = null,
            Moe = null,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Writes a synthetic Gemma 3 safetensors fixture: embed_tokens, lm_head,
    /// model.norm, and per-layer input_layernorm, self_attn.{q,k,v,o}_proj,
    /// the Gemma four-norm set (post_attention_layernorm, pre_feedforward_layernorm,
    /// post_feedforward_layernorm), and mlp.{gate,up,down}_proj — all HF row-major
    /// F32.
    /// <para>
    /// The norm weights are emitted as Gemma stores them (offsets from 1.0): the
    /// loader adds 1.0 at load (<c>(1+w)</c> absorption). The post-attention and
    /// post-feedforward norms are emitted with a non-trivial per-element ramp so a
    /// fixture written with <paramref name="postNormAmplitude"/> &gt; 0 produces an
    /// output that depends on those weights (the discriminator for the four-norm
    /// layout). When <paramref name="writeQkNorm"/> is true, per-head q_norm/k_norm
    /// tensors are emitted too.
    /// </para>
    /// </summary>
    private static void WriteFixture(string path, int seed, float lmHeadAmplitude = 0.1f,
                                     float postNormAmplitude = 0.10f, bool writeQkNorm = false,
                                     float gateUpAmplitude = 0.05f)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;     // = HiddenSize
        int kvStride = NumHeads * HeadDim;    // KV heads = NumHeads here

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        // Final norm is a Gemma (1+w) norm too — emit small offsets from 0.
        AddRand(b, "model.norm.weight", [HiddenSize], amplitude: 0.05f, seed: seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], lmHeadAmplitude, seed + 2);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";

            // Gemma RMSNorm weights are stored as offsets from 1.0; emit small
            // amplitudes (centred on 0) so the loader's +1.0 absorption lands
            // them near 1.0.
            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 0);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
                    amplitude: postNormAmplitude, seed: s + 1);
            AddRand(b, $"{prefix}.pre_feedforward_layernorm.weight", [HiddenSize],
                    amplitude: 0.05f, seed: s + 9);
            AddRand(b, $"{prefix}.post_feedforward_layernorm.weight", [HiddenSize],
                    amplitude: postNormAmplitude, seed: s + 10);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.1f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.1f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.1f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.1f, s + 5);

            if (writeQkNorm)
            {
                AddRand(b, $"{prefix}.self_attn.q_norm.weight", [HeadDim], amplitude: 0.05f, seed: s + 11);
                AddRand(b, $"{prefix}.self_attn.k_norm.weight", [HeadDim], amplitude: 0.05f, seed: s + 12);
            }

            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], gateUpAmplitude, s + 6);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], gateUpAmplitude, s + 7);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 8);
        }

        b.WriteTo(path);
    }

    /// <summary>
    /// Deterministic small-magnitude cos-based fill, shared with the MLA
    /// forward tests (<see cref="TransformerModelMlaForwardTests"/>).
    /// </summary>
    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed,
                                float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            values[i] = jitter > 0f ? center + jitter * cos : amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }

    private static unsafe LogitStats ComputeStats(ITensor logits)
    {
        int total = 1;
        for (int i = 0; i < logits.Shape.Rank; i++) total *= logits.Shape[i];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);

        int finite = 0;
        double sum = 0, sumSq = 0;
        float min = float.PositiveInfinity, max = float.NegativeInfinity;
        foreach (float v in span)
        {
            if (float.IsFinite(v))
            {
                finite++;
                sum += v;
                sumSq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
        }
        double mean = finite > 0 ? sum / finite : 0.0;
        double variance = finite > 0 ? (sumSq / finite) - (mean * mean) : 0.0;
        double stddev = Math.Sqrt(Math.Max(0.0, variance));
        return new LogitStats(total, finite, (float)mean, (float)stddev, min, max);
    }

    private readonly record struct LogitStats(
        int TotalCount, int FiniteCount, float Mean, float StdDev, float Min, float Max);
}
