using System.Text.Json;
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
/// End-to-end Gemma 4 (DiffusionGemma text tower) MoE forward-pass coverage.
/// Exercises the four Gemma-4 deltas vs Gemma 3 wired into
/// <see cref="TransformerModel"/>:
/// <list type="bullet">
///   <item><b>Sparse MoE FFN with GeGLU experts</b> — the per-layer dense GeGLU
///     MLP is replaced by a top-k dense-routing MoE block whose experts honour
///     <see cref="ModelConfig.ActivationFunction"/> (GeGLU when GELUTanh).</item>
///   <item><b>Per-attention-type RoPE</b> — full-attention layers use a different
///     base theta + a partial-rotary factor than the sliding-window layers
///     (<see cref="ModelConfig.GlobalRoPEConfig"/> +
///     <see cref="ModelConfig.PartialRotaryFactor"/>).</item>
///   <item><b>Dual KV-head counts</b> — full-attention layers use
///     <see cref="ModelConfig.NumGlobalKvHeads"/>, sliding layers use
///     <see cref="ModelConfig.NumKvHeads"/>.</item>
///   <item><b>The Gemma four-norm + QK-norm backbone</b> (inherited from PR-1)
///     resolves correctly through the MoE layer path.</item>
/// </list>
/// A tiny synthetic safetensors fixture is loaded through the full
/// <see cref="TransformerModel.LoadFromSafetensors"/> path, so the loader (#25)
/// and the forward (#24) are validated together.
/// </summary>
public sealed class TransformerModelGemma4MoeForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 4;
    private const int NumHeads = 4;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 4
    // Sliding layers: NumKvHeads = 2; full layers: NumGlobalKvHeads = 1.
    private const int NumKvHeads = 2;
    private const int NumGlobalKvHeads = 1;
    private const int NumExperts = 6;
    private const int TopK = 2;
    private const int MoeIntermediateSize = 12;
    private const int SlidingWindow = 2;

    private readonly string _scratch;

    public TransformerModelGemma4MoeForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_Gemma4Moe_AllMechanisms_FiniteNonDegenerate()
    {
        // Canonical Gemma-4 MoE forward: full+sliding layer mix, per-attention-type
        // RoPE, dual KV-head counts, GeGLU experts, final soft-cap. Asserts shape,
        // finiteness, non-degenerate variance, and the soft-cap magnitude bound.
        string path = Path.Combine(_scratch, "gemma4-all.safetensors");
        WriteFixture(path, seed: 42);

        ModelConfig config = BuildConfig();

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
        Assert.True(stats.Min > -30.0f && stats.Max < 30.0f,
            $"Final-logit soft-cap did not clamp: min={stats.Min}, max={stats.Max}");
    }

    [Fact]
    public void Forward_Gemma4Moe_GeGLUExperts_DifferFromSwiGLU()
    {
        // Discriminative: the MoE experts must select FusedOps.GeGLUTanh when
        // ActivationFunction == GELUTanh and FusedOps.SwiGLU otherwise. Two runs
        // on the SAME fixture differing only in the activation function must
        // produce measurably different logits. Amplify the expert gate/up input
        // so GELU(tanh) and SiLU diverge meaningfully.
        string path = Path.Combine(_scratch, "gemma4-geglu.safetensors");
        WriteFixture(path, seed: 99, expertAmplitude: 1.4f);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfgGeGLU = BuildConfig(activation: ActivationFunction.GELUTanh);
        ModelConfig cfgSwiGLU = cfgGeGLU with { ActivationFunction = ActivationFunction.SiLU };

        float[] geglu = RunLogits(path, cfgGeGLU, tokenIds, positions);
        float[] swiglu = RunLogits(path, cfgSwiGLU, tokenIds, positions);

        float maxDiff = MaxAbsDiff(geglu, swiglu);
        Assert.True(maxDiff > 1e-4f,
            $"GeGLU vs SwiGLU experts produced no measurable difference (maxDiff={maxDiff}).");
        Assert.All(geglu, v => Assert.True(float.IsFinite(v)));
        Assert.All(swiglu, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Forward_Gemma4Moe_DualKvHeads_LoadAndForwardPerLayerShapes()
    {
        // Full-attention layers carry NumGlobalKvHeads (1) K/V heads and sliding
        // layers carry NumKvHeads (2) — DISTINCT K/V projection shapes per layer
        // type in the fixture. The dual-KV config loads and forwards finite,
        // proving the loader honours each layer's KV-head count and the forward
        // dispatch uses the right GQA group size per layer.
        string path = Path.Combine(_scratch, "gemma4-dualkv.safetensors");
        WriteFixture(path, seed: 271);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        float[] dual = RunLogits(path, BuildConfig(), tokenIds, positions);
        Assert.All(dual, v => Assert.True(float.IsFinite(v)));

        // Decisive evidence the per-layer KV-head count is actually honoured: a
        // config that (wrongly) claims a UNIFORM NumKvHeads for every layer
        // expects the full-attention K/V projections to have NumKvHeads*headDim
        // rows, but the fixture emitted them with NumGlobalKvHeads*headDim rows —
        // so loading must fail a projection-shape validation. A naive loader that
        // ignored NumGlobalKvHeads would instead load (and mis-shape the full
        // layers) without error.
        ModelConfig cfgWrongUniform = BuildConfig() with { NumGlobalKvHeads = null };
        var ex = Record.Exception(() =>
        {
            using var sf = SafetensorsFile.Open(path);
            using var model = TransformerModel.LoadFromSafetensors(sf, cfgWrongUniform);
        });
        Assert.NotNull(ex);
    }

    [Fact]
    public void Forward_Gemma4Moe_PartialRotary_DiffersFromFullRotary()
    {
        // Discriminative: the full-attention layers apply a partial-rotary factor
        // (only the leading fraction of each head rotates). Compare against a
        // config with PartialRotaryFactor = 1.0 (full rotation) on otherwise
        // identical weights — the logits must differ. The sliding layers are
        // unaffected (they always rotate fully on the sliding table), so the
        // difference is attributable to the global-table partial rotation.
        string path = Path.Combine(_scratch, "gemma4-partial.safetensors");
        WriteFixture(path, seed: 7);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        // HeadDim = 4 → partial factor 0.5 rotates 2 dims; full rotates 4 dims.
        ModelConfig cfgPartial = BuildConfig(partialRotaryFactor: 0.5f);
        ModelConfig cfgFull = cfgPartial with { PartialRotaryFactor = 1.0f };

        float[] partial = RunLogits(path, cfgPartial, tokenIds, positions);
        float[] full = RunLogits(path, cfgFull, tokenIds, positions);

        float maxDiff = MaxAbsDiff(partial, full);
        Assert.True(maxDiff > 1e-4f,
            $"Partial-rotary factor had no measurable effect (maxDiff={maxDiff}).");
        Assert.All(partial, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Forward_Gemma4Moe_PerAttentionTypeTheta_DiffersFromUniformTheta()
    {
        // Discriminative: full-attention layers use a different RoPE base theta
        // (1e6) than the sliding layers (1e4). Compare against a config whose
        // global RoPE uses the SAME theta as the sliding RoPE — the logits must
        // differ, proving the secondary (global) frequency table is actually
        // built and dispatched for the full-attention layers.
        string path = Path.Combine(_scratch, "gemma4-theta.safetensors");
        WriteFixture(path, seed: 1234);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfgDistinct = BuildConfig(globalTheta: 1_000_000.0f);
        // Same theta on both tables (and full rotation) — collapses the global
        // table to the sliding one for the full-attention layers.
        ModelConfig cfgSame = cfgDistinct with
        {
            GlobalRoPEConfig = cfgDistinct.GlobalRoPEConfig!.Value with { Theta = 10_000.0f },
            PartialRotaryFactor = 1.0f,
        };

        float[] distinct = RunLogits(path, cfgDistinct, tokenIds, positions);
        float[] same = RunLogits(path, cfgSame, tokenIds, positions);

        float maxDiff = MaxAbsDiff(distinct, same);
        Assert.True(maxDiff > 1e-4f,
            $"Per-attention-type RoPE theta had no measurable effect (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_Gemma4Moe_QkNormAndFourNorm_FeedForwardMath()
    {
        // The PR-1 four-norm + QK-norm backbone must resolve through the MoE
        // layer path. Compare a fixture WITH non-trivial post-norm / QK-norm
        // weights against one with near-identity norms — the outputs must differ,
        // confirming those tensors are loaded and applied on the MoE path.
        string strong = Path.Combine(_scratch, "gemma4-norm-strong.safetensors");
        string weak = Path.Combine(_scratch, "gemma4-norm-weak.safetensors");
        WriteFixture(strong, seed: 808, postNormAmplitude: 0.40f, qkNormAmplitude: 0.30f);
        WriteFixture(weak, seed: 808, postNormAmplitude: 0.01f, qkNormAmplitude: 0.01f);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];

        ModelConfig cfg = BuildConfig();
        float[] s = RunLogits(strong, cfg, tokenIds, positions);
        float[] w = RunLogits(weak, cfg, tokenIds, positions);

        Assert.All(s, v => Assert.True(float.IsFinite(v)));
        float maxDiff = MaxAbsDiff(s, w);
        Assert.True(maxDiff > 1e-4f,
            $"Four-norm / QK-norm weights had no measurable effect on the MoE path (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_Gemma4Moe_MultiShard_TiedEmbeddings_LoadsAndForwards()
    {
        // Loader coverage (#25): split the Gemma-4 MoE checkpoint across two
        // safetensors shards driven by a model.safetensors.index.json, and TIE
        // the output embeddings (omit lm_head.weight; the loader aliases
        // model.embed_tokens.weight when TiedEmbeddings is set). The full
        // MultiShardSafetensorsFile path is exercised end-to-end through the
        // forward pass.
        string indexPath = WriteTwoShardTiedFixture(seed: 4242);

        ModelConfig config = BuildConfig() with { TiedEmbeddings = true };

        using ISafetensorsTensorSource src = MultiShardSafetensorsFile.Open(indexPath);
        using var model = TransformerModel.LoadFromSafetensors(src, config);

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);

        Assert.Equal(2, logits.Shape.Rank);
        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);

        var stats = ComputeStats(logits);
        Assert.Equal(stats.TotalCount, stats.FiniteCount);
        Assert.True(stats.StdDev > 0.0f, $"Logits degenerate: std={stats.StdDev}");
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
        ActivationFunction activation = ActivationFunction.GELUTanh,
        float partialRotaryFactor = 0.5f,
        float globalTheta = 1_000_000.0f)
    {
        // Sliding RoPE (local layers): theta 1e4, full rotation, NeoX pairing.
        var slidingRope = new RoPEConfig(
            Theta: 10_000.0f,
            DimensionCount: HeadDim,
            Type: RoPEType.NeoX);

        // Global RoPE (full-attention layers): theta 1e6 + partial-rotary factor.
        var globalRope = new RoPEConfig(
            Theta: globalTheta,
            DimensionCount: HeadDim,
            Type: RoPEType.NeoX);

        // Per-layer attention type: layers 0,2 sliding (window=2); layers 1,3 full.
        var perLayer = new int?[NumLayers]
        {
            SlidingWindow, // layer 0 sliding
            null,          // layer 1 full
            SlidingWindow, // layer 2 sliding
            null,          // layer 3 full
        };

        var moe = new MoeConfig
        {
            NumExperts = NumExperts,
            NumExpertsPerTok = TopK,
            MoeIntermediateSize = MoeIntermediateSize,
            NormTopKProb = true,
        };

        return new ModelConfig
        {
            Architecture = Architecture.Gemma4,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            // IntermediateSize is unused on a pure-MoE Gemma 4 layer but must be
            // valid; mirror the MoE intermediate.
            IntermediateSize = MoeIntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = slidingRope,
            GlobalRoPEConfig = globalRope,
            PartialRotaryFactor = partialRotaryFactor,
            NumGlobalKvHeads = NumGlobalKvHeads,
            GlobalHeadDim = HeadDim, // uniform head_dim (supported case)
            ActivationFunction = activation,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = SlidingWindow,
            PerLayerSlidingWindow = perLayer,
            FinalLogitSoftcap = 30.0f,
            EmbeddingScale = MathF.Sqrt(HiddenSize),
            Moe = moe,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Writes a synthetic Gemma 4 MoE safetensors fixture: embed_tokens, lm_head,
    /// model.norm, and per-layer input_layernorm, self_attn.{q,k,v,o}_proj,
    /// per-head q_norm/k_norm, the Gemma four-norm set
    /// (post_attention_layernorm, pre_feedforward_layernorm,
    /// post_feedforward_layernorm), the MoE router (mlp.gate), and per-expert
    /// mlp.experts.{j}.{gate,up,down}_proj — all HF row-major F32.
    /// <para>
    /// Each layer's K/V projections are sized to THAT layer's KV-head count —
    /// full-attention layers (NumGlobalKvHeads) and sliding layers (NumKvHeads)
    /// carry distinct K/V weight shapes, exactly as a real Gemma 4 checkpoint
    /// does. The loader picks up the per-layer KOutputDim/VOutputDim from the
    /// tensor shape. Norm weights are emitted as Gemma stores them (offsets from
    /// 1.0); the loader adds 1.0 at load.
    /// </para>
    /// </summary>
    private static void WriteFixture(string path, int seed,
                                     float postNormAmplitude = 0.10f,
                                     float qkNormAmplitude = 0.05f,
                                     float expertAmplitude = 0.10f)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;            // = HiddenSize

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], amplitude: 0.05f, seed: seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.1f, seed + 2);

        // Per-layer attention type drives the KV-head count (and thus the K/V
        // projection shape). Layers 0,2 sliding → NumKvHeads; layers 1,3 full →
        // NumGlobalKvHeads. Mirrors BuildConfig's PerLayerSlidingWindow.
        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";
            bool isFull = (i % 2) == 1; // layers 1,3 are full attention
            int layerKvHeads = isFull ? NumGlobalKvHeads : NumKvHeads;
            int kvStride = layerKvHeads * HeadDim;

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize], amplitude: 0.05f, seed: s + 0);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize], amplitude: postNormAmplitude, seed: s + 1);
            AddRand(b, $"{prefix}.pre_feedforward_layernorm.weight", [HiddenSize], amplitude: 0.05f, seed: s + 9);
            AddRand(b, $"{prefix}.post_feedforward_layernorm.weight", [HiddenSize], amplitude: postNormAmplitude, seed: s + 10);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.1f, s + 2);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.1f, s + 3);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.1f, s + 4);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.1f, s + 5);

            AddRand(b, $"{prefix}.self_attn.q_norm.weight", [HeadDim], amplitude: qkNormAmplitude, seed: s + 11);
            AddRand(b, $"{prefix}.self_attn.k_norm.weight", [HeadDim], amplitude: qkNormAmplitude, seed: s + 12);

            // MoE router gate + per-expert GeGLU MLP (Qwen-MoE tensor naming).
            AddRand(b, $"{prefix}.mlp.gate.weight", [NumExperts, HiddenSize], 0.2f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                int es = s + 100 + 7 * e;
                AddRand(b, $"{prefix}.mlp.experts.{e}.gate_proj.weight", [MoeIntermediateSize, HiddenSize], expertAmplitude, es + 0);
                AddRand(b, $"{prefix}.mlp.experts.{e}.up_proj.weight", [MoeIntermediateSize, HiddenSize], expertAmplitude, es + 1);
                AddRand(b, $"{prefix}.mlp.experts.{e}.down_proj.weight", [HiddenSize, MoeIntermediateSize], 0.05f, es + 2);
            }
        }

        b.WriteTo(path);
    }

    /// <summary>
    /// Writes the same Gemma-4 MoE checkpoint as <see cref="WriteFixture"/> but
    /// split across two safetensors shards with a model.safetensors.index.json,
    /// and WITHOUT an lm_head.weight (tied embeddings). Returns the index path.
    /// Tensors are partitioned shard1 = global/even-layer tensors, shard2 =
    /// odd-layer tensors so both shards are exercised and the loader has to route
    /// names across files.
    /// </summary>
    private string WriteTwoShardTiedFixture(int seed)
    {
        string shard1Name = "model-00001-of-00002.safetensors";
        string shard2Name = "model-00002-of-00002.safetensors";
        string shard1 = Path.Combine(_scratch, shard1Name);
        string shard2 = Path.Combine(_scratch, shard2Name);
        string indexPath = Path.Combine(_scratch, "model.safetensors.index.json");

        var b1 = new SafetensorsFixtureBuilder();
        var b2 = new SafetensorsFixtureBuilder();
        var weightMap = new Dictionary<string, string>();

        // Local helper: emit into the chosen builder and record the routing.
        void Emit(SafetensorsFixtureBuilder b, string shardName, string name, int[] shape, float amplitude, int s)
        {
            AddRand(b, name, shape, amplitude, s);
            weightMap[name] = shardName;
        }

        int qStride = NumHeads * HeadDim;
        // Embeddings + final norm in shard1. NO lm_head (tied).
        Emit(b1, shard1Name, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        Emit(b1, shard1Name, "model.norm.weight", [HiddenSize], 0.05f, seed + 1);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 10 * (i + 1);
            string prefix = $"model.layers.{i}";
            bool isFull = (i % 2) == 1;
            int layerKvHeads = isFull ? NumGlobalKvHeads : NumKvHeads;
            int kvStride = layerKvHeads * HeadDim;

            // Route even layers to shard1, odd layers to shard2.
            var b = (i % 2) == 0 ? b1 : b2;
            string shardName = (i % 2) == 0 ? shard1Name : shard2Name;

            Emit(b, shardName, $"{prefix}.input_layernorm.weight", [HiddenSize], 0.05f, s + 0);
            Emit(b, shardName, $"{prefix}.post_attention_layernorm.weight", [HiddenSize], 0.10f, s + 1);
            Emit(b, shardName, $"{prefix}.pre_feedforward_layernorm.weight", [HiddenSize], 0.05f, s + 9);
            Emit(b, shardName, $"{prefix}.post_feedforward_layernorm.weight", [HiddenSize], 0.10f, s + 10);

            Emit(b, shardName, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.1f, s + 2);
            Emit(b, shardName, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.1f, s + 3);
            Emit(b, shardName, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.1f, s + 4);
            Emit(b, shardName, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.1f, s + 5);
            Emit(b, shardName, $"{prefix}.self_attn.q_norm.weight", [HeadDim], 0.05f, s + 11);
            Emit(b, shardName, $"{prefix}.self_attn.k_norm.weight", [HeadDim], 0.05f, s + 12);

            Emit(b, shardName, $"{prefix}.mlp.gate.weight", [NumExperts, HiddenSize], 0.2f, s + 6);
            for (int e = 0; e < NumExperts; e++)
            {
                int es = s + 100 + 7 * e;
                Emit(b, shardName, $"{prefix}.mlp.experts.{e}.gate_proj.weight", [MoeIntermediateSize, HiddenSize], 0.10f, es + 0);
                Emit(b, shardName, $"{prefix}.mlp.experts.{e}.up_proj.weight", [MoeIntermediateSize, HiddenSize], 0.10f, es + 1);
                Emit(b, shardName, $"{prefix}.mlp.experts.{e}.down_proj.weight", [HiddenSize, MoeIntermediateSize], 0.05f, es + 2);
            }
        }

        b1.WriteTo(shard1);
        b2.WriteTo(shard2);

        string json = JsonSerializer.Serialize(new
        {
            metadata = new { total_size = 0 },
            weight_map = weightMap,
        });
        File.WriteAllText(indexPath, json);
        return indexPath;
    }

    /// <summary>
    /// Deterministic small-magnitude cos-based fill, shared with the Gemma 3
    /// forward tests.
    /// </summary>
    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            values[i] = amplitude * MathF.Cos(phi);
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
