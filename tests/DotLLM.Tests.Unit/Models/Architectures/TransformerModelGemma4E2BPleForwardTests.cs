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
/// End-to-end coverage for the Gemma-4 <b>dense</b> text tower with Per-Layer
/// Embeddings (PLE) — the <c>gemma4_text</c> family (e.g. <c>google/gemma-4-E2B</c>).
/// Unlike the Gemma-4 26B-A4B MoE backbone, E2B is a dense GeGLU-tanh block; the one
/// novel piece is PLE: a second embedding table (<c>embed_tokens_per_layer</c>) plus
/// a context projection, combined once and injected as a gated residual into every
/// decoder layer's output. A tiny synthetic safetensors fixture is loaded through the
/// full <see cref="TransformerModel.LoadFromSafetensors"/> path (loader + forward
/// validated together). Real gated-weight numerical parity vs HF is a documented
/// remaining gate (see .planning/notes/gemma4-e2b-ple-matformer-design.md).
/// </summary>
public sealed class TransformerModelGemma4E2BPleForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 3;
    private const int NumHeads = 4;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 4
    private const int NumKvHeads = 1;                  // MQA (E2B)
    private const int IntermediateSize = 12;
    private const int PleDim = 6;                       // hidden_size_per_layer_input
    private const int SlidingWindow = 2;

    private readonly string _scratch;

    public TransformerModelGemma4E2BPleForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma4e2b-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_E2BPle_LoadsAndForwards_FiniteNonDegenerate()
    {
        string path = Path.Combine(_scratch, "e2b-ple.safetensors");
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
    public void Forward_E2BPle_ChangesOutput_VsNoPle()
    {
        // Discriminative: PLE must materially change the forward output. Load the
        // SAME fixture twice — once with PerLayerEmbedding set (loads the PLE tables
        // and injects per layer) and once with it null (the PLE tensors in the file
        // are simply ignored and no injection runs). The logits must differ.
        string path = Path.Combine(_scratch, "e2b-ple-disc.safetensors");
        WriteFixture(path, seed: 99);

        ModelConfig withPle = BuildConfig();
        ModelConfig noPle = withPle with { PerLayerEmbedding = null };

        float[] a = RunLogits(path, withPle);
        float[] b = RunLogits(path, noPle);

        Assert.All(a, v => Assert.True(float.IsFinite(v)));
        Assert.All(b, v => Assert.True(float.IsFinite(v)));
        float maxDiff = MaxAbsDiff(a, b);
        Assert.True(maxDiff > 1e-4f,
            $"PLE injection had no measurable effect on the forward output (maxDiff={maxDiff}).");
    }

    // ───────────────────────── helpers ─────────────────────────

    private static float[] RunLogits(string path, ModelConfig config)
    {
        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];
        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, config);
        using ITensor l = model.Forward(tokenIds, positions, deviceId: -1);
        return CopyLogits(l);
    }

    private static ModelConfig BuildConfig()
    {
        var rope = new RoPEConfig(Theta: 10_000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX);

        // Layers 0,1 sliding; layer 2 full attention (uniform KV/head-dim — the PLE
        // test isn't about the dual-RoPE/dual-KV path, which the MoE tests cover).
        var perLayer = new int?[NumLayers] { SlidingWindow, SlidingWindow, null };

        return new ModelConfig
        {
            Architecture = Architecture.Gemma4,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = 16,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.GELUTanh,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = SlidingWindow,
            PerLayerSlidingWindow = perLayer,
            FinalLogitSoftcap = 30.0f,
            EmbeddingScale = MathF.Sqrt(HiddenSize),
            PerLayerEmbedding = new PerLayerEmbeddingConfig { VocabSize = VocabSize, PerLayerDim = PleDim },
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Writes a synthetic Gemma-4 E2B dense fixture: embed_tokens, lm_head,
    /// model.norm; per-layer Gemma four-norms + MQA self_attn.{q,k,v,o}_proj +
    /// per-head q_norm/k_norm + dense mlp.{gate,up,down}_proj; and the PLE tables —
    /// model.embed_tokens_per_layer, model.per_layer_model_projection,
    /// model.per_layer_projection_norm plus per-layer per_layer_input_gate /
    /// per_layer_projection / post_per_layer_input_norm. All HF row-major F32; norm
    /// weights emitted as Gemma stores them (offset from 1.0 — loader adds 1.0).
    /// </summary>
    private static void WriteFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;   // = HiddenSize
        int kvStride = NumKvHeads * HeadDim; // MQA
        int lp = NumLayers * PleDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 0.05f, seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.1f, seed + 2);

        // PLE model-level tables.
        AddRand(b, "model.embed_tokens_per_layer.weight", [VocabSize, lp], 0.1f, seed + 3);
        AddRand(b, "model.per_layer_model_projection.weight", [lp, HiddenSize], 0.1f, seed + 4);
        AddRand(b, "model.per_layer_projection_norm.weight", [PleDim], 0.05f, seed + 5);

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 20 * (i + 1);
            string prefix = $"model.layers.{i}";

            AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize], 0.05f, s + 0);
            AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize], 0.10f, s + 1);
            AddRand(b, $"{prefix}.pre_feedforward_layernorm.weight", [HiddenSize], 0.05f, s + 2);
            AddRand(b, $"{prefix}.post_feedforward_layernorm.weight", [HiddenSize], 0.10f, s + 3);

            AddRand(b, $"{prefix}.self_attn.q_proj.weight", [qStride, HiddenSize], 0.1f, s + 4);
            AddRand(b, $"{prefix}.self_attn.k_proj.weight", [kvStride, HiddenSize], 0.1f, s + 5);
            AddRand(b, $"{prefix}.self_attn.v_proj.weight", [kvStride, HiddenSize], 0.1f, s + 6);
            AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, qStride], 0.1f, s + 7);
            AddRand(b, $"{prefix}.self_attn.q_norm.weight", [HeadDim], 0.05f, s + 8);
            AddRand(b, $"{prefix}.self_attn.k_norm.weight", [HeadDim], 0.05f, s + 9);

            // Dense GeGLU MLP.
            AddRand(b, $"{prefix}.mlp.gate_proj.weight", [IntermediateSize, HiddenSize], 0.1f, s + 10);
            AddRand(b, $"{prefix}.mlp.up_proj.weight", [IntermediateSize, HiddenSize], 0.1f, s + 11);
            AddRand(b, $"{prefix}.mlp.down_proj.weight", [HiddenSize, IntermediateSize], 0.05f, s + 12);

            // PLE per-layer weights.
            AddRand(b, $"{prefix}.per_layer_input_gate.weight", [PleDim, HiddenSize], 0.2f, s + 13);
            AddRand(b, $"{prefix}.per_layer_projection.weight", [HiddenSize, PleDim], 0.2f, s + 14);
            AddRand(b, $"{prefix}.post_per_layer_input_norm.weight", [HiddenSize], 0.10f, s + 15);
        }

        b.WriteTo(path);
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
