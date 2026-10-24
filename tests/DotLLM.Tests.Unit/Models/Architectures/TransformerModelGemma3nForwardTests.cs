using System.Runtime.InteropServices;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using Xunit;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// End-to-end coverage for the Gemma-3n text tower (issue #136): Per-Layer
/// Embeddings (reused from the Gemma-4 dense tower — PLE originated in
/// Gemma 3n), AltUp (Alternating Updates — multi-stream predict/correct), the
/// Laurel block (Learned Augmented Residual Layer), and per-layer FFN
/// activation sparsity. A tiny synthetic safetensors fixture is loaded through
/// the full <see cref="TransformerModel.LoadFromSafetensors"/> path (loader +
/// forward validated together), mirroring
/// <c>TransformerModelGemma4E2BPleForwardTests</c>. Real gated-weight numerical
/// parity vs HF <c>transformers</c> is documented follow-up (no real Gemma-3n
/// weights available in this environment).
/// </summary>
/// <remarks>
/// Scope: <c>NumSharedKvLayers</c> is 0 here (every layer owns its KV) — the
/// shared-KV donor mechanism is architecture-agnostic
/// (<see cref="ModelConfig.SharedKvDonorLayer"/>) and already exercised by the
/// Gemma-4 E2B/E4B synthetic fixtures; this test's synthetic layer count (3) is
/// too small to reproduce the real donor pattern faithfully, so it is left
/// disabled to keep the fixture focused on the genuinely new Gemma-3n pieces.
/// </remarks>
public sealed class TransformerModelGemma3nForwardTests : IDisposable
{
    private const int HiddenSize = 16;
    private const int NumLayers = 3;
    private const int NumHeads = 4;
    private const int VocabSize = 8;
    private const int HeadDim = HiddenSize / NumHeads; // 4
    private const int NumKvHeads = 2;                  // GQA
    private const int IntermediateSize = 12;
    private const int PleDim = 6;                      // hidden_size_per_layer_input
    private const int SlidingWindow = 2;
    private const int AltUpNumInputs = 4;
    private const int LaurelRank = 4;

    private readonly string _scratch;

    public TransformerModelGemma3nForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-gemma3n-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [Fact]
    public void Forward_Gemma3n_LoadsAndForwards_FiniteNonDegenerate()
    {
        string path = Path.Combine(_scratch, "gemma3n.safetensors");
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
    public void Forward_Gemma3n_ChangesOutput_VsAltUpDisabledLikeGemma4()
    {
        // Discriminative: with AltUp/Laurel/PLE all active, the forward output must
        // differ materially from the same weights run through the plain Gemma-4
        // dense path (Architecture.Gemma4, Gemma3n config stripped) — which skips
        // AltUp/Laurel entirely and injects PLE the Gemma-4 way. This proves the
        // Gemma-3n branch is actually being exercised, not silently falling back to
        // the Gemma-4 path.
        string path = Path.Combine(_scratch, "gemma3n-disc.safetensors");
        WriteFixture(path, seed: 99);

        ModelConfig g3n = BuildConfig();
        ModelConfig g4 = g3n with { Architecture = Architecture.Gemma4, Gemma3n = null };

        float[] a = RunLogits(path, g3n);
        float[] b = RunLogits(path, g4);

        Assert.All(a, v => Assert.True(float.IsFinite(v)));
        Assert.All(b, v => Assert.True(float.IsFinite(v)));
        float maxDiff = MaxAbsDiff(a, b);
        Assert.True(maxDiff > 1e-3f,
            $"Gemma-3n AltUp/Laurel path had no measurable effect vs the plain Gemma-4 path (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_Gemma3n_ActivationSparsity_ChangesOutput()
    {
        // Discriminative: a non-zero activation_sparsity_pattern must materially
        // change the forward output vs all-zero (plain GeGLU, no gate).
        string path = Path.Combine(_scratch, "gemma3n-sparsity.safetensors");
        WriteFixture(path, seed: 7);

        ModelConfig sparse = BuildConfig();
        ModelConfig dense = sparse with
        {
            Gemma3n = sparse.Gemma3n! with
            {
                ActivationSparsityPattern = new float[NumLayers], // all-zero
            },
        };

        float[] a = RunLogits(path, sparse);
        float[] b = RunLogits(path, dense);

        Assert.All(a, v => Assert.True(float.IsFinite(v)));
        Assert.All(b, v => Assert.True(float.IsFinite(v)));
        float maxDiff = MaxAbsDiff(a, b);
        Assert.True(maxDiff > 1e-4f,
            $"Activation sparsity had no measurable effect on the forward output (maxDiff={maxDiff}).");
    }

    [Fact]
    public void KvCacheDecode_MatchesCachelessForward()
    {
        // Prefill-then-decode through SimpleKvCache must reproduce the cacheless
        // oracle. AltUp/Laurel/PLE/activation-sparsity are all per-token
        // operations with zero cross-token coupling (only attention mixes
        // tokens), so a decode call that reprocesses just the last token through
        // its own fresh AltUp stream stack — attending over the cached K/V from
        // the prefill call — must match the cacheless multi-token forward.
        string path = Path.Combine(_scratch, "gemma3n-kvcache.safetensors");
        WriteFixture(path, seed: 17);
        ModelConfig config = BuildConfig();

        int[] tokenIds = [0, 1, 2, 3, 4];
        int[] positions = [0, 1, 2, 3, 4];
        int last = tokenIds.Length - 1;

        using var sf = SafetensorsFile.Open(path);
        using var model = TransformerModel.LoadFromSafetensors(sf, config);

        float[] cachelessLast;
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1))
            cachelessLast = LastRow(logits, VocabSize);

        float[] decodeLast;
        using (var kv = new SimpleKvCache(KvGeometry.FromConfig(config), maxSeqLen: 16))
        {
            using (model.Forward(tokenIds.AsSpan(0, last), positions.AsSpan(0, last), deviceId: -1, kvCache: kv)) { }
            using ITensor logits = model.Forward(tokenIds.AsSpan(last, 1), positions.AsSpan(last, 1), deviceId: -1, kvCache: kv);
            decodeLast = LastRow(logits, VocabSize);
        }

        const float absTol = 1e-3f, relTol = 1e-3f;
        for (int c = 0; c < VocabSize; c++)
        {
            float a = cachelessLast[c], b = decodeLast[c];
            Assert.True(MathF.Abs(a - b) <= absTol + relTol * MathF.Abs(a),
                $"col {c}: cacheless={a:F6} vs decode={b:F6}");
        }
    }

    // ───────────────────────── helpers ─────────────────────────

    private static unsafe float[] LastRow(ITensor logits, int vocab)
    {
        int seqLen = logits.Shape[0];
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocab);
        return span.Slice((seqLen - 1) * vocab, vocab).ToArray();
    }

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
        var globalRope = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX);

        // Layers 0,1 sliding; layer 2 full attention.
        var perLayer = new int?[NumLayers] { SlidingWindow, SlidingWindow, null };

        var gemma3nConfig = new Gemma3nConfig
        {
            NumInputs = AltUpNumInputs,
            ActiveIdx = 0,
            CoefClip = 120.0f,
            CorrectOutputScale = true,
            LaurelRank = LaurelRank,
            // Sparsity on layer 0 only — exercises GaussianTopK without forcing
            // every layer's gate through it.
            ActivationSparsityPattern = new float[NumLayers] { 0.5f, 0f, 0f },
        };

        return new ModelConfig
        {
            Architecture = Architecture.Gemma3n,
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
            GlobalRoPEConfig = globalRope,
            ActivationFunction = ActivationFunction.GELUTanh,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
            SlidingWindowSize = SlidingWindow,
            PerLayerSlidingWindow = perLayer,
            FinalLogitSoftcap = 30.0f,
            EmbeddingScale = MathF.Sqrt(HiddenSize),
            PerLayerEmbedding = new PerLayerEmbeddingConfig { VocabSize = VocabSize, PerLayerDim = PleDim },
            Gemma3n = gemma3nConfig,
            ChatTemplate = null,
        };
    }

    /// <summary>
    /// Writes a synthetic Gemma-3n fixture: embed_tokens, lm_head, model.norm;
    /// per-layer Gemma four-norms + GQA self_attn.{q,k,v,o}_proj + per-head
    /// q_norm/k_norm + dense mlp.{gate,up,down}_proj; the PLE tables (identical
    /// shape to the Gemma-4 E2B/E4B fixture); AND the Gemma-3n-only tensors:
    /// model-level <c>altup_projections</c>/<c>altup_unembed_projections</c>, and
    /// per-layer <c>altup.*</c> + <c>laurel.*</c>. All HF row-major F32; norm
    /// weights emitted as Gemma stores them (offset from 1.0 — loader adds 1.0).
    /// </summary>
    private static void WriteFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();
        int qStride = NumHeads * HeadDim;
        int kvStride = NumKvHeads * HeadDim;
        int lp = NumLayers * PleDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.1f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 0.05f, seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.1f, seed + 2);

        // PLE model-level tables.
        AddRand(b, "model.embed_tokens_per_layer.weight", [VocabSize, lp], 0.1f, seed + 3);
        AddRand(b, "model.per_layer_model_projection.weight", [lp, HiddenSize], 0.1f, seed + 4);
        AddRand(b, "model.per_layer_projection_norm.weight", [PleDim], 0.05f, seed + 5);

        // Gemma-3n model-level AltUp stream projections (numInputs-1 pairs).
        for (int i = 0; i < AltUpNumInputs - 1; i++)
        {
            AddRand(b, $"model.altup_projections.{i}.weight", [HiddenSize, HiddenSize], 0.05f, seed + 6 + i);
            AddRand(b, $"model.altup_unembed_projections.{i}.weight", [HiddenSize, HiddenSize], 0.05f, seed + 9 + i);
        }

        for (int i = 0; i < NumLayers; i++)
        {
            int s = seed + 40 * (i + 1);
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

            // Gemma-3n per-layer AltUp weights.
            AddRand(b, $"{prefix}.altup.correction_coefs.weight", [AltUpNumInputs, AltUpNumInputs], 0.1f, s + 16);
            AddRand(b, $"{prefix}.altup.prediction_coefs.weight",
                [AltUpNumInputs * AltUpNumInputs, AltUpNumInputs], 0.1f, s + 17);
            AddRand(b, $"{prefix}.altup.modality_router.weight", [AltUpNumInputs, HiddenSize], 0.1f, s + 18);
            AddRand(b, $"{prefix}.altup.router_norm.weight", [HiddenSize], 0.05f, s + 19);
            AddRand(b, $"{prefix}.altup.correct_output_scale", [HiddenSize], 0.1f, s + 20);

            // Gemma-3n per-layer Laurel weights.
            AddRand(b, $"{prefix}.laurel.linear_left.weight", [LaurelRank, HiddenSize], 0.1f, s + 21);
            AddRand(b, $"{prefix}.laurel.linear_right.weight", [HiddenSize, LaurelRank], 0.1f, s + 22);
            AddRand(b, $"{prefix}.laurel.post_laurel_norm.weight", [HiddenSize], 0.10f, s + 23);
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

    [StructLayout(LayoutKind.Sequential)]
    private readonly record struct LogitStats(
        int TotalCount, int FiniteCount, float Mean, float StdDev, float Min, float Max);
}
