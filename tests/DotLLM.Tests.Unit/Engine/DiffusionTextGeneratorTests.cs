using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine;
using DotLLM.Engine.Samplers;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Tokenizers;
using Xunit;

namespace DotLLM.Tests.Unit.Engine;

/// <summary>
/// Unit tests for <see cref="DiffusionTextGenerator"/> — the iterative masked-diffusion decode loop
/// (issue #28, DiffusionGemma PR-6). All tests are CPU-only and run against either a fully-scripted
/// stub <see cref="IModel"/> (precise control over per-position logits → entropy) or the tiny
/// synthetic safetensors <see cref="TransformerModel"/> fixture (real cacheless hybrid forward).
/// No model downloads; real-weight end-to-end is issue #32.
/// </summary>
public sealed class DiffusionTextGeneratorTests : IDisposable
{
    private const int VocabSize = 16;
    private const int MaskTokenId = 15; // within [0, VocabSize)
    private const int EosTokenId = 0;

    private readonly string _scratch;

    public DiffusionTextGeneratorTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-diff-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    // ───────────────────────── End-to-end on the real synthetic model ─────────────────────────

    [Fact]
    public void Generate_RealSyntheticModel_RunsPrefillAndDenoiseAndReturnsTokens()
    {
        using var model = LoadSyntheticModel(seed: 42);
        var tok = new StubTokenizer();
        var diffusion = new DiffusionConfig
        {
            CanvasLength = 4,
            MaxDenoisingSteps = 6,
            MaskTokenId = MaskTokenId,
        };
        var gen = new DiffusionTextGenerator(model, tok, sampler: null, diffusion);

        var result = gen.Generate([1, 2, 3]);

        Assert.Equal(4, result.GeneratedTokenCount);
        Assert.Equal(4, result.GeneratedTokenIds.Length);
        Assert.Equal(1, result.CanvasCount);
        Assert.True(result.TotalDenoisingSteps >= 1);
        // No mask token may survive into the finished output.
        Assert.DoesNotContain(MaskTokenId, result.GeneratedTokenIds);
        Assert.False(string.IsNullOrEmpty(result.Text));
    }

    // ───────────────────────── Monotone non-increasing mask count ─────────────────────────

    [Fact]
    public void Generate_MaskCountIsMonotoneNonIncreasing_AcrossSteps()
    {
        // Scripted model: every position confidently prefers token 7. The scheduler's proportional
        // budget unmasks a subset each step; the canvas-snapshot mask counts must never increase.
        var model = new ScriptedModel(VocabSize, ConfidentLogitsFor(7));
        var tok = new StubTokenizer();
        var diffusion = new DiffusionConfig
        {
            CanvasLength = 8,
            MaxDenoisingSteps = 8,
            EntropyBound = 100f,        // generous → unmask whole budget each step
            ConfidenceThreshold = -1f,  // disable confidence early-stop for this test
            StabilityThreshold = 99,
            MaskTokenId = MaskTokenId,
        };
        var gen = new DiffusionTextGenerator(model, tok, new EntropyBoundSampler(), diffusion);

        var masks = new List<int>();
        gen.Generate([5, 5], onCanvasStep: s => { if (!s.Completed) masks.Add(s.MaskedCount); });

        Assert.NotEmpty(masks);
        Assert.Equal(0, masks[^1]); // ends fully unmasked
        for (int i = 1; i < masks.Count; i++)
            Assert.True(masks[i] <= masks[i - 1],
                $"mask count increased at step {i}: {masks[i - 1]} → {masks[i]}");
    }

    // ───────────────────────── Confidence early stop ─────────────────────────

    [Fact]
    public void Generate_ConfidentCanvas_StopsBeforeMaxSteps()
    {
        // A maximally-confident (near-zero entropy) canvas should trip the confidence early-stop
        // well before the max-step budget. Compare against an unbounded step budget.
        var model = new ScriptedModel(VocabSize, ConfidentLogitsFor(3));
        var tok = new StubTokenizer();
        var diffusion = new DiffusionConfig
        {
            CanvasLength = 4,
            MaxDenoisingSteps = 50,     // generous cap
            EntropyBound = 100f,        // commit the whole budget each step
            ConfidenceThreshold = 0.01f, // near-zero entropy trips this
            StabilityThreshold = 99,    // isolate confidence as the stop cause
            MaskTokenId = MaskTokenId,
        };
        var gen = new DiffusionTextGenerator(model, tok, new EntropyBoundSampler(), diffusion);

        var result = gen.Generate([9]);

        Assert.Single(result.CanvasStats);
        Assert.Equal(DenoiseStopResult.Confidence, result.CanvasStats[0].StopReason);
        Assert.True(result.TotalDenoisingSteps < 50,
            $"expected an early stop, but ran {result.TotalDenoisingSteps} steps.");
    }

    // ───────────────────────── Multi-canvas (block-autoregressive) ─────────────────────────

    [Fact]
    public void Generate_TargetLongerThanCanvas_ProducesMultipleCanvases()
    {
        var model = new ScriptedModel(VocabSize, ConfidentLogitsFor(4));
        var tok = new StubTokenizer();
        var diffusion = new DiffusionConfig
        {
            CanvasLength = 3,
            MaxDenoisingSteps = 6,
            EntropyBound = 100f,
            ConfidenceThreshold = -1f,
            StabilityThreshold = 99,
            MaskTokenId = MaskTokenId,
        };
        var gen = new DiffusionTextGenerator(model, tok, new EntropyBoundSampler(), diffusion);

        // target 7 > canvas 3 → ceil(7/3) = 3 canvases (3 + 3 + 1).
        var result = gen.Generate([1], targetLength: 7);

        Assert.True(result.CanvasCount >= 2, $"expected >= 2 canvases, got {result.CanvasCount}.");
        Assert.Equal(7, result.GeneratedTokenCount);
        Assert.Equal(FinishReason.Length, result.FinishReason);
    }

    // ───────────────────────── Streaming callback ─────────────────────────

    [Fact]
    public void Generate_StreamingCallback_ObservesDecreasingMaskCounts()
    {
        var model = new ScriptedModel(VocabSize, ConfidentLogitsFor(6));
        var tok = new StubTokenizer();
        var diffusion = new DiffusionConfig
        {
            CanvasLength = 6,
            MaxDenoisingSteps = 6,
            EntropyBound = 100f,
            ConfidenceThreshold = -1f,
            StabilityThreshold = 99,
            MaskTokenId = MaskTokenId,
        };
        var gen = new DiffusionTextGenerator(model, tok, new EntropyBoundSampler(), diffusion);

        var observed = new List<DiffusionCanvasState>();
        gen.Generate([2, 2], onCanvasStep: observed.Add);

        Assert.NotEmpty(observed);
        // First snapshot starts fully masked except the first committed step; final snapshot 0-masked.
        Assert.Equal(0, observed[^1].MaskedCount);
        // At least one intermediate snapshot has a strictly-positive mask count (decreasing toward 0).
        Assert.Contains(observed, s => s.MaskedCount > 0);
        // A completion snapshot is delivered.
        Assert.Contains(observed, s => s.Completed);
        // Canvas snapshots are defensive copies (independent arrays).
        Assert.True(observed.Count >= 2);
        Assert.False(ReferenceEquals(observed[0].Canvas, observed[1].Canvas));
    }

    // ───────────────────────── Cacheless hybrid forward contract ─────────────────────────

    [Fact]
    public void Generate_UsesCachelessHybridForward_NeverPassesKvCache()
    {
        var model = new ScriptedModel(VocabSize, ConfidentLogitsFor(1));
        var tok = new StubTokenizer();
        var diffusion = new DiffusionConfig
        {
            CanvasLength = 4,
            MaxDenoisingSteps = 4,
            EntropyBound = 100f,
            ConfidenceThreshold = -1f,
            StabilityThreshold = 99,
            MaskTokenId = MaskTokenId,
        };
        var gen = new DiffusionTextGenerator(model, tok, new EntropyBoundSampler(), diffusion);

        gen.Generate([1, 2, 3]);

        Assert.True(model.ForwardCount >= 1);
        Assert.True(model.AllForwardsWereCacheless, "a KV-cache was passed to the non-causal forward (would throw).");
        Assert.True(model.AllForwardsWereHybrid, "a non-hybrid mask spec reached the diffusion forward.");
        // prefixLen of the first canvas equals the prompt length (3).
        Assert.Equal(3, model.FirstHybridPrefixLen);
        // Working sequence length on the first step = promptLen + canvasLen = 3 + 4.
        Assert.Equal(7, model.FirstSeqLen);
    }

    // ───────────────────────── helpers ─────────────────────────

    /// <summary>Per-position logit row that confidently (near-deterministically) prefers <paramref name="token"/>.</summary>
    private static float[] ConfidentLogitsFor(int token)
    {
        var row = new float[VocabSize];
        for (int i = 0; i < VocabSize; i++) row[i] = 0f;
        row[token] = 40f; // huge margin → near-zero entropy after softmax
        return row;
    }

    private TransformerModel LoadSyntheticModel(int seed)
    {
        string path = Path.Combine(_scratch, $"diff-{seed}.safetensors");
        WriteFixture(path, seed);
        var sf = SafetensorsFile.Open(path);
        return TransformerModel.LoadFromSafetensors(sf, BuildSyntheticConfig());
    }

    private static ModelConfig BuildSyntheticConfig()
    {
        const int hidden = 16, layers = 2, heads = 2, headDim = 8, inter = 24;
        var rope = new RoPEConfig(Theta: 10000.0f, DimensionCount: headDim, Type: RoPEType.NeoX);
        return new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = VocabSize,
            HiddenSize = hidden,
            IntermediateSize = inter,
            NumLayers = layers,
            NumAttentionHeads = heads,
            NumKvHeads = heads,
            HeadDim = headDim,
            MaxSequenceLength = 64,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-6f,
            TiedEmbeddings = false,
        };
    }

    private static void WriteFixture(string path, int seed)
    {
        const int hidden = 16, layers = 2, heads = 2, headDim = 8, inter = 24;
        var b = new SafetensorsFixtureBuilder();
        int stride = heads * headDim;

        AddRand(b, "model.embed_tokens.weight", [VocabSize, hidden], 0.2f, seed + 0);
        AddRand(b, "model.norm.weight", [hidden], 1.0f, seed + 1);
        AddRand(b, "lm_head.weight", [VocabSize, hidden], 0.2f, seed + 2);

        for (int i = 0; i < layers; i++)
        {
            int s = seed + 10 * (i + 1);
            string p = $"model.layers.{i}";
            AddRand(b, $"{p}.input_layernorm.weight", [hidden], 1.0f, s + 0);
            AddRand(b, $"{p}.post_attention_layernorm.weight", [hidden], 1.0f, s + 1);
            AddRand(b, $"{p}.self_attn.q_proj.weight", [stride, hidden], 0.2f, s + 2);
            AddRand(b, $"{p}.self_attn.k_proj.weight", [stride, hidden], 0.2f, s + 3);
            AddRand(b, $"{p}.self_attn.v_proj.weight", [stride, hidden], 0.2f, s + 4);
            AddRand(b, $"{p}.self_attn.o_proj.weight", [hidden, stride], 0.2f, s + 5);
            AddRand(b, $"{p}.mlp.gate_proj.weight", [inter, hidden], 0.2f, s + 6);
            AddRand(b, $"{p}.mlp.up_proj.weight", [inter, hidden], 0.2f, s + 7);
            AddRand(b, $"{p}.mlp.down_proj.weight", [hidden, inter], 0.2f, s + 8);
        }
        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape, float amp, int seed)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        var v = new float[n];
        for (long i = 0; i < n; i++)
            v[i] = amp * MathF.Cos(0.61803398875f * (i + 1) + seed * 0.37f);
        b.AddFloat32(name, shape, v);
    }

    // ───────────────────────── stubs ─────────────────────────

    /// <summary>
    /// Deterministic stub model returning a fixed per-position logit row for every input position.
    /// Records how the diffusion loop invokes <c>Forward</c> so the cacheless / hybrid contract and
    /// sequence shape can be asserted.
    /// </summary>
    private sealed class ScriptedModel : IModel
    {
        private readonly float[] _rowLogits;

        public ScriptedModel(int vocab, float[] rowLogits)
        {
            if (rowLogits.Length != vocab) throw new ArgumentException("row length must equal vocab.");
            Config = new ModelConfig
            {
                Architecture = Architecture.Llama,
                VocabSize = vocab,
                HiddenSize = 8,
                IntermediateSize = 8,
                NumLayers = 1,
                NumAttentionHeads = 1,
                NumKvHeads = 1,
                HeadDim = 8,
                MaxSequenceLength = 256,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.None,
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-6f,
            };
            _rowLogits = rowLogits;
        }

        public ModelConfig Config { get; }
        public long ComputeMemoryBytes => 0;

        public int ForwardCount { get; private set; }
        public bool AllForwardsWereCacheless { get; private set; } = true;
        public bool AllForwardsWereHybrid { get; private set; } = true;
        public int FirstHybridPrefixLen { get; private set; } = -1;
        public int FirstSeqLen { get; private set; } = -1;

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId)
            => Build(tokenIds.Length);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId, IKvCache? kvCache)
            => Build(tokenIds.Length);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
            IKvCache? kvCache, ILoraAdapter? adapter)
            => Build(tokenIds.Length);

        public ITensor Forward(ReadOnlySpan<int> tokenIds, ReadOnlySpan<int> positions, int deviceId,
            IKvCache? kvCache, ILoraAdapter? adapter, AttentionMaskSpec maskSpec)
        {
            if (kvCache is not null) AllForwardsWereCacheless = false;
            if (maskSpec.Mode != AttentionMaskMode.Hybrid) AllForwardsWereHybrid = false;
            if (ForwardCount == 0)
            {
                FirstHybridPrefixLen = maskSpec.PrefixLength;
                FirstSeqLen = tokenIds.Length;
            }
            ForwardCount++;
            return Build(tokenIds.Length);
        }

        private unsafe ITensor Build(int rows)
        {
            // CPU-style [seqLen, vocab] all-position logits; every row is the scripted preference.
            var t = UnmanagedTensor.Allocate(new TensorShape(rows, Config.VocabSize), DType.Float32, -1);
            float* p = (float*)t.DataPointer;
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < Config.VocabSize; c++)
                    p[(long)r * Config.VocabSize + c] = _rowLogits[c];
            return t;
        }

        public void Dispose() { }
    }

    /// <summary>Trivial tokenizer: identity ids ↔ "t{id}" tokens. EOS = 0, mask within vocab.</summary>
    private sealed class StubTokenizer : ITokenizer
    {
        public int VocabSize => DiffusionTextGeneratorTests.VocabSize;
        public int BosTokenId => 1;
        public int EosTokenId => DiffusionTextGeneratorTests.EosTokenId;

        public int[] Encode(string text) => string.IsNullOrEmpty(text) ? [] : [1];
        public string Decode(ReadOnlySpan<int> tokenIds)
        {
            var sb = new System.Text.StringBuilder();
            foreach (int id in tokenIds) sb.Append('t').Append(id).Append(' ');
            return sb.ToString().TrimEnd();
        }
        public string DecodeToken(int tokenId) => $"t{tokenId}";
        public int CountTokens(string text) => Encode(text).Length;
    }
}
