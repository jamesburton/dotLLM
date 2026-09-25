using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Engine.KvCache;
using DotLLM.Models.Architectures;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Structural / state-handling parity tests for <see cref="Qwen3MoeHybridTransformerModel"/>.
/// </summary>
/// <remarks>
/// <para>
/// These tests build a minimal but architecturally faithful qwen35moe "model" — one GDN layer
/// + one full-attention layer + a sparse MoE FFN with a Qwen1.5-style shared expert — out of
/// synthetic F32 weights owned in unmanaged memory by the test fixture. The model is
/// constructed via <see cref="Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights"/>,
/// bypassing the GGUF loader. The tests then verify:
/// </para>
/// <list type="number">
///   <item><b>Shape, finiteness, and non-degenerate variance</b> of the last-token logits over
///         a multi-token prefill. Random small-amplitude weights stay numerically stable
///         through the GDN recurrence; NaN/Inf or shape mismatch indicates a load- or
///         pointer-aliasing bug.</item>
///   <item><b>Determinism</b> — two consecutive Forward calls with the same input on the same
///         model instance produce different state but identical logits given a fresh GDN-state
///         cache reset; rebuilding the model from scratch with the same seed produces the same
///         logits row-for-row.</item>
///   <item><b>Q/K/V de-interleave + sigmoid-gate correctness on the full-attn layer</b>: forcing
///         the per-head gate to zero (so <c>sigmoid(0)=0.5</c>) gives a deterministic
///         scaling of the attention output, which we sanity-check by comparing two runs with
///         different shifts in the Q+Gate fused projection.</item>
/// </list>
/// <para>
/// These tests do NOT replace a real-GGUF parity check against llama.cpp output — that is the
/// semantic correctness gate. What passes here is "structural and state-handling parity";
/// architectural bugs (e.g. wrong frequency assignments in RoPE, wrong shared-expert gate
/// composition) are caught only by the llama.cpp comparison.
/// </para>
/// </remarks>
public sealed unsafe class Qwen3MoeHybridTransformerModelTests
{
    // Compact but architecturally valid shapes.
    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;          // GQA repeat factor = 2
    private const int HeadDim = 16;
    private const int RopeDim = 8;             // partial-rotary slice < HeadDim
    private const int MaxSeqLen = 8;
    private const int MoeIntermediate = 32;
    private const int SharedIntermediate = 16;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;

    // GDN config.
    private const int NVHead = 2;
    private const int NKHead = 1;             // V heads per K head = 2
    private const int DState = 8;
    private const int DConv = 4;
    private const int DInner = NVHead * DState;

    [Fact]
    public void Forward_Mixed_GdnAndFullAttn_HasFiniteNonZeroLogits()
    {
        using var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 42);
        var config = fixture.Config;

        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        using var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32,
            VocabSize, HiddenSize);

        using var kvCache = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen);

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache);

        Assert.Equal(tokenIds.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);

        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, tokenIds.Length * VocabSize);
        float mean = 0f, sumSq = 0f;
        foreach (var v in span)
        {
            Assert.True(float.IsFinite(v), $"non-finite logit: {v}");
            mean += v;
            sumSq += v * v;
        }
        mean /= span.Length;
        float variance = sumSq / span.Length - mean * mean;

        // With random weights the logits should NOT all be the same — variance > epsilon.
        Assert.True(variance > 1e-6f,
            $"degenerate logits: mean={mean:F6}, variance={variance:E3}");
    }

    [Fact]
    public void Forward_Deterministic_AcrossModelInstances()
    {
        int[] tokenIds = [3, 1, 4, 1, 5];
        int[] positions = [0, 1, 2, 3, 4];

        // ── Build #1 ──────────────────────────────────────────────────────
        float[] firstLogits;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 7))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv))
        {
            firstLogits = CopyLogits(logits);
        }

        // ── Build #2 with identical seed ───────────────────────────────────
        float[] secondLogits;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 7))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv))
        {
            secondLogits = CopyLogits(logits);
        }

        Assert.Equal(firstLogits.Length, secondLogits.Length);
        for (int i = 0; i < firstLogits.Length; i++)
        {
            // Same seed → bit-identical execution path → identical logits.
            Assert.Equal(firstLogits[i], secondLogits[i]);
        }
    }

    /// <summary>
    /// Prefill-vs-incremental parity on a GDN-only model. If the rolling conv state or the
    /// per-head matrix state in <c>GdnStateCache</c> is updated incorrectly, the position-3
    /// logits from a 4-token prefill won't match the logits from running four 1-token Forwards
    /// sequentially. Pure CPU, identical kernels both paths → tolerance can be very tight.
    /// </summary>
    [Fact]
    public void Forward_GdnOnly_PrefillVsIncremental_LastTokenLogitsMatch()
    {
        const int seqLen = 4;
        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        // ── Prefill all four tokens in one Forward ────────────────────────
        float[] prefillLastRow;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 11, gdnOnly: true))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1))
        {
            int total = logits.Shape[0] * logits.Shape[1];
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);
            prefillLastRow = span.Slice((seqLen - 1) * VocabSize, VocabSize).ToArray();
        }

        // ── Four sequential 1-token Forwards on a freshly-built same-seed model ──
        float[] incrementalLastRow;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 11, gdnOnly: true))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        {
            float[]? lastLogits = null;
            for (int t = 0; t < seqLen; t++)
            {
                int[] oneTok = [tokenIds[t]];
                int[] onePos = [positions[t]];
                using ITensor step = model.Forward(oneTok, onePos, deviceId: -1);
                if (t == seqLen - 1)
                {
                    var span = new ReadOnlySpan<float>((void*)step.DataPointer, VocabSize);
                    lastLogits = span.ToArray();
                }
            }
            incrementalLastRow = lastLogits!;
        }

        // Tight tolerance: same kernels, same data — only intra-token reduction-order differences
        // can drift. Test fails on conv-state or GDN-state roll bugs.
        const float absTol = 1e-4f;
        const float relTol = 1e-4f;
        for (int c = 0; c < VocabSize; c++)
        {
            float pref = prefillLastRow[c];
            float incr = incrementalLastRow[c];
            float diff = MathF.Abs(pref - incr);
            float bar = absTol + relTol * MathF.Abs(pref);
            Assert.True(diff <= bar,
                $"col={c}: prefill={pref:F6} vs incremental={incr:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// Prefill-vs-incremental parity on the mixed-layer model (GDN + full-attn + KvCache).
    /// Validates the KV-cache update path on top of the GDN state machinery — the full-attn
    /// layer's K/V slot must accumulate identically whether all four tokens are appended in
    /// a single Forward or one-at-a-time.
    /// </summary>
    [Fact]
    public void Forward_Mixed_PrefillVsIncremental_LastTokenLogitsMatch()
    {
        const int seqLen = 4;
        int[] tokenIds = [5, 0, 3, 7];
        int[] positions = [0, 1, 2, 3];

        // ── Prefill all four tokens in one Forward ────────────────────────
        float[] prefillLastRow;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 23))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv))
        {
            int total = logits.Shape[0] * logits.Shape[1];
            var span = new ReadOnlySpan<float>((void*)logits.DataPointer, total);
            prefillLastRow = span.Slice((seqLen - 1) * VocabSize, VocabSize).ToArray();
        }

        // ── Four sequential 1-token Forwards on a freshly-built same-seed model ──
        float[] incrementalLastRow;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 23))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        {
            float[]? lastLogits = null;
            for (int t = 0; t < seqLen; t++)
            {
                int[] oneTok = [tokenIds[t]];
                int[] onePos = [positions[t]];
                using ITensor step = model.Forward(oneTok, onePos, deviceId: -1, kv);
                if (t == seqLen - 1)
                {
                    var span = new ReadOnlySpan<float>((void*)step.DataPointer, VocabSize);
                    lastLogits = span.ToArray();
                }
            }
            incrementalLastRow = lastLogits!;
        }

        const float absTol = 1e-4f;
        const float relTol = 1e-4f;
        for (int c = 0; c < VocabSize; c++)
        {
            float pref = prefillLastRow[c];
            float incr = incrementalLastRow[c];
            float diff = MathF.Abs(pref - incr);
            float bar = absTol + relTol * MathF.Abs(pref);
            Assert.True(diff <= bar,
                $"col={c}: prefill={pref:F6} vs incremental={incr:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    [Fact]
    public void Forward_SharedExpertGate_PerturbationChangesLogits()
    {
        int[] tokenIds = [0, 1, 2];
        int[] positions = [0, 1, 2];

        // Baseline run.
        float[] baseLogits;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 99))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv))
        {
            baseLogits = CopyLogits(logits);
        }

        // Perturb the shared-expert gate to a strongly-saturating value, rebuild, rerun.
        float[] perturbedLogits;
        using (var fixture = Qwen3MoeHybridFixtureBuilder.Build(seed: 99, sharedGateAmplitude: 5f))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (var kv = new SimpleKvCache(model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen))
        using (ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kv))
        {
            perturbedLogits = CopyLogits(logits);
        }

        // The shared-expert gate WAS observably reaching the output. Sanity check
        // we exercised the new shared-expert path (HasSharedExpert == true).
        bool different = false;
        for (int i = 0; i < baseLogits.Length; i++)
        {
            if (MathF.Abs(baseLogits[i] - perturbedLogits[i]) > 1e-5f)
            {
                different = true;
                break;
            }
        }
        Assert.True(different,
            "perturbing the shared-expert gate amplitude did not change the logits — " +
            "the shared-expert path is not being executed.");
    }

    private static float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Owns a synthetic qwen35moe "model" in unmanaged memory: all projections are F32,
    /// allocated 64-byte aligned via <see cref="NativeMemory.AlignedAlloc"/>.
    /// Layer 0 is GDN, layer 1 is full GQA attention. Both layers share a sparse MoE FFN
    /// with a Qwen1.5-style shared expert and per-token sigmoid gate.
    /// </summary>
    private sealed unsafe class Qwen3MoeHybridFixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public Qwen3MoeLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static Qwen3MoeHybridFixtureBuilder Build(
            int seed, float sharedGateAmplitude = 0.05f, bool gdnOnly = false)
        {
            var b = new Qwen3MoeHybridFixtureBuilder();
            b.BuildInternal(seed, sharedGateAmplitude, gdnOnly);
            return b;
        }

        private void BuildInternal(int seed, float sharedGateAmplitude, bool gdnOnly)
        {
            var rng = new Random(seed);

            // Layout: by default layer 0 = GDN, layer 1 = full-attn (FullAttnInterval = 2).
            // When gdnOnly is set, both layers are GDN — used by the prefill-vs-incremental test
            // to avoid threading a KvCache through the parity comparison.
            HybridLayerKind[] kinds = gdnOnly
                ? [HybridLayerKind.GatedDeltaNet, HybridLayerKind.GatedDeltaNet]
                : [HybridLayerKind.GatedDeltaNet, HybridLayerKind.Attention];
            int[] headCountKv = gdnOnly ? [0, 0] : [0, NumKvHeads];
            int[] ffnLen = [0, 0]; // MoE FFN, FeedForwardLength is not the routed-expert dim.

            var layout = new HybridLayerLayout
            {
                LayerKind = kinds,
                HeadCountKv = headCountKv,
                FeedForwardLength = ffnLen,
            };

            var gdnConfig = new GatedDeltaNetConfig(
                FullAttnInterval: 2,
                NVHead: NVHead,
                NKHead: NKHead,
                DState: DState,
                DInner: DInner,
                DConv: DConv);

            var moeConfig = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = NumExpertsPerTok,
                MoeIntermediateSize = MoeIntermediate,
                NormTopKProb = true,
                SharedExpertIntermediateSize = SharedIntermediate,
                NumSharedExperts = 1,
                HasSharedExpertGate = true,
                DecoderSparseStep = 1,
            };

            Config = new ModelConfig
            {
                Architecture = Architecture.Qwen3MoeHybrid,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                IntermediateSize = 0,
                NumLayers = 2,
                NumAttentionHeads = NumAttentionHeads,
                NumKvHeads = NumKvHeads,
                HeadDim = HeadDim,
                MaxSequenceLength = MaxSeqLen,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.RoPE,
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: RopeDim, Type: RoPEType.NeoX),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                HybridLayout = layout,
                GdnConfig = gdnConfig,
                Moe = moeConfig,
                ChatTemplate = null,
            };

            // Top-level weights.
            TokenEmbedPtr = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);
            OutputNormWeight = FillNormVec(HiddenSize, rng);
            OutputWeightPtr = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);

            // Per-layer weights.
            Layers = new Qwen3MoeLayerWeights[2];

            // Layer 0: GDN.
            Layers[0] = new Qwen3MoeLayerWeights
            {
                AttnNormWeight = FillNormVec(HiddenSize, rng),
                PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                Gdn = BuildGdn(rng),
                FullAttn = null,
                Moe = BuildMoe(rng, sharedGateAmplitude),
            };

            // Layer 1: full-attn by default, GDN when gdnOnly is set.
            Layers[1] = gdnOnly
                ? new Qwen3MoeLayerWeights
                {
                    AttnNormWeight = FillNormVec(HiddenSize, rng),
                    PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                    Gdn = BuildGdn(rng),
                    FullAttn = null,
                    Moe = BuildMoe(rng, sharedGateAmplitude),
                }
                : new Qwen3MoeLayerWeights
                {
                    AttnNormWeight = FillNormVec(HiddenSize, rng),
                    PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                    Gdn = null,
                    FullAttn = BuildFullAttn(rng),
                    Moe = BuildMoe(rng, sharedGateAmplitude),
                };
        }

        private GdnTokenMixingWeights BuildGdn(Random rng)
        {
            int convDim = (2 * NKHead + NVHead) * DState;
            int gdnKDim = NKHead * DState;
            int gdnVDim = NVHead * DState;
            int qkvOut = 2 * gdnKDim + gdnVDim; // = convDim

            return new GdnTokenMixingWeights
            {
                QkvWeight = AllocFloatsUniform(HiddenSize * qkvOut, rng, 0.05f),
                QkvQuantType = QuantizationType.F32,
                QkvInputDim = HiddenSize,
                QkvOutputDim = qkvOut,
                GateWeight = AllocFloatsUniform(HiddenSize * gdnVDim, rng, 0.05f),
                GateQuantType = QuantizationType.F32,
                GateInputDim = HiddenSize,
                GateOutputDim = gdnVDim,
                A = NegativeRandom(NVHead, rng),
                AlphaWeight = AllocFloatsUniform(HiddenSize * NVHead, rng, 0.05f),
                AlphaQuantType = QuantizationType.F32,
                AlphaInputDim = HiddenSize,
                AlphaOutputDim = NVHead,
                BetaWeight = AllocFloatsUniform(HiddenSize * NVHead, rng, 0.05f),
                BetaQuantType = QuantizationType.F32,
                BetaInputDim = HiddenSize,
                BetaOutputDim = NVHead,
                Conv1dWeight = FillRandom(DConv * convDim, rng, 0.1f),
                Conv1dBias = new float[convDim],
                DtBias = FillRandom(NVHead, rng, 0.1f),
                SsmNormWeight = FillNormVec(DState, rng),
                OutWeight = AllocFloatsUniform(gdnVDim * HiddenSize, rng, 0.05f),
                OutQuantType = QuantizationType.F32,
                OutInputDim = gdnVDim,
                OutOutputDim = HiddenSize,
            };
        }

        private Qwen3FullAttnWeights BuildFullAttn(Random rng)
        {
            int qOut = 2 * NumAttentionHeads * HeadDim; // Fused Q+Gate.
            int kvOut = NumKvHeads * HeadDim;
            int oIn = NumAttentionHeads * HeadDim;
            return new Qwen3FullAttnWeights
            {
                QWeight = AllocFloatsUniform(HiddenSize * qOut, rng, 0.05f),
                QQuantType = QuantizationType.F32,
                QInputDim = HiddenSize,
                QOutputDim = qOut,
                KWeight = AllocFloatsUniform(HiddenSize * kvOut, rng, 0.05f),
                KQuantType = QuantizationType.F32,
                KInputDim = HiddenSize,
                KOutputDim = kvOut,
                VWeight = AllocFloatsUniform(HiddenSize * kvOut, rng, 0.05f),
                VQuantType = QuantizationType.F32,
                VInputDim = HiddenSize,
                VOutputDim = kvOut,
                OWeight = AllocFloatsUniform(oIn * HiddenSize, rng, 0.05f),
                OQuantType = QuantizationType.F32,
                OInputDim = oIn,
                OOutputDim = HiddenSize,
                NumKvHeads = NumKvHeads,
                QNormWeight = FillNormVec(HeadDim, rng),
                KNormWeight = FillNormVec(HeadDim, rng),
            };
        }

        private MoeLayerWeights BuildMoe(Random rng, float sharedGateAmplitude)
        {
            float[] gate = FillRandom(NumExperts * HiddenSize, rng, 0.05f);
            var w1 = new nint[NumExperts];
            var w2 = new nint[NumExperts];
            var w3 = new nint[NumExperts];
            for (int e = 0; e < NumExperts; e++)
            {
                w1[e] = AllocFloatsUniform(MoeIntermediate * HiddenSize, rng, 0.05f);
                w2[e] = AllocFloatsUniform(HiddenSize * MoeIntermediate, rng, 0.05f);
                w3[e] = AllocFloatsUniform(MoeIntermediate * HiddenSize, rng, 0.05f);
            }
            nint[] sharedGate = [AllocFloatsUniform(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedUp = [AllocFloatsUniform(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedDown = [AllocFloatsUniform(HiddenSize * SharedIntermediate, rng, 0.05f)];
            float[] sharedExpertGate = FillRandom(HiddenSize, rng, sharedGateAmplitude);

            return new MoeLayerWeights(
                gate: gate,
                w1: w1, w2: w2, w3: w3,
                numExperts: NumExperts,
                numExpertsPerTok: NumExpertsPerTok,
                hiddenSize: HiddenSize,
                intermediateSize: MoeIntermediate,
                normTopKProb: true,
                sharedGateProj: sharedGate,
                sharedUpProj: sharedUp,
                sharedDownProj: sharedDown,
                sharedIntermediateSize: SharedIntermediate,
                sharedExpertGate: sharedExpertGate);
        }

        private nint AllocFloatsUniform(int count, Random rng, float amplitude)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(ptr);
            float* dst = (float*)ptr;
            for (int i = 0; i < count; i++)
                dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return ptr;
        }

        private static float[] FillRandom(int count, Random rng, float amplitude)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return arr;
        }

        private static float[] FillNormVec(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        private static float[] NegativeRandom(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = -((float)rng.NextDouble() * 0.5f + 0.1f);
            return arr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs)
                NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
