using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Lora;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Models.Architectures;

/// <summary>
/// Regression coverage for the routed-MoE LoRA gap in <c>Qwen3MoeHybridTransformerModel</c>
/// (qwen35moe — Gated DeltaNet + sparse MoE hybrid): its <c>ForwardMoeBody</c> delegates
/// expert compute to the shared <c>MoeSwiGluMlp.ExecuteRoutedFromAssignments</c> kernel, which
/// already applies per-expert LoRA deltas via <c>ApplyLoraDelta(..., "mlp.experts.{j}.{proj}",
/// ...)</c> — but the call site used to hardcode <c>loraAdapter: null, loraLayer: 0</c>, and
/// this class had NO <see cref="ILoraAdapter"/> parameter anywhere in its public
/// <c>Forward</c> overload chain at all (unlike <c>TransformerModel</c>, which already had
/// <c>_currentAdapter</c> plumbed through). This adds the 5-arg
/// <c>Forward(..., IKvCache?, ILoraAdapter?)</c> overload and wires the real layer index /
/// adapter into the MoE call site.
/// </summary>
public sealed unsafe class Qwen3MoeHybridLoraTests
{
    private const int VocabSize = 8;
    private const int HiddenSize = 16;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 8;
    private const int RopeDim = 8;
    private const int MaxSeqLen = 8;
    private const int MoeIntermediate = 16;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;

    // Single-layer GDN-only model — the MoE FFN sub-layer runs on every layer regardless
    // of the token-mixing kind, so a 1-layer GDN model is sufficient to exercise the
    // routed-expert LoRA call site without needing a KV-cache / full-attn layer at all.
    private const int NVHead = 2;
    private const int NKHead = 1;
    private const int DState = 4;
    private const int DConv = 4;
    private const int DInner = NVHead * DState;

    [Fact]
    public void Forward_WithPerExpertMoeLoraAdapter_ChangesOutput()
    {
        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        // Two FRESH same-seed model instances — see the no-op test below for why reusing
        // one instance across two Forward calls would confound the comparison with GDN
        // recurrent-state drift instead of isolating the LoRA effect.
        float[] withoutAdapter;
        using (var fixture = Fixture.Build(seed: 5))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (ITensor baseline = model.Forward(tokenIds, positions, deviceId: -1, kvCache: null))
        {
            withoutAdapter = CopyLogits(baseline);
        }

        float[] withAdapter;
        using (var adapter = BuildPerExpertMoeAdapter(layer: 0, rank: 4, alpha: 8f))
        using (var fixture = Fixture.Build(seed: 5))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (ITensor adapted = model.Forward(tokenIds, positions, deviceId: -1, kvCache: null, adapter))
        {
            withAdapter = CopyLogits(adapted);
        }

        float maxDiff = MaxAbsDiff(withoutAdapter, withAdapter);
        Assert.True(maxDiff > 1e-5f,
            $"Per-expert LoRA adapter had no measurable effect on Qwen3MoeHybrid's routed-MoE branch (maxDiff={maxDiff}).");
    }

    [Fact]
    public void Forward_WithAdapterTargetingUnsupportedProjection_IsSafeNoOp()
    {
        int[] tokenIds = [0, 1, 2, 3];
        int[] positions = [0, 1, 2, 3];

        // Two FRESH same-seed model instances — NOT one model reused across two Forward
        // calls. Reusing one instance would advance its internally-owned GDN recurrent
        // state (_gdnCache) on the first call, so the second call's output would differ
        // for a state reason unrelated to LoRA, defeating the "byte-identical" comparison.
        float[] withoutAdapter;
        using (var fixture = Fixture.Build(seed: 5))
        using (var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            fixture.Config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize))
        using (ITensor baseline = model.Forward(tokenIds, positions, deviceId: -1, kvCache: null))
        {
            withoutAdapter = CopyLogits(baseline);
        }

        // An adapter that targets GDN-recurrent-mixing and full-attention projection names —
        // neither of which this class has a LoRA hook for. GetLayerWeights(layer, proj) never
        // matches "mlp.experts.{j}.{proj}", so this must be a silent no-op: no throw, and
        // byte-identical output to the no-adapter baseline.
        float[] withAdapter;
        using (var adapter = new LoraAdapter(
            "unsupported-proj-probe", rank: 4, alpha: 8f,
            targetModules: ["q_proj", "k_proj", "v_proj", "o_proj", "ssm_alpha_proj"]))
        {
            AddDenseEntry(adapter, layer: 0, proj: "q_proj",
                inputDim: HiddenSize, outputDim: NumAttentionHeads * HeadDim, rank: 4);
            AddDenseEntry(adapter, layer: 0, proj: "ssm_alpha_proj",
                inputDim: HiddenSize, outputDim: NVHead, rank: 4);

            using var fixture = Fixture.Build(seed: 5);
            using var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                fixture.Config, fixture.Layers, fixture.OutputNormWeight,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize);
            using ITensor adapted = model.Forward(tokenIds, positions, deviceId: -1, kvCache: null, adapter);
            withAdapter = CopyLogits(adapted);
        }

        Assert.Equal(withoutAdapter.Length, withAdapter.Length);
        for (int i = 0; i < withoutAdapter.Length; i++)
            Assert.Equal(withoutAdapter[i], withAdapter[i]);
    }

    private static LoraAdapter BuildPerExpertMoeAdapter(int layer, int rank, float alpha)
    {
        var adapter = new LoraAdapter(
            "per-expert-moe-probe", rank, alpha,
            targetModules: ["experts.gate_proj", "experts.up_proj", "experts.down_proj"]);

        var rng = new Random(0xC0FFEE);
        for (int e = 0; e < NumExperts; e++)
        {
            AddDenseEntry(adapter, layer, $"mlp.experts.{e}.gate_proj", HiddenSize, MoeIntermediate, rank, rng);
            AddDenseEntry(adapter, layer, $"mlp.experts.{e}.up_proj", HiddenSize, MoeIntermediate, rank, rng);
            AddDenseEntry(adapter, layer, $"mlp.experts.{e}.down_proj", MoeIntermediate, HiddenSize, rank, rng);
        }
        return adapter;
    }

    private static void AddDenseEntry(
        LoraAdapter adapter, int layer, string proj, int inputDim, int outputDim, int rank, Random? rng = null)
    {
        rng ??= new Random(1234);
        nint a = LoraAdapter.AllocAligned((long)rank * inputDim);
        nint b = LoraAdapter.AllocAligned((long)outputDim * rank);
        FillRandom((float*)a, rank * inputDim, rng, 0.05f);
        FillRandom((float*)b, outputDim * rank, rng, 0.05f);
        adapter.AddLayerWeights(layer, proj, new LoraLayerWeights(a, b, inputDim, outputDim));
    }

    private static void FillRandom(float* dst, int count, Random rng, float amplitude)
    {
        for (int i = 0; i < count; i++)
            dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
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

    private static float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        var copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Owns a synthetic single-layer GDN+MoE qwen35moe "model" in unmanaged memory.
    /// Deliberately smaller/simpler than <c>Qwen3MoeHybridTransformerModelTests</c>'s fixture —
    /// this test only needs the MoE call-site LoRA plumbing to be exercised, not the full
    /// GDN/full-attn/KV-cache structural surface.
    /// </summary>
    private sealed unsafe class Fixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public Qwen3MoeLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static Fixture Build(int seed)
        {
            var f = new Fixture();
            f.BuildInternal(seed);
            return f;
        }

        private void BuildInternal(int seed)
        {
            var rng = new Random(seed);

            var layout = new HybridLayerLayout
            {
                LayerKind = [HybridLayerKind.GatedDeltaNet],
                HeadCountKv = [0],
                FeedForwardLength = [0],
            };

            var gdnConfig = new GatedDeltaNetConfig(
                FullAttnInterval: 100, // no full-attn layer at NumLayers=1
                NVHead: NVHead, NKHead: NKHead, DState: DState, DInner: DInner, DConv: DConv);

            var moeConfig = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = NumExpertsPerTok,
                MoeIntermediateSize = MoeIntermediate,
                NormTopKProb = true,
                SharedExpertIntermediateSize = 0,
                NumSharedExperts = 0,
                HasSharedExpertGate = false,
                DecoderSparseStep = 1,
            };

            Config = new ModelConfig
            {
                Architecture = Architecture.Qwen3MoeHybrid,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                IntermediateSize = 0,
                NumLayers = 1,
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

            TokenEmbedPtr = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);
            OutputNormWeight = FillNormVec(HiddenSize, rng);
            OutputWeightPtr = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);

            Layers = new Qwen3MoeLayerWeights[1];
            Layers[0] = new Qwen3MoeLayerWeights
            {
                AttnNormWeight = FillNormVec(HiddenSize, rng),
                PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                Gdn = BuildGdn(rng),
                FullAttn = null,
                Moe = BuildMoe(rng),
            };
        }

        private GdnTokenMixingWeights BuildGdn(Random rng)
        {
            int convDim = (2 * NKHead + NVHead) * DState;
            int gdnKDim = NKHead * DState;
            int gdnVDim = NVHead * DState;
            int qkvOut = 2 * gdnKDim + gdnVDim;

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

        private MoeLayerWeights BuildMoe(Random rng)
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

            return new MoeLayerWeights(
                gate: gate,
                w1: w1, w2: w2, w3: w3,
                numExperts: NumExperts,
                numExpertsPerTok: NumExpertsPerTok,
                hiddenSize: HiddenSize,
                intermediateSize: MoeIntermediate,
                normTopKProb: true,
                sharedGateProj: Array.Empty<nint>(),
                sharedUpProj: Array.Empty<nint>(),
                sharedDownProj: Array.Empty<nint>(),
                sharedIntermediateSize: 0,
                sharedExpertGate: null);
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
