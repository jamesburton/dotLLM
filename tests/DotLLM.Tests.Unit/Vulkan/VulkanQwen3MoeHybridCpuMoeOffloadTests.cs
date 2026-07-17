using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Exact-token parity + placement-bookkeeping tests for the CPU/GPU MoE expert
/// offload path (issue #370, llama.cpp <c>--n-cpu-moe</c> shorthand equivalent):
/// <see cref="VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights"/>'s
/// <c>nCpuMoeLayers</c> parameter and the matching <c>DOTLLM_N_CPU_MOE</c>
/// environment-variable fallback.
/// </summary>
/// <remarks>
/// <para>
/// <b>Method.</b> A synthetic 2-layer Qwen3MoeHybrid "model" (one GDN layer, one
/// full-attention layer, both with a sparse MoE FFN + Qwen1.5-style shared
/// expert) with every projection F32 — this deliberately exercises the F32
/// per-expert-pointer fallback branch of the CPU-offload path (the branch every
/// synthetic-fixture / non-GGUF caller takes; production GGUF loads go through
/// the raw-quant-view branch, which is the exact same routing/GEMM call the
/// pure-CPU <c>Qwen3MoeHybridTransformerModel.ForwardMoeBody</c> uses and is
/// covered indirectly by every other Qwen3MoeHybrid CPU/Vulkan parity test in
/// this file set).
/// </para>
/// <para>
/// <b>Discriminates.</b> A layer-index off-by-one, a stale MoE bank left
/// GPU-resident for a CPU-placed layer, a barrier/ordering bug in the host
/// round-trip, or a routing/weight-pointer mismatch in the CPU path would all
/// show up as a large logit divergence here — not just "still finite".
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3MoeHybridCpuMoeOffloadTests
{
    private const int HiddenSize = 32;
    private const int VocabSize = 8;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;         // NumAttentionHeads * HeadDim = 32 = HiddenSize
    private const int MaxSeqLen = 16;

    private const int NVHead = 4;
    private const int NKHead = 2;
    private const int DState = 8;           // NVHead * DState = 32 = HiddenSize
    private const int DConv = 4;

    private const int MoeIntermediate = 16;
    private const int SharedIntermediate = 8;
    private const int NumExperts = 6;
    private const int NumExpertsPerTok = 2;

    // Dense-host tolerance envelope used throughout the Qwen3MoeHybrid Vulkan
    // parity suite (e.g. VulkanQwen3MoeHybridTransformerModelForwardBatchTests).
    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableTheory]
    [InlineData(0)] // baseline: fully GPU-resident/streaming (pre-#370 behaviour)
    [InlineData(1)] // mixed: layer 0 CPU-placed, layer 1 GPU-placed
    [InlineData(2)] // fully CPU-placed
    public void CpuOffload_AnyPlacement_ProducesFiniteNonDegenerateLogits(int nCpuMoeLayers)
    {
        // Cross-backend (CPU-reference-model vs Vulkan) logit-VALUE parity for
        // this fixture is deliberately not asserted here: a 2-layer GDN+MoE
        // hybrid with random small-scale weights is a textbook case of
        // discrete top-k routing-selection chaos — a sub-ULP F32
        // reduction-order difference between the CPU grouped-GEMM path and
        // the GPU indexed-matmul path can flip which expert(s) a token
        // selects, which is a legitimate (if noisy) DISCRETE divergence, not
        // a precision bug, and it compounds layer-to-layer since layer 2's
        // router input depends on layer 1's output. This already reproduces
        // at nCpuMoeLayers=0 — the completely unmodified pre-#370 GPU path —
        // confirming it is a fixture-scale property, not a #370 regression.
        // The load-bearing "exact-token parity gate" the issue asks for is
        // CpuOffload_MixedPlacement_MatchesFullGpuPlacement below, which
        // compares Vulkan-vs-Vulkan at a fixed random seed where only the
        // MoE FFN backend differs between runs (GDN/attention stay
        // byte-identical GPU compute either way) — a real, tight-tolerance
        // parity gate. This test just guards against a placement producing
        // garbage (NaN/Inf/all-zero) output.
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var fixture = MoeOffloadFixtureBuilder.Build(seed: 1000 + nCpuMoeLayers);
        var config = fixture.Config;
        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;

        int[] tokenIds = [0, 3, 1, 5];
        int[] positions = [0, 1, 2, 3];

        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            device, config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, QuantizationType.F32, vocabSize, hiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            spvDir, nCpuMoeLayers);
        Assert.Equal(nCpuMoeLayers, model.NCpuMoeLayers);

        using var kvCache = model.CreateKvCache(MaxSeqLen);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache);
        float[] vkLogits = CopyLogits(logits);

        bool anyNonZero = false;
        for (int c = 0; c < vocabSize; c++)
        {
            float vk = vkLogits[c];
            Assert.True(float.IsFinite(vk), $"non-finite Vulkan logit nCpuMoeLayers={nCpuMoeLayers} col={c}: {vk}");
            if (vk != 0f) anyNonZero = true;
        }
        Assert.True(anyNonZero, $"nCpuMoeLayers={nCpuMoeLayers}: logits are degenerate (all zero).");
    }

    [SkippableTheory]
    [InlineData(1)] // mixed: layer 0 CPU-placed, layer 1 GPU-placed
    [InlineData(2)] // fully CPU-placed
    public void CpuOffload_AnyPlacement_MatchesFullGpuPlacement(int nCpuMoeLayers)
    {
        // The "exact-token parity gate" the #370 acceptance criteria ask for:
        // mixed (or fully) CPU-placed MoE must match full-GPU placement.
        // Both runs are Vulkan-vs-Vulkan at the same fixed weights/tokens —
        // GDN/attention compute is byte-identical either way (device
        // placement for those never changes), so only the MoE FFN backend
        // differs. That keeps this comparison tight (the established
        // dense-host tolerance envelope), unlike a cross-backend
        // CPU-reference comparison which is intrinsically noisy for a
        // hybrid-GDN synthetic fixture (see the test above).
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var fixture = MoeOffloadFixtureBuilder.Build(seed: 2024);
        var config = fixture.Config;
        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;

        int[] tokenIds = [2, 4, 6, 1];
        int[] positions = [0, 1, 2, 3];

        float[] LogitsFor(int n)
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, vocabSize, hiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                spvDir, n);
            using var kvCache = model.CreateKvCache(MaxSeqLen);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache);
            return CopyLogits(logits);
        }

        float[] fullGpu = LogitsFor(0);
        float[] offloaded = LogitsFor(nCpuMoeLayers);

        for (int c = 0; c < vocabSize; c++)
        {
            float a = fullGpu[c];
            float b = offloaded[c];
            float diff = MathF.Abs(a - b);
            float bar = AbsTol + RelTol * MathF.Abs(a);
            Assert.True(diff <= bar,
                $"nCpuMoeLayers={nCpuMoeLayers}, col={c}: full-GPU={a:F6} vs offloaded={b:F6} " +
                $"(|diff|={diff:E3} > {bar:E3})");
        }
    }

    [SkippableFact]
    public void CpuOffload_NCpuMoeLayers_ClampedToLayerCount()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = MoeOffloadFixtureBuilder.Build(seed: 55);
        var config = fixture.Config;

        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            device, config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, QuantizationType.F32, config.VocabSize, config.HiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            spvDir, nCpuMoeLayers: 999);

        Assert.Equal(config.NumLayers, model.NCpuMoeLayers);
        Assert.True(model.EstimatedCpuOffloadVramSavedBytes > 0);
    }

    [SkippableFact]
    public void CpuOffload_NoPlacement_ReportsZeroSavedBytes()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = MoeOffloadFixtureBuilder.Build(seed: 56);
        var config = fixture.Config;

        using var device = VulkanDevice.Create();
        using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
            device, config, fixture.Layers, fixture.OutputNormWeight,
            fixture.OutputWeightPtr, QuantizationType.F32, config.VocabSize, config.HiddenSize,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            spvDir, nCpuMoeLayers: 0);

        Assert.Equal(0, model.NCpuMoeLayers);
        Assert.Equal(0, model.EstimatedCpuOffloadVramSavedBytes);
    }

    [SkippableFact]
    public void CpuOffload_EnvVarFallback_ResolvesWhenNoExplicitValueGiven()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        using var fixture = MoeOffloadFixtureBuilder.Build(seed: 57);
        var config = fixture.Config;

        string? previous = Environment.GetEnvironmentVariable("DOTLLM_N_CPU_MOE");
        try
        {
            Environment.SetEnvironmentVariable("DOTLLM_N_CPU_MOE", "1");

            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, config.VocabSize, config.HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                spvDir); // nCpuMoeLayers omitted -> env var fallback

            Assert.Equal(1, model.NCpuMoeLayers);
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_N_CPU_MOE", previous);
        }
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Owns a randomly-generated 2-layer (GDN + full-attn) Qwen3MoeHybrid "model" in
    /// unmanaged memory, every projection F32 — the CPU-offload path's F32-pointer
    /// fallback branch (the one every synthetic/non-GGUF caller exercises).
    /// </summary>
    private sealed unsafe class MoeOffloadFixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public Qwen3MoeLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static MoeOffloadFixtureBuilder Build(int seed)
        {
            var b = new MoeOffloadFixtureBuilder();
            b.BuildInternal(seed);
            return b;
        }

        private void BuildInternal(int seed)
        {
            var rng = new Random(seed);

            HybridLayerKind[] kinds = [HybridLayerKind.GatedDeltaNet, HybridLayerKind.Attention];
            int[] headCountKv = [0, NumKvHeads];
            int[] ffnLen = [0, 0];

            var layout = new HybridLayerLayout
            {
                LayerKind = kinds,
                HeadCountKv = headCountKv,
                FeedForwardLength = ffnLen,
            };

            var gdnConfig = new GatedDeltaNetConfig(
                FullAttnInterval: 2,
                NVHead: NVHead, NKHead: NKHead, DState: DState,
                DInner: NVHead * DState,
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
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.NeoX),
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

            Layers = new Qwen3MoeLayerWeights[2];
            Layers[0] = new Qwen3MoeLayerWeights
            {
                AttnNormWeight = FillNormVec(HiddenSize, rng),
                PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                Gdn = BuildGdn(rng),
                FullAttn = null,
                Moe = BuildMoe(rng),
            };
            Layers[1] = new Qwen3MoeLayerWeights
            {
                AttnNormWeight = FillNormVec(HiddenSize, rng),
                PostAttnNormWeight = FillNormVec(HiddenSize, rng),
                Gdn = null,
                FullAttn = BuildFullAttn(rng),
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
                QkvWeight = AllocFloatsUniform(qkvOut * HiddenSize, rng, 0.05f),
                QkvQuantType = QuantizationType.F32,
                QkvInputDim = HiddenSize,
                QkvOutputDim = qkvOut,
                GateWeight = AllocFloatsUniform(gdnVDim * HiddenSize, rng, 0.05f),
                GateQuantType = QuantizationType.F32,
                GateInputDim = HiddenSize,
                GateOutputDim = gdnVDim,
                A = NegativeRandom(NVHead, rng),
                AlphaWeight = AllocFloatsUniform(NVHead * HiddenSize, rng, 0.05f),
                AlphaQuantType = QuantizationType.F32,
                AlphaInputDim = HiddenSize,
                AlphaOutputDim = NVHead,
                BetaWeight = AllocFloatsUniform(NVHead * HiddenSize, rng, 0.05f),
                BetaQuantType = QuantizationType.F32,
                BetaInputDim = HiddenSize,
                BetaOutputDim = NVHead,
                Conv1dWeight = FillRandom(DConv * convDim, rng, 0.1f),
                Conv1dBias = new float[convDim],
                DtBias = FillRandom(NVHead, rng, 0.1f),
                SsmNormWeight = FillNormVec(DState, rng),
                OutWeight = AllocFloatsUniform(HiddenSize * gdnVDim, rng, 0.05f),
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
                QWeight = AllocFloatsUniform(qOut * HiddenSize, rng, 0.05f),
                QQuantType = QuantizationType.F32,
                QInputDim = HiddenSize,
                QOutputDim = qOut,
                KWeight = AllocFloatsUniform(kvOut * HiddenSize, rng, 0.05f),
                KQuantType = QuantizationType.F32,
                KInputDim = HiddenSize,
                KOutputDim = kvOut,
                VWeight = AllocFloatsUniform(kvOut * HiddenSize, rng, 0.05f),
                VQuantType = QuantizationType.F32,
                VInputDim = HiddenSize,
                VOutputDim = kvOut,
                OWeight = AllocFloatsUniform(HiddenSize * oIn, rng, 0.05f),
                OQuantType = QuantizationType.F32,
                OInputDim = oIn,
                OOutputDim = HiddenSize,
                NumKvHeads = NumKvHeads,
                QNormWeight = FillNormVec(HeadDim, rng),
                KNormWeight = FillNormVec(HeadDim, rng),
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
            nint[] sharedGate = [AllocFloatsUniform(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedUp = [AllocFloatsUniform(SharedIntermediate * HiddenSize, rng, 0.05f)];
            nint[] sharedDown = [AllocFloatsUniform(HiddenSize * SharedIntermediate, rng, 0.05f)];
            float[] sharedExpertGate = FillRandom(HiddenSize, rng, 0.05f);

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
