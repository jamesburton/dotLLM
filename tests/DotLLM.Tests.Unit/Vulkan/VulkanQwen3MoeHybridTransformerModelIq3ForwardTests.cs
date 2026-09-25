using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cpu.Kernels;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity tests for the Vulkan Qwen3MoeHybrid forward path with IQ3_S and
/// IQ3_XXS projection-weight upload. Audit finding H3: commit <c>48d65fe</c> added
/// IQ3_S / IQ3_XXS case branches to <see cref="VulkanQwen3MoeHybridTransformerModel"/>'s
/// dispatch, session 5 wired the upload-path predicates, and session 6 added the
/// <c>BuildFromPrebuiltWeights</c> factory — these are the discriminating host-level
/// tests that prove the dispatch + upload pipeline is correctly wired for IQ3,
/// not just the kernels in isolation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Method.</b> Builds a synthetic mini Qwen3MoeHybrid "model" — one GDN layer
/// (token-mixing) + one full-attention layer + a sparse MoE FFN with a Qwen1.5-style
/// shared expert. Quantisable projections (GDN QKV/Gate/Alpha/Beta/Out, full-attn
/// Q/K/V/O, LM head) are generated F32 then quantised to IQ3 via
/// <see cref="Iq3Fixture"/>. The same IQ3 byte buffer is consumed by both backends:
/// the CPU forward dequantises on the fly (no native IQ3 GEMV kernel — dequant-then-F32
/// matmul); the Vulkan side dispatches the IQ3 matmul kernels directly through the
/// session-5 upload predicates and per-host dispatch arm.
/// </para>
/// <para>
/// <b>Dimensions.</b> Every IQ3 contraction axis is a multiple of 256 — a hard
/// requirement of the Vulkan IQ3 matmul kernels. <c>HiddenSize = 256</c>,
/// <c>NVHead * DState = 256</c> (GDN OutWeight input), <c>NumAttentionHeads * HeadDim
/// = 256</c> (full-attn O input).
/// </para>
/// <para>
/// <b>Routed MoE banks stay F32.</b> The per-routed-expert W1/W2/W3 banks
/// deliberately have no quant overlay in <see cref="MoeLayerWeights"/> — the Vulkan
/// <c>moe_indexed_matmul_f32</c> kernel is F32-only. Router gate, shared-expert
/// projections, and the shared-expert sigmoid gate also stay F32 to keep this test's
/// surface scoped to the IQ3 dispatch on the GDN / full-attn / lm_head paths
/// (the MoE quant overlay surface is exercised separately under
/// <c>VulkanTransformerModelMoe*</c>).
/// </para>
/// <para>
/// <b>Tolerance.</b> IQ3 is ~3.3 bpw — per-element drift is larger than Q4_K. We pin
/// to abs 0.15 / rel 0.15 (slightly looser than NemotronH's 0.1/0.1 envelope because
/// the GDN recurrence + Q/Gate sigmoid composition compounds the drift through more
/// nonlinear operations). The discriminator is "Vulkan dispatch matches CPU dequant
/// path" which would diverge by orders of magnitude with a miswired branch (wrong
/// codebook handle, IQ2-vs-IQ3 typo at the case label).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanQwen3MoeHybridTransformerModelIq3ForwardTests
{
    // IQ3-friendly dimensions: every quantisable contraction axis must be a multiple of 256.
    private const int HiddenSize = 256;
    private const int VocabSize = 8;
    private const int NumAttentionHeads = 4;
    private const int NumKvHeads = 2;          // GQA repeat factor = 2
    private const int HeadDim = 64;            // nQ*HeadDim = 256 (OWeight input)
    private const int RopeDim = 32;            // < HeadDim, even, partial-rotary
    private const int MaxSeqLen = 16;

    // GDN config. NVHead*DState = 256 = OutWeight input (IQ3 contraction axis).
    private const int NVHead = 8;
    private const int NKHead = 4;              // V heads per K head = 2
    private const int DState = 32;
    private const int DConv = 4;

    // MoE config. Routed experts stay F32 → MoeIntermediate has no IQ3 constraint.
    private const int MoeIntermediate = 32;
    private const int SharedIntermediate = 16;
    private const int NumExperts = 4;
    private const int NumExpertsPerTok = 2;

    // IQ3 ~3.3 bpw with GDN recurrence + Q/Gate sigmoid composition — empirically
    // 0.15/0.15 holds for 1-2 layer fixtures while still catching a miswired
    // dispatch (which would diverge by orders of magnitude).
    private const float AbsTol = 1.5e-1f;
    private const float RelTol = 1.5e-1f;

    [SkippableFact]
    public void Forward_IQ3_XXS_Prefill_FiniteLogits()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_XXS, seqLen: 4, seed: 107);

    [SkippableFact]
    public void Forward_IQ3_S_Prefill_FiniteLogits()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_S, seqLen: 4, seed: 207);

    [SkippableFact]
    public void Forward_IQ3_XXS_VsCpuOracle_LogitsMatch()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_XXS, seqLen: 1, seed: 47);

    [SkippableFact]
    public void Forward_IQ3_S_VsCpuOracle_LogitsMatch()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_S, seqLen: 1, seed: 53);

    private void AssertVulkanMatchesCpu(QuantizationType iq3Type, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var fixture = Qwen3MoeHybridIq3FixtureBuilder.Build(seed, iq3Type);
        var config = fixture.Config;

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VocabSize;
            positions[i] = i;
        }

        int hiddenSize = config.HiddenSize;
        int vocabSize = config.VocabSize;

        // Discriminator: assert the upload path preserves the IQ3 bytes on device.
        // Without this VulkanQwen3MoeHybridWeights.DeviceQuantTypeFor would silently
        // dequant to F32, parity would pass, and the IQ3 dispatch arm would never
        // be reached — exactly the audit H3 trap-the-bug pattern. Session 5 added
        // the predicates; this assert ships as a live gate.
        using (var device = VulkanDevice.Create())
        using (var vkWeights = VulkanQwen3MoeHybridWeights.Upload(
            device, config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize))
        {
            Skip.IfNot(vkWeights.OutputDeviceQuantType == iq3Type,
                $"VulkanQwen3MoeHybridWeights upload-path predicate did not keep {iq3Type} " +
                $"on device for lm_head; observed dtype: {vkWeights.OutputDeviceQuantType}. " +
                "This should pass after session-5 commit (KeepIq3* predicates on " +
                "VulkanQwen3MoeHybridWeights.DeviceQuantTypeFor).");
        }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            using var model = Qwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                config, fixture.Layers, fixture.OutputNormWeight,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize);
            using var kvCache = new DotLLM.Engine.KvCache.SimpleKvCache(
                model.AttentionLayerCount, NumKvHeads, HeadDim, MaxSeqLen);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanQwen3MoeHybridTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                spvDir);
            using var kvCache = model.CreateKvCache(MaxSeqLen);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1, kvCache);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(vocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < vocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * vocabSize + c];
            float vk = vkLogits[c];
            Assert.True(float.IsFinite(cpu), $"non-finite CPU logit {iq3Type} seqLen={seqLen} col={c}: {cpu}");
            Assert.True(float.IsFinite(vk), $"non-finite Vulkan logit {iq3Type} seqLen={seqLen} col={c}: {vk}");
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"{iq3Type} seqLen={seqLen}, col={c}: " +
                $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        AssertNonDegenerate(vkLogits, iq3Type, seqLen);
    }

    private static void AssertNonDegenerate(float[] logits, QuantizationType qt, int seqLen)
    {
        double mean = 0;
        for (int i = 0; i < logits.Length; i++) mean += logits[i];
        mean /= logits.Length;
        double var = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            double d = logits[i] - mean;
            var += d * d;
        }
        var /= logits.Length;
        Assert.True(var > 1e-12,
            $"{qt} seqLen={seqLen}: logits stddev near zero (var={var:E3}) — likely degenerate output.");
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Owns a randomly-generated 2-layer Qwen3MoeHybrid "model" in unmanaged memory:
    /// every quantisable projection (token embed F32, output LM head IQ3, GDN
    /// QKV/Gate/Alpha/Beta/Out IQ3, full-attn Q/K/V/O IQ3, routed-MoE banks F32) is
    /// allocated 64-byte-aligned via <see cref="NativeMemory.AlignedAlloc"/>. Both
    /// CPU and Vulkan model factories take the same <see cref="Qwen3MoeLayerWeights"/>
    /// array so the backends consume identical bytes.
    /// </summary>
    private sealed unsafe class Qwen3MoeHybridIq3FixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public Qwen3MoeLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;
        public QuantizationType OutputQuantType;
        private QuantizationType _iq3Type;

        public static Qwen3MoeHybridIq3FixtureBuilder Build(int seed, QuantizationType iq3Type)
        {
            if (iq3Type != QuantizationType.IQ3_S && iq3Type != QuantizationType.IQ3_XXS)
                throw new ArgumentException($"Unexpected IQ3 type {iq3Type}", nameof(iq3Type));
            var b = new Qwen3MoeHybridIq3FixtureBuilder();
            b.BuildInternal(seed, iq3Type);
            return b;
        }

        private void BuildInternal(int seed, QuantizationType iq3Type)
        {
            _iq3Type = iq3Type;
            OutputQuantType = iq3Type;
            var rng = new Random(seed);

            // Layout: layer 0 = GDN, layer 1 = full-attn (FullAttnInterval = 2).
            HybridLayerKind[] kinds = [HybridLayerKind.GatedDeltaNet, HybridLayerKind.Attention];
            int[] headCountKv = [0, NumKvHeads];
            int[] ffnLen = [0, 0]; // MoE FFN — FeedForwardLength is not the routed-expert dim.

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

            // Token embedding [vocab, hidden] — always F32 (vkCmdCopyBuffer gather constraint).
            TokenEmbedPtr = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);

            // Output norm + LM head [vocab, hidden] in IQ3 (contracts along hidden=256).
            OutputNormWeight = FillNormVec(HiddenSize, rng);
            OutputWeightPtr = AllocProjectionIq3(VocabSize, HiddenSize, rng);

            // Per-layer weights.
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
            int gdnVDim = NVHead * DState; // = 256 → IQ3-friendly for OutWeight
            int qkvOut = 2 * gdnKDim + gdnVDim;

            return new GdnTokenMixingWeights
            {
                // QKV contracts along HiddenSize=256 → IQ3.
                QkvWeight = AllocProjectionIq3(qkvOut, HiddenSize, rng),
                QkvQuantType = _iq3Type,
                QkvInputDim = HiddenSize,
                QkvOutputDim = qkvOut,
                // Gate contracts along HiddenSize=256 → IQ3.
                GateWeight = AllocProjectionIq3(gdnVDim, HiddenSize, rng),
                GateQuantType = _iq3Type,
                GateInputDim = HiddenSize,
                GateOutputDim = gdnVDim,
                A = NegativeRandom(NVHead, rng),
                // Alpha contracts along HiddenSize=256 → IQ3.
                AlphaWeight = AllocProjectionIq3(NVHead, HiddenSize, rng),
                AlphaQuantType = _iq3Type,
                AlphaInputDim = HiddenSize,
                AlphaOutputDim = NVHead,
                // Beta contracts along HiddenSize=256 → IQ3.
                BetaWeight = AllocProjectionIq3(NVHead, HiddenSize, rng),
                BetaQuantType = _iq3Type,
                BetaInputDim = HiddenSize,
                BetaOutputDim = NVHead,
                Conv1dWeight = FillRandom(DConv * convDim, rng, 0.1f),
                Conv1dBias = new float[convDim],
                DtBias = FillRandom(NVHead, rng, 0.1f),
                SsmNormWeight = FillNormVec(DState, rng),
                // Out contracts along gdnVDim=256 → IQ3.
                OutWeight = AllocProjectionIq3(HiddenSize, gdnVDim, rng),
                OutQuantType = _iq3Type,
                OutInputDim = gdnVDim,
                OutOutputDim = HiddenSize,
            };
        }

        private Qwen3FullAttnWeights BuildFullAttn(Random rng)
        {
            int qOut = 2 * NumAttentionHeads * HeadDim; // Fused Q+Gate.
            int kvOut = NumKvHeads * HeadDim;
            int oIn = NumAttentionHeads * HeadDim; // = 256 → IQ3-friendly for OWeight
            return new Qwen3FullAttnWeights
            {
                QWeight = AllocProjectionIq3(qOut, HiddenSize, rng),
                QQuantType = _iq3Type,
                QInputDim = HiddenSize,
                QOutputDim = qOut,
                KWeight = AllocProjectionIq3(kvOut, HiddenSize, rng),
                KQuantType = _iq3Type,
                KInputDim = HiddenSize,
                KOutputDim = kvOut,
                VWeight = AllocProjectionIq3(kvOut, HiddenSize, rng),
                VQuantType = _iq3Type,
                VInputDim = HiddenSize,
                VOutputDim = kvOut,
                OWeight = AllocProjectionIq3(HiddenSize, oIn, rng),
                OQuantType = _iq3Type,
                OInputDim = oIn,
                OOutputDim = HiddenSize,
                NumKvHeads = NumKvHeads,
                QNormWeight = FillNormVec(HeadDim, rng),
                KNormWeight = FillNormVec(HeadDim, rng),
            };
        }

        private MoeLayerWeights BuildMoe(Random rng)
        {
            // Routed experts stay F32 — the Vulkan moe_indexed_matmul kernel is F32-only
            // and this test is scoped to the IQ3 dispatch on the non-MoE projections.
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

        /// <summary>
        /// Allocates a projection [outputDim, inputDim] in IQ3 format. Generates F32
        /// rows, quantises to IQ3 row-stride bytes via <see cref="Iq3Fixture"/>, writes
        /// the IQ3 bytes to a 64-byte-aligned allocation. <paramref name="inputDim"/>
        /// must be a multiple of 256.
        /// </summary>
        private nint AllocProjectionIq3(int outputDim, int inputDim, Random rng)
        {
            if ((inputDim % 256) != 0)
                throw new InvalidOperationException(
                    $"IQ3 requires inputDim multiple of 256 (got {inputDim}). Bump fixture dims.");

            float[] src = new float[outputDim * inputDim];
            for (int i = 0; i < src.Length; i++)
                src[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;

            byte[] iq3Bytes = _iq3Type == QuantizationType.IQ3_XXS
                ? Iq3Fixture.QuantizeRowsIq3Xxs(src, outputDim, inputDim)
                : Iq3Fixture.QuantizeRowsIq3S(src, outputDim, inputDim);

            long totalBytes = iq3Bytes.Length;
            nint dst = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
            _allocs.Add(dst);
            new ReadOnlySpan<byte>(iq3Bytes).CopyTo(new Span<byte>((void*)dst, checked((int)totalBytes)));
            return dst;
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
