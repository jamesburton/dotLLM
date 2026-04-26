using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using Architecture = DotLLM.Core.Configuration.Architecture;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// End-to-end parity test for the Vulkan NemotronH hybrid forward pass against the CPU
/// reference. Builds a synthetic mini NemotronH "model" with all-F32 random projections
/// owned in unmanaged memory by the test fixture, loads it on both backends, runs
/// <c>Forward</c>, and asserts the last-token logits row matches.
/// </summary>
/// <remarks>
/// All projection weights are F32, sidestepping quant-handling complexity for this first
/// cut. Drift comes from F32 reduction-order differences between the CPU and Vulkan
/// matmul / scan / softmax pipelines; tolerance is abs 5e-3 / rel 1e-3 on last-token
/// logits, matching the MoE/MLA forward parity tests.
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHTransformerModelForwardTests
{
    private const int HiddenSize = 16;
    private const int VocabSize = 8;
    private const int HeadDim = 8;
    private const int NumHeads = 2;
    private const int NumKvHeads = 2;
    private const int IntermediateSize = 24;
    private const int DInner = 16;
    private const int DConv = 4;
    private const int DState = 8;
    private const int NGroup = 2;
    private const int NHead = 2;       // d_inner / n_head = 8 = head_dim_ssm
    private const int MaxSeqLen = 16;

    private const float AbsTol = 5e-3f;
    private const float RelTol = 1e-3f;

    [SkippableFact]
    public void Forward_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(kinds, seqLen: 1, seed: 42);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(kinds, seqLen: 3, seed: 7);
    }

    [SkippableFact]
    public void Forward_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertVulkanMatchesCpu(kinds, seqLen: 1, seed: 13);
    }

    private void AssertVulkanMatchesCpu(HybridLayerKind[] layerKinds, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var fixture = NemotronHFixtureBuilder.Build(layerKinds, seed);
        var config = fixture.Config;

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++)
        {
            tokenIds[i] = i % VocabSize;
            positions[i] = i;
        }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            using var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
                config, fixture.Layers, fixture.OutputNormWeight,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, QuantizationType.F32, VocabSize, HiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(VocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < VocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * VocabSize + c];
            float vk = vkLogits[c];
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"layers={string.Join(',', layerKinds)}, seqLen={seqLen}, col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
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
    /// Owns a randomly-generated NemotronH "model" in unmanaged memory: every projection
    /// weight (token embed, output, plus per-layer ssm/attn/ffn projections) is allocated
    /// 64-byte-aligned via <see cref="NativeMemory.AlignedAlloc"/>. Both the CPU and Vulkan
    /// model factories take the same <see cref="NemotronHLayerWeights"/> array so the
    /// backends consume identical bytes.
    /// </summary>
    private sealed unsafe class NemotronHFixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public NemotronHLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;

        public static NemotronHFixtureBuilder Build(HybridLayerKind[] layerKinds, int seed)
        {
            var b = new NemotronHFixtureBuilder();
            b.BuildInternal(layerKinds, seed);
            return b;
        }

        private void BuildInternal(HybridLayerKind[] layerKinds, int seed)
        {
            int numLayers = layerKinds.Length;
            var rng = new Random(seed);

            // Per-layer kinds: hard-code per-layer KV head count and FFN length to match.
            var headCountKv = new int[numLayers];
            var ffnLength = new int[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                headCountKv[i] = layerKinds[i] == HybridLayerKind.Attention ? NumKvHeads : 0;
                ffnLength[i] = layerKinds[i] == HybridLayerKind.Ffn ? IntermediateSize : 0;
            }

            var layout = new HybridLayerLayout
            {
                LayerKind = layerKinds,
                HeadCountKv = headCountKv,
                FeedForwardLength = ffnLength,
            };

            var ssmConfig = new MambaSsmConfig(
                DConv: DConv, DInner: DInner, DState: DState, NGroup: NGroup, NHead: NHead);

            Config = new ModelConfig
            {
                Architecture = Architecture.NemotronH,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                IntermediateSize = IntermediateSize,
                NumLayers = numLayers,
                NumAttentionHeads = NumHeads,
                NumKvHeads = NumKvHeads,
                HeadDim = HeadDim,
                MaxSequenceLength = MaxSeqLen,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.RoPE,
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                HybridLayout = layout,
                SsmConfig = ssmConfig,
                ChatTemplate = null,
            };

            // Token embedding [vocab, hidden]
            TokenEmbedPtr = AllocAndFillUniform(VocabSize * HiddenSize, rng, amplitude: 0.05f);

            // Output norm + LM head.
            OutputNormWeight = FillNormVec(HiddenSize, rng);
            OutputWeightPtr = AllocAndFillUniform(VocabSize * HiddenSize, rng, amplitude: 0.05f);

            // Per-layer weights.
            Layers = new NemotronHLayerWeights[numLayers];
            for (int i = 0; i < numLayers; i++)
            {
                float[] attnNorm = FillNormVec(HiddenSize, rng);
                switch (layerKinds[i])
                {
                    case HybridLayerKind.Ssm:
                        Layers[i] = new NemotronHLayerWeights
                        {
                            AttnNormWeight = attnNorm,
                            Ssm = BuildSsm(rng, ssmConfig),
                        };
                        break;
                    case HybridLayerKind.Attention:
                        Layers[i] = new NemotronHLayerWeights
                        {
                            AttnNormWeight = attnNorm,
                            Attention = BuildAttn(rng, NumKvHeads),
                        };
                        break;
                    case HybridLayerKind.Ffn:
                        Layers[i] = new NemotronHLayerWeights
                        {
                            AttnNormWeight = attnNorm,
                            Ffn = BuildFfn(rng, IntermediateSize),
                        };
                        break;
                    default:
                        throw new InvalidOperationException();
                }
            }
        }

        private NemotronHSsmWeights BuildSsm(Random rng, MambaSsmConfig ssm)
        {
            int convDim = ssm.ConvDim;
            int inProjDim = ssm.InputProjectionDim;
            // ssm_in: [hidden, in_proj_dim] in GGUF; CPU oracle stores InOutputDim=in_proj_dim
            // and InInputDim=hidden, with the matmul reading nint as M=outputDim x K=inputDim.
            nint inWeight = AllocAndFillUniform(inProjDim * HiddenSize, rng, amplitude: 0.05f);
            nint outWeight = AllocAndFillUniform(HiddenSize * ssm.DInner, rng, amplitude: 0.05f);

            float[] conv1dWeight = FillRandom(ssm.DConv * convDim, rng, amplitude: 0.1f);
            float[] conv1dBias = FillRandom(convDim, rng, amplitude: 0.1f);
            // A: forced negative so exp(dt*A) decays.
            float[] a = NegativeRandom(ssm.NHead, rng);
            float[] d = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            float[] dtBias = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            // ssm_norm.weight is [d_inner] (group-wise gain).
            float[] normWeight = FillNormVec(ssm.DInner, rng);

            return new NemotronHSsmWeights
            {
                InWeight = inWeight,
                InQuantType = QuantizationType.F32,
                InInputDim = HiddenSize,
                InOutputDim = inProjDim,
                Conv1dWeight = conv1dWeight,
                Conv1dBias = conv1dBias,
                A = a,
                D = d,
                DtBias = dtBias,
                NormWeight = normWeight,
                OutWeight = outWeight,
                OutQuantType = QuantizationType.F32,
                OutInputDim = ssm.DInner,
                OutOutputDim = HiddenSize,
            };
        }

        private NemotronHAttentionWeights BuildAttn(Random rng, int numKvHeads)
        {
            int qOut = NumHeads * HeadDim;
            int kvOut = numKvHeads * HeadDim;
            return new NemotronHAttentionWeights
            {
                QWeight = AllocAndFillUniform(qOut * HiddenSize, rng, amplitude: 0.05f),
                QQuantType = QuantizationType.F32,
                QInputDim = HiddenSize, QOutputDim = qOut,
                KWeight = AllocAndFillUniform(kvOut * HiddenSize, rng, amplitude: 0.05f),
                KQuantType = QuantizationType.F32,
                KInputDim = HiddenSize, KOutputDim = kvOut,
                VWeight = AllocAndFillUniform(kvOut * HiddenSize, rng, amplitude: 0.05f),
                VQuantType = QuantizationType.F32,
                VInputDim = HiddenSize, VOutputDim = kvOut,
                OWeight = AllocAndFillUniform(HiddenSize * qOut, rng, amplitude: 0.05f),
                OQuantType = QuantizationType.F32,
                OInputDim = qOut, OOutputDim = HiddenSize,
                NumKvHeads = numKvHeads,
            };
        }

        private NemotronHFfnWeights BuildFfn(Random rng, int intermediateSize)
        {
            return new NemotronHFfnWeights
            {
                UpWeight = AllocAndFillUniform(intermediateSize * HiddenSize, rng, amplitude: 0.05f),
                UpQuantType = QuantizationType.F32,
                UpInputDim = HiddenSize, UpOutputDim = intermediateSize,
                DownWeight = AllocAndFillUniform(HiddenSize * intermediateSize, rng, amplitude: 0.05f),
                DownQuantType = QuantizationType.F32,
                DownInputDim = intermediateSize, DownOutputDim = HiddenSize,
                IntermediateSize = intermediateSize,
            };
        }

        private nint AllocAndFillUniform(int count, Random rng, float amplitude)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(ptr);
            float* dst = (float*)ptr;
            for (int i = 0; i < count; i++)
            {
                dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            }
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
            // Centred at 1.0 with small jitter, like RMSNorm gamma weights.
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        private static float[] NegativeRandom(int count, Random rng)
        {
            // A forced negative so the SSM scan exp(dt*A) decays.
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = -((float)rng.NextDouble() * 0.5f + 0.1f);
            return arr;
        }

        public void Dispose()
        {
            foreach (var p in _allocs)
            {
                NativeMemory.AlignedFree((void*)p);
            }
            _allocs.Clear();
        }
    }
}
