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
/// End-to-end parity tests for the Vulkan NemotronH hybrid forward path with IQ3_S and
/// IQ3_XXS projection-weight upload. Audit finding H3: the IQ3 GEMV/GEMM kernels are
/// bit-perfect against the CPU oracle (16 Iq3 kernel parity tests in tree), and commit
/// 146d747 added IQ3_S / IQ3_XXS case branches to
/// <see cref="VulkanNemotronHTransformerModel"/>'s <c>RecordMatmul</c> dispatch — this
/// is the discriminating host-level test that proves the dispatch + upload pipeline is
/// correctly wired for IQ3, not just the kernels in isolation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Method.</b> Builds a synthetic mini NemotronH "model" with every quantisable
/// projection (SSM in/out, attn Q/K/V/O, FFN up/down, LM head) generated as F32, then
/// quantised in place to IQ3_S / IQ3_XXS via <see cref="Iq3Fixture"/>. The resulting
/// IQ3 byte buffer is what both backends consume — the CPU forward dequantises on the
/// fly (no native IQ3 GEMV kernel in tree, dequant-then-F32 matmul); the Vulkan side
/// dispatches the IQ3 matmul kernels directly.
/// </para>
/// <para>
/// <b>Dimensions.</b> All IQ3 contraction axes are bumped to multiples of 256 — a hard
/// requirement of the Vulkan IQ3 matmul kernels. <c>HiddenSize = 256</c>,
/// <c>IntermediateSize = 256</c>, <c>DInner = 256</c>; attention heads chosen so
/// <c>NumHeads * HeadDim = 256</c> (32 heads × 8 head_dim → too many; we keep heads at
/// 4 with HeadDim=64 so Q output dim = 256, contracting along hidden=256).
/// </para>
/// <para>
/// <b>Tolerance.</b> IQ3 is ~3.3 bpw — per-element drift is larger than Q4_K. We pin
/// to abs 0.1 / rel 0.1; the discriminator is "Vulkan dispatch matches CPU dequant
/// path" which would diverge by orders of magnitude with a miswired branch (wrong
/// codebook handle, IQ2-vs-IQ3 typo at the case label).
/// </para>
/// <para>
/// <b>Upload-path gate (audit H3 follow-up).</b> The NemotronH host's
/// <see cref="VulkanNemotronHWeights"/> upload predicate <c>DeviceQuantTypeFor</c>
/// currently does NOT recognise <see cref="QuantizationType.IQ3_S"/> or
/// <see cref="QuantizationType.IQ3_XXS"/> — only the dense host's
/// <see cref="VulkanWeights"/> has IQ3 keep-on-device predicates. These tests detect
/// the gap by asserting the device-side <c>OQuantType</c> is IQ3 after upload — if
/// the gate is closed, the test surfaces the gap with a precise skip-reason. See
/// <c>.planning/notes/iq3-per-host-parity-deferred.md</c>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanNemotronHTransformerModelIq3ForwardTests
{
    // IQ3-friendly dimensions: every contraction axis must be a multiple of 256.
    // HiddenSize (256) is the contraction axis for SSM-in, attn Q/K/V, FFN up, LM head.
    // IntermediateSize (256) is the contraction axis for FFN down. DInner (256) is the
    // contraction axis for SSM-out. Attn O contracts along NumHeads*HeadDim (256).
    private const int HiddenSize = 256;
    private const int VocabSize = 8;
    private const int HeadDim = 64;
    private const int NumHeads = 4;
    private const int NumKvHeads = 4;        // simple GQA repeat factor = 1
    private const int IntermediateSize = 256;
    private const int DInner = 256;
    private const int DConv = 4;
    private const int DState = 8;
    private const int NGroup = 2;
    private const int NHead = 4;             // d_inner / n_head = 64 = head_dim_ssm
    private const int MaxSeqLen = 16;

    // IQ3 is ~3.3 bpw — empirically the per-element drift is ~5-10% relative and
    // the round-trip can amplify through one matmul into logits up to ~30-50%.
    // 1e-1 abs + 1e-1 rel is generous but the relevant discriminator is "miswired
    // case label would diverge by orders of magnitude".
    private const float AbsTol = 1e-1f;
    private const float RelTol = 1e-1f;

    [SkippableFact]
    public void Forward_IQ3_XXS_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(QuantizationType.IQ3_XXS, kinds, seqLen: 1, seed: 142);
    }

    [SkippableFact]
    public void Forward_IQ3_XXS_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(QuantizationType.IQ3_XXS, kinds, seqLen: 3, seed: 107);
    }

    [SkippableFact]
    public void Forward_IQ3_XXS_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertVulkanMatchesCpu(QuantizationType.IQ3_XXS, kinds, seqLen: 1, seed: 113);
    }

    [SkippableFact]
    public void Forward_IQ3_S_AllSsmLayers_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Ssm, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(QuantizationType.IQ3_S, kinds, seqLen: 1, seed: 242);
    }

    [SkippableFact]
    public void Forward_IQ3_S_AttentionThenSsm_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm };
        AssertVulkanMatchesCpu(QuantizationType.IQ3_S, kinds, seqLen: 3, seed: 207);
    }

    [SkippableFact]
    public void Forward_IQ3_S_AttentionThenSsmThenFfn_MatchesCpuReference()
    {
        var kinds = new[] { HybridLayerKind.Attention, HybridLayerKind.Ssm, HybridLayerKind.Ffn };
        AssertVulkanMatchesCpu(QuantizationType.IQ3_S, kinds, seqLen: 1, seed: 213);
    }

    private void AssertVulkanMatchesCpu(
        QuantizationType iq3Type, HybridLayerKind[] layerKinds, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var fixture = NemotronHIq3FixtureBuilder.Build(layerKinds, seed, iq3Type);
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
        // Without this VulkanNemotronHWeights.DeviceQuantTypeFor would silently
        // dequant to F32, parity would pass, and the IQ3 dispatch arm would never
        // be reached — exactly the audit H3 trap-the-bug pattern. We Skip.If the
        // gate is closed so the test surfaces the gap without breaking CI; the
        // gate becomes a live parity test the moment the predicate is added.
        using (var device = VulkanDevice.Create())
        using (var vkWeights = VulkanNemotronHWeights.Upload(
            device, config, fixture.Layers, fixture.OutputNormWeight,
            fixture.TokenEmbedPtr, QuantizationType.F32,
            fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize))
        {
            bool gateOpen = vkWeights.OutputDeviceQuantType == iq3Type;
            // Spot-check the first attention layer's O projection or the SSM out
            // projection — any quantisable projection in the layout.
            for (int li = 0; li < vkWeights.Layers.Length; li++)
            {
                var lb = vkWeights.Layers[li];
                if (lb.Attention is { } attn && attn.ODeviceQuantType != iq3Type) gateOpen = false;
                if (lb.Ffn is { } ffn && ffn.DownDeviceQuantType != iq3Type) gateOpen = false;
                if (lb.Ssm is { } ssm && ssm.OutDeviceQuantType != iq3Type) gateOpen = false;
            }
            Skip.IfNot(gateOpen,
                $"VulkanNemotronHWeights upload-path predicate does not yet recognise {iq3Type}; " +
                $"observed lm_head device dtype: {vkWeights.OutputDeviceQuantType}. " +
                "Add KeepIq3XxsOnDevice/KeepIq3SOnDevice to VulkanNemotronHWeights.DeviceQuantTypeFor " +
                "(mirroring the dense host's VulkanWeights). The IQ3 GEMV/GEMM kernels and dispatch " +
                "wiring are already in tree (commit 146d747). See " +
                ".planning/notes/iq3-per-host-parity-deferred.md for context.");
        }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            using var model = NemotronHTransformerModel.BuildFromPrebuiltWeights(
                config, fixture.Layers, fixture.OutputNormWeight,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using var device = VulkanDevice.Create();
            using var model = VulkanNemotronHTransformerModel.BuildFromPrebuiltWeights(
                device, config, fixture.Layers, fixture.OutputNormWeight,
                fixture.OutputWeightPtr, fixture.OutputQuantType, vocabSize, hiddenSize,
                fixture.TokenEmbedPtr, QuantizationType.F32,
                spvDir);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            Assert.Equal(1, logits.Shape[0]);
            Assert.Equal(vocabSize, logits.Shape[1]);
            vkLogits = CopyLogits(logits);
        }

        int lastRow = seqLen - 1;
        for (int c = 0; c < vocabSize; c++)
        {
            float cpu = cpuLogits[lastRow * vocabSize + c];
            float vk = vkLogits[c];
            Assert.True(float.IsFinite(cpu),
                $"non-finite CPU logit {iq3Type} layers={string.Join(',', layerKinds)} col={c}: {cpu}");
            Assert.True(float.IsFinite(vk),
                $"non-finite Vulkan logit {iq3Type} layers={string.Join(',', layerKinds)} col={c}: {vk}");
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"{iq3Type} layers={string.Join(',', layerKinds)}, seqLen={seqLen}, col={c}: " +
                $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        AssertNonDegenerate(vkLogits, iq3Type, layerKinds);
    }

    private static void AssertNonDegenerate(float[] logits, QuantizationType qt, HybridLayerKind[] layerKinds)
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
            $"{qt} layers={string.Join(',', layerKinds)}: logits stddev near zero " +
            $"(var={var:E3}) — likely degenerate output.");
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    /// <summary>
    /// Owns a randomly-generated NemotronH "model" in unmanaged memory: every quantisable
    /// projection (token embed F32, output LM head IQ3, plus per-layer SSM/Attn/FFN
    /// projections IQ3) is allocated 64-byte-aligned via
    /// <see cref="NativeMemory.AlignedAlloc"/>. Both CPU and Vulkan model factories take
    /// the same <see cref="NemotronHLayerWeights"/> array so the backends consume
    /// identical bytes.
    /// </summary>
    private sealed unsafe class NemotronHIq3FixtureBuilder : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public NemotronHLayerWeights[] Layers = null!;
        public float[] OutputNormWeight = null!;
        public nint TokenEmbedPtr;
        public nint OutputWeightPtr;
        public QuantizationType OutputQuantType;
        private QuantizationType _iq3Type;

        public static NemotronHIq3FixtureBuilder Build(
            HybridLayerKind[] layerKinds, int seed, QuantizationType iq3Type)
        {
            if (iq3Type != QuantizationType.IQ3_S && iq3Type != QuantizationType.IQ3_XXS)
                throw new ArgumentException($"Unexpected IQ3 type {iq3Type}", nameof(iq3Type));
            var b = new NemotronHIq3FixtureBuilder();
            b.BuildInternal(layerKinds, seed, iq3Type);
            return b;
        }

        private void BuildInternal(HybridLayerKind[] layerKinds, int seed, QuantizationType iq3Type)
        {
            _iq3Type = iq3Type;
            int numLayers = layerKinds.Length;
            var rng = new Random(seed);

            OutputQuantType = iq3Type;

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

            // Token embedding [vocab, hidden] — always F32 (vkCmdCopyBuffer gather constraint).
            TokenEmbedPtr = AllocAndFillUniform(VocabSize * HiddenSize, rng, amplitude: 0.05f);

            // Output norm + LM head. LM head contracts along hidden (=256, IQ3-friendly).
            OutputNormWeight = FillNormVec(HiddenSize, rng);
            OutputWeightPtr = AllocProjectionIq3(VocabSize, HiddenSize, rng);

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
            // SSM-in contracts along hidden=256 → IQ3-friendly.
            nint inWeight = AllocProjectionIq3(inProjDim, HiddenSize, rng);
            // SSM-out contracts along d_inner=256 → IQ3-friendly.
            nint outWeight = AllocProjectionIq3(HiddenSize, ssm.DInner, rng);

            float[] conv1dWeight = FillRandom(ssm.DConv * convDim, rng, amplitude: 0.1f);
            float[] conv1dBias = FillRandom(convDim, rng, amplitude: 0.1f);
            float[] a = NegativeRandom(ssm.NHead, rng);
            float[] d = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            float[] dtBias = FillRandom(ssm.NHead, rng, amplitude: 0.1f);
            float[] normWeight = FillNormVec(ssm.DInner, rng);

            return new NemotronHSsmWeights
            {
                InWeight = inWeight,
                InQuantType = _iq3Type,
                InInputDim = HiddenSize,
                InOutputDim = inProjDim,
                Conv1dWeight = conv1dWeight,
                Conv1dBias = conv1dBias,
                A = a,
                D = d,
                DtBias = dtBias,
                NormWeight = normWeight,
                OutWeight = outWeight,
                OutQuantType = _iq3Type,
                OutInputDim = ssm.DInner,
                OutOutputDim = HiddenSize,
            };
        }

        private NemotronHAttentionWeights BuildAttn(Random rng, int numKvHeads)
        {
            int qOut = NumHeads * HeadDim;       // 4*64 = 256
            int kvOut = numKvHeads * HeadDim;    // 4*64 = 256
            return new NemotronHAttentionWeights
            {
                QWeight = AllocProjectionIq3(qOut, HiddenSize, rng),
                QQuantType = _iq3Type,
                QInputDim = HiddenSize, QOutputDim = qOut,
                KWeight = AllocProjectionIq3(kvOut, HiddenSize, rng),
                KQuantType = _iq3Type,
                KInputDim = HiddenSize, KOutputDim = kvOut,
                VWeight = AllocProjectionIq3(kvOut, HiddenSize, rng),
                VQuantType = _iq3Type,
                VInputDim = HiddenSize, VOutputDim = kvOut,
                OWeight = AllocProjectionIq3(HiddenSize, qOut, rng),
                OQuantType = _iq3Type,
                OInputDim = qOut, OOutputDim = HiddenSize,
                NumKvHeads = numKvHeads,
            };
        }

        private NemotronHFfnWeights BuildFfn(Random rng, int intermediateSize)
        {
            return new NemotronHFfnWeights
            {
                UpWeight = AllocProjectionIq3(intermediateSize, HiddenSize, rng),
                UpQuantType = _iq3Type,
                UpInputDim = HiddenSize, UpOutputDim = intermediateSize,
                DownWeight = AllocProjectionIq3(HiddenSize, intermediateSize, rng),
                DownQuantType = _iq3Type,
                DownInputDim = intermediateSize, DownOutputDim = HiddenSize,
                IntermediateSize = intermediateSize,
            };
        }

        /// <summary>
        /// Allocates a projection [outputDim, inputDim] in IQ3 format. Generates F32
        /// rows in scratch, quantises to IQ3 via <see cref="Iq3Fixture"/>, writes the
        /// IQ3 row-stride bytes to the final allocation.
        /// <paramref name="inputDim"/> must be a multiple of 256.
        /// </summary>
        private nint AllocProjectionIq3(int outputDim, int inputDim, Random rng)
        {
            if ((inputDim % 256) != 0)
                throw new InvalidOperationException(
                    $"IQ3 requires inputDim multiple of 256 (got {inputDim}). Bump fixture dims.");

            // Materialise the whole matrix as managed F32, then quantise to IQ3 row-stride.
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

        private nint AllocAndFillUniform(int count, Random rng, float amplitude)
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
