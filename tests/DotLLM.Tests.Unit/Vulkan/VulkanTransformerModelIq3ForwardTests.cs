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
/// End-to-end parity tests for the Vulkan dense transformer forward path with IQ3_S
/// and IQ3_XXS projection-weight upload. Audit finding H3: the IQ3 GEMV/GEMM kernels
/// are bit-perfect against the CPU oracle (16 Iq3 kernel parity tests in tree), and
/// commit 07f391f added IQ3_S / IQ3_XXS case branches to
/// <see cref="VulkanTransformerModel"/>'s <c>RecordMatmul</c> dispatch. This is the
/// discriminating host-level test that proves the dispatch + upload pipeline is
/// correctly wired for IQ3, not just the kernels in isolation.
/// </summary>
/// <remarks>
/// <para>
/// <b>Method.</b> Builds a synthetic dense Llama-style "model" with every projection
/// (Q/K/V/O attention + Gate/Up/Down FFN + LM head) generated as F32, then quantised
/// in place to IQ3 via <see cref="Iq3Fixture"/>. The raw IQ3 bytes are pinned to the
/// <see cref="TransformerLayerWeights.QWeight"/> et al pointers with the matching
/// <see cref="QuantizationType"/> field; both backends consume the same bytes
/// (CPU dequants on the fly, Vulkan dispatches the IQ3 matmul kernels).
/// </para>
/// <para>
/// <b>Dimensions.</b> All IQ3 contraction axes are bumped to multiples of 256:
/// <c>HiddenSize = 256</c>, <c>IntermediateSize = 256</c>. Attention dim
/// <c>NumHeads * HeadDim = 256</c> with NumHeads=4 / HeadDim=64. The O projection
/// contracts along <c>NumHeads * HeadDim</c> = 256 → IQ3-friendly.
/// </para>
/// <para>
/// <b>Tolerance.</b> abs 1e-1 / rel 1e-1 — generous IQ3 envelope. The discriminator
/// "Vulkan dispatch matches CPU dequant-then-F32" would diverge by orders of
/// magnitude if any of the four dispatch branches (IQ3_XXS GEMV, IQ3_XXS GEMM,
/// IQ3_S GEMV, IQ3_S GEMM) was miswired.
/// </para>
/// <para>
/// <b>Upload-path gate (audit H3 follow-up).</b> The dense host's
/// <see cref="VulkanWeights"/> already has <c>KeepIq3XxsOnDevice</c> /
/// <c>KeepIq3SOnDevice</c> predicates in its <c>DeviceQuantTypeFor</c> dispatch —
/// this test EXPECTS the gate to be open and parity to hold. If the test surfaces a
/// failure here, it means the IQ3 dispatch wiring in
/// <see cref="VulkanTransformerModel"/> diverged from the CPU oracle (the audit's
/// trap-the-bug shape).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelIq3ForwardTests : IDisposable
{
    // IQ3-friendly dimensions: every contraction axis must be a multiple of 256.
    private const int HiddenSize = 256;
    private const int IntermediateSize = 256;
    private const int VocabSize = 8;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int NumKvHeads = 4;       // GQA repeat factor 1
    private const int HeadDim = 64;          // NumHeads * HeadDim = 256
    private const int MaxSeqLen = 8;

    private const float AbsTol = 1e-1f;
    private const float RelTol = 1e-1f;

    // Note: allocations are passed to TransformerWeights via ownedAllocations and
    // freed by TransformerWeights.Dispose() (called from TransformerModel.Dispose()).
    // The test does NOT double-track them — doing so would cause a double-free when
    // the test fixture disposes.
    public void Dispose() { }

    [SkippableFact]
    public void Forward_IQ3_XXS_SingleToken_MatchesCpuReference()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_XXS, seqLen: 1, seed: 42);

    [SkippableFact]
    public void Forward_IQ3_XXS_Prefill_MatchesCpuReference()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_XXS, seqLen: 3, seed: 7);

    [SkippableFact]
    public void Forward_IQ3_S_SingleToken_MatchesCpuReference()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_S, seqLen: 1, seed: 142);

    [SkippableFact]
    public void Forward_IQ3_S_Prefill_MatchesCpuReference()
        => AssertVulkanMatchesCpu(QuantizationType.IQ3_S, seqLen: 3, seed: 107);

    private void AssertVulkanMatchesCpu(QuantizationType iq3Type, int seqLen, int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        ModelConfig config = BuildConfig();

        // Stage the IQ3 byte blob ONCE deterministically, then materialise two
        // independent TransformerWeights copies (CPU + Vulkan) so each model owns
        // its own allocations and can dispose independently. Re-quantising twice
        // would be wasteful (the IQ3_S quantiser is O(33M comparisons / pair) and
        // takes minutes on a [256,256] matrix); pre-staging the bytes lets both
        // backends consume identical fixture data while sidestepping double-free.
        var blob = StageIq3Blob(seed, iq3Type);

        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            TransformerWeights cpuWeights = MaterialiseWeights(blob, config, iq3Type);
            using var model = TransformerModel.BuildFromPrebuiltWeights(cpuWeights, config);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            TransformerWeights vkWeights = MaterialiseWeights(blob, config, iq3Type);
            using var device = VulkanDevice.Create();
            using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, config, vkWeights, spvDir);
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
            Assert.True(float.IsFinite(cpu), $"non-finite CPU logit {iq3Type} seqLen={seqLen} col={c}: {cpu}");
            Assert.True(float.IsFinite(vk), $"non-finite Vulkan logit {iq3Type} seqLen={seqLen} col={c}: {vk}");
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"{iq3Type} dense seqLen={seqLen}, col={c}: " +
                $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        AssertNonDegenerate(vkLogits, iq3Type, seqLen);
    }

    /// <summary>
    /// Per-layer IQ3 byte blobs (already quantised) + per-vector F32 norm buffers.
    /// Generated once per test and consumed twice (CPU + Vulkan) via
    /// <see cref="MaterialiseWeights"/>, each materialisation allocating its own
    /// native copies so each <see cref="TransformerWeights"/> owns disjoint memory.
    /// </summary>
    private sealed class Iq3StagedBlob
    {
        public float[] TokenEmbedF32 = null!;
        public byte[] LmHeadIq3 = null!;
        public float[] OutputNormWeight = null!;
        public LayerBlob[] Layers = null!;

        public sealed class LayerBlob
        {
            public float[] AttnNormWeight = null!;
            public float[] FfnNormWeight = null!;
            public byte[] QIq3 = null!;
            public byte[] KIq3 = null!;
            public byte[] VIq3 = null!;
            public byte[] OIq3 = null!;
            public byte[] GateIq3 = null!;
            public byte[] UpIq3 = null!;
            public byte[] DownIq3 = null!;
        }
    }

    private static Iq3StagedBlob StageIq3Blob(int seed, QuantizationType iq3Type)
    {
        var rng = new Random(seed);
        var b = new Iq3StagedBlob
        {
            TokenEmbedF32 = FillF32Array(VocabSize * HiddenSize, rng, 0.05f),
            OutputNormWeight = FillNormVec(HiddenSize, rng),
            LmHeadIq3 = QuantiseToIq3Bytes(VocabSize, HiddenSize, rng, iq3Type),
            Layers = new Iq3StagedBlob.LayerBlob[NumLayers],
        };
        for (int i = 0; i < NumLayers; i++)
        {
            b.Layers[i] = new Iq3StagedBlob.LayerBlob
            {
                AttnNormWeight = FillNormVec(HiddenSize, rng),
                FfnNormWeight = FillNormVec(HiddenSize, rng),
                QIq3 = QuantiseToIq3Bytes(NumHeads * HeadDim, HiddenSize, rng, iq3Type),
                KIq3 = QuantiseToIq3Bytes(NumKvHeads * HeadDim, HiddenSize, rng, iq3Type),
                VIq3 = QuantiseToIq3Bytes(NumKvHeads * HeadDim, HiddenSize, rng, iq3Type),
                OIq3 = QuantiseToIq3Bytes(HiddenSize, NumHeads * HeadDim, rng, iq3Type),
                GateIq3 = QuantiseToIq3Bytes(IntermediateSize, HiddenSize, rng, iq3Type),
                UpIq3 = QuantiseToIq3Bytes(IntermediateSize, HiddenSize, rng, iq3Type),
                DownIq3 = QuantiseToIq3Bytes(HiddenSize, IntermediateSize, rng, iq3Type),
            };
        }
        return b;
    }

    private static byte[] QuantiseToIq3Bytes(int outputDim, int inputDim, Random rng, QuantizationType iq3Type)
    {
        if ((inputDim % 256) != 0)
            throw new InvalidOperationException(
                $"IQ3 requires inputDim multiple of 256 (got {inputDim}). Bump fixture dims.");
        float[] src = new float[outputDim * inputDim];
        for (int i = 0; i < src.Length; i++)
            src[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
        return iq3Type == QuantizationType.IQ3_XXS
            ? Iq3Fixture.QuantizeRowsIq3Xxs(src, outputDim, inputDim)
            : Iq3Fixture.QuantizeRowsIq3S(src, outputDim, inputDim);
    }

    /// <summary>
    /// Materialises a fresh <see cref="TransformerWeights"/> from a pre-staged IQ3
    /// blob — every nint pointer is a brand-new 64-byte-aligned allocation owned by
    /// the returned weights instance (freed via <c>ownedAllocations</c> on
    /// disposal). Callers can build a CPU and a Vulkan model from two
    /// materialisations of the same blob without double-free risk.
    /// </summary>
    private static unsafe TransformerWeights MaterialiseWeights(
        Iq3StagedBlob blob, ModelConfig config, QuantizationType iq3Type)
    {
        var owned = new List<nint>();

        nint tokenEmbed = CopyF32ToNative(blob.TokenEmbedF32, owned);
        nint lmHead = CopyBytesToNative(blob.LmHeadIq3, owned);

        var layers = new TransformerLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            var lb = blob.Layers[i];
            int qOut = NumHeads * HeadDim;
            int kvOut = NumKvHeads * HeadDim;
            layers[i] = new TransformerLayerWeights(
                attnNormWeight: (float[])lb.AttnNormWeight.Clone(),
                qWeight: CopyBytesToNative(lb.QIq3, owned), qQuantType: iq3Type, qOutputDim: qOut, qInputDim: HiddenSize,
                kWeight: CopyBytesToNative(lb.KIq3, owned), kQuantType: iq3Type, kOutputDim: kvOut, kInputDim: HiddenSize,
                vWeight: CopyBytesToNative(lb.VIq3, owned), vQuantType: iq3Type, vOutputDim: kvOut, vInputDim: HiddenSize,
                oWeight: CopyBytesToNative(lb.OIq3, owned), oQuantType: iq3Type, oOutputDim: HiddenSize, oInputDim: qOut,
                ffnNormWeight: (float[])lb.FfnNormWeight.Clone(),
                gateWeight: CopyBytesToNative(lb.GateIq3, owned), gateQuantType: iq3Type, gateOutputDim: IntermediateSize, gateInputDim: HiddenSize,
                upWeight: CopyBytesToNative(lb.UpIq3, owned), upQuantType: iq3Type, upOutputDim: IntermediateSize, upInputDim: HiddenSize,
                downWeight: CopyBytesToNative(lb.DownIq3, owned), downQuantType: iq3Type, downOutputDim: HiddenSize, downInputDim: IntermediateSize);
        }

        return TransformerWeights.CreateFromSafetensors(
            tokenEmbedWeight: tokenEmbed, tokenEmbedQt: QuantizationType.F32,
            vocabSize: VocabSize, hiddenSize: HiddenSize,
            layers: layers,
            outputNormWeight: (float[])blob.OutputNormWeight.Clone(),
            outputWeight: lmHead, outputQt: iq3Type, outputM: VocabSize, outputK: HiddenSize,
            ownedAllocations: owned);
    }

    private static unsafe nint CopyF32ToNative(float[] src, List<nint> owned)
    {
        long bytes = (long)src.Length * sizeof(float);
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        owned.Add(ptr);
        src.AsSpan().CopyTo(new Span<float>((void*)ptr, src.Length));
        return ptr;
    }

    private static unsafe nint CopyBytesToNative(byte[] src, List<nint> owned)
    {
        long bytes = src.Length;
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)bytes, 64);
        owned.Add(ptr);
        src.AsSpan().CopyTo(new Span<byte>((void*)ptr, src.Length));
        return ptr;
    }

    private static float[] FillF32Array(int count, Random rng, float amplitude)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
        return arr;
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
            $"{qt} dense seqLen={seqLen}: logits stddev near zero " +
            $"(var={var:E3}) — likely degenerate output.");
    }

    private static float[] FillNormVec(int count, Random rng)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
        return arr;
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }

    private static ModelConfig BuildConfig()
    {
        var rope = new RoPEConfig(Theta: 10000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm);
        return new ModelConfig
        {
            Architecture = Architecture.Llama,
            VocabSize = VocabSize,
            HiddenSize = HiddenSize,
            IntermediateSize = IntermediateSize,
            NumLayers = NumLayers,
            NumAttentionHeads = NumHeads,
            NumKvHeads = NumKvHeads,
            HeadDim = HeadDim,
            MaxSequenceLength = MaxSeqLen,
            AttentionType = AttentionType.GQA,
            PositionEncodingType = PositionEncodingType.RoPE,
            RoPEConfig = rope,
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            ChatTemplate = null,
        };
    }
}
