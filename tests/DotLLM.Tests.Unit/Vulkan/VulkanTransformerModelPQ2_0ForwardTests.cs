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
/// End-to-end parity test for the Vulkan dense transformer forward path with PQ2_0
/// (PrismML Bonsai ternary) projection-weight upload — issue #205. Proves the
/// <see cref="VulkanWeights.KeepPQ2_0OnDevice"/>-equivalent upload gate and the
/// <see cref="VulkanTransformerModel"/> <c>RecordMatmul</c> PQ2_0 dispatch branch are
/// correctly wired end-to-end (weight upload -> GEMV dispatch -> logits), not just the
/// isolated <c>MatMulPQ2_0GemvF32Kernel</c> in <c>VulkanMatMulPQ2_0GemvF32KernelTests</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Method.</b> Builds a synthetic dense Llama-style "model" where every projection
/// (Q/K/V/O attention + Gate/Up/Down FFN + LM head) is generated directly in the PQ2_0
/// on-disk byte layout — random ternary codes with a random per-(row,group) fp16 scale,
/// packed exactly as <see cref="MatMulPQ2_0GemvF32Kernel"/>'s doc comment specifies. Both
/// backends consume the identical bytes: the CPU oracle dequantises via
/// <c>Dequantize.DequantizePQ2_0</c> on the fly, Vulkan keeps the raw bytes on device and
/// dispatches <c>matmul_pq2_0_f32_gemv.comp</c>. No separate float source is needed since
/// both backends derive their view of the weights from the same packed bytes.
/// </para>
/// <para>
/// <b>Decode-only scope.</b> Only <c>seqLen == 1</c> (single-token decode) is exercised —
/// PQ2_0 GEMM/prefill is not yet implemented on Vulkan (#205 explicit follow-on); the
/// dispatcher throws <see cref="NotSupportedException"/> for <c>seqLen &gt; 1</c> today.
/// </para>
/// <para>
/// <b>Dimensions.</b> All PQ2_0 contraction axes are bumped to multiples of 128 (the PQ2_0
/// group size): <c>HiddenSize = 256</c>, <c>IntermediateSize = 256</c>,
/// <c>NumHeads * HeadDim = 256</c> (NumHeads=4, HeadDim=64).
/// </para>
/// <para>
/// <b>Tolerance.</b> abs 1e-1 / rel 1e-1 — same generous envelope as the IQ3/I2_S dense
/// forward parity tests; the discriminator is "Vulkan dispatch matches CPU dequant", which
/// would diverge by orders of magnitude if the PQ2_0 upload gate or dispatch branch were
/// miswired (e.g. group scale applied once at the end instead of per-group, or a byte
/// offset off by the 2-byte scale header).
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelPQ2_0ForwardTests : IDisposable
{
    private const int HiddenSize = 256;
    private const int IntermediateSize = 256;
    private const int VocabSize = 8;
    private const int NumLayers = 2;
    private const int NumHeads = 4;
    private const int NumKvHeads = 4;       // GQA repeat factor 1
    private const int HeadDim = 64;          // NumHeads * HeadDim = 256
    private const int MaxSeqLen = 8;
    private const int GroupSize = 128;
    private const int GroupBytes = 34;

    private const float AbsTol = 1e-1f;
    private const float RelTol = 1e-1f;

    // Note: allocations are passed to TransformerWeights via ownedAllocations and
    // freed by TransformerWeights.Dispose() (called from TransformerModel.Dispose()).
    public void Dispose() { }

    [SkippableFact]
    public void Forward_PQ2_0_SingleToken_MatchesCpuReference()
        => AssertVulkanMatchesCpu(seed: 205);

    private void AssertVulkanMatchesCpu(int seed)
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        ModelConfig config = BuildConfig();
        var blob = StagePQ2_0Blob(seed);

        const int seqLen = 1;
        int[] tokenIds = new int[seqLen];
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        // ── CPU oracle ────────────────────────────────────────────────
        float[] cpuLogits;
        {
            TransformerWeights cpuWeights = MaterialiseWeights(blob, config);
            using var model = TransformerModel.BuildFromPrebuiltWeights(cpuWeights, config);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        // ── Vulkan under test ─────────────────────────────────────────
        float[] vkLogits;
        {
            using TransformerWeights vkWeights = MaterialiseWeights(blob, config);
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
            Assert.True(float.IsFinite(cpu), $"non-finite CPU logit col={c}: {cpu}");
            Assert.True(float.IsFinite(vk), $"non-finite Vulkan logit col={c}: {vk}");
            float diff = MathF.Abs(cpu - vk);
            float bar = AbsTol + RelTol * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"PQ2_0 dense seqLen={seqLen}, col={c}: " +
                $"cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }

        AssertNonDegenerate(vkLogits);
    }

    /// <summary>Per-layer PQ2_0 byte blobs + per-vector F32 norm buffers, generated once
    /// and consumed twice (CPU + Vulkan) via <see cref="MaterialiseWeights"/>.</summary>
    private sealed class PQ2_0StagedBlob
    {
        public float[] TokenEmbedF32 = null!;
        public byte[] LmHeadPQ2_0 = null!;
        public float[] OutputNormWeight = null!;
        public LayerBlob[] Layers = null!;

        public sealed class LayerBlob
        {
            public float[] AttnNormWeight = null!;
            public float[] FfnNormWeight = null!;
            public byte[] QPQ2_0 = null!;
            public byte[] KPQ2_0 = null!;
            public byte[] VPQ2_0 = null!;
            public byte[] OPQ2_0 = null!;
            public byte[] GatePQ2_0 = null!;
            public byte[] UpPQ2_0 = null!;
            public byte[] DownPQ2_0 = null!;
        }
    }

    private static PQ2_0StagedBlob StagePQ2_0Blob(int seed)
    {
        var rng = new Random(seed);
        var b = new PQ2_0StagedBlob
        {
            TokenEmbedF32 = FillF32Array(VocabSize * HiddenSize, rng, 0.05f),
            OutputNormWeight = FillNormVec(HiddenSize, rng),
            LmHeadPQ2_0 = QuantiseToPQ2_0Bytes(VocabSize, HiddenSize, rng),
            Layers = new PQ2_0StagedBlob.LayerBlob[NumLayers],
        };
        for (int i = 0; i < NumLayers; i++)
        {
            b.Layers[i] = new PQ2_0StagedBlob.LayerBlob
            {
                AttnNormWeight = FillNormVec(HiddenSize, rng),
                FfnNormWeight = FillNormVec(HiddenSize, rng),
                QPQ2_0 = QuantiseToPQ2_0Bytes(NumHeads * HeadDim, HiddenSize, rng),
                KPQ2_0 = QuantiseToPQ2_0Bytes(NumKvHeads * HeadDim, HiddenSize, rng),
                VPQ2_0 = QuantiseToPQ2_0Bytes(NumKvHeads * HeadDim, HiddenSize, rng),
                OPQ2_0 = QuantiseToPQ2_0Bytes(HiddenSize, NumHeads * HeadDim, rng),
                GatePQ2_0 = QuantiseToPQ2_0Bytes(IntermediateSize, HiddenSize, rng),
                UpPQ2_0 = QuantiseToPQ2_0Bytes(IntermediateSize, HiddenSize, rng),
                DownPQ2_0 = QuantiseToPQ2_0Bytes(HiddenSize, IntermediateSize, rng),
            };
        }
        return b;
    }

    /// <summary>
    /// Generates a row-major <c>[outputDim, inputDim]</c> matrix directly in the PQ2_0 byte
    /// layout: random ternary codes {-1,0,+1} with a random fp16 scale per (row, 128-element
    /// group), packed exactly as <see cref="MatMulPQ2_0GemvF32Kernel"/> decodes (2-byte
    /// little-endian fp16 scale + 32 packed code bytes per 34-byte group; byte <c>gp</c>
    /// packs group-relative positions {gp, gp+32, gp+64, gp+96} at bit offsets {6,4,2,0}).
    /// No separate F32 source is needed — CPU and Vulkan both derive their view directly from
    /// these bytes.
    /// </summary>
    private static byte[] QuantiseToPQ2_0Bytes(int outputDim, int inputDim, Random rng)
    {
        if ((inputDim % GroupSize) != 0)
            throw new InvalidOperationException(
                $"PQ2_0 requires inputDim multiple of {GroupSize} (got {inputDim}). Bump fixture dims.");

        int groups = inputDim / GroupSize;
        int rowBytes = groups * GroupBytes;
        byte[] buf = new byte[(long)outputDim * rowBytes];

        for (int r = 0; r < outputDim; r++)
        {
            int rowByteBase = r * rowBytes;
            for (int g = 0; g < groups; g++)
            {
                int groupByteBase = rowByteBase + g * GroupBytes;
                Half scale = (Half)(((float)rng.NextDouble() * 0.5f + 0.5f) * 0.05f);
                BitConverter.GetBytes(scale).CopyTo(buf, groupByteBase);
                int codeBase = groupByteBase + 2;
                for (int p = 0; p < GroupSize; p++)
                {
                    int code = rng.Next(3);                 // {0,1,2} -> {-1,0,+1}
                    int byteInGroup = p % 32;
                    int shift = 6 - 2 * (p / 32);
                    buf[codeBase + byteInGroup] |= (byte)(code << shift);
                }
            }
        }
        return buf;
    }

    private static unsafe TransformerWeights MaterialiseWeights(PQ2_0StagedBlob blob, ModelConfig config)
    {
        var owned = new List<nint>();

        nint tokenEmbed = CopyF32ToNative(blob.TokenEmbedF32, owned);
        nint lmHead = CopyBytesToNative(blob.LmHeadPQ2_0, owned);

        var layers = new TransformerLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            var lb = blob.Layers[i];
            int qOut = NumHeads * HeadDim;
            int kvOut = NumKvHeads * HeadDim;
            layers[i] = new TransformerLayerWeights(
                attnNormWeight: (float[])lb.AttnNormWeight.Clone(),
                qWeight: CopyBytesToNative(lb.QPQ2_0, owned), qQuantType: QuantizationType.PQ2_0, qOutputDim: qOut, qInputDim: HiddenSize,
                kWeight: CopyBytesToNative(lb.KPQ2_0, owned), kQuantType: QuantizationType.PQ2_0, kOutputDim: kvOut, kInputDim: HiddenSize,
                vWeight: CopyBytesToNative(lb.VPQ2_0, owned), vQuantType: QuantizationType.PQ2_0, vOutputDim: kvOut, vInputDim: HiddenSize,
                oWeight: CopyBytesToNative(lb.OPQ2_0, owned), oQuantType: QuantizationType.PQ2_0, oOutputDim: HiddenSize, oInputDim: qOut,
                ffnNormWeight: (float[])lb.FfnNormWeight.Clone(),
                gateWeight: CopyBytesToNative(lb.GatePQ2_0, owned), gateQuantType: QuantizationType.PQ2_0, gateOutputDim: IntermediateSize, gateInputDim: HiddenSize,
                upWeight: CopyBytesToNative(lb.UpPQ2_0, owned), upQuantType: QuantizationType.PQ2_0, upOutputDim: IntermediateSize, upInputDim: HiddenSize,
                downWeight: CopyBytesToNative(lb.DownPQ2_0, owned), downQuantType: QuantizationType.PQ2_0, downOutputDim: HiddenSize, downInputDim: IntermediateSize);
        }

        return TransformerWeights.CreateFromSafetensors(
            tokenEmbedWeight: tokenEmbed, tokenEmbedQt: QuantizationType.F32,
            vocabSize: VocabSize, hiddenSize: HiddenSize,
            layers: layers,
            outputNormWeight: (float[])blob.OutputNormWeight.Clone(),
            outputWeight: lmHead, outputQt: QuantizationType.PQ2_0, outputM: VocabSize, outputK: HiddenSize,
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

    private static void AssertNonDegenerate(float[] logits)
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
            $"PQ2_0 dense: logits stddev near zero (var={var:E3}) — likely degenerate output.");
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
