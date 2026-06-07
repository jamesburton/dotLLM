using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Models.SafeTensors;
using DotLLM.Tests.Unit.Models.SafeTensors;
using DotLLM.Vulkan;
using Xunit;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>End-to-end coverage for the grouped F16 coopmat MoE path.</summary>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed class VulkanTransformerModelMoeF16GroupedForwardTests : IDisposable
{
    private const int HiddenSize = 32;
    private const int NumLayers = 1;
    private const int NumHeads = 4;
    private const int NumKvHeads = 2;
    private const int HeadDim = 8;
    private const int VocabSize = 8;
    private const int IntermediateSize = 32;
    private const int NumExperts = 4;
    private const int TopK = 2;

    private readonly string _scratch;
    private readonly List<nint> _allocs = new();

    public VulkanTransformerModelMoeF16GroupedForwardTests()
    {
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-moe-f16-vk-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public unsafe void Dispose()
    {
        foreach (nint ptr in _allocs)
            NativeMemory.AlignedFree((void*)ptr);
        _allocs.Clear();
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    [SkippableFact]
    public void Forward_F16GroupedExperts_Prefill_MatchesCpuReference()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);

        using var device = VulkanDevice.Create();
        Skip.IfNot(device.HasCooperativeMatrix, "VK_KHR_cooperative_matrix not available on this Vulkan device.");

        const int seqLen = 5;
        string path = Path.Combine(_scratch, "moe-f16-grouped.safetensors");
        WriteFixture(path, seed: 171);
        ModelConfig config = BuildConfig();

        int[] tokenIds = Enumerable.Range(0, seqLen).Select(i => i % VocabSize).ToArray();
        int[] positions = Enumerable.Range(0, seqLen).ToArray();

        float[] cpuLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            var weights = TransformerWeightsSafetensorsLoader.Load(sf, config);
            ApplyF16RoutedExpertRawViews(weights);
            using var model = TransformerModel.BuildFromPrebuiltWeights(weights, config);
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            cpuLogits = CopyLogits(logits);
        }

        float[] vkLogits;
        {
            using var sf = SafetensorsFile.Open(path);
            var weights = TransformerWeightsSafetensorsLoader.Load(sf, config);
            ApplyF16RoutedExpertRawViews(weights);
            using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, config, weights, spvDir);
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
            float bar = 7e-3f + 2e-3f * MathF.Abs(cpu);
            Assert.True(diff <= bar,
                $"col={c}: cpu={cpu:F6} vs vulkan={vk:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    private unsafe void ApplyF16RoutedExpertRawViews(TransformerWeights weights)
    {
        for (int layer = 0; layer < weights.Layers.Length; layer++)
        {
            var old = weights.Layers[layer];
            var moe = old.Moe!;

            nint[] w1 = new nint[NumExperts];
            nint[] w2 = new nint[NumExperts];
            nint[] w3 = new nint[NumExperts];

            nint w1Raw = BuildF16BankAndRoundedF32(moe.W1, w1, IntermediateSize, HiddenSize);
            nint w2Raw = BuildF16BankAndRoundedF32(moe.W2, w2, HiddenSize, IntermediateSize);
            nint w3Raw = BuildF16BankAndRoundedF32(moe.W3, w3, IntermediateSize, HiddenSize);

            var f16Moe = new MoeLayerWeights(
                moe.Gate, w1, w2, w3,
                moe.NumExperts, moe.NumExpertsPerTok, moe.HiddenSize, moe.IntermediateSize,
                moe.NormTopKProb,
                Array.Empty<nint>(), Array.Empty<nint>(), Array.Empty<nint>(),
                sharedIntermediateSize: 0, sharedExpertGate: null,
                gateExpsRaw: w1Raw, gateExpsRawQt: QuantizationType.F16,
                gateExpsMDim: IntermediateSize, gateExpsKDim: HiddenSize,
                upExpsRaw: w3Raw, upExpsRawQt: QuantizationType.F16,
                upExpsMDim: IntermediateSize, upExpsKDim: HiddenSize,
                downExpsRaw: w2Raw, downExpsRawQt: QuantizationType.F16,
                downExpsMDim: HiddenSize, downExpsKDim: IntermediateSize,
                sharedGateRaw: Array.Empty<nint>(), sharedGateRawQt: QuantizationType.F32,
                sharedUpRaw: Array.Empty<nint>(), sharedUpRawQt: QuantizationType.F32,
                sharedDownRaw: Array.Empty<nint>(), sharedDownRawQt: QuantizationType.F32);

            weights.Layers[layer] = new TransformerLayerWeights(
                old.AttnNormWeight,
                old.QWeight, old.QQuantType, old.QOutputDim, old.QInputDim,
                old.KWeight, old.KQuantType, old.KOutputDim, old.KInputDim,
                old.VWeight, old.VQuantType, old.VOutputDim, old.VInputDim,
                old.OWeight, old.OQuantType, old.OOutputDim, old.OInputDim,
                old.FfnNormWeight,
                old.GateWeight, old.GateQuantType, old.GateOutputDim, old.GateInputDim,
                old.UpWeight, old.UpQuantType, old.UpOutputDim, old.UpInputDim,
                old.DownWeight, old.DownQuantType, old.DownOutputDim, old.DownInputDim,
                old.QBias, old.KBias, old.VBias, old.OBias,
                old.GateBias, old.UpBias, old.DownBias,
                old.QNormWeight, old.KNormWeight,
                moe: f16Moe,
                mla: old.Mla);
        }
    }

    private unsafe nint BuildF16BankAndRoundedF32(nint[] srcF32, nint[] roundedF32, int rows, int cols)
    {
        long elemsPerExpert = (long)rows * cols;
        long bytesPerExpert = elemsPerExpert * sizeof(ushort);
        long totalBytes = bytesPerExpert * NumExperts;
        nint raw = (nint)NativeMemory.AlignedAlloc((nuint)totalBytes, 64);
        _allocs.Add(raw);

        for (int expert = 0; expert < NumExperts; expert++)
        {
            var src = new ReadOnlySpan<float>((void*)srcF32[expert], checked((int)elemsPerExpert));
            byte[] f16 = F16Bf16Fixture.QuantizeRowsF16(src.ToArray(), rows, cols);
            Marshal.Copy(f16, 0, raw + (nint)(expert * bytesPerExpert), f16.Length);

            nint rounded = (nint)NativeMemory.AlignedAlloc((nuint)(elemsPerExpert * sizeof(float)), 64);
            _allocs.Add(rounded);
            float[] decoded = F16Bf16Fixture.DecodeF16(f16, rows, cols);
            new ReadOnlySpan<float>(decoded).CopyTo(new Span<float>((void*)rounded, checked((int)elemsPerExpert)));
            roundedF32[expert] = rounded;
        }

        return raw;
    }

    private static ModelConfig BuildConfig()
    {
        var rope = new RoPEConfig(Theta: 1_000_000.0f, DimensionCount: HeadDim, Type: RoPEType.Norm);
        return new ModelConfig
        {
            Architecture = Architecture.Mixtral,
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
            ActivationFunction = ActivationFunction.SiLU,
            NormType = NormType.RMSNorm,
            NormEpsilon = 1e-5f,
            TiedEmbeddings = false,
            Moe = new MoeConfig
            {
                NumExperts = NumExperts,
                NumExpertsPerTok = TopK,
                MoeIntermediateSize = IntermediateSize,
            },
            ChatTemplate = null,
        };
    }

    private static void WriteFixture(string path, int seed)
    {
        var b = new SafetensorsFixtureBuilder();

        AddRand(b, "model.embed_tokens.weight", [VocabSize, HiddenSize], 0.05f, seed + 0);
        AddRand(b, "model.norm.weight", [HiddenSize], 1.0f, seed + 1, center: 1.0f, jitter: 0.05f);
        AddRand(b, "lm_head.weight", [VocabSize, HiddenSize], 0.05f, seed + 2);

        const int i = 0;
        int s = seed + 100;
        string prefix = $"model.layers.{i}";

        AddRand(b, $"{prefix}.input_layernorm.weight", [HiddenSize],
            amplitude: 0.05f, seed: s + 0, center: 1.0f, jitter: 0.05f);
        AddRand(b, $"{prefix}.post_attention_layernorm.weight", [HiddenSize],
            amplitude: 0.05f, seed: s + 1, center: 1.0f, jitter: 0.05f);

        AddRand(b, $"{prefix}.self_attn.q_proj.weight", [NumHeads * HeadDim, HiddenSize], 0.05f, s + 2);
        AddRand(b, $"{prefix}.self_attn.k_proj.weight", [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 3);
        AddRand(b, $"{prefix}.self_attn.v_proj.weight", [NumKvHeads * HeadDim, HiddenSize], 0.05f, s + 4);
        AddRand(b, $"{prefix}.self_attn.o_proj.weight", [HiddenSize, NumHeads * HeadDim], 0.05f, s + 5);

        AddRand(b, $"{prefix}.block_sparse_moe.gate.weight", [NumExperts, HiddenSize], 0.1f, s + 6);
        for (int e = 0; e < NumExperts; e++)
        {
            AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w1.weight", [IntermediateSize, HiddenSize], 0.1f, s + 10 + e * 3);
            AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w2.weight", [HiddenSize, IntermediateSize], 0.1f, s + 11 + e * 3);
            AddRand(b, $"{prefix}.block_sparse_moe.experts.{e}.w3.weight", [IntermediateSize, HiddenSize], 0.1f, s + 12 + e * 3);
        }

        b.WriteTo(path);
    }

    private static void AddRand(SafetensorsFixtureBuilder b, string name, int[] shape,
                                float amplitude, int seed,
                                float center = 0.0f, float jitter = 0.0f)
    {
        long n = 1;
        for (int i = 0; i < shape.Length; i++) n *= shape[i];
        float[] values = new float[n];
        for (long i = 0; i < n; i++)
        {
            float phi = 0.61803398875f * (i + 1) + seed * 0.37f;
            float cos = MathF.Cos(phi);
            values[i] = jitter > 0f ? center + jitter * cos : amplitude * cos;
        }
        b.AddFloat32(name, shape, values);
    }

    private static unsafe float[] CopyLogits(ITensor logits)
    {
        int total = checked(logits.Shape[0] * logits.Shape[1]);
        float[] copy = new float[total];
        new ReadOnlySpan<float>((void*)logits.DataPointer, total).CopyTo(copy);
        return copy;
    }
}
