using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Single-device parity proof for the Vulkan pipeline-parallel (layer-spanning) primitives (#366):
/// a model split into stage-0 (layers <c>[0..K)</c>) + stage-1 (layers <c>[K..L)</c> + final norm + LM
/// head) must produce logits identical to the un-split full model. Stage-1 holds a windowed device-weight
/// upload (<c>VulkanWeights.Upload firstLayer=K</c>) and resumes from the hidden state handed off by
/// stage-0 (<c>DownloadHiddenState</c> → <c>ForwardFromHidden</c>), with its CPU-side per-layer lookups
/// offset by <c>_firstLayer</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Single-device on purpose.</b> This isolates the hard correctness deltas — windowed weights, the
/// <c>_firstLayer</c> offset (cpu weights + per-layer sliding window), and resume-from-hidden — WITHOUT
/// the cross-device handoff, so it runs on any single Vulkan GPU (incl. the Strix Halo iGPU). The actual
/// two-device run (prefill on GPU0 → host → decode on GPU1) is <c>VulkanCrossDevicePipelineParityTests</c>
/// on the Framework box.
/// </para>
/// <para>
/// <b>Tight tolerance.</b> Both the full model and both stages run the identical Vulkan FP32 kernels; the
/// only difference is the hidden state round-tripping through host FP32 between layers K-1 and K, which is
/// lossless. So logits should match to near-bit-exact — a tight band (2e-3 abs / 1e-2 rel) discriminates a
/// real off-by-layer bug (which shifts logits by ~0.5+) from FP32 dispatch-order noise.
/// </para>
/// <para>
/// <b>Split parameterisation</b> {1, 2, 3} over a 4-layer fixture mirrors <c>HybridVulkanCudaParityTests</c>:
/// split=1 catches missing-offset bugs where local index 0 ≡ global 0 would mask a reindex error; split=3
/// exercises the single-stage-1-layer edge and the <c>firstLayer=3</c> device-upload window.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanPipelineParityTests
{
    private readonly ITestOutputHelper _out;
    public VulkanPipelineParityTests(ITestOutputHelper output) => _out = output;

    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 16;
    private const int IntermediateSize = 32;
    private const int NumLayers = 4;
    private const int MaxSeqLen = 8;

    public static TheoryData<int> SplitPoints => new() { 1, 2, 3 };

    private const float AbsTol = 2e-3f;
    private const float RelTol = 1e-2f;

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineSplit_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();
        _out.WriteLine($"device: {device.DeviceName}; split at layer {splitAt} of {NumLayers}");

        float[] full = RunFull(device, fixture, tokenIds, positions, spvDir!);
        float[] split = RunSplitPrefill(device, device, fixture, tokenIds, positions, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"prefill/split={splitAt}");
    }

    /// <summary>
    /// Cross-device spanning: stage-0 on physical device 0, stage-1 on physical device 1, hidden state
    /// handed off device0 → host → device1. Logits must match the single-device full model. Gated on ≥2
    /// Vulkan devices (skips on single-GPU boxes); runs on the Framework RTX 3060 + Intel Arc pair.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineSplit_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2, "Cross-device spanning needs >= 2 Vulkan devices.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"stage0 device 0: {dev0.DeviceName}; stage1 device 1: {dev1.DeviceName}; split at {splitAt}/{NumLayers}");

        float[] full = RunFull(dev0, fixture, tokenIds, positions, spvDir!);
        float[] split = RunSplitPrefill(dev0, dev1, fixture, tokenIds, positions, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"xdev-prefill/split={splitAt}");
    }

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineSplit_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] prefillIds = [3, 1, 4, 2];
        int[] prefillPos = [0, 1, 2, 3];
        int[] decodeIds = [5];
        int[] decodePos = [4];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var device = VulkanDevice.Create();
        _out.WriteLine($"device: {device.DeviceName}; split at layer {splitAt} of {NumLayers} (decode)");

        float[] full = RunFullDecode(device, fixture, prefillIds, prefillPos, decodeIds, decodePos, spvDir!);
        float[] split = RunSplitDecode(device, device, fixture, prefillIds, prefillPos, decodeIds, decodePos, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"decode/split={splitAt}");
    }

    /// <summary>
    /// Cross-device decode spanning: prefill + one decode step with stage-0 on device 0 and stage-1 on
    /// device 1 (each with its own device-local KV cache). Gated on ≥2 Vulkan devices.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineSplit_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(VulkanDevice.IsAvailable(), "No Vulkan loader/driver available.");
        Skip.If(VulkanDevice.PhysicalDeviceCount() < 2, "Cross-device spanning needs >= 2 Vulkan devices.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] prefillIds = [3, 1, 4, 2];
        int[] prefillPos = [0, 1, 2, 3];
        int[] decodeIds = [5];
        int[] decodePos = [4];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var dev0 = VulkanDevice.Create(0);
        using var dev1 = VulkanDevice.Create(1);
        _out.WriteLine($"stage0 device 0: {dev0.DeviceName}; stage1 device 1: {dev1.DeviceName}; split at {splitAt}/{NumLayers} (decode)");

        float[] full = RunFullDecode(dev0, fixture, prefillIds, prefillPos, decodeIds, decodePos, spvDir!);
        float[] split = RunSplitDecode(dev0, dev1, fixture, prefillIds, prefillPos, decodeIds, decodePos, splitAt, spvDir!);

        AssertLogitsMatch(full, split, $"xdev-decode/split={splitAt}");
    }

    // ── Runners ──────────────────────────────────────────────────────────────

    private static float[] RunFull(VulkanDevice device, DenseFixture fx, int[] ids, int[] pos, string spvDir)
    {
        using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, fx.Config, fx.Weights, spvDir);
        using ITensor logits = model.Forward(ids, pos, deviceId: 0, kvCache: null);
        return LastRow(logits);
    }

    private static float[] RunSplitPrefill(
        VulkanDevice stage0Device, VulkanDevice stage1Device,
        DenseFixture fx, int[] ids, int[] pos, int splitAt, string spvDir)
    {
        int seqLen = ids.Length;

        // Stage 0: layers [0..splitAt) on stage0Device. Run the full forward, discard logits, read hidden.
        var cfg0 = fx.Config with { NumLayers = splitAt };
        using var stage0 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage0Device, cfg0, fx.Weights, spvDir, firstLayer: 0);
        using (var _ = stage0.Forward(ids, pos, deviceId: 0, kvCache: null)) { }
        using ITensor hidden = stage0.DownloadHiddenState(seqLen); // device0 → host

        // Stage 1: layers [splitAt..L) + final norm + LM head on stage1Device, resumed from the host hidden.
        var cfg1 = fx.Config with { NumLayers = NumLayers - splitAt };
        using var stage1 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage1Device, cfg1, fx.Weights, spvDir, firstLayer: splitAt);
        var hiddenSpan = new ReadOnlySpan<float>((void*)hidden.DataPointer, seqLen * HiddenSize);
        using ITensor logits = stage1.ForwardFromHidden(hiddenSpan, pos, kvCache: null); // host → device1
        return LastRow(logits);
    }

    private static float[] RunFullDecode(
        VulkanDevice device, DenseFixture fx, int[] preIds, int[] prePos, int[] decIds, int[] decPos, string spvDir)
    {
        using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, fx.Config, fx.Weights, spvDir);
        using var kv = model.CreateKvCache(MaxSeqLen);
        using (var _ = model.Forward(preIds, prePos, deviceId: 0, kv)) { }
        using ITensor logits = model.Forward(decIds, decPos, deviceId: 0, kv);
        return LastRow(logits);
    }

    private static float[] RunSplitDecode(
        VulkanDevice stage0Device, VulkanDevice stage1Device,
        DenseFixture fx, int[] preIds, int[] prePos, int[] decIds, int[] decPos,
        int splitAt, string spvDir)
    {
        var cfg0 = fx.Config with { NumLayers = splitAt };
        var cfg1 = fx.Config with { NumLayers = NumLayers - splitAt };
        using var stage0 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage0Device, cfg0, fx.Weights, spvDir, firstLayer: 0);
        using var stage1 = VulkanTransformerModel.BuildFromPrebuiltWeights(stage1Device, cfg1, fx.Weights, spvDir, firstLayer: splitAt);
        using var kv0 = stage0.CreateKvCache(MaxSeqLen); // K layers, local-indexed
        using var kv1 = stage1.CreateKvCache(MaxSeqLen); // L-K layers, local-indexed

        // Prefill through both stages.
        using (var _ = stage0.Forward(preIds, prePos, deviceId: 0, kv0)) { }
        using (ITensor h0 = stage0.DownloadHiddenState(preIds.Length))
        {
            var span0 = new ReadOnlySpan<float>((void*)h0.DataPointer, preIds.Length * HiddenSize);
            using var __ = stage1.ForwardFromHidden(span0, prePos, kv1);
        }

        // Decode one token through both stages.
        using (var _ = stage0.Forward(decIds, decPos, deviceId: 0, kv0)) { }
        using ITensor h1 = stage0.DownloadHiddenState(decIds.Length);
        var span1 = new ReadOnlySpan<float>((void*)h1.DataPointer, decIds.Length * HiddenSize);
        using ITensor logits = stage1.ForwardFromHidden(span1, decPos, kv1);
        return LastRow(logits);
    }

    private static float[] LastRow(ITensor logits)
    {
        // VulkanTransformerModel returns [rows, vocab]; take the last row.
        int rows = logits.Shape[0];
        int vocab = logits.Shape[1];
        Assert.Equal(VocabSize, vocab);
        int offset = (rows - 1) * vocab;
        return new ReadOnlySpan<float>((void*)(logits.DataPointer + (nint)offset * sizeof(float)), vocab).ToArray();
    }

    private void AssertLogitsMatch(float[] reference, float[] split, string variant)
    {
        Assert.Equal(reference.Length, split.Length);
        float maxAbs = 0f;
        for (int c = 0; c < reference.Length; c++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[c] - split[c]));
        _out.WriteLine($"[{variant}] max|diff|={maxAbs:E3} (AbsTol={AbsTol:E3})");

        for (int c = 0; c < reference.Length; c++)
        {
            float refVal = reference[c], splitVal = split[c];
            Assert.True(float.IsFinite(refVal) && float.IsFinite(splitVal),
                $"{variant} col={c}: non-finite logit (ref={refVal}, split={splitVal}).");
            float diff = MathF.Abs(refVal - splitVal);
            float bar = AbsTol + RelTol * MathF.Abs(refVal);
            Assert.True(diff <= bar,
                $"{variant} col={c}: ref={refVal:F6} vs split={splitVal:F6} (|diff|={diff:E3} > {bar:E3}).");
        }
    }

    private static string? FindSpvDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "spv"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "vulkan", "spv"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0)
                return full;
        }
        return null;
    }

    // ── Synthetic 4-layer dense fixture (mirrors HybridVulkanCudaParityTests.DenseFixture) ──
    private sealed unsafe class DenseFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public TransformerWeights Weights = null!;

        public static DenseFixture Build(int seed, RoPEType ropeType)
        {
            var b = new DenseFixture();
            b.BuildInternal(seed, ropeType);
            return b;
        }

        private void BuildInternal(int seed, RoPEType ropeType)
        {
            var rng = new Random(seed);
            Config = new ModelConfig
            {
                Architecture = Architecture.Llama,
                VocabSize = VocabSize,
                HiddenSize = HiddenSize,
                IntermediateSize = IntermediateSize,
                NumLayers = NumLayers,
                NumAttentionHeads = NumAttentionHeads,
                NumKvHeads = NumKvHeads,
                HeadDim = HeadDim,
                MaxSequenceLength = MaxSeqLen,
                AttentionType = AttentionType.GQA,
                PositionEncodingType = PositionEncodingType.RoPE,
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: RopeDim, Type: ropeType),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                ChatTemplate = null,
            };

            nint tokenEmbed = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);
            float[] outputNorm = FillNormVec(HiddenSize, rng);
            nint output = AllocFloatsUniform(VocabSize * HiddenSize, rng, 0.05f);

            int qOut = NumAttentionHeads * HeadDim;
            int kvOut = NumKvHeads * HeadDim;
            int oIn = NumAttentionHeads * HeadDim;

            var layers = new TransformerLayerWeights[NumLayers];
            for (int i = 0; i < NumLayers; i++)
            {
                float[] attnNorm = FillNormVec(HiddenSize, rng);
                float[] ffnNorm = FillNormVec(HiddenSize, rng);
                nint qW = AllocFloatsUniform(qOut * HiddenSize, rng, 0.05f);
                nint kW = AllocFloatsUniform(kvOut * HiddenSize, rng, 0.05f);
                nint vW = AllocFloatsUniform(kvOut * HiddenSize, rng, 0.05f);
                nint oW = AllocFloatsUniform(HiddenSize * oIn, rng, 0.05f);
                nint gateW = AllocFloatsUniform(IntermediateSize * HiddenSize, rng, 0.05f);
                nint upW = AllocFloatsUniform(IntermediateSize * HiddenSize, rng, 0.05f);
                nint downW = AllocFloatsUniform(HiddenSize * IntermediateSize, rng, 0.05f);

                layers[i] = new TransformerLayerWeights(
                    attnNormWeight: attnNorm,
                    qWeight: qW, qQuantType: QuantizationType.F32, qOutputDim: qOut, qInputDim: HiddenSize,
                    kWeight: kW, kQuantType: QuantizationType.F32, kOutputDim: kvOut, kInputDim: HiddenSize,
                    vWeight: vW, vQuantType: QuantizationType.F32, vOutputDim: kvOut, vInputDim: HiddenSize,
                    oWeight: oW, oQuantType: QuantizationType.F32, oOutputDim: HiddenSize, oInputDim: oIn,
                    ffnNormWeight: ffnNorm,
                    gateWeight: gateW, gateQuantType: QuantizationType.F32, gateOutputDim: IntermediateSize, gateInputDim: HiddenSize,
                    upWeight: upW, upQuantType: QuantizationType.F32, upOutputDim: IntermediateSize, upInputDim: HiddenSize,
                    downWeight: downW, downQuantType: QuantizationType.F32, downOutputDim: HiddenSize, downInputDim: IntermediateSize);
            }

            Weights = TransformerWeights.CreateFromSafetensors(
                tokenEmbedWeight: tokenEmbed, tokenEmbedQt: QuantizationType.F32,
                vocabSize: VocabSize, hiddenSize: HiddenSize,
                layers: layers,
                outputNormWeight: outputNorm,
                outputWeight: output, outputQt: QuantizationType.F32,
                outputM: VocabSize, outputK: HiddenSize,
                ownedAllocations: new List<nint>());
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

        private static float[] FillNormVec(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++)
                arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return arr;
        }

        public void Dispose()
        {
            Weights?.Dispose();
            foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
