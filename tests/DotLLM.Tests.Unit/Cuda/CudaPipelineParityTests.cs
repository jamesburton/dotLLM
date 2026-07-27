using System.Runtime.InteropServices;
using DotLLM.Core.Attention;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Parity proof for the CUDA pipeline-parallel (layer-spanning) primitives (#367): a model split into
/// stage-0 (layers <c>[0..K)</c> on CUDA device 0) + stage-1 (layers <c>[K..L)</c> + final norm + LM head on
/// CUDA device 1) must produce last-token logits identical to the un-split full
/// <see cref="CudaTransformerModel"/>. Stage-1 holds a windowed device-weight upload
/// (<see cref="CudaWeights.LoadFromGguf"/> with <c>firstLayer=K</c>) and resumes from the hidden state handed
/// off device0 → host (FP32) → device1. This is the CUDA→CUDA mirror of <c>VulkanPipelineParityTests</c>.
/// </summary>
/// <remarks>
/// <para>
/// <b>Single-device by default.</b> The <c>PipelineModel_*</c> theories run both stages on CUDA device 0
/// (two distinct contexts), isolating the hard correctness deltas — windowed weights, the per-stage layer
/// reindex, and resume-from-hidden — WITHOUT needing two physical GPUs. They run on any single CUDA box
/// (e.g. the T5500 RTX 3060). The <c>CrossDevicePipelineModel_*</c> theories place stage-0 on device 0 and
/// stage-1 on device 1 and are gated on <see cref="CudaDevice.GetDeviceCount"/> ≥ 2 (Kaggle T4 ×2).
/// </para>
/// <para>
/// <b>Tight tolerance.</b> Both the full model and the two stages run the identical CUDA FP16 kernels; the
/// only difference is the hidden state round-tripping through host FP32 between layers K-1 and K, which is
/// lossless (FP32 exactly represents the FP16 boundary value). So logits match to near-bit-exact — a tight
/// band (abs 5e-3 / rel 5e-3) discriminates a real off-by-layer bug (which shifts logits by ~0.5+) from FP
/// dispatch noise.
/// </para>
/// <para>
/// <b>Split parameterisation</b> {1, 2, 3} over a 4-layer fixture: split=1 catches missing-offset bugs where
/// local index 0 ≡ global 0 would mask a reindex error; split=3 exercises the single-stage-1-layer edge and
/// the <c>firstLayer=3</c> device-upload window.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
public sealed unsafe class CudaPipelineParityTests
{
    private readonly ITestOutputHelper _out;
    public CudaPipelineParityTests(ITestOutputHelper output) => _out = output;

    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 16;
    private const int IntermediateSize = 32;
    private const int NumLayers = 4;
    private const int MaxSeqLen = 16;

    public static TheoryData<int> SplitPoints => new() { 1, 2, 3 };

    // Same kernels, lossless FP32 round-trip at the boundary → tight tolerance.
    private const float AbsTol = 5e-3f;
    private const float RelTol = 5e-3f;

    // ── Single-device (both stages on CUDA device 0) ──

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineModel_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX kernel files not found; build the CUDA native kernels.");

        int[] ids = [3, 1, 4, 2];
        int[] pos = [0, 1, 2, 3];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);

        float[] full = RunFull(fixture, ids, pos, deviceId: 0, ptxDir!);
        float[] pipe = RunPipelinePrefill(fixture, ids, pos, splitAt, dev0: 0, dev1: 0, ptxDir!);
        AssertLogitsMatch(full, pipe, $"model-prefill/split={splitAt}");
    }

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void PipelineModel_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX kernel files not found; build the CUDA native kernels.");

        int[] preIds = [3, 1, 4, 2]; int[] prePos = [0, 1, 2, 3];
        int[] decIds = [5]; int[] decPos = [4];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);

        float[] full = RunFullDecode(fixture, preIds, prePos, decIds, decPos, deviceId: 0, ptxDir!);
        float[] pipe = RunPipelineDecode(fixture, preIds, prePos, decIds, decPos, splitAt, dev0: 0, dev1: 0, ptxDir!);
        AssertLogitsMatch(full, pipe, $"model-decode/split={splitAt}");
    }

    // ── Cross-device (stage-0 on device 0, stage-1 on device 1; needs >= 2 GPUs) ──

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineModel_PrefillVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        Skip.If(CudaDevice.GetDeviceCount() < 2, "Cross-device spanning needs >= 2 CUDA GPUs (e.g. Kaggle T4 x2).");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX kernel files not found; build the CUDA native kernels.");

        int[] ids = [3, 1, 4, 2];
        int[] pos = [0, 1, 2, 3];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);

        float[] full = RunFull(fixture, ids, pos, deviceId: 0, ptxDir!);
        float[] pipe = RunPipelinePrefill(fixture, ids, pos, splitAt, dev0: 0, dev1: 1, ptxDir!);
        AssertLogitsMatch(full, pipe, $"xdev-model-prefill/split={splitAt}");
    }

    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void CrossDevicePipelineModel_DecodeVsFull_LastTokenLogitsMatch(int splitAt)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        Skip.If(CudaDevice.GetDeviceCount() < 2, "Cross-device spanning needs >= 2 CUDA GPUs (e.g. Kaggle T4 x2).");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX kernel files not found; build the CUDA native kernels.");

        int[] preIds = [3, 1, 4, 2]; int[] prePos = [0, 1, 2, 3];
        int[] decIds = [5]; int[] decPos = [4];
        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);

        float[] full = RunFullDecode(fixture, preIds, prePos, decIds, decPos, deviceId: 0, ptxDir!);
        float[] pipe = RunPipelineDecode(fixture, preIds, prePos, decIds, decPos, splitAt, dev0: 0, dev1: 1, ptxDir!);
        AssertLogitsMatch(full, pipe, $"xdev-model-decode/split={splitAt}");
    }

    // ── VRAM trim (#123, CUDA sibling of Vulkan #368): stage 1 must shed the embed table ──

    /// <summary>
    /// Discriminating check for the stage VRAM trim: stage 1 (only ever seeded from stage 0's hidden
    /// state) must NOT hold the token-embedding table on device, while stage 0 must. Guards against a
    /// silent regression that re-uploads the skipped table — invisible to the logit-parity theories.
    /// (The head half of the trim needs no test: <c>CudaWeights.LoadFromGguf</c> has always skipped the
    /// output norm + LM head for a non-final window.)
    /// </summary>
    [SkippableFact]
    public void TrimmedStage1_ShedsTokenEmbedTable()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX kernel files not found; build the CUDA native kernels.");

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        using var model = CudaPipelineTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, splitLayer: 2, device0Id: 0, device1Id: 0, ptxDir!);

        Assert.True(model.Stage0.HasTokenEmbed, "stage 0 gathers embeddings and must hold the table");
        Assert.False(model.Stage1.HasTokenEmbed, "stage 1 is hidden-seeded and must shed the table");
    }

    /// <summary>An embed-less stage entered via the embedding path must fail loudly, not launch against a null pointer.</summary>
    [SkippableFact]
    public void EmbedlessStage_EnqueueFromEmbedding_Throws()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX kernel files not found; build the CUDA native kernels.");

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);
        fixture.Weights.RepackWeights();
        using var stage = CudaPipelineStage.Build(
            fixture.Config, fixture.Weights, deviceId: 0, ptxDir!,
            firstLayer: 2, layerCount: 2, isFinalStage: true, skipTokenEmbed: true);

        Assert.False(stage.HasTokenEmbed);
        Assert.Throws<InvalidOperationException>(
            () => stage.EnqueueFromEmbedding([3, 1], [0, 1], seqLen: 2, kvCache: null));
    }

    // ── Runners ──────────────────────────────────────────────────────────────

    private static float[] RunFull(DenseFixture fx, int[] ids, int[] pos, int deviceId, string ptxDir)
    {
        using var model = CudaTransformerModel.BuildFromPrebuiltWeights(fx.Weights, fx.Config, deviceId, ptxDir);
        using ITensor logits = model.Forward(ids, pos, deviceId, kvCache: null);
        return LastTokenLogits(logits);
    }

    private static float[] RunFullDecode(
        DenseFixture fx, int[] preIds, int[] prePos, int[] decIds, int[] decPos, int deviceId, string ptxDir)
    {
        using var model = CudaTransformerModel.BuildFromPrebuiltWeights(fx.Weights, fx.Config, deviceId, ptxDir);
        using var kv = model.CreateKvCache(MaxSeqLen);
        using (var _ = model.Forward(preIds, prePos, deviceId, kv)) { }
        using ITensor logits = model.Forward(decIds, decPos, deviceId, kv);
        return LastTokenLogits(logits);
    }

    private static float[] RunPipelinePrefill(
        DenseFixture fx, int[] ids, int[] pos, int splitAt, int dev0, int dev1, string ptxDir)
    {
        using var model = CudaPipelineTransformerModel.BuildFromPrebuiltWeights(
            fx.Weights, fx.Config, splitAt, dev0, dev1, ptxDir);
        using ITensor logits = model.Forward(ids, pos, deviceId: 0, kvCache: null);
        return LastTokenLogits(logits);
    }

    private static float[] RunPipelineDecode(
        DenseFixture fx, int[] preIds, int[] prePos, int[] decIds, int[] decPos,
        int splitAt, int dev0, int dev1, string ptxDir)
    {
        using var model = CudaPipelineTransformerModel.BuildFromPrebuiltWeights(
            fx.Weights, fx.Config, splitAt, dev0, dev1, ptxDir);
        using var kv = model.CreateKvCache(MaxSeqLen);
        using (var _ = model.Forward(preIds, prePos, deviceId: 0, kv)) { }
        using ITensor logits = model.Forward(decIds, decPos, deviceId: 0, kv);
        return LastTokenLogits(logits);
    }

    private static float[] LastTokenLogits(ITensor logits)
    {
        // CudaTransformerModel and CudaPipelineTransformerModel both return [1, vocab] (last token only).
        return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
    }

    private void AssertLogitsMatch(float[] reference, float[] actual, string variant)
    {
        Assert.Equal(reference.Length, actual.Length);
        float maxAbs = 0f;
        for (int c = 0; c < reference.Length; c++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[c] - actual[c]));
        _out.WriteLine($"[{variant}] max|diff|={maxAbs:E3} (AbsTol={AbsTol:E3})");

        for (int c = 0; c < reference.Length; c++)
        {
            float refVal = reference[c], actVal = actual[c];
            Assert.True(float.IsFinite(refVal) && float.IsFinite(actVal),
                $"{variant} col={c}: non-finite logit (ref={refVal}, actual={actVal}).");
            float bar = AbsTol + RelTol * MathF.Abs(refVal);
            Assert.True(MathF.Abs(refVal - actVal) <= bar,
                $"{variant} col={c}: full={refVal:F6} vs pipeline={actVal:F6} " +
                $"(|diff|={MathF.Abs(refVal - actVal):E3} > {bar:E3}).");
        }
    }

    private static string? FindPtxDir()
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, "ptx"),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", "ptx"),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0)
                return full;
        }
        return null;
    }

    // ── Synthetic 4-layer dense GQA fixture (mirrors VulkanPipelineParityTests.DenseFixture) ──
    private sealed class DenseFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public TransformerWeights Weights { get; private set; } = null!;

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

            nint tokenEmbed = Alloc(VocabSize * HiddenSize, rng);
            float[] outputNorm = Norm(HiddenSize, rng);
            nint output = Alloc(VocabSize * HiddenSize, rng);

            int qOut = NumAttentionHeads * HeadDim;
            int kvOut = NumKvHeads * HeadDim;
            int oIn = NumAttentionHeads * HeadDim;

            var layers = new TransformerLayerWeights[NumLayers];
            for (int i = 0; i < NumLayers; i++)
            {
                layers[i] = new TransformerLayerWeights(
                    attnNormWeight: Norm(HiddenSize, rng),
                    qWeight: Alloc(qOut * HiddenSize, rng), qQuantType: QuantizationType.F32, qOutputDim: qOut, qInputDim: HiddenSize,
                    kWeight: Alloc(kvOut * HiddenSize, rng), kQuantType: QuantizationType.F32, kOutputDim: kvOut, kInputDim: HiddenSize,
                    vWeight: Alloc(kvOut * HiddenSize, rng), vQuantType: QuantizationType.F32, vOutputDim: kvOut, vInputDim: HiddenSize,
                    oWeight: Alloc(HiddenSize * oIn, rng), oQuantType: QuantizationType.F32, oOutputDim: HiddenSize, oInputDim: oIn,
                    ffnNormWeight: Norm(HiddenSize, rng),
                    gateWeight: Alloc(IntermediateSize * HiddenSize, rng), gateQuantType: QuantizationType.F32, gateOutputDim: IntermediateSize, gateInputDim: HiddenSize,
                    upWeight: Alloc(IntermediateSize * HiddenSize, rng), upQuantType: QuantizationType.F32, upOutputDim: IntermediateSize, upInputDim: HiddenSize,
                    downWeight: Alloc(HiddenSize * IntermediateSize, rng), downQuantType: QuantizationType.F32, downOutputDim: HiddenSize, downInputDim: IntermediateSize);
            }

            Weights?.Dispose();
            Weights = TransformerWeights.CreateFromSafetensors(
                tokenEmbedWeight: tokenEmbed, tokenEmbedQt: QuantizationType.F32,
                vocabSize: VocabSize, hiddenSize: HiddenSize,
                layers: layers, outputNormWeight: outputNorm,
                outputWeight: output, outputQt: QuantizationType.F32,
                outputM: VocabSize, outputK: HiddenSize,
                ownedAllocations: new List<nint>());
        }

        private nint Alloc(int count, Random rng)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(ptr);
            float* dst = (float*)ptr;
            for (int i = 0; i < count; i++) dst[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
            return ptr;
        }

        private static float[] Norm(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
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
