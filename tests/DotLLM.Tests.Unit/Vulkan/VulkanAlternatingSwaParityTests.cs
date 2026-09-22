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
/// Vulkan twin of <c>CudaAlternatingSwaParityTests</c> (#366) for issue #480: the dense
/// <see cref="VulkanTransformerModel"/> must honour <see cref="ModelConfig.SlidingWindowPattern"/>
/// (gpt-oss-style alternating sliding/dense attention: pattern = 2 → even layers windowed, odd
/// layers dense) exactly like the CPU oracle <c>TransformerModel.GetLayerSlidingWindow</c>.
/// Before #480 <c>VulkanTransformerModel.GetLayerSlidingWindow</c> ignored the pattern and
/// applied the window to every layer.
/// </summary>
/// <remarks>
/// <para>
/// <b>Why this shape discriminates.</b> 4 layers, window = 8, seqLen = 24. With pattern = 2 the
/// dense layers 1/3 see the full 24-token causal history at the last token; the pattern-ignoring
/// mutant (window on every layer — the pre-#480 code) collapses their receptive field to 8. A
/// sequence no longer than the window could not tell the two apart. <see cref="Fixture_PatternChangesCpuLogits"/>
/// proves on the CPU oracle alone that the two masks give logits far outside the parity
/// tolerance, so a pass here cannot be an accident of the fixture.
/// </para>
/// <para>
/// <b>Coverage.</b> Every Vulkan call site of <c>GetLayerSlidingWindow</c> on the dense path:
/// cacheless prefill and KV-cache prefill+decode (<c>Forward</c>, which also exercises the
/// S == 1 split-KV decode kernel at ctx 24), the fused batched path (<c>ForwardBatchSimple</c>),
/// and a pipeline stage that starts at an odd global layer (the <c>_firstLayer</c> offset: using
/// the stage-local index there flips every layer's windowed/dense role).
/// </para>
/// <para>
/// <b>Tolerance / mutant evidence</b> (Strix Halo gfx1151, 2026-09-22; F32 weights on both
/// sides): see <see cref="AbsTol"/>.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class VulkanAlternatingSwaParityTests
{
    private readonly ITestOutputHelper _out;
    public VulkanAlternatingSwaParityTests(ITestOutputHelper output) => _out = output;

    private const int VocabSize = 10;
    private const int HiddenSize = 12;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 6;
    private const int RopeDim = 6;
    private const int IntermediateSize = 20;
    private const int NumLayers = 4;
    private const int SeqLen = 24;
    private const int MaxSeqLen = 32;
    private const int SlidingWindowSize = 8;
    private const int SlidingWindowPattern = 2;

    // Calibrated on this fixture (see class remarks and the report on #480).
    private const float AbsTol = 5e-4f;
    private const float RelTol = 2e-3f;

    // ── Pure resolver semantics (no GPU) ─────────────────────────────────────

    [Theory]
    // gpt-oss shape: window=128, pattern=2 → even windowed, odd dense.
    [InlineData(128, 2, 0, 128)]
    [InlineData(128, 2, 1, 0)]
    [InlineData(128, 2, 2, 128)]
    [InlineData(128, 2, 3, 0)]
    // pattern=3 → layers 0,1 windowed, 2 dense, 3,4 windowed, 5 dense.
    [InlineData(64, 3, 1, 64)]
    [InlineData(64, 3, 2, 0)]
    [InlineData(64, 3, 5, 0)]
    // pattern<=0 → uniform window.
    [InlineData(4096, 0, 0, 4096)]
    [InlineData(4096, 0, 5, 4096)]
    // no window → dense regardless of pattern.
    [InlineData(0, 2, 0, 0)]
    public void ResolveLayerSlidingWindow_PatternSemanticsMatchCpu(
        int window, int pattern, int layer, int expected)
    {
        Assert.Equal(expected,
            VulkanTransformerModel.ResolveLayerSlidingWindow(window, pattern, perLayer: null, layer));
    }

    [Fact]
    public void ResolveLayerSlidingWindow_PerLayerListWinsOverPattern()
    {
        var perLayer = new int?[] { null, 256, null, 64 };
        Assert.Equal(0, VulkanTransformerModel.ResolveLayerSlidingWindow(128, 2, perLayer, 0));
        Assert.Equal(256, VulkanTransformerModel.ResolveLayerSlidingWindow(128, 2, perLayer, 1));
        Assert.Equal(0, VulkanTransformerModel.ResolveLayerSlidingWindow(128, 2, perLayer, 2));
        Assert.Equal(64, VulkanTransformerModel.ResolveLayerSlidingWindow(128, 2, perLayer, 3));
    }

    // ── Fixture self-check: the pattern must matter on this shape ───────────

    /// <summary>
    /// CPU-only: the same weights with pattern = 2 vs pattern = 0 (window on every layer — what
    /// pre-#480 Vulkan computed) must differ by well over the Vulkan parity bar. If this ever
    /// fails the GPU tests below have stopped discriminating.
    /// </summary>
    [Fact]
    public void Fixture_PatternChangesCpuLogits()
    {
        var (ids, pos) = Tokens(SeqLen, seed: 7);
        using var fx = SwaFixture.Build(seed: 23);
        float[] patterned = RunCpuLastRow(fx.Weights, fx.Config, ids, pos);
        float[] uniform = RunCpuLastRow(fx.Weights, fx.Config with { SlidingWindowPattern = 0 }, ids, pos);
        float maxDiff = MaxAbsDiff(patterned, uniform);
        _out.WriteLine($"CPU pattern=2 vs pattern=0 (mutant mask): max |diff| = {maxDiff:E3}");
        Assert.True(maxDiff > 10 * AbsTol,
            $"fixture does not discriminate: pattern-ignoring mask moves logits only {maxDiff:E3}");
    }

    // ── Vulkan vs CPU oracle ────────────────────────────────────────────────

    [SkippableFact]
    public void VulkanForward_AlternatingSwa_PrefillVsCpu_LastTokenLogitsMatch()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (ids, pos) = Tokens(SeqLen, seed: 7);
        using var fx = SwaFixture.Build(seed: 23);
        float[] cpu = RunCpuLastRow(fx.Weights, fx.Config, ids, pos);

        using var device = VulkanDevice.Create();
        using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, fx.Config, fx.Weights, spvDir);
        using ITensor logits = model.Forward(ids, pos, deviceId: -1);
        AssertLogitsMatch(cpu, LastRow(logits), "prefill");
    }

    [SkippableFact]
    public void VulkanForward_AlternatingSwa_KvPrefillThenDecodeVsCpu_LogitsMatch()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (ids, pos) = Tokens(SeqLen, seed: 7);
        using var fx = SwaFixture.Build(seed: 23);
        float[] cpu = RunCpuLastRow(fx.Weights, fx.Config, ids, pos);

        using var device = VulkanDevice.Create();
        using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, fx.Config, fx.Weights, spvDir);
        using var kv = model.CreateKvCache(MaxSeqLen);
        int p = SeqLen - 1;
        using (var _ = model.Forward(ids.AsSpan(0, p), pos.AsSpan(0, p), deviceId: -1, kvCache: kv)) { }
        using ITensor logits = model.Forward(ids.AsSpan(p, 1), pos.AsSpan(p, 1), deviceId: -1, kvCache: kv);
        AssertLogitsMatch(cpu, LastRow(logits), "decode@ctx24");
    }

    [SkippableFact]
    public void VulkanForwardBatch_AlternatingSwa_TwoSequencesVsCpu_LogitsMatch()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (idsA, posA) = Tokens(SeqLen, seed: 7);
        var (idsB, posB) = Tokens(20, seed: 11);
        using var fx = SwaFixture.Build(seed: 23);
        float[] cpuA = RunCpuLastRow(fx.Weights, fx.Config, idsA, posA);
        float[] cpuB = RunCpuLastRow(fx.Weights, fx.Config, idsB, posB);

        using var device = VulkanDevice.Create();
        using var model = VulkanTransformerModel.BuildFromPrebuiltWeights(device, fx.Config, fx.Weights, spvDir);
        using var kvA = model.CreateKvCache(MaxSeqLen);
        using var kvB = model.CreateKvCache(MaxSeqLen);
        var requests = new[]
        {
            new SequenceForwardRequest { TokenIds = idsA.AsMemory(), Positions = posA.AsMemory(), KvCache = kvA },
            new SequenceForwardRequest { TokenIds = idsB.AsMemory(), Positions = posB.AsMemory(), KvCache = kvB },
        };
        var results = model.ForwardBatch(requests, deviceId: -1);
        try
        {
            Assert.Equal(2, results.Count);
            AssertLogitsMatch(cpuA, LastRow(results[0]), "batch[0] len24");
            AssertLogitsMatch(cpuB, LastRow(results[1]), "batch[1] len20");
        }
        finally
        {
            foreach (var t in results) t.Dispose();
        }
    }

    /// <summary>
    /// Pipeline split at global layer 1: stage 1 covers global layers 1..3, whose local indices
    /// 0..2 have the OPPOSITE parity. Discriminates a resolver that uses the stage-local index.
    /// </summary>
    [SkippableFact]
    public void VulkanPipeline_AlternatingSwa_SplitAtOddLayerVsCpu_LogitsMatch()
    {
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out string spvDir);
        var (ids, pos) = Tokens(SeqLen, seed: 7);
        using var fx = SwaFixture.Build(seed: 23);
        float[] cpu = RunCpuLastRow(fx.Weights, fx.Config, ids, pos);

        using var device = VulkanDevice.Create();
        using var model = VulkanPipelineTransformerModel.BuildFromPrebuiltWeights(
            device, device, fx.Config, fx.Weights, splitLayer: 1, spvDir);
        using ITensor logits = model.Forward(ids, pos, deviceId: -1);
        AssertLogitsMatch(cpu, LastRow(logits), "pipeline split=1");
    }

    // ── helpers ─────────────────────────────────────────────────────────────

    private static (int[] Ids, int[] Pos) Tokens(int n, int seed)
    {
        var rng = new Random(seed);
        var ids = new int[n];
        var pos = new int[n];
        for (int i = 0; i < n; i++) { ids[i] = rng.Next(VocabSize); pos[i] = i; }
        return (ids, pos);
    }

    private static float[] RunCpuLastRow(TransformerWeights weights, ModelConfig config, int[] ids, int[] pos)
    {
        using var model = TransformerModel.BuildFromPrebuiltWeights(weights, config);
        using ITensor logits = model.Forward(ids, pos, deviceId: -1);
        Assert.Equal(ids.Length, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, ids.Length * VocabSize);
        return span.Slice((ids.Length - 1) * VocabSize, VocabSize).ToArray();
    }

    /// <summary>Vulkan returns last-token-only logits, <c>[1, vocab]</c>.</summary>
    private static float[] LastRow(ITensor logits)
    {
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0f;
        for (int i = 0; i < a.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    private void AssertLogitsMatch(float[] cpu, float[] vk, string label)
    {
        Assert.Equal(cpu.Length, vk.Length);
        _out.WriteLine($"[{label}] max |diff| = {MaxAbsDiff(cpu, vk):E3}");
        for (int c = 0; c < cpu.Length; c++)
        {
            Assert.True(float.IsFinite(vk[c]), $"[{label}] col={c}: vulkan logit non-finite: {vk[c]}");
            float diff = MathF.Abs(cpu[c] - vk[c]);
            float bar = AbsTol + RelTol * MathF.Abs(cpu[c]);
            Assert.True(diff <= bar,
                $"[{label}] col={c}: cpu={cpu[c]:F6} vs vulkan={vk[c]:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    /// <summary>
    /// Synthetic 4-layer F32 alternating-SWA fixture — same construction as
    /// <c>CudaAlternatingSwaParityTests.AlternatingSwaFixture</c>. Owns every allocation and the
    /// <see cref="TransformerWeights"/>; the CPU and Vulkan <c>BuildFromPrebuiltWeights</c> entry
    /// points leave ownership with the caller.
    /// </summary>
    private sealed class SwaFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config { get; private set; } = null!;
        public TransformerWeights Weights { get; private set; } = null!;

        public static SwaFixture Build(int seed)
        {
            var f = new SwaFixture();
            f.BuildInternal(seed);
            return f;
        }

        private void BuildInternal(int seed)
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
                RoPEConfig = new RoPEConfig(Theta: 10000.0f, DimensionCount: RopeDim, Type: RoPEType.NeoX),
                ActivationFunction = ActivationFunction.SiLU,
                NormType = NormType.RMSNorm,
                NormEpsilon = 1e-5f,
                TiedEmbeddings = false,
                ChatTemplate = null,
                SlidingWindowSize = SlidingWindowSize,
                SlidingWindowPattern = SlidingWindowPattern,
            };

            nint tokenEmbed = AllocUniform(VocabSize * HiddenSize, rng, 0.05f);
            float[] outputNorm = NormVec(HiddenSize, rng);
            nint output = AllocUniform(VocabSize * HiddenSize, rng, 0.05f);

            int qOut = NumAttentionHeads * HeadDim;
            int kvOut = NumKvHeads * HeadDim;
            var layers = new TransformerLayerWeights[NumLayers];
            for (int i = 0; i < NumLayers; i++)
            {
                float[] attnNorm = NormVec(HiddenSize, rng);
                float[] ffnNorm = NormVec(HiddenSize, rng);
                nint qW = AllocUniform(qOut * HiddenSize, rng, 0.05f);
                nint kW = AllocUniform(kvOut * HiddenSize, rng, 0.05f);
                nint vW = AllocUniform(kvOut * HiddenSize, rng, 0.05f);
                nint oW = AllocUniform(HiddenSize * qOut, rng, 0.05f);
                nint gateW = AllocUniform(IntermediateSize * HiddenSize, rng, 0.05f);
                nint upW = AllocUniform(IntermediateSize * HiddenSize, rng, 0.05f);
                nint downW = AllocUniform(HiddenSize * IntermediateSize, rng, 0.05f);

                layers[i] = new TransformerLayerWeights(
                    attnNormWeight: attnNorm,
                    qWeight: qW, qQuantType: QuantizationType.F32, qOutputDim: qOut, qInputDim: HiddenSize,
                    kWeight: kW, kQuantType: QuantizationType.F32, kOutputDim: kvOut, kInputDim: HiddenSize,
                    vWeight: vW, vQuantType: QuantizationType.F32, vOutputDim: kvOut, vInputDim: HiddenSize,
                    oWeight: oW, oQuantType: QuantizationType.F32, oOutputDim: HiddenSize, oInputDim: qOut,
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

        private nint AllocUniform(int count, Random rng, float amplitude)
        {
            nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
            _allocs.Add(ptr);
            float* dst = (float*)ptr;
            for (int i = 0; i < count; i++)
                dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amplitude;
            return ptr;
        }

        private static float[] NormVec(int count, Random rng)
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
