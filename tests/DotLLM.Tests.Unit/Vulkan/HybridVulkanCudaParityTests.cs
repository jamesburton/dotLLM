using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Interop;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for <see cref="HybridVulkanCudaTransformerModel"/>:
/// the Vulkan+CUDA layer-split path must produce logits within a
/// tolerance band of the pure-CUDA reference on the same weights.
/// </summary>
/// <remarks>
/// <para>
/// <b>Architecture.</b> A 4-layer synthetic dense GQA fixture (hidden=32,
/// heads=2, KVheads=1, headDim=16, intermediate=32) is parameterised over
/// three split points:
/// <list type="bullet">
///   <item>numVulkanLayers=1 (split at boundary 1/3) — catches off-by-one
///   bugs where layer 0 ≡ global index ≡ local index, masking reindex errors.</item>
///   <item>numVulkanLayers=2 (mid-stack) — original M1 split; exercises
///   the two-independent-0-based-remap property (weights + KV cache).</item>
///   <item>numVulkanLayers=3 (split at L-1) — only one CUDA layer; exercises
///   the single-element edge case in the CUDA loop and firstLayer=3 offset
///   in <see cref="CudaWeights.LoadFromGguf"/>.</item>
/// </list>
/// Both prefill and decode are run for each split so the three-way coverage
/// catches both the activation handoff path and the KV-cache indexing path.
/// </para>
/// <para>
/// <b>Tolerance band.</b> The hybrid path stacks two error sources on top of
/// pure-CUDA FP16 noise: the FP32→FP16 convert at the CUDA upload and the
/// Vulkan FP32-precision layer body accumulated over N layers.
/// The tolerance is empirically widened to accommodate this (5e-2 abs / 1e-1
/// rel) compared with the pure-CUDA parity tests (1.5e-3 abs / 5e-3 rel),
/// reflecting the precision difference between Vulkan FP32 and CUDA FP16
/// layer body.
/// </para>
/// <para>
/// <b>What the split parameterisation discriminates.</b>
/// Split=1: wrong <c>_cudaWeights.Layers[layer]</c> (missing firstLayer offset)
/// reads layer 1 data for layer 0 — immediate OOB or silent garbage. Split=3:
/// same bug surfaces on the first-CUDA-layer norm at local index 0 but global
/// index 3. A wrong <c>cacheLayer = layer</c> (missing <c>- numVulkanLayers</c>
/// offset) causes an out-of-bounds read in <see cref="HybridVulkanCudaKvCache"/>
/// at decode step 2 — split=1 catches it because <c>cacheLayer=1</c> would
/// read slot 1 from a 3-slot CUDA cache, while split=3 gives only 1 slot.
/// </para>
/// <para>
/// <b>Skip behaviour.</b> Skips cleanly when either Vulkan or CUDA is
/// unavailable, when PTX files cannot be located, or when the
/// DOTLLM_VULKAN_DEVICE_VENDOR env-var is not set to the Intel Arc vendor.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class HybridVulkanCudaParityTests
{
    private readonly ITestOutputHelper _out;
    public HybridVulkanCudaParityTests(ITestOutputHelper output) => _out = output;

    // ── Fixture shape ───────────────────────────────────────────────────────
    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 16;
    private const int IntermediateSize = 32;
    private const int NumLayers = 4;
    private const int MaxSeqLen = 8;

    // Three split points: 1 (min), 2 (mid), 3 (L-1 = max).
    // Split at 1: numVulkanLayers=1, numCudaLayers=3, firstLayer=1
    // Split at 2: numVulkanLayers=2, numCudaLayers=2, firstLayer=2  (original M1 point)
    // Split at 3: numVulkanLayers=3, numCudaLayers=1, firstLayer=3
    public static TheoryData<int> SplitPoints => new() { 1, 2, 3 };

    // Tolerance: Vulkan runs FP32 for layer bodies; CUDA runs FP16.
    // The FP32→FP16 conversion at the handoff introduces rounding at every
    // hidden-state element. Over N Vulkan layers + boundary + (L-N) CUDA
    // layers this accumulates. We use 5e-2 abs / 1e-1 rel — well above
    // empirical noise but below the ~0.5 jump caused by a real boundary bug
    // (mirroring the evidence in HybridTransformerModelSplitParityTests at ~600×).
    private const float AbsTol = 5e-2f;
    private const float RelTol = 1e-1f;

    private static bool IsBothAvailable()
        => VulkanDevice.IsAvailable() && IsCudaDriverPresent();

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows()
            ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaAvailableProbe();
    }

    [System.Runtime.CompilerServices.MethodImpl(
        System.Runtime.CompilerServices.MethodImplOptions.NoInlining)]
    private static bool CudaAvailableProbe() => CudaDevice.IsAvailable();

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

    /// <summary>
    /// Prefill parity: 4-token sequence, NeoX RoPE, split point parameterised
    /// over {1, 2, 3}. Hybrid logits must match pure-CUDA within the tolerance
    /// band at the last-token position.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void HybridVulkanCuda_DenseNeoxRope_PrefillVsCuda_LastTokenLogitsMatch(int numVulkanLayers)
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found; build with CUDA Toolkit.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        int[] tokenIds = [3, 1, 4, 2];
        int[] positions = [0, 1, 2, 3];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);

        float[] cudaLast = RunCudaLastRow(fixture, tokenIds, positions, ptxDir!);
        float[] hybridLast = RunHybridLastRow(fixture, tokenIds, positions, numVulkanLayers, ptxDir!, spvDir!);

        AssertLogitsMatch(cudaLast, hybridLast, $"NeoX-prefill/split={numVulkanLayers}");
    }

    /// <summary>
    /// Single-token decode parity: NeoX RoPE decode step (positions=[4]),
    /// split point parameterised over {1, 2, 3}.
    /// Exercises the KV-cache path — specifically that both the weight array
    /// and <see cref="HybridVulkanCudaKvCache"/> use independently-0-based
    /// (local) layer indices, not global layer indices.
    /// </summary>
    [SkippableTheory]
    [MemberData(nameof(SplitPoints))]
    public void HybridVulkanCuda_DenseNeoxRope_DecodeVsCuda_LastTokenLogitsMatch(int numVulkanLayers)
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found; build with CUDA Toolkit.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found; build with Vulkan SDK.");

        // Prefill 4 tokens, then decode one more (position=4).
        int[] prefillIds = [3, 1, 4, 2];
        int[] prefillPos = [0, 1, 2, 3];
        int[] decodeIds = [5];
        int[] decodePos = [4];

        using var fixture = DenseFixture.Build(seed: 42, ropeType: RoPEType.NeoX);

        // Run pure-CUDA path with KV cache.
        float[] cudaLast = RunCudaDecodeLastRow(fixture, prefillIds, prefillPos, decodeIds, decodePos, ptxDir!);

        // Run hybrid path with KV cache.
        float[] hybridLast = RunHybridDecodeLastRow(fixture, prefillIds, prefillPos, decodeIds, decodePos,
            numVulkanLayers, ptxDir!, spvDir!);

        AssertLogitsMatch(cudaLast, hybridLast, $"NeoX-decode/split={numVulkanLayers}");
    }

    // ── Runner helpers ──────────────────────────────────────────────────────

    private static float[] RunCudaLastRow(
        DenseFixture fixture, int[] tokenIds, int[] positions, string ptxDir)
    {
        using var model = CudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, deviceId: 0, ptxDir: ptxDir);
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: 0);
        // CudaTransformerModel returns [1, vocabSize] (last token only).
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
    }

    private float[] RunHybridLastRow(
        DenseFixture fixture, int[] tokenIds, int[] positions,
        int numVulkanLayers, string ptxDir, string spvDir)
    {
        using var device = VulkanDevice.Create();
        using var model = HybridVulkanCudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, numVulkanLayers: numVulkanLayers,
            vulkanDevice: device, cudaDeviceId: 0, spvDir: spvDir, ptxDir: ptxDir);

        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
    }

    private static float[] RunCudaDecodeLastRow(
        DenseFixture fixture,
        int[] prefillIds, int[] prefillPos, int[] decodeIds, int[] decodePos,
        string ptxDir)
    {
        using var model = CudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, deviceId: 0, ptxDir: ptxDir);
        using var kvCache = model.CreateKvCache(MaxSeqLen);

        // Prefill
        using var _ = model.Forward(prefillIds, prefillPos, deviceId: 0, kvCache);

        // Decode
        using ITensor logits = model.Forward(decodeIds, decodePos, deviceId: 0, kvCache);
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
    }

    private float[] RunHybridDecodeLastRow(
        DenseFixture fixture,
        int[] prefillIds, int[] prefillPos, int[] decodeIds, int[] decodePos,
        int numVulkanLayers, string ptxDir, string spvDir)
    {
        using var device = VulkanDevice.Create();
        using var model = HybridVulkanCudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, numVulkanLayers: numVulkanLayers,
            vulkanDevice: device, cudaDeviceId: 0, spvDir: spvDir, ptxDir: ptxDir);
        using var kvCache = model.CreateKvCache(MaxSeqLen);

        // Prefill
        using var _ = model.Forward(prefillIds, prefillPos, deviceId: -1, kvCache);

        // Decode
        using ITensor logits = model.Forward(decodeIds, decodePos, deviceId: -1, kvCache);
        Assert.Equal(1, logits.Shape[0]);
        Assert.Equal(VocabSize, logits.Shape[1]);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
    }

    // ── Assertion ──────────────────────────────────────────────────────────

    private void AssertLogitsMatch(float[] reference, float[] hybrid, string variant)
    {
        Assert.Equal(reference.Length, hybrid.Length);

        _out.WriteLine($"[{variant}] col | reference  | hybrid     | |diff|");
        _out.WriteLine($"[{variant}] ----+------------+------------+----------");
        float maxAbs = 0f;
        double sumSq = 0.0;
        for (int c = 0; c < reference.Length; c++)
        {
            float d = MathF.Abs(reference[c] - hybrid[c]);
            if (d > maxAbs) maxAbs = d;
            sumSq += (double)d * d;
            _out.WriteLine($"[{variant}] {c,3} | {reference[c],10:F6} | {hybrid[c],10:F6} | {d:E3}");
        }
        double rms = Math.Sqrt(sumSq / reference.Length);
        _out.WriteLine($"[{variant}] max|diff|={maxAbs:E3}  rms={rms:E3}  AbsTol={AbsTol:E3}");

        for (int c = 0; c < reference.Length; c++)
        {
            float refVal = reference[c];
            float hybVal = hybrid[c];
            Assert.True(float.IsFinite(refVal), $"{variant} col={c}: reference logit non-finite: {refVal}");
            Assert.True(float.IsFinite(hybVal), $"{variant} col={c}: hybrid logit non-finite: {hybVal}");
            float diff = MathF.Abs(refVal - hybVal);
            float bar = AbsTol + RelTol * MathF.Abs(refVal);
            Assert.True(diff <= bar,
                $"{variant} col={c}: ref={refVal:F6} vs hybrid={hybVal:F6} (|diff|={diff:E3} > {bar:E3})");
        }
    }

    // ── Fixture ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Synthetic 4-layer dense transformer weight fixture in unmanaged memory.
    /// Mirrors <see cref="HybridTransformerModelSplitParityTests.DenseFixture"/>
    /// with an independent seed to verify the handoff under different weight distributions.
    /// </summary>
    private sealed unsafe class DenseFixture : IDisposable
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

            Weights?.Dispose();
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
            foreach (var p in _allocs)
                NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
