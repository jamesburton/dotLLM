using System.Runtime.InteropServices;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.PositionEncoding;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Vulkan;

/// <summary>
/// Parity tests for the M3 pipelined batched forward
/// (<see cref="HybridVulkanCudaTransformerModel.ForwardBatchedPipelined"/>): the
/// overlapped path must produce, per stream, the same logits as the synchronous
/// serial M2 <see cref="HybridVulkanCudaTransformerModel.Forward"/>.
/// </summary>
/// <remarks>
/// <para>
/// The overlapped and serial paths run the <b>same kernels</b> and differ only in
/// CUDA-stream synchronisation timing, so the tolerance is the tight hybrid band
/// (abs 5e-3) rather than the wide cross-precision band — any larger divergence
/// indicates a staging-buffer aliasing bug introduced by the pipeline, not FP
/// noise.
/// </para>
/// <para>
/// <b>Discriminating inputs.</b> The multi-stream case feeds <i>distinct</i> token
/// sequences per stream, so a buffer mix-up between streams (e.g. the persistent
/// temp-F32 staging being overwritten while still in flight) changes the logits
/// and fails the test. Same-token streams could not catch that.
/// </para>
/// <para>
/// <b>Batch=1.</b> A single-request pipelined call must exactly track the serial
/// path (it degenerates to Vulkan → enqueue → finish), confirming the overlap is
/// a no-op win at batch=1.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class HybridVulkanCudaPipelineParityTests
{
    private readonly ITestOutputHelper _out;
    public HybridVulkanCudaPipelineParityTests(ITestOutputHelper output) => _out = output;

    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 16;
    private const int IntermediateSize = 32;
    private const int NumLayers = 4;
    private const int MaxSeqLen = 16;
    private const int NumVulkanLayers = 2;

    // Same kernels, sync-only difference → tight tolerance.
    private const float AbsTol = 5e-3f;
    private const float RelTol = 5e-3f;

    private static bool IsBothAvailable()
        => VulkanDevice.IsAvailable() && IsCudaDriverPresent();

    private static bool IsCudaDriverPresent()
    {
        string lib = OperatingSystem.IsWindows()
            ? "nvcuda.dll" : "libcuda.so.1";
        if (!NativeLibrary.TryLoad(lib, out nint h)) return false;
        NativeLibrary.Free(h);
        return CudaDevice.IsAvailable();
    }

    private static string? FindPtxDir() => FindDir("ptx", "*.ptx", "ptx");
    private static string? FindSpvDir() => FindDir("spv", "*.spv", Path.Combine("vulkan", "spv"));

    private static string? FindDir(string baseName, string pattern, string nativeSub)
    {
        var candidates = new[]
        {
            Path.Combine(AppContext.BaseDirectory, baseName),
            Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "..", "native", nativeSub),
        };
        foreach (var dir in candidates)
        {
            var full = Path.GetFullPath(dir);
            if (Directory.Exists(full) && Directory.GetFiles(full, pattern).Length > 0) return full;
        }
        return null;
    }

    /// <summary>Batch=1: pipelined single-request decode must equal serial decode.</summary>
    [SkippableFact]
    public void Pipelined_Batch1_MatchesSerialDecode()
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");
        string? ptxDir = FindPtxDir(); Skip.If(ptxDir is null, "PTX files not found.");
        string? spvDir = FindSpvDir(); Skip.If(spvDir is null, "SPIR-V shader files not found.");

        int[] prefillIds = [3, 1, 4, 2];
        int[] prefillPos = [0, 1, 2, 3];
        int[] decodeIds = [5];
        int[] decodePos = [4];

        float[] serial = RunSerialDecode(prefillIds, prefillPos, decodeIds, decodePos, ptxDir!, spvDir!);
        float[] pipelined = RunPipelinedDecode(
            new[] { (prefillIds, prefillPos, decodeIds, decodePos) }, ptxDir!, spvDir!)[0];

        AssertMatch(serial, pipelined, "batch1");
    }

    /// <summary>
    /// Multi-stream: 3 streams with distinct token sequences, each decoded one
    /// step. Each stream's pipelined logits must match that stream's serial logits.
    /// </summary>
    [SkippableFact]
    public void Pipelined_MultiStream_MatchesSerialPerStream()
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");
        string? ptxDir = FindPtxDir(); Skip.If(ptxDir is null, "PTX files not found.");
        string? spvDir = FindSpvDir(); Skip.If(spvDir is null, "SPIR-V shader files not found.");

        // Distinct prompts per stream so a cross-stream buffer mixup is detectable.
        var streams = new[]
        {
            (new[] { 3, 1, 4, 2 }, new[] { 0, 1, 2, 3 }, new[] { 5 }, new[] { 4 }),
            (new[] { 7, 6, 5, 0 }, new[] { 0, 1, 2, 3 }, new[] { 2 }, new[] { 4 }),
            (new[] { 1, 1, 2, 3 }, new[] { 0, 1, 2, 3 }, new[] { 6 }, new[] { 4 }),
        };

        // Serial reference per stream (independent model instance per run keeps KV clean).
        var serial = new float[streams.Length][];
        for (int i = 0; i < streams.Length; i++)
            serial[i] = RunSerialDecode(streams[i].Item1, streams[i].Item2,
                streams[i].Item3, streams[i].Item4, ptxDir!, spvDir!);

        float[][] pipelined = RunPipelinedDecode(streams, ptxDir!, spvDir!);

        for (int i = 0; i < streams.Length; i++)
            AssertMatch(serial[i], pipelined[i], $"stream{i}");
    }

    // ── Runners ──────────────────────────────────────────────────────────────

    private float[] RunSerialDecode(
        int[] prefillIds, int[] prefillPos, int[] decodeIds, int[] decodePos,
        string ptxDir, string spvDir)
    {
        using var fixture = DenseFixture.Build(seed: 7);
        using var device = VulkanDevice.Create();
        using var model = HybridVulkanCudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, numVulkanLayers: NumVulkanLayers,
            vulkanDevice: device, cudaDeviceId: 0, spvDir: spvDir, ptxDir: ptxDir);
        using var kv = model.CreateKvCache(MaxSeqLen);

        using (var _ = model.Forward(prefillIds, prefillPos, deviceId: -1, kv)) { }
        using ITensor logits = model.Forward(decodeIds, decodePos, deviceId: -1, kv);
        return new ReadOnlySpan<float>((void*)logits.DataPointer, VocabSize).ToArray();
    }

    private float[][] RunPipelinedDecode(
        (int[] prefillIds, int[] prefillPos, int[] decodeIds, int[] decodePos)[] streams,
        string ptxDir, string spvDir)
    {
        using var fixture = DenseFixture.Build(seed: 7);
        using var device = VulkanDevice.Create();
        using var model = HybridVulkanCudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, numVulkanLayers: NumVulkanLayers,
            vulkanDevice: device, cudaDeviceId: 0, spvDir: spvDir, ptxDir: ptxDir);

        // One KV cache per stream; prefill each serially (prefill is not the
        // overlapped path — the decode tick is).
        var caches = new HybridVulkanCudaKvCache[streams.Length];
        try
        {
            for (int i = 0; i < streams.Length; i++)
            {
                caches[i] = model.CreateKvCache(MaxSeqLen);
                using var _ = model.Forward(streams[i].prefillIds, streams[i].prefillPos, deviceId: -1, caches[i]);
            }

            var requests = new PipelinedRequest[streams.Length];
            for (int i = 0; i < streams.Length; i++)
                requests[i] = new PipelinedRequest
                {
                    TokenIds = streams[i].decodeIds,
                    Positions = streams[i].decodePos,
                    KvCache = caches[i],
                };

            ITensor[] results = model.ForwardBatchedPipelined(requests);
            try
            {
                var outArr = new float[streams.Length][];
                for (int i = 0; i < results.Length; i++)
                    outArr[i] = new ReadOnlySpan<float>((void*)results[i].DataPointer, VocabSize).ToArray();
                return outArr;
            }
            finally
            {
                foreach (var r in results) r.Dispose();
            }
        }
        finally
        {
            foreach (var c in caches) c?.Dispose();
        }
    }

    private void AssertMatch(float[] reference, float[] actual, string variant)
    {
        Assert.Equal(reference.Length, actual.Length);
        float maxAbs = 0f;
        for (int c = 0; c < reference.Length; c++)
            maxAbs = MathF.Max(maxAbs, MathF.Abs(reference[c] - actual[c]));
        _out.WriteLine($"[{variant}] max|diff|={maxAbs:E3}  AbsTol={AbsTol:E3}");

        for (int c = 0; c < reference.Length; c++)
        {
            float refVal = reference[c], actVal = actual[c];
            Assert.True(float.IsFinite(actVal), $"{variant} col={c}: non-finite {actVal}");
            float bar = AbsTol + RelTol * MathF.Abs(refVal);
            Assert.True(MathF.Abs(refVal - actVal) <= bar,
                $"{variant} col={c}: serial={refVal:F6} pipelined={actVal:F6} " +
                $"(|diff|={MathF.Abs(refVal - actVal):E3} > {bar:E3})");
        }
    }

    // ── Fixture (4-layer dense GQA, NeoX RoPE) ──────────────────────────────

    private sealed class DenseFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public TransformerWeights Weights { get; private set; } = null!;

        public static DenseFixture Build(int seed)
        {
            var b = new DenseFixture();
            b.BuildInternal(seed);
            return b;
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
