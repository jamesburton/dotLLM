using System.Diagnostics;
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
/// Latency baseline for the M2 (fence-serialized) Vulkan+CUDA hybrid forward pass.
/// These measurements provide the "before" reference for M3 async pipelining.
/// Not a correctness test — numbers are printed to xUnit output and captured
/// in the commit message / test log.
/// </summary>
/// <remarks>
/// Uses the same 4-layer synthetic fixture as <see cref="HybridVulkanCudaParityTests"/>
/// so results are reproducible across runs. Warm-up iterations excluded from the
/// median. Numbers are in milliseconds (wall clock via <see cref="Stopwatch"/>).
/// </remarks>
[Trait("Category", "GPU")]
[Collection("VulkanKernels")]
public sealed unsafe class HybridVulkanCudaLatencyTests
{
    private const int VocabSize = 8;
    private const int HiddenSize = 32;
    private const int NumAttentionHeads = 2;
    private const int NumKvHeads = 1;
    private const int HeadDim = 16;
    private const int RopeDim = 16;
    private const int IntermediateSize = 32;
    private const int NumLayers = 4;
    private const int MaxSeqLen = 64;

    private const int WarmupIter = 5;
    private const int MeasureIter = 20;

    private readonly ITestOutputHelper _out;
    public HybridVulkanCudaLatencyTests(ITestOutputHelper output) => _out = output;

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
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.ptx").Length > 0) return full;
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
            if (Directory.Exists(full) && Directory.GetFiles(full, "*.spv").Length > 0) return full;
        }
        return null;
    }

    /// <summary>
    /// Measures median decode latency for the fence-serialized Vulkan+CUDA
    /// hybrid (M2 baseline). Printed output is the M3 "before" reference number.
    /// </summary>
    [SkippableFact]
    public void M2_FenceSerializedBaseline_DecodeLatency()
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found.");

        using var fixture = BuildFixture(seed: 42);
        using var device = VulkanDevice.Create();

        // Split at 2: representative mid-stack split (same as M1 baseline).
        const int NumVulkanLayers = 2;
        using var model = HybridVulkanCudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config,
            numVulkanLayers: NumVulkanLayers,
            vulkanDevice: device, cudaDeviceId: 0,
            spvDir: spvDir, ptxDir: ptxDir);

        using var kvCache = model.CreateKvCache(MaxSeqLen);

        // Prefill step (not measured — just populates KV cache).
        int[] prefillIds = [3, 1, 4, 2, 5, 6, 7, 1];
        int[] prefillPos  = [0, 1, 2, 3, 4, 5, 6, 7];
        using (var _ = model.Forward(prefillIds, prefillPos, deviceId: -1, kvCache)) { }

        // Decode step — single token, KV-cache path.
        int[] decodeIds = [2];
        int[] decodePos  = [8];

        // Warm-up.
        for (int i = 0; i < WarmupIter; i++)
        {
            using var t = model.Forward(decodeIds, decodePos, deviceId: -1, kvCache);
        }

        // Measure.
        var samples = new double[MeasureIter];
        var sw = new Stopwatch();
        for (int i = 0; i < MeasureIter; i++)
        {
            sw.Restart();
            using var t = model.Forward(decodeIds, decodePos, deviceId: -1, kvCache);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(samples);
        double median = Median(samples);
        double p95    = samples[(int)(MeasureIter * 0.95)];
        double min    = samples[0];

        _out.WriteLine("=== M2 fence-serialized Vulkan+CUDA decode latency ===");
        _out.WriteLine($"  split={NumVulkanLayers}/{NumLayers}  seqLen=1 (decode step at pos=8)");
        _out.WriteLine($"  warmup={WarmupIter}  measure={MeasureIter}");
        _out.WriteLine($"  median={median:F3} ms  p95={p95:F3} ms  min={min:F3} ms");
        _out.WriteLine("=== This is the M3 'before' baseline ===");

        // Sanity: must be finite and positive (not a real correctness assertion).
        Assert.True(median > 0 && median < 10_000, $"Median {median:F3} ms looks wrong.");
    }

    /// <summary>
    /// Measures median prefill latency for the fence-serialized Vulkan+CUDA
    /// hybrid (M2 baseline) at seqLen=32.
    /// </summary>
    [SkippableFact]
    public void M2_FenceSerializedBaseline_PrefillLatency_SeqLen32()
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");
        string? ptxDir = FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found.");
        string? spvDir = FindSpvDir();
        Skip.If(spvDir is null, "SPIR-V shader files not found.");

        using var fixture = BuildFixture(seed: 42);
        using var device = VulkanDevice.Create();

        const int SeqLen = 32;
        const int NumVulkanLayers = 2;
        using var model = HybridVulkanCudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config,
            numVulkanLayers: NumVulkanLayers,
            vulkanDevice: device, cudaDeviceId: 0,
            spvDir: spvDir, ptxDir: ptxDir);

        int[] tokenIds = new int[SeqLen];
        int[] positions = new int[SeqLen];
        for (int i = 0; i < SeqLen; i++) { tokenIds[i] = i % VocabSize; positions[i] = i; }

        // Warm-up.
        for (int i = 0; i < WarmupIter; i++)
        {
            using var t = model.Forward(tokenIds, positions, deviceId: -1);
        }

        // Measure.
        var samples = new double[MeasureIter];
        var sw = new Stopwatch();
        for (int i = 0; i < MeasureIter; i++)
        {
            sw.Restart();
            using var t = model.Forward(tokenIds, positions, deviceId: -1);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(samples);
        double median = Median(samples);
        double p95    = samples[(int)(MeasureIter * 0.95)];
        double min    = samples[0];

        _out.WriteLine("=== M2 fence-serialized Vulkan+CUDA prefill latency (seqLen=32) ===");
        _out.WriteLine($"  split={NumVulkanLayers}/{NumLayers}");
        _out.WriteLine($"  warmup={WarmupIter}  measure={MeasureIter}");
        _out.WriteLine($"  median={median:F3} ms  p95={p95:F3} ms  min={min:F3} ms");
        _out.WriteLine("=== This is the M3 'before' baseline ===");

        Assert.True(median > 0 && median < 10_000, $"Median {median:F3} ms looks wrong.");
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    private static double Median(double[] sorted)
    {
        int n = sorted.Length;
        return n % 2 == 0 ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 : sorted[n / 2];
    }

    private static LatencyFixture BuildFixture(int seed)
    {
        var rng = new Random(seed);
        var config = new ModelConfig
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

        var allocs = new List<nint>();
        nint tokenEmbed = AllocUniform(VocabSize * HiddenSize, rng, 0.05f, allocs);
        float[] outputNorm = FillNorm(HiddenSize, rng);
        nint output = AllocUniform(VocabSize * HiddenSize, rng, 0.05f, allocs);

        int qOut = NumAttentionHeads * HeadDim;
        int kvOut = NumKvHeads * HeadDim;
        int oIn  = NumAttentionHeads * HeadDim;

        var layers = new TransformerLayerWeights[NumLayers];
        for (int i = 0; i < NumLayers; i++)
        {
            layers[i] = new TransformerLayerWeights(
                attnNormWeight: FillNorm(HiddenSize, rng),
                qWeight: AllocUniform(qOut * HiddenSize, rng, 0.05f, allocs),
                qQuantType: QuantizationType.F32, qOutputDim: qOut, qInputDim: HiddenSize,
                kWeight: AllocUniform(kvOut * HiddenSize, rng, 0.05f, allocs),
                kQuantType: QuantizationType.F32, kOutputDim: kvOut, kInputDim: HiddenSize,
                vWeight: AllocUniform(kvOut * HiddenSize, rng, 0.05f, allocs),
                vQuantType: QuantizationType.F32, vOutputDim: kvOut, vInputDim: HiddenSize,
                oWeight: AllocUniform(HiddenSize * oIn, rng, 0.05f, allocs),
                oQuantType: QuantizationType.F32, oOutputDim: HiddenSize, oInputDim: oIn,
                ffnNormWeight: FillNorm(HiddenSize, rng),
                gateWeight: AllocUniform(IntermediateSize * HiddenSize, rng, 0.05f, allocs),
                gateQuantType: QuantizationType.F32, gateOutputDim: IntermediateSize, gateInputDim: HiddenSize,
                upWeight: AllocUniform(IntermediateSize * HiddenSize, rng, 0.05f, allocs),
                upQuantType: QuantizationType.F32, upOutputDim: IntermediateSize, upInputDim: HiddenSize,
                downWeight: AllocUniform(HiddenSize * IntermediateSize, rng, 0.05f, allocs),
                downQuantType: QuantizationType.F32, downOutputDim: HiddenSize, downInputDim: IntermediateSize);
        }

        var weights = TransformerWeights.CreateFromSafetensors(
            tokenEmbedWeight: tokenEmbed, tokenEmbedQt: QuantizationType.F32,
            vocabSize: VocabSize, hiddenSize: HiddenSize,
            layers: layers,
            outputNormWeight: outputNorm,
            outputWeight: output, outputQt: QuantizationType.F32,
            outputM: VocabSize, outputK: HiddenSize,
            ownedAllocations: new List<nint>());

        return new LatencyFixture(config, weights, allocs);
    }

    private static unsafe nint AllocUniform(int count, Random rng, float amp, List<nint> allocs)
    {
        nint ptr = (nint)NativeMemory.AlignedAlloc((nuint)((long)count * sizeof(float)), 64);
        allocs.Add(ptr);
        float* dst = (float*)ptr;
        for (int i = 0; i < count; i++)
            dst[i] = ((float)rng.NextDouble() * 2f - 1f) * amp;
        return ptr;
    }

    private static float[] FillNorm(int count, Random rng)
    {
        var arr = new float[count];
        for (int i = 0; i < count; i++)
            arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.05f;
        return arr;
    }

    // Simple RAII wrapper so tests don't leak unmanaged memory.
    private sealed unsafe class LatencyFixture : IDisposable
    {
        public readonly ModelConfig Config;
        public readonly TransformerWeights Weights;
        private readonly List<nint> _allocs;
        public LatencyFixture(ModelConfig c, TransformerWeights w, List<nint> a)
        {
            Config = c; Weights = w; _allocs = a;
        }
        public void Dispose()
        {
            Weights?.Dispose();
            foreach (var p in _allocs) NativeMemory.AlignedFree((void*)p);
            _allocs.Clear();
        }
    }
}
