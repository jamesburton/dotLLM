using System.Diagnostics;
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
/// Throughput benchmark for the M3 host-pipelined overlap: a batched / multi-
/// stream decode tick (N independent sequences, one decode step each) run via the
/// synchronous serial M2 path (a loop of <see cref="HybridVulkanCudaTransformerModel.Forward"/>)
/// versus the overlapped
/// <see cref="HybridVulkanCudaTransformerModel.ForwardBatchedPipelined"/>.
/// </summary>
/// <remarks>
/// <para>
/// Uses a <b>scaled</b> synthetic fixture (hidden=2048, ffn=5504, 16 layers) so
/// each device's per-stream decode stage is multi-millisecond and the overlap is
/// observable — the tiny parity fixture is dominated by submit/launch overhead.
/// The Vulkan side is forced onto the Intel Arc iGPU so the two stages run on
/// physically different devices (iGPU Vulkan ∥ eGPU CUDA).
/// </para>
/// <para>
/// <b>Analytical ceiling.</b> For a depth-1 host pipeline over N streams with
/// per-stream Vulkan stage Tv and CUDA stage Tc, pipelined wall time is
/// <c>Tv + (N-1)·max(Tv,Tc) + Tc</c> and serial is <c>N·(Tv+Tc)</c>, so the
/// speedup is bounded by <c>N·(Tv+Tc) / (Tv+(N-1)·max(Tv,Tc)+Tc)</c>. With the
/// balanced split (Tv≈Tc) that ceiling is <c>2N/(N+1)</c> — 1.33x@N=2, 1.6x@N=4,
/// 1.78x@N=8. The benchmark sweeps N at the balanced split and reports the
/// measured speedup against this ceiling; matching the <i>curve shape</i> is
/// robust to absolute-time throttling (it is a ratio) and is the real evidence
/// the overlap works.
/// </para>
/// <para>
/// <b>Balanced split.</b> Equal layer counts are <i>not</i> balanced: the iGPU
/// LPDDR5x (~100 GB/s) is ~3x slower per byte than the 3060 (360 GB/s) and
/// seqLen=1 decode is weight-bandwidth-bound, so a balanced split puts roughly
/// 3x more layers on the CUDA eGPU. We empirically pick the split whose measured
/// Tv≈Tc and run the N-sweep there.
/// </para>
/// <para>
/// Measurement is stabilised: fresh caches per variant prefilled to the same
/// length, KV rolled back after every rep so each rep does identical work,
/// serial/pipelined truly interleaved (S,P,S,P) to share thermal state, a short
/// cooldown between reps, and min/median/max reported so variance is visible.
/// Not a hard correctness gate — numbers print to xUnit output; assertions are
/// sanity + a ceiling guard that flags physically-impossible (contaminated)
/// speedups.
/// </para>
/// </remarks>
[Trait("Category", "GPU")]
[Trait("Category", "Benchmark")]
[Collection("VulkanKernels")]
public sealed unsafe class HybridVulkanCudaPipelineBenchTests
{
    private const int VocabSize = 256;
    private const int HiddenSize = 2048;
    private const int NumAttentionHeads = 16;
    private const int NumKvHeads = 4;
    private const int HeadDim = 128;
    private const int RopeDim = 128;
    private const int IntermediateSize = 5504;
    private const int NumLayers = 16;
    private const int MaxSeqLen = 64;
    private const int PrefillLen = 4;

    private const int Warmup = 3;
    private const int Repeats = 10;
    private const int CooldownMs = 250;

    // iGPU LPDDR5x (~100 GB/s) is ~3x slower per byte than the 3060 (360 GB/s),
    // and seqLen=1 decode is weight-bandwidth-bound, so the balanced split puts
    // ~3x more layers on the CUDA eGPU: Tv ∝ split·3, Tc ∝ (L-split)·1, balanced
    // when split·3 = (L-split)·1 → split = L/4 = 4 for L=16.
    private const double IgpuRelCostPerLayer = 3.0;
    private const int BalancedSplit = NumLayers / 4; // = 4

    private readonly ITestOutputHelper _out;
    public HybridVulkanCudaPipelineBenchTests(ITestOutputHelper output) => _out = output;

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

    /// <summary>
    /// Picks the balanced split (Tv≈Tc), sweeps N over {1,2,4,8} there, and
    /// compares pipelined vs serial against the 2N/(N+1) ceiling.
    /// </summary>
    [SkippableFact]
    public void Bench_BatchedDecode_PipelinedVsSerial()
    {
        Skip.IfNot(IsBothAvailable(), "Both Vulkan and CUDA GPU must be available.");
        string? ptxDir = FindPtxDir(); Skip.If(ptxDir is null, "PTX files not found.");
        string? spvDir = FindSpvDir(); Skip.If(spvDir is null, "SPIR-V shader files not found.");

        using var fixture = ScaledFixture.Build(seed: 11);

        // Force the Vulkan side onto the Intel Arc iGPU so the two stages run on
        // physically different devices. Without this the scorer picks the dGPU
        // for Vulkan too and both stages serialize on the one 3060 (no overlap).
        string? priorVendor = Environment.GetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_VENDOR");
        Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_VENDOR", "0x8086");
        VulkanDevice device;
        try { device = VulkanDevice.Create(); }
        catch { Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_VENDOR", priorVendor); throw; }

        try
        {
            _out.WriteLine($"Vulkan device: {device.DeviceName} (vendor=0x{device.VendorId:X4})");
            _out.WriteLine($"Model: hidden={HiddenSize} ffn={IntermediateSize} layers={NumLayers} " +
                $"heads={NumAttentionHeads}/{NumKvHeads} headDim={HeadDim}");
            Skip.If(device.VendorId == 0x10DE,
                "Vulkan selected the NVIDIA dGPU — both stages would run on the 3060 and cannot overlap.");

            // Pinned balanced split (derived from the ~3x iGPU/eGPU bandwidth ratio,
            // see BalancedSplit). At the balanced split Tv≈Tc, so the depth-1 pipeline
            // ceiling is 2N/(N+1). For reference we also print the general per-split
            // ceiling that accounts for imbalance.
            int split = BalancedSplit;
            double tvShare = split * IgpuRelCostPerLayer;
            double tcShare = (NumLayers - split) * 1.0;
            _out.WriteLine($"Balanced split = {split}/{NumLayers}  (modelled Tv:Tc = {tvShare:F0}:{tcShare:F0})");
            _out.WriteLine("");
            _out.WriteLine($"--- N-sweep at split {split}/{NumLayers} (interleaved S/P, min of {Repeats}) ---");
            _out.WriteLine("  N | serial(min) | pipe(min) | speedup | ceiling");
            _out.WriteLine("  --+-------------+-----------+---------+--------");

            double n1Speedup = double.NaN;
            foreach (int n in new[] { 1, 2, 4, 8 })
            {
                (double serialMin, double serialMed, double serialMax,
                 double pipeMin, double pipeMed, double pipeMax) =
                    MeasureInterleaved(fixture, device, split, n, ptxDir!, spvDir!);

                double speedup = serialMin / pipeMin;
                // General depth-1 ceiling: N(Tv+Tc) / (Tv + Tc + (N-1)·max(Tv,Tc)).
                // At the balanced split this reduces to 2N/(N+1).
                double maxStage = Math.Max(tvShare, tcShare);
                double ceiling = n * (tvShare + tcShare) / (tvShare + tcShare + (n - 1) * maxStage);
                if (n == 1) { ceiling = 1.0; n1Speedup = speedup; }

                _out.WriteLine($"  {n,1} | {serialMin,9:F2}   | {pipeMin,7:F2}   | {speedup,5:F3}x  | {ceiling:F3}x");
                _out.WriteLine($"      serial[min/med/max]={serialMin:F1}/{serialMed:F1}/{serialMax:F1}  " +
                    $"pipe[min/med/max]={pipeMin:F1}/{pipeMed:F1}/{pipeMax:F1}");

                Assert.True(double.IsFinite(speedup) && serialMin > 0 && pipeMin > 0,
                    $"N={n}: bad timings.");
                if (n > 1 && speedup > ceiling * 1.15)
                    _out.WriteLine($"      WARNING: speedup {speedup:F3}x exceeds ceiling {ceiling:F3}x +15% " +
                        "— baseline likely contaminated (throttle/contention); re-run.");
            }
            _out.WriteLine("");
            _out.WriteLine($"Rig check — N=1 (same code path both ways, must be ≈1.0): {n1Speedup:F3}x");
            if (n1Speedup is < 0.90 or > 1.10)
                _out.WriteLine("  WARNING: N=1 outside [0.90,1.10] — measurement noise floor is high; " +
                    "treat batched speedups as order-of-magnitude and re-run on a quiet box.");
        }
        finally
        {
            device.Dispose();
            Environment.SetEnvironmentVariable("DOTLLM_VULKAN_DEVICE_VENDOR", priorVendor);
        }
    }

    // ── Interleaved serial-vs-pipelined measurement at a fixed split & N ──

    private (double sMin, double sMed, double sMax, double pMin, double pMed, double pMax)
        MeasureInterleaved(ScaledFixture fixture, VulkanDevice device, int split, int n,
                           string ptxDir, string spvDir)
    {
        using var model = HybridVulkanCudaTransformerModel.BuildFromPrebuiltWeights(
            fixture.Weights, fixture.Config, numVulkanLayers: split,
            vulkanDevice: device, cudaDeviceId: 0, spvDir: spvDir, ptxDir: ptxDir);

        var caches = new HybridVulkanCudaKvCache[n];
        var dIds = new int[n][];
        var dPos = new int[n][];
        try
        {
            for (int i = 0; i < n; i++)
            {
                caches[i] = model.CreateKvCache(MaxSeqLen);
                int[] pIds = [(3 + i) % VocabSize, (7 + i) % VocabSize, (1 + i) % VocabSize, (5 + i) % VocabSize];
                int[] pPos = [0, 1, 2, 3];
                using var _ = model.Forward(pIds, pPos, deviceId: -1, caches[i]);
                dIds[i] = [(2 + i) % VocabSize];
                dPos[i] = [PrefillLen];
            }

            var requests = new PipelinedRequest[n];
            for (int i = 0; i < n; i++)
                requests[i] = new PipelinedRequest { TokenIds = dIds[i], Positions = dPos[i], KvCache = caches[i] };

            // Warmup both paths.
            for (int w = 0; w < Warmup; w++)
            {
                RunSerial(model, caches, dIds, dPos, n);
                RunPipelined(model, requests);
            }

            var sSamples = new double[Repeats];
            var pSamples = new double[Repeats];
            var sw = new Stopwatch();
            for (int r = 0; r < Repeats; r++)
            {
                // Serial.
                sw.Restart(); RunSerial(model, caches, dIds, dPos, n); sw.Stop();
                sSamples[r] = sw.Elapsed.TotalMilliseconds;
                Cooldown();
                // Pipelined.
                sw.Restart(); RunPipelined(model, requests); sw.Stop();
                pSamples[r] = sw.Elapsed.TotalMilliseconds;
                Cooldown();
            }

            Array.Sort(sSamples); Array.Sort(pSamples);
            return (sSamples[0], Median(sSamples), sSamples[^1],
                    pSamples[0], Median(pSamples), pSamples[^1]);
        }
        finally
        {
            foreach (var c in caches) c?.Dispose();
        }
    }

    private void RunSerial(HybridVulkanCudaTransformerModel model, HybridVulkanCudaKvCache[] caches,
                          int[][] dIds, int[][] dPos, int n)
    {
        for (int i = 0; i < n; i++)
        {
            using var _ = model.Forward(dIds[i], dPos[i], deviceId: -1, caches[i]);
            caches[i].Rollback(PrefillLen); // keep every rep doing identical work
        }
    }

    private void RunPipelined(HybridVulkanCudaTransformerModel model, PipelinedRequest[] requests)
    {
        var results = model.ForwardBatchedPipelined(requests);
        foreach (var t in results) t.Dispose();
        foreach (var req in requests) req.KvCache?.Rollback(PrefillLen);
    }

    private static void Cooldown()
    {
        if (CooldownMs > 0) Thread.Sleep(CooldownMs);
    }

    private static double Median(double[] sorted)
    {
        int m = sorted.Length;
        return m % 2 == 0 ? (sorted[m / 2 - 1] + sorted[m / 2]) / 2.0 : sorted[m / 2];
    }

    // ── Scaled synthetic fixture ────────────────────────────────────────────

    private sealed class ScaledFixture : IDisposable
    {
        private readonly List<nint> _allocs = new();
        public ModelConfig Config = null!;
        public TransformerWeights Weights { get; private set; } = null!;

        public static ScaledFixture Build(int seed)
        {
            var b = new ScaledFixture();
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
            for (int i = 0; i < count; i++) dst[i] = ((float)rng.NextDouble() * 2f - 1f) * 0.02f;
            return ptr;
        }

        private static float[] Norm(int count, Random rng)
        {
            var arr = new float[count];
            for (int i = 0; i < count; i++) arr[i] = 1.0f + ((float)rng.NextDouble() * 2f - 1f) * 0.02f;
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
