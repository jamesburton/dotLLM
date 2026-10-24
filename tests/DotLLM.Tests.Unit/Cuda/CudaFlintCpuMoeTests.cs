using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Models.Flint;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// CPU-expert MoE layers on the generic CUDA host (<c>--cpu-moe</c> / <c>--n-cpu-moe</c>, #1187) on the hermetic
/// <see cref="SyntheticFlintGguf"/> with routed Q4_1, shared Q8_0 and F32 attention / dense (MoE layers 2-5, layer 4 a
/// NoPE full layer, ungated shared expert, sinks, window 4 over 16 tokens). The CPU leg runs the CPU model's own routed
/// kernels (Q8_1 activations) on the mmap, so against the CPU model on the same file the remaining difference is CUDA's
/// FP16 residual stream and how that rounds the Q8_1 expert inputs. Prefill plus 6 decode steps through the CUDA KV
/// cache. A fully resident CUDA run is printed for reference: it runs the routed experts on F32 activations, so its distance to the CPU model is the Q8_1 activation noise
/// (and, on this random-weight model, an occasional routing near-tie).
/// </summary>
/// <remarks>
/// The bound can see a dropped or misplaced routed contribution: <c>CudaFlintParitySyntheticTests</c> shows the top-k
/// renormalisation alone moves these logits by more than 5x the bound, and renormalisation only acts through the routed sum.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaFlintCpuMoeTests : IDisposable
{
    /// <summary>
    /// CPU-expert CUDA vs the CPU model. Both quantise the expert inputs to Q8_1, but CUDA's inputs went through the FP16
    /// residual first, so the rounding does not cancel: measured up to 7.1e-2 (RTX 3060). Each Flint feature moves these
    /// logits by 3.0-9.1 (<c>CudaFlintParitySyntheticTests</c>), 30x above this bound.
    /// </summary>
    private const float AbsTol = 1e-1f;

    private const int PrefillTokens = 10;

    private static readonly int[] Ids = FlintSyntheticGoldenTests.Ids;

    private readonly ITestOutputHelper _out;
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"dotllm-flint-cpumoe-{Guid.NewGuid():N}.gguf");

    public CudaFlintCpuMoeTests(ITestOutputHelper output)
    {
        _out = output;
        File.WriteAllBytes(_path, SyntheticFlintGguf.Build(FlintSyntheticGoldenTests.Quantized));
    }

    public void Dispose()
    {
        try { File.Delete(_path); } catch { /* best effort */ }
    }

    /// <summary>
    /// Placements: every MoE layer on the CPU, a split (layers 2-3 CPU, 4-5 device), and <c>--moe-offload on</c>; each with
    /// the prefill GPU leg (#1202) off, on with its cost model, and forced to split every CPU layer half / half in waves of
    /// two experts (several waves per layer, ring slots reused across layers, the next layer prefetched, a CPU part).
    /// </summary>
    public static TheoryData<string, string> Placements
    {
        get
        {
            var data = new TheoryData<string, string>();
            foreach (string p in new[] { "cpu-moe", "n-cpu-moe-4", "offload-on" })
                foreach (string stream in new[] { "stream-off", "stream-auto", "stream-split" })
                    data.Add(p, stream);

            // The CPU part's F16 upload with its power-of-two scale (forced: these logits never reach the real limit).
            data.Add("cpu-moe", "stream-split-scaled");
            return data;
        }
    }

    [SkippableTheory]
    [MemberData(nameof(Placements))]
    public void CpuExpertLayers_MatchCpuModel_PrefillAndDecode(string placement, string stream)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(_path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        float[] cpu;
        using (var cpuModel = TransformerModel.LoadFromGguf(gguf, cfg, ThreadingConfig.SingleThreaded))
        using (ITensor t = cpuModel.Forward(Ids, Enumerable.Range(0, Ids.Length).ToArray(), deviceId: -1))
            cpu = Copy(t);
        List<float[]> resident = RunCuda(gguf, cfg, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, null);
        int vocab = SyntheticFlintGguf.Vocab;

        var options = placement switch
        {
            "cpu-moe" => new MoeOffloadOptions { CpuMoeLayers = MoeOffloadOptions.AllLayers, CpuThreads = 3 },
            "n-cpu-moe-4" => new MoeOffloadOptions { CpuMoeLayers = 4, CpuThreads = 1 },
            _ => new MoeOffloadOptions { Mode = MoeOffloadMode.On, CpuThreads = 2 },
        };
        int expectedCpuLayers = placement == "n-cpu-moe-4" ? 2 : 4;

        List<float[]> offloaded;
        bool streamWas = CudaTransformerModel.StreamCpuExperts;
        try
        {
            CudaTransformerModel.StreamCpuExperts = stream != "stream-off";
            if (stream.StartsWith("stream-split", StringComparison.Ordinal))
            {
                CudaMoeExpertStream.TestWaveExperts = 2;
                CudaMoeExpertStream.TestSplit = experts => experts / 2;
            }
            if (stream == "stream-split-scaled")
                CudaMoeExpertStream.CpuPartF16Limit = 1e-3f;
            offloaded = RunCuda(gguf, cfg, options, expectedCpuLayers, stream);
        }
        finally
        {
            CudaTransformerModel.StreamCpuExperts = streamWas;
            CudaMoeExpertStream.TestWaveExperts = null;
            CudaMoeExpertStream.TestSplit = null;
            CudaMoeExpertStream.CpuPartF16Limit = 16384f;
        }
        for (int r = 0; r < offloaded.Count; r++)
        {
            int at = PrefillTokens - 1 + r;
            string what = $"{placement} {stream} {(r == 0 ? "prefill" : "decode")}@{at}";
            _out.WriteLine($"{what,-28} resident vs CPU model {MaxAbs(cpu.AsSpan(at * vocab, vocab), resident[r]):E3}, "
                + $"vs CPU experts {MaxAbs(resident[r], offloaded[r]):E3}");
            AssertMatch(what, cpu.AsSpan(at * vocab, vocab).ToArray(), offloaded[r]);
        }
    }

    /// <summary>Prefill of <see cref="PrefillTokens"/> then one-token decodes; returns the last row of each call.</summary>
    private static List<float[]> RunCuda(GgufFile gguf, ModelConfig cfg, MoeOffloadOptions options, int? expectedCpuLayers,
        string stream = "stream-off", string? embed = null, Action<CudaTransformerModel>? check = null)
    {
        var rows = new List<float[]>();
        var (model, kvFactory) = CudaModelLoader.CreateFromGguf(gguf, cfg, deviceId: 0, moeOffload: options);
        using (model)
        {
            var cuda = Assert.IsType<CudaTransformerModel>(model);
            Assert.Equal(expectedCpuLayers is null, cuda.CpuMoeLayers == 0);
            if (embed is not null)
                Assert.Equal(embed == "host", cuda.TokenEmbeddingsOnHost);
            using var kv = kvFactory(64);
            int[] pos = Enumerable.Range(0, Ids.Length).ToArray();
            using (ITensor p = cuda.Forward(Ids.AsSpan(0, PrefillTokens), pos.AsSpan(0, PrefillTokens), deviceId: -1, kv))
                rows.Add(Copy(p));
            // Prefill: the CPU layers' experts ran on the CPU leg, the GPU leg, or (forced split) both.
            Assert.Equal(expectedCpuLayers is not null, cuda.CpuMoeExpertsComputed + cuda.CpuMoeExpertsStreamed > 0);
            if (expectedCpuLayers is not null && stream == "stream-off")
                Assert.Equal(0, cuda.CpuMoeExpertsStreamed);
            if (expectedCpuLayers is not null && stream.StartsWith("stream-split", StringComparison.Ordinal))
                Assert.True(cuda.CpuMoeExpertsComputed > 0 && cuda.CpuMoeExpertsStreamed > 0, "both legs must run");
            long streamedAfterPrefill = cuda.CpuMoeExpertsStreamed;
            for (int i = PrefillTokens; i < Ids.Length; i++)
            {
                long before = cuda.CpuMoeExpertsComputed;
                using ITensor d = cuda.Forward(Ids.AsSpan(i, 1), pos.AsSpan(i, 1), deviceId: -1, kv);
                rows.Add(Copy(d));

                // One decode token routes TopK distinct experts in each CPU layer, all on the CPU leg.
                Assert.Equal((expectedCpuLayers ?? 0) * SyntheticFlintGguf.TopK, cuda.CpuMoeExpertsComputed - before);
                Assert.Equal(streamedAfterPrefill, cuda.CpuMoeExpertsStreamed);
            }
            check?.Invoke(cuda);
        }
        return rows;
    }

    private static float MaxAbs(ReadOnlySpan<float> a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < b.Length; i++) m = MathF.Max(m, MathF.Abs(a[i] - b[i]));
        return m;
    }

    [Fact]
    public void Plan_ExplicitOnOffAndAutoFit()
    {
        long[] routed = [0, 0, 100, 100, 100, 100];
        var auto = MoeOffloadOptions.Default;
        Assert.Equal(3, CudaTransformerModel.PlanCpuMoeLayers(auto with { CpuMoeLayers = 3 }, routed, true, 1000, 10, 0));
        Assert.Equal(MoeOffloadOptions.AllLayers,
            CudaTransformerModel.PlanCpuMoeLayers(auto with { Mode = MoeOffloadMode.On }, routed, true, 500, 10_000, 0));
        Assert.Equal(0, CudaTransformerModel.PlanCpuMoeLayers(auto with { Mode = MoeOffloadMode.Off }, routed, true, 500, 10, 0));

        // Fits: nothing moves. Over by 150: layers 2 and 3 move (the first layer that frees enough is index 3, so N = 4).
        Assert.Equal(0, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 600, 50));
        Assert.Equal(4, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 400, 50));
        Assert.Equal(3, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 450, 50));

        // Even every expert on the CPU does not fit; auto still moves all of them. Not eligible: auto never moves any.
        Assert.Equal(MoeOffloadOptions.AllLayers, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, true, 500, 50, 50));
        Assert.Equal(0, CudaTransformerModel.PlanCpuMoeLayers(auto, routed, false, 500, 50, 50));
    }

    /// <summary>
    /// Device-routed decode (#1182): the resident grouped Q4_1 decode body with pointer arrays built on the device from the
    /// top-k ids must give logits bit-identical to the host-routed body (same kernels, same expert order).
    /// </summary>
    [SkippableFact]
    public void ResidentDecode_DeviceRouting_BitIdenticalToHostRouting()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(_path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var off = new MoeOffloadOptions { Mode = MoeOffloadMode.Off };
        long before = CudaMoeFfn.DeviceRoutedCalls;
        List<float[]> device = RunCuda(gguf, cfg, off, null);
        Assert.True(CudaMoeFfn.DeviceRoutedCalls > before, "the device-routed body did not run");
        List<float[]> host;
        CudaMoeFfn.DeviceRoutingEnabled = false;
        try
        {
            host = RunCuda(gguf, cfg, off, null);
        }
        finally
        {
            CudaMoeFfn.DeviceRoutingEnabled = true;
        }
        Assert.Equal(host.Count, device.Count);
        for (int r = 0; r < host.Count; r++)
            for (int i = 0; i < host[r].Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(host[r][i]) == BitConverter.SingleToInt32Bits(device[r][i]),
                    $"row {r} logit {i}: device-routed {device[r][i]} vs host-routed {host[r][i]}");
    }

    /// <summary>
    /// Host-gathered F32 token embeddings (#1183): the table stays in the mmap and the forward uploads the rows as F16. The
    /// logits must be bit-identical to the device lookup's (same F32 to F16 rounding), and the table must not be on the device.
    /// </summary>
    [SkippableFact]
    public void HostTokenEmbeddings_BitIdenticalToDeviceTable()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(_path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var options = new MoeOffloadOptions { CpuMoeLayers = 4, CpuThreads = 1 };
        List<float[]> host = RunCuda(gguf, cfg, options, 2, stream: "stream-auto", embed: "host");
        List<float[]> device;
        Environment.SetEnvironmentVariable("DOTLLM_CUDA_HOST_EMBED", "0");
        try
        {
            device = RunCuda(gguf, cfg, options, 2, stream: "stream-auto", embed: "device");
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_CUDA_HOST_EMBED", null);
        }
        for (int r = 0; r < host.Count; r++)
            for (int i = 0; i < host[r].Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(host[r][i]) == BitConverter.SingleToInt32Bits(device[r][i]),
                    $"row {r} logit {i}: host-gathered {host[r][i]} vs device table {device[r][i]}");
    }

    /// <summary>
    /// Runtime VRAM (#1211): the FP16 forward allocates no F32 activation shadows and never the dequant scratch (every Flint
    /// projection runs in place), and the dense / shared-expert weights are packed into the arena. Packing changes only where
    /// the tensors live, so the logits must be bit-identical to one allocation per tensor.
    /// </summary>
    [SkippableFact]
    public void Fp16Forward_NoF32ShadowsNoDequantScratch_ArenaBitIdentical()
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(_path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var options = new MoeOffloadOptions { CpuMoeLayers = 4, CpuThreads = 1 };
        List<float[]> packed = RunCuda(gguf, cfg, options, 2, stream: "stream-auto", check: m =>
        {
            Assert.Equal(0, m.ForwardState.HiddenStateF32);
            Assert.Equal(0, m.ForwardState.ResidualF32);
            Assert.Equal(0, m.ForwardState.SiluOutputF32);
            Assert.False(m.ForwardState.DequantScratchAllocated, "no Flint projection dequantises");
            Assert.True(m.DeviceWeights.ArenaUsedBytes > 0, "the weights were not packed");
        });
        List<float[]> separate;
        Environment.SetEnvironmentVariable("DOTLLM_CUDA_WEIGHT_ARENA", "0");
        try
        {
            separate = RunCuda(gguf, cfg, options, 2, stream: "stream-auto", check: m => Assert.Equal(0, m.DeviceWeights.ArenaUsedBytes));
        }
        finally
        {
            Environment.SetEnvironmentVariable("DOTLLM_CUDA_WEIGHT_ARENA", null);
        }
        for (int r = 0; r < packed.Count; r++)
            for (int i = 0; i < packed[r].Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(packed[r][i]) == BitConverter.SingleToInt32Bits(separate[r][i]),
                    $"row {r} logit {i}: arena {packed[r][i]} vs one allocation per tensor {separate[r][i]}");
    }

    /// <summary>
    /// The captured MoE decode (#1210) replays the eager body: resident layers device-routed, CPU-expert layers as host-function
    /// nodes, the NoPE full layer, sinks and the 4-token window read from the same kernels with device-side position and
    /// length. Decode steps 2-6 run from the graph (step 1 is the eager warm-up that captures nothing) and must give logits
    /// bit-identical to the eager body's, for every placement.
    /// </summary>
    [SkippableTheory]
    [InlineData("resident")]
    [InlineData("n-cpu-moe-4")]
    [InlineData("cpu-moe")]
    public void MoeDecodeGraph_BitIdenticalToEager(string placement)
    {
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");
        using var gguf = GgufFile.Open(_path);
        ModelConfig cfg = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (options, cpuLayers) = placement switch
        {
            "resident" => (new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, (int?)null),
            "n-cpu-moe-4" => (new MoeOffloadOptions { CpuMoeLayers = 4, CpuThreads = 2 }, 2),
            _ => (new MoeOffloadOptions { CpuMoeLayers = MoeOffloadOptions.AllLayers, CpuThreads = 1 }, 4),
        };
        int decodeSteps = Ids.Length - PrefillTokens;
        List<float[]> graph = RunCuda(gguf, cfg, options, cpuLayers, stream: "stream-auto", check: m =>
        {
            Assert.Equal(1, m.GraphCaptureCount);
            Assert.Equal(decodeSteps - 1, m.GraphReplayCount);
            Assert.Equal(1, m.EagerDecodeCount);
        });
        List<float[]> eager;
        CudaTransformerModel.MoeDecodeGraphEnabled = false;
        try
        {
            eager = RunCuda(gguf, cfg, options, cpuLayers, stream: "stream-auto", check: m => Assert.Equal(0, m.GraphReplayCount));
        }
        finally
        {
            CudaTransformerModel.MoeDecodeGraphEnabled = true;
        }
        Assert.Equal(eager.Count, graph.Count);
        for (int r = 0; r < eager.Count; r++)
            for (int i = 0; i < eager[r].Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(eager[r][i]) == BitConverter.SingleToInt32Bits(graph[r][i]),
                    $"{placement} row {r} logit {i}: graph {graph[r][i]} vs eager {eager[r][i]}");
    }

    [Fact]
    public void PlanSplit_BalancesLinkAgainstCpu()
    {
        // 4 experts with 1, 1, 10, 10 assignments.
        int[] offsets = [0, 1, 2, 12, 22];

        // Free link: everything on the GPU. Free CPU: nothing on the GPU.
        Assert.Equal(4, CudaMoeExpertStream.PlanSplit(offsets, 4, 0, expertMs: 0.001, cpuMsPerAssignment: 1));
        Assert.Equal(0, CudaMoeExpertStream.PlanSplit(offsets, 4, 0, expertMs: 1, cpuMsPerAssignment: 0.001));

        // 1 ms per expert vs 0.1 ms per assignment: split 3 costs max(3, 1.0) = 3, split 2 max(2, 2.0) = 2, split 1
        // max(1, 2.1) = 2.1. Prefetched experts stay on the GPU even when that costs more.
        Assert.Equal(2, CudaMoeExpertStream.PlanSplit(offsets, 4, 0, expertMs: 1, cpuMsPerAssignment: 0.1));
        Assert.Equal(3, CudaMoeExpertStream.PlanSplit(offsets, 4, 3, expertMs: 1, cpuMsPerAssignment: 0.1));

        // Experts without rows cost no link time.
        int[] sparse = [0, 0, 0, 10, 10];
        Assert.Equal(3, CudaMoeExpertStream.PlanSplit(sparse, 4, 0, expertMs: 1, cpuMsPerAssignment: 1));
    }

    private void AssertMatch(string what, float[] cpu, float[] offloaded)
    {
        Assert.Equal(cpu.Length, offloaded.Length);
        Assert.All(offloaded, v => Assert.True(float.IsFinite(v), $"{what}: non-finite logit"));
        float diff = MaxAbs(cpu, offloaded);
        _out.WriteLine($"{what,-28} max |CPU model - CPU experts| = {diff:E3}");
        Assert.True(diff <= AbsTol, $"{what}: max |CPU model - CPU experts| = {diff:E3} > {AbsTol:E3}");
    }

    private static unsafe float[] Copy(ITensor t)
    {
        var f = new float[checked((int)t.Shape.ElementCount)];
        new ReadOnlySpan<float>((void*)t.DataPointer, f.Length).CopyTo(f);
        return f;
    }
}
