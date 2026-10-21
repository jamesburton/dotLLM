using DotLLM.Core.Configuration;
using DotLLM.Core.Moe;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Gguf;
using DotLLM.Tests.Unit.Vulkan;
using DotLLM.Vulkan;
using Xunit;
using Xunit.Abstractions;
using Architecture = DotLLM.Core.Configuration.Architecture;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1083: cold experts on the integrated GPU (<c>--moe-cold igpu</c> / <c>cpu+igpu</c>), held to the same model with every expert
/// resident, on PR A's synthetic Bolt / Lightning fixture and on the real <c>blk9</c> truncation. Needs a CUDA device AND an
/// integrated Vulkan GPU (the dev box's RTX 3060 + Arc, in a quiet window); skips otherwise. The tolerance is PR C's
/// (<see cref="CudaNemotronHMoeColdCpuTests.AssertSameSelection"/>, <see cref="CudaNemotronHMoeColdCpuTests.AssertLogitsWithinTheCrossBackendBound"/>):
/// the iGPU rounds its activations to Q8_1 / F16 rather than the CPU's Q8_K, the same cross-backend class.
/// </summary>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronHMoeColdIgpuTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaNemotronHMoeColdIgpuTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-moe-igpu-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <param name="cold">igpu, or cpu+igpu with a fixed cost model that gives both legs work.</param>
    /// <param name="latent">true = Bolt shape (LatentMoE); false = Lightning shape.</param>
    /// <param name="blockTokens">MoE token block; 3 splits the prefill into three layer calls.</param>
    [SkippableTheory]
    [InlineData(MoeColdMode.Igpu, true, 512)]
    [InlineData(MoeColdMode.Igpu, false, 3)]
    [InlineData(MoeColdMode.CpuIgpu, true, 512)]
    [InlineData(MoeColdMode.CpuIgpu, false, 512)]
    public void ColdIgpu_SmallCache_MatchesResident_SameSelection(MoeColdMode cold, bool latent, int blockTokens)
    {
        string ptxDir = RequireCudaAndIntegratedGpu();
        string path = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent, zeroBias: false, f32Banks: false);
        string noBias = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent, zeroBias: true, f32Banks: false);
        string exact = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent, zeroBias: false, f32Banks: true);
        float[] cpu = CudaNemotronHMoeColdCpuTests.RunCpu(path);
        float[] cpuExact = CudaNemotronHMoeColdCpuTests.RunCpu(exact);
        float control = CudaNemotronHMoeColdCpuTests.MaxDiff(cpu, CudaNemotronHMoeColdCpuTests.RunCpu(noBias));
        float cpuVsExact = CudaNemotronHMoeColdCpuTests.MaxDiff(cpu, cpuExact);
        Assert.True(control > 0.05f, $"fixture insensitive to the selection bias (max logit delta {control:E3})");

        int savedBlock = CudaNemotronHTransformerModel.MoeBlockTokens;
        CudaNemotronHTransformerModel.MoeBlockTokens = blockTokens;
        try
        {
            var residentTrace = new List<(int, int[])>();
            CudaNemotronHTransformerModel.MoeRoutingTraceHook = (l, ids) => residentTrace.Add((l, ids));
            float[] resident = CudaNemotronHMoeColdCpuTests.Run(path, ptxDir, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, out _, out _);

            var options = Options(cold);
            if (cold == MoeColdMode.CpuIgpu)
            {
                // Equal per-expert costs, a free submission and one CPU worker: every multi-expert cold set is split across both legs.
                options = options with
                {
                    CostModelOverride = new MoeCostModel(1e6, 0,
                        [.. CudaNemotronHMoeColdCpuTests.ExpertLayouts(path).Select(l =>
                            new MoeLayoutCost(l, new LinearCost(100, 10), new LinearCost(1, 0), new LinearCost(100, 10)))],
                        IgpuSubmitUs: 0),
                };
            }

            var coldTrace = new List<(int, int[])>();
            CudaNemotronHTransformerModel.MoeRoutingTraceHook = (l, ids) => coldTrace.Add((l, ids));
            var start = MoeTelemetry.Snapshot();
            float[] coldLogits = CudaNemotronHMoeColdCpuTests.Run(path, ptxDir, options, out var info, out var cache);
            var tel = MoeTelemetry.Snapshot().Since(start);

            Assert.NotNull(info);
            Assert.Equal(cold, info!.Cold);
            Assert.NotNull(info.IgpuDevice);
            _output.WriteLine($"{MoeOffloadOptions.FormatCold(cold)} latent={latent} block={blockTokens}: iGPU {info.IgpuDevice} " +
                $"({info.IgpuLayers} layers), cache {info.CacheSlots} slots, hits {cache.Hits}, cold misses {cache.ColdMisses}, " +
                $"promotions {cache.Promotions}, CPU {tel.ColdExpertsCpu} / iGPU {tel.ColdExpertsIgpu} of {tel.Routed}; {tel.FormatSummary()}");

            CudaNemotronHMoeColdCpuTests.AssertSameSelection(_output, residentTrace, coldTrace);
            AssertPinnedBanksImportedZeroCopy(info);
            Assert.True(tel.ColdExpertsIgpu > 0, "no expert ran on the iGPU");
            if (cold == MoeColdMode.CpuIgpu) Assert.True(tel.ColdExpertsCpu > 0, "cpu+igpu never used the CPU leg");
            // igpu: only the layers the iGPU refused (none on a device with integer dot product) fall back to the CPU.
            if (cold == MoeColdMode.Igpu && info.IgpuLayers == 4) Assert.Equal(0, tel.ColdExpertsCpu);
            Assert.True(cache.ColdMisses > 0, "no cold miss");
            Assert.Equal(0, cache.Insertions + cache.Bypasses);   // never copies on a miss
            Assert.Equal(tel.Routed, cache.Hits + cache.Misses);
            Assert.Equal(cache.ColdMisses, tel.ColdExpertsCpu + tel.ColdExpertsIgpu);
            Assert.True(tel.IgpuComputeMicros > 0, "no iGPU compute time recorded");

            CudaNemotronHMoeColdCpuTests.AssertLogitsWithinTheCrossBackendBound(_output, resident, coldLogits, cpuExact, control, cpuVsExact);
        }
        finally
        {
            CudaNemotronHTransformerModel.MoeBlockTokens = savedBlock;
            CudaNemotronHTransformerModel.MoeRoutingTraceHook = null;
        }
    }

    /// <summary>
    /// With a pinned host store the iGPU must import the pinned banks, not the mmap (#1083 window 2): the pinned copy is private
    /// read-write memory, which the Arc imports zero-copy, whereas the default read-only GGUF mapping is refused and staged, a third copy
    /// of the banks in RAM. So every bank of every iGPU layer is imported and none is staged, whatever the GGUF's mapping mode.
    /// </summary>
    /// <param name="info">The offloaded model's info.</param>
    internal static void AssertPinnedBanksImportedZeroCopy(CudaMoeOffloadInfo info)
    {
        if (!info.PinnedStore) return;
        Assert.Equal(0, info.IgpuStagedBanks);
        Assert.Equal(0L, info.IgpuStagedBytes);
        Assert.Equal(2 * info.IgpuLayers, info.IgpuImportedBanks);
    }

    /// <summary>
    /// Auto with a fixed cost model that has no iGPU costs never opens the iGPU (no device, no leg, bit-identical to the CPU-only
    /// auto of #1071); with one where the iGPU is cheapest it puts the cold experts there.
    /// </summary>
    [SkippableFact]
    public void Auto_OpensTheIgpuOnlyWhenItsCostsCanWin()
    {
        string ptxDir = RequireCudaAndIntegratedGpu();
        string path = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent: true, zeroBias: false, f32Banks: false);
        var layouts = CudaNemotronHMoeColdCpuTests.ExpertLayouts(path);

        var noIgpu = Options(MoeColdMode.Auto) with
        {
            CostModelOverride = new MoeCostModel(1e6, 0, [.. layouts.Select(l => new MoeLayoutCost(l, new LinearCost(0, 0), new LinearCost(1, 0)))]),
        };
        CudaNemotronHMoeColdCpuTests.Run(path, ptxDir, noIgpu, out var info, out _);
        Assert.Null(info!.IgpuDevice);

        var igpuWins = Options(MoeColdMode.Auto) with
        {
            CostModelOverride = new MoeCostModel(1e6, 0,
                [.. layouts.Select(l => new MoeLayoutCost(l, new LinearCost(1e6, 0), new LinearCost(1, 0), new LinearCost(1, 0)))],
                IgpuSubmitUs: 0),
        };
        var start = MoeTelemetry.Snapshot();
        CudaNemotronHMoeColdCpuTests.Run(path, ptxDir, igpuWins, out info, out var cache);
        var tel = MoeTelemetry.Snapshot().Since(start);
        Assert.NotNull(info!.IgpuDevice);
        Assert.True(tel.ColdExpertsIgpu > 0);
        Assert.Equal(0, cache.Insertions + cache.Bypasses);
    }

    /// <summary>
    /// Real weights (<c>DOTLLM_BOLT_BLK9_GGUF</c>, NVIDIA's IQ2_XXS experts): iGPU cold experts within PR A's real-model bounds of
    /// resident (mean KL, top-1), prefill (one 16-token layer call per MoE layer) and four decode steps.
    /// </summary>
    [SkippableTheory]
    [Trait("Category", "RealModel")]
    [InlineData(MoeColdMode.Igpu, 512)]
    [InlineData(MoeColdMode.Igpu, 4)]
    [InlineData(MoeColdMode.CpuIgpu, 512)]
    public void Blk9_ColdIgpu_WithinTheCrossBackendBoundOfResident(MoeColdMode cold, int blockTokens)
    {
        string? path = Environment.GetEnvironmentVariable("DOTLLM_BOLT_BLK9_GGUF");
        Skip.If(string.IsNullOrEmpty(path) || !File.Exists(path), "DOTLLM_BOLT_BLK9_GGUF not set or file missing");
        string ptxDir = RequireCudaAndIntegratedGpu();

        int savedBlock = CudaNemotronHTransformerModel.MoeBlockTokens;
        CudaNemotronHTransformerModel.MoeBlockTokens = blockTokens;
        try
        {
            using var gguf = GgufFile.Open(path!);
            var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
            Assert.Equal(Architecture.NemotronHMoe, config.Architecture);
            var (resident, decodeTokens) = CudaNemotronHBoltBlk9ColdCpuRealModelTests.Run(gguf, config, ptxDir,
                new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, null, out _, out _);
            var options = Options(cold) with { CacheMib = 256, ModelPath = path };
            var start = MoeTelemetry.Snapshot();
            var (coldRows, _) = CudaNemotronHBoltBlk9ColdCpuRealModelTests.Run(gguf, config, ptxDir, options, decodeTokens, out var info, out var cache);
            var tel = MoeTelemetry.Snapshot().Since(start);
            _output.WriteLine($"{MoeOffloadOptions.FormatCold(cold)} block {blockTokens}: iGPU {info.IgpuDevice} ({info.IgpuLayers} layers), " +
                $"{info.CacheSlots} slots, hits {cache.Hits}, cold {cache.ColdMisses}; {tel.FormatSummary()}");
            Assert.NotNull(info.IgpuDevice);
            AssertPinnedBanksImportedZeroCopy(info);
            Assert.True(tel.ColdExpertsIgpu > 0 || cold == MoeColdMode.CpuIgpu, "no expert ran on the iGPU");
            Assert.True(tel.ColdExpertsIgpu + tel.ColdExpertsCpu > 0, "nothing ran cold");

            int rows = resident.Count, agree = 0;
            double klSum = 0;
            for (int r = 0; r < rows; r++)
            {
                klSum += CudaNemotronHBoltBlk9ColdCpuRealModelTests.Kl(resident[r], coldRows[r]);
                if (CudaNemotronHBoltBlk9ColdCpuRealModelTests.ArgMax(resident[r]) == CudaNemotronHBoltBlk9ColdCpuRealModelTests.ArgMax(coldRows[r])) agree++;
            }
            double meanKl = klSum / rows, top1 = (double)agree / rows;
            _output.WriteLine($"mean KL {meanKl:E4}, top-1 {agree}/{rows}");
            Assert.True(meanKl <= 0.02, $"mean KL(resident || iGPU cold) {meanKl:E4} > 0.02");
            Assert.True(top1 >= 0.9, $"top-1 agreement {top1:P1} < 90%");
        }
        finally
        {
            CudaNemotronHTransformerModel.MoeBlockTokens = savedBlock;
        }
    }

    private MoeOffloadOptions Options(MoeColdMode cold) => new()
    {
        Mode = MoeOffloadMode.On,
        Cold = cold,
        CacheMib = 1,
        // One CPU worker: with equal costs the balance then splits any two-expert cold set across the CPU and the iGPU.
        CpuThreads = 1,
        CalibrationPath = Path.Combine(_scratch, "moe-calibration.json"),
        Log = _output.WriteLine,
    };

    /// <summary>Skips unless a CUDA device and an integrated (non-NVIDIA) Vulkan GPU both exist.</summary>
    private static string RequireCudaAndIntegratedGpu()
    {
        string ptxDir = CudaNemotronHMoeColdCpuTests.RequireCuda();
        VulkanMatMulF32KernelTests.SkipIfUnavailable(out _);
        Skip.If(VulkanIgpuDeviceSelector.Select(VulkanDevice.EnumeratePhysicalDevices(),
            Environment.GetEnvironmentVariable(MoeOffloadOptions.IgpuDeviceEnvVar), out string reason) is null, $"no integrated Vulkan GPU: {reason}");
        return ptxDir;
    }
}
