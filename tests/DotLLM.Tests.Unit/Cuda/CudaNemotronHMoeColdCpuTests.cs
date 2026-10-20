using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Moe;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using DotLLM.Models.Architectures;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Unit.Cuda;

/// <summary>
/// Issue #1071: cold experts on the CPU at the latent cut, on the synthetic Bolt / Lightning fixture of
/// <see cref="CudaNemotronHMoeParityTests"/>, held to the same model with every expert resident.
/// <list type="bullet">
///   <item>The routing is the host's either way, so the per-layer top-k selection must be identical (routing hook).</item>
///   <item>The CPU experts run the CPU model's kernels (Q8_K-quantised activations), so the logits differ from the resident run by
///   the cross-backend amount: bounded like PR A's CUDA-vs-CPU parity, and small against the selection-bias control.</item>
///   <item>Static placement (<c>--n-cpu-moe</c>, <c>--cpu-moe</c>) keeps those layers out of the cache entirely.</item>
///   <item><c>--moe-cold auto</c> with a CPU that never wins computes nothing on the CPU, so it is bit-identical to resident.</item>
/// </list>
/// </summary>
/// <remarks>
/// The fixture has MoE layers at 1, 2, 4 and 5 (6 / 8 / 5 / 7 experts). A 1 MiB cache holds a fraction of them, so decode both
/// hits (promoted experts) and computes misses on the CPU. Calibration writes to a scratch file, never to %LOCALAPPDATA%.
/// </remarks>
[Trait("Category", "GPU")]
[Collection(CudaCollection.Name)]
public sealed class CudaNemotronHMoeColdCpuTests : IDisposable
{
    private const int Vocab = 16;
    private static readonly int[] Prompt = [3, 9, 1, 14, 6, 0, 11];
    private static readonly int[] DecodeTokens = [5, 2, 13, 8, 0, 15, 7, 4];

    private readonly ITestOutputHelper _output;
    private readonly string _scratch;

    public CudaNemotronHMoeColdCpuTests(ITestOutputHelper output)
    {
        _output = output;
        _scratch = Path.Combine(Path.GetTempPath(), $"dotllm-cuda-moe-cold-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_scratch);
    }

    public void Dispose()
    {
        try { Directory.Delete(_scratch, recursive: true); } catch { /* best-effort */ }
    }

    /// <param name="latent">true = Bolt shape (LatentMoE); false = Lightning shape.</param>
    /// <param name="blockTokens">MoE token block; 3 splits the prefill into three layer calls.</param>
    /// <param name="threads">CPU cold-leg threads (1 = the forward's own thread only).</param>
    /// <param name="pinned">true = pinned host store; false = memory-mapped store through the staging ring.</param>
    [SkippableTheory]
    [InlineData(true, 512, 3, true)]
    [InlineData(true, 3, 1, false)]
    [InlineData(false, 512, 4, true)]
    public void ColdCpu_SmallCache_MatchesResident_SameSelection(bool latent, int blockTokens, int threads, bool pinned)
    {
        string ptxDir = RequireCuda();
        string path = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent, zeroBias: false, f32Banks: false);
        string noBias = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent, zeroBias: true, f32Banks: false);
        string exact = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent, zeroBias: false, f32Banks: true);
        float[] cpu = RunCpu(path), cpuNoBias = RunCpu(noBias), cpuExact = RunCpu(exact);
        float control = MaxDiff(cpu, cpuNoBias);
        float cpuVsExact = MaxDiff(cpu, cpuExact);
        Assert.True(control > 0.05f, $"fixture insensitive to the selection bias (max logit delta {control:E3})");

        int savedBlock = CudaNemotronHTransformerModel.MoeBlockTokens;
        CudaNemotronHTransformerModel.MoeBlockTokens = blockTokens;
        try
        {
            var residentTrace = new List<(int, int[])>();
            CudaNemotronHTransformerModel.MoeRoutingTraceHook = (l, ids) => residentTrace.Add((l, ids));
            float[] resident = Run(path, ptxDir, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, out _, out _);

            var coldTrace = new List<(int, int[])>();
            CudaNemotronHTransformerModel.MoeRoutingTraceHook = (l, ids) => coldTrace.Add((l, ids));
            var start = MoeTelemetry.Snapshot();
            float[] cold = Run(path, ptxDir, Options(MoeColdMode.Cpu, threads, pinned), out var info, out var cache);
            var tel = MoeTelemetry.Snapshot().Since(start);

            Assert.NotNull(info);
            Assert.Equal(MoeColdMode.Cpu, info!.Cold);
            Assert.Equal(threads, info.CpuThreads);
            Assert.Equal(pinned, info.PinnedStore);
            _output.WriteLine($"latent={latent} block={blockTokens} threads={threads} pinned={pinned}: cache {info.CacheSlots} slots, " +
                $"hits {cache.Hits}, cold misses {cache.ColdMisses}, promotions {cache.Promotions}, CPU experts {tel.ColdExpertsCpu} " +
                $"of {tel.Routed}; {tel.FormatSummary()}");

            // Selection: identical per layer call (the host routes either way).
            Assert.Equal(residentTrace.Count, coldTrace.Count);
            for (int i = 0; i < residentTrace.Count; i++)
            {
                Assert.Equal(residentTrace[i].Item1, coldTrace[i].Item1);
                Assert.Equal(residentTrace[i].Item2, coldTrace[i].Item2);
            }

            // The CPU leg ran, the cache was used and promote-only admission copied something in.
            Assert.True(tel.ColdExpertsCpu > 0, "no expert ran on the CPU");
            Assert.True(cache.ColdMisses > 0, "no cold miss");
            Assert.True(cache.Promotions > 0, "promote-only admission never promoted");
            Assert.Equal(0, cache.Insertions + cache.Bypasses);   // cpu mode never copies on miss
            Assert.Equal(tel.Routed, cache.Hits + cache.Misses);   // every routed expert went through the cache
            Assert.Equal(cache.ColdMisses, tel.ColdExpertsCpu);

            // Logits: the cross-backend bound (PR A: CUDA vs the CPU oracle is small against the bias control), and the CPU leg
            // cannot be further from the resident run than the CPU oracle is from the exact F32-bank oracle plus CUDA's own error.
            float vsResident = MaxDiff(resident, cold);
            float residentVsExact = MaxDiff(resident, cpuExact);
            _output.WriteLine($"|cold-resident| {vsResident:E3}, |resident-exact| {residentVsExact:E3}, |cpu-exact| {cpuVsExact:E3}, " +
                $"control {control:E3}");
            Assert.True(vsResident * 4 < control, $"|cold - resident| {vsResident:E3} is not small against the bias effect {control:E3}");
            Assert.True(vsResident <= 2 * (cpuVsExact + residentVsExact) + 1e-3f,
                $"|cold - resident| {vsResident:E3} exceeds the CPU and CUDA oracle errors ({cpuVsExact:E3} + {residentVsExact:E3})");
            Assert.True(vsResident > 0, "bit-identical to resident: the CPU leg did not run its own kernels");
            for (int row = 0; row < resident.Length / Vocab; row++)
                Assert.Equal(ArgMax(resident, row), ArgMax(cold, row));
        }
        finally
        {
            CudaNemotronHTransformerModel.MoeBlockTokens = savedBlock;
            CudaNemotronHTransformerModel.MoeRoutingTraceHook = null;
        }
    }

    [SkippableFact]
    public void StaticCpuLayers_NeverReachTheCache_AndCpuMoeCachesNothing()
    {
        string ptxDir = RequireCuda();
        string path = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent: true, zeroBias: false, f32Banks: false);
        string noBias = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent: true, zeroBias: true, f32Banks: false);
        float control = MaxDiff(RunCpu(path), RunCpu(noBias));
        float[] resident = Run(path, ptxDir, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, out _, out _);

        var lookups = new List<int>();
        CudaNemotronHTransformerModel.MoeLookupTraceHook = (layer, _) => lookups.Add(layer);
        try
        {
            // --n-cpu-moe 3: layers 1 and 2 (indices below 3) are static CPU layers; layers 4 and 5 copy on miss. Mode auto: the
            // static layers alone force offload of a model that fits.
            var start = MoeTelemetry.Snapshot();
            var options = Options(MoeColdMode.Copy, threads: 2, pinned: true) with { Mode = MoeOffloadMode.Auto, CpuMoeLayers = 3 };
            float[] nCpu = Run(path, ptxDir, options, out var info, out var cache);
            var tel = MoeTelemetry.Snapshot().Since(start);
            Assert.NotNull(info);
            Assert.Equal(2, info!.StaticCpuLayers);
            Assert.Equal(0, cache.ExpertsPerLayer[1]);
            Assert.Equal(0, cache.ExpertsPerLayer[2]);
            Assert.True(cache.ExpertsPerLayer[4] > 0 && cache.ExpertsPerLayer[5] > 0);
            Assert.DoesNotContain(lookups, l => l < 3);
            Assert.Contains(4, lookups);
            Assert.True(tel.ColdExpertsCpu > 0);
            Assert.Equal(tel.Routed - tel.ColdExpertsCpu, cache.Hits + cache.Misses);
            float d = MaxDiff(resident, nCpu);
            _output.WriteLine($"--n-cpu-moe 3: |diff| {d:E3}, control {control:E3}, CPU experts {tel.ColdExpertsCpu} of {tel.Routed}");
            Assert.True(d > 0 && d * 4 < control, $"|n-cpu-moe - resident| {d:E3} vs control {control:E3}");

            // --cpu-moe: every MoE layer on the CPU, no cache at all, no copies.
            lookups.Clear();
            start = MoeTelemetry.Snapshot();
            float[] all = Run(path, ptxDir, options with { CpuMoeLayers = MoeOffloadOptions.AllLayers }, out info, out cache);
            tel = MoeTelemetry.Snapshot().Since(start);
            Assert.Equal(0, info!.CacheBytes);
            Assert.Equal(4, info.StaticCpuLayers);
            Assert.Empty(lookups);
            Assert.Equal(tel.Routed, tel.ColdExpertsCpu);
            Assert.Equal(0, tel.BytesH2D);
            float dAll = MaxDiff(resident, all);
            Assert.True(dAll * 4 < control, $"|cpu-moe - resident| {dAll:E3} vs control {control:E3}");
        }
        finally
        {
            CudaNemotronHTransformerModel.MoeLookupTraceHook = null;
        }
    }

    [SkippableFact]
    public void Auto_WithACpuThatNeverWins_IsBitIdenticalToResident_AndWithACpuThatAlwaysWins_UsesIt()
    {
        string ptxDir = RequireCuda();
        string path = CudaNemotronHMoeParityTests.WriteFixture(_scratch, latent: true, zeroBias: false, f32Banks: false);
        float[] resident = Run(path, ptxDir, new MoeOffloadOptions { Mode = MoeOffloadMode.Off }, out _, out _);

        // No calibrated layout: CPU cost is infinite, so every miss is copied (the copy-mode path through the auto planner).
        var never = Options(MoeColdMode.Auto, 2, true) with { CostModelOverride = new MoeCostModel(5e9, 50, []) };
        var start = MoeTelemetry.Snapshot();
        float[] a = Run(path, ptxDir, never, out var info, out var cache);
        Assert.Equal(MoeColdMode.Auto, info!.Cold);
        Assert.Equal(0, MoeTelemetry.Snapshot().Since(start).ColdExpertsCpu);
        Assert.True(cache.Insertions > 0);
        for (int i = 0; i < resident.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(resident[i]) == BitConverter.SingleToInt32Bits(a[i]),
                $"logit {i}: resident {resident[i]:R} != auto {a[i]:R}");

        // A CPU that costs nothing and a link that costs a lot: every uncached expert goes to the CPU.
        var layouts = ExpertLayouts(path);
        var always = Options(MoeColdMode.Auto, 2, true) with
        {
            CostModelOverride = new MoeCostModel(1e6, 0,
                [.. layouts.Select(l => new MoeLayoutCost(l, new LinearCost(0, 0), new LinearCost(1, 0)))]),
        };
        start = MoeTelemetry.Snapshot();
        Run(path, ptxDir, always, out _, out cache);
        var tel = MoeTelemetry.Snapshot().Since(start);
        Assert.Equal(0, cache.Insertions + cache.Bypasses);
        Assert.Equal(cache.ColdMisses, tel.ColdExpertsCpu);
        Assert.True(tel.ColdExpertsCpu > 0);
    }

    // ───────────────────────── helpers ─────────────────────────

    private MoeOffloadOptions Options(MoeColdMode cold, int threads, bool pinned) => new()
    {
        Mode = MoeOffloadMode.On,
        Cold = cold,
        CacheMib = 1,
        CpuThreads = threads,
        PinBudgetFraction = pinned ? MoeOffloadOptions.DefaultPinBudgetFraction : 0,
        CalibrationPath = Path.Combine(_scratch, "moe-calibration.json"),
        Log = _output.WriteLine,
    };

    private static string RequireCuda()
    {
        Skip.IfNot(CudaNemotronMoeTestSupport.IsCudaAvailable(), "No CUDA GPU available");
        string? ptxDir = CudaNemotronMoeTestSupport.FindPtxDir();
        Skip.If(ptxDir is null, "PTX files not found");
        return ptxDir!;
    }

    /// <summary>The distinct expert layouts of the fixture (read from the CPU model's layer weights).</summary>
    private static List<ExpertLayout> ExpertLayouts(string path)
    {
        using var gguf = GgufFile.Open(path);
        using var cpu = NemotronHTransformerModel.LoadFromGguf(gguf, GgufModelConfigExtractor.Extract(gguf.Metadata),
            ThreadingConfig.SingleThreaded);
        return [.. cpu.LayerWeights.Where(l => l.Moe is not null)
            .Select(l => new ExpertLayout(l.Moe!.UpPerExpertBytes, l.Moe.DownPerExpertBytes)).Distinct()];
    }

    /// <summary>Cache counters and shape after a run (read before the model is disposed).</summary>
    private sealed record CacheCounters(long Hits, long Misses, long Insertions, long Bypasses, long ColdMisses, long Promotions,
        int[] ExpertsPerLayer);

    /// <summary>Prompt prefill then <see cref="DecodeTokens"/> one at a time; rows = prompt rows + one per decode step.</summary>
    private static unsafe float[] Run(string path, string ptxDir, MoeOffloadOptions options, out CudaMoeOffloadInfo? info,
        out CacheCounters counters)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        var (model, _) = CudaModelLoader.CreateFromGguf(gguf, config, deviceId: 0, ptxDir, options);
        using (model)
        {
            var cuda = Assert.IsType<CudaNemotronHTransformerModel>(model);
            var rows = new List<float>();
            using (var logits = model.Forward(Prompt, [.. Enumerable.Range(0, Prompt.Length)], deviceId: 0))
                rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Prompt.Length * Vocab).ToArray());
            for (int s = 0; s < DecodeTokens.Length; s++)
            {
                using var logits = model.Forward([DecodeTokens[s]], [Prompt.Length + s], deviceId: 0);
                rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Vocab).ToArray());
            }

            info = cuda.MoeOffload;
            var c = cuda.MoeExpertCache;
            counters = c is null
                ? new CacheCounters(0, 0, 0, 0, 0, 0, [])
                : new CacheCounters(c.Hits, c.Misses, c.Insertions, c.Bypasses, c.ColdMisses, c.Promotions,
                    [.. Enumerable.Range(0, config.NumLayers).Select(c.ExpertsIn)]);
            return [.. rows];
        }
    }

    /// <summary>The CPU model on the same file: prompt prefill then the decode tokens (cacheless; the SSM state carries).</summary>
    private static unsafe float[] RunCpu(string path)
    {
        using var gguf = GgufFile.Open(path);
        var config = GgufModelConfigExtractor.Extract(gguf.Metadata);
        using var model = NemotronHTransformerModel.LoadFromGguf(gguf, config, ThreadingConfig.SingleThreaded);
        var rows = new List<float>();
        using (var logits = model.Forward(Prompt, [.. Enumerable.Range(0, Prompt.Length)], deviceId: 0))
            rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Prompt.Length * Vocab).ToArray());
        for (int s = 0; s < DecodeTokens.Length; s++)
        {
            using var logits = model.Forward([DecodeTokens[s]], [Prompt.Length + s], deviceId: 0);
            rows.AddRange(new ReadOnlySpan<float>((void*)logits.DataPointer, Vocab).ToArray());
        }
        return [.. rows];
    }

    private static float MaxDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float worst = 0;
        for (int i = 0; i < a.Length; i++) worst = MathF.Max(worst, MathF.Abs(a[i] - b[i]));
        return worst;
    }

    private static int ArgMax(float[] logits, int row)
    {
        int best = 0;
        for (int i = 1; i < Vocab; i++) if (logits[row * Vocab + i] > logits[row * Vocab + best]) best = i;
        return best;
    }
}
