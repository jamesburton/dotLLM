using System.Diagnostics;
using DotLLM.Models.Quantization.Mach1;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Quantization.Mach1;

/// <summary>
/// Issue #266 Phase C: validates <see cref="Mach1FusedExpertGemv"/> (via
/// <see cref="Mach1PackedCheckpoint.GemvExpertProjection"/>) against the
/// already-Phase-B-proven dense-decode path
/// (<see cref="Mach1PackedCheckpoint.DecodeExpertProjection"/>) on REAL
/// expert weights from the <c>SyzygyResearch/Mach-1-Additive-35B</c> fixture,
/// then measures resident-memory and wall-clock deltas between the two
/// paths. Deliberately scoped to a handful of experts in one layer per the
/// session's RAM/time budget — NOT a full 40-layer / 256-expert run (Phase
/// B's own loader test already documents that needs ~70-120 GB).
/// </summary>
public sealed class Mach1FusedExpertGemvRealFixtureTests
{
    private const int Layer = 0;
    private const int HiddenSize = 2048;
    private const int ExpertIntermediateSize = 512;

    private readonly ITestOutputHelper _output;

    public Mach1FusedExpertGemvRealFixtureTests(ITestOutputHelper output) => _output = output;

    [SkippableFact]
    public void GemvExpertProjection_Expert0AllProjections_MatchesDenseDecodeThenMatVec()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        Skip.If(!File.Exists(Path.Combine(root!, "packed", "experts", "L00.safetensors")),
            "packed/experts/L00.safetensors not staged.");

        using var checkpoint = Mach1PackedCheckpoint.Open(root!);
        var rng = new Random(20260808);

        foreach ((string proj, int m0, int n0) in new[]
                 {
                     ("gate", ExpertIntermediateSize, HiddenSize),
                     ("up", ExpertIntermediateSize, HiddenSize),
                     ("down", HiddenSize, ExpertIntermediateSize),
                 })
        {
            float[] x = new float[n0];
            for (int i = 0; i < n0; i++)
                x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            float[] dense = new float[m0 * n0];
            checkpoint.DecodeExpertProjection(Layer, expertIndex: 0, proj, m0, n0, dense);
            float[] yRef = new float[m0];
            for (int i = 0; i < m0; i++)
            {
                float acc = 0f;
                int rowBase = i * n0;
                for (int j = 0; j < n0; j++)
                    acc += dense[rowBase + j] * x[j];
                yRef[i] = acc;
            }

            float[] yFused = new float[m0];
            checkpoint.GemvExpertProjection(Layer, expertIndex: 0, proj, m0, n0, x, yFused);

            double maxAbsErr = 0, maxRelErr = 0, refNorm = 0;
            for (int i = 0; i < m0; i++)
            {
                double diff = Math.Abs(yFused[i] - yRef[i]);
                double rel = diff / (Math.Abs((double)yRef[i]) + 1e-6);
                maxAbsErr = Math.Max(maxAbsErr, diff);
                maxRelErr = Math.Max(maxRelErr, rel);
                refNorm += (double)yRef[i] * yRef[i];
            }
            refNorm = Math.Sqrt(refNorm);

            _output.WriteLine($"expert=0 proj={proj} [{m0}x{n0}]: |y_ref|={refNorm:E4} maxAbsErr={maxAbsErr:E4} maxRelErr={maxRelErr:E4}");

            Assert.True(maxRelErr < 1e-2 || maxAbsErr < 1e-2 * Math.Max(refNorm / Math.Sqrt(m0), 1e-3),
                $"expert=0 proj={proj}: fused GEMV diverged from dense-decode reference " +
                $"(maxAbsErr={maxAbsErr:E4}, maxRelErr={maxRelErr:E4}, |y_ref|={refNorm:E4}).");
        }
    }

    /// <summary>
    /// Real-data perf: interleaved A/B trials (dense-decode-then-matvec vs.
    /// fused GEMV) for expert 0's "gate" projection, alternating so both
    /// share the same thermal/scheduling conditions (this project's
    /// established methodology, e.g. <c>MtpBenchProfile</c> for issue #253).
    /// Reports median wall-clock per call; does not assert a win — issue
    /// #266's acceptance bar explicitly allows a correctness-first kernel to
    /// be slower than dense-decode initially.
    /// </summary>
    [SkippableFact]
    public void Perf_InterleavedAB_DenseDecodeVsFusedGemv_Expert0Gate()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        Skip.If(!File.Exists(Path.Combine(root!, "packed", "experts", "L00.safetensors")),
            "packed/experts/L00.safetensors not staged.");

        using var checkpoint = Mach1PackedCheckpoint.Open(root!);
        const int m0 = ExpertIntermediateSize, n0 = HiddenSize;
        float[] x = new float[n0];
        var rng = new Random(1);
        for (int i = 0; i < n0; i++)
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] dense = new float[m0 * n0];
        float[] y = new float[m0];

        const int warmup = 2, trials = 10;
        var denseTimes = new List<double>();
        var fusedTimes = new List<double>();

        for (int i = 0; i < warmup; i++)
        {
            checkpoint.DecodeExpertProjection(Layer, 0, "gate", m0, n0, dense);
            checkpoint.GemvExpertProjection(Layer, 0, "gate", m0, n0, x, y);
        }

        var sw = new Stopwatch();
        for (int i = 0; i < trials; i++)
        {
            sw.Restart();
            checkpoint.DecodeExpertProjection(Layer, 0, "gate", m0, n0, dense);
            for (int r = 0; r < m0; r++)
            {
                float acc = 0f;
                int rb = r * n0;
                for (int c = 0; c < n0; c++)
                    acc += dense[rb + c] * x[c];
                y[r] = acc;
            }
            sw.Stop();
            denseTimes.Add(sw.Elapsed.TotalMilliseconds);

            sw.Restart();
            checkpoint.GemvExpertProjection(Layer, 0, "gate", m0, n0, x, y);
            sw.Stop();
            fusedTimes.Add(sw.Elapsed.TotalMilliseconds);
        }

        denseTimes.Sort();
        fusedTimes.Sort();
        double denseMedian = denseTimes[trials / 2];
        double fusedMedian = fusedTimes[trials / 2];

        _output.WriteLine($"Dense decode+matvec median: {denseMedian:F2} ms/call over {trials} interleaved trials");
        _output.WriteLine($"Fused GEMV median:          {fusedMedian:F2} ms/call over {trials} interleaved trials");
        _output.WriteLine($"Ratio (fused/dense):        {fusedMedian / denseMedian:F2}x");
    }

    /// <summary>
    /// Resident-memory comparison: decoding N experts' "gate" projection to
    /// dense fp32 and RETAINING them (the pattern a dense-decode-then-cache
    /// design would need) vs. computing the fused GEMV for the same N
    /// experts and retaining only the tiny [m0] outputs. Measures managed
    /// heap growth via <see cref="GC.GetTotalMemory"/> (forced full
    /// collections bracket each phase) — this isolates the payload the fused
    /// path avoids keeping resident, independent of OS page-cache effects
    /// from the memory-mapped packed file itself.
    /// </summary>
    [SkippableFact]
    public void Memory_DenseDecodeRetained_vs_FusedDiscarded_NExperts()
    {
        string? root = ResolveFixtureRoot();
        Skip.If(root is null, SkipReason);
        Skip.If(!File.Exists(Path.Combine(root!, "packed", "experts", "L00.safetensors")),
            "packed/experts/L00.safetensors not staged.");

        const int expertCount = 32; // one full chunk (ChunkSize)
        const int m0 = ExpertIntermediateSize, n0 = HiddenSize;

        float[] x = new float[n0];
        var rng = new Random(7);
        for (int i = 0; i < n0; i++)
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        long baseline, afterDense, afterFusedBaseline, afterFused;
        var retainedDense = new List<float[]>(expertCount);

        using (var checkpoint = Mach1PackedCheckpoint.Open(root!))
        {
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            baseline = GC.GetTotalMemory(forceFullCollection: true);

            for (int e = 0; e < expertCount; e++)
            {
                float[] dense = new float[m0 * n0];
                checkpoint.DecodeExpertProjection(Layer, e, "gate", m0, n0, dense);
                retainedDense.Add(dense); // retained -- simulates a dense-decode-then-cache design
            }
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            afterDense = GC.GetTotalMemory(forceFullCollection: true);
        }

        double denseRetainedMb = (afterDense - baseline) / (1024.0 * 1024.0);
        _output.WriteLine($"Dense-decode-retained: {expertCount} experts x gate [{m0}x{n0}] fp32 = " +
                           $"{denseRetainedMb:F1} MB managed-heap growth " +
                           $"(theoretical: {expertCount * m0 * n0 * 4L / (1024.0 * 1024.0):F1} MB)");

        // Drop the retained dense arrays and force a clean baseline before measuring the fused path.
        retainedDense.Clear();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        afterFusedBaseline = GC.GetTotalMemory(forceFullCollection: true);

        using (var checkpoint = Mach1PackedCheckpoint.Open(root!))
        {
            float[] y = new float[m0];
            for (int e = 0; e < expertCount; e++)
            {
                checkpoint.GemvExpertProjection(Layer, e, "gate", m0, n0, x, y);
                // y is reused/discarded each iteration -- nothing dense-sized is retained.
            }
            GC.Collect(2, GCCollectionMode.Forced, blocking: true);
            afterFused = GC.GetTotalMemory(forceFullCollection: true);
        }

        double fusedGrowthMb = (afterFused - afterFusedBaseline) / (1024.0 * 1024.0);
        _output.WriteLine($"Fused GEMV (discarded):  {expertCount} experts x gate [{m0}x{n0}]: " +
                           $"{fusedGrowthMb:F1} MB managed-heap growth after the run");
        _output.WriteLine($"Avoided resident payload: ~{denseRetainedMb - fusedGrowthMb:F1} MB for {expertCount} experts, 1 projection, 1 layer");

        // Extrapolation using the model's own documented totals (issue #266):
        // 256 experts x 3 projections x 40 layers dense fp32 ~= 64 GB, vs 6.21 GB packed.
        _output.WriteLine("Full-model extrapolation (from issue #266's own figures, not re-measured here): " +
                           "256 experts x 3 proj x 40 layers dense fp32 ~64 GB vs 6.21 GB packed-resident " +
                           "(the fused path's resident footprint is the packed bytes plus O(m0+n0) transient scratch per call).");
    }

    private const string SkipReason =
        "Mach-1-Additive-35B fixture not found. Set DOTLLM_MACH1_35B_DIR or populate " +
        "~/.dotllm/test-cache/SyzygyResearch/Mach-1-Additive-35B/ (see docs/QUANTIZATION.md).";

    private static string? ResolveFixtureRoot()
    {
        string? env = Environment.GetEnvironmentVariable("DOTLLM_MACH1_35B_DIR");
        if (!string.IsNullOrWhiteSpace(env) && Directory.Exists(env))
            return env;

        string conventional = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
            ".dotllm", "test-cache", "SyzygyResearch", "Mach-1-Additive-35B");
        return Directory.Exists(conventional) ? conventional : null;
    }
}
