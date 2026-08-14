using System.Diagnostics;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Cuda;
using DotLLM.Cuda.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Cuda;

/// <summary>
/// Opt-in timing harness for <see cref="CudaMamba3TransformerModel"/>'s host-side per-token
/// prep costs (issue #382: per-layer managed allocations, redundant static-weight
/// re-downloads, the dead <c>_trapDevice</c> upload, and the full-<c>seqLen</c> final norm).
/// Runs against the real <c>ib-ssm/mamba3-370M-10BT</c> checkpoint — the tiny synthetic
/// parity fixture (8 hidden, 16 vocab) does not surface any of these costs at a measurable
/// scale. Mirrors <see cref="DotLLM.Tests.Unit.Cuda.CudaForwardPerfHarness"/>'s
/// opt-in-env-var-gated-xUnit-fact structure and
/// <see cref="IbSsmMamba3CudaParityTests"/>'s checkpoint resolution.
/// </summary>
/// <remarks>
/// Correctness is NOT this harness's job — the three parity test classes
/// (<c>CudaMamba3ParitySyntheticTests</c>, <c>CudaMamba3MimoParitySyntheticTests</c>,
/// <see cref="IbSsmMamba3CudaParityTests"/>) gate that. This harness exists purely to give
/// issue #382's four fixes measured before/after numbers instead of assumed ones (CLAUDE.md:
/// "benchmark before/after kernel changes").
/// </remarks>
[Trait("Category", "GPU")]
public sealed class CudaMamba3ForwardPerfHarness
{
    private const string PerfEnvVar = "DOTLLM_CUDA_MAMBA3_PERF";
    private const string CheckpointPathEnvVar = "DOTLLM_IBSSM_CHECKPOINT_PATH";
    private const string SafetensorsName = "model.safetensors";
    private const string ConventionalDir = "C:/temp/dotllm-ibssm";
    private const string UserProfileFallbackDir = "dotllm-ibssm-370m";

    private readonly ITestOutputHelper _output;
    public CudaMamba3ForwardPerfHarness(ITestOutputHelper output) => _output = output;

    private static string? ResolveCheckpointPath()
    {
        string? env = Environment.GetEnvironmentVariable(CheckpointPathEnvVar);
        if (!string.IsNullOrWhiteSpace(env))
        {
            if (File.Exists(env)) return env;
            if (Directory.Exists(env))
            {
                string candidate = Path.Combine(env, SafetensorsName);
                if (File.Exists(candidate)) return candidate;
            }
        }

        string conventional = Path.Combine(ConventionalDir, SafetensorsName);
        if (File.Exists(conventional)) return conventional;

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        if (!string.IsNullOrWhiteSpace(home))
        {
            string fallback = Path.Combine(home, UserProfileFallbackDir, SafetensorsName);
            if (File.Exists(fallback)) return fallback;
        }
        return null;
    }

    [SkippableFact]
    public void MeasureHostPrepLatencyAndAllocations()
    {
        Skip.IfNot(
            string.Equals(Environment.GetEnvironmentVariable(PerfEnvVar), "1", StringComparison.Ordinal),
            $"{PerfEnvVar}=1 not set.");
        string? checkpointPath = ResolveCheckpointPath();
        Skip.If(checkpointPath is null,
            $"ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
            + $"to the safetensors file or its directory, or place it at {ConventionalDir}/{SafetensorsName}.");
        Skip.IfNot(CudaDevice.IsAvailable(), "No CUDA GPU available.");

        var loadSw = Stopwatch.StartNew();
        var (model, source, config) = CudaModelLoader.LoadMamba3FromSafetensors(checkpointPath!);
        loadSw.Stop();
        _output.WriteLine($"load_ms={loadSw.Elapsed.TotalMilliseconds:F1}");

        try
        {
            int vocabSize = config.VocabSize;

            // ---- Warm-up: pay JIT / cuBLAS-handle / first-kernel-launch cost once, unmeasured. ----
            RunPrefill(model, vocabSize, seqLen: 8);

            // ---- Prefill latency at two seqLens (item 1's LOH-sized `proj` array scales with
            // seqLen * dInProj — the effect should be visible as reduced GC pause/alloc time,
            // not necessarily raw wall-clock, since the D2H/H2D transfer volume is unchanged). ----
            const int PrefillRepeats = 5;
            double[] prefill128 = new double[PrefillRepeats];
            for (int i = 0; i < PrefillRepeats; i++)
                prefill128[i] = RunPrefill(model, vocabSize, seqLen: 128);
            ReportStats("prefill_seqlen128_ms", prefill128);

            double[] prefill512 = new double[PrefillRepeats];
            for (int i = 0; i < PrefillRepeats; i++)
                prefill512[i] = RunPrefill(model, vocabSize, seqLen: 512);
            ReportStats("prefill_seqlen512_ms", prefill512);

            // ---- Managed allocation delta around a single seqLen=512 forward (item 1's target). ----
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            long allocBefore = GC.GetAllocatedBytesForCurrentThread();
            RunPrefill(model, vocabSize, seqLen: 512);
            long allocAfter = GC.GetAllocatedBytesForCurrentThread();
            _output.WriteLine($"managed_alloc_bytes_per_prefill512={allocAfter - allocBefore}");

            // ---- Decode latency AFTER a long prefill (scratch capacity already grown to >=512).
            // This is the configuration where a cap-sized-but-seqLen-unaware transfer bug (e.g.
            // uploading a cap-sized host array instead of a seqLen-sized slice) would show up as
            // a decode regression — the harness's own regression check for item 1's fix. ----
            using var state = new CudaMamba3StateCache(config.Mamba3Config!, config.NumLayers);
            int[] longPrefillTokens = MakeTokenIds(512, vocabSize, seed: 7);
            int[] longPrefillPositions = MakePositions(512);
            using (ITensor _ = model.Forward(longPrefillTokens, longPrefillPositions, deviceId: -1, state))
            {
                // Discarded — only used to grow scratch capacity + seed state.
            }

            const int DecodeSteps = 20;
            double[] decodeMs = new double[DecodeSteps];
            for (int i = 0; i < DecodeSteps; i++)
            {
                int[] tok = [1 + (i % (vocabSize - 1))];
                int[] pos = [512 + i];
                var sw = Stopwatch.StartNew();
                using ITensor logits = model.Forward(tok, pos, deviceId: -1, state);
                sw.Stop();
                decodeMs[i] = sw.Elapsed.TotalMilliseconds;
            }
            ReportStats("decode_after_512prefill_ms", decodeMs);
            double decodeAvg = Average(decodeMs);
            _output.WriteLine($"decode_after_512prefill_tok_per_sec={(decodeAvg > 0 ? 1000.0 / decodeAvg : 0.0):F2}");
        }
        finally
        {
            model.Dispose();
            source.Dispose();
        }
    }

    private static double RunPrefill(CudaMamba3TransformerModel model, int vocabSize, int seqLen)
    {
        int[] tokenIds = MakeTokenIds(seqLen, vocabSize, seed: seqLen);
        int[] positions = MakePositions(seqLen);
        var sw = Stopwatch.StartNew();
        using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }

    private static int[] MakeTokenIds(int seqLen, int vocabSize, int seed)
    {
        var rng = new Random(seed);
        int[] ids = new int[seqLen];
        for (int i = 0; i < seqLen; i++) ids[i] = rng.Next(0, vocabSize);
        return ids;
    }

    private static int[] MakePositions(int seqLen)
    {
        int[] positions = new int[seqLen];
        for (int i = 0; i < seqLen; i++) positions[i] = i;
        return positions;
    }

    private void ReportStats(string label, double[] samples)
    {
        double avg = Average(samples);
        double min = double.PositiveInfinity, max = 0.0;
        foreach (double v in samples) { if (v < min) min = v; if (v > max) max = v; }
        _output.WriteLine($"{label}: avg={avg:F2} min={min:F2} max={max:F2} n={samples.Length}");
        for (int i = 0; i < samples.Length; i++)
            _output.WriteLine($"  {label}[{i}]={samples[i]:F2}");
    }

    private static double Average(double[] samples)
    {
        if (samples.Length == 0) return 0.0;
        double sum = 0.0;
        foreach (double v in samples) sum += v;
        return sum / samples.Length;
    }
}
