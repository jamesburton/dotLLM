using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Core.Tensors;
using DotLLM.Models;
using DotLLM.Models.Architectures;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Models.Loaders;

/// <summary>
/// End-to-end verification that <see cref="ModelLoader.LoadFromSafetensors"/>
/// dispatches and successfully forwards a pass through the real
/// <c>ib-ssm/mamba3-370M-10BT</c> checkpoint (1.55 GB, 48 layers, 32 heads,
/// d_state=128, vocab=32000).
/// </summary>
/// <remarks>
/// <para>
/// <b>Gating.</b> The real checkpoint is not committed to the repo and not
/// fetched by CI. Tests skip gracefully unless a checkpoint is available, via
/// either:
/// </para>
/// <list type="number">
///   <item>
///     The <c>DOTLLM_IBSSM_CHECKPOINT_PATH</c> environment variable, pointing
///     at either <c>model.safetensors</c> directly or at the checkpoint
///     directory containing <c>model.safetensors</c> + <c>config.json</c>.
///   </item>
///   <item>
///     The conventional local path <c>C:/temp/dotllm-ibssm/model.safetensors</c>.
///   </item>
///   <item>
///     The user-profile fallback <c>%USERPROFILE%/dotllm-ibssm-370m/model.safetensors</c>.
///   </item>
/// </list>
/// <para>
/// <b>To run locally.</b> Either download the checkpoint to one of the
/// auto-detected paths above, or set <c>DOTLLM_IBSSM_CHECKPOINT_PATH</c> to
/// the safetensors file (or its containing directory). Then:
/// <code>
///   $env:DOTLLM_IBSSM_CHECKPOINT_PATH = "C:/temp/dotllm-ibssm/model.safetensors"
///   dotnet test tests/DotLLM.Tests.Integration/DotLLM.Tests.Integration.csproj `
///     --filter FullyQualifiedName~IbSsmMamba3RealWeights
/// </code>
/// </para>
/// <para>
/// <b>Canonical reference comparison.</b> The third test
/// (<see cref="ForwardMatchesCanonicalReference"/>) is gated by
/// <c>DOTLLM_IBSSM_REF_COMPARE=1</c>. It is skipped by default because the
/// canonical <c>state-spaces/mamba</c> path needs Triton+CUDA which isn't
/// viable on Windows+CPU at 370M dims; the algorithm-level
/// <c>Mamba3CanonicalReferenceCompareTests</c> cover the block math
/// against Python reference fixtures at tractable scales.
/// </para>
/// </remarks>
public sealed class IbSsmMamba3RealWeightsLoadTests
{
    private const string CheckpointPathEnvVar = "DOTLLM_IBSSM_CHECKPOINT_PATH";
    private const string RefCompareEnvVar = "DOTLLM_IBSSM_REF_COMPARE";
    private const string SafetensorsName = "model.safetensors";
    private const string ConventionalDir = "C:/temp/dotllm-ibssm";
    private const string UserProfileFallbackDir = "dotllm-ibssm-370m";

    private readonly ITestOutputHelper _output;

    public IbSsmMamba3RealWeightsLoadTests(ITestOutputHelper output)
    {
        _output = output;
    }

    /// <summary>
    /// Resolves a safetensors path from the env-var override, then from the
    /// conventional <c>C:/temp/dotllm-ibssm/</c> path, then from the
    /// user-profile fallback. Returns null if none resolve to an existing file
    /// (caller skips).
    /// </summary>
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

    /// <summary>
    /// Loads the ib-ssm/mamba3-370M-10BT checkpoint and asserts the
    /// <see cref="ModelConfig"/> + architecture dispatch match the published
    /// dimensions. No forward pass — the 48-layer CPU prefill is exercised by
    /// <see cref="ForwardProducesFiniteVocabLogits"/>.
    /// </summary>
    [Fact]
    public void LoadConfig_ReturnsExpectedDimensions()
    {
        string? checkpointPath = ResolveCheckpointPath();
        if (checkpointPath is null)
        {
            _output.WriteLine(
                $"[SKIP] ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
                + $"to the safetensors file or its directory, or place it at {ConventionalDir}/model.safetensors "
                + "or %USERPROFILE%/dotllm-ibssm-370m/model.safetensors.");
            return;
        }

        long fileBytes = new FileInfo(checkpointPath).Length;
        _output.WriteLine($"Checkpoint: {checkpointPath}  ({fileBytes:N0} bytes)");

        var sw = Stopwatch.StartNew();
        var (model, file, config) = ModelLoader.LoadFromSafetensors(checkpointPath);
        sw.Stop();

        try
        {
            _output.WriteLine(
                $"Load: arch={config.Architecture} vocab={config.VocabSize} "
                + $"hidden={config.HiddenSize} layers={config.NumLayers} "
                + $"heads={config.Mamba3Config!.NumHeads} head_dim={config.Mamba3Config.HeadDim} "
                + $"d_state={config.Mamba3Config.StateSize} "
                + $"d_in_proj={config.Mamba3Config.InputProjectionDim} "
                + $"num_rope_angles={config.Mamba3Config.NumRopeAngles} "
                + $"rope_fraction={config.Mamba3Config.RopeFraction} "
                + $"is_mimo={config.Mamba3Config.IsMimo} "
                + $"mimo_rank={config.Mamba3Config.MimoRank} "
                + $"tied={config.TiedEmbeddings} "
                + $"in {sw.Elapsed.TotalMilliseconds:F1} ms");

            Assert.Equal(Architecture.Mamba3, config.Architecture);
            Assert.IsType<Mamba3TransformerModel>(model);
            Assert.Equal(1024, config.HiddenSize);
            Assert.Equal(48, config.NumLayers);
            Assert.Equal(32, config.NumAttentionHeads);
            Assert.Equal(32, config.Mamba3Config.NumHeads);
            Assert.Equal(64, config.Mamba3Config.HeadDim);
            Assert.Equal(128, config.Mamba3Config.StateSize);
            Assert.Equal(32000, config.VocabSize);
            Assert.Equal(1, config.Mamba3Config.NumGroups);
            Assert.False(config.Mamba3Config.IsMimo);
            Assert.Equal(0.5f, config.Mamba3Config.RopeFraction, precision: 5);
            Assert.Equal(4480, config.Mamba3Config.InputProjectionDim);
            Assert.Equal(32, config.Mamba3Config.NumRopeAngles);
        }
        finally
        {
            model.Dispose();
            file.Dispose();
        }
    }

    /// <summary>
    /// Forwards 5 tokens through the full 48-layer 370M SSM and asserts every
    /// one of the 5 × 32 000 = 160 000 logits is finite with nonzero per-position
    /// stddev. Logs min/max/mean/stddev + top-1 argmax per position and elapsed
    /// time, so regressions (NaN emergence, degeneracy, slowdowns) are visible
    /// in the test output.
    /// </summary>
    [Fact]
    public void ForwardProducesFiniteVocabLogits()
    {
        string? checkpointPath = ResolveCheckpointPath();
        if (checkpointPath is null)
        {
            _output.WriteLine(
                $"[SKIP] ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar} "
                + "to enable this test.");
            return;
        }

        long fileBytes = new FileInfo(checkpointPath).Length;
        _output.WriteLine($"Checkpoint: {checkpointPath}  ({fileBytes:N0} bytes)");

        var loadWatch = Stopwatch.StartNew();
        var (model, file, config) = ModelLoader.LoadFromSafetensors(checkpointPath);
        loadWatch.Stop();
        _output.WriteLine($"Load: {loadWatch.Elapsed.TotalMilliseconds:F1} ms");

        try
        {
            // Five tokens spanning the vocab range. Head, quantile, tail, vocab-minus-one.
            int[] tokenIds = [0, 100, 1000, 10000, 31999];
            int[] positions = [0, 1, 2, 3, 4];

            var fwdWatch = Stopwatch.StartNew();
            using ITensor logits = model.Forward(tokenIds, positions, deviceId: -1);
            fwdWatch.Stop();

            Assert.Equal(2, logits.Shape.Rank);
            Assert.Equal(tokenIds.Length, logits.Shape[0]);
            Assert.Equal(config.VocabSize, logits.Shape[1]);

            var stats = ComputePerPositionStats(logits, tokenIds.Length, config.VocabSize);
            _output.WriteLine(
                $"Forward: shape=[{logits.Shape[0]}, {logits.Shape[1]}] "
                + $"finite={stats.TotalFinite}/{stats.TotalCount} "
                + $"in {fwdWatch.Elapsed.TotalSeconds:F1} s "
                + $"({fwdWatch.Elapsed.TotalSeconds / tokenIds.Length:F2} s/token)");

            for (int p = 0; p < tokenIds.Length; p++)
            {
                var ps = stats.Per[p];
                _output.WriteLine(
                    $"  pos[{p}] token={tokenIds[p]}: "
                    + $"min={ps.Min:G4} max={ps.Max:G4} mean={ps.Mean:G4} "
                    + $"stddev={ps.StdDev:G4} argmax={ps.ArgMax} ({ps.ArgMaxValue:G4})");
            }

            Assert.Equal(stats.TotalCount, stats.TotalFinite);
            for (int p = 0; p < tokenIds.Length; p++)
            {
                Assert.True(stats.Per[p].StdDev > 0,
                    $"Position {p} logits have zero variance — forward pass likely degenerate.");
            }
        }
        finally
        {
            model.Dispose();
            file.Dispose();
        }
    }

    /// <summary>
    /// Optional gated canonical-reference comparison. Skipped by default;
    /// enable by setting <c>DOTLLM_IBSSM_REF_COMPARE=1</c> alongside
    /// <c>DOTLLM_IBSSM_CHECKPOINT_PATH</c> (or one of the auto-detected paths).
    /// </summary>
    /// <remarks>
    /// <para>
    /// The canonical <c>state-spaces/mamba</c> forward path depends on Triton
    /// + CUDA kernels that do not work on Windows+CPU. At 370M dimensions the
    /// pure-Python fallback is prohibitively slow (48 layers × 1024 hidden ×
    /// 5 tokens pushes into tens of minutes even with careful vectorisation).
    /// </para>
    /// <para>
    /// This stub preserves the contract and prints the recommended invocation
    /// so a future environment with Triton+CUDA (or a fully-patched pure-Python
    /// reference) can opt in. When the stub flips to a real comparison, the
    /// expected tolerances are intentionally loose: <c>AbsTol=5e-3</c>,
    /// <c>RelTol=5e-2</c> — 48-layer F32 drift between the attention-style SSD
    /// scan and the trapezoidal-kernel reference routinely exceeds the strict
    /// algorithm-level tolerances used by the small-scale comparators.
    /// </para>
    /// </remarks>
    [Fact]
    public void ForwardMatchesCanonicalReference()
    {
        string? enable = Environment.GetEnvironmentVariable(RefCompareEnvVar);
        if (!string.Equals(enable, "1", StringComparison.Ordinal))
        {
            _output.WriteLine(
                $"[SKIP] Canonical reference comparison disabled. To enable, set "
                + $"{RefCompareEnvVar}=1 plus {CheckpointPathEnvVar}. Note: the canonical "
                + "state-spaces/mamba path requires Triton+CUDA (not viable on Windows+CPU "
                + "at 370M dims); the pure-Python fallback from capture_fixtures_canonical.py "
                + "takes tens of minutes to run 48 layers × 1024 hidden × 5 tokens. "
                + "Algorithm-level drift at block scale is covered by "
                + "Mamba3CanonicalReferenceCompareTests.");
            return;
        }

        string? checkpointPath = ResolveCheckpointPath();
        if (checkpointPath is null)
        {
            _output.WriteLine(
                $"[SKIP] ib-ssm/mamba3-370M-10BT checkpoint not found. Set {CheckpointPathEnvVar}.");
            return;
        }

        // Full-scale reference import is not wired here — the test remains a
        // documented opt-in until a Triton+CUDA CI path exists.
        throw new SkipException(
            "Canonical reference comparison at 370M dims is not yet wired; "
            + "set DOTLLM_IBSSM_REF_COMPARE=1 AND run from a Linux+CUDA host with "
            + "state-spaces/mamba available, then wire the reference call in this test.");
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static unsafe FullStats ComputePerPositionStats(ITensor logits, int seqLen, int vocabSize)
    {
        var span = new ReadOnlySpan<float>((void*)logits.DataPointer, seqLen * vocabSize);
        var per = new PositionStats[seqLen];
        int totalFinite = 0;
        for (int p = 0; p < seqLen; p++)
        {
            int finite = 0;
            double sum = 0, sumSq = 0;
            float min = float.PositiveInfinity, max = float.NegativeInfinity;
            int argmax = -1;
            float argmaxVal = float.NegativeInfinity;
            for (int v = 0; v < vocabSize; v++)
            {
                float x = span[p * vocabSize + v];
                if (float.IsFinite(x))
                {
                    finite++;
                    sum += x;
                    sumSq += (double)x * x;
                    if (x < min) min = x;
                    if (x > max)
                    {
                        max = x;
                        argmax = v;
                        argmaxVal = x;
                    }
                }
            }
            double mean = finite > 0 ? sum / finite : 0.0;
            double variance = finite > 0 ? (sumSq / finite) - mean * mean : 0.0;
            double stddev = Math.Sqrt(Math.Max(0.0, variance));
            per[p] = new PositionStats(finite, (float)mean, (float)stddev, min, max, argmax, argmaxVal);
            totalFinite += finite;
        }
        return new FullStats(seqLen * vocabSize, totalFinite, per);
    }

    private readonly record struct PositionStats(
        int FiniteCount, float Mean, float StdDev, float Min, float Max, int ArgMax, float ArgMaxValue);

    private sealed record FullStats(int TotalCount, int TotalFinite, PositionStats[] Per);

    /// <summary>
    /// xUnit has no built-in "skip from within a test body" — this sentinel
    /// lets <see cref="ForwardMatchesCanonicalReference"/> express an opt-in
    /// gate as a specific exception type if someone does enable
    /// <c>DOTLLM_IBSSM_REF_COMPARE=1</c> without a working reference path.
    /// </summary>
    private sealed class SkipException(string message) : Xunit.Sdk.XunitException(message);
}
