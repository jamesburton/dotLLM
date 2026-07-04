using DotLLM.Core.Configuration;
using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Entry points for the #33 capability harness (<see cref="CapabilityHarness"/>):
/// <list type="bullet">
/// <item>a real-checkpoint eval gated on <c>DOTLLM_CAP_DIFFUSION_GGUF</c> /
/// <c>DOTLLM_CAP_AR_GGUF</c> (each optional — whichever is set runs; both set produces the
/// side-by-side comparison). Self-skips when neither is set, so CI never depends on
/// multi-gig checkpoints;</item>
/// <item>a synthetic smoke test that exercises the full plumbing (load → generate → score →
/// report) on the self-generated tiny fixtures. Its SCORES are garbage (random weights) —
/// it asserts only that the harness runs, scores, and writes the report. CPU-only CI safe.</item>
/// </list>
/// Report: always dumped to the test output; additionally written to the path in
/// <c>DOTLLM_CAP_REPORT</c> when set. Canvas/steps/max-tokens come from
/// <c>DOTLLM_CAP_CANVAS</c> / <c>DOTLLM_CAP_STEPS</c> / <c>DOTLLM_CAP_MAXTOK</c>
/// (defaults 32 / 16 / 48), diffusion prompt-KV from <c>DOTLLM_CAP_PKV=1</c>.
/// </summary>
public sealed class DiffusionCapabilityHarnessTests
{
    private const string DiffusionEnv = "DOTLLM_CAP_DIFFUSION_GGUF";
    private const string ArEnv = "DOTLLM_CAP_AR_GGUF";
    private const string ReportEnv = "DOTLLM_CAP_REPORT";

    private readonly ITestOutputHelper _out;

    public DiffusionCapabilityHarnessTests(ITestOutputHelper output) => _out = output;

    private static string? ResolveEnvPath(string envVar)
    {
        string? path = Environment.GetEnvironmentVariable(envVar);
        return string.IsNullOrWhiteSpace(path) || !File.Exists(path) ? null : path;
    }

    /// <summary>
    /// Real-checkpoint capability eval. Runs whichever of the diffusion / AR models is
    /// configured, scores the fixed prompt set, and emits the comparable markdown report.
    /// SLOW on CPU with a 26B diffusion checkpoint — size the canvas/steps via env.
    /// </summary>
    [SkippableFact]
    public void RealModels_CapabilityEval_EmitsComparableReport()
    {
        string? diffusionPath = ResolveEnvPath(DiffusionEnv);
        string? arPath = ResolveEnvPath(ArEnv);
        Skip.If(diffusionPath is null && arPath is null,
            $"Set {DiffusionEnv} (diffusion GGUF) and/or {ArEnv} (AR baseline GGUF) to run the capability eval.");

        var set = CapabilityPromptSet.LoadDefault();
        var opts = CapabilityRunOptions.FromEnvironment();
        _out.WriteLine($"[capability] canvas={opts.CanvasLength} steps={opts.DenoiseSteps} "
            + $"maxtok={opts.MaxTokens} pkv={(opts.EnablePromptKv ? "on" : "off")} prompts={set.Prompts.Count}");

        var reports = new List<CapabilityModelReport>();
        if (diffusionPath is not null)
            reports.Add(CapabilityHarness.RunDiffusion(diffusionPath, set, opts, _out.WriteLine));
        if (arPath is not null)
            reports.Add(CapabilityHarness.RunAutoregressive(arPath, set, opts, _out.WriteLine));

        string markdown = CapabilityHarness.BuildMarkdownReport(reports, set, opts);
        _out.WriteLine(string.Empty);
        _out.WriteLine(markdown);
        WriteReportIfConfigured(markdown);

        // Structural assertions only — real-model pass rates are REPORTED, not gated (the
        // harness must not turn a capability regression into a red build on a random box).
        Assert.NotEmpty(reports);
        foreach (CapabilityModelReport report in reports)
        {
            Assert.Equal(set.Prompts.Count, report.Results.Count);
            foreach (CapabilityPromptResult r in report.Results)
            {
                Assert.True(r.WallSeconds >= 0 && double.IsFinite(r.WallSeconds), $"bad wall time for {r.PromptId}");
                Assert.True(r.EffectiveTokensPerSec >= 0 && double.IsFinite(r.EffectiveTokensPerSec),
                    $"bad effective tok/s for {r.PromptId}");
            }
        }
    }

    /// <summary>
    /// Synthetic end-to-end smoke: generates the tiny diffusion-gemma AND gemma4 AR fixtures
    /// (<see cref="SyntheticGemma4Gguf"/>), runs BOTH harness paths over the full prompt set
    /// with a tiny canvas/step/token budget, builds the comparison report, and writes it to
    /// disk. Random weights ⇒ scoring outcome is meaningless; the assertions cover plumbing:
    /// every prompt produced a result with sane metrics, and the report contains every prompt
    /// row plus both model columns.
    /// </summary>
    [Fact]
    public void SyntheticSmoke_HarnessRunsScoresAndWritesReport()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"dotllm-cap-smoke-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string diffusionPath = SyntheticGemma4Gguf.WriteDiffusionGemma(Path.Combine(dir, "syn-diffusion.gguf"));
            string arPath = SyntheticGemma4Gguf.WriteGemma4(Path.Combine(dir, "syn-ar.gguf"));

            var set = CapabilityPromptSet.LoadDefault();
            Assert.Equal(10, set.Prompts.Count); // the checked-in fixed set

            var opts = new CapabilityRunOptions
            {
                CanvasLength = 8,
                DenoiseSteps = 3,
                MaxTokens = 8,
                Threading = ThreadingConfig.SingleThreaded,
            };

            CapabilityModelReport diffusion = CapabilityHarness.RunDiffusion(diffusionPath, set, opts, _out.WriteLine);
            CapabilityModelReport ar = CapabilityHarness.RunAutoregressive(arPath, set, opts, _out.WriteLine);

            // Every prompt scored on both engines, with finite metrics.
            foreach (CapabilityModelReport report in new[] { diffusion, ar })
            {
                Assert.Equal(set.Prompts.Count, report.Results.Count);
                Assert.Equal(
                    set.Prompts.Select(p => p.Id),
                    report.Results.Select(r => r.PromptId));
                foreach (CapabilityPromptResult r in report.Results)
                {
                    Assert.True(r.WallSeconds >= 0 && double.IsFinite(r.WallSeconds), $"bad wall for {r.PromptId}");
                    Assert.True(r.GeneratedTokens >= 0, $"negative token count for {r.PromptId}");
                    Assert.True(r.EffectiveTokensPerSec >= 0 && double.IsFinite(r.EffectiveTokensPerSec),
                        $"bad eff tok/s for {r.PromptId}");
                }
            }
            // Engine-specific metrics landed on the right sides.
            Assert.All(diffusion.Results, r => Assert.NotNull(r.DenoiseSteps));
            Assert.All(diffusion.Results, r => Assert.Null(r.DecodeTokensPerSec));
            Assert.All(ar.Results, r => Assert.Null(r.DenoiseSteps));
            Assert.All(ar.Results, r => Assert.NotNull(r.DecodeTokensPerSec));

            // Report renders and persists: every prompt row + both labels + the comparison table.
            string markdown = CapabilityHarness.BuildMarkdownReport([diffusion, ar], set, opts);
            string reportPath = Path.Combine(dir, "capability-report.md");
            File.WriteAllText(reportPath, markdown);

            Assert.True(File.Exists(reportPath), "report file must be written");
            string persisted = File.ReadAllText(reportPath);
            foreach (CapabilityPrompt p in set.Prompts)
                Assert.Contains(p.Id, persisted, StringComparison.Ordinal);
            Assert.Contains("## Per-prompt comparison", persisted, StringComparison.Ordinal);
            Assert.Contains("## Detail — diffusion", persisted, StringComparison.Ordinal);
            Assert.Contains("## Detail — ar", persisted, StringComparison.Ordinal);

            _out.WriteLine(string.Empty);
            _out.WriteLine(markdown);
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best-effort temp cleanup */ }
        }
    }

    private void WriteReportIfConfigured(string markdown)
    {
        string? reportPath = Environment.GetEnvironmentVariable(ReportEnv);
        if (string.IsNullOrWhiteSpace(reportPath))
            return;
        string? parent = Path.GetDirectoryName(Path.GetFullPath(reportPath));
        if (!string.IsNullOrEmpty(parent))
            Directory.CreateDirectory(parent);
        File.WriteAllText(reportPath, markdown);
        _out.WriteLine($"[capability] report written to {reportPath}");
    }
}
