using System.Diagnostics;
using System.Globalization;
using System.Text;
using DotLLM.Core.Configuration;
using DotLLM.Engine;
using DotLLM.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Run parameters for the capability harness. Defaults match a CPU smoke; the GPU box
/// overrides via env (<c>DOTLLM_CAP_CANVAS</c> / <c>DOTLLM_CAP_STEPS</c> /
/// <c>DOTLLM_CAP_MAXTOK</c> / <c>DOTLLM_CAP_PKV</c>) through <see cref="FromEnvironment"/>.
/// </summary>
public sealed record CapabilityRunOptions
{
    /// <summary>Diffusion canvas length override (env <c>DOTLLM_CAP_CANVAS</c>, default 32).</summary>
    public int CanvasLength { get; init; } = 32;

    /// <summary>Max denoise steps per canvas (env <c>DOTLLM_CAP_STEPS</c>, default 16).</summary>
    public int DenoiseSteps { get; init; } = 16;

    /// <summary>Max generated tokens per prompt, both engines (env <c>DOTLLM_CAP_MAXTOK</c>, default 48).</summary>
    public int MaxTokens { get; init; } = 48;

    /// <summary>Opt-in diffusion prompt-KV prefill/decode optimisation (env <c>DOTLLM_CAP_PKV=1</c>).</summary>
    public bool EnablePromptKv { get; init; }

    /// <summary>Optional CPU threading override (tests use SingleThreaded for determinism).</summary>
    public ThreadingConfig? Threading { get; init; }

    /// <summary>Builds options from the <c>DOTLLM_CAP_*</c> environment variables.</summary>
    public static CapabilityRunOptions FromEnvironment() => new()
    {
        CanvasLength = ParseInt("DOTLLM_CAP_CANVAS", 32),
        DenoiseSteps = ParseInt("DOTLLM_CAP_STEPS", 16),
        MaxTokens = ParseInt("DOTLLM_CAP_MAXTOK", 48),
        EnablePromptKv = Environment.GetEnvironmentVariable("DOTLLM_CAP_PKV") == "1",
    };

    private static int ParseInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer,
               CultureInfo.InvariantCulture, out int v) && v > 0 ? v : fallback;
}

/// <summary>Per-prompt outcome: objective pass/fail plus the throughput observations.</summary>
public sealed record CapabilityPromptResult
{
    /// <summary>Prompt id (row key).</summary>
    public required string PromptId { get; init; }

    /// <summary>Task family (<c>qa</c> / <c>typo</c> / <c>code</c>).</summary>
    public required string Family { get; init; }

    /// <summary>Whether the generated text satisfied the prompt's objective rule.</summary>
    public required bool Passed { get; init; }

    /// <summary>Wall-clock generation time for this prompt (seconds; excludes model load).</summary>
    public required double WallSeconds { get; init; }

    /// <summary>Number of generated tokens.</summary>
    public required int GeneratedTokens { get; init; }

    /// <summary>Total denoise steps (diffusion runs only).</summary>
    public int? DenoiseSteps { get; init; }

    /// <summary>Decode-phase tokens/sec (AR runs only).</summary>
    public double? DecodeTokensPerSec { get; init; }

    /// <summary>Effective throughput: generated tokens / wall seconds.</summary>
    public required double EffectiveTokensPerSec { get; init; }

    /// <summary>The generated text (raw; report rendering truncates/escapes it).</summary>
    public required string OutputText { get; init; }
}

/// <summary>Aggregated capability result for one model (one engine mode).</summary>
public sealed record CapabilityModelReport
{
    /// <summary>Short label used in report columns (e.g. <c>diffusion</c>, <c>ar</c>).</summary>
    public required string Label { get; init; }

    /// <summary>Path to the GGUF that produced these results.</summary>
    public required string ModelPath { get; init; }

    /// <summary>Model + tokenizer load wall time (seconds).</summary>
    public required double LoadSeconds { get; init; }

    /// <summary>Per-prompt outcomes, in prompt-set order.</summary>
    public required IReadOnlyList<CapabilityPromptResult> Results { get; init; }

    /// <summary>Number of prompts that passed their objective rule.</summary>
    public int PassCount => Results.Count(r => r.Passed);

    /// <summary>Total generation wall time across all prompts (seconds).</summary>
    public double TotalWallSeconds => Results.Sum(r => r.WallSeconds);

    /// <summary>Mean effective tokens/sec over prompts that generated at least one token.</summary>
    public double MeanEffectiveTokensPerSec
    {
        get
        {
            var withTokens = Results.Where(r => r.GeneratedTokens > 0).ToArray();
            return withTokens.Length > 0 ? withTokens.Average(r => r.EffectiveTokensPerSec) : 0;
        }
    }
}

/// <summary>
/// Capability half of the #33 benchmark harness: runs the fixed
/// <see cref="CapabilityPromptSet"/> through a dotLLM diffusion model
/// (<see cref="DiffusionTextGenerator"/>) and/or an AR baseline
/// (<see cref="TextGenerator"/>), scores each output with the prompt's objective rule,
/// and renders one comparable markdown report (EVAL.md house style). Throughput lives in
/// <c>CrossBackendTimingHarness</c> / <c>docs/diffusiongemma/REAL-26B-RESULTS.md</c>;
/// this harness adds the scored capability axis.
/// </summary>
public static class CapabilityHarness
{
    /// <summary>
    /// Runs the prompt set through a masked-diffusion model. Wiring mirrors
    /// <c>DiffusionGemmaGgufForwardTests</c>: <see cref="ModelLoader.LoadFromGguf"/>,
    /// <see cref="GgufBpeTokenizerFactory"/>, a small-canvas <c>DiffusionConfig</c> override,
    /// and a BOS prepend on the encoded prompt. Greedy (temperature 0) for reproducible scoring.
    /// </summary>
    /// <param name="ggufPath">Path to a diffusion GGUF (non-null <c>DiffusionConfig</c> required).</param>
    /// <param name="set">The fixed prompt set.</param>
    /// <param name="opts">Canvas/steps/max-token overrides.</param>
    /// <param name="log">Optional per-prompt progress sink (e.g. <c>ITestOutputHelper.WriteLine</c>).</param>
    public static CapabilityModelReport RunDiffusion(
        string ggufPath, CapabilityPromptSet set, CapabilityRunOptions opts, Action<string>? log = null)
    {
        var loadSw = Stopwatch.StartNew();
        var (model, gguf, config) = ModelLoader.LoadFromGguf(ggufPath, opts.Threading);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        loadSw.Stop();
        using var _ = gguf;
        using var __ = model;

        if (config.DiffusionConfig is null)
            throw new InvalidOperationException(
                $"'{ggufPath}' has no DiffusionConfig — not a diffusion model (arch {config.Architecture}).");

        // Small-canvas override; temperature 0 => greedy argmax unmasking, deterministic
        // and objectively scoreable (matches the PKV-equivalence tests' deterministic mode).
        var diff = config.DiffusionConfig with
        {
            CanvasLength = opts.CanvasLength,
            MaxDenoisingSteps = opts.DenoiseSteps,
            TemperatureMax = 0f,
            TemperatureMin = 0f,
        };
        var generator = new DiffusionTextGenerator(model, tokenizer, diffusionConfig: diff,
            enablePromptKv: opts.EnablePromptKv);

        var results = new List<CapabilityPromptResult>(set.Prompts.Count);
        foreach (CapabilityPrompt p in set.Prompts)
        {
            // BOS prepend (Gemma add_bos_token=True) — same wiring as the real-26B forward tests.
            int[] enc = tokenizer.Encode(p.Prompt);
            int[] promptIds = new int[enc.Length + 1];
            promptIds[0] = tokenizer.BosTokenId;
            Array.Copy(enc, 0, promptIds, 1, enc.Length);

            var sw = Stopwatch.StartNew();
            DiffusionResult r = generator.Generate(promptIds, targetLength: opts.MaxTokens);
            sw.Stop();

            double wall = sw.Elapsed.TotalSeconds;
            bool passed = p.Rule.Score(r.Text);
            results.Add(new CapabilityPromptResult
            {
                PromptId = p.Id,
                Family = p.Family,
                Passed = passed,
                WallSeconds = wall,
                GeneratedTokens = r.GeneratedTokenCount,
                DenoiseSteps = r.TotalDenoisingSteps,
                EffectiveTokensPerSec = wall > 0 ? r.GeneratedTokenCount / wall : 0,
                OutputText = r.Text,
            });
            log?.Invoke($"[diffusion] {p.Id}: {(passed ? "pass" : "FAIL")}  wall {wall:F1}s  "
                + $"steps {r.TotalDenoisingSteps}  toks {r.GeneratedTokenCount}  text: {Preview(r.Text)}");
        }

        return new CapabilityModelReport
        {
            Label = "diffusion",
            ModelPath = ggufPath,
            LoadSeconds = loadSw.Elapsed.TotalSeconds,
            Results = results,
        };
    }

    /// <summary>
    /// Runs the prompt set through an autoregressive baseline via the standard
    /// <see cref="TextGenerator"/> (default KV-cache, greedy, <see cref="CapabilityRunOptions.MaxTokens"/> cap).
    /// </summary>
    /// <param name="ggufPath">Path to any AR GGUF <see cref="ModelLoader"/> can load.</param>
    /// <param name="set">The fixed prompt set.</param>
    /// <param name="opts">Max-token override (canvas/steps are diffusion-only).</param>
    /// <param name="log">Optional per-prompt progress sink.</param>
    public static CapabilityModelReport RunAutoregressive(
        string ggufPath, CapabilityPromptSet set, CapabilityRunOptions opts, Action<string>? log = null)
    {
        var loadSw = Stopwatch.StartNew();
        var (model, gguf, config) = ModelLoader.LoadFromGguf(ggufPath, opts.Threading);
        var tokenizer = GgufBpeTokenizerFactory.Load(gguf.Metadata);
        loadSw.Stop();
        using var _ = gguf;
        using var __ = model;

        var generator = new TextGenerator(model, tokenizer);
        var options = new InferenceOptions { Temperature = 0f, MaxTokens = opts.MaxTokens };

        var results = new List<CapabilityPromptResult>(set.Prompts.Count);
        foreach (CapabilityPrompt p in set.Prompts)
        {
            var sw = Stopwatch.StartNew();
            InferenceResponse r = generator.Generate(p.Prompt, options);
            sw.Stop();

            double wall = sw.Elapsed.TotalSeconds;
            bool passed = p.Rule.Score(r.Text);
            results.Add(new CapabilityPromptResult
            {
                PromptId = p.Id,
                Family = p.Family,
                Passed = passed,
                WallSeconds = wall,
                GeneratedTokens = r.GeneratedTokenCount,
                DecodeTokensPerSec = r.Timings.DecodeTokensPerSec,
                EffectiveTokensPerSec = wall > 0 ? r.GeneratedTokenCount / wall : 0,
                OutputText = r.Text,
            });
            log?.Invoke($"[ar] {p.Id}: {(passed ? "pass" : "FAIL")}  wall {wall:F1}s  "
                + $"decode {r.Timings.DecodeTokensPerSec:F1} tok/s  toks {r.GeneratedTokenCount}  text: {Preview(r.Text)}");
        }

        return new CapabilityModelReport
        {
            Label = "ar",
            ModelPath = ggufPath,
            LoadSeconds = loadSw.Elapsed.TotalSeconds,
            Results = results,
        };
    }

    /// <summary>
    /// Renders the single comparable markdown report: configuration, per-model detail tables,
    /// family-level aggregate, and (with two or more models) a per-prompt pass/fail comparison.
    /// </summary>
    /// <param name="reports">One report per model that ran (diffusion and/or AR).</param>
    /// <param name="set">The prompt set the reports were scored against.</param>
    /// <param name="opts">The run options, echoed into the configuration table.</param>
    public static string BuildMarkdownReport(
        IReadOnlyList<CapabilityModelReport> reports, CapabilityPromptSet set, CapabilityRunOptions opts)
    {
        var families = set.Prompts.Select(p => p.Family).Distinct().ToArray();
        var sb = new StringBuilder();

        sb.AppendLine("# Capability eval — diffusion vs AR baseline (#33)");
        sb.AppendLine();
        string dateLine = string.Create(CultureInfo.InvariantCulture,
            $"**Date:** {DateTime.Now:yyyy-MM-dd} · machine `{Environment.MachineName}` · dotLLM capability harness ");
        dateLine += $"(`DiffusionCapabilityHarnessTests`), fixed prompt set `capability-prompts.json` "
            + $"({set.Prompts.Count} prompts: {string.Join(" / ", families)}). "
            + "Scoring is objective (case-insensitive keyword/regex containment) — no LLM judging.";
        sb.AppendLine(dateLine);
        sb.AppendLine();

        sb.AppendLine("## Configuration");
        sb.AppendLine("| Setting | Value |");
        sb.AppendLine("|---|---:|");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Canvas length (diffusion) | {opts.CanvasLength} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Max denoise steps / canvas | {opts.DenoiseSteps} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Max generated tokens | {opts.MaxTokens} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Prompt-KV (diffusion) | {(opts.EnablePromptKv ? "on" : "off")} |");
        sb.AppendLine("| Sampling | greedy (temperature 0) |");
        sb.AppendLine();

        sb.AppendLine("## Models");
        sb.AppendLine("| Label | GGUF | Load (s) |");
        sb.AppendLine("|---|---|---:|");
        foreach (var r in reports)
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"| {r.Label} | `{Path.GetFileName(r.ModelPath)}` | {r.LoadSeconds:F1} |");
        sb.AppendLine();

        sb.AppendLine("## Aggregate");
        sb.Append("| Model |");
        foreach (string f in families) sb.Append(CultureInfo.InvariantCulture, $" {f} |");
        sb.AppendLine(" Total | Wall (s) | Mean eff tok/s |");
        sb.Append("|---|");
        foreach (string _ in families) sb.Append("---:|");
        sb.AppendLine("---:|---:|---:|");
        foreach (var r in reports)
        {
            sb.Append(CultureInfo.InvariantCulture, $"| {r.Label} |");
            foreach (string f in families)
            {
                int total = r.Results.Count(x => x.Family == f);
                int pass = r.Results.Count(x => x.Family == f && x.Passed);
                sb.Append(CultureInfo.InvariantCulture, $" {pass}/{total} |");
            }
            double rate = r.Results.Count > 0 ? 100.0 * r.PassCount / r.Results.Count : 0;
            sb.AppendLine(CultureInfo.InvariantCulture,
                $" **{r.PassCount}/{r.Results.Count}** ({rate:F0}%) | {r.TotalWallSeconds:F1} | {r.MeanEffectiveTokensPerSec:F2} |");
        }
        sb.AppendLine();

        if (reports.Count >= 2)
        {
            sb.AppendLine("## Per-prompt comparison");
            sb.Append("| Prompt | Family | Rule |");
            foreach (var r in reports) sb.Append(CultureInfo.InvariantCulture, $" {r.Label} |");
            sb.AppendLine();
            sb.Append("|---|---|---|");
            foreach (var _ in reports) sb.Append("---|");
            sb.AppendLine();
            foreach (CapabilityPrompt p in set.Prompts)
            {
                sb.Append(CultureInfo.InvariantCulture, $"| {p.Id} | {p.Family} | {Escape(p.Rule.Describe())} |");
                foreach (var r in reports)
                {
                    var row = r.Results.FirstOrDefault(x => x.PromptId == p.Id);
                    sb.Append(CultureInfo.InvariantCulture, $" {(row is null ? "—" : row.Passed ? "pass" : "FAIL")} |");
                }
                sb.AppendLine();
            }
            sb.AppendLine();
        }

        foreach (var r in reports)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $"## Detail — {r.Label} (`{Path.GetFileName(r.ModelPath)}`)");
            sb.AppendLine("| Prompt | Family | Pass | Wall (s) | Gen toks | Steps | Decode tok/s | Eff tok/s | Output |");
            sb.AppendLine("|---|---|---|---:|---:|---:|---:|---:|---|");
            foreach (var x in r.Results)
            {
                string steps = x.DenoiseSteps?.ToString(CultureInfo.InvariantCulture) ?? "—";
                string decode = x.DecodeTokensPerSec is { } d ? d.ToString("F1", CultureInfo.InvariantCulture) : "—";
                sb.AppendLine(CultureInfo.InvariantCulture,
                    $"| {x.PromptId} | {x.Family} | {(x.Passed ? "pass" : "FAIL")} | {x.WallSeconds:F1} | "
                    + $"{x.GeneratedTokens} | {steps} | {decode} | {x.EffectiveTokensPerSec:F2} | {Escape(Preview(x.OutputText))} |");
            }
            sb.AppendLine();
        }

        return sb.ToString();
    }

    /// <summary>Single-line preview of generated text, truncated for table rendering.</summary>
    private static string Preview(string text, int max = 80)
    {
        string oneLine = text.Replace("\r", " ").Replace("\n", " ").Trim();
        return oneLine.Length <= max ? oneLine : oneLine[..max] + "…";
    }

    /// <summary>Escapes markdown table cell content (pipes and backticks in generated text).</summary>
    private static string Escape(string cell) => cell.Replace("|", "\\|");
}
