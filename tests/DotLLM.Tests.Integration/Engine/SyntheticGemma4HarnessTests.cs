using DotLLM.Models.Gguf;
using Xunit;
using Xunit.Abstractions;

namespace DotLLM.Tests.Integration.Engine;

/// <summary>
/// Timing harness + emit-to-disk helpers for the synthetic Gemma-4 fixture. These run in CI
/// (they generate their own fixture) and double as the entry points a 12 GB T5500 / the
/// BenchmarkDotNet project use to consume the same <c>.gguf</c> on Vulkan/CUDA/HIP.
/// </summary>
public sealed class SyntheticGemma4HarnessTests
{
    private readonly ITestOutputHelper _out;

    public SyntheticGemma4HarnessTests(ITestOutputHelper output) => _out = output;

    /// <summary>
    /// Runs the per-phase Stopwatch harness for the TINY preset and prints CSV-friendly lines
    /// (phase,name,ms,tokens_per_sec). Asserts every phase produced a finite, non-negative time.
    /// </summary>
    [Fact]
    public void TimingHarness_Tiny_EmitsPerPhaseCsv()
    {
        var rows = SyntheticGemma4Harness.Run(
            SyntheticGemma4Gguf.Tiny, presetName: "tiny",
            prefillForwards: 4, prefillTokens: 16, diffusionSteps: 3, canvasLength: 8);

        _out.WriteLine(PhaseTiming.CsvHeader);
        foreach (var r in rows)
        {
            _out.WriteLine(r.ToCsv());
            Assert.True(r.Milliseconds >= 0 && double.IsFinite(r.Milliseconds), $"bad ms for {r.Phase}/{r.Name}");
        }

        // The expected phase set must be present.
        Assert.Contains(rows, r => r.Phase == "gen");
        Assert.Contains(rows, r => r.Phase == "load");
        Assert.Contains(rows, r => r.Phase == "warmup");
        Assert.Contains(rows, r => r.Phase == "prefill_avg" && r.TokensPerSec > 0);
        Assert.Contains(rows, r => r.Phase == "diffusion_step");
    }

    /// <summary>
    /// Emit-to-disk helper: writes BOTH the tiny gemma4 AR fixture and the diffusion-gemma
    /// fixture to a known path under the system temp dir, so the T5500 can consume them for
    /// CUDA kernel dev. Set <c>DOTLLM_SYNTH_OUT_DIR</c> to control the directory; otherwise a
    /// stable temp subdir is used. Reports the exact paths + file sizes.
    /// </summary>
    [Fact]
    public void EmitFixturesToDisk_WritesGemma4AndDiffusion()
    {
        string dir = Environment.GetEnvironmentVariable("DOTLLM_SYNTH_OUT_DIR")
            ?? Path.Combine(Path.GetTempPath(), "dotllm-synth-gemma4");
        Directory.CreateDirectory(dir);

        string arPath = Path.Combine(dir, "synthetic-gemma4-tiny.gguf");
        string diffPath = Path.Combine(dir, "synthetic-diffusion-gemma-tiny.gguf");
        SyntheticGemma4Gguf.WriteGemma4(arPath);
        SyntheticGemma4Gguf.WriteDiffusionGemma(diffPath);

        long arSize = new FileInfo(arPath).Length;
        long diffSize = new FileInfo(diffPath).Length;
        _out.WriteLine($"[emit] gemma4 AR     : {arPath} ({arSize:N0} bytes)");
        _out.WriteLine($"[emit] diffusion     : {diffPath} ({diffSize:N0} bytes)");
        _out.WriteLine("Point the Vulkan/CUDA/HIP GGUF loaders or benchmarks/DotLLM.Benchmarks at these paths.");

        Assert.True(arSize > 0 && diffSize > 0);
        // Diffusion fixture is strictly larger (self-cond + per-layer enc scale tensors).
        Assert.True(diffSize > arSize, "diffusion fixture should be larger than the AR fixture.");
    }
}
