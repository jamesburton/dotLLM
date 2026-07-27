using System.Diagnostics;
using DotLLM.Core.Configuration;
using DotLLM.Tokenizers;

namespace DotLLM.Engine;

/// <summary>
/// Executes warm-up inference passes to trigger JIT compilation and CUDA kernel loading.
/// </summary>
public static class WarmupRunner
{
    /// <summary>
    /// Runs warm-up inference passes using the given generator and tokenizer.
    /// Each iteration exercises the full inference pipeline: tokenize → prefill → decode → sample.
    /// </summary>
    /// <param name="generator">The text generator to warm up.</param>
    /// <param name="tokenizer">The tokenizer (used for logging prompt token count).</param>
    /// <param name="options">Warm-up configuration. If null, uses <see cref="WarmupOptions.Default"/>.</param>
    public static void Run(TextGenerator generator, ITokenizer tokenizer, WarmupOptions? options = null)
    {
        options ??= WarmupOptions.Default;
        if (!options.Enabled || options.Iterations <= 0)
            return;

        int promptTokens = tokenizer.Encode(options.DummyPrompt).Length;
        Console.WriteLine($"[dotllm] Warming up ({options.Iterations} iterations, " +
                          $"{promptTokens} prompt tokens, {options.MaxTokens} max gen tokens)...");

        long totalStart = Stopwatch.GetTimestamp();

        var inferenceOptions = new InferenceOptions
        {
            MaxTokens = options.MaxTokens,
            Temperature = 0f,
        };

        for (int i = 0; i < options.Iterations; i++)
        {
            long iterStart = Stopwatch.GetTimestamp();
            generator.Generate(options.DummyPrompt, inferenceOptions);
            double iterMs = Stopwatch.GetElapsedTime(iterStart).TotalMilliseconds;
            Console.WriteLine($"[dotllm]   Iteration {i + 1}/{options.Iterations}: {iterMs:F0}ms");
        }

        double totalMs = Stopwatch.GetElapsedTime(totalStart).TotalMilliseconds;
        Console.WriteLine($"[dotllm] Warm-up complete in {totalMs:F0}ms");
    }

    /// <summary>
    /// Runs warm-up denoising passes for a masked text-diffusion model. Drives a single small
    /// canvas through <see cref="DiffusionTextGenerator.Generate(string, int?, System.Action{DiffusionCanvasState}?, DotLLM.Core.Lora.ILoraAdapter?)"/>
    /// to trigger JIT compilation of the cacheless hybrid forward + denoise loop. CPU-only — uses
    /// no GPU dependency.
    /// </summary>
    /// <param name="generator">The diffusion generator to warm up.</param>
    /// <param name="tokenizer">The tokenizer (used for logging prompt token count).</param>
    /// <param name="options">Warm-up configuration. If null, uses <see cref="WarmupOptions.Default"/>.</param>
    public static void RunDiffusion(DiffusionTextGenerator generator, ITokenizer tokenizer, WarmupOptions? options = null)
    {
        options ??= WarmupOptions.Default;
        if (!options.Enabled || options.Iterations <= 0)
            return;

        int promptTokens = tokenizer.Encode(options.DummyPrompt).Length;
        // Keep the warm-up canvas small and bounded regardless of the model's full canvas length.
        int warmupTarget = Math.Max(1, Math.Min(options.MaxTokens, 8));
        Console.WriteLine($"[dotllm] Warming up diffusion ({options.Iterations} iterations, " +
                          $"{promptTokens} prompt tokens, {warmupTarget} canvas target)...");

        long totalStart = Stopwatch.GetTimestamp();
        for (int i = 0; i < options.Iterations; i++)
        {
            long iterStart = Stopwatch.GetTimestamp();
            generator.Generate(options.DummyPrompt, targetLength: warmupTarget);
            double iterMs = Stopwatch.GetElapsedTime(iterStart).TotalMilliseconds;
            Console.WriteLine($"[dotllm]   Iteration {i + 1}/{options.Iterations}: {iterMs:F0}ms");
        }

        double totalMs = Stopwatch.GetElapsedTime(totalStart).TotalMilliseconds;
        Console.WriteLine($"[dotllm] Diffusion warm-up complete in {totalMs:F0}ms");
    }
}
