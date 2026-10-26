using DotLLM.HuggingFace;
using DotLLM.Tests.Integration.Fixtures;

namespace DotLLM.Benchmarks.Lora;

/// <summary>
/// Resolves a local GGUF checkpoint for the LoRA macro-bench, in priority order:
/// <list type="number">
/// <item>The <c>DOTLLM_BENCH_MODEL_PATH</c> env var (manual override).</item>
/// <item>A TinyLlama GGUF anywhere under <c>~/.dotllm/models/</c> (or the legacy test-cache).</item>
/// <item>Llama-3.2-1B-Instruct Q8_0 GGUF (the closest available stand-in).</item>
/// <item>SmolLM-135M Q8_0 GGUF (smallest fallback).</item>
/// </list>
/// We deliberately do NOT trigger downloads here — Phase 4d.3 is a measurement
/// follow-up, not a fixture provisioner, and the parent agent is offline-tolerant.
/// </summary>
internal static class LoraMacroBenchFixture
{
    /// <summary>
    /// Searches the local model store for a usable checkpoint. Returns the
    /// resolved path + a short label, or <c>null</c> + a skip reason.
    /// </summary>
    public static string? ResolveCheckpoint(out string label, out string? skipReason)
    {
        // 1. Manual override.
        var envPath = Environment.GetEnvironmentVariable("DOTLLM_BENCH_MODEL_PATH");
        if (!string.IsNullOrEmpty(envPath) && File.Exists(envPath))
        {
            label = Path.GetFileNameWithoutExtension(envPath);
            skipReason = null;
            return envPath;
        }

        // 2. TinyLlama GGUF — preferred per Phase 4d.3 spec.
        //    Probe common filename stems used by HF mirrors, in the model store then the legacy test-cache.
        string[] roots =
        [
            HuggingFaceDownloader.DefaultModelsDirectory,
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".dotllm", "test-cache"),
        ];
        foreach (string root in roots.Where(Directory.Exists))
        {
            foreach (var pat in TinyLlamaPatterns)
            {
                foreach (var hit in Directory.EnumerateFiles(root, pat, SearchOption.AllDirectories))
                {
                    label = "TinyLlama";
                    skipReason = null;
                    return hit;
                }
            }
        }

        // 3. Llama-3.2-1B Q8_0 — same scale class, exercises the same forward
        //    path on the same architecture family.
        // 4. SmolLM-135M Q8_0 — smallest fallback. Useful for CI smoke runs but
        //    note its absolute tok/s aren't representative of TinyLlama-class
        //    bandwidth pressure — the delta % vs base is still meaningful.
        (string Label, string Org, string Repo, string File)[] fallbacks =
        [
            ("Llama32_1B", "bartowski", "Llama-3.2-1B-Instruct-GGUF", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
            ("SmolLM_135M", "QuantFactory", "SmolLM-135M-GGUF", "SmolLM-135M.Q8_0.gguf"),
            ("SmolLM2_135M", "bartowski", "SmolLM2-135M-Instruct-GGUF", "SmolLM2-135M-Instruct-Q8_0.gguf"),
        ];
        foreach (var (name, org, repo, file) in fallbacks)
        {
            var found = TestFixtureResolver.ResolveFile("DOTLLM_BENCH_MODEL_PATH", org, repo, file);
            if (found.Found)
            {
                label = name;
                skipReason = null;
                return found.Path;
            }
        }

        label = string.Empty;
        skipReason =
            "no usable GGUF checkpoint found under ~/.dotllm/models/. "
          + "Set DOTLLM_BENCH_MODEL_PATH=<path-to-gguf> or place a TinyLlama / "
          + "Llama-3.2-1B / SmolLM-135M GGUF in the model store.";
        return null;
    }

    private static readonly string[] TinyLlamaPatterns =
    [
        "*TinyLlama*Q8_0*.gguf",
        "*tinyllama*q8_0*.gguf",
        "*TinyLlama*Q4_K_M*.gguf",
        "*tinyllama*q4_k_m*.gguf",
        "*TinyLlama*.gguf",
        "*tinyllama*.gguf",
    ];
}
