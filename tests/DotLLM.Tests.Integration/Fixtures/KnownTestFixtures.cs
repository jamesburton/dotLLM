namespace DotLLM.Tests.Integration.Fixtures;

/// <summary>
/// Named lookups for the real-model fixtures that several suites share, so the environment
/// override, cache layout and repo coordinates are declared once (issue #308) rather than
/// re-spelled — and drifting — in every suite that consumes them.
/// </summary>
/// <remarks>
/// Each accessor re-probes on access (the underlying probes are a handful of
/// <see cref="File.Exists(string)"/> calls) so a test that sets an override at runtime is
/// honoured, matching the behaviour of the environment-variable reads these replaced.
/// </remarks>
internal static class KnownTestFixtures
{
    /// <summary>
    /// BitNet b1.58-2B-4T, the natively-ternary I2_S GGUF —
    /// <c>microsoft/bitnet-b1.58-2B-4T-gguf/ggml-model-i2_s.gguf</c>.
    /// </summary>
    public static FixtureLocation BitNetI2S => TestFixtureResolver.ResolveFile(
        "DOTLLM_BITNET_GGUF",
        "microsoft",
        "bitnet-b1.58-2B-4T-gguf",
        "ggml-model-i2_s.gguf");

    /// <summary>Human-readable name for <see cref="BitNetI2S"/> skip messages.</summary>
    public const string BitNetI2SDescription = "BitNet b1.58-2B-4T I2_S GGUF";

    /// <summary>
    /// Gemma-4-26B-A4B-it, Q4_K_M (~15.7 GB) —
    /// <c>unsloth/gemma-4-26B-A4B-it-GGUF/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf</c>.
    /// </summary>
    public static FixtureLocation Gemma4_26B_A4B_Q4KM => TestFixtureResolver.ResolveFile(
        "DOTLLM_GEMMA4_GGUF",
        "unsloth",
        "gemma-4-26B-A4B-it-GGUF",
        "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "gemma-4-26B-A4B-it-Q4_K_M.gguf");

    /// <summary>Human-readable name for <see cref="Gemma4_26B_A4B_Q4KM"/> skip messages.</summary>
    public const string Gemma4_26BDescription = "Gemma-4-26B-A4B-it Q4_K_M GGUF (~15.7 GB)";

    /// <summary>
    /// Qwen2.5-0.5B-Instruct, Q8_0 (~0.5 GB) — <c>Qwen/Qwen2.5-0.5B-Instruct-GGUF</c>.
    /// Carries <c>tokenizer.ggml.pre = qwen2</c>, the value on every Qwen2/Qwen3 GGUF (#397).
    /// </summary>
    public static FixtureLocation Qwen2_5_0_5B_Q8_0 => TestFixtureResolver.ResolveFile(
        "DOTLLM_QWEN25_0_5B_Q8_0_GGUF",
        "Qwen",
        "Qwen2.5-0.5B-Instruct-GGUF",
        "qwen2.5-0.5b-instruct-q8_0.gguf");

    /// <summary>Human-readable name for <see cref="Qwen2_5_0_5B_Q8_0"/> skip messages.</summary>
    public const string Qwen2_5_0_5BDescription = "Qwen2.5-0.5B-Instruct Q8_0 GGUF (tokenizer.ggml.pre = qwen2)";

    /// <summary>
    /// Ternary-Bonsai-27B, Q2_0 (~7.2 GB) — <c>prism-ml/Ternary-Bonsai-27B-gguf</c>.
    /// Carries <c>tokenizer.ggml.pre = qwen35</c> (#397).
    /// </summary>
    public static FixtureLocation TernaryBonsai27B_Q2_0 => TestFixtureResolver.ResolveFile(
        "DOTLLM_BONSAI_PQ2_0_GGUF",
        "prism-ml",
        "Ternary-Bonsai-27B-gguf",
        "Ternary-Bonsai-27B-Q2_0.gguf");

    /// <summary>Human-readable name for <see cref="TernaryBonsai27B_Q2_0"/> skip messages.</summary>
    public const string TernaryBonsai27BDescription = "Ternary-Bonsai-27B Q2_0 GGUF (tokenizer.ggml.pre = qwen35)";

    /// <summary>
    /// DeepSeek-V2-Lite HF safetensors snapshot directory —
    /// <c>deepseek-ai/DeepSeek-V2-Lite</c>.
    /// </summary>
    public static FixtureLocation DeepSeekV2LiteCheckpoint => TestFixtureResolver.ResolveDirectory(
        ["DOTLLM_DEEPSEEK_V2_LITE_CHECKPOINT_PATH"],
        "deepseek-ai",
        "DeepSeek-V2-Lite",
        ["config.json"],
        // Legacy path the suites documented before #308 added cache resolution.
        ["C:/temp/dotllm-deepseek-v2-lite"]);

    /// <summary>Human-readable name for <see cref="DeepSeekV2LiteCheckpoint"/> skip messages.</summary>
    public const string DeepSeekV2LiteDescription = "DeepSeek-V2-Lite HF safetensors snapshot";
}
