namespace DotLLM.Vulkan;

/// <summary>The context-threshold rules of the lazily attached Vulkan DSpark drafter (issue #2046), pure so that a test can fail when the threshold is ignored.</summary>
internal static class DSparkAttachPolicy
{
    /// <summary>Parses <c>DOTLLM_DSPARK_MIN_CONTEXT</c>: empty = <paramref name="defaultValue"/>, <c>off</c> = never, otherwise a token count &gt;= 0 (0 = every request).</summary>
    internal static int ParseMinContext(string? text, int defaultValue)
    {
        if (string.IsNullOrWhiteSpace(text)) return defaultValue;
        if (text.Equals("off", StringComparison.OrdinalIgnoreCase)) return int.MaxValue;
        return int.TryParse(text, out int n) && n >= 0 ? n : throw new ArgumentException($"{VulkanInkling2TransformerModel.MinContextEnvVar}='{text}': expected a token count >= 0 or 'off'.");
    }

    /// <summary>True when a deferred drafter must be attached for a request of <paramref name="contextTokens"/>.</summary>
    internal static bool ShouldAttach(bool hasDeferred, int contextTokens, int minContext) => hasDeferred && contextTokens >= minContext;

    /// <summary>True when an attached drafter may speculate on a request of <paramref name="contextTokens"/>: always for an explicit attach, else only from the threshold.</summary>
    internal static bool MayUse(bool forced, int contextTokens, int minContext) => forced || contextTokens >= minContext;
}
