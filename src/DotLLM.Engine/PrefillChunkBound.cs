namespace DotLLM.Engine;

/// <summary>
/// Resolves the prompt-prefill chunk against a model's hard per-forward limit (#1892): an explicit size above it is clamped (one log
/// line at the caller), and "one pass" (0) becomes the limit. One rule for every engine path that prefills.
/// </summary>
public static class PrefillChunkBound
{
    /// <summary>The chunk a prefill uses.</summary>
    /// <param name="configured">The configured chunk size; 0 = not set.</param>
    /// <param name="preferred">The model's preferred chunk (<c>IModel.PreferredPrefillChunkTokens</c>); used when nothing is configured.</param>
    /// <param name="hardMax">The model's hard limit (<c>IModel.MaxPrefillForwardTokens</c>); 0 = none.</param>
    /// <param name="clamped">True when an explicit <paramref name="configured"/> value was above the limit and was lowered to it.</param>
    /// <returns>Tokens per forward pass; 0 = one pass (only when there is no limit).</returns>
    public static int Resolve(int configured, int preferred, int hardMax, out bool clamped)
    {
        clamped = false;
        int chunk = configured > 0 ? configured : Math.Max(0, preferred);
        if (hardMax <= 0) return chunk;
        if (chunk == 0 || chunk > hardMax)
        {
            clamped = configured > hardMax;
            return hardMax;
        }
        return chunk;
    }

    /// <summary>The one log line for a clamped explicit chunk size.</summary>
    /// <param name="configured">The size that was asked for.</param>
    /// <param name="limit">The limit it was lowered to.</param>
    /// <returns>The message.</returns>
    public static string ClampMessage(int configured, int limit)
        => $"[dotllm] --prefill-chunk-size {configured} is above the {limit} rows one forward may take on this model's TurboQuant KV " +
           $"(the Vulkan dispatch grid bounds it); using {limit}.";
}
