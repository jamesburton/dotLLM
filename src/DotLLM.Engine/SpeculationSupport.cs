using DotLLM.Core.Attention;
using DotLLM.Core.Models;

namespace DotLLM.Engine;

/// <summary>
/// When draft-model and prompt-lookup speculation cannot keep greedy output equal to plain decode (#2030). A rejected draft is
/// undone by cutting the KV-cache length (plus a recurrent checkpoint where the model has one), and the speculative forwards carry
/// no adapter. Anything those two do not cover must decode without speculation.
/// </summary>
public static class SpeculationSupport
{
    /// <summary>Why this target can never speculate, or null.</summary>
    /// <param name="target">The model that verifies drafts.</param>
    /// <returns>The reason, or null.</returns>
    public static string? TargetBlocker(IModel target)
    {
        ArgumentNullException.ThrowIfNull(target);
        return target.RequiresPerSequenceState && !target.SupportsRecurrentStateCheckpoint
            ? $"{target.GetType().Name} keeps sequence state it cannot checkpoint, so a rejected draft could not be undone"
            : null;
    }

    /// <summary>Why this request cannot speculate, or null.</summary>
    /// <param name="target">The model that verifies drafts.</param>
    /// <param name="kvCache">The request's target KV-cache.</param>
    /// <param name="hasAdapter">True when the request carries a LoRA adapter.</param>
    /// <returns>The reason, or null.</returns>
    public static string? RequestBlocker(IModel target, IKvCache kvCache, bool hasAdapter)
    {
        if (TargetBlocker(target) is { } blocker) return blocker;
        if (hasAdapter) return "the speculative forwards do not apply a LoRA adapter";
        // Rows pushed out of the FP16 window are quantized in place; cutting the length does not bring them back.
        return kvCache is IQuantizedKvCache { WindowCapacity: > 0 }
            ? "a quantized KV-cache with an FP16 window cannot be rolled back once rows have left the window"
            : null;
    }
}
