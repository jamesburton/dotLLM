using System.Collections.Concurrent;
using DotLLM.Vulkan.Kernels;

namespace DotLLM.Vulkan;

/// <summary>
/// One-shot warnings for prefill attention fast paths that did not engage (issue #441).
/// </summary>
/// <remarks>
/// <para>
/// The bug this exists to prevent is not a wrong number, it is <i>silence</i>. Bonsai 2 declares
/// <c>attention.key_length = value_length = 256</c>; <c>attention_flash_f32.comp</c> was compiled
/// with <c>MAX_HEAD_DIM = 128</c>; so every full-attention layer fell through to the per-token
/// <see cref="AttentionF32Kernel"/> on every prefill pass, with nothing anywhere saying so. The
/// bucket was ~20 % of the prefill pass and it took a per-op profiling campaign to notice.
/// </para>
/// <para>
/// Every Vulkan model that gates flash attention on a head-dimension bound MUST report through
/// here when the gate closes. The message names the model, the head dim, the bound that rejected
/// it and what to do about it, so the next person reads it instead of rediscovering it.
/// </para>
/// <para>
/// Warnings are de-duplicated by full text, so a 64-layer model prints one line, not 64. Output
/// goes to stderr (the convention across this backend - see <c>VulkanDevice</c>'s
/// <c>[vulkan-mem]</c> lines) and can be silenced with
/// <c>DOTLLM_VULKAN_QUIET_ATTENTION_FALLBACK=1</c> for benchmark harnesses that parse stderr.
/// The reported set stays queryable either way so tests can assert on it.
/// </para>
/// </remarks>
public static class VulkanAttentionFallbackDiagnostics
{
    /// <summary>Set to <c>1</c> to suppress the stderr lines (the reported set is still recorded).</summary>
    public const string QuietEnvVar = "DOTLLM_VULKAN_QUIET_ATTENTION_FALLBACK";

    private static readonly ConcurrentDictionary<string, byte> Seen = new(StringComparer.Ordinal);

    /// <summary>Every distinct warning raised in this process, in no particular order.</summary>
    public static IReadOnlyCollection<string> Reported => (IReadOnlyCollection<string>)Seen.Keys;

    /// <summary>Clears the de-duplication set. Test hook; not for production use.</summary>
    public static void Reset() => Seen.Clear();

    /// <summary>
    /// Reports that the prefill flash-attention kernel will not be used for this model because
    /// its head dimension exceeds what the loaded shader variants can dispatch.
    /// </summary>
    /// <param name="modelKind">Model class raising the warning, e.g. <c>Qwen3HybridDense</c>.</param>
    /// <param name="headDim">The model's per-head dimension.</param>
    /// <param name="supportedMaxHeadDim">Largest head dimension the loaded flash variants accept.</param>
    public static void ReportHeadDimTooWide(string modelKind, int headDim, int supportedMaxHeadDim)
        => Report(
            $"[vulkan-attn] {modelKind}: prefill flash attention DISABLED - headDim {headDim} > " +
            $"shader MAX_HEAD_DIM {supportedMaxHeadDim}. Falling back to the per-token attention " +
            $"kernel, which re-reads every K/V row once per query token instead of once per " +
            $"{VulkanFlashAttentionF32Kernel.QueryTileRows}-row tile. Fix: add/compile a wider " +
            $"attention_flash_f32_hd*.comp variant (issue #441), or set {VulkanFlashAttentionF32Kernel.WideVariantEnvVar} " +
            $"if a suitable one exists but was switched off.");

    /// <summary>
    /// Reports that the flash-attention SPV could not be loaded at all, so prefill runs on the
    /// per-token kernel.
    /// </summary>
    /// <param name="modelKind">Model class raising the warning.</param>
    /// <param name="reason">Short human-readable cause (missing SPV, disabled by env var, ...).</param>
    public static void ReportUnavailable(string modelKind, string reason)
        => Report(
            $"[vulkan-attn] {modelKind}: prefill flash attention DISABLED - {reason}. " +
            $"Falling back to the per-token attention kernel.");

    /// <summary>
    /// The single place a Vulkan model should obtain its prefill flash-attention kernel.
    /// Applies every gate (env-var opt-out, head-dim bound, missing SPV) and reports a one-shot
    /// warning whenever one of them closes, so the fallback can never again be silent.
    /// </summary>
    /// <param name="device">Device to create the kernel on.</param>
    /// <param name="spvDir">Directory holding the compiled SPIR-V blobs.</param>
    /// <param name="headDim">The model's per-head dimension.</param>
    /// <param name="modelKind">Model class name for the warning text.</param>
    /// <returns>A kernel that can dispatch <paramref name="headDim"/>, or <c>null</c>.</returns>
    public static VulkanFlashAttentionF32Kernel? CreatePrefillFlashAttention(
        VulkanDevice device, string spvDir, int headDim, string modelKind)
    {
        if (VulkanTransformerModel.IsFlashAttentionDisabled())
        {
            ReportUnavailable(modelKind, $"{VulkanTransformerModel.DisableFlashAttentionEnvVar}=1");
            return null;
        }

        if (headDim > VulkanFlashAttentionF32Kernel.MaxSupportedHeadDim)
        {
            ReportHeadDimTooWide(modelKind, headDim, VulkanFlashAttentionF32Kernel.MaxSupportedHeadDim);
            return null;
        }

        VulkanFlashAttentionF32Kernel? kernel = VulkanFlashAttentionF32Kernel.TryCreate(device, spvDir);
        if (kernel is null)
        {
            ReportUnavailable(modelKind, "attention_flash_f32.spv is missing or its pipeline failed to build");
            return null;
        }

        if (headDim > kernel.SupportedMaxHeadDim)
        {
            // The wide SPV was absent from this build (or switched off): the base shader alone
            // cannot take this head. Report and drop the pipelines we would never dispatch.
            ReportHeadDimTooWide(modelKind, headDim, kernel.SupportedMaxHeadDim);
            kernel.Dispose();
            return null;
        }

        return kernel;
    }

    private static void Report(string message)
    {
        if (!Seen.TryAdd(message, 0)) return;
        if (Environment.GetEnvironmentVariable(QuietEnvVar) == "1") return;
        Console.Error.WriteLine(message);
    }
}
