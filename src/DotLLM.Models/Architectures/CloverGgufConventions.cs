using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Architectures;

/// <summary>
/// Load-time tensor conventions of DiffuSpace Clover GGUFs (<c>clover_moe</c>, issue #1794), shared by every Qwen3MoeHybrid host
/// (CPU, and through it Vulkan; CUDA loads its own copy).
/// </summary>
/// <remarks>
/// <para>
/// <b><c>ssm_a</c> holds the raw HF <c>A_log</c>.</b> qwen35moe GGUFs store <c>ssm_a = -exp(A_log)</c> (strictly negative; the
/// GDN gate is <c>g = ssm_a * softplus(alpha + dt_bias)</c>). The official <c>Diffuspace/Clover-35B-A3B-Preview-GGUF</c> files were
/// written by DiffuSpace's own converter, which stores <c>A_log</c> itself under the <c>ssm_a</c> name: layer 0's values span
/// -3.98..+4.28, and Qwen3.6-35B-A3B's <c>ssm_a</c> is exactly <c>-exp</c> of them (-exp(-3.3125) = -0.03643 vs -0.036425). Read as
/// qwen35moe, the model emits <c>GGGG…</c>. llama.cpp PR #31306's converter (<c>conversion/clover.py</c>) instead writes the qwen35moe
/// form and drops the bidirectional tensors, so both layouts exist.
/// </para>
/// <para>
/// Detection, per GDN layer: <c>-exp(x)</c> is strictly negative, so any positive entry proves <c>A_log</c>; the presence of the
/// DiffuSpace-only <c>ssm_conv1d.weight_right</c> tensor identifies the converter that writes <c>A_log</c> even if every entry
/// happened to be negative. A file with neither is the llama.cpp-converter form and is left alone.
/// </para>
/// </remarks>
public static class CloverGgufConventions
{
    /// <summary>The four bidirectional-preview GDN tensor suffixes the causal forward never reads (kept for #1794 stage 3).</summary>
    public static readonly string[] BidirectionalPreviewSuffixes =
    [
        "ssm_conv1d.weight_right",   // HF conv_right_weight            [3, convDim]
        "ssm_bd_logit_scale.weight", // HF bd_softmax_logit_scale       [nVHead]
        "ssm_bd_history.bias",       // HF bd_softmax_history_bias      [nVHead]
        "ssm_bd_history.weight",     // HF bd_softmax_history_c         [nVHead]
    ];

    /// <summary>
    /// Whether <paramref name="a"/> (a GDN layer's <c>ssm_a</c>, read verbatim) is the raw <c>A_log</c> that must be turned into
    /// <c>-exp(A_log)</c> before use. Always <see langword="false"/> for a non-Clover model.
    /// </summary>
    /// <param name="config">Model configuration (Clover iff <see cref="ModelConfig.Clover"/> is non-null).</param>
    /// <param name="tensors">The GGUF tensor directory.</param>
    /// <param name="layerPrefix">The layer prefix, e.g. <c>blk.0</c>.</param>
    /// <param name="a">The layer's <c>ssm_a</c> values as stored.</param>
    public static bool SsmAIsALog(ModelConfig config, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
                                  string layerPrefix, ReadOnlySpan<float> a)
    {
        if (config.Clover is null)
            return false;
        if (tensors.ContainsKey($"{layerPrefix}.ssm_conv1d.weight_right"))
            return true;
        foreach (float v in a)
            if (v > 0f)
                return true;
        return false;
    }

    /// <summary>In place: <c>a[i] = -exp(a[i])</c> (HF <c>-A_log.float().exp()</c>), the qwen35moe <c>ssm_a</c> form.</summary>
    /// <param name="a">The per-head <c>A_log</c> values; overwritten.</param>
    public static void ALogToSsmA(Span<float> a)
    {
        for (int i = 0; i < a.Length; i++)
            a[i] = -MathF.Exp(a[i]);
    }

    /// <summary>
    /// Writes one load-log line naming the Clover tensors the causal forward leaves unconsumed (they stay in the mmap). No-op for
    /// non-Clover models or files without them.
    /// </summary>
    /// <param name="config">Model configuration.</param>
    /// <param name="tensors">The GGUF tensor directory.</param>
    /// <param name="host">Host name for the log line.</param>
    public static void LogUnconsumedTensors(ModelConfig config, IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string host)
    {
        if (config.Clover is null)
            return;
        var counts = new int[BidirectionalPreviewSuffixes.Length];
        long bytes = 0;
        foreach (var (name, desc) in tensors)
        {
            for (int s = 0; s < BidirectionalPreviewSuffixes.Length; s++)
            {
                if (name.EndsWith("." + BidirectionalPreviewSuffixes[s], StringComparison.Ordinal))
                {
                    counts[s]++;
                    bytes += desc.Shape.ElementCount * sizeof(float);
                }
            }
        }
        int total = counts.Sum();
        if (total == 0)
            return;
        var parts = BidirectionalPreviewSuffixes.Select((sfx, i) => $"{sfx} x{counts[i]}");
        Console.Error.WriteLine(
            $"[clover] {host}: causal (AR) forward only; ignoring {total} bidirectional block-diffusion tensors "
            + $"({string.Join(", ", parts)}; ~{bytes / 1024} KiB, left in the mmap). Block-diffusion decoding is not implemented (#1794).");
    }
}
