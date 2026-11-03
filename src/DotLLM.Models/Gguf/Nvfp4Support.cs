using DotLLM.Core.Configuration;

namespace DotLLM.Models.Gguf;

/// <summary>
/// Load-time rules for NVFP4 weights (issue #1038). An NVFP4 weight <c>X.weight</c> usually ships with
/// a sibling <c>X.scale</c> (F32, one element; llama.cpp's "scale2") that multiplies the matmul output
/// (<c>build_lora_mm</c>: <c>res = mul_mat(w, x) * w_s</c>). In NVIDIA's Nemotron 3.5 Bolt GGUF those
/// scales are about 1e-4, so a loader that silently skips one produces outputs thousands of times too
/// large. Every loader therefore either names the scales it applies or refuses the file.
/// </summary>
public static class Nvfp4Support
{
    /// <summary>Suffix of a weight tensor name.</summary>
    private const string WeightSuffix = ".weight";

    /// <summary>Suffix of the per-tensor second-level scale tensor name.</summary>
    private const string ScaleSuffix = ".scale";

    /// <summary>
    /// Name of the per-tensor scale that belongs to <paramref name="weightName"/>
    /// (<c>blk.0.ssm_in.weight</c> → <c>blk.0.ssm_in.scale</c>).
    /// </summary>
    /// <param name="weightName">A tensor name ending in <c>.weight</c>.</param>
    /// <returns>The sibling scale tensor name.</returns>
    /// <exception cref="ArgumentException"><paramref name="weightName"/> does not end in <c>.weight</c>.</exception>
    public static string ScaleNameFor(string weightName)
    {
        ArgumentNullException.ThrowIfNull(weightName);
        if (!weightName.EndsWith(WeightSuffix, StringComparison.Ordinal))
            throw new ArgumentException($"'{weightName}' is not a weight tensor name.", nameof(weightName));
        return string.Concat(weightName.AsSpan(0, weightName.Length - WeightSuffix.Length), ScaleSuffix);
    }

    /// <summary>
    /// The scale tensor names that sit next to an NVFP4 weight, in tensor-name order.
    /// </summary>
    /// <param name="tensors">The GGUF tensor table.</param>
    /// <returns>Every <c>X.scale</c> whose <c>X.weight</c> is NVFP4.</returns>
    public static IReadOnlyList<string> ScalesOfNvfp4Weights(IReadOnlyDictionary<string, GgufTensorDescriptor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var result = new List<string>();
        foreach (var (name, desc) in tensors)
        {
            if (desc.QuantizationType != QuantizationType.NVFP4 || !name.EndsWith(WeightSuffix, StringComparison.Ordinal))
                continue;
            string scale = ScaleNameFor(name);
            if (tensors.ContainsKey(scale))
                result.Add(scale);
        }

        result.Sort(StringComparer.Ordinal);
        return result;
    }

    /// <summary>
    /// Refuses the file when an NVFP4 weight has a per-tensor scale that <paramref name="loader"/> does
    /// not apply. Call it from every loader that can meet NVFP4 tensors, naming the scales it consumes.
    /// </summary>
    /// <param name="tensors">The GGUF tensor table.</param>
    /// <param name="consumedScales">Scale tensor names the loader applies, or <see langword="null"/> for none.</param>
    /// <param name="loader">The loader's name, for the message.</param>
    /// <exception cref="NotSupportedException">An NVFP4 weight scale would be ignored.</exception>
    public static void RejectUnappliedScales(
        IReadOnlyDictionary<string, GgufTensorDescriptor> tensors,
        IReadOnlySet<string>? consumedScales,
        string loader)
    {
        var unapplied = ScalesOfNvfp4Weights(tensors)
            .Where(s => consumedScales is null || !consumedScales.Contains(s))
            .ToList();
        if (unapplied.Count == 0)
            return;

        throw new NotSupportedException(
            $"{loader} cannot apply the per-tensor scales of {unapplied.Count} NVFP4 weight(s) " +
            $"(first: '{unapplied[0]}'); running without them would scale those projections by ~1/scale. " +
            "NVFP4 is implemented for the Nemotron-H SSM projections only (CPU, Vulkan and CUDA) (#1038).");
    }

    /// <summary>
    /// The scale tensors the generic CPU transformer model (#1578, #1579) applies: the per-tensor scalar of every dense,
    /// attention and LM-head projection, plus - when the architecture's MoE layers go through the grouped quantised MoE kernel
    /// (<paramref name="groupedMoe"/>) - the per-expert vector of every routed bank and the scalar of every shared projection.
    /// Scales of other NVFP4 weights (e.g. an NVFP4 token embedding) are not in the set, so
    /// <see cref="RejectUnappliedScales"/> still refuses a file that carries them.
    /// </summary>
    /// <param name="tensors">The GGUF tensor table.</param>
    /// <param name="groupedMoe">True when routed experts are consumed by the grouped quantised MoE kernel (Flint, MiMo-V2).</param>
    /// <returns>Scale tensor names the CPU transformer loader consumes.</returns>
    public static IReadOnlySet<string> ScalesAppliedByCpuTransformer(IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, bool groupedMoe)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var applied = new HashSet<string>(StringComparer.Ordinal);
        foreach (string scale in ScalesOfNvfp4Weights(tensors))
        {
            string weight = scale[..^ScaleSuffix.Length] + WeightSuffix;
            bool moeWeight = weight.Contains("_exps.", StringComparison.Ordinal) || weight.Contains("_shexp.", StringComparison.Ordinal);
            if (IsCpuTransformerScaledWeight(weight) && (groupedMoe || !moeWeight))
                applied.Add(scale);
        }
        return applied;
    }

    private static readonly System.Text.RegularExpressions.Regex CpuTransformerScaledWeight = new(
        @"^(output|blk\.\d+\.(attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down|ffn_(gate|up|down)_(exps|shexp)))\.weight$",
        System.Text.RegularExpressions.RegexOptions.Compiled | System.Text.RegularExpressions.RegexOptions.CultureInvariant);

    private static bool IsCpuTransformerScaledWeight(string weightName) => CpuTransformerScaledWeight.IsMatch(weightName);

    /// <summary>
    /// Refuses the file when it holds any NVFP4 tensor. For backend paths with no NVFP4 support.
    /// </summary>
    /// <param name="tensors">The GGUF tensor table.</param>
    /// <param name="backend">The backend's name, for the message.</param>
    /// <exception cref="NotSupportedException">An NVFP4 tensor is present.</exception>
    public static void RejectAny(IReadOnlyDictionary<string, GgufTensorDescriptor> tensors, string backend)
        => QuantSupport.RejectAny(tensors, QuantizationType.NVFP4, backend, 1038);
}
