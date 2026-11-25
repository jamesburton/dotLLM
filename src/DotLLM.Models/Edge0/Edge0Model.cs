using DotLLM.Core.Configuration;
using DotLLM.Core.Models;
using DotLLM.Models.Gguf;

namespace DotLLM.Models.Edge0;

/// <summary>
/// Recognises and opens an Edge0 model folder from its skeleton GGUF. An Edge0 folder (HF <c>Edge0/Edge0-145B-q3k</c>) is a
/// <c>qwen4exp</c> GGUF WITHOUT the routed experts (<c>model-skeleton.gguf</c>) next to the strata pack (<see cref="Edge0Pack"/>) and the FP8
/// n-gram table (<c>ple-fp8.gguf</c>); pointing the CLI at the skeleton is enough, the siblings are found in its directory.
/// </summary>
public static class Edge0Model
{
    /// <summary>File name of the FP8 n-gram table next to the skeleton.</summary>
    public const string PleFileName = "ple-fp8.gguf";

    /// <summary>Environment variable selecting the diagnostic column-scale arm (<c>inverse</c> | <c>direct</c> | <c>none</c>); unset = <c>inverse</c>.</summary>
    public const string ColumnScaleEnv = "DOTLLM_EDGE0_COLSCALE";

    /// <summary>
    /// Parses <see cref="ColumnScaleEnv"/>. Only the three names are accepted (an enum parse would also take numbers, and an
    /// unrecognised value would silently run the default): anything else throws. A non-default arm produces wrong output by
    /// design, so it is announced on stderr.
    /// </summary>
    /// <param name="value">The variable's value, or null when unset.</param>
    /// <returns>The arm to run.</returns>
    /// <exception cref="ArgumentException"><paramref name="value"/> is set but is not one of the three names.</exception>
    public static Edge0ColumnScaleMode ColumnScaleModeFromEnvironment(string? value)
    {
        if (string.IsNullOrWhiteSpace(value)) return Edge0ColumnScaleMode.Inverse;
        Edge0ColumnScaleMode mode = value.Trim().ToLowerInvariant() switch
        {
            "inverse" => Edge0ColumnScaleMode.Inverse,
            "direct" => Edge0ColumnScaleMode.Direct,
            "none" => Edge0ColumnScaleMode.None,
            _ => throw new ArgumentException($"{ColumnScaleEnv}='{value}' is not one of inverse | direct | none.", nameof(value)),
        };
        if (mode != Edge0ColumnScaleMode.Inverse)
            Console.Error.WriteLine($"warning: {ColumnScaleEnv}={value}: Edge0 column scale is a diagnostic arm ({mode}); output is deliberately wrong.");
        return mode;
    }

    /// <summary>
    /// True when <paramref name="gguf"/> is a qwen4exp skeleton: it has the block-0 router but no fused routed-expert banks
    /// (a stock qwen4exp file always carries <c>blk.0.ffn_gate_exps.weight</c>).
    /// </summary>
    public static bool IsSkeleton(GgufFile gguf)
    {
        ArgumentNullException.ThrowIfNull(gguf);
        var t = gguf.TensorsByName;
        return t.ContainsKey("blk.0.ffn_gate_inp.weight") && !t.ContainsKey("blk.0.ffn_gate_exps.weight");
    }

    /// <summary>
    /// Opens the pack and the FP8 n-gram table that sit in the same directory as the skeleton GGUF (the path it was opened from, so an
    /// HF snapshot link directory works; nothing is copied). The caller (or the model it hands them to) owns both.
    /// </summary>
    /// <param name="skeleton">The opened skeleton.</param>
    /// <param name="config">Its configuration.</param>
    /// <exception cref="FileNotFoundException">A sibling file is missing (the message names it and the file it was needed for).</exception>
    public static (Edge0Pack Pack, GgufFile Ple) OpenSiblings(GgufFile skeleton, ModelConfig config)
    {
        ArgumentNullException.ThrowIfNull(skeleton);
        ArgumentNullException.ThrowIfNull(config);
        string dir = Path.GetDirectoryName(Path.GetFullPath(skeleton.ShardPaths[0])) ?? ".";
        string pleFile = Path.Combine(dir, PleFileName);
        if (!File.Exists(pleFile))
            throw new FileNotFoundException(
                $"This GGUF has no routed experts (an Edge0 skeleton), so its pack and n-gram table must sit beside it, but {pleFile} is missing.", pleFile);
        var moe = config.Moe ?? throw new InvalidDataException("Edge0 skeleton has no MoE configuration.");
        bool q4 = string.Equals(Environment.GetEnvironmentVariable("DOTLLM_EDGE0_LATENT"), "q4", StringComparison.OrdinalIgnoreCase);
        var pack = Edge0Pack.Open(dir, config.HiddenSize, moe.MoeIntermediateSize, preferLatentQ4: q4);
        try
        {
            pack.ColumnScaleMode = ColumnScaleModeFromEnvironment(Environment.GetEnvironmentVariable(ColumnScaleEnv));
            return (pack, GgufFile.Open(pleFile));
        }
        catch
        {
            pack.Dispose();
            throw;
        }
    }
}
