using System.Globalization;

namespace DotLLM.Cli.Commands;

/// <summary>
/// Parses a <c>--lora</c> specifier of the form <c>path</c> or <c>path=weight</c>.
/// Windows drive letters (e.g. <c>C:/x</c>) are handled correctly — the colon
/// after a single drive letter is never mistaken for a weight separator because
/// <see cref="Parse"/> uses the <em>last</em> <c>=</c> sign, not the colon.
/// </summary>
public static class LoraSpec
{
    /// <summary>
    /// Parses <paramref name="spec"/> into a (Path, Weight) tuple.
    /// </summary>
    /// <param name="spec">
    /// A path string, optionally followed by <c>=weight</c>
    /// (e.g. <c>C:/adapters/lora1</c> or <c>C:/adapters/lora1=0.7</c>).
    /// </param>
    /// <returns>
    /// The resolved path and blend weight. Weight defaults to <c>1.0</c> when not specified.
    /// </returns>
    /// <exception cref="ArgumentException">Thrown when <paramref name="spec"/> is null or empty.</exception>
    public static (string Path, float Weight) Parse(string spec)
    {
        ArgumentException.ThrowIfNullOrEmpty(spec);

        int eq = spec.LastIndexOf('=');
        if (eq > 0 && float.TryParse(spec.AsSpan(eq + 1), NumberStyles.Float,
                                     CultureInfo.InvariantCulture, out float w))
            return (spec[..eq], w);

        return (spec, 1f);
    }
}
