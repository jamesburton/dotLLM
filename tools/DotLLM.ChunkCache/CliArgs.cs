using System.Globalization;

namespace DotLLM.ChunkCache;

/// <summary>A command-line usage error (exit code <see cref="ExitCodes.Usage"/>).</summary>
internal sealed class UsageException(string message) : Exception(message);

/// <summary>A <c>--strict</c> check failed (exit code <see cref="ExitCodes.StrictFailure"/>).</summary>
internal sealed class StrictFailureException(string message) : Exception(message);

/// <summary>The process exit codes (documented in the help text).</summary>
internal static class ExitCodes
{
    /// <summary>Success.</summary>
    public const int Ok = 0;

    /// <summary>Runtime error: I/O, a busy store lock, a model that cannot be loaded, a symbol not found, an unreadable index.</summary>
    public const int Error = 1;

    /// <summary>Usage error: unknown verb or option, missing or malformed value.</summary>
    public const int Usage = 2;

    /// <summary>A <c>--strict</c> check failed (build: skipped projects, load failures or excluded files; status: not up to date).</summary>
    public const int StrictFailure = 3;
}

/// <summary>How an option takes values.</summary>
internal enum OptionKind
{
    /// <summary>No value (<c>--json</c>).</summary>
    Flag,

    /// <summary>One value; given twice is a usage error.</summary>
    Single,

    /// <summary>Repeatable, and each value is split on commas (<c>--path a,b --path c</c>).</summary>
    List,
}

/// <summary>
/// Parsed arguments of one verb: <c>--name value</c>, <c>--name=value</c> or a flag <c>--name</c>. Unknown options, a missing
/// value or a positional argument are usage errors.
/// </summary>
internal sealed class CliArgs
{
    private readonly Dictionary<string, List<string>> _values = new(StringComparer.Ordinal);
    private readonly HashSet<string> _flags = new(StringComparer.Ordinal);

    private CliArgs(string verb) => Verb = verb;

    /// <summary>The verb.</summary>
    public string Verb { get; }

    /// <summary>Parses <paramref name="args"/> (after the verb) against <paramref name="spec"/>.</summary>
    public static CliArgs Parse(string verb, IEnumerable<string> args, IReadOnlyDictionary<string, OptionKind> spec)
    {
        var parsed = new CliArgs(verb);
        using IEnumerator<string> e = args.GetEnumerator();
        while (e.MoveNext())
        {
            string arg = e.Current;
            if (!arg.StartsWith("--", StringComparison.Ordinal) || arg.Length == 2)
                throw new UsageException($"{verb}: unexpected argument '{arg}' (options start with --).");
            string name = arg[2..];
            string? inline = null;
            int eq = name.IndexOf('=', StringComparison.Ordinal);
            if (eq >= 0)
            {
                inline = name[(eq + 1)..];
                name = name[..eq];
            }
            if (!spec.TryGetValue(name, out OptionKind kind))
                throw new UsageException($"{verb}: unknown option '--{name}'. Run 'chunkcache {verb} --help'.");
            if (kind == OptionKind.Flag)
            {
                if (inline is not null) throw new UsageException($"{verb}: --{name} takes no value.");
                parsed._flags.Add(name);
                continue;
            }
            string value;
            if (inline is not null) value = inline;
            else if (e.MoveNext()) value = e.Current;
            else throw new UsageException($"{verb}: --{name} needs a value.");
            if (!parsed._values.TryGetValue(name, out List<string>? list)) parsed._values[name] = list = [];
            else if (kind == OptionKind.Single) throw new UsageException($"{verb}: --{name} given more than once.");
            if (kind == OptionKind.List)
                list.AddRange(value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
            else list.Add(value);
        }
        return parsed;
    }

    /// <summary>Whether flag <paramref name="name"/> was given.</summary>
    public bool Flag(string name) => _flags.Contains(name);

    /// <summary>The value of <paramref name="name"/>, or null.</summary>
    public string? Value(string name) => _values.TryGetValue(name, out List<string>? v) ? v[0] : null;

    /// <summary>The value of <paramref name="name"/>; a usage error when absent.</summary>
    public string Required(string name) => Value(name) ?? throw new UsageException($"{Verb}: --{name} is required.");

    /// <summary>The values of list option <paramref name="name"/> (empty when absent).</summary>
    public IReadOnlyList<string> List(string name) => _values.TryGetValue(name, out List<string>? v) ? v : [];

    /// <summary>An integer option in [<paramref name="min"/>, <paramref name="max"/>], or <paramref name="fallback"/>.</summary>
    public int Int(string name, int fallback, int min = int.MinValue, int max = int.MaxValue) => IntOrNull(name, min, max) ?? fallback;

    /// <summary>An integer option in range, or null when absent.</summary>
    public int? IntOrNull(string name, int min = int.MinValue, int max = int.MaxValue)
    {
        string? s = Value(name);
        if (s is null) return null;
        if (!int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) || v < min || v > max)
            throw new UsageException($"{Verb}: --{name} must be an integer in [{min}, {max}], not '{s}'.");
        return v;
    }

    /// <summary>A non-negative number of seconds, or <paramref name="fallback"/>.</summary>
    public TimeSpan Seconds(string name, TimeSpan fallback)
    {
        string? s = Value(name);
        if (s is null) return fallback;
        if (!double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) || v < 0 || !double.IsFinite(v))
            throw new UsageException($"{Verb}: --{name} must be a non-negative number of seconds, not '{s}'.");
        return TimeSpan.FromSeconds(v);
    }
}
