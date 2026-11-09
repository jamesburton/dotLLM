using System.Text;

namespace DotLLM.ChunkCache;

/// <summary>
/// One option of a verb: the single registration the parser (<see cref="VerbDef.Spec"/>) and the help text
/// (<see cref="HelpWriter"/>) both read, so an option cannot be accepted without being documented (a test fails on one with
/// no help text).
/// </summary>
/// <param name="Name">The option name without the leading dashes.</param>
/// <param name="Kind">How it takes values (<see cref="OptionKind"/>).</param>
/// <param name="Help">What it does, in one or two sentences.</param>
/// <param name="Value">The value form for the help line (<c>FILE.gguf</c>, <c>N</c>, <c>symbol|fixed</c>); null for a flag.</param>
/// <param name="Default">What applies when the option is not given (<c>off</c>, <c>required</c>, <c>none</c>, <c>512</c>).</param>
/// <param name="Values">The valid values or range with its unit, when the value form does not say it (<c>integer 1..1048576, tokens</c>).</param>
/// <param name="Env">The environment variable that supplies or complements it, if any.</param>
internal sealed record OptionSpec(string Name, OptionKind Kind, string Help, string? Value = null, string Default = "", string? Values = null, string? Env = null)
{
    /// <summary>The help line header: <c>--name VALUE</c>.</summary>
    public string Header => Kind == OptionKind.Flag ? "--" + Name : $"--{Name} {Value}";

    /// <summary>The description with its <c>Values</c>, <c>Default</c>, repeatability and <c>Env</c> clauses (format <c>Text. Values: v. Default: d. Env: E.</c>).</summary>
    public string Description => Compose(withDefault: true);

    /// <summary>The description without the <c>Default</c> clause (the docs page shows the default in its own column).</summary>
    public string DescriptionWithoutDefault => Compose(withDefault: false);

    private string Compose(bool withDefault)
    {
        var sb = new StringBuilder(Help.Trim());
        if (Values is { Length: > 0 }) sb.Append(" Values: ").Append(Sentence(Values));
        if (Kind == OptionKind.List) sb.Append(" Repeatable: one value per flag, values may not contain commas.");
        if (withDefault && Default.Length > 0) sb.Append(" Default: ").Append(Sentence(Default));
        if (Env is { Length: > 0 }) sb.Append(" Env: ").Append(Sentence(Env));
        return sb.ToString();
    }

    private static string Sentence(string s) => s.EndsWith('.') ? s : s + ".";
}

/// <summary>
/// A verb: its option registry (the parser's <see cref="Spec"/> is derived from <see cref="Options"/>) and everything its
/// <c>--help</c> prints. <see cref="ChunkCacheCli"/> parses with <see cref="Spec"/> and prints <see cref="Help"/>.
/// </summary>
internal sealed class VerbDef
{
    private readonly Lazy<IReadOnlyDictionary<string, OptionKind>> _spec;
    private readonly Lazy<string> _help;

    /// <summary>Creates a verb definition.</summary>
    /// <param name="name">Verb name.</param>
    /// <param name="summary">One line: what the verb does (also its line in the top-level usage).</param>
    /// <param name="usage">Synopsis lines, one per form.</param>
    /// <param name="description">Paragraphs (may be empty), as authored: wrapped by hand.</param>
    /// <param name="options">The registered options.</param>
    /// <param name="jsonFields">One line per <c>--json</c> field (<c>name  what it is</c>); every property of <paramref name="jsonTypes"/> must appear.</param>
    /// <param name="jsonTypes">The result types whose properties the JSON fields document (a test compares them).</param>
    /// <param name="examples">Complete command lines, each starting with <c>chunkcache &lt;verb&gt;</c> (a test parses them).</param>
    /// <param name="strictMeaning">What exit code 3 means for this verb, or null when the verb never exits 3.</param>
    /// <param name="extraExitNotes">Verb-specific details of the other exit codes, or null.</param>
    public VerbDef(string name, string summary, IReadOnlyList<string> usage, string description, IReadOnlyList<OptionSpec> options, string jsonFields,
        IReadOnlyList<Type> jsonTypes, IReadOnlyList<string> examples, string? strictMeaning = null, string? extraExitNotes = null)
    {
        Name = name;
        Summary = summary;
        Usage = usage;
        Description = description;
        Options = options;
        JsonFields = jsonFields;
        JsonTypes = jsonTypes;
        Examples = examples;
        StrictMeaning = strictMeaning;
        ExtraExitNotes = extraExitNotes;
        _spec = new(() => options.ToDictionary(o => o.Name, o => o.Kind, StringComparer.Ordinal));
        _help = new(() => HelpWriter.Render(this));
    }

    /// <summary>Verb name.</summary>
    public string Name { get; }

    /// <summary>One-line summary.</summary>
    public string Summary { get; }

    /// <summary>Synopsis lines.</summary>
    public IReadOnlyList<string> Usage { get; }

    /// <summary>Hand-wrapped description paragraphs.</summary>
    public string Description { get; }

    /// <summary>The registered options (the single registration).</summary>
    public IReadOnlyList<OptionSpec> Options { get; }

    /// <summary>The <c>--json</c> field list, one line per field.</summary>
    public string JsonFields { get; }

    /// <summary>The result types the <see cref="JsonFields"/> document.</summary>
    public IReadOnlyList<Type> JsonTypes { get; }

    /// <summary>Example command lines.</summary>
    public IReadOnlyList<string> Examples { get; }

    /// <summary>What exit code 3 means here (null: never used).</summary>
    public string? StrictMeaning { get; }

    /// <summary>Verb-specific exit-code notes.</summary>
    public string? ExtraExitNotes { get; }

    /// <summary>The parser's option table, derived from <see cref="Options"/>.</summary>
    public IReadOnlyDictionary<string, OptionKind> Spec => _spec.Value;

    /// <summary>The full <c>--help</c> text.</summary>
    public string Help => _help.Value;

    /// <summary>The synopsis and a pointer to the full help (printed after a usage error).</summary>
    public string UsageHint => "usage: " + string.Join(Environment.NewLine + "       ", Usage) + Environment.NewLine + $"Run 'chunkcache {Name} --help' for every option, the exit codes and examples.";
}

/// <summary>Renders <see cref="VerbDef"/> into <c>--help</c> text.</summary>
internal static class HelpWriter
{
    private const int Width = 110;

    /// <summary>The <c>--verbose</c> line, shared by every verb (it is stripped before parsing, so no option table holds it).</summary>
    public const string VerboseHelp = "Add the stack trace of an error to the message on stderr. Allowed anywhere on the command line; never changes the exit code or the --json output. Default: off.";

    /// <summary>The exit codes common to every verb.</summary>
    public static IReadOnlyList<(int Code, string Meaning)> ExitCodeLines { get; } =
    [
        (ExitCodes.Ok, "ok"),
        (ExitCodes.Error, "runtime error: I/O, a busy store lock, a store owned by a server or another writer, a model that cannot be loaded, a symbol not found, an unreadable index or ledger, a server error or an unreachable server"),
        (ExitCodes.Usage, "usage error: unknown verb or option, a missing or malformed value, options that contradict each other"),
    ];

    /// <summary>The full help of <paramref name="verb"/>.</summary>
    public static string Render(VerbDef verb)
    {
        var sb = new StringBuilder();
        sb.Append("chunkcache ").Append(verb.Name).Append(": ").AppendLine(verb.Summary).AppendLine();
        sb.AppendLine("usage:");
        foreach (string line in verb.Usage) sb.Append("  ").AppendLine(line);
        if (verb.Description.Length > 0) sb.AppendLine().AppendLine(verb.Description.TrimEnd());
        sb.AppendLine().AppendLine("options:");
        foreach (OptionSpec o in verb.Options)
        {
            sb.Append("  ").AppendLine(o.Header);
            Wrap(sb, o.Description, "        ");
        }
        sb.AppendLine().AppendLine("global option (any verb):");
        sb.AppendLine("  --verbose");
        Wrap(sb, VerboseHelp, "        ");
        string?[] envs = verb.Options.Select(o => o.Env).Where(e => e is not null).Distinct(StringComparer.Ordinal).ToArray();
        if (envs.Length > 0)
        {
            sb.AppendLine().AppendLine("environment:");
            foreach (string? env in envs)
                if (env == ServerConnection.ApiKeyVariable)
                {
                    sb.Append("  ").AppendLine(env);
                    Wrap(sb, "The API key for --server. Read from the environment only (there is no key option, so it never shows in a process list or a shell history), sent as X-API-Key, and replaced by *** in every message, including the --json error object. Redirects are not followed, so a redirect cannot carry it to another origin.", "        ");
                }
        }
        sb.AppendLine().AppendLine("exit codes:");
        foreach ((int code, string meaning) in ExitCodeLines) AppendExit(sb, code, meaning);
        AppendExit(sb, ExitCodes.StrictFailure, verb.StrictMeaning ?? "not used by this verb");
        if (verb.ExtraExitNotes is { Length: > 0 }) Wrap(sb, verb.ExtraExitNotes, "      ");
        sb.AppendLine().AppendLine("--json output (one JSON document on stdout, snake_case names; progress and warnings go to stderr):");
        foreach (string line in verb.JsonFields.Trim().Split('\n')) sb.Append("  ").AppendLine(line.TrimEnd());
        sb.Append("  A failure prints {\"error\": {\"code\", \"message\", \"exit_code\"}} instead, with the non-zero exit code.").AppendLine();
        sb.AppendLine().AppendLine("examples:");
        foreach (string example in verb.Examples) sb.Append("  ").AppendLine(example);
        return sb.ToString().TrimEnd();
    }

    private static void AppendExit(StringBuilder sb, int code, string meaning) => Wrap(sb, $"{code}  {meaning}", "  ", "      ");

    /// <summary>Appends <paramref name="text"/> word-wrapped to <see cref="Width"/> columns; the first line starts with <paramref name="first"/>.</summary>
    private static void Wrap(StringBuilder sb, string text, string first, string? rest = null)
    {
        rest ??= first;
        string prefix = first;
        int column = prefix.Length;
        sb.Append(prefix);
        bool lineStart = true;
        foreach (string word in text.Split([' ', '\n'], StringSplitOptions.RemoveEmptyEntries))
        {
            if (!lineStart && column + 1 + word.Length > Width)
            {
                sb.AppendLine().Append(rest);
                column = rest.Length;
                lineStart = true;
            }
            if (!lineStart) { sb.Append(' '); column++; }
            sb.Append(word);
            column += word.Length;
            lineStart = false;
        }
        sb.AppendLine();
    }
}
