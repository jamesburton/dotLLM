using System.Globalization;
using System.Reflection;
using System.Text;
using System.Text.RegularExpressions;
using System.Xml.Linq;
using DotLLM.Cli;
using Spectre.Console;
using Spectre.Console.Cli;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>One argument or option of a command, as Spectre reports it plus the default read from the settings class.</summary>
internal sealed record CliParam(
    bool IsArgument, string Name, string Short, string Description, string Default, string TypeLabel, bool Required, bool IsFlag, string Env)
{
    /// <summary>Markdown cell for the first column of the generated tables.</summary>
    public string DisplayName
    {
        get
        {
            if (IsArgument) return Required ? $"`<{Name}>`" : $"`[{Name}]`";
            string s = $"`--{Name}`" + (Short.Length > 0 ? $", `-{Short}`" : "");
            return IsFlag || TypeLabel.Length == 0 ? s : s + $" `<{TypeLabel}>`";
        }
    }
}

/// <summary>A command (or branch) of the dotllm tree.</summary>
internal sealed record CliCommand(string Path, string Description, bool IsBranch, IReadOnlyList<CliParam> Params, string? SettingsType)
{
    /// <summary>True for the Debug-build-only <c>debug</c> branch, which Release builds do not register.</summary>
    public bool IsDebug => Path == "debug" || Path.StartsWith("debug ", StringComparison.Ordinal);
}

/// <summary>
/// The real dotllm command tree, read through Spectre's own <c>cli xmldoc</c> command over <see cref="CliConfiguration"/>, with
/// defaults taken from the settings classes by reflection (xmldoc does not report them). This is the "generated from truth" source for
/// the CLI tests and for the option tables of <c>docs/CLI.md</c>.
/// </summary>
internal static class CliCommandModel
{
    private static readonly Regex EnvRegex = new(@"\b(?:DOTLLM_[A-Z0-9_]*[A-Z0-9]|HF_HOME|HF_HUB_CACHE|HF_TOKEN|OLLAMA_MODELS)\b", RegexOptions.Compiled | RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(5));

    private static readonly Lazy<IReadOnlyList<CliCommand>> s_all = new(Build);

    /// <summary>Every command and branch in registration order (depth first).</summary>
    public static IReadOnlyList<CliCommand> All => s_all.Value;

    private static IReadOnlyList<CliCommand> Build()
    {
        var sw = new StringWriter();
        var console = AnsiConsole.Create(new AnsiConsoleSettings
        {
            Ansi = AnsiSupport.No,
            ColorSystem = ColorSystemSupport.NoColors,
            Out = new AnsiConsoleOutput(sw),
            Interactive = InteractionSupport.No,
        });
        console.Profile.Width = 1_000_000;   // no wrapping: a wrap can split a word inside a description

        var app = new CommandApp();
        app.Configure(c =>
        {
            CliConfiguration.Configure(c);
            c.ConfigureConsole(console);
        });
        int rc = app.Run(["cli", "xmldoc"]);
        if (rc != 0)
            throw new InvalidOperationException($"'cli xmldoc' exited {rc} (ValidateExamples failure?): {sw}");

        var doc = XDocument.Parse(sw.ToString());
        var list = new List<CliCommand>();
        foreach (var top in doc.Root!.Elements("Command"))
            Walk(top, "", list);
        return list;
    }

    private static void Walk(XElement cmd, string parentPath, List<CliCommand> into)
    {
        string name = (string)cmd.Attribute("Name")!;
        string path = parentPath.Length == 0 ? name : parentPath + " " + name;
        bool isBranch = string.Equals((string?)cmd.Attribute("IsBranch"), "true", StringComparison.OrdinalIgnoreCase);
        string? settingsName = (string?)cmd.Attribute("Settings");
        Type? settings = settingsName is null ? null : typeof(CliConfiguration).Assembly.GetType(settingsName);

        var ps = new List<CliParam>();
        foreach (var p in cmd.Element("Parameters")?.Elements() ?? [])
            ps.Add(ToParam(p, settings));

        into.Add(new CliCommand(path, Collapse(cmd.Element("Description")?.Value), isBranch, ps, settingsName));
        foreach (var child in cmd.Elements("Command"))
            Walk(child, path, into);
    }

    private static CliParam ToParam(XElement p, Type? settings)
    {
        bool isArg = p.Name.LocalName == "Argument";
        string desc = Collapse(p.Element("Description")?.Value);
        string clr = ((string?)p.Attribute("ClrType") ?? "").Replace("System.", "");
        string kind = (string?)p.Attribute("Kind") ?? "scalar";
        bool required = string.Equals((string?)p.Attribute("Required"), "true", StringComparison.OrdinalIgnoreCase);
        bool isFlag = !isArg && (kind == "flag" || clr is "Boolean");

        string name = isArg ? (string)p.Attribute("Name")! : (string)p.Attribute("Long")!;
        string shortName = isArg ? "" : (string?)p.Attribute("Short") ?? "";
        int position = isArg ? int.Parse((string)p.Attribute("Position")!, CultureInfo.InvariantCulture) : -1;

        string def = settings is null ? "" : DefaultOf(settings, isArg, name, position);
        string label = clr switch
        {
            "Int32" or "Int64" => "int",
            "Single" or "Double" => "number",
            "String" => "text",
            "Boolean" => "",
            _ when clr.EndsWith("[]", StringComparison.Ordinal) => "text",
            _ => clr.ToLowerInvariant(),
        };
        if (kind == "vector") label = "text";
        string env = string.Join(", ", EnvRegex.Matches(desc).Select(m => m.Value).Distinct().Select(v => $"`{v}`"));
        return new CliParam(isArg, name, shortName, desc, def, label, required, isFlag, env);
    }

    private static string DefaultOf(Type settings, bool isArg, string name, int position)
    {
        PropertyInfo? prop = null;
        foreach (var pr in settings.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (isArg)
            {
                var a = pr.GetCustomAttribute<CommandArgumentAttribute>();
                if (a is not null && a.Position == position && string.Equals(a.ValueName, name, StringComparison.Ordinal)) { prop = pr; break; }
            }
            else
            {
                var o = pr.GetCustomAttribute<CommandOptionAttribute>();
                if (o is not null && o.LongNames.Contains(name, StringComparer.Ordinal)) { prop = pr; break; }
            }
        }
        if (prop is null || isArg) return "";

        object? value = prop.GetCustomAttribute<System.ComponentModel.DefaultValueAttribute>() is { } dv ? dv.Value : null;
        if (value is null)
        {
            try { value = prop.GetValue(Activator.CreateInstance(settings, nonPublic: true)); }
            catch { value = null; }
        }
        return value switch
        {
            null => "",
            bool b => b ? "on" : "off",
            string s => s.Length == 0 ? "" : $"`{s}`",
            float f => f.ToString("R", CultureInfo.InvariantCulture),
            double d => d.ToString("R", CultureInfo.InvariantCulture),
            Array { Length: 0 } => "",
            IFormattable fmt => fmt.ToString(null, CultureInfo.InvariantCulture),
            _ => value.ToString() ?? "",
        };
    }

    private static string Collapse(string? s) => Regex.Replace(s ?? "", @"\s+", " ", RegexOptions.None, TimeSpan.FromSeconds(5)).Trim();

    /// <summary>The generated option table for one command (the block between its <c>options:</c> markers in docs/CLI.md).</summary>
    public static string RenderTable(CliCommand cmd)
    {
        var sb = new StringBuilder();
        sb.Append("| Option | Default | Env | Description |\n|---|---|---|---|\n");
        foreach (var p in cmd.Params.OrderBy(x => x.IsArgument ? 0 : 1))
        {
            string def = p.Default.Length == 0 ? (p.IsFlag ? "off" : p.IsArgument ? "" : "unset") : p.Default;
            sb.Append("| ").Append(p.DisplayName).Append(" | ").Append(def).Append(" | ").Append(p.Env).Append(" | ")
              .Append(p.Description.Replace("|", "\\|", StringComparison.Ordinal)).Append(" |\n");
        }
        return sb.ToString().TrimEnd('\n');
    }
}
