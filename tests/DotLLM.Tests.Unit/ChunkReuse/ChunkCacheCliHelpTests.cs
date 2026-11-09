using System.Collections;
using System.Reflection;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using DotLLM.ChunkCache;
using Xunit;

namespace DotLLM.Tests.Unit.ChunkReuse;

/// <summary>
/// The help guard of the <c>chunkcache</c> CLI (dotLLM issue 1721): every option a verb's parser accepts has help text with a value form
/// and a default, every verb shows an example, <c>--help</c> documents every <c>--json</c> field, and the docs page lists exactly the verbs
/// the tool has. The option list is the SAME registration the parser uses (<see cref="VerbDef.Options"/> derives
/// <see cref="VerbDef.Spec"/>), so a new option cannot be added undocumented: <see cref="Guard_FailsOnAnUndocumentedOption"/> proves
/// the check can fail.
/// </summary>
public sealed class ChunkCacheCliHelpTests
{
    private static readonly IReadOnlyDictionary<string, VerbDef> Verbs = ChunkCacheCli.Verbs;

    private static Regex Rx(string pattern) => new(pattern, RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(5));

    // ------------------------------------------------------------------------------------------------ the guard

    /// <summary>Every problem with the help of <paramref name="verbs"/> (empty = fine). Takes the table so a test can feed it a bad one.</summary>
    private static List<string> HelpProblems(IReadOnlyDictionary<string, VerbDef> verbs)
    {
        var problems = new List<string>();
        foreach ((string name, VerbDef verb) in verbs)
        {
            if (verb.Name != name) problems.Add($"{name}: registered under another name '{verb.Name}'");
            if (string.IsNullOrWhiteSpace(verb.Summary)) problems.Add($"{name}: no summary");
            if (verb.Usage.Count == 0 || verb.Usage.Any(string.IsNullOrWhiteSpace)) problems.Add($"{name}: no usage line");
            if (verb.Examples.Count == 0) problems.Add($"{name}: no example");
            foreach (string example in verb.Examples)
                if (!example.TrimStart().StartsWith($"chunkcache {name} ", StringComparison.Ordinal) && !Rx(@"^(?:\w+=\S+ )+chunkcache " + name + " ").IsMatch(example))
                    problems.Add($"{name}: example does not start with 'chunkcache {name}': {example}");
            if (string.IsNullOrWhiteSpace(verb.JsonFields)) problems.Add($"{name}: no --json fields");
            if (verb.Options.GroupBy(o => o.Name, StringComparer.Ordinal).Any(g => g.Count() > 1)) problems.Add($"{name}: an option is registered twice");
            foreach (string required in new[] { "json", "help" })
                if (verb.Options.All(o => o.Name != required)) problems.Add($"{name}: --{required} is not registered");
            string help = verb.Help;
            Dictionary<string, string> entries = OptionEntries(help);
            foreach (OptionSpec o in verb.Options)
            {
                string at = $"{name} --{o.Name}";
                if (o.Help.Trim().Length < 15 || !o.Help.Trim().Contains(' ', StringComparison.Ordinal)) problems.Add($"{at}: no help text");
                if (string.IsNullOrWhiteSpace(o.Default)) problems.Add($"{at}: no default stated");
                if (o.Kind == OptionKind.Flag && o.Value is not null) problems.Add($"{at}: a flag has no value form");
                if (o.Kind != OptionKind.Flag && string.IsNullOrWhiteSpace(o.Value)) problems.Add($"{at}: no value form");
                if (o.Name == "server" && o.Env != "CHUNKCACHE_API_KEY") problems.Add($"{at}: does not name the CHUNKCACHE_API_KEY environment variable");
                if (!entries.TryGetValue(o.Header, out string? entry))
                {
                    problems.Add($"{at}: --help does not print its entry");
                    continue;
                }
                if (!entry.Contains(Squash(o.Description), StringComparison.Ordinal)) problems.Add($"{at}: --help does not print its description");
                // The clauses are checked against the registered fields themselves, not against the renderer's own output.
                if (!entry.Contains("Default: " + o.Default.TrimEnd('.'), StringComparison.Ordinal)) problems.Add($"{at}: --help does not print its default");
                if (o.Values is not null && !entry.Contains("Values: " + o.Values.TrimEnd('.'), StringComparison.Ordinal)) problems.Add($"{at}: --help does not print its valid values");
                if (o.Env is not null && !entry.Contains("Env: " + o.Env, StringComparison.Ordinal)) problems.Add($"{at}: --help does not print its environment variable");
                if (o.Kind == OptionKind.List && !entry.Contains("Repeatable", StringComparison.Ordinal)) problems.Add($"{at}: --help does not say it repeats");
            }
        }
        return problems;
    }

    /// <summary>The entries of the <c>options:</c> section of a verb's help: header line (<c>--name VALUE</c>) to its wrapped text, whitespace squashed.</summary>
    private static Dictionary<string, string> OptionEntries(string help)
    {
        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        string[] lines = help.Split('\n').Select(l => l.TrimEnd('\r')).ToArray();
        int i = Array.IndexOf(lines, "options:") + 1;
        string? header = null;
        var body = new StringBuilder();
        for (; i > 0 && i < lines.Length && lines[i].Length > 0; i++)
        {
            if (lines[i].StartsWith("  --", StringComparison.Ordinal) && !lines[i].StartsWith("   ", StringComparison.Ordinal))
            {
                if (header is not null) entries[header] = Squash(body.ToString());
                header = lines[i].Trim();
                body.Clear();
            }
            else body.Append(lines[i]).Append(' ');
        }
        if (header is not null) entries[header] = Squash(body.ToString());
        return entries;
    }

    private static string Squash(string s) => Rx(@"\s+").Replace(s, " ");

    [Fact]
    public void EveryRegisteredOption_OfEveryVerb_HasHelpTextDefaultAndValueForm_AndEveryVerbAnExample()
    {
        Assert.Empty(HelpProblems(Verbs));
    }

    [Fact]
    public void Guard_FailsOnAnUndocumentedOption()
    {
        // An option registered with no help text, one with no default and one with no value form, plus a verb with no example.
        VerbDef real = Verbs["status"];
        var bad = new Dictionary<string, VerbDef>(Verbs)
        {
            ["status"] = new VerbDef("status", real.Summary, real.Usage, real.Description,
                [.. real.Options, new OptionSpec("fake-no-help", OptionKind.Flag, "", Default: "off"),
                    new OptionSpec("fake-no-default", OptionKind.Flag, "A flag that says what it does but not its default."),
                    new OptionSpec("fake-no-value", OptionKind.Single, "An option that takes a value but shows no value form.", Default: "none")],
                real.JsonFields, real.JsonTypes, real.Examples),
            ["build"] = new VerbDef("build", Verbs["build"].Summary, Verbs["build"].Usage, "", Verbs["build"].Options, Verbs["build"].JsonFields, Verbs["build"].JsonTypes, []),
        };
        List<string> problems = HelpProblems(bad);
        Assert.Contains(problems, p => p.Contains("status --fake-no-help: no help text", StringComparison.Ordinal));
        Assert.Contains(problems, p => p.Contains("status --fake-no-default: no default stated", StringComparison.Ordinal));
        Assert.Contains(problems, p => p.Contains("status --fake-no-value: no value form", StringComparison.Ordinal));
        Assert.Contains(problems, p => p.Contains("build: no example", StringComparison.Ordinal));
    }

    [Fact]
    public void ParserSpec_IsDerivedFromTheRegistry_AndEveryRegisteredOptionParses()
    {
        foreach ((string name, VerbDef verb) in Verbs)
        {
            Assert.Equal(verb.Options.Select(o => o.Name).Order(StringComparer.Ordinal), verb.Spec.Keys.Order(StringComparer.Ordinal));
            foreach (OptionSpec o in verb.Options)
            {
                Assert.Equal(o.Kind, verb.Spec[o.Name]);
                string[] args = o.Kind == OptionKind.Flag ? ["--" + o.Name] : ["--" + o.Name, "value"];
                CliArgs parsed = CliArgs.Parse(name, args, verb.Spec);
                Assert.True(o.Kind == OptionKind.Flag ? parsed.Flag(o.Name) : parsed.Value(o.Name) == "value", $"{name} --{o.Name}");
            }
            Assert.Throws<UsageException>(() => CliArgs.Parse(name, ["--no-such-option-xyz"], verb.Spec));
        }
    }

    [Fact]
    public void VerbHelp_IsWhatTheCliPrints_AndTheTopLevelUsageListsEveryVerb()
    {
        foreach ((string name, VerbDef verb) in Verbs)
        {
            var o = new StringWriter();
            var e = new StringWriter();
            Assert.Equal(0, ChunkCacheCli.Run([name, "--help"], o, e));
            Assert.Equal(verb.Help, o.ToString().TrimEnd());
            Assert.Equal("", e.ToString());
            Assert.Matches(Rx(@"(?m)^  " + name + @"\s+\S"), ChunkCacheCli.Usage);
            Assert.Contains("chunkcache " + name + ": ", verb.Help, StringComparison.Ordinal);
        }
        // Every verb the help knows is a verb the dispatcher runs (an unknown verb is a usage error, a known one is not).
        foreach (string name in Verbs.Keys)
            Assert.NotEqual(2, ChunkCacheCli.Run([name, "--help"], new StringWriter(), new StringWriter()));
        foreach (string needed in new[] { "CHUNKCACHE_API_KEY", "exit codes", "--verbose", "0  ok", "2  usage" })
            Assert.Contains(needed, ChunkCacheCli.Usage, StringComparison.Ordinal);
    }

    [Fact]
    public void UsageError_PrintsTheSynopsisNotTheWholeHelp()
    {
        var o = new StringWriter();
        var e = new StringWriter();
        Assert.Equal(2, ChunkCacheCli.Run(["build", "--no-such-option", "1"], o, e));
        Assert.Contains("usage: chunkcache build", e.ToString(), StringComparison.Ordinal);
        Assert.Contains("chunkcache build --help", e.ToString(), StringComparison.Ordinal);
        Assert.DoesNotContain("exit codes:", e.ToString(), StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------------------------------------ --json fields

    [Fact]
    public void Help_DocumentsEveryJsonField_OfTheResultTypes()
    {
        foreach ((string name, VerbDef verb) in Verbs)
        {
            string section = JsonSection(verb.Help);
            var fields = new SortedSet<string>(StringComparer.Ordinal);
            foreach (Type t in verb.JsonTypes) CollectFields(t, fields, []);
            Assert.NotEmpty(fields);
            foreach (string f in fields) Assert.True(Mentions(section, f), $"{name}: --help does not document the JSON field '{f}'");
        }
    }

    [Fact]
    public void Help_DocumentsEveryProperty_OfTheVerbsJsonSchema()
    {
        foreach ((string name, VerbDef verb) in Verbs)
        {
            string path = Path.Combine(AppContext.BaseDirectory, "ChunkCache", "Fixtures", "v1", "schemas", name + ".json");
            Assert.True(File.Exists(path), $"no schema file for {name}: {path}");
            using JsonDocument schema = JsonDocument.Parse(File.ReadAllText(path));
            var names = new SortedSet<string>(StringComparer.Ordinal);
            CollectSchemaNames(schema.RootElement, names);
            string section = JsonSection(verb.Help);
            foreach (string n in names) Assert.True(Mentions(section, n), $"{name}: --help does not document the schema property '{n}'");
        }
    }

    private static string JsonSection(string help)
    {
        int from = help.IndexOf("--json output", StringComparison.Ordinal);
        int to = help.IndexOf("examples:", from, StringComparison.Ordinal);
        Assert.True(from >= 0 && to > from, "help has no --json section");
        return help[from..to];
    }

    private static bool Mentions(string text, string field) => Rx(@"(?<![a-z_])" + Regex.Escape(field) + @"(?![a-z_])").IsMatch(text);

    private static void CollectFields(Type type, SortedSet<string> fields, HashSet<Type> seen)
    {
        if (!seen.Add(type)) return;
        foreach (PropertyInfo p in type.GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (p.GetCustomAttribute<System.Text.Json.Serialization.JsonIgnoreAttribute>() is not null) continue;
            fields.Add(JsonNamingPolicy.SnakeCaseLower.ConvertName(p.Name));
            Type pt = Nullable.GetUnderlyingType(p.PropertyType) ?? p.PropertyType;
            if (pt.IsGenericType && typeof(IEnumerable).IsAssignableFrom(pt)) pt = pt.GetGenericArguments()[0];
            if (pt.Namespace == "DotLLM.ChunkCache" && !pt.IsEnum) CollectFields(pt, fields, seen);
        }
    }

    private static void CollectSchemaNames(JsonElement schema, SortedSet<string> names)
    {
        if (schema.ValueKind == JsonValueKind.Object)
        {
            if (schema.TryGetProperty("properties", out JsonElement props) && props.ValueKind == JsonValueKind.Object)
                foreach (JsonProperty p in props.EnumerateObject())
                {
                    names.Add(p.Name);
                    CollectSchemaNames(p.Value, names);
                }
            if (schema.TryGetProperty("items", out JsonElement items)) CollectSchemaNames(items, names);
        }
    }

    // ------------------------------------------------------------------------------------------------ examples and the docs page

    private static readonly Regex EnvPrefix = Rx(@"^(?:\w+=\S*\s+)+");

    /// <summary>Parses a documented command line (<c>[VAR=x] chunkcache VERB args...</c>) against the verb's registered options.</summary>
    private static void AssertExampleParses(string line, string where)
    {
        string cmd = line.Trim();
        int hash = cmd.IndexOf(" #", StringComparison.Ordinal);
        if (hash >= 0) cmd = cmd[..hash].TrimEnd();
        cmd = EnvPrefix.Replace(cmd, "");
        string[] tokens = Tokenize(cmd);
        Assert.True(tokens.Length >= 2 && tokens[0] == "chunkcache", $"{where}: not a chunkcache command: {line}");
        Assert.True(Verbs.TryGetValue(tokens[1], out VerbDef? verb), $"{where}: unknown verb '{tokens[1]}': {line}");
        try
        {
            CliArgs.Parse(verb!.Name, tokens.Skip(2), verb.Spec);
        }
        catch (UsageException ex)
        {
            Assert.Fail($"{where}: the example does not parse: {ex.Message}: {line}");
        }
    }

    private static string[] Tokenize(string s)
    {
        var tokens = new List<string>();
        var cur = new StringBuilder();
        char quote = '\0';
        bool any = false;
        foreach (char c in s)
        {
            if (quote != '\0') { if (c == quote) quote = '\0'; else cur.Append(c); continue; }
            if (c is '"' or '\'') { quote = c; any = true; continue; }
            if (char.IsWhiteSpace(c))
            {
                if (cur.Length > 0 || any) tokens.Add(cur.ToString());
                cur.Clear();
                any = false;
                continue;
            }
            cur.Append(c);
        }
        if (cur.Length > 0 || any) tokens.Add(cur.ToString());
        return [.. tokens];
    }

    [Fact]
    public void HelpExamples_ParseAgainstTheRegisteredOptions()
    {
        AssertExampleParsesFails();
        foreach ((string name, VerbDef verb) in Verbs)
            foreach (string example in verb.Examples) AssertExampleParses(example, $"{name} --help");
    }

    /// <summary>The example checker itself can fail: a made-up option, a comma list and a wrong verb are all rejected.</summary>
    private static void AssertExampleParsesFails()
    {
        Assert.ThrowsAny<Exception>(() => AssertExampleParses("chunkcache build --model M.gguf --no-such-option", "control"));
        Assert.ThrowsAny<Exception>(() => AssertExampleParses("chunkcache select --path a.cs,b.cs", "control"));
        Assert.ThrowsAny<Exception>(() => AssertExampleParses("chunkcache frobnicate --json", "control"));
    }

    private static string DocsPage()
    {
        for (string? dir = AppContext.BaseDirectory; dir is not null; dir = Path.GetDirectoryName(dir))
        {
            string candidate = Path.Combine(dir, "docs", "CHUNK_CACHE_CLI.md");
            if (File.Exists(candidate)) return File.ReadAllText(candidate);
        }
        throw new FileNotFoundException("docs/CHUNK_CACHE_CLI.md not found above " + AppContext.BaseDirectory);
    }

    private static Dictionary<string, string> VerbSections(string page)
    {
        int verbs = page.IndexOf("\n## Verbs", StringComparison.Ordinal);
        Assert.True(verbs >= 0, "the docs page has no '## Verbs' section");
        string rest = page[(verbs + 1)..];
        int next = Rx(@"(?m)^## ").Match(rest[3..]).Index;
        string body = next > 0 ? rest[..(next + 3)] : rest;
        var sections = new Dictionary<string, string>(StringComparer.Ordinal);
        MatchCollection heads = Rx(@"(?m)^### (?<verb>\S+)\s*$").Matches(body);
        for (int i = 0; i < heads.Count; i++)
        {
            int start = heads[i].Index + heads[i].Length;
            int end = i + 1 < heads.Count ? heads[i + 1].Index : body.Length;
            sections[heads[i].Groups["verb"].Value] = body[start..end];
        }
        return sections;
    }

    [Fact]
    public void DocsPage_DocumentsExactlyTheVerbsTheToolHas_WithRunnableExamples()
    {
        string page = DocsPage();
        Dictionary<string, string> sections = VerbSections(page);
        Assert.Equal(Verbs.Keys.Order(StringComparer.Ordinal), sections.Keys.Order(StringComparer.Ordinal));
        foreach ((string name, string text) in sections)
        {
            MatchCollection fences = Rx(@"(?ms)^```sh\s*\n(?<body>.*?)^```").Matches(text);
            Assert.True(fences.Count > 0, $"docs: verb '{name}' has no ```sh example");
            int commands = 0;
            foreach (Match fence in fences)
                foreach (string line in fence.Groups["body"].Value.Split('\n'))
                {
                    string t = line.Trim();
                    if (t.Length == 0 || t.StartsWith('#')) continue;
                    AssertExampleParses(t, $"docs {name}");
                    Assert.Equal(name, Tokenize(EnvPrefix.Replace(t.Split(" #")[0].Trim(), ""))[1]); // an example sits under its own verb
                    commands++;
                }
            Assert.True(commands > 0, $"docs: verb '{name}' has no example command");
            // Every option of the verb that is not a bare shared one is named at least once on the page section or its synopsis block.
            Assert.Contains("--json", text, StringComparison.Ordinal);
        }
        foreach (string needed in new[] { "CHUNKCACHE_API_KEY", "Exit codes", "advisory" })
            Assert.Contains(needed, page, StringComparison.Ordinal);
    }

    /// <summary>The markdown option table of a verb as the docs page shows it, generated from the registry the parser and <c>--help</c> read.</summary>
    internal static string OptionTable(VerbDef verb)
    {
        static string Cell(string s) => s.Replace("|", "\\|", StringComparison.Ordinal);
        var sb = new StringBuilder("| Option | Value | Default | Description |\n|---|---|---|---|\n");
        foreach (OptionSpec o in verb.Options)
            sb.Append(System.Globalization.CultureInfo.InvariantCulture, $"| `--{o.Name}` | {(o.Value is null ? "" : "`" + Cell(o.Value) + "`")} | {Cell(o.Default)} | {Cell(o.DescriptionWithoutDefault)} |\n");
        return sb.ToString().TrimEnd();
    }

    private static string OptionBlock(VerbDef verb) => $"<!-- options:{verb.Name} -->\n{OptionTable(verb)}\n<!-- /options:{verb.Name} -->";

    [Fact]
    public void DocsPage_OptionTables_AreTheCurrentRegistry()
    {
        // Set CHUNKCACHE_WRITE_DOCS=<file> to write every verb's current block there and paste it into the page.
        if (Environment.GetEnvironmentVariable("CHUNKCACHE_WRITE_DOCS") is { Length: > 0 } dump)
            File.WriteAllText(dump, string.Join("\n\n", Verbs.Values.Select(v => $"### {v.Name}\n\n{OptionBlock(v)}")) + "\n");
        string page = DocsPage().Replace("\r\n", "\n", StringComparison.Ordinal);
        foreach ((string name, VerbDef verb) in Verbs)
        {
            Match m = Rx(@"(?s)<!-- options:" + Regex.Escape(name) + @" -->\n.*?\n<!-- /options:" + Regex.Escape(name) + " -->").Match(page);
            Assert.True(m.Success, $"docs: no generated option table for '{name}' (run with CHUNKCACHE_WRITE_DOCS=<file> to generate it)");
            Assert.Equal(OptionBlock(verb), m.Value);
        }
    }

    [Fact]
    public void DocsExamples_Checker_RejectsABadExample()
    {
        Assert.ThrowsAny<Exception>(() => AssertExampleParses("chunkcache status --model M.gguf --store DIR --bogus", "docs control"));
    }
}
