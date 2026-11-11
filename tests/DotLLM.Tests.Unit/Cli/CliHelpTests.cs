using System.Text.RegularExpressions;
using Xunit;

namespace DotLLM.Tests.Unit.Cli;

/// <summary>
/// Issue #1720: <c>--help</c> and <c>docs/CLI.md</c> must stay complete. The command tree is read through Spectre's
/// <c>cli xmldoc</c> over the real <c>CliConfiguration</c>, so a command or option added tomorrow is covered without touching this file.
/// </summary>
public sealed class CliHelpTests
{
    private const string RegenVar = "DOTLLM_REGEN_CLI_DOCS";

    private static string RepoRoot()
    {
        string? dir = AppContext.BaseDirectory;
        while (dir is not null && !File.Exists(Path.Combine(dir, "dotLLM.slnx"))) dir = Path.GetDirectoryName(dir);
        return dir ?? throw new InvalidOperationException("repo root (dotLLM.slnx) not found above " + AppContext.BaseDirectory);
    }

    private static string DocPath() => Path.Combine(RepoRoot(), "docs", "CLI.md");

    /// <summary>The doc with LF line endings, so a CRLF checkout (autocrlf) compares equal to the generated tables.</summary>
    private static string ReadDoc() => File.ReadAllText(DocPath()).Replace("\r\n", "\n", StringComparison.Ordinal);

    /// <summary>
    /// The help-quality rules of issue #1721 over a command list: every command, option and argument has a description; every runnable
    /// (non-branch) command shows at least one example; an option with a declared default states it in its help text. Pure, so the
    /// guard itself can be run against deliberately broken trees.
    /// </summary>
    internal static List<string> HelpViolations(IEnumerable<CliCommand> commands)
    {
        var v = new List<string>();
        foreach (var c in commands)
        {
            if (string.IsNullOrWhiteSpace(c.Description)) v.Add($"command '{c.Path}' has no description");
            if (!c.IsBranch && c.ExampleCount == 0) v.Add($"command '{c.Path}' has no example");
            foreach (var p in c.Params)
            {
                string what = $"'{c.Path}' {(p.IsArgument ? "argument" : "option")} '{p.Name}'";
                if (string.IsNullOrWhiteSpace(p.Description)) { v.Add($"{what} has no description"); continue; }
                bool hasDefault = !p.IsArgument && !p.IsFlag && p.Default.Length > 0;
                if (hasDefault && !p.Description.Contains("default", StringComparison.OrdinalIgnoreCase)
                               && !p.Description.Contains(p.Default.Trim('`'), StringComparison.OrdinalIgnoreCase))
                    v.Add($"{what} has default {p.Default} but its description does not state it (add 'Default: {p.Default.Trim('`')}.')");
            }
        }
        return v;
    }

    [Fact]
    public void EveryCommandOptionAndArgument_HasADescription_AnExample_AndItsDefault()
    {
        var violations = HelpViolations(CliCommandModel.All);
        Assert.True(violations.Count == 0, "Help-quality violations: " + string.Join("; ", violations));
    }

    [Fact]
    public void HelpGuard_FailsOnAnUndescribedOption_AMissingExample_AndAnUnstatedDefault()
    {
        // Prove the guard can fail: take a real, passing command and break it three ways.
        var good = CliCommandModel.All.First(c => !c.IsBranch && !c.IsDebug && c.ExampleCount > 0 && c.Params.Any(p => !p.IsArgument && !p.IsFlag && p.Default.Length > 0));
        Assert.Empty(HelpViolations([good]));

        var opt = good.Params.First(p => !p.IsArgument && !p.IsFlag && p.Default.Length > 0);
        Assert.Contains(HelpViolations([good with { Params = [.. good.Params, opt with { Name = "undescribed", Description = "" }] }]), m => m.Contains("no description", StringComparison.Ordinal));
        Assert.Contains(HelpViolations([good with { ExampleCount = 0 }]), m => m.Contains("no example", StringComparison.Ordinal));
        Assert.Contains(HelpViolations([good with { Params = [opt with { Description = "Something unrelated." }] }]), m => m.Contains("does not state", StringComparison.Ordinal));
    }

    [Fact]
    public void EveryCommand_HelpRenders_WithoutMarkupErrors()
    {
        // Spectre feeds descriptions through its markup parser, so an unescaped "[:quant]" or "[...]" in an option text
        // breaks --help for that command ("Could not find color or style"). xmldoc prints the raw text and cannot see it.
        var failures = new List<string>();
        foreach (var path in new[] { "" }.Concat(CliCommandModel.All.Select(c => c.Path)))
        {
            string[] args = [.. path.Split(' ', StringSplitOptions.RemoveEmptyEntries), "--help"];
            var (rc, output) = CliCommandModel.Run(args);
            if (rc != 0 || output.Contains("Error:", StringComparison.Ordinal))
                failures.Add($"'dotllm {path} --help' (exit {rc}): {output.Split(Convert.ToChar(10)).FirstOrDefault(l => l.Contains("Error", StringComparison.Ordinal))}");
        }
        Assert.True(failures.Count == 0, string.Join("; ", failures));
    }

    [Fact]
    public void TopLevelList_ExistsAndSharesModelListOptions()
    {
        var top = CliCommandModel.All.Single(c => c.Path == "list");
        var nested = CliCommandModel.All.Single(c => c.Path == "model list");
        Assert.Equal(nested.Params.Select(p => p.Name), top.Params.Select(p => p.Name));
        Assert.Contains(top.Params, p => p.Name == "json");
    }

    [Fact]
    public void CliMd_OptionTables_MatchTheCommandTree()
    {
        string doc = ReadDoc();
        bool regen = Environment.GetEnvironmentVariable(RegenVar) == "1";
        var problems = new List<string>();

        foreach (var c in CliCommandModel.All.Where(c => !c.IsDebug && c.Params.Count > 0))
        {
            string begin = $"<!-- BEGIN options:{c.Path} -->";
            string end = $"<!-- END options:{c.Path} -->";
            int b = doc.IndexOf(begin, StringComparison.Ordinal);
            int e = doc.IndexOf(end, StringComparison.Ordinal);
            if (b < 0 || e < b) { problems.Add($"docs/CLI.md has no '{begin}' ... '{end}' block for 'dotllm {c.Path}'"); continue; }

            string expected = CliCommandModel.RenderTable(c);
            string current = doc[(b + begin.Length)..e].Trim('\r', '\n');
            if (current == expected) continue;
            if (regen) doc = doc[..(b + begin.Length)] + "\n" + expected + "\n" + doc[e..];
            else problems.Add($"option table for 'dotllm {c.Path}' is stale");
        }

        if (regen) File.WriteAllText(DocPath(), doc);
        else Assert.True(problems.Count == 0,
            string.Join("; ", problems) + $". Regenerate with {RegenVar}=1 (dotnet test --filter FullyQualifiedName~CliHelpTests), review the diff, commit.");
        if (regen) Assert.True(problems.Count == 0, string.Join("; ", problems));
    }

    [Fact]
    public void CliMd_MentionsEveryCommand_AndTheUserFacingEnvironmentVariables()
    {
        string doc = ReadDoc();
        string root = RepoRoot();

        foreach (var c in CliCommandModel.All.Where(c => !c.IsDebug && !c.IsBranch))
            Assert.True(doc.Contains($"dotllm {c.Path}", StringComparison.Ordinal), $"docs/CLI.md never mentions 'dotllm {c.Path}'");

        // Every DOTLLM_* name that the layers a user touches (CLI, server, model store, core options) read or name must be in the doc.
        // Kernel/backend tuning knobs live in the appendix, regenerated by scripts/gen-cli-env-appendix.py.
        var lit = new Regex("\"DOTLLM_[A-Z0-9_]*[A-Z0-9]\"", RegexOptions.Compiled | RegexOptions.ExplicitCapture, TimeSpan.FromSeconds(5));
        var undocumented = new SortedSet<string>(StringComparer.Ordinal);
        foreach (string proj in new[] { "DotLLM.Cli", "DotLLM.Server", "DotLLM.HuggingFace", "DotLLM.Core" })
        {
            string dir = Path.Combine(root, "src", proj);
            foreach (string f in Directory.EnumerateFiles(dir, "*.cs", SearchOption.AllDirectories))
            {
                if (f.Contains($"{Path.DirectorySeparatorChar}obj{Path.DirectorySeparatorChar}") || f.Contains($"{Path.DirectorySeparatorChar}bin{Path.DirectorySeparatorChar}")) continue;
                foreach (Match m in lit.Matches(File.ReadAllText(f)))
                {
                    string name = m.Value.Trim('"');
                    if (!Regex.IsMatch(doc, $@"\b{name}\b", RegexOptions.None, TimeSpan.FromSeconds(5))) undocumented.Add(name);
                }
            }
        }
        Assert.True(undocumented.Count == 0,
            "Environment variables read by the CLI/server/core layers but missing from docs/CLI.md: " + string.Join(", ", undocumented) +
            ". Add them to the user-facing tables, or run `python scripts/gen-cli-env-appendix.py --write`.");
    }
}
