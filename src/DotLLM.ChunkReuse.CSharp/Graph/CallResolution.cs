using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DotLLM.ChunkReuse.Graph;

/// <summary>Invocation sites in the measured files of some compilations and how many of them bind.</summary>
/// <param name="Invocations">Invocation expressions (<c>M(...)</c>, <c>x.M(...)</c>, <c>x?.M(...)</c>, delegate calls) in the measured files.</param>
/// <param name="Resolved">Of those, the ones that bind to a single method symbol (source or metadata).</param>
/// <param name="ResolvedToFile">Of those, the ones whose method is declared in one of the measured files (a possible Call edge).</param>
public readonly record struct CallResolutionStats(int Invocations, int Resolved, int ResolvedToFile)
{
    /// <summary><see cref="Resolved"/> / <see cref="Invocations"/> (0 when there are none).</summary>
    public double ResolvedShare => Invocations == 0 ? 0 : (double)Resolved / Invocations;

    /// <summary><see cref="ResolvedToFile"/> / <see cref="Invocations"/> (0 when there are none).</summary>
    public double ResolvedToFileShare => Invocations == 0 ? 0 : (double)ResolvedToFile / Invocations;
}

/// <summary>What an invocation site binds to.</summary>
public enum CallTargetKind
{
    /// <summary>No single method symbol (unknown name, overload-resolution failure, ambiguity: candidates do not count).</summary>
    Unresolved,

    /// <summary>A method that is not declared in a measured file (metadata, or a source file outside the measured set).</summary>
    Elsewhere,

    /// <summary>A method declared in a measured file.</summary>
    File,
}

/// <summary>One invocation site and its binding.</summary>
/// <param name="Path">Corpus path of the file holding the invocation.</param>
/// <param name="Offset">Character offset of the invocation expression in that file.</param>
/// <param name="Length">Length of the invocation expression (nested invocations such as <c>a.B().C()</c> share a start).</param>
/// <param name="Kind">What it binds to.</param>
/// <param name="Method">The bound method (display string), or null when unresolved.</param>
/// <param name="DefinitionPath">For <see cref="CallTargetKind.File"/>: corpus path of the first declaring measured file.</param>
/// <param name="DefinitionOffset">For <see cref="CallTargetKind.File"/>: offset of that declaration's name; otherwise -1.</param>
public sealed record CallSite(string Path, int Offset, int Length, CallTargetKind Kind, string? Method, string? DefinitionPath, int DefinitionOffset);

/// <summary>
/// Measures how many invocation sites bind, to compare compilation set-ups over the same files (BCL-only versus one
/// compilation per project). Only <c>GetSymbolInfo(invocation).Symbol</c> counts, as in <see cref="RoslynGraphBuilder"/>:
/// candidate symbols (overload-resolution failure, ambiguity) are unresolved. A delegate invocation binds to the delegate's
/// <c>Invoke</c> (resolved, declared where the delegate type is). <c>nameof(...)</c> is not an invocation and is not counted.
/// </summary>
public static class CallResolution
{
    /// <summary>Counts the invocation sites of <paramref name="paths"/> (all owned files when null) in <paramref name="projects"/>.</summary>
    public static CallResolutionStats Measure(IReadOnlyList<ProjectCompilation> projects, IReadOnlySet<string>? paths = null)
    {
        int invocations = 0, resolved = 0, inFiles = 0;
        foreach (CallSite s in Sites(projects, paths))
        {
            invocations++;
            if (s.Kind != CallTargetKind.Unresolved) resolved++;
            if (s.Kind == CallTargetKind.File) inFiles++;
        }
        return new CallResolutionStats(invocations, resolved, inFiles);
    }

    /// <summary>Every invocation site of <paramref name="paths"/> (all owned files when null), by path, offset, then length.</summary>
    public static IReadOnlyList<CallSite> Sites(IReadOnlyList<ProjectCompilation> projects, IReadOnlySet<string>? paths = null)
    {
        ArgumentNullException.ThrowIfNull(projects);
        // Roslyn's binder recurses per nesting level: bind on the large-stack thread (the caller's stack may be ordinary), and
        // leave out a file nested deeper than the cap that thread makes safe.
        return CodeCorpus.RoslynParsing.RunWithLargeStack(() => SitesCore(projects, paths, CodeCorpus.RoslynParsing.DefaultFileBudget));
    }

    /// <summary><see cref="Sites"/> with a per-file binding budget (tests).</summary>
    internal static IReadOnlyList<CallSite> SitesWithin(IReadOnlyList<ProjectCompilation> projects, TimeSpan fileBudget) =>
        CodeCorpus.RoslynParsing.RunWithLargeStack(() => SitesCore(projects, null, fileBudget));

    private static IReadOnlyList<CallSite> SitesCore(IReadOnlyList<ProjectCompilation> projects, IReadOnlySet<string>? paths, TimeSpan fileBudget)
    {
        var corpusPath = new Dictionary<string, string>(RoslynGraphBuilder.PathComparer);
        foreach (ProjectCompilation p in projects)
            foreach ((string path, SyntaxTree tree) in p.Trees)
                if (paths is null || paths.Contains(path)) corpusPath.TryAdd(tree.FilePath, path);

        var sites = new List<CallSite>();
        foreach (ProjectCompilation p in projects)
        {
            foreach ((string path, SyntaxTree tree) in p.Trees)
            {
                if (paths is not null && !paths.Contains(path)) continue;
                int depth = CodeCorpus.RoslynParsing.SyntaxDepth(tree.GetRoot());
                if (depth > CodeCorpus.RoslynParsing.MaxSyntaxDepth)
                {
                    System.Diagnostics.Trace.TraceWarning($"{path}: syntax depth {depth} exceeds {CodeCorpus.RoslynParsing.MaxSyntaxDepth}; no call sites measured.");
                    continue;
                }
                SemanticModel model = p.Compilation.GetSemanticModel(tree);
                using var budget = new CancellationTokenSource(fileBudget);
                var treeSites = new List<CallSite>();
                try
                {
                foreach (InvocationExpressionSyntax inv in tree.GetRoot().DescendantNodes().OfType<InvocationExpressionSyntax>())
                {
                    budget.Token.ThrowIfCancellationRequested();
                    SymbolInfo info = model.GetSymbolInfo(inv, budget.Token);
                    if (inv.Expression is IdentifierNameSyntax { Identifier.ValueText: "nameof" } && info.Symbol is null) continue;
                    int at = inv.SpanStart, length = inv.Span.Length;
                    if (info.Symbol is not IMethodSymbol method)
                    {
                        treeSites.Add(new CallSite(path, at, length, CallTargetKind.Unresolved, null, null, -1));
                        continue;
                    }
                    IMethodSymbol m = (method.ReducedFrom ?? method).OriginalDefinition;
                    (string Path, int Offset)? def = m.Locations
                        .Where(l => l.IsInSource && l.SourceTree is not null && corpusPath.ContainsKey(l.SourceTree.FilePath))
                        .Select(l => (Path: corpusPath[l.SourceTree!.FilePath], Offset: l.SourceSpan.Start))
                        .OrderBy(d => d.Path, StringComparer.Ordinal).ThenBy(d => d.Offset)
                        .Select(d => ((string, int)?)d).FirstOrDefault();
                    string display = m.ToDisplayString();
                    treeSites.Add(def is { } d
                        ? new CallSite(path, at, length, CallTargetKind.File, display, d.Item1, d.Item2)
                        : new CallSite(path, at, length, CallTargetKind.Elsewhere, display, null, -1));
                }
                }
                catch (OperationCanceledException)
                {
                    System.Diagnostics.Trace.TraceWarning($"{path}: binding exceeded {fileBudget.TotalSeconds:F0} s; no call sites measured.");
                    continue;
                }
                sites.AddRange(treeSites);
            }
        }
        sites.Sort((a, b) => string.CompareOrdinal(a.Path, b.Path) is var c && c != 0 ? c : a.Offset != b.Offset ? a.Offset.CompareTo(b.Offset) : a.Length.CompareTo(b.Length));
        return sites;
    }
}
