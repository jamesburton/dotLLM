using DotLLM.ChunkReuse.CodeCorpus;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;

namespace DotLLM.ChunkReuse.Graph;

/// <summary>Invocation sites in the owned files of some compilations and how many of them bind.</summary>
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

/// <summary>
/// Measures how many invocation sites bind, to compare compilation set-ups over the same files (BCL-only versus per-project
/// compilations from <see cref="SolutionScanner"/>). <c>nameof(...)</c> is not an invocation and is not counted.
/// </summary>
public static class CallResolution
{
    /// <summary>Counts the invocation sites of <paramref name="paths"/> (all owned files when null) in <paramref name="projects"/>.</summary>
    public static CallResolutionStats Measure(IReadOnlyList<ProjectCompilation> projects, IReadOnlySet<string>? paths = null)
    {
        ArgumentNullException.ThrowIfNull(projects);
        var treePaths = new HashSet<string>(StringComparer.Ordinal);
        foreach (ProjectCompilation p in projects)
            foreach ((string path, SyntaxTree tree) in p.Trees)
                if (paths is null || paths.Contains(path)) treePaths.Add(tree.FilePath);

        int invocations = 0, resolved = 0, inFiles = 0;
        foreach (ProjectCompilation p in projects)
        {
            foreach ((string path, SyntaxTree tree) in p.Trees.OrderBy(kv => kv.Key, StringComparer.Ordinal))
            {
                if (paths is not null && !paths.Contains(path)) continue;
                SemanticModel model = p.Compilation.GetSemanticModel(tree);
                foreach (InvocationExpressionSyntax inv in tree.GetRoot().DescendantNodes().OfType<InvocationExpressionSyntax>())
                {
                    if (inv.Expression is IdentifierNameSyntax { Identifier.ValueText: "nameof" } && model.GetSymbolInfo(inv).Symbol is null)
                        continue;
                    invocations++;
                    if (model.GetSymbolInfo(inv).Symbol is not IMethodSymbol method) continue;
                    resolved++;
                    IMethodSymbol m = (method.ReducedFrom ?? method).OriginalDefinition;
                    if (m.Locations.Any(l => l.IsInSource && l.SourceTree is { } t && treePaths.Contains(t.FilePath))) inFiles++;
                }
            }
        }
        return new CallResolutionStats(invocations, resolved, inFiles);
    }
}
