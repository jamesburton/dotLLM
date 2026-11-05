using Microsoft.CodeAnalysis;

namespace DotLLM.ChunkReuse.Graph;

/// <summary>
/// One project's compilation for <see cref="RoslynGraphBuilder.Build(IReadOnlyList{ProjectCompilation}, IReadOnlyList{CodeCorpus.Chunk})"/>:
/// the files the project owns (corpus-relative path → its syntax tree in <paramref name="Compilation"/>). Symbols in other
/// projects resolve through the compilation's project references; their locations are matched to corpus paths by the trees'
/// file paths, so every tree a reference can point into must be owned by one of the projects passed together.
/// </summary>
/// <param name="Name">Project name (for diagnostics).</param>
/// <param name="Compilation">The project's compilation (its own parse options, defines and references).</param>
/// <param name="Trees">Owned files: corpus-relative path → syntax tree of <paramref name="Compilation"/>. A path is owned by one project only.</param>
public sealed record ProjectCompilation(string Name, Compilation Compilation, IReadOnlyDictionary<string, SyntaxTree> Trees)
{
    /// <summary>
    /// The BCL-only set-up as one "project": a single compilation over <paramref name="files"/> with the running runtime's
    /// BCL references and no defines (as <see cref="CodeCorpus.Corpus"/>), for comparisons with scanned projects.
    /// </summary>
    public static ProjectCompilation BclOnly(IEnumerable<CodeCorpus.SourceFile> files, bool implicitUsings = true)
    {
        ArgumentNullException.ThrowIfNull(files);
        (Microsoft.CodeAnalysis.CSharp.CSharpCompilation compilation, Dictionary<string, SyntaxTree> trees) =
            CodeCorpus.Corpus.CreateCompilation(files.OrderBy(f => f.Path, StringComparer.Ordinal), implicitUsings);
        return new ProjectCompilation("bcl-only", compilation, trees);
    }
}
