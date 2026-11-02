using DotLLM.ChunkReuse.Graph;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.MSBuild;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.SolutionPersistence.Model;
using Microsoft.VisualStudio.SolutionPersistence.Serializer;

namespace DotLLM.ChunkReuse.CodeCorpus;

/// <summary>A project the scanner did not scan, and why (<see cref="Reason"/> is one of the constants below).</summary>
/// <param name="ProjectPath">Project file path relative to <see cref="ScanResult.RootDirectory"/>, '/' separated.</param>
/// <param name="Reason">Category: <see cref="LoadFailed"/>, <see cref="UnsupportedLanguage"/> or <see cref="ExtraTargetFramework"/>.</param>
/// <param name="Message">The loader's message(s), or the scanner's explanation.</param>
public sealed record SkippedProject(string ProjectPath, string Reason, string Message)
{
    /// <summary>MSBuild could not evaluate or design-time build the project (missing import or SDK, bad XML, missing file).</summary>
    public const string LoadFailed = "load-failed";

    /// <summary>The project is not a C# project (Visual Basic, F#, or any other project type).</summary>
    public const string UnsupportedLanguage = "unsupported-language";

    /// <summary>
    /// A further target framework of a multi-targeted project: its files are scanned once, from the target framework whose
    /// Roslyn project name sorts first (named in the message).
    /// </summary>
    public const string ExtraTargetFramework = "extra-target-framework";
}

/// <summary>A loaded C# project.</summary>
/// <param name="Name">Roslyn project name (with the target framework in parentheses for multi-targeted projects).</param>
/// <param name="ProjectPath">Project file path relative to <see cref="ScanResult.RootDirectory"/>, '/' separated.</param>
/// <param name="Files">Files this project owns (a file compiled by several projects is owned by the first by project path), ordinal order.</param>
/// <param name="Diagnostics">Loader warnings and failures that name this project (e.g. unresolved references).</param>
/// <param name="Compilation">The project's compilation and owned trees, for <see cref="RoslynGraphBuilder"/>.</param>
public sealed record ScannedProject(string Name, string ProjectPath, IReadOnlyList<string> Files, IReadOnlyList<string> Diagnostics,
    ProjectCompilation Compilation);

/// <summary>
/// The C# files of a solution or project (<see cref="Files"/>: every C# document of the scanned projects, generated ones
/// included and flagged), the projects that were not scanned and why (<see cref="Skipped"/>), and one compilation per scanned
/// project.
/// </summary>
/// <param name="Files">Files in ordinal path order; paths relative to <see cref="RootDirectory"/> ('/' separated, may start with
/// <c>../</c> for files outside it); text with CRLF → LF, as <see cref="Corpus.FromDirectory"/>.</param>
/// <param name="Skipped">Projects not scanned, by project path.</param>
public sealed record ScanResult(IReadOnlyList<SourceFile> Files, IReadOnlyList<SkippedProject> Skipped)
{
    /// <summary>Directory of the solution or project file; all paths are relative to it.</summary>
    public string RootDirectory { get; init; } = "";

    /// <summary>Scanned projects, ordinal by <see cref="ScannedProject.ProjectPath"/>.</summary>
    public IReadOnlyList<ScannedProject> Projects { get; init; } = [];

    /// <summary>
    /// Paths of generated files: under a <c>bin</c> or <c>obj</c> directory (SDK implicit usings, assembly info), or
    /// <see cref="SymbolChunker.IsGenerated"/> (file-name suffix, <c>&lt;auto-generated</c> header, <c>[GeneratedCode]</c>).
    /// Source-generator output is not a document and is not in <see cref="Files"/>; it is in the compilations.
    /// </summary>
    public IReadOnlySet<string> Generated { get; init; } = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>Loader diagnostics that name no project of the solution.</summary>
    public IReadOnlyList<string> Diagnostics { get; init; } = [];

    /// <summary>True when <paramref name="path"/> is a generated file (see <see cref="Generated"/>).</summary>
    public bool IsGenerated(string path) => Generated.Contains(path);

    /// <summary>The files that are not generated, in order (what a corpus normally chunks).</summary>
    public IReadOnlyList<SourceFile> AuthoredFiles => Files.Where(f => !IsGenerated(f.Path)).ToArray();

    /// <summary>One compilation per scanned project, for <see cref="RoslynGraphBuilder.Build(IReadOnlyList{ProjectCompilation}, IReadOnlyList{Chunk})"/>.</summary>
    public IReadOnlyList<ProjectCompilation> Compilations => Projects.Select(p => p.Compilation).ToArray();
}

/// <summary>
/// Loads a solution (<c>.sln</c>, <c>.slnx</c>) or a project with Roslyn's <see cref="MSBuildWorkspace"/> (design-time builds
/// in Roslyn's out-of-process build host, which locates the .NET SDK itself — no in-process MSBuild registration) and returns
/// its C# files and per-project compilations. Projects that do not load, are not C#, or repeat a target framework are listed
/// in <see cref="ScanResult.Skipped"/> with the reason; a load failure never aborts the scan silently.
/// </summary>
/// <remarks>
/// Projects should be restored first (as after a build); an unrestored project loads without its package and framework
/// references, and the workspace's warning is recorded in <see cref="ScannedProject.Diagnostics"/>. Document texts are
/// replaced by their CRLF → LF form before the compilations are made, so syntax-tree offsets equal <see cref="SourceFile"/>
/// offsets (and chunk offsets).
/// </remarks>
public static class SolutionScanner
{
    /// <summary>Loads <paramref name="slnOrProjectPath"/> (a <c>.sln</c>/<c>.slnx</c>, or a project file with its project references).</summary>
    public static async Task<ScanResult> LoadAsync(string slnOrProjectPath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(slnOrProjectPath);
        string full = Path.GetFullPath(slnOrProjectPath);
        if (!File.Exists(full)) throw new FileNotFoundException("Solution or project file not found.", full);
        string root = Path.GetDirectoryName(full)!;
        string ext = Path.GetExtension(full);
        bool isSolution = ext.Equals(".sln", StringComparison.OrdinalIgnoreCase) || ext.Equals(".slnx", StringComparison.OrdinalIgnoreCase);

        using MSBuildWorkspace workspace = MSBuildWorkspace.Create();
        workspace.SkipUnrecognizedProjects = true;
        workspace.LoadMetadataForReferencedProjects = false;

        // Projects the solution lists (or the one project): any of them that does not end up scanned is reported.
        List<string> expected;
        Solution solution;
        var skipped = new List<SkippedProject>();
        string? rootFailure = null;
        if (isSolution)
        {
            expected = await ListSolutionProjectsAsync(full, cancellationToken).ConfigureAwait(false);
            solution = await workspace.OpenSolutionAsync(full, progress: null, cancellationToken).ConfigureAwait(false);
        }
        else
        {
            expected = [full];
            try
            {
                Project project = await workspace.OpenProjectAsync(full, progress: null, cancellationToken).ConfigureAwait(false);
                solution = project.Solution;
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                rootFailure = ex.Message;
                solution = workspace.CurrentSolution;
            }
        }

        string Rel(string path) => Path.GetRelativePath(root, path).Replace('\\', '/');
        List<WorkspaceDiagnostic> diagnostics = workspace.Diagnostics.ToList();
        var usedDiagnostics = new HashSet<WorkspaceDiagnostic>();
        List<WorkspaceDiagnostic> DiagnosticsOf(string projectPath)
        {
            var found = diagnostics.Where(d => d.Message.Contains(projectPath, StringComparison.OrdinalIgnoreCase)).ToList();
            usedDiagnostics.UnionWith(found);
            return found;
        }
        List<string> DiagnosticsFor(string projectPath) => DiagnosticsOf(projectPath).Select(Format).ToList();

        // Roslyn projects by project file; a multi-targeted project gives one per target framework.
        var byFile = solution.Projects.Where(p => p.FilePath is not null)
            .GroupBy(p => Path.GetFullPath(p.FilePath!), StringComparer.OrdinalIgnoreCase)
            .ToDictionary(g => g.Key, g => g.OrderBy(p => p.Name, StringComparer.Ordinal).ToList(), StringComparer.OrdinalIgnoreCase);

        foreach (string path in expected.Where(p => !byFile.ContainsKey(p)))
        {
            List<string> messages = DiagnosticsFor(path);
            if (rootFailure is not null && path.Equals(full, StringComparison.OrdinalIgnoreCase)) messages.Insert(0, rootFailure);
            bool csharp = Path.GetExtension(path).Equals(".csproj", StringComparison.OrdinalIgnoreCase);
            string message = messages.Count > 0 ? string.Join("\n", messages) : csharp ? "The workspace did not load the project." : "Not a C# project.";
            skipped.Add(new SkippedProject(Rel(path), csharp ? SkippedProject.LoadFailed : SkippedProject.UnsupportedLanguage, message));
        }

        // Scanned projects: C#, loaded without a failure, first target framework by name.
        var primary = new List<(string Path, Project Project, List<string> Diagnostics)>();
        foreach ((string path, List<Project> projects) in byFile.OrderBy(kv => Rel(kv.Key), StringComparer.Ordinal))
        {
            Project first = projects[0];
            List<WorkspaceDiagnostic> own = DiagnosticsOf(path);
            if (first.Language != LanguageNames.CSharp)
            {
                string message = string.Join("\n", own.Select(Format).Prepend($"Language '{first.Language}'."));
                skipped.Add(new SkippedProject(Rel(path), SkippedProject.UnsupportedLanguage, message));
                continue;
            }
            // The workspace keeps a project whose evaluation or design-time build failed (e.g. a missing SDK), with whatever
            // items it got: not a reliable file list or compilation.
            if (own.Any(d => d.Kind == WorkspaceDiagnosticKind.Failure))
            {
                skipped.Add(new SkippedProject(Rel(path), SkippedProject.LoadFailed,
                    string.Join("\n", own.Where(d => d.Kind == WorkspaceDiagnosticKind.Failure).Select(Format))));
                continue;
            }
            primary.Add((path, first, own.Select(Format).ToList()));
            foreach (Project extra in projects.Skip(1))
                skipped.Add(new SkippedProject(Rel(path), SkippedProject.ExtraTargetFramework,
                    $"Project '{extra.Name}': files scanned from '{first.Name}'."));
        }

        // LF texts (every document, so a file compiled by several projects reads the same everywhere).
        var texts = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        foreach (Project project in solution.Projects.Where(p => p.Language == LanguageNames.CSharp))
        {
            foreach (Document doc in project.Documents.Where(d => d.FilePath is not null))
            {
                cancellationToken.ThrowIfCancellationRequested();
                if (!texts.TryGetValue(doc.FilePath!, out string? lf))
                {
                    SourceText text = await doc.GetTextAsync(cancellationToken).ConfigureAwait(false);
                    lf = text.ToString().Replace("\r\n", "\n", StringComparison.Ordinal);
                    texts[doc.FilePath!] = lf;
                }
                solution = solution.WithDocumentText(doc.Id, SourceText.From(lf, System.Text.Encoding.UTF8));
            }
        }

        var files = new List<SourceFile>();
        var owned = new HashSet<string>(StringComparer.Ordinal);
        var generated = new HashSet<string>(StringComparer.Ordinal);
        var scanned = new List<ScannedProject>();
        foreach ((string path, Project p, List<string> projectDiagnostics) in primary)
        {
            Project project = solution.GetProject(p.Id)!;
            Compilation compilation = (await project.GetCompilationAsync(cancellationToken).ConfigureAwait(false))!;
            // A project that loads without its references (not restored, or a design-time build that silently produced none)
            // binds almost nothing; say so rather than give a quietly poor graph.
            if (compilation.GetSpecialType(SpecialType.System_Object).TypeKind == TypeKind.Error)
                projectDiagnostics.Add("Warning: no core library reference (System.Object does not resolve): project not restored, or its references did not resolve.");
            var trees = new Dictionary<string, SyntaxTree>(StringComparer.Ordinal);
            foreach (Document doc in project.Documents.Where(d => d.FilePath is not null).OrderBy(d => Rel(d.FilePath!), StringComparer.Ordinal))
            {
                string rel = Rel(doc.FilePath!);
                if (!owned.Add(rel)) continue;
                var file = new SourceFile(rel, texts[doc.FilePath!]);
                files.Add(file);
                if (IsGeneratedPath(rel) || SymbolChunker.IsGenerated(file)) generated.Add(rel);
                trees[rel] = (await doc.GetSyntaxTreeAsync(cancellationToken).ConfigureAwait(false))!;
            }
            scanned.Add(new ScannedProject(project.Name, Rel(path), trees.Keys.Order(StringComparer.Ordinal).ToArray(), projectDiagnostics,
                new ProjectCompilation(project.Name, compilation, trees)));
        }

        return new ScanResult(files.OrderBy(f => f.Path, StringComparer.Ordinal).ToArray(),
            skipped.OrderBy(s => s.ProjectPath, StringComparer.Ordinal).ThenBy(s => s.Reason, StringComparer.Ordinal).ToArray())
        {
            RootDirectory = root,
            Projects = scanned,
            Generated = generated,
            Diagnostics = diagnostics.Where(d => !usedDiagnostics.Contains(d)).Select(Format).ToArray(),
        };
    }

    private static string Format(WorkspaceDiagnostic d) => $"{d.Kind}: {d.Message}";

    /// <summary>A path with a <c>bin</c> or <c>obj</c> directory segment (build output, generated by the SDK).</summary>
    private static bool IsGeneratedPath(string relativePath) =>
        relativePath.Split('/')[..^1].Any(d => d.Equals("obj", StringComparison.OrdinalIgnoreCase) || d.Equals("bin", StringComparison.OrdinalIgnoreCase));

    /// <summary>Full paths of the projects a solution file lists (solution folders excluded).</summary>
    private static async Task<List<string>> ListSolutionProjectsAsync(string solutionPath, CancellationToken cancellationToken)
    {
        var serializer = SolutionSerializers.GetSerializerByMoniker(solutionPath)
            ?? throw new NotSupportedException($"Unrecognised solution file '{solutionPath}'.");
        SolutionModel model = await serializer.OpenAsync(solutionPath, cancellationToken).ConfigureAwait(false);
        string dir = Path.GetDirectoryName(solutionPath)!;
        return model.SolutionProjects.Select(p => Path.GetFullPath(Path.Combine(dir, p.FilePath.Replace('\\', Path.DirectorySeparatorChar)))).ToList();
    }
}
