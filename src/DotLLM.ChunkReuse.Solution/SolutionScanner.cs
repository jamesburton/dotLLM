using System.Text.RegularExpressions;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.MSBuild;
using Microsoft.CodeAnalysis.Text;
using Microsoft.VisualStudio.SolutionPersistence.Model;
using Microsoft.VisualStudio.SolutionPersistence.Serializer;

namespace DotLLM.ChunkReuse.SolutionScan;

/// <summary>
/// A project the scanner did not scan, and why (<see cref="Reason"/> is one of the constants below). Consumers that look for
/// failures filter on <see cref="Reason"/>: <see cref="ExtraTargetFramework"/> is informational.
/// </summary>
/// <param name="ProjectPath">Project file path relative to <see cref="ScanResult.RootDirectory"/>, '/' separated.</param>
/// <param name="Reason">Category: <see cref="LoadFailed"/>, <see cref="UnsupportedLanguage"/> or <see cref="ExtraTargetFramework"/>.</param>
/// <param name="Message">The loader's message(s), or the scanner's explanation.</param>
public sealed record SkippedProject(string ProjectPath, string Reason, string Message)
{
    /// <summary>MSBuild could not evaluate or design-time build the project (missing SDK, bad XML, missing file).</summary>
    public const string LoadFailed = "load-failed";

    /// <summary>The project is not a C# project (Visual Basic, F#, a shared project, or any other project type).</summary>
    public const string UnsupportedLanguage = "unsupported-language";

    /// <summary>
    /// A further target framework of a multi-targeted project (informational). The project is scanned once, from the target
    /// framework <see cref="SolutionScanner"/> prefers (see there); files that only this target compiles are in <see cref="Files"/>
    /// and are not scanned.
    /// </summary>
    public const string ExtraTargetFramework = "extra-target-framework";

    /// <summary>For <see cref="ExtraTargetFramework"/>: files (paths as in <see cref="ScanResult.Files"/>) that only this target compiles,
    /// except its own <c>obj</c>/<c>bin</c> output (implicit usings, assembly info); otherwise empty.</summary>
    public IReadOnlyList<string> Files { get; init; } = [];
}

/// <summary>A scanned C# project.</summary>
/// <param name="Name">Roslyn project name (with the target framework in parentheses for multi-targeted projects).</param>
/// <param name="ProjectPath">Project file path relative to <see cref="ScanResult.RootDirectory"/>, '/' separated.</param>
/// <param name="Files">Files this project owns, ordinal order (see <see cref="SolutionScanner"/> for the owner of a shared file).</param>
/// <param name="Diagnostics">Loader warnings and failures attributed to this project, and every unattributed loader failure.</param>
/// <param name="Compilation">The project's compilation and owned trees, for <see cref="RoslynGraphBuilder"/>.</param>
public sealed record ScannedProject(string Name, string ProjectPath, IReadOnlyList<string> Files, IReadOnlyList<string> Diagnostics,
    ProjectCompilation Compilation)
{
    /// <summary>The scanned target framework (from the Roslyn project name), or null for a single-target project.</summary>
    public string? TargetFramework { get; init; }

    /// <summary>Project paths (relative, as <see cref="ProjectPath"/>) of the projects this one references through the workspace, ordinal order.</summary>
    public IReadOnlyList<string> ProjectReferences { get; init; } = [];
}

/// <summary>
/// The C# files of a solution or project (<see cref="Files"/>: every C# document of the scanned projects, generated ones
/// included and flagged), the projects that were not scanned and why (<see cref="Skipped"/>), and one compilation per scanned
/// project.
/// </summary>
/// <param name="Files">Files in ordinal path order; paths relative to <see cref="RootDirectory"/> ('/' separated; <c>../</c>
/// for files outside it, an absolute path for a file on another drive), with on-disk casing on Windows; text with CRLF → LF,
/// as <see cref="Corpus.FromDirectory"/>.</param>
/// <param name="Skipped">Projects not scanned, by project path.</param>
public sealed record ScanResult(IReadOnlyList<SourceFile> Files, IReadOnlyList<SkippedProject> Skipped)
{
    private (IReadOnlyList<SourceFile> Source, IReadOnlySet<string> Generated, IReadOnlyList<SourceFile> Value)? _authored;
    private (IReadOnlyList<ScannedProject> Source, IReadOnlyList<ProjectCompilation> Value)? _compilations;

    /// <summary>Directory of the solution or project file; all paths are relative to it.</summary>
    public string RootDirectory { get; init; } = "";

    /// <summary>Scanned projects, ordinal by <see cref="ScannedProject.ProjectPath"/>.</summary>
    public IReadOnlyList<ScannedProject> Projects { get; init; } = [];

    /// <summary>Paths of generated files (<see cref="GeneratedFiles.IsGenerated"/>). Source-generator output is not a document and is not in <see cref="Files"/>.</summary>
    public IReadOnlySet<string> Generated { get; init; } = new HashSet<string>(StringComparer.Ordinal);

    /// <summary>Loader diagnostics that could not be attributed to one project (failures among them are also on every scanned project).</summary>
    public IReadOnlyList<string> Diagnostics { get; init; } = [];

    /// <summary>True when <paramref name="path"/> is a generated file (see <see cref="Generated"/>).</summary>
    public bool IsGenerated(string path) => Generated.Contains(path);

    /// <summary>The files that are not generated, in order (what a corpus normally chunks). Computed once.</summary>
    public IReadOnlyList<SourceFile> AuthoredFiles
    {
        get
        {
            if (_authored is { } a && ReferenceEquals(a.Source, Files) && ReferenceEquals(a.Generated, Generated)) return a.Value;
            IReadOnlyList<SourceFile> value = Files.Where(f => !IsGenerated(f.Path)).ToArray();
            _authored = (Files, Generated, value);
            return value;
        }
    }

    /// <summary>One compilation per scanned project, for <see cref="RoslynGraphBuilder.Build(IReadOnlyList{ProjectCompilation}, IReadOnlyList{Chunk})"/>. Computed once.</summary>
    public IReadOnlyList<ProjectCompilation> Compilations
    {
        get
        {
            if (_compilations is { } c && ReferenceEquals(c.Source, Projects)) return c.Value;
            IReadOnlyList<ProjectCompilation> value = Projects.Select(p => p.Compilation).ToArray();
            _compilations = (Projects, value);
            return value;
        }
    }
}

/// <summary>Generated-code detection for scanned files, by path and syntax (not by raw text).</summary>
public static class GeneratedFiles
{
    /// <summary>
    /// True for generated code: a <c>bin</c> or <c>obj</c> directory segment (SDK implicit usings, assembly info; a real source
    /// folder named <c>bin</c>/<c>obj</c> is flagged too), a <c>.g.cs</c>, <c>.g.i.cs</c>, <c>.designer.cs</c> or
    /// <c>.generated.cs</c> file name, an <c>&lt;auto-generated</c> / <c>&lt;autogenerated</c> marker in the comments before the
    /// first token, or a <c>[GeneratedCode]</c> attribute (real attribute syntax). Marker strings anywhere else — string
    /// literals, comments further down, a regex — do not count, unlike <see cref="SymbolChunker.IsGenerated"/>.
    /// </summary>
    public static bool IsGenerated(string relativePath, SyntaxTree tree)
    {
        ArgumentNullException.ThrowIfNull(relativePath);
        ArgumentNullException.ThrowIfNull(tree);
        string[] parts = relativePath.Split('/');
        if (parts[..^1].Any(d => d.Equals("obj", StringComparison.OrdinalIgnoreCase) || d.Equals("bin", StringComparison.OrdinalIgnoreCase)))
            return true;
        string name = parts[^1].ToLowerInvariant();
        if (name.EndsWith(".g.cs", StringComparison.Ordinal) || name.EndsWith(".g.i.cs", StringComparison.Ordinal)
            || name.EndsWith(".designer.cs", StringComparison.Ordinal) || name.EndsWith(".generated.cs", StringComparison.Ordinal))
            return true;
        SyntaxNode root = tree.GetRoot();
        foreach (SyntaxTrivia trivia in root.GetFirstToken(includeZeroWidth: true).LeadingTrivia)
        {
            if (!trivia.IsKind(SyntaxKind.SingleLineCommentTrivia) && !trivia.IsKind(SyntaxKind.MultiLineCommentTrivia)) continue;
            string text = trivia.ToString();
            if (text.Contains("<auto-generated", StringComparison.OrdinalIgnoreCase) || text.Contains("<autogenerated", StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return root.DescendantNodes(n => n is not (BlockSyntax or ArrowExpressionClauseSyntax or EqualsValueClauseSyntax))
            .OfType<AttributeSyntax>().Any(a => RightmostName(a.Name) is "GeneratedCode" or "GeneratedCodeAttribute");
    }

    private static string RightmostName(NameSyntax name) => name switch
    {
        QualifiedNameSyntax q => q.Right.Identifier.ValueText,
        AliasQualifiedNameSyntax a => a.Name.Identifier.ValueText,
        SimpleNameSyntax s => s.Identifier.ValueText,
        _ => "",
    };
}

/// <summary>
/// Loads a solution (<c>.sln</c>, <c>.slnx</c>) or a project with Roslyn's <see cref="MSBuildWorkspace"/> and returns its C#
/// files and per-project compilations. Projects that do not load, are not C#, or repeat a target framework are listed in
/// <see cref="ScanResult.Skipped"/> with the reason; a load failure never aborts the scan silently.
/// </summary>
/// <remarks>
/// <para>
/// Design-time builds run in Roslyn's out-of-process BuildHost, which locates MSBuild and the .NET SDK itself, so no in-process
/// MSBuild registration (<c>Microsoft.Build.Locator</c>) is needed. Projects should be restored first (as after a build); a
/// project that loads without a core library gets a warning in <see cref="ScannedProject.Diagnostics"/>.
/// </para>
/// <para>
/// Loader diagnostics are attributed by the project path Roslyn quotes in them (normalised, <see cref="RoslynGraphBuilder.PathComparer"/>);
/// a failure that names no project is added to every scanned project and to <see cref="ScanResult.Diagnostics"/>. A project
/// with an attributed failure (the workspace keeps it, with whatever items it got) is skipped as <see cref="SkippedProject.LoadFailed"/>.
/// </para>
/// <para>
/// Multi-targeted projects: the scanned target framework is the highest .NET (5+) version, then .NET Core, .NET Standard,
/// .NET Framework, platform-neutral before platform-specific, then ordinal name. Files only the other targets compile are
/// listed on their <see cref="SkippedProject.ExtraTargetFramework"/> entries.
/// </para>
/// <para>
/// A file compiled by several projects is scanned once and owned by the project whose directory holds it (the deepest), else by
/// the first project by path. On Windows, paths take their on-disk casing, so different spellings of one file are one file.
/// Document texts are replaced by their CRLF → LF form before the compilations are made, so syntax-tree offsets equal
/// <see cref="SourceFile"/> offsets (and chunk offsets).
/// </para>
/// </remarks>
public static class SolutionScanner
{
    private static readonly StringComparer PathComparer = RoslynGraphBuilder.PathComparer;
    private static readonly StringComparison PathComparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;

    /// <summary>Loads <paramref name="slnOrProjectPath"/> (a <c>.sln</c>/<c>.slnx</c>, or a project file with its project references).</summary>
    public static async Task<ScanResult> LoadAsync(string slnOrProjectPath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(slnOrProjectPath);
        cancellationToken.ThrowIfCancellationRequested();
        var canonical = new PathCanonicalizer();
        string full = canonical.Of(Path.GetFullPath(slnOrProjectPath));
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
            expected = (await ListSolutionProjectsAsync(full, cancellationToken).ConfigureAwait(false)).Select(canonical.Of).ToList();
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

        // Roslyn projects by (canonical) project file; a multi-targeted project gives one per target framework.
        var byFile = solution.Projects.Where(p => p.FilePath is not null)
            .GroupBy(p => canonical.Of(Path.GetFullPath(p.FilePath!)), PathComparer)
            .ToDictionary(g => g.Key, g => g.OrderByDescending(p => p, TargetFrameworkPreference.Instance).ToList(), PathComparer);

        (Dictionary<string, List<WorkspaceDiagnostic>> attributed, List<WorkspaceDiagnostic> unattributed) =
            AttributeDiagnostics(workspace.Diagnostics, expected.Concat(byFile.Keys), d => d.Message, canonical.Of);
        List<WorkspaceDiagnostic> DiagnosticsOf(string projectPath) =>
            attributed.TryGetValue(projectPath, out List<WorkspaceDiagnostic>? list) ? list : [];
        List<string> unattributedFailures = unattributed.Where(d => d.Kind == WorkspaceDiagnosticKind.Failure)
            .Select(d => "Unattributed failure: " + Format(d)).ToList();

        foreach (string path in expected.Where(p => !byFile.ContainsKey(p)))
        {
            List<string> messages = DiagnosticsOf(path).Select(Format).ToList();
            if (rootFailure is not null && PathComparer.Equals(path, full)) messages.Insert(0, rootFailure);
            string pext = Path.GetExtension(path);
            bool csharp = pext.Equals(".csproj", StringComparison.OrdinalIgnoreCase);
            string fallback = pext.Equals(".shproj", StringComparison.OrdinalIgnoreCase)
                ? "Shared project: not a project of its own; its files are scanned through the projects that import it."
                : csharp ? "The workspace did not load the project." : "Not a C# project.";
            string message = messages.Count > 0 ? string.Join("\n", messages) : fallback;
            skipped.Add(new SkippedProject(Rel(path), csharp ? SkippedProject.LoadFailed : SkippedProject.UnsupportedLanguage, message));
        }

        // Scanned projects: C#, loaded without an attributed failure, the preferred target framework.
        var primary = new List<(string Path, Project Project, List<Project> Extras, List<string> Diagnostics)>();
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
            primary.Add((path, first, projects.Skip(1).ToList(), own.Select(Format).Concat(unattributedFailures).ToList()));
        }

        // LF texts (every C# document, so a file compiled by several projects reads the same everywhere).
        var texts = new Dictionary<string, string>(PathComparer);
        foreach (Project project in solution.Projects.Where(p => p.Language == LanguageNames.CSharp))
        {
            foreach (Document doc in project.Documents.Where(d => d.FilePath is not null))
            {
                cancellationToken.ThrowIfCancellationRequested();
                string docPath = canonical.Of(doc.FilePath!);
                if (!texts.TryGetValue(docPath, out string? lf))
                {
                    SourceText text = await doc.GetTextAsync(cancellationToken).ConfigureAwait(false);
                    lf = text.ToString().Replace("\r\n", "\n", StringComparison.Ordinal);
                    texts[docPath] = lf;
                }
                solution = solution.WithDocumentText(doc.Id, SourceText.From(lf, System.Text.Encoding.UTF8));
            }
        }

        // Owner of each file: the scanned project whose directory holds it (deepest), else the first by project path.
        var owner = new Dictionary<string, string>(PathComparer); // canonical file → canonical project path
        foreach ((string path, Project p, _, _) in primary)
        {
            string projectDir = Path.GetDirectoryName(path)! + Path.DirectorySeparatorChar;
            foreach (Document doc in p.Documents.Where(d => d.FilePath is not null))
            {
                string file = canonical.Of(doc.FilePath!);
                bool inside = file.StartsWith(projectDir, PathComparison);
                if (!owner.TryGetValue(file, out string? current)) { owner[file] = path; continue; }
                string currentDir = Path.GetDirectoryName(current)! + Path.DirectorySeparatorChar;
                bool currentInside = file.StartsWith(currentDir, PathComparison);
                if (inside && (!currentInside || projectDir.Length > currentDir.Length)) owner[file] = path;
            }
        }

        var files = new List<SourceFile>();
        var generated = new HashSet<string>(StringComparer.Ordinal);
        var scanned = new List<ScannedProject>();
        var scannedFiles = new HashSet<string>(PathComparer);
        foreach ((string path, Project p, List<Project> extras, List<string> projectDiagnostics) in primary)
        {
            Project project = solution.GetProject(p.Id)!;
            Compilation compilation = (await project.GetCompilationAsync(cancellationToken).ConfigureAwait(false))!;
            if (compilation.GetSpecialType(SpecialType.System_Object).TypeKind == TypeKind.Error)
                projectDiagnostics.Add("Warning: no core library reference (System.Object does not resolve): project not restored, or its references did not resolve.");
            var trees = new Dictionary<string, SyntaxTree>(StringComparer.Ordinal);
            foreach (Document doc in project.Documents.Where(d => d.FilePath is not null))
            {
                string file = canonical.Of(doc.FilePath!);
                if (!PathComparer.Equals(owner[file], path) || !scannedFiles.Add(file)) continue;
                string rel = Rel(file);
                SyntaxTree tree = (await doc.GetSyntaxTreeAsync(cancellationToken).ConfigureAwait(false))!;
                files.Add(new SourceFile(rel, texts[file]));
                if (GeneratedFiles.IsGenerated(rel, tree)) generated.Add(rel);
                trees[rel] = tree;
            }
            string? tfm = TargetFrameworkPreference.Of(p);
            scanned.Add(new ScannedProject(project.Name, Rel(path), trees.Keys.Order(StringComparer.Ordinal).ToArray(), projectDiagnostics,
                new ProjectCompilation(project.Name, compilation, trees))
            {
                TargetFramework = tfm,
                ProjectReferences = project.ProjectReferences.Select(r => solution.GetProject(r.ProjectId)?.FilePath)
                    .Where(f => f is not null).Select(f => Rel(canonical.Of(f!))).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray(),
            });
            foreach (Project extra in extras)
            {
                var primaryDocs = p.Documents.Where(d => d.FilePath is not null).Select(d => canonical.Of(d.FilePath!)).ToHashSet(PathComparer);
                string[] only = extra.Documents.Where(d => d.FilePath is not null).Select(d => canonical.Of(d.FilePath!))
                    .Where(f => !primaryDocs.Contains(f)).Select(Rel).Where(r => !IsBuildOutput(r)).Distinct(StringComparer.Ordinal).Order(StringComparer.Ordinal).ToArray();
                string message = $"Project '{extra.Name}': files scanned from '{project.Name}'."
                    + (only.Length == 0 ? "" : $" Not scanned (only this target compiles them): {string.Join(", ", only)}.");
                skipped.Add(new SkippedProject(Rel(path), SkippedProject.ExtraTargetFramework, message) { Files = only });
            }
        }

        return new ScanResult(files.OrderBy(f => f.Path, StringComparer.Ordinal).ToArray(),
            skipped.OrderBy(s => s.ProjectPath, StringComparer.Ordinal).ThenBy(s => s.Reason, StringComparer.Ordinal)
                .ThenBy(s => s.Message, StringComparer.Ordinal).ToArray())
        {
            RootDirectory = root,
            Projects = scanned,
            Generated = generated,
            Diagnostics = unattributed.Select(Format).ToArray(),
        };
    }

    private static string Format(WorkspaceDiagnostic d) => $"{d.Kind}: {d.Message}";

    /// <summary>A path with an <c>obj</c> or <c>bin</c> directory segment (per-target SDK output, not listed as a lost file).</summary>
    private static bool IsBuildOutput(string relativePath) => relativePath.Split('/')[..^1]
        .Any(d => d.Equals("obj", StringComparison.OrdinalIgnoreCase) || d.Equals("bin", StringComparison.OrdinalIgnoreCase));

    // Roslyn's MSBuild loader quotes the project file: "Msbuild failed when processing the file '<path>' with message: ...".
    private static readonly Regex ProcessingFile = new(@"processing the file '([^']+)'", RegexOptions.CultureInvariant);
    private static readonly Regex Quoted = new(@"'([^']+)'", RegexOptions.CultureInvariant);

    /// <summary>
    /// Attributes loader diagnostics to projects by the project path quoted in the message: the path after "processing the
    /// file" when present, otherwise the single quoted path that is one of <paramref name="projectPaths"/>. Paths are compared
    /// after <see cref="Path.GetFullPath(string)"/> and <paramref name="canonical"/>, with <see cref="RoslynGraphBuilder.PathComparer"/>.
    /// A path merely contained in the text (unquoted, or as a prefix of a longer path) does not attribute. The loader also
    /// reports a referenced project's failure again with an empty path (<c>processing the file ''</c>), which stays unattributed.
    /// </summary>
    internal static (Dictionary<string, List<T>> ByProject, List<T> Unattributed) AttributeDiagnostics<T>(IEnumerable<T> diagnostics,
        IEnumerable<string> projectPaths, Func<T, string> message, Func<string, string> canonical)
    {
        var known = new HashSet<string>(projectPaths, PathComparer);
        var byProject = new Dictionary<string, List<T>>(PathComparer);
        var unattributed = new List<T>();
        foreach (T d in diagnostics)
        {
            string text = message(d);
            string? target = null;
            Match m = ProcessingFile.Match(text);
            if (m.Success)
            {
                target = Normalise(m.Groups[1].Value);
                if (target is not null && !known.Contains(target)) target = null;
            }
            else
            {
                string[] hits = Quoted.Matches(text).Select(q => Normalise(q.Groups[1].Value)).OfType<string>()
                    .Where(known.Contains).Distinct(PathComparer).ToArray();
                if (hits.Length == 1) target = hits[0];
            }
            if (target is null) unattributed.Add(d);
            else
            {
                if (!byProject.TryGetValue(target, out List<T>? list)) byProject[target] = list = [];
                list.Add(d);
            }
        }
        return (byProject, unattributed);

        string? Normalise(string p)
        {
            try { return Path.IsPathRooted(p) ? canonical(Path.GetFullPath(p)) : null; }
            catch (Exception e) when (e is ArgumentException or NotSupportedException or PathTooLongException) { return null; }
        }
    }

    /// <summary>Full paths of the projects a solution file lists (solution folders excluded).</summary>
    private static async Task<List<string>> ListSolutionProjectsAsync(string solutionPath, CancellationToken cancellationToken)
    {
        var serializer = SolutionSerializers.GetSerializerByMoniker(solutionPath)
            ?? throw new NotSupportedException($"Unrecognised solution file '{solutionPath}'.");
        SolutionModel model = await serializer.OpenAsync(solutionPath, cancellationToken).ConfigureAwait(false);
        string dir = Path.GetDirectoryName(solutionPath)!;
        return model.SolutionProjects.Select(p => Path.GetFullPath(Path.Combine(dir, p.FilePath.Replace('\\', Path.DirectorySeparatorChar)))).ToList();
    }

    /// <summary>Orders Roslyn projects of one multi-targeted project file by preference (greater = preferred).</summary>
    internal sealed class TargetFrameworkPreference : IComparer<Project>
    {
        public static readonly TargetFrameworkPreference Instance = new();

        /// <summary>The target framework in a Roslyn project name such as <c>Lib(net10.0)</c>, or null.</summary>
        public static string? Of(Project p) => FromName(p.Name);

        internal static string? FromName(string name)
        {
            int open = name.LastIndexOf('(');
            return open > 0 && name.EndsWith(')') ? name[(open + 1)..^1] : null;
        }

        public int Compare(Project? x, Project? y) => CompareNames(x?.Name ?? "", y?.Name ?? "");

        /// <summary>Compares two Roslyn project names by their target framework (see <see cref="SolutionScanner"/> remarks).</summary>
        internal static int CompareNames(string x, string y)
        {
            (int fx, Version vx, bool nx) = Rank(FromName(x));
            (int fy, Version vy, bool ny) = Rank(FromName(y));
            int c = fx.CompareTo(fy);
            if (c == 0) c = vx.CompareTo(vy);
            if (c == 0) c = nx.CompareTo(ny);
            // Last: ordinal name, the smaller name preferred (so the preferred one sorts first in a descending order).
            return c != 0 ? c : string.CompareOrdinal(y, x);
        }

        /// <summary>Family (4 net5+, 3 netcoreapp, 2 netstandard, 1 .NET Framework, 0 unknown), version, platform-neutral.</summary>
        private static (int Family, Version Version, bool Neutral) Rank(string? tfm)
        {
            if (tfm is null) return (0, new Version(0, 0), true);
            string t = tfm.ToLowerInvariant();
            int dash = t.IndexOf('-');
            bool neutral = dash < 0;
            string core = neutral ? t : t[..dash];
            if (core.StartsWith("netcoreapp", StringComparison.Ordinal)) return (3, V(core[10..]), neutral);
            if (core.StartsWith("netstandard", StringComparison.Ordinal)) return (2, V(core[11..]), neutral);
            if (core.StartsWith("net", StringComparison.Ordinal))
            {
                string rest = core[3..];
                if (rest.Contains('.')) return (4, V(rest), neutral);                 // net5.0 … net10.0
                if (rest.Length > 0 && rest.All(char.IsAsciiDigit))                    // net48, net472 (no dot)
                    return (1, V(string.Join('.', rest.ToCharArray())), neutral);
            }
            return (0, new Version(0, 0), neutral);
        }

        private static Version V(string s) => Version.TryParse(s.Contains('.') ? s : s + ".0", out Version? v) ? v : new Version(0, 0);
    }

    /// <summary>Gives full paths their on-disk casing on Windows (cached per directory); identity elsewhere.</summary>
    private sealed class PathCanonicalizer
    {
        private readonly Dictionary<string, string> _cache = new(StringComparer.OrdinalIgnoreCase);

        public string Of(string fullPath)
        {
            if (!OperatingSystem.IsWindows()) return fullPath;
            if (_cache.TryGetValue(fullPath, out string? hit)) return hit;
            string result = fullPath;
            try
            {
                string? dir = Path.GetDirectoryName(fullPath);
                string name = Path.GetFileName(fullPath);
                if (dir is not null && name.Length > 0 && name.IndexOfAny(['*', '?']) < 0)
                {
                    string cdir = Of(dir);
                    var info = new DirectoryInfo(cdir);
                    string? actual = info.Exists ? info.EnumerateFileSystemInfos(name).Select(i => i.Name)
                        .FirstOrDefault(n => n.Equals(name, StringComparison.OrdinalIgnoreCase)) : null;
                    result = Path.Combine(cdir, actual ?? name);
                }
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException or System.Security.SecurityException)
            {
                result = fullPath;
            }
            _cache[fullPath] = result;
            return result;
        }
    }
}
