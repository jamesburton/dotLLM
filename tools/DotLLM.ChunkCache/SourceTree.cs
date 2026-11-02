using System.Diagnostics;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.SolutionScan;
using DotLLM.ChunkReuse.Store;
using DotLLM.Tokenizers;

namespace DotLLM.ChunkCache;

/// <summary>A skipped project or file, as reported by <c>build</c> (and <c>--strict</c>).</summary>
/// <param name="Project">Project path (or file, for an excluded file).</param>
/// <param name="Reason">Why (<see cref="SkippedProject"/> reasons, or <c>other-volume</c>).</param>
/// <param name="Message">Detail.</param>
/// <param name="FailsStrict">Whether it makes <c>--strict</c> fail (informational entries such as an extra target framework do not).</param>
internal sealed record SkippedEntry(string Project, string Reason, string Message, bool FailsStrict);

/// <summary>Where a tree comes from: a solution/project file or a directory.</summary>
/// <param name="Kind"><c>solution</c> or <c>directory</c>.</param>
/// <param name="Path">Absolute path.</param>
/// <param name="Configuration">MSBuild configuration (solutions only).</param>
internal sealed record SourceSpec(string Kind, string Path, string? Configuration);

/// <summary>
/// The source files of a solution (Roslyn <see cref="SolutionScanner"/>, authored files only) or of a directory
/// (<see cref="Corpus.FromDirectory"/>'s file rules), turned into index input with <see cref="ChunkIndex.RelativizeScan"/>:
/// one absolute <see cref="Root"/> and '/'-separated relative paths. Files on another volume than the root are
/// <see cref="Excluded"/> (a warning; a <c>--strict</c> failure).
/// </summary>
internal sealed class SourceTree
{
    private readonly IReadOnlyList<ProjectCompilation>? _compilations;

    private SourceTree(SourceSpec spec, IReadOnlyList<SourceFile> files, string root, IReadOnlyList<string> excluded,
        IReadOnlyList<SkippedEntry> skipped, IReadOnlyList<ProjectCompilation>? compilations, string summary)
    {
        Spec = spec;
        Files = files;
        Root = root;
        Excluded = excluded;
        Skipped = skipped;
        _compilations = compilations;
        Summary = summary;
    }

    /// <summary>Where the tree comes from.</summary>
    public SourceSpec Spec { get; }

    /// <summary>The files, with index paths relative to <see cref="Root"/>.</summary>
    public IReadOnlyList<SourceFile> Files { get; }

    /// <summary>Absolute root directory of the index paths.</summary>
    public string Root { get; }

    /// <summary>Files left out because they are on another volume than the root (no relative form).</summary>
    public IReadOnlyList<string> Excluded { get; }

    /// <summary>Skipped projects, unattributed load failures and excluded files.</summary>
    public IReadOnlyList<SkippedEntry> Skipped { get; }

    /// <summary>One line describing the scan.</summary>
    public string Summary { get; }

    /// <summary>Whether a <c>--strict</c> build must fail.</summary>
    public bool FailsStrict => Skipped.Any(s => s.FailsStrict);

    /// <summary>The source spec from <c>--solution</c> / <c>--directory</c> / <c>--configuration</c>, or null when neither is given.</summary>
    public static SourceSpec? SpecFrom(CliContext ctx, CliArgs args)
    {
        string? solution = args.Value("solution"), directory = args.Value("directory");
        if (solution is not null && directory is not null)
            throw new UsageException($"{args.Verb}: give --solution or --directory, not both.");
        string? configuration = args.Value("configuration");
        if (directory is not null)
        {
            if (configuration is not null) throw new UsageException($"{args.Verb}: --configuration applies to --solution only.");
            return new SourceSpec("directory", ctx.FullPath(directory), null);
        }
        return solution is null ? null : new SourceSpec("solution", ctx.FullPath(solution), configuration ?? "Release");
    }

    /// <summary>Scans <paramref name="spec"/>.</summary>
    public static SourceTree Load(CliContext ctx, SourceSpec spec, ITokenizer tokenizer)
    {
        var sw = Stopwatch.StartNew();
        if (spec.Kind == "directory")
        {
            if (!Directory.Exists(spec.Path)) throw new DirectoryNotFoundException($"Directory not found: {spec.Path}");
            Corpus corpus = Corpus.FromDirectory(spec.Path, new NoChunks(), tokenizer);
            (IReadOnlyList<SourceFile> files, string root, IReadOnlyList<string> excluded) = ChunkIndex.RelativizeScan(corpus.Files, spec.Path);
            var skipped = excluded.Select(e => new SkippedEntry(e, "other-volume", "on another volume than the root; not indexed", true)).ToList();
            string summary = $"directory {spec.Path}: {files.Count} files ({string.Join(", ", corpus.Excluded.GroupBy(e => e.Reason).Select(g => $"{g.Count()} {g.Key}").DefaultIfEmpty("none"))} left out by the directory rules) [{sw.Elapsed.TotalSeconds:F1} s]";
            return new SourceTree(spec, files, root, excluded, skipped, null, summary);
        }

        if (!File.Exists(spec.Path)) throw new FileNotFoundException($"Solution or project not found: {spec.Path}", spec.Path);
        ScanResult scan = SolutionScanner.LoadAsync(spec.Path, new SolutionScanOptions(spec.Configuration ?? "Release")).GetAwaiter().GetResult();
        (IReadOnlyList<SourceFile> sfiles, string sroot, IReadOnlyList<string> sexcluded) = ChunkIndex.RelativizeScan(scan.AuthoredFiles, scan.RootDirectory);
        var entries = new List<SkippedEntry>();
        foreach (SkippedProject p in scan.Skipped)
            entries.Add(new SkippedEntry(p.ProjectPath, p.Reason, p.Message, p.Reason != SkippedProject.ExtraTargetFramework));
        // Unattributed loader failures ("Failure: ...") also fail --strict (review-task5-rereview N10); warnings do not.
        foreach (string d in scan.Diagnostics.Where(d => d.StartsWith("Failure", StringComparison.Ordinal)))
            entries.Add(new SkippedEntry("(unattributed)", "load-diagnostic", d, true));
        foreach (string e in sexcluded)
            entries.Add(new SkippedEntry(e, "other-volume", "on another volume than the root; not indexed", true));
        IReadOnlyList<ProjectCompilation> compilations = Rekey(scan.Compilations, scan.RootDirectory, sroot);
        string ssummary = $"solution {spec.Path} (Configuration={spec.Configuration}): {scan.Projects.Count} projects, {scan.Skipped.Count} skipped, "
            + $"{scan.Files.Count} files ({scan.Generated.Count} generated), {sfiles.Count} authored files indexed; root {sroot} [{sw.Elapsed.TotalSeconds:F1} s]";
        return new SourceTree(spec, sfiles, sroot, sexcluded, entries, compilations, ssummary);
    }

    /// <summary>
    /// The reference graph over <paramref name="chunks"/> (chunks of <see cref="Files"/>): per-project compilations for a
    /// solution (references across projects resolve), one BCL-only compilation for a directory.
    /// </summary>
    public ReferenceGraph BuildGraph(ITokenizer tokenizer, IReadOnlyList<Chunk> chunks)
    {
        var builder = new RoslynGraphBuilder(tokenizer);
        return _compilations is null ? builder.Build(Files, chunks) : builder.Build(_compilations, chunks);
    }

    // The scan's compilations are keyed by scan paths (relative to the scan root); when RelativizeScan widened the root, re-key
    // them to index paths so graph chunk paths and compilation trees agree. Trees of files with no relative form keep their key.
    private static IReadOnlyList<ProjectCompilation> Rekey(IReadOnlyList<ProjectCompilation> projects, string scanRoot, string root)
    {
        string a = Path.TrimEndingDirectorySeparator(Path.GetFullPath(scanRoot)), b = Path.TrimEndingDirectorySeparator(Path.GetFullPath(root));
        if (ChunkIndex.PathComparer.Equals(a, b)) return projects;
        string volume = Path.GetPathRoot(b)!;
        string Map(string key)
        {
            string full = Path.IsPathFullyQualified(key) ? Path.GetFullPath(key) : Path.GetFullPath(Path.Combine(a, key));
            return ChunkIndex.PathComparer.Equals(Path.GetPathRoot(full), volume) ? ChunkIndex.NormalizePath(Path.GetRelativePath(b, full)) : key;
        }
        return projects.Select(p => p with { Trees = p.Trees.ToDictionary(kv => Map(kv.Key), kv => kv.Value, RoslynGraphBuilder.PathComparer) }).ToList();
    }

    /// <summary>A chunker that produces nothing: <see cref="Corpus.FromDirectory"/> is used for its file rules only.</summary>
    private sealed class NoChunks : IChunker
    {
        public string Name => "none";

        public int MaxChunkTokens => 0;

        public IReadOnlyList<Chunk> Chunk(SourceFile file) => [];
    }
}
