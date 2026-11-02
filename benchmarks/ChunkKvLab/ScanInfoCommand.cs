using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.SolutionScan;

namespace ChunkKvLab;

/// <summary>
/// <c>scan-info</c> (Phase S Task 5, tokenizer only): loads a solution with <see cref="SolutionScanner"/> and prints its
/// projects, skipped projects (with reasons), files (generated / authored; generated files outside <c>obj/</c> by path), then
/// compares per-project compilations with the BCL-only compilation over the same authored files: invocation sites that bind
/// (<see cref="CallResolution"/>) for all projects, without test projects, and without test and benchmark projects (the
/// <see cref="Corpus.FromDirectory"/> scope); site-by-site differences; and reference graph edges by kind.
/// </summary>
internal static class ScanInfoCommand
{
    public static int Run(string modelPath, string solution, string chunkerName, int chunkTokens)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        var sw = Stopwatch.StartNew();
        ScanResult scan = SolutionScanner.LoadAsync(solution).GetAwaiter().GetResult();
        Console.WriteLine($"scan {solution}: {sw.Elapsed.TotalSeconds:F1} s (workspace load, LF texts, compilations)");

        IReadOnlyList<SourceFile> authored = scan.AuthoredFiles;
        var authoredSet = authored.Select(f => f.Path).ToHashSet(StringComparer.Ordinal);
        int objGenerated = scan.Generated.Count(p => IsObjOrBin(p));
        Console.WriteLine($"projects scanned {scan.Projects.Count}, skipped entries {scan.Skipped.Count}; files {scan.Files.Count} "
            + $"(generated {scan.Generated.Count}: {objGenerated} under obj/bin, {scan.Generated.Count - objGenerated} by name/header/attribute; "
            + $"authored {authored.Count}); authored chars {authored.Sum(f => (long)f.Text.Length)}; "
            + $"files outside the solution directory {scan.Files.Count(f => f.Path.StartsWith("../", StringComparison.Ordinal) || Path.IsPathRooted(f.Path))}");
        foreach (string g in scan.Generated.Where(p => !IsObjOrBin(p)).Order(StringComparer.Ordinal))
            Console.WriteLine($"  generated (not obj/bin): {g}");
        foreach (ScannedProject p in scan.Projects)
            Console.WriteLine($"  project {p.ProjectPath} [{p.Name}]: files {p.Files.Count}, authored {p.Files.Count(authoredSet.Contains)}, "
                + $"diagnostics {p.Diagnostics.Count}, project references {p.ProjectReferences.Count}{(IsTest(p) ? ", test" : IsBenchmark(p) ? ", benchmark" : "")}");
        foreach (ScannedProject p in scan.Projects)
            foreach (string d in p.Diagnostics)
                Console.WriteLine($"  diagnostic {p.ProjectPath}: {OneLine(d)}");
        foreach (SkippedProject s in scan.Skipped)
            Console.WriteLine($"  skipped {s.ProjectPath}: {s.Reason}: {OneLine(s.Message)}");
        foreach (string d in scan.Diagnostics)
            Console.WriteLine($"  unattributed diagnostic: {OneLine(d)}");

        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(modelPath);
        var tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        IChunker chunker = TasksRunner.MakeChunker(chunkerName, tokenizer, chunkTokens);
        sw.Restart();
        Chunk[] chunks = authored.SelectMany(chunker.Chunk).ToArray();
        long tokens = chunks.Sum(c => (long)c.TokenIds.Length);
        Console.WriteLine($"chunker {chunker.Name}: {chunks.Length} chunks, {tokens} tokens ({sw.Elapsed.TotalSeconds:F1} s)");

        // Invocation sites under both set-ups (same LF texts, so (path, offset) identifies a site in both).
        ProjectCompilation bcl = ProjectCompilation.BclOnly(authored);
        sw.Restart();
        IReadOnlyList<CallSite> projectSites = CallResolution.Sites(scan.Compilations, authoredSet);
        double projectSitesS = sw.Elapsed.TotalSeconds;
        sw.Restart();
        IReadOnlyList<CallSite> bclSites = CallResolution.Sites([bcl], authoredSet);
        double bclSitesS = sw.Elapsed.TotalSeconds;
        Console.WriteLine($"call sites: projects {projectSites.Count} [{projectSitesS:F1} s], bcl-only {bclSites.Count} [{bclSitesS:F1} s]");

        var ownerOf = new Dictionary<string, ScannedProject>(StringComparer.Ordinal);
        foreach (ScannedProject p in scan.Projects) foreach (string f in p.Files) ownerOf[f] = p;
        Func<ScannedProject, bool>[] scopes = [_ => true, p => !IsTest(p), p => !IsTest(p) && !IsBenchmark(p)];
        string[] scopeNames = ["all projects", "without test projects", "without test and benchmark projects (Corpus.FromDirectory scope)"];
        Console.WriteLine("binding by scope (each set-up over its own sites): projects bind / bcl-only bind; projects to-file / bcl-only to-file");
        for (int i = 0; i < scopes.Length; i++)
        {
            Func<ScannedProject, bool> inScope = scopes[i];
            int nProjects = scan.Projects.Count(inScope);
            Stats a = Stats.Of(projectSites.Where(s => inScope(ownerOf[s.Path]))), b = Stats.Of(bclSites.Where(s => inScope(ownerOf[s.Path])));
            Console.WriteLine($"  {scopeNames[i]} ({nProjects} projects): sites {a.N} / {b.N}; bind {Pct(a.Bind, a.N)} / {Pct(b.Bind, b.N)}; "
                + $"to an authored file {Pct(a.File, a.N)} / {Pct(b.File, b.N)} ({a.File} / {b.File})");
        }

        // Site by site (sites present in both set-ups).
        var bclByKey = bclSites.ToDictionary(s => (s.Path, s.Offset));
        var pairs = projectSites.Where(s => bclByKey.ContainsKey((s.Path, s.Offset))).Select(s => (P: s, B: bclByKey[(s.Path, s.Offset)])).ToArray();
        Console.WriteLine($"sites in both set-ups {pairs.Length} (projects only {projectSites.Count - pairs.Length}, bcl-only only {bclSites.Count - pairs.Length}); projects kind x bcl-only kind:");
        foreach (var g in pairs.GroupBy(x => (x.P.Kind, x.B.Kind)).OrderBy(g => g.Key.Item1).ThenBy(g => g.Key.Item2))
        {
            int differentTarget = g.Key is (CallTargetKind.File, CallTargetKind.File)
                ? g.Count(x => x.P.DefinitionPath != x.B.DefinitionPath || x.P.DefinitionOffset != x.B.DefinitionOffset) : 0;
            Console.WriteLine($"  {g.Key.Item1,-10} x {g.Key.Item2,-10} {g.Count(),7}{(differentTarget > 0 ? $" (different declaration: {differentTarget})" : "")}");
        }
        // Where BCL-only binds to an authored file and per-project does not (or to another declaration): the sites to explain.
        var reverse = pairs.Where(x => x.B.Kind == CallTargetKind.File
            && (x.P.Kind != CallTargetKind.File || x.P.DefinitionPath != x.B.DefinitionPath || x.P.DefinitionOffset != x.B.DefinitionOffset)).ToArray();
        Console.WriteLine($"bcl-only binds to an authored file where per-project does not (or elsewhere): {reverse.Length}; by project: "
            + string.Join(", ", reverse.GroupBy(x => ownerOf[x.P.Path].ProjectPath).OrderByDescending(g => g.Count()).Select(g => $"{g.Key} {g.Count()}")));
        var byPattern = reverse.GroupBy(x => (x.P.Kind, P: x.P.Method ?? "-", B: x.B.Method ?? "-")).OrderByDescending(g => g.Count()).ToArray();
        foreach (var g in byPattern.Take(40))
        {
            var first = g.First();
            Console.WriteLine($"  {g.Count(),4}x projects {g.Key.Kind} {g.Key.P} | bcl-only {g.Key.B} @ {first.B.DefinitionPath}  (e.g. {first.P.Path}@{first.P.Offset}: \"{Snippet(scan, first.P)}\")");
        }
        if (byPattern.Length > 40) Console.WriteLine($"  ... {byPattern.Length - 40} more patterns");

        // Graph edges.
        var graphs = new Dictionary<string, ReferenceGraph>(StringComparer.Ordinal);
        var builder = new RoslynGraphBuilder(tokenizer);
        foreach (string name in new[] { "projects", "bcl-only" })
        {
            sw.Restart();
            ReferenceGraph graph = name == "projects" ? builder.Build(scan.Compilations, chunks) : builder.Build(authored, chunks);
            graphs[name] = graph;
            Console.WriteLine($"{name}: graph edges {graph.Edges.Count} ({graph.Edges.Count / (tokens / 1000.0):F2} per 1K tokens) [{sw.Elapsed.TotalSeconds:F1} s]; "
                + string.Join(", ", Enum.GetValues<SymbolEdgeKind>().Select(k => $"{k} {graph.Edges.Count(e => e.Kind == k)}")));
        }
        var projectEdges = graphs["projects"].Edges.ToHashSet();
        var bclEdges = graphs["bcl-only"].Edges.ToHashSet();
        int both = projectEdges.Count(bclEdges.Contains);
        Console.WriteLine($"edge overlap: both {both}, projects only {projectEdges.Count - both}, bcl-only only {bclEdges.Count - both}");
        foreach (SymbolEdgeKind k in Enum.GetValues<SymbolEdgeKind>())
        {
            int p = projectEdges.Count(e => e.Kind == k), b = bclEdges.Count(e => e.Kind == k), pb = projectEdges.Count(e => e.Kind == k && bclEdges.Contains(e));
            Console.WriteLine($"  {k,-10} projects {p}, bcl-only {b}, both {pb}");
        }

        // Per project: binding under both set-ups, each over its own sites in the project's authored files.
        Console.WriteLine("per project (authored files): sites projects / bcl-only; bind share projects vs bcl-only; to-authored-file share projects vs bcl-only");
        foreach (ScannedProject p in scan.Projects)
        {
            Stats a = Stats.Of(projectSites.Where(s => ownerOf[s.Path] == p)), b = Stats.Of(bclSites.Where(s => ownerOf[s.Path] == p));
            if (a.N + b.N == 0) continue;
            Console.WriteLine($"  {p.ProjectPath}: {a.N} / {b.N}; {Pct(a.Bind, a.N)} vs {Pct(b.Bind, b.N)}; {Pct(a.File, a.N)} vs {Pct(b.File, b.N)}");
        }
        return 0;
    }

    private readonly record struct Stats(int N, int Bind, int File)
    {
        public static Stats Of(IEnumerable<CallSite> sites)
        {
            int n = 0, bind = 0, file = 0;
            foreach (CallSite s in sites)
            {
                n++;
                if (s.Kind != CallTargetKind.Unresolved) bind++;
                if (s.Kind == CallTargetKind.File) file++;
            }
            return new Stats(n, bind, file);
        }
    }

    private static bool IsObjOrBin(string path) =>
        path.Split('/')[..^1].Any(d => d.Equals("obj", StringComparison.OrdinalIgnoreCase) || d.Equals("bin", StringComparison.OrdinalIgnoreCase));

    /// <summary>A test project: a directory segment of its path or its name as <see cref="Corpus.FromDirectory"/> treats test directories.</summary>
    private static bool IsTest(ScannedProject p) => p.ProjectPath.Split('/').Any(d =>
    {
        string s = d.ToLowerInvariant().Replace(".csproj", "", StringComparison.Ordinal);
        return s is "test" or "tests" || s.EndsWith(".tests", StringComparison.Ordinal) || s.EndsWith(".test", StringComparison.Ordinal)
            || s.EndsWith("tests", StringComparison.Ordinal);
    });

    private static bool IsBenchmark(ScannedProject p) => p.ProjectPath.Split('/').Any(d =>
    {
        string s = d.ToLowerInvariant().Replace(".csproj", "", StringComparison.Ordinal);
        return s is "benchmarks" || s.EndsWith(".benchmarks", StringComparison.Ordinal);
    });

    private static string Snippet(ScanResult scan, CallSite site)
    {
        string text = scan.Files.First(f => f.Path == site.Path).Text;
        int end = text.IndexOf('\n', site.Offset);
        string s = text[site.Offset..(end < 0 ? text.Length : end)].Trim();
        return s.Length > 70 ? s[..70] + "…" : s;
    }

    private static string OneLine(string s) => s.Replace("\r", "", StringComparison.Ordinal).Replace('\n', ' ');

    private static string Pct(int part, int whole) => whole == 0 ? "n/a" : (100.0 * part / whole).ToString("F2", CultureInfo.InvariantCulture) + " %";
}
