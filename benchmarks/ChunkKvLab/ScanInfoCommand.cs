using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;

namespace ChunkKvLab;

/// <summary>
/// <c>scan-info</c> (Phase S Task 5, tokenizer only): loads a solution with <see cref="SolutionScanner"/> and prints its
/// projects, skipped projects (with reasons), files (generated / authored), then compares per-project compilations with the
/// BCL-only compilation over the same authored files: invocation sites that bind (<see cref="CallResolution"/>) and reference
/// graph edges by kind (chunks of the authored files with the given chunker).
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
        Console.WriteLine($"projects scanned {scan.Projects.Count}, skipped entries {scan.Skipped.Count}; files {scan.Files.Count} "
            + $"(generated {scan.Generated.Count}, authored {authored.Count}); authored chars {authored.Sum(f => (long)f.Text.Length)}");
        foreach (ScannedProject p in scan.Projects)
            Console.WriteLine($"  project {p.ProjectPath} [{p.Name}]: files {p.Files.Count}, authored {p.Files.Count(authoredSet.Contains)}, "
                + $"diagnostics {p.Diagnostics.Count}");
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

        ProjectCompilation bcl = ProjectCompilation.BclOnly(authored);
        var modes = new (string Name, IReadOnlyList<ProjectCompilation> Compilations)[] { ("projects", scan.Compilations), ("bcl-only", [bcl]) };
        var graphs = new Dictionary<string, ReferenceGraph>(StringComparer.Ordinal);
        var builder = new RoslynGraphBuilder(tokenizer);
        foreach ((string name, IReadOnlyList<ProjectCompilation> compilations) in modes)
        {
            sw.Restart();
            CallResolutionStats calls = CallResolution.Measure(compilations, authoredSet);
            double callS = sw.Elapsed.TotalSeconds;
            sw.Restart();
            ReferenceGraph graph = name == "projects" ? builder.Build(compilations, chunks) : builder.Build(authored, chunks);
            double graphS = sw.Elapsed.TotalSeconds;
            graphs[name] = graph;
            Console.WriteLine($"{name}: invocations {calls.Invocations}, bind {calls.Resolved} ({Pct(calls.ResolvedShare)}), "
                + $"to an authored file {calls.ResolvedToFile} ({Pct(calls.ResolvedToFileShare)}) [{callS:F1} s]; "
                + $"graph edges {graph.Edges.Count} ({graph.Edges.Count / (tokens / 1000.0):F2} per 1K tokens) [{graphS:F1} s]; "
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

        // Per project: invocation binding share under both set-ups, on the project's own authored files.
        Console.WriteLine("per project (authored files): invocations, bind share projects vs bcl-only, to-authored-file share projects vs bcl-only");
        foreach (ScannedProject p in scan.Projects)
        {
            var own = p.Files.Where(authoredSet.Contains).ToHashSet(StringComparer.Ordinal);
            if (own.Count == 0) continue;
            CallResolutionStats a = CallResolution.Measure(scan.Compilations, own), b = CallResolution.Measure([bcl], own);
            Console.WriteLine($"  {p.ProjectPath}: {a.Invocations}; {Pct(a.ResolvedShare)} vs {Pct(b.ResolvedShare)}; "
                + $"{Pct(a.ResolvedToFileShare)} vs {Pct(b.ResolvedToFileShare)}");
        }
        return 0;
    }

    private static string OneLine(string s) => s.Replace("\r", "", StringComparison.Ordinal).Replace('\n', ' ');

    private static string Pct(double share) => (100 * share).ToString("F1", CultureInfo.InvariantCulture) + " %";
}
