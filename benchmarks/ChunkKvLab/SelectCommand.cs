using System.Diagnostics;
using System.Globalization;
using DotLLM.ChunkReuse.CodeCorpus;
using DotLLM.ChunkReuse.Graph;
using DotLLM.ChunkReuse.SolutionScan;
using DotLLM.ChunkReuse.Store;
using DotLLM.ChunkReuse.Tooling;

namespace ChunkKvLab;

/// <summary>
/// <c>select</c> (Phase S Task 6, tokenizer only): scans a solution, relativises its authored files
/// (<see cref="ChunkIndex.RelativizeScan"/>), chunks them, builds the per-project reference graph and runs
/// <see cref="ContextSelector"/>; prints the symbol's definitions, counts by reason and hop, tokens, what the budget trimmed and
/// the first selected chunks. The same request without a budget is run first, to show the whole neighbourhood.
/// </summary>
internal static class SelectCommand
{
    public static int Run(string modelPath, string solution, string configuration, string chunkerName, int chunkTokens,
        SelectionRequest request, int show)
    {
        CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;
        var sw = Stopwatch.StartNew();
        ScanResult scan = SolutionScanner.LoadAsync(solution, new SolutionScanOptions(configuration)).GetAwaiter().GetResult();
        double scanS = sw.Elapsed.TotalSeconds;
        (IReadOnlyList<SourceFile> files, string root, IReadOnlyList<string> excluded) = ChunkIndex.RelativizeScan(scan.AuthoredFiles, scan.RootDirectory);
        if (!string.Equals(Path.GetFullPath(root), Path.GetFullPath(scan.RootDirectory), StringComparison.OrdinalIgnoreCase))
            throw new InvalidOperationException($"RelativizeScan widened the root to {root}; the per-project graph is keyed by scan paths, so this command needs the scan root.");
        Console.WriteLine($"scan {solution} (Configuration={configuration}): {scan.Projects.Count} projects, {scan.Skipped.Count} skipped entries, "
            + $"{scan.Files.Count} files ({scan.Generated.Count} generated), {files.Count} authored files after RelativizeScan "
            + $"(excluded {excluded.Count}); root {root} [{scanS:F1} s]");

        using var gguf = DotLLM.Models.Gguf.GgufFile.Open(modelPath);
        var tokenizer = DotLLM.Models.Gguf.GgufBpeTokenizerFactory.Load(gguf.Metadata);
        IChunker chunker = TasksRunner.MakeChunker(chunkerName, tokenizer, chunkTokens);
        sw.Restart();
        Chunk[] chunks = files.SelectMany(chunker.Chunk).ToArray();
        Console.WriteLine($"chunker {chunker.Name}: {chunks.Length} chunks, {chunks.Sum(c => (long)c.TokenIds.Length)} tokens [{sw.Elapsed.TotalSeconds:F1} s]");
        sw.Restart();
        ReferenceGraph graph = new RoslynGraphBuilder(tokenizer).Build(scan.Compilations, chunks);
        Console.WriteLine($"graph (per-project compilations): {graph.Edges.Count} edges [{sw.Elapsed.TotalSeconds:F1} s]");

        request = request with { ChunkKind = ChunkRef.KindOf(chunker) };
        IGitInfo git = new GitCliInfo(root);
        foreach (SelectionRequest r in request.TokenBudget is null ? new[] { request } : new[] { request with { TokenBudget = null }, request })
        {
            sw.Restart();
            SelectionResult result = ContextSelector.Select(r, files, graph, git, root);
            double selS = sw.Elapsed.TotalSeconds;
            Console.WriteLine();
            Console.WriteLine($"request: symbol={r.Symbol ?? "-"} hops={r.Hops} paths=[{string.Join(", ", r.Paths)}] exclude=[{string.Join(", ", r.Exclude)}] "
                + $"since={r.ChangedSince ?? "-"}{(r.ChangedSinceMergeBase ? " (merge base)" : "")} max-per-hop={r.MaxNeighboursPerHop?.ToString() ?? "off"} budget={(r.TokenBudget is int b ? b.ToString() : "none")} [{selS * 1000:F0} ms]");
            foreach (SymbolDefinition d in result.SymbolDefinitions)
                Console.WriteLine($"  definition {d.QualifiedName} ({d.Kind}) {d.FilePath}:{d.Line}");
            Console.WriteLine($"  candidates {result.Count + result.Trimmed.Count} chunks / {result.CandidateTokens} tokens in "
                + $"{result.Concat(result.Trimmed).Select(c => c.Chunk.FilePath).Distinct().Count()} files; selected {result.Count} chunks / {result.SelectedTokens} tokens in "
                + $"{result.Select(c => c.Chunk.FilePath).Distinct().Count()} files; trimmed {result.Trimmed.Count} chunks / {result.CandidateTokens - result.SelectedTokens} tokens");
            var selectedIds = result.Select(c => c.Chunk.Id).ToHashSet(StringComparer.Ordinal);
            Console.WriteLine("  by reason (selected / candidates, tokens selected / candidates):");
            foreach (var g in result.Concat(result.Trimmed).GroupBy(c => (c.Reason, c.Hop)).OrderByDescending(g => g.Max(c => c.Priority)).ThenBy(g => g.Key.Hop))
            {
                var sel = g.Where(c => selectedIds.Contains(c.Chunk.Id)).ToArray();
                Console.WriteLine($"    {g.Key.Reason,-15} hop {g.Key.Hop,2}: {sel.Length,5} / {g.Count(),5} chunks, {sel.Sum(c => c.Tokens),7} / {g.Sum(c => c.Tokens),7} tokens");
            }
            foreach (string n in result.Notes) Console.WriteLine($"  note: {n}");
            foreach (string u in result.UnmatchedPatterns) Console.WriteLine($"  unmatched pattern: {u}");
            if (result.UnmatchedChangedFiles.Count > 0)
                Console.WriteLine($"  changed files with no chunk ({result.UnmatchedChangedFiles.Count}): {string.Join(", ", result.UnmatchedChangedFiles.Take(20))}");
            if (r.TokenBudget is not null)
            {
                Console.WriteLine($"  first {Math.Min(show, result.Count)} selected (rank order):");
                foreach (SelectedChunk c in result.Take(show))
                    Console.WriteLine($"    {c.Priority,4} w{c.Weight,-3} {c.Tokens,5} tok  {c.Chunk.FilePath}:{c.Chunk.StartLine}-{c.Chunk.EndLine}  {c.Detail}");
                if (result.Trimmed.Count > 0)
                {
                    SelectedChunk t = result.Trimmed[0];
                    Console.WriteLine($"  first trimmed: {t.Priority} w{t.Weight} {t.Tokens} tok {t.Chunk.FilePath}:{t.Chunk.StartLine}-{t.Chunk.EndLine} {t.Detail}");
                }
            }
        }
        return 0;
    }
}
